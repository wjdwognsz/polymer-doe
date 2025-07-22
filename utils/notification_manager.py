"""
데스크톱 알림 시스템 관리자
크로스 플랫폼 시스템 알림 및 알림 히스토리 관리
"""
import os
import sys
import platform
import json
import logging
import threading
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import time

# 플랫폼별 알림 라이브러리
try:
    from plyer import notification as plyer_notification
    PLYER_AVAILABLE = True
except ImportError:
    PLYER_AVAILABLE = False
    logging.warning("plyer not available, using fallback notification system")

# Windows 전용
if platform.system() == "Windows":
    try:
        from win10toast import ToastNotifier
        WIN_TOAST_AVAILABLE = True
    except ImportError:
        WIN_TOAST_AVAILABLE = False

# 프로젝트 임포트
from config.local_config import LOCAL_CONFIG
from config.app_config import NOTIFICATION_CONFIG


class NotificationType(Enum):
    """알림 유형"""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    SYSTEM = "system"
    PROJECT = "project"
    COLLABORATION = "collaboration"
    AI_COMPLETE = "ai_complete"


class NotificationPriority(Enum):
    """알림 우선순위"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class Notification:
    """알림 데이터 모델"""
    id: Optional[str] = None
    title: str = ""
    message: str = ""
    type: NotificationType = NotificationType.INFO
    priority: NotificationPriority = NotificationPriority.NORMAL
    timestamp: datetime = None
    data: Optional[Dict[str, Any]] = None
    actions: Optional[List[Dict[str, str]]] = None
    timeout: int = 10  # 초
    icon: Optional[str] = None
    sound: bool = True
    persistent: bool = False
    read: bool = False
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.id is None:
            self.id = f"{self.type.value}_{int(self.timestamp.timestamp() * 1000)}"
        if self.data is None:
            self.data = {}
        if self.actions is None:
            self.actions = []


class NotificationManager:
    """데스크톱 알림 관리자"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.platform = platform.system()
        self._notification_queue = queue.Queue()
        self._history = []
        self._callbacks: Dict[str, List[Callable]] = {}
        self._settings = self._load_settings()
        self._running = False
        self._worker_thread = None
        
        # 플랫폼별 알림 시스템 초기화
        self._init_platform_notifier()
        
        # 알림 아이콘 경로 설정
        self._setup_icons()
        
        # 데이터베이스 테이블 생성
        if self.db_manager:
            self._init_database()
        
        # 알림 워커 시작
        self.start()
    
    def _init_platform_notifier(self):
        """플랫폼별 알림 시스템 초기화"""
        self.notifier = None
        
        if self.platform == "Windows" and WIN_TOAST_AVAILABLE:
            self.notifier = ToastNotifier()
            self._notify_func = self._notify_windows
        elif PLYER_AVAILABLE:
            self._notify_func = self._notify_plyer
        else:
            self._notify_func = self._notify_fallback
            logging.warning("No native notification system available, using fallback")
    
    def _setup_icons(self):
        """알림 아이콘 설정"""
        # PyInstaller 환경 처리
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(os.path.dirname(__file__))
        
        self.icons = {
            NotificationType.INFO: os.path.join(base_path, 'assets', 'icons', 'info.ico'),
            NotificationType.SUCCESS: os.path.join(base_path, 'assets', 'icons', 'success.ico'),
            NotificationType.WARNING: os.path.join(base_path, 'assets', 'icons', 'warning.ico'),
            NotificationType.ERROR: os.path.join(base_path, 'assets', 'icons', 'error.ico'),
            NotificationType.SYSTEM: os.path.join(base_path, 'assets', 'icons', 'system.ico'),
            NotificationType.PROJECT: os.path.join(base_path, 'assets', 'icons', 'project.ico'),
            NotificationType.COLLABORATION: os.path.join(base_path, 'assets', 'icons', 'collab.ico'),
            NotificationType.AI_COMPLETE: os.path.join(base_path, 'assets', 'icons', 'ai.ico'),
        }
        
        # 기본 아이콘
        self.default_icon = os.path.join(base_path, 'assets', 'icons', 'app.ico')
    
    def _init_database(self):
        """데이터베이스 테이블 초기화"""
        try:
            self.db_manager._get_connection().execute('''
                CREATE TABLE IF NOT EXISTS notifications (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    type TEXT NOT NULL,
                    priority INTEGER DEFAULT 2,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    data TEXT,
                    read INTEGER DEFAULT 0,
                    dismissed INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            self.db_manager._get_connection().execute('''
                CREATE TABLE IF NOT EXISTS notification_settings (
                    user_id INTEGER PRIMARY KEY,
                    enabled INTEGER DEFAULT 1,
                    sound INTEGER DEFAULT 1,
                    types_enabled TEXT DEFAULT '{}',
                    quiet_hours_start TEXT,
                    quiet_hours_end TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            self.db_manager._get_connection().commit()
        except Exception as e:
            logging.error(f"Failed to initialize notification database: {e}")
    
    def _load_settings(self) -> Dict[str, Any]:
        """사용자 알림 설정 로드"""
        default_settings = {
            'enabled': True,
            'sound': True,
            'types_enabled': {t.value: True for t in NotificationType},
            'quiet_hours_start': None,
            'quiet_hours_end': None,
            'priority_threshold': NotificationPriority.NORMAL.value
        }
        
        if self.db_manager:
            try:
                # DB에서 설정 로드
                result = self.db_manager._get_connection().execute(
                    "SELECT * FROM notification_settings WHERE user_id = ?",
                    (self._get_current_user_id(),)
                ).fetchone()
                
                if result:
                    settings = dict(result)
                    settings['types_enabled'] = json.loads(settings.get('types_enabled', '{}'))
                    return settings
            except Exception as e:
                logging.error(f"Failed to load notification settings: {e}")
        
        return default_settings
    
    def _save_settings(self):
        """설정 저장"""
        if self.db_manager:
            try:
                conn = self.db_manager._get_connection()
                conn.execute('''
                    INSERT OR REPLACE INTO notification_settings 
                    (user_id, enabled, sound, types_enabled, quiet_hours_start, quiet_hours_end)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    self._get_current_user_id(),
                    self._settings['enabled'],
                    self._settings['sound'],
                    json.dumps(self._settings['types_enabled']),
                    self._settings.get('quiet_hours_start'),
                    self._settings.get('quiet_hours_end')
                ))
                conn.commit()
            except Exception as e:
                logging.error(f"Failed to save notification settings: {e}")
    
    def start(self):
        """알림 워커 시작"""
        if not self._running:
            self._running = True
            self._worker_thread = threading.Thread(target=self._notification_worker, daemon=True)
            self._worker_thread.start()
            logging.info("Notification manager started")
    
    def stop(self):
        """알림 워커 중지"""
        self._running = False
        self._notification_queue.put(None)  # 워커 깨우기
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        logging.info("Notification manager stopped")
    
    def _notification_worker(self):
        """백그라운드 알림 처리 워커"""
        while self._running:
            try:
                notification = self._notification_queue.get(timeout=1)
                if notification is None:
                    continue
                
                # 설정 확인
                if not self._should_show_notification(notification):
                    continue
                
                # 플랫폼별 알림 표시
                self._notify_func(notification)
                
                # 히스토리 저장
                self._save_to_history(notification)
                
                # 콜백 실행
                self._trigger_callbacks(notification)
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Notification worker error: {e}")
    
    def _should_show_notification(self, notification: Notification) -> bool:
        """알림 표시 여부 결정"""
        # 전체 알림 비활성화
        if not self._settings['enabled']:
            return False
        
        # 유형별 필터
        if not self._settings['types_enabled'].get(notification.type.value, True):
            return False
        
        # 우선순위 필터
        if notification.priority.value < self._settings['priority_threshold']:
            return False
        
        # 방해금지 시간 확인
        if self._in_quiet_hours():
            return notification.priority == NotificationPriority.URGENT
        
        return True
    
    def _in_quiet_hours(self) -> bool:
        """방해금지 시간 확인"""
        start = self._settings.get('quiet_hours_start')
        end = self._settings.get('quiet_hours_end')
        
        if not start or not end:
            return False
        
        now = datetime.now().time()
        start_time = datetime.strptime(start, "%H:%M").time()
        end_time = datetime.strptime(end, "%H:%M").time()
        
        if start_time <= end_time:
            return start_time <= now <= end_time
        else:  # 자정을 넘는 경우
            return now >= start_time or now <= end_time
    
    def _notify_windows(self, notification: Notification):
        """Windows 알림 표시"""
        try:
            icon_path = notification.icon or self.icons.get(notification.type, self.default_icon)
            
            # Windows 10 Toast Notification
            self.notifier.show_toast(
                notification.title,
                notification.message,
                icon_path=icon_path if os.path.exists(icon_path) else None,
                duration=notification.timeout,
                threaded=False
            )
        except Exception as e:
            logging.error(f"Windows notification error: {e}")
            self._notify_fallback(notification)
    
    def _notify_plyer(self, notification: Notification):
        """Plyer를 사용한 크로스플랫폼 알림"""
        try:
            plyer_notification.notify(
                title=notification.title,
                message=notification.message,
                app_name="Universal DOE Platform",
                timeout=notification.timeout
            )
        except Exception as e:
            logging.error(f"Plyer notification error: {e}")
            self._notify_fallback(notification)
    
    def _notify_fallback(self, notification: Notification):
        """폴백 알림 (로깅)"""
        logging.info(f"[{notification.type.value.upper()}] {notification.title}: {notification.message}")
        
        # Streamlit 세션에 알림 추가 (UI에서 표시)
        try:
            import streamlit as st
            if 'notifications' not in st.session_state:
                st.session_state.notifications = []
            st.session_state.notifications.append(notification)
        except:
            pass
    
    def _save_to_history(self, notification: Notification):
        """알림 히스토리 저장"""
        # 메모리 히스토리
        self._history.append(notification)
        if len(self._history) > 100:  # 최대 100개 유지
            self._history.pop(0)
        
        # 데이터베이스 저장
        if self.db_manager:
            try:
                conn = self.db_manager._get_connection()
                conn.execute('''
                    INSERT INTO notifications 
                    (id, user_id, title, message, type, priority, timestamp, data, read)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    notification.id,
                    self._get_current_user_id(),
                    notification.title,
                    notification.message,
                    notification.type.value,
                    notification.priority.value,
                    notification.timestamp,
                    json.dumps(notification.data),
                    0
                ))
                conn.commit()
            except Exception as e:
                logging.error(f"Failed to save notification to database: {e}")
    
    def _trigger_callbacks(self, notification: Notification):
        """알림 콜백 실행"""
        # 전체 콜백
        for callback in self._callbacks.get('*', []):
            try:
                callback(notification)
            except Exception as e:
                logging.error(f"Notification callback error: {e}")
        
        # 유형별 콜백
        for callback in self._callbacks.get(notification.type.value, []):
            try:
                callback(notification)
            except Exception as e:
                logging.error(f"Notification callback error: {e}")
    
    def notify(self, title: str, message: str, 
              type: NotificationType = NotificationType.INFO,
              priority: NotificationPriority = NotificationPriority.NORMAL,
              **kwargs) -> str:
        """알림 발송"""
        notification = Notification(
            title=title,
            message=message,
            type=type,
            priority=priority,
            **kwargs
        )
        
        self._notification_queue.put(notification)
        return notification.id
    
    def notify_info(self, title: str, message: str, **kwargs) -> str:
        """정보 알림"""
        return self.notify(title, message, NotificationType.INFO, **kwargs)
    
    def notify_success(self, title: str, message: str, **kwargs) -> str:
        """성공 알림"""
        return self.notify(title, message, NotificationType.SUCCESS, **kwargs)
    
    def notify_warning(self, title: str, message: str, **kwargs) -> str:
        """경고 알림"""
        return self.notify(title, message, NotificationType.WARNING, **kwargs)
    
    def notify_error(self, title: str, message: str, **kwargs) -> str:
        """에러 알림"""
        return self.notify(title, message, NotificationType.ERROR, 
                          priority=NotificationPriority.HIGH, **kwargs)
    
    def notify_project_complete(self, project_name: str, **kwargs) -> str:
        """프로젝트 완료 알림"""
        return self.notify(
            "프로젝트 완료",
            f"'{project_name}' 프로젝트가 완료되었습니다.",
            NotificationType.PROJECT,
            priority=NotificationPriority.HIGH,
            **kwargs
        )
    
    def notify_ai_complete(self, task: str, result_summary: str = "", **kwargs) -> str:
        """AI 작업 완료 알림"""
        message = f"AI {task} 작업이 완료되었습니다."
        if result_summary:
            message += f"\n{result_summary}"
        
        return self.notify(
            "AI 작업 완료",
            message,
            NotificationType.AI_COMPLETE,
            **kwargs
        )
    
    def get_history(self, limit: int = 50, 
                   type_filter: Optional[NotificationType] = None,
                   unread_only: bool = False) -> List[Notification]:
        """알림 히스토리 조회"""
        if self.db_manager:
            try:
                query = "SELECT * FROM notifications WHERE user_id = ?"
                params = [self._get_current_user_id()]
                
                if type_filter:
                    query += " AND type = ?"
                    params.append(type_filter.value)
                
                if unread_only:
                    query += " AND read = 0"
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                results = self.db_manager._get_connection().execute(query, params).fetchall()
                
                notifications = []
                for row in results:
                    data = dict(row)
                    data['type'] = NotificationType(data['type'])
                    data['priority'] = NotificationPriority(data['priority'])
                    data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                    data['data'] = json.loads(data.get('data', '{}'))
                    notifications.append(Notification(**data))
                
                return notifications
                
            except Exception as e:
                logging.error(f"Failed to get notification history: {e}")
        
        # 메모리 히스토리 폴백
        history = self._history[-limit:]
        if type_filter:
            history = [n for n in history if n.type == type_filter]
        if unread_only:
            history = [n for n in history if not n.read]
        
        return history
    
    def mark_as_read(self, notification_id: str):
        """알림을 읽음으로 표시"""
        # 메모리 업데이트
        for notification in self._history:
            if notification.id == notification_id:
                notification.read = True
                break
        
        # 데이터베이스 업데이트
        if self.db_manager:
            try:
                conn = self.db_manager._get_connection()
                conn.execute(
                    "UPDATE notifications SET read = 1 WHERE id = ?",
                    (notification_id,)
                )
                conn.commit()
            except Exception as e:
                logging.error(f"Failed to mark notification as read: {e}")
    
    def mark_all_as_read(self):
        """모든 알림을 읽음으로 표시"""
        # 메모리 업데이트
        for notification in self._history:
            notification.read = True
        
        # 데이터베이스 업데이트
        if self.db_manager:
            try:
                conn = self.db_manager._get_connection()
                conn.execute(
                    "UPDATE notifications SET read = 1 WHERE user_id = ?",
                    (self._get_current_user_id(),)
                )
                conn.commit()
            except Exception as e:
                logging.error(f"Failed to mark all notifications as read: {e}")
    
    def get_unread_count(self) -> int:
        """읽지 않은 알림 수"""
        if self.db_manager:
            try:
                result = self.db_manager._get_connection().execute(
                    "SELECT COUNT(*) FROM notifications WHERE user_id = ? AND read = 0",
                    (self._get_current_user_id(),)
                ).fetchone()
                return result[0] if result else 0
            except:
                pass
        
        return sum(1 for n in self._history if not n.read)
    
    def clear_old_notifications(self, days: int = 30):
        """오래된 알림 정리"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # 메모리 정리
        self._history = [n for n in self._history if n.timestamp > cutoff_date]
        
        # 데이터베이스 정리
        if self.db_manager:
            try:
                conn = self.db_manager._get_connection()
                conn.execute(
                    "DELETE FROM notifications WHERE timestamp < ?",
                    (cutoff_date,)
                )
                conn.commit()
            except Exception as e:
                logging.error(f"Failed to clear old notifications: {e}")
    
    def register_callback(self, callback: Callable, type_filter: Optional[str] = None):
        """알림 콜백 등록"""
        key = type_filter or '*'
        if key not in self._callbacks:
            self._callbacks[key] = []
        self._callbacks[key].append(callback)
    
    def update_settings(self, **kwargs):
        """설정 업데이트"""
        self._settings.update(kwargs)
        self._save_settings()
    
    def get_settings(self) -> Dict[str, Any]:
        """현재 설정 반환"""
        return self._settings.copy()
    
    def _get_current_user_id(self) -> int:
        """현재 사용자 ID 반환"""
        try:
            import streamlit as st
            if 'user' in st.session_state and st.session_state.user:
                return st.session_state.user.get('id', 0)
        except:
            pass
        return 0
    
    def render_notification_center(self):
        """Streamlit UI용 알림 센터 렌더링"""
        import streamlit as st
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("🔔 알림 센터")
        with col2:
            unread_count = self.get_unread_count()
            if unread_count > 0:
                st.warning(f"새 알림 {unread_count}개")
        
        # 알림 필터
        tab1, tab2, tab3 = st.tabs(["전체", "읽지 않음", "설정"])
        
        with tab1:
            notifications = self.get_history(limit=50)
            self._render_notification_list(notifications)
        
        with tab2:
            unread_notifications = self.get_history(limit=50, unread_only=True)
            self._render_notification_list(unread_notifications)
            
            if unread_notifications and st.button("모두 읽음으로 표시"):
                self.mark_all_as_read()
                st.rerun()
        
        with tab3:
            self._render_notification_settings()
    
    def _render_notification_list(self, notifications: List[Notification]):
        """알림 목록 렌더링"""
        import streamlit as st
        
        if not notifications:
            st.info("알림이 없습니다.")
            return
        
        for notification in notifications:
            with st.container():
                col1, col2, col3 = st.columns([1, 8, 1])
                
                with col1:
                    icon_map = {
                        NotificationType.INFO: "ℹ️",
                        NotificationType.SUCCESS: "✅",
                        NotificationType.WARNING: "⚠️",
                        NotificationType.ERROR: "❌",
                        NotificationType.PROJECT: "📁",
                        NotificationType.AI_COMPLETE: "🤖"
                    }
                    st.write(icon_map.get(notification.type, "📢"))
                
                with col2:
                    if not notification.read:
                        st.markdown(f"**{notification.title}**")
                    else:
                        st.markdown(notification.title)
                    
                    st.caption(notification.message)
                    st.caption(f"{notification.timestamp.strftime('%Y-%m-%d %H:%M')}")
                
                with col3:
                    if not notification.read:
                        if st.button("✓", key=f"read_{notification.id}"):
                            self.mark_as_read(notification.id)
                            st.rerun()
                
                st.divider()
    
    def _render_notification_settings(self):
        """알림 설정 UI"""
        import streamlit as st
        
        st.subheader("알림 설정")
        
        # 전체 알림 활성화
        enabled = st.checkbox(
            "알림 활성화",
            value=self._settings['enabled'],
            help="모든 알림을 켜거나 끕니다"
        )
        
        if enabled != self._settings['enabled']:
            self.update_settings(enabled=enabled)
        
        # 소리 설정
        sound = st.checkbox(
            "알림 소리",
            value=self._settings['sound'],
            disabled=not enabled
        )
        
        if sound != self._settings['sound']:
            self.update_settings(sound=sound)
        
        # 알림 유형별 설정
        st.write("알림 유형별 설정:")
        
        type_labels = {
            NotificationType.INFO: "정보 알림",
            NotificationType.SUCCESS: "성공 알림",
            NotificationType.WARNING: "경고 알림",
            NotificationType.ERROR: "오류 알림",
            NotificationType.PROJECT: "프로젝트 알림",
            NotificationType.AI_COMPLETE: "AI 작업 완료"
        }
        
        types_enabled = self._settings['types_enabled'].copy()
        
        for notification_type in NotificationType:
            enabled = st.checkbox(
                type_labels.get(notification_type, notification_type.value),
                value=types_enabled.get(notification_type.value, True),
                disabled=not self._settings['enabled']
            )
            types_enabled[notification_type.value] = enabled
        
        if types_enabled != self._settings['types_enabled']:
            self.update_settings(types_enabled=types_enabled)
        
        # 방해금지 시간
        st.write("방해금지 시간:")
        
        col1, col2 = st.columns(2)
        with col1:
            quiet_start = st.time_input(
                "시작 시간",
                value=None,
                disabled=not enabled
            )
        with col2:
            quiet_end = st.time_input(
                "종료 시간",
                value=None,
                disabled=not enabled
            )
        
        if quiet_start and quiet_end:
            self.update_settings(
                quiet_hours_start=quiet_start.strftime("%H:%M"),
                quiet_hours_end=quiet_end.strftime("%H:%M")
            )
        
        # 알림 정리
        st.divider()
        if st.button("30일 이상 된 알림 삭제"):
            self.clear_old_notifications(days=30)
            st.success("오래된 알림이 삭제되었습니다.")


# 싱글톤 인스턴스
_notification_manager: Optional[NotificationManager] = None

def get_notification_manager(db_manager=None) -> NotificationManager:
    """NotificationManager 싱글톤 인스턴스 반환"""
    global _notification_manager
    if _notification_manager is None:
        _notification_manager = NotificationManager(db_manager)
    return _notification_manager


# 편의 함수들
def notify(title: str, message: str, **kwargs) -> str:
    """간편 알림 발송"""
    return get_notification_manager().notify(title, message, **kwargs)

def notify_success(title: str, message: str, **kwargs) -> str:
    """성공 알림"""
    return get_notification_manager().notify_success(title, message, **kwargs)

def notify_error(title: str, message: str, **kwargs) -> str:
    """에러 알림"""
    return get_notification_manager().notify_error(title, message, **kwargs)

def notify_project_complete(project_name: str, **kwargs) -> str:
    """프로젝트 완료 알림"""
    return get_notification_manager().notify_project_complete(project_name, **kwargs)

def notify_ai_complete(task: str, result_summary: str = "", **kwargs) -> str:
    """AI 작업 완료 알림"""
    return get_notification_manager().notify_ai_complete(task, result_summary, **kwargs)
