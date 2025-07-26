"""
notification_manager.py - 알림 시스템 관리자
Universal DOE Platform의 중앙 알림 시스템으로 다양한 채널을 통한 알림 전송 및 관리
"""

import os
import sys
import platform
import json
import logging
import threading
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import time
import uuid
from collections import defaultdict, deque
import hashlib

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

# Streamlit 통합
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# 프로젝트 임포트
from config.app_config import APP_CONFIG
from config.local_config import LOCAL_CONFIG


class NotificationType(Enum):
    """확장된 알림 유형"""
    # 시스템
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    SYSTEM = "system"
    
    # 프로젝트
    PROJECT_CREATED = "project_created"
    PROJECT_SHARED = "project_shared"
    PROJECT_UPDATED = "project_updated"
    PROJECT_COMPLETED = "project_completed"
    PROJECT_DELETED = "project_deleted"
    
    # 실험
    EXPERIMENT_CREATED = "experiment_created"
    EXPERIMENT_ASSIGNED = "experiment_assigned"
    EXPERIMENT_STARTED = "experiment_started"
    EXPERIMENT_COMPLETED = "experiment_completed"
    EXPERIMENT_FAILED = "experiment_failed"
    DATA_UPLOADED = "data_uploaded"
    ANALYSIS_COMPLETED = "analysis_completed"
    
    # 협업
    COMMENT_ADDED = "comment_added"
    MENTIONED = "mentioned"
    TASK_ASSIGNED = "task_assigned"
    TASK_COMPLETED = "task_completed"
    FILE_SHARED = "file_shared"
    COLLABORATOR_ADDED = "collaborator_added"
    PERMISSION_CHANGED = "permission_changed"
    
    # AI
    AI_COMPLETE = "ai_complete"
    AI_ERROR = "ai_error"
    
    # 성과
    ACHIEVEMENT_UNLOCKED = "achievement_unlocked"
    MILESTONE_REACHED = "milestone_reached"


class NotificationCategory(Enum):
    """알림 카테고리"""
    SYSTEM = "system"
    PROJECT = "project"
    EXPERIMENT = "experiment"
    COLLABORATION = "collaboration"
    ACHIEVEMENT = "achievement"


class NotificationPriority(Enum):
    """알림 우선순위"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class NotificationChannel(Enum):
    """알림 채널"""
    IN_APP = "in_app"
    DESKTOP = "desktop"
    EMAIL = "email"
    PUSH = "push"  # 미래 확장


@dataclass
class NotificationAction:
    """알림 액션 버튼"""
    label: str
    action: str
    style: str = "primary"  # primary, secondary, danger
    data: Optional[Dict[str, Any]] = None


@dataclass
class Notification:
    """확장된 알림 데이터 모델"""
    # 기본 정보
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: NotificationType = NotificationType.INFO
    category: Optional[NotificationCategory] = None
    priority: NotificationPriority = NotificationPriority.NORMAL
    
    # 콘텐츠
    title: str = ""
    message: str = ""
    icon: Optional[str] = None
    image: Optional[str] = None
    
    # 메타데이터
    user_id: Optional[str] = None
    sender_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    
    # 액션
    actions: List[NotificationAction] = field(default_factory=list)
    link: Optional[str] = None
    
    # 채널
    channels: List[NotificationChannel] = field(default_factory=lambda: [NotificationChannel.IN_APP])
    
    # 상태
    status: str = "pending"  # pending, sent, delivered, read, failed
    read: bool = False
    
    # 타임스탬프
    created_at: datetime = field(default_factory=datetime.now)
    sent_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    # 그룹화
    group_id: Optional[str] = None
    parent_id: Optional[str] = None  # 스레드
    
    # 기타
    sound: bool = True
    persistent: bool = False
    retry_count: int = 0
    
    def __post_init__(self):
        # 카테고리 자동 설정
        if not self.category:
            self.category = self._infer_category()
    
    def _infer_category(self) -> NotificationCategory:
        """알림 유형에서 카테고리 추론"""
        type_name = self.type.value
        if type_name.startswith("project_"):
            return NotificationCategory.PROJECT
        elif type_name.startswith("experiment_"):
            return NotificationCategory.EXPERIMENT
        elif type_name in ["comment_added", "mentioned", "task_", "collaborator_", "file_shared"]:
            return NotificationCategory.COLLABORATION
        elif type_name.startswith("achievement_") or type_name == "milestone_reached":
            return NotificationCategory.ACHIEVEMENT
        else:
            return NotificationCategory.SYSTEM
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        data = asdict(self)
        data['type'] = self.type.value
        data['category'] = self.category.value if self.category else None
        data['priority'] = self.priority.value
        data['channels'] = [ch.value for ch in self.channels]
        data['created_at'] = self.created_at.isoformat()
        data['sent_at'] = self.sent_at.isoformat() if self.sent_at else None
        data['read_at'] = self.read_at.isoformat() if self.read_at else None
        data['expires_at'] = self.expires_at.isoformat() if self.expires_at else None
        data['actions'] = [asdict(action) for action in self.actions]
        return data


@dataclass
class NotificationSettings:
    """사용자 알림 설정"""
    user_id: str
    enabled: bool = True
    
    # 채널별 설정
    channels: Dict[str, bool] = field(default_factory=lambda: {
        'in_app': True,
        'desktop': True,
        'email': False,
        'push': False
    })
    
    # 카테고리별 설정
    categories: Dict[str, bool] = field(default_factory=lambda: {
        'system': True,
        'project': True,
        'experiment': True,
        'collaboration': True,
        'achievement': True
    })
    
    # 유형별 세부 설정
    types: Dict[str, bool] = field(default_factory=dict)
    
    # 우선순위 임계값
    priority_threshold: int = NotificationPriority.LOW.value
    
    # 방해금지 모드
    quiet_hours: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': False,
        'start': "22:00",
        'end': "08:00",
        'timezone': 'local',
        'allow_urgent': True
    })
    
    # 이메일 다이제스트
    email_digest: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': False,
        'frequency': 'daily',  # immediately, hourly, daily, weekly
        'time': "09:00"
    })
    
    # 그룹화 설정
    grouping: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'time_window': 300,  # 5분
        'max_group_size': 10
    })
    
    # 커스텀 규칙
    rules: List[Dict[str, Any]] = field(default_factory=list)


class NotificationQueue:
    """우선순위 기반 알림 큐"""
    
    def __init__(self):
        self._queue = queue.PriorityQueue()
        self._failed_queue = deque(maxlen=100)
        self._offline_queues: Dict[str, List[Notification]] = defaultdict(list)
        self._processing = False
        self._lock = threading.Lock()
    
    def enqueue(self, notification: Notification):
        """알림을 큐에 추가"""
        # 우선순위 역순 (높은 우선순위가 먼저)
        priority = -notification.priority.value
        timestamp = notification.created_at.timestamp()
        
        self._queue.put((priority, timestamp, notification))
    
    def dequeue(self, timeout: float = 1.0) -> Optional[Notification]:
        """큐에서 알림 추출"""
        try:
            _, _, notification = self._queue.get(timeout=timeout)
            return notification
        except queue.Empty:
            return None
    
    def enqueue_batch(self, notifications: List[Notification]):
        """대량 알림 추가"""
        for notification in notifications:
            self.enqueue(notification)
    
    def add_to_offline_queue(self, user_id: str, notification: Notification):
        """오프라인 사용자 큐에 추가"""
        with self._lock:
            self._offline_queues[user_id].append(notification)
            # 최대 100개까지만 보관
            if len(self._offline_queues[user_id]) > 100:
                self._offline_queues[user_id].pop(0)
    
    def get_offline_notifications(self, user_id: str) -> List[Notification]:
        """오프라인 동안 쌓인 알림 반환"""
        with self._lock:
            notifications = self._offline_queues.get(user_id, [])
            self._offline_queues[user_id] = []
            return notifications
    
    def add_failed(self, notification: Notification):
        """실패한 알림 추가"""
        notification.retry_count += 1
        self._failed_queue.append(notification)
    
    def get_failed_notifications(self) -> List[Notification]:
        """실패한 알림 반환"""
        failed = list(self._failed_queue)
        self._failed_queue.clear()
        return failed
    
    def size(self) -> int:
        """큐 크기"""
        return self._queue.qsize()
    
    def is_empty(self) -> bool:
        """큐가 비어있는지 확인"""
        return self._queue.empty()


class NotificationGrouper:
    """알림 그룹화 관리자"""
    
    def __init__(self, time_window: int = 300):  # 5분
        self.time_window = time_window
        self._groups: Dict[str, List[Notification]] = defaultdict(list)
        self._group_timestamps: Dict[str, datetime] = {}
        self._lock = threading.Lock()
    
    def should_group(self, notification: Notification, existing: Notification) -> bool:
        """두 알림을 그룹화할지 결정"""
        # 같은 유형
        if notification.type != existing.type:
            return False
        
        # 같은 카테고리
        if notification.category != existing.category:
            return False
        
        # 시간 차이 확인
        time_diff = (notification.created_at - existing.created_at).total_seconds()
        if abs(time_diff) > self.time_window:
            return False
        
        # 같은 프로젝트/실험
        if notification.data.get('project_id') != existing.data.get('project_id'):
            return False
        
        return True
    
    def add_to_group(self, notification: Notification) -> Optional[str]:
        """알림을 그룹에 추가하고 그룹 ID 반환"""
        with self._lock:
            # 그룹 키 생성
            group_key = self._generate_group_key(notification)
            
            # 기존 그룹 확인
            for gid, group in self._groups.items():
                if group and self.should_group(notification, group[0]):
                    notification.group_id = gid
                    group.append(notification)
                    return gid
            
            # 새 그룹 생성
            group_id = str(uuid.uuid4())
            notification.group_id = group_id
            self._groups[group_id] = [notification]
            self._group_timestamps[group_id] = notification.created_at
            
            return group_id
    
    def _generate_group_key(self, notification: Notification) -> str:
        """그룹 키 생성"""
        parts = [
            notification.type.value,
            notification.user_id or "",
            notification.data.get('project_id', ''),
            notification.data.get('experiment_id', '')
        ]
        return hashlib.md5("_".join(parts).encode()).hexdigest()
    
    def get_group(self, group_id: str) -> List[Notification]:
        """그룹의 알림들 반환"""
        return self._groups.get(group_id, [])
    
    def clean_old_groups(self):
        """오래된 그룹 정리"""
        with self._lock:
            now = datetime.now()
            expired = []
            
            for gid, timestamp in self._group_timestamps.items():
                if (now - timestamp).total_seconds() > self.time_window * 2:
                    expired.append(gid)
            
            for gid in expired:
                del self._groups[gid]
                del self._group_timestamps[gid]


class RealtimeNotifier:
    """실시간 알림 전송자 (Streamlit 환경)"""
    
    def __init__(self):
        self._subscriptions: Dict[str, Set[str]] = defaultdict(set)  # channel -> users
        self._online_users: Set[str] = set()
        self._notification_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self._lock = threading.Lock()
    
    def subscribe(self, user_id: str, channel: str):
        """채널 구독"""
        with self._lock:
            self._subscriptions[channel].add(user_id)
            self._online_users.add(user_id)
    
    def unsubscribe(self, user_id: str, channel: str):
        """채널 구독 해제"""
        with self._lock:
            self._subscriptions[channel].discard(user_id)
    
    def set_online(self, user_id: str):
        """사용자 온라인 상태 설정"""
        with self._lock:
            self._online_users.add(user_id)
    
    def set_offline(self, user_id: str):
        """사용자 오프라인 상태 설정"""
        with self._lock:
            self._online_users.discard(user_id)
    
    def is_online(self, user_id: str) -> bool:
        """사용자 온라인 여부"""
        return user_id in self._online_users
    
    def broadcast(self, notification: Notification, channel: Optional[str] = None):
        """채널에 알림 브로드캐스트"""
        with self._lock:
            if channel:
                users = self._subscriptions.get(channel, set())
            else:
                users = {notification.user_id} if notification.user_id else set()
            
            for user_id in users:
                if user_id in self._online_users:
                    self._notification_buffer[user_id].append(notification)
    
    def get_notifications(self, user_id: str) -> List[Notification]:
        """사용자의 버퍼된 알림 반환"""
        with self._lock:
            notifications = list(self._notification_buffer[user_id])
            self._notification_buffer[user_id].clear()
            return notifications
    
    def update_streamlit_state(self, notification: Notification):
        """Streamlit 세션 상태 업데이트"""
        if STREAMLIT_AVAILABLE and hasattr(st, 'session_state'):
            if 'notifications' not in st.session_state:
                st.session_state.notifications = []
            if 'unread_count' not in st.session_state:
                st.session_state.unread_count = 0
            
            st.session_state.notifications.insert(0, notification)
            if not notification.read:
                st.session_state.unread_count += 1
            
            # 최대 100개까지만 유지
            st.session_state.notifications = st.session_state.notifications[:100]


class NotificationManager:
    """확장된 알림 관리자"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.platform = platform.system()
        
        # 핵심 컴포넌트
        self._queue = NotificationQueue()
        self._grouper = NotificationGrouper()
        self._realtime = RealtimeNotifier()
        
        # 상태 관리
        self._settings: Dict[str, NotificationSettings] = {}
        self._statistics: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # 워커 관리
        self._running = False
        self._workers = []
        
        # 플랫폼별 알림 시스템
        self._init_platform_notifier()
        
        # 아이콘 설정
        self._setup_icons()
        
        # 데이터베이스 초기화
        if self.db_manager:
            self._init_database()
        
        # 워커 시작
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
            logging.info("Using fallback notification system")
    
    def _setup_icons(self):
        """알림 아이콘 설정"""
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(os.path.dirname(__file__))
        
        icons_dir = os.path.join(base_path, 'assets', 'icons')
        
        # 카테고리별 아이콘
        self.category_icons = {
            NotificationCategory.SYSTEM: 'system.ico',
            NotificationCategory.PROJECT: 'project.ico',
            NotificationCategory.EXPERIMENT: 'experiment.ico',
            NotificationCategory.COLLABORATION: 'collaboration.ico',
            NotificationCategory.ACHIEVEMENT: 'achievement.ico'
        }
        
        # 유형별 아이콘 (선택적)
        self.type_icons = {
            NotificationType.SUCCESS: 'success.ico',
            NotificationType.WARNING: 'warning.ico',
            NotificationType.ERROR: 'error.ico',
            NotificationType.AI_COMPLETE: 'ai.ico'
        }
        
        # 전체 경로로 변환
        for category, icon in self.category_icons.items():
            self.category_icons[category] = os.path.join(icons_dir, icon)
        
        for type_, icon in self.type_icons.items():
            self.type_icons[type_] = os.path.join(icons_dir, icon)
        
        self.default_icon = os.path.join(icons_dir, 'app.ico')
    
    def _init_database(self):
        """데이터베이스 테이블 초기화"""
        try:
            conn = self.db_manager._get_connection()
            
            # 알림 테이블
            conn.execute('''
                CREATE TABLE IF NOT EXISTS notifications (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    type TEXT NOT NULL,
                    category TEXT,
                    priority INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    data TEXT,
                    status TEXT DEFAULT 'pending',
                    read INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    sent_at TIMESTAMP,
                    read_at TIMESTAMP,
                    expires_at TIMESTAMP,
                    group_id TEXT,
                    parent_id TEXT,
                    retry_count INTEGER DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # 알림 설정 테이블
            conn.execute('''
                CREATE TABLE IF NOT EXISTS notification_settings (
                    user_id TEXT PRIMARY KEY,
                    enabled INTEGER DEFAULT 1,
                    channels TEXT DEFAULT '{}',
                    categories TEXT DEFAULT '{}',
                    types TEXT DEFAULT '{}',
                    priority_threshold INTEGER DEFAULT 1,
                    quiet_hours TEXT DEFAULT '{}',
                    email_digest TEXT DEFAULT '{}',
                    grouping TEXT DEFAULT '{}',
                    rules TEXT DEFAULT '[]',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # 알림 통계 테이블
            conn.execute('''
                CREATE TABLE IF NOT EXISTS notification_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    type TEXT,
                    category TEXT,
                    date DATE,
                    sent_count INTEGER DEFAULT 0,
                    read_count INTEGER DEFAULT 0,
                    click_count INTEGER DEFAULT 0,
                    dismiss_count INTEGER DEFAULT 0,
                    UNIQUE(user_id, type, date),
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # 인덱스 생성
            conn.execute('CREATE INDEX IF NOT EXISTS idx_notifications_user_status ON notifications(user_id, status)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_notifications_created ON notifications(created_at DESC)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_notifications_group ON notifications(group_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_stats_user_date ON notification_stats(user_id, date)')
            
            conn.commit()
            logging.info("Notification database tables initialized")
            
        except Exception as e:
            logging.error(f"Failed to initialize notification database: {e}")
    
    def start(self):
        """알림 시스템 시작"""
        if self._running:
            return
        
        self._running = True
        
        # 메인 워커
        main_worker = threading.Thread(target=self._notification_worker, daemon=True)
        main_worker.start()
        self._workers.append(main_worker)
        
        # 그룹 정리 워커
        cleanup_worker = threading.Thread(target=self._cleanup_worker, daemon=True)
        cleanup_worker.start()
        self._workers.append(cleanup_worker)
        
        # 통계 워커
        stats_worker = threading.Thread(target=self._stats_worker, daemon=True)
        stats_worker.start()
        self._workers.append(stats_worker)
        
        logging.info("Notification manager started")
    
    def stop(self):
        """알림 시스템 중지"""
        self._running = False
        
        # 워커 종료 대기
        for worker in self._workers:
            worker.join(timeout=5)
        
        self._workers.clear()
        logging.info("Notification manager stopped")
    
    def _notification_worker(self):
        """메인 알림 처리 워커"""
        while self._running:
            try:
                # 큐에서 알림 가져오기
                notification = self._queue.dequeue(timeout=1.0)
                if not notification:
                    continue
                
                # 처리
                self._process_notification(notification)
                
            except Exception as e:
                logging.error(f"Notification worker error: {e}")
    
    def _cleanup_worker(self):
        """정리 작업 워커"""
        while self._running:
            try:
                # 5분마다 실행
                time.sleep(300)
                
                # 오래된 그룹 정리
                self._grouper.clean_old_groups()
                
                # 만료된 알림 정리
                self._clean_expired_notifications()
                
                # 실패한 알림 재시도
                self._retry_failed_notifications()
                
            except Exception as e:
                logging.error(f"Cleanup worker error: {e}")
    
    def _stats_worker(self):
        """통계 업데이트 워커"""
        while self._running:
            try:
                # 1분마다 실행
                time.sleep(60)
                
                # 통계 데이터베이스에 저장
                self._save_statistics()
                
            except Exception as e:
                logging.error(f"Stats worker error: {e}")
    
    def _process_notification(self, notification: Notification):
        """알림 처리"""
        try:
            # 사용자 설정 확인
            if not self._should_send_notification(notification):
                return
            
            # 그룹화 처리
            if self._should_group(notification):
                self._grouper.add_to_group(notification)
            
            # 채널별 전송
            for channel in notification.channels:
                if channel == NotificationChannel.IN_APP:
                    self._send_in_app(notification)
                elif channel == NotificationChannel.DESKTOP:
                    self._send_desktop(notification)
                elif channel == NotificationChannel.EMAIL:
                    self._queue_email(notification)
            
            # 상태 업데이트
            notification.status = "sent"
            notification.sent_at = datetime.now()
            
            # 데이터베이스 저장
            self._save_notification(notification)
            
            # 통계 업데이트
            self._update_statistics(notification, 'sent')
            
            # 콜백 실행
            self._trigger_callbacks(notification)
            
        except Exception as e:
            logging.error(f"Failed to process notification: {e}")
            notification.status = "failed"
            self._queue.add_failed(notification)
    
    def _should_send_notification(self, notification: Notification) -> bool:
        """알림 전송 여부 결정"""
        if not notification.user_id:
            return True  # 시스템 알림
        
        settings = self.get_user_settings(notification.user_id)
        
        # 전체 비활성화
        if not settings.enabled:
            return False
        
        # 카테고리 확인
        if notification.category and not settings.categories.get(notification.category.value, True):
            return False
        
        # 유형별 확인
        if notification.type.value in settings.types and not settings.types[notification.type.value]:
            return False
        
        # 우선순위 확인
        if notification.priority.value < settings.priority_threshold:
            return False
        
        # 방해금지 시간 확인
        if self._in_quiet_hours(settings) and notification.priority != NotificationPriority.URGENT:
            return False
        
        return True
    
    def _should_group(self, notification: Notification) -> bool:
        """그룹화 여부 결정"""
        if not notification.user_id:
            return False
        
        settings = self.get_user_settings(notification.user_id)
        return settings.grouping.get('enabled', True)
    
    def _in_quiet_hours(self, settings: NotificationSettings) -> bool:
        """방해금지 시간 확인"""
        quiet = settings.quiet_hours
        if not quiet.get('enabled'):
            return False
        
        now = datetime.now().time()
        start = datetime.strptime(quiet['start'], "%H:%M").time()
        end = datetime.strptime(quiet['end'], "%H:%M").time()
        
        if start <= end:
            return start <= now <= end
        else:
            return now >= start or now <= end
    
    def _send_in_app(self, notification: Notification):
        """인앱 알림 전송"""
        # 실시간 전송
        self._realtime.broadcast(notification)
        
        # Streamlit 상태 업데이트
        self._realtime.update_streamlit_state(notification)
    
    def _send_desktop(self, notification: Notification):
        """데스크톱 알림 전송"""
        if notification.sound and not self._in_quiet_hours(self.get_user_settings(notification.user_id)):
            self._play_notification_sound()
        
        # 플랫폼별 알림
        self._notify_func(notification)
    
    def _queue_email(self, notification: Notification):
        """이메일 큐에 추가"""
        # TODO: EmailNotifier 통합
        logging.info(f"Email queued for notification {notification.id}")
    
    def _notify_windows(self, notification: Notification):
        """Windows 알림"""
        try:
            icon_path = self._get_icon_path(notification)
            
            self.notifier.show_toast(
                notification.title,
                notification.message,
                icon_path=icon_path if os.path.exists(icon_path) else None,
                duration=10,
                threaded=False
            )
        except Exception as e:
            logging.error(f"Windows notification error: {e}")
            self._notify_fallback(notification)
    
    def _notify_plyer(self, notification: Notification):
        """Plyer 크로스플랫폼 알림"""
        try:
            plyer_notification.notify(
                title=notification.title,
                message=notification.message,
                app_name="Universal DOE Platform",
                timeout=10
            )
        except Exception as e:
            logging.error(f"Plyer notification error: {e}")
            self._notify_fallback(notification)
    
    def _notify_fallback(self, notification: Notification):
        """폴백 알림"""
        logging.info(f"[{notification.type.value.upper()}] {notification.title}: {notification.message}")
    
    def _get_icon_path(self, notification: Notification) -> str:
        """알림 아이콘 경로 반환"""
        if notification.icon and os.path.exists(notification.icon):
            return notification.icon
        
        # 유형별 아이콘
        if notification.type in self.type_icons:
            return self.type_icons[notification.type]
        
        # 카테고리별 아이콘
        if notification.category in self.category_icons:
            return self.category_icons[notification.category]
        
        return self.default_icon
    
    def _play_notification_sound(self):
        """알림 소리 재생"""
        # TODO: 플랫폼별 소리 재생
        pass
    
    def _save_notification(self, notification: Notification):
        """데이터베이스에 알림 저장"""
        if not self.db_manager:
            return
        
        try:
            conn = self.db_manager._get_connection()
            conn.execute('''
                INSERT INTO notifications 
                (id, user_id, type, category, priority, title, message, data, 
                 status, read, created_at, sent_at, expires_at, group_id, parent_id, retry_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                notification.id,
                notification.user_id,
                notification.type.value,
                notification.category.value if notification.category else None,
                notification.priority.value,
                notification.title,
                notification.message,
                json.dumps(notification.data),
                notification.status,
                int(notification.read),
                notification.created_at,
                notification.sent_at,
                notification.expires_at,
                notification.group_id,
                notification.parent_id,
                notification.retry_count
            ))
            conn.commit()
        except Exception as e:
            logging.error(f"Failed to save notification: {e}")
    
    def _update_statistics(self, notification: Notification, action: str):
        """통계 업데이트"""
        key = f"{notification.user_id}:{notification.type.value}:{datetime.now().date()}"
        self._statistics[key][f"{action}_count"] += 1
    
    def _save_statistics(self):
        """통계를 데이터베이스에 저장"""
        if not self.db_manager or not self._statistics:
            return
        
        try:
            conn = self.db_manager._get_connection()
            
            for key, stats in self._statistics.items():
                user_id, type_str, date_str = key.split(':')
                
                # UPSERT 구현
                conn.execute('''
                    INSERT INTO notification_stats 
                    (user_id, type, date, sent_count, read_count, click_count, dismiss_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, type, date) DO UPDATE SET
                        sent_count = sent_count + excluded.sent_count,
                        read_count = read_count + excluded.read_count,
                        click_count = click_count + excluded.click_count,
                        dismiss_count = dismiss_count + excluded.dismiss_count
                ''', (
                    user_id, type_str, date_str,
                    stats.get('sent_count', 0),
                    stats.get('read_count', 0),
                    stats.get('click_count', 0),
                    stats.get('dismiss_count', 0)
                ))
            
            conn.commit()
            self._statistics.clear()
            
        except Exception as e:
            logging.error(f"Failed to save statistics: {e}")
    
    def _trigger_callbacks(self, notification: Notification):
        """콜백 실행"""
        # 전체 콜백
        for callback in self._callbacks.get('*', []):
            try:
                callback(notification)
            except Exception as e:
                logging.error(f"Callback error: {e}")
        
        # 유형별 콜백
        for callback in self._callbacks.get(notification.type.value, []):
            try:
                callback(notification)
            except Exception as e:
                logging.error(f"Type callback error: {e}")
    
    def _clean_expired_notifications(self):
        """만료된 알림 정리"""
        if not self.db_manager:
            return
        
        try:
            conn = self.db_manager._get_connection()
            conn.execute('''
                DELETE FROM notifications 
                WHERE expires_at IS NOT NULL AND expires_at < ?
            ''', (datetime.now(),))
            conn.commit()
        except Exception as e:
            logging.error(f"Failed to clean expired notifications: {e}")
    
    def _retry_failed_notifications(self):
        """실패한 알림 재시도"""
        failed = self._queue.get_failed_notifications()
        
        for notification in failed:
            if notification.retry_count < 3:  # 최대 3회
                self._queue.enqueue(notification)
    
    # === 공개 API ===
    
    def send_notification(self, 
                         title: str,
                         message: str,
                         type: NotificationType = NotificationType.INFO,
                         user_id: Optional[str] = None,
                         **kwargs) -> str:
        """알림 전송"""
        notification = Notification(
            title=title,
            message=message,
            type=type,
            user_id=user_id,
            **kwargs
        )
        
        self._queue.enqueue(notification)
        return notification.id
    
    def send_bulk_notifications(self, 
                               user_ids: List[str],
                               title: str,
                               message: str,
                               **kwargs) -> List[str]:
        """대량 알림 전송"""
        notification_ids = []
        
        for user_id in user_ids:
            nid = self.send_notification(
                title=title,
                message=message,
                user_id=user_id,
                **kwargs
            )
            notification_ids.append(nid)
        
        return notification_ids
    
    def get_notifications(self,
                         user_id: str,
                         limit: int = 50,
                         offset: int = 0,
                         unread_only: bool = False,
                         category: Optional[NotificationCategory] = None,
                         type_filter: Optional[NotificationType] = None) -> List[Dict[str, Any]]:
        """알림 조회"""
        if not self.db_manager:
            return []
        
        try:
            query = '''
                SELECT * FROM notifications 
                WHERE user_id = ?
            '''
            params = [user_id]
            
            if unread_only:
                query += ' AND read = 0'
            
            if category:
                query += ' AND category = ?'
                params.append(category.value)
            
            if type_filter:
                query += ' AND type = ?'
                params.append(type_filter.value)
            
            query += ' ORDER BY created_at DESC LIMIT ? OFFSET ?'
            params.extend([limit, offset])
            
            conn = self.db_manager._get_connection()
            rows = conn.execute(query, params).fetchall()
            
            notifications = []
            for row in rows:
                data = dict(row)
                data['data'] = json.loads(data.get('data', '{}'))
                notifications.append(data)
            
            return notifications
            
        except Exception as e:
            logging.error(f"Failed to get notifications: {e}")
            return []
    
    def mark_as_read(self, notification_id: str, user_id: str):
        """읽음 표시"""
        if not self.db_manager:
            return
        
        try:
            conn = self.db_manager._get_connection()
            conn.execute('''
                UPDATE notifications 
                SET read = 1, read_at = ?
                WHERE id = ? AND user_id = ?
            ''', (datetime.now(), notification_id, user_id))
            conn.commit()
            
            # 통계 업데이트
            self._statistics[f"{user_id}:read:{datetime.now().date()}"]["read_count"] += 1
            
            # Streamlit 상태 업데이트
            if STREAMLIT_AVAILABLE and hasattr(st, 'session_state'):
                if st.session_state.get('unread_count', 0) > 0:
                    st.session_state.unread_count -= 1
                    
        except Exception as e:
            logging.error(f"Failed to mark as read: {e}")
    
    def mark_all_as_read(self, user_id: str):
        """모두 읽음 표시"""
        if not self.db_manager:
            return
        
        try:
            conn = self.db_manager._get_connection()
            result = conn.execute('''
                UPDATE notifications 
                SET read = 1, read_at = ?
                WHERE user_id = ? AND read = 0
            ''', (datetime.now(), user_id))
            conn.commit()
            
            count = result.rowcount
            
            # Streamlit 상태 업데이트
            if STREAMLIT_AVAILABLE and hasattr(st, 'session_state'):
                st.session_state.unread_count = 0
                
        except Exception as e:
            logging.error(f"Failed to mark all as read: {e}")
    
    def delete_notification(self, notification_id: str, user_id: str):
        """알림 삭제"""
        if not self.db_manager:
            return
        
        try:
            conn = self.db_manager._get_connection()
            conn.execute('''
                DELETE FROM notifications 
                WHERE id = ? AND user_id = ?
            ''', (notification_id, user_id))
            conn.commit()
        except Exception as e:
            logging.error(f"Failed to delete notification: {e}")
    
    def get_unread_count(self, user_id: str) -> int:
        """읽지 않은 알림 수"""
        if not self.db_manager:
            return 0
        
        try:
            conn = self.db_manager._get_connection()
            result = conn.execute('''
                SELECT COUNT(*) FROM notifications 
                WHERE user_id = ? AND read = 0
            ''', (user_id,)).fetchone()
            
            return result[0] if result else 0
            
        except Exception as e:
            logging.error(f"Failed to get unread count: {e}")
            return 0
    
    def get_user_settings(self, user_id: str) -> NotificationSettings:
        """사용자 설정 조회"""
        if user_id in self._settings:
            return self._settings[user_id]
        
        # 데이터베이스에서 로드
        if self.db_manager:
            try:
                conn = self.db_manager._get_connection()
                row = conn.execute('''
                    SELECT * FROM notification_settings 
                    WHERE user_id = ?
                ''', (user_id,)).fetchone()
                
                if row:
                    settings = NotificationSettings(
                        user_id=user_id,
                        enabled=bool(row['enabled']),
                        channels=json.loads(row['channels']),
                        categories=json.loads(row['categories']),
                        types=json.loads(row['types']),
                        priority_threshold=row['priority_threshold'],
                        quiet_hours=json.loads(row['quiet_hours']),
                        email_digest=json.loads(row['email_digest']),
                        grouping=json.loads(row['grouping']),
                        rules=json.loads(row['rules'])
                    )
                    self._settings[user_id] = settings
                    return settings
                    
            except Exception as e:
                logging.error(f"Failed to load user settings: {e}")
        
        # 기본 설정 반환
        settings = NotificationSettings(user_id=user_id)
        self._settings[user_id] = settings
        return settings
    
    def update_user_settings(self, user_id: str, **updates):
        """사용자 설정 업데이트"""
        settings = self.get_user_settings(user_id)
        
        # 업데이트 적용
        for key, value in updates.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
        
        # 캐시 업데이트
        self._settings[user_id] = settings
        
        # 데이터베이스 저장
        if self.db_manager:
            try:
                conn = self.db_manager._get_connection()
                conn.execute('''
                    INSERT OR REPLACE INTO notification_settings 
                    (user_id, enabled, channels, categories, types, priority_threshold,
                     quiet_hours, email_digest, grouping, rules, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user_id,
                    int(settings.enabled),
                    json.dumps(settings.channels),
                    json.dumps(settings.categories),
                    json.dumps(settings.types),
                    settings.priority_threshold,
                    json.dumps(settings.quiet_hours),
                    json.dumps(settings.email_digest),
                    json.dumps(settings.grouping),
                    json.dumps(settings.rules),
                    datetime.now()
                ))
                conn.commit()
            except Exception as e:
                logging.error(f"Failed to save user settings: {e}")
    
    def get_notification_stats(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """알림 통계 조회"""
        if not self.db_manager:
            return {}
        
        try:
            conn = self.db_manager._get_connection()
            
            # 기간별 통계
            start_date = datetime.now() - timedelta(days=days)
            
            stats = conn.execute('''
                SELECT 
                    type,
                    SUM(sent_count) as total_sent,
                    SUM(read_count) as total_read,
                    SUM(click_count) as total_click,
                    SUM(dismiss_count) as total_dismiss
                FROM notification_stats
                WHERE user_id = ? AND date >= ?
                GROUP BY type
            ''', (user_id, start_date.date())).fetchall()
            
            # 일별 추이
            daily = conn.execute('''
                SELECT 
                    date,
                    SUM(sent_count) as sent,
                    SUM(read_count) as read
                FROM notification_stats
                WHERE user_id = ? AND date >= ?
                GROUP BY date
                ORDER BY date
            ''', (user_id, start_date.date())).fetchall()
            
            return {
                'by_type': [dict(row) for row in stats],
                'daily_trend': [dict(row) for row in daily],
                'total_sent': sum(row['total_sent'] for row in stats),
                'total_read': sum(row['total_read'] for row in stats),
                'read_rate': sum(row['total_read'] for row in stats) / max(sum(row['total_sent'] for row in stats), 1)
            }
            
        except Exception as e:
            logging.error(f"Failed to get notification stats: {e}")
            return {}
    
    def register_callback(self, event: str, callback: Callable):
        """콜백 등록"""
        self._callbacks[event].append(callback)
    
    def unregister_callback(self, event: str, callback: Callable):
        """콜백 해제"""
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)
    
    # === 편의 메서드 ===
    
    def notify_success(self, title: str, message: str, **kwargs) -> str:
        """성공 알림"""
        return self.send_notification(title, message, NotificationType.SUCCESS, **kwargs)
    
    def notify_error(self, title: str, message: str, **kwargs) -> str:
        """에러 알림"""
        return self.send_notification(title, message, NotificationType.ERROR, 
                                    priority=NotificationPriority.HIGH, **kwargs)
    
    def notify_project_complete(self, project_name: str, project_id: str, user_id: str, **kwargs) -> str:
        """프로젝트 완료 알림"""
        return self.send_notification(
            f"프로젝트 완료: {project_name}",
            f"'{project_name}' 프로젝트가 성공적으로 완료되었습니다.",
            NotificationType.PROJECT_COMPLETED,
            user_id=user_id,
            data={'project_id': project_id, 'project_name': project_name},
            priority=NotificationPriority.HIGH,
            **kwargs
        )
    
    def notify_experiment_complete(self, experiment_name: str, experiment_id: str, 
                                 project_id: str, user_id: str, **kwargs) -> str:
        """실험 완료 알림"""
        return self.send_notification(
            f"실험 완료: {experiment_name}",
            f"실험이 완료되었습니다. 결과를 확인하세요.",
            NotificationType.EXPERIMENT_COMPLETED,
            user_id=user_id,
            data={'experiment_id': experiment_id, 'project_id': project_id},
            actions=[
                NotificationAction("결과 보기", f"/experiment/{experiment_id}/results"),
                NotificationAction("분석하기", f"/experiment/{experiment_id}/analyze", "secondary")
            ],
            **kwargs
        )
    
    def notify_ai_complete(self, task: str, result_summary: str = "", user_id: str = None, **kwargs) -> str:
        """AI 작업 완료 알림"""
        message = f"AI {task} 작업이 완료되었습니다."
        if result_summary:
            message += f"\n{result_summary}"
        
        return self.send_notification(
            "AI 작업 완료",
            message,
            NotificationType.AI_COMPLETE,
            user_id=user_id,
            **kwargs
        )
    
    def notify_collaboration(self, action: str, from_user: str, to_user: str, 
                           context: Dict[str, Any], **kwargs) -> str:
        """협업 관련 알림"""
        type_map = {
            'comment': NotificationType.COMMENT_ADDED,
            'mention': NotificationType.MENTIONED,
            'share': NotificationType.FILE_SHARED,
            'assign': NotificationType.TASK_ASSIGNED,
            'invite': NotificationType.COLLABORATOR_ADDED
        }
        
        notification_type = type_map.get(action, NotificationType.INFO)
        
        return self.send_notification(
            f"{from_user}님의 {action}",
            context.get('message', ''),
            notification_type,
            user_id=to_user,
            data={
                'from_user': from_user,
                'action': action,
                **context
            },
            **kwargs
        )


# 싱글톤 인스턴스
_notification_manager: Optional[NotificationManager] = None


def get_notification_manager(db_manager=None) -> NotificationManager:
    """NotificationManager 싱글톤 인스턴스 반환"""
    global _notification_manager
    if _notification_manager is None:
        _notification_manager = NotificationManager(db_manager)
    return _notification_manager


# === Streamlit UI 컴포넌트 ===

def render_notification_center():
    """알림 센터 UI 렌더링"""
    if not STREAMLIT_AVAILABLE:
        return
    
    nm = get_notification_manager()
    user_id = st.session_state.get('user_id')
    
    if not user_id:
        return
    
    # 알림 아이콘과 카운트
    unread_count = nm.get_unread_count(user_id)
    
    with st.container():
        col1, col2 = st.columns([1, 9])
        
        with col1:
            if unread_count > 0:
                st.markdown(f'<div style="position: relative;">🔔 <span style="position: absolute; top: -8px; right: -8px; background: red; color: white; border-radius: 50%; padding: 2px 6px; font-size: 12px;">{unread_count}</span></div>', unsafe_allow_html=True)
            else:
                st.write("🔔")
        
        with col2:
            if st.button("알림 센터", use_container_width=True):
                st.session_state.show_notifications = not st.session_state.get('show_notifications', False)
    
    # 알림 목록 표시
    if st.session_state.get('show_notifications'):
        with st.expander("알림", expanded=True):
            # 액션 버튼
            col1, col2, col3 = st.columns([1, 1, 3])
            with col1:
                if st.button("모두 읽음"):
                    nm.mark_all_as_read(user_id)
                    st.rerun()
            
            with col2:
                if st.button("새로고침"):
                    st.rerun()
            
            # 필터
            with st.container():
                filter_col1, filter_col2 = st.columns(2)
                with filter_col1:
                    show_unread = st.checkbox("읽지 않은 알림만", value=False)
                with filter_col2:
                    category_filter = st.selectbox(
                        "카테고리",
                        ["전체"] + [c.value for c in NotificationCategory],
                        index=0
                    )
            
            # 알림 목록
            notifications = nm.get_notifications(
                user_id,
                limit=20,
                unread_only=show_unread,
                category=NotificationCategory(category_filter) if category_filter != "전체" else None
            )
            
            if notifications:
                for notif in notifications:
                    _render_notification_item(notif, nm, user_id)
            else:
                st.info("알림이 없습니다.")


def _render_notification_item(notification: Dict[str, Any], nm: NotificationManager, user_id: str):
    """개별 알림 아이템 렌더링"""
    # 읽음 여부에 따른 스타일
    style = "background-color: #f0f0f0;" if not notification['read'] else ""
    
    with st.container():
        col1, col2, col3 = st.columns([1, 6, 2])
        
        with col1:
            # 아이콘
            icon_map = {
                'success': '✅',
                'error': '❌',
                'warning': '⚠️',
                'info': 'ℹ️',
                'project': '📁',
                'experiment': '🧪',
                'collaboration': '👥',
                'ai_complete': '🤖'
            }
            icon = icon_map.get(notification['type'], '📌')
            st.write(icon)
        
        with col2:
            # 제목과 메시지
            st.markdown(f"**{notification['title']}**")
            st.write(notification['message'])
            
            # 시간
            created_at = datetime.fromisoformat(notification['created_at'])
            time_ago = _format_time_ago(created_at)
            st.caption(time_ago)
        
        with col3:
            # 액션 버튼
            if not notification['read']:
                if st.button("읽음", key=f"read_{notification['id']}"):
                    nm.mark_as_read(notification['id'], user_id)
                    st.rerun()
            
            if st.button("삭제", key=f"del_{notification['id']}"):
                nm.delete_notification(notification['id'], user_id)
                st.rerun()


def _format_time_ago(timestamp: datetime) -> str:
    """시간 차이를 사람이 읽기 쉬운 형식으로 변환"""
    now = datetime.now()
    diff = now - timestamp
    
    if diff.days > 7:
        return timestamp.strftime("%Y-%m-%d")
    elif diff.days > 0:
        return f"{diff.days}일 전"
    elif diff.seconds > 3600:
        return f"{diff.seconds // 3600}시간 전"
    elif diff.seconds > 60:
        return f"{diff.seconds // 60}분 전"
    else:
        return "방금 전"


def render_notification_settings():
    """알림 설정 UI"""
    if not STREAMLIT_AVAILABLE:
        return
    
    nm = get_notification_manager()
    user_id = st.session_state.get('user_id')
    
    if not user_id:
        st.warning("로그인이 필요합니다.")
        return
    
    settings = nm.get_user_settings(user_id)
    
    st.header("알림 설정")
    
    # 전체 설정
    enabled = st.checkbox("알림 받기", value=settings.enabled)
    if enabled != settings.enabled:
        nm.update_user_settings(user_id, enabled=enabled)
    
    if not enabled:
        st.info("알림이 비활성화되어 있습니다.")
        return
    
    # 채널 설정
    st.subheader("알림 채널")
    
    channels = settings.channels.copy()
    col1, col2 = st.columns(2)
    
    with col1:
        channels['in_app'] = st.checkbox("앱 내 알림", value=channels.get('in_app', True))
        channels['desktop'] = st.checkbox("데스크톱 알림", value=channels.get('desktop', True))
    
    with col2:
        channels['email'] = st.checkbox("이메일", value=channels.get('email', False))
        channels['push'] = st.checkbox("푸시 알림 (준비중)", value=False, disabled=True)
    
    if channels != settings.channels:
        nm.update_user_settings(user_id, channels=channels)
    
    # 카테고리 설정
    st.subheader("알림 카테고리")
    
    categories = settings.categories.copy()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        categories['project'] = st.checkbox("프로젝트", value=categories.get('project', True))
        categories['experiment'] = st.checkbox("실험", value=categories.get('experiment', True))
    
    with col2:
        categories['collaboration'] = st.checkbox("협업", value=categories.get('collaboration', True))
        categories['achievement'] = st.checkbox("성과", value=categories.get('achievement', True))
    
    with col3:
        categories['system'] = st.checkbox("시스템", value=categories.get('system', True))
    
    if categories != settings.categories:
        nm.update_user_settings(user_id, categories=categories)
    
    # 방해금지 설정
    st.subheader("방해금지 모드")
    
    quiet_hours = settings.quiet_hours.copy()
    quiet_enabled = st.checkbox("방해금지 시간 설정", value=quiet_hours.get('enabled', False))
    
    if quiet_enabled:
        col1, col2 = st.columns(2)
        with col1:
            start_time = st.time_input("시작 시간", 
                                      value=datetime.strptime(quiet_hours.get('start', '22:00'), '%H:%M').time())
            quiet_hours['start'] = start_time.strftime('%H:%M')
        
        with col2:
            end_time = st.time_input("종료 시간",
                                    value=datetime.strptime(quiet_hours.get('end', '08:00'), '%H:%M').time())
            quiet_hours['end'] = end_time.strftime('%H:%M')
        
        quiet_hours['allow_urgent'] = st.checkbox("긴급 알림은 허용", 
                                                 value=quiet_hours.get('allow_urgent', True))
    
    quiet_hours['enabled'] = quiet_enabled
    
    if quiet_hours != settings.quiet_hours:
        nm.update_user_settings(user_id, quiet_hours=quiet_hours)
    
    # 이메일 다이제스트
    st.subheader("이메일 다이제스트")
    
    email_digest = settings.email_digest.copy()
    digest_enabled = st.checkbox("이메일 요약 받기", value=email_digest.get('enabled', False))
    
    if digest_enabled:
        frequency = st.selectbox("빈도", 
                               ['immediately', 'hourly', 'daily', 'weekly'],
                               index=['immediately', 'hourly', 'daily', 'weekly'].index(email_digest.get('frequency', 'daily')))
        email_digest['frequency'] = frequency
        
        if frequency in ['daily', 'weekly']:
            digest_time = st.time_input("전송 시간",
                                       value=datetime.strptime(email_digest.get('time', '09:00'), '%H:%M').time())
            email_digest['time'] = digest_time.strftime('%H:%M')
    
    email_digest['enabled'] = digest_enabled
    
    if email_digest != settings.email_digest:
        nm.update_user_settings(user_id, email_digest=email_digest)
    
    # 저장 완료 메시지
    if st.button("설정 저장"):
        st.success("알림 설정이 저장되었습니다.")


# === 테스트 및 예제 ===

if __name__ == "__main__":
    # 테스트 코드
    logging.basicConfig(level=logging.INFO)
    
    # 매니저 생성
    nm = get_notification_manager()
    
    # 테스트 알림 전송
    test_user = "test_user_123"
    
    # 다양한 알림 테스트
    nm.notify_success("테스트 성공", "알림 시스템이 정상 작동합니다.", user_id=test_user)
    
    nm.notify_project_complete("고분자 합성 프로젝트", "proj_123", test_user)
    
    nm.notify_experiment_complete("용매 최적화 실험", "exp_456", "proj_123", test_user)
    
    nm.notify_collaboration("comment", "김연구원", test_user, {
        'message': "실험 결과가 흥미롭네요!",
        'project_id': "proj_123"
    })
    
    # 통계 확인
    import pprint
    stats = nm.get_notification_stats(test_user, days=7)
    pprint.pprint(stats)
    
    print("알림 시스템 테스트 완료!")
