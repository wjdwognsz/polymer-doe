"""
온/오프라인 동기화 관리자
SQLite(로컬)와 Google Sheets(클라우드) 간 데이터 동기화를 담당합니다.
"""
import threading
import time
import logging
import requests
import queue
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json
from dataclasses import dataclass, asdict
from pathlib import Path

from config.offline_config import (
    SYNC_CONFIGURATION, 
    SyncStrategy, 
    get_sync_strategy,
    get_sync_priority
)
from config.local_config import LOCAL_CONFIG

logger = logging.getLogger(__name__)


# ============================================================================
# 데이터 모델
# ============================================================================

class SyncStatus(Enum):
    """동기화 상태"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CONFLICT = "conflict"


@dataclass
class SyncItem:
    """동기화 항목"""
    id: Optional[int] = None
    table_name: str = ""
    record_id: int = 0
    action: str = ""  # insert, update, delete
    data: Dict[str, Any] = None
    local_timestamp: datetime = None
    sync_status: SyncStatus = SyncStatus.PENDING
    priority: int = 5
    retry_count: int = 0
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}
        if self.local_timestamp is None:
            self.local_timestamp = datetime.now()
        # 우선순위 자동 설정
        if self.table_name:
            self.priority = get_sync_priority(self.table_name)


@dataclass
class SyncConflict:
    """동기화 충돌 정보"""
    table_name: str
    record_id: int
    local_data: Dict[str, Any]
    remote_data: Dict[str, Any]
    local_timestamp: datetime
    remote_timestamp: datetime
    resolution_strategy: SyncStrategy
    resolved: bool = False
    resolution: Optional[Dict[str, Any]] = None


# ============================================================================
# 메인 동기화 관리자
# ============================================================================

class SyncManager:
    """온/오프라인 동기화 관리자"""
    
    def __init__(self, db_manager, sheets_manager=None):
        """
        초기화
        
        Args:
            db_manager: DatabaseManager 인스턴스
            sheets_manager: GoogleSheetsManager 인스턴스 (선택)
        """
        self.db_manager = db_manager
        self.sheets_manager = sheets_manager
        self.config = SYNC_CONFIGURATION
        
        # 상태 관리
        self.is_online = False
        self.last_sync = {}  # {table_name: datetime}
        self.sync_in_progress = False
        
        # 동기화 큐 (우선순위 큐)
        self.sync_queue = queue.PriorityQueue()
        self.conflict_queue = queue.Queue()
        
        # 스레드 관리
        self.sync_thread = None
        self.monitor_thread = None
        self._stop_event = threading.Event()
        
        # 콜백 함수들
        self.callbacks = {
            'on_online': [],
            'on_offline': [],
            'on_sync_start': [],
            'on_sync_complete': [],
            'on_sync_error': [],
            'on_conflict': []
        }
        
        # 동기화 통계
        self.stats = {
            'total_synced': 0,
            'failed_syncs': 0,
            'conflicts_resolved': 0,
            'last_successful_sync': None
        }
        
        # 시작
        self._initialize()
    
    def _initialize(self):
        """초기화 작업"""
        # 오프라인 모드로 시작
        self.is_online = False
        
        # 대기 중인 동기화 항목 로드
        self._load_pending_syncs()
        
        # 자동 시작 설정 확인
        if self.config['auto_sync']['enabled']:
            self.start()
    
    # ============================================================================
    # 서비스 제어
    # ============================================================================
    
    def start(self):
        """동기화 서비스 시작"""
        if self.sync_thread and self.sync_thread.is_alive():
            logger.warning("Sync service is already running")
            return
        
        self._stop_event.clear()
        
        # 연결 모니터링 스레드
        self.monitor_thread = threading.Thread(
            target=self._connection_monitor_loop,
            daemon=True,
            name="SyncMonitor"
        )
        self.monitor_thread.start()
        
        # 동기화 처리 스레드
        self.sync_thread = threading.Thread(
            target=self._sync_loop,
            daemon=True,
            name="SyncWorker"
        )
        self.sync_thread.start()
        
        logger.info("Sync service started")
    
    def stop(self):
        """동기화 서비스 중지"""
        logger.info("Stopping sync service...")
        self._stop_event.set()
        
        # 스레드 종료 대기
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        if self.sync_thread:
            self.sync_thread.join(timeout=10)
        
        # 종료 시 동기화 처리
        if self.config['auto_sync']['on_app_close']:
            self.sync_all()
        
        logger.info("Sync service stopped")
    
    # ============================================================================
    # 연결 상태 관리
    # ============================================================================
    
    def check_connection(self) -> bool:
        """인터넷 연결 확인"""
        test_urls = [
            'https://www.google.com',
            'https://sheets.googleapis.com',
            'https://1.1.1.1'
        ]
        
        for url in test_urls:
            try:
                response = requests.get(url, timeout=3)
                if response.status_code == 200:
                    return True
            except:
                continue
        
        return False
    
    def _connection_monitor_loop(self):
        """연결 상태 모니터링 루프"""
        check_interval = LOCAL_CONFIG['offline_mode']['check_interval']
        
        while not self._stop_event.is_set():
            try:
                was_online = self.is_online
                self.is_online = self.check_connection()
                
                # 상태 변경 감지
                if was_online != self.is_online:
                    self._handle_connection_change(was_online, self.is_online)
                
            except Exception as e:
                logger.error(f"Connection monitor error: {str(e)}")
            
            # 대기
            self._stop_event.wait(check_interval)
    
    def _handle_connection_change(self, was_online: bool, is_online: bool):
        """연결 상태 변경 처리"""
        if is_online and not was_online:
            # 오프라인 → 온라인
            logger.info("Connection restored")
            self._trigger_callbacks('on_online')
            
            # 자동 동기화 시작
            if self.config['auto_sync']['on_connection_restore']:
                self._schedule_sync(priority=1)
        
        elif not is_online and was_online:
            # 온라인 → 오프라인
            logger.info("Connection lost")
            self._trigger_callbacks('on_offline')
    
    # ============================================================================
    # 동기화 처리
    # ============================================================================
    
    def _sync_loop(self):
        """동기화 처리 루프"""
        while not self._stop_event.is_set():
            try:
                # 큐에서 항목 가져오기 (우선순위 순)
                priority, sync_item = self.sync_queue.get(timeout=1)
                
                if self.is_online and self.sheets_manager:
                    self._process_sync_item(sync_item)
                else:
                    # 오프라인이면 다시 큐에 추가
                    self.sync_queue.put((priority, sync_item))
                    time.sleep(5)  # 대기
                
            except queue.Empty:
                # 주기적 동기화 확인
                self._check_periodic_sync()
            except Exception as e:
                logger.error(f"Sync loop error: {str(e)}")
    
    def _process_sync_item(self, item: SyncItem):
        """개별 동기화 항목 처리"""
        item.sync_status = SyncStatus.IN_PROGRESS
        self._trigger_callbacks('on_sync_start', item)
        
        try:
            # 액션별 처리
            if item.action == 'insert':
                success = self._sync_insert(item)
            elif item.action == 'update':
                success = self._sync_update(item)
            elif item.action == 'delete':
                success = self._sync_delete(item)
            else:
                raise ValueError(f"Unknown action: {item.action}")
            
            if success:
                # 성공
                item.sync_status = SyncStatus.COMPLETED
                self.db_manager.mark_synced(item.id, success=True)
                self.stats['total_synced'] += 1
                self.stats['last_successful_sync'] = datetime.now()
                self._trigger_callbacks('on_sync_complete', item)
            else:
                # 실패
                self._handle_sync_failure(item)
                
        except Exception as e:
            logger.error(f"Sync processing error: {str(e)}")
            item.error_message = str(e)
            self._handle_sync_failure(item)
    
    def _sync_insert(self, item: SyncItem) -> bool:
        """INSERT 동기화"""
        try:
            # Sheets에 새 행 추가
            result = self.sheets_manager.append_row(
                sheet_name=item.table_name,
                values=item.data
            )
            
            # 원격 ID 업데이트
            if result and 'id' in result:
                self.db_manager.update_sync_mapping(
                    table=item.table_name,
                    local_id=item.record_id,
                    remote_id=result['id']
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Insert sync failed: {str(e)}")
            return False
    
    def _sync_update(self, item: SyncItem) -> bool:
        """UPDATE 동기화"""
        try:
            # 충돌 확인
            remote_data = self.sheets_manager.get_row(
                sheet_name=item.table_name,
                row_id=item.record_id
            )
            
            if remote_data:
                # 타임스탬프 비교
                conflict = self._detect_conflict(item, remote_data)
                if conflict:
                    self._handle_conflict(conflict)
                    return False
            
            # 업데이트 실행
            success = self.sheets_manager.update_row(
                sheet_name=item.table_name,
                row_id=item.record_id,
                values=item.data
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Update sync failed: {str(e)}")
            return False
    
    def _sync_delete(self, item: SyncItem) -> bool:
        """DELETE 동기화"""
        try:
            # Sheets에서 삭제 (soft delete)
            success = self.sheets_manager.soft_delete_row(
                sheet_name=item.table_name,
                row_id=item.record_id
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Delete sync failed: {str(e)}")
            return False
    
    # ============================================================================
    # 충돌 처리
    # ============================================================================
    
    def _detect_conflict(self, item: SyncItem, remote_data: Dict) -> Optional[SyncConflict]:
        """충돌 감지"""
        # 타임스탬프 비교
        local_time = item.data.get('updated_at', item.local_timestamp)
        remote_time = remote_data.get('updated_at')
        
        if not remote_time:
            return None
        
        # 원격이 더 최신인 경우 충돌
        if isinstance(remote_time, str):
            remote_time = datetime.fromisoformat(remote_time)
        
        if remote_time > local_time:
            return SyncConflict(
                table_name=item.table_name,
                record_id=item.record_id,
                local_data=item.data,
                remote_data=remote_data,
                local_timestamp=local_time,
                remote_timestamp=remote_time,
                resolution_strategy=get_sync_strategy(item.table_name)
            )
        
        return None
    
    def _handle_conflict(self, conflict: SyncConflict):
        """충돌 처리"""
        logger.warning(f"Sync conflict detected: {conflict.table_name}#{conflict.record_id}")
        
        # 전략별 해결
        if conflict.resolution_strategy == SyncStrategy.NEWEST_WINS:
            # 최신 데이터 선택
            if conflict.local_timestamp > conflict.remote_timestamp:
                conflict.resolution = conflict.local_data
            else:
                conflict.resolution = conflict.remote_data
            conflict.resolved = True
            
        elif conflict.resolution_strategy == SyncStrategy.LOCAL_FIRST:
            # 로컬 우선
            conflict.resolution = conflict.local_data
            conflict.resolved = True
            
        elif conflict.resolution_strategy == SyncStrategy.REMOTE_FIRST:
            # 원격 우선
            conflict.resolution = conflict.remote_data
            conflict.resolved = True
            
        elif conflict.resolution_strategy == SyncStrategy.MERGE:
            # 병합 시도
            conflict.resolution = self._merge_data(
                conflict.local_data,
                conflict.remote_data
            )
            conflict.resolved = True
            
        else:  # MANUAL
            # 수동 해결 필요
            self.conflict_queue.put(conflict)
            self._trigger_callbacks('on_conflict', conflict)
            return
        
        # 자동 해결된 경우 적용
        if conflict.resolved:
            self._apply_conflict_resolution(conflict)
            self.stats['conflicts_resolved'] += 1
    
    def _merge_data(self, local: Dict, remote: Dict) -> Dict:
        """데이터 병합"""
        merged = remote.copy()
        
        # 기본 병합 규칙
        for key, value in local.items():
            if key in ['updated_at', 'sync_version']:
                continue
            
            # 리스트는 합치기
            if isinstance(value, list) and isinstance(remote.get(key), list):
                merged[key] = list(set(value + remote[key]))
            # 딕셔너리는 업데이트
            elif isinstance(value, dict) and isinstance(remote.get(key), dict):
                merged[key] = {**remote[key], **value}
            # 나머지는 로컬 값 사용
            else:
                merged[key] = value
        
        merged['updated_at'] = datetime.now().isoformat()
        return merged
    
    def _apply_conflict_resolution(self, conflict: SyncConflict):
        """충돌 해결 적용"""
        if not conflict.resolved or not conflict.resolution:
            return
        
        # 로컬 업데이트
        self.db_manager.update_record(
            table=conflict.table_name,
            record_id=conflict.record_id,
            data=conflict.resolution
        )
        
        # 원격 업데이트
        if self.sheets_manager:
            self.sheets_manager.update_row(
                sheet_name=conflict.table_name,
                row_id=conflict.record_id,
                values=conflict.resolution
            )
    
    # ============================================================================
    # 일괄 동기화
    # ============================================================================
    
    def sync_all(self, tables: Optional[List[str]] = None):
        """전체 동기화 실행"""
        if not self.is_online or not self.sheets_manager:
            logger.warning("Cannot sync: offline or no sheets manager")
            return
        
        logger.info("Starting full sync...")
        self.sync_in_progress = True
        
        try:
            # 대기 중인 모든 변경사항 가져오기
            pending_syncs = self.db_manager.get_pending_sync(limit=1000)
            
            # 테이블 필터링
            if tables:
                pending_syncs = [s for s in pending_syncs if s['table_name'] in tables]
            
            # 우선순위별 정렬 후 큐에 추가
            for sync_data in pending_syncs:
                item = SyncItem(**sync_data)
                self.add_to_queue(item)
            
            logger.info(f"Added {len(pending_syncs)} items to sync queue")
            
        finally:
            self.sync_in_progress = False
    
    def sync_table(self, table_name: str):
        """특정 테이블 동기화"""
        self.sync_all(tables=[table_name])
    
    # ============================================================================
    # 동기화 큐 관리
    # ============================================================================
    
    def add_to_queue(self, item: SyncItem):
        """동기화 큐에 추가"""
        # 중복 확인
        if self._is_duplicate(item):
            return
        
        # 큐 크기 확인
        max_size = self.config['queue_management']['max_queue_size']
        if self.sync_queue.qsize() >= max_size:
            logger.warning(f"Sync queue is full ({max_size} items)")
            return
        
        # 우선순위 큐에 추가 (낮은 숫자가 높은 우선순위)
        self.sync_queue.put((item.priority, item))
    
    def _is_duplicate(self, item: SyncItem) -> bool:
        """중복 항목 확인"""
        # 간단한 중복 확인 (실제로는 더 정교한 로직 필요)
        return False
    
    def clear_queue(self):
        """동기화 큐 비우기"""
        while not self.sync_queue.empty():
            try:
                self.sync_queue.get_nowait()
            except queue.Empty:
                break
    
    def get_queue_status(self) -> Dict[str, Any]:
        """큐 상태 조회"""
        return {
            'size': self.sync_queue.qsize(),
            'is_syncing': self.sync_in_progress,
            'conflicts_pending': self.conflict_queue.qsize(),
            'stats': self.stats.copy()
        }
    
    # ============================================================================
    # 주기적 동기화
    # ============================================================================
    
    def _check_periodic_sync(self):
        """주기적 동기화 확인"""
        if not self.config['auto_sync']['enabled']:
            return
        
        interval = self.config['auto_sync']['interval']
        
        for table_name in self.config['sync_priorities']:
            last_sync = self.last_sync.get(table_name)
            
            if not last_sync or (datetime.now() - last_sync) > interval:
                # 동기화 예약
                self._schedule_sync(table=table_name, priority=5)
                self.last_sync[table_name] = datetime.now()
    
    def _schedule_sync(self, table: Optional[str] = None, priority: int = 5):
        """동기화 예약"""
        if table:
            self.sync_table(table)
        else:
            self.sync_all()
    
    # ============================================================================
    # 실패 처리
    # ============================================================================
    
    def _handle_sync_failure(self, item: SyncItem):
        """동기화 실패 처리"""
        item.sync_status = SyncStatus.FAILED
        item.retry_count += 1
        
        # 재시도 여부 확인
        max_retries = self.config['queue_management']['max_retries']
        
        if item.retry_count < max_retries:
            # 재시도 스케줄
            retry_delay = self.config['queue_management']['retry_delay']
            threading.Timer(
                retry_delay.total_seconds(),
                lambda: self.add_to_queue(item)
            ).start()
            
            logger.info(f"Scheduled retry {item.retry_count}/{max_retries} for sync item")
        else:
            # 최대 재시도 초과
            logger.error(f"Max retries exceeded for sync item: {item.table_name}#{item.record_id}")
            self.db_manager.mark_synced(item.id, success=False, error_message=item.error_message)
            self.stats['failed_syncs'] += 1
            self._trigger_callbacks('on_sync_error', item)
    
    def _load_pending_syncs(self):
        """시작 시 대기 중인 동기화 로드"""
        try:
            pending = self.db_manager.get_pending_sync()
            for sync_data in pending:
                item = SyncItem(**sync_data)
                self.add_to_queue(item)
            
            logger.info(f"Loaded {len(pending)} pending sync items")
        except Exception as e:
            logger.error(f"Failed to load pending syncs: {str(e)}")
    
    # ============================================================================
    # 콜백 관리
    # ============================================================================
    
    def register_callback(self, event: str, callback: Callable):
        """콜백 함수 등록"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
        else:
            logger.warning(f"Unknown event: {event}")
    
    def _trigger_callbacks(self, event: str, *args, **kwargs):
        """콜백 실행"""
        for callback in self.callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Callback error ({event}): {str(e)}")
    
    # ============================================================================
    # 수동 충돌 해결
    # ============================================================================
    
    def get_pending_conflicts(self) -> List[SyncConflict]:
        """대기 중인 충돌 목록"""
        conflicts = []
        while not self.conflict_queue.empty():
            try:
                conflict = self.conflict_queue.get_nowait()
                conflicts.append(conflict)
            except queue.Empty:
                break
        
        # 다시 큐에 넣기
        for conflict in conflicts:
            self.conflict_queue.put(conflict)
        
        return conflicts
    
    def resolve_conflict(self, conflict: SyncConflict, resolution: Dict[str, Any]):
        """수동 충돌 해결"""
        conflict.resolution = resolution
        conflict.resolved = True
        self._apply_conflict_resolution(conflict)
        
        # 통계 업데이트
        self.stats['conflicts_resolved'] += 1
    
    # ============================================================================
    # 유틸리티
    # ============================================================================
    
    def force_sync(self):
        """강제 동기화 (온라인 상태 무시)"""
        was_online = self.is_online
        self.is_online = True
        
        try:
            self.sync_all()
        finally:
            self.is_online = was_online
    
    def export_sync_log(self, filepath: Path):
        """동기화 로그 내보내기"""
        logs = self.db_manager.get_sync_history(limit=10000)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2, default=str)
        
        logger.info(f"Exported sync log to {filepath}")
    
    def reset_sync_state(self):
        """동기화 상태 초기화"""
        logger.warning("Resetting sync state...")
        
        # 큐 비우기
        self.clear_queue()
        
        # 통계 초기화
        self.stats = {
            'total_synced': 0,
            'failed_syncs': 0,
            'conflicts_resolved': 0,
            'last_successful_sync': None
        }
        
        # 동기화 로그 초기화
        self.db_manager.clear_sync_log()
        
        logger.info("Sync state reset complete")


# ============================================================================
# 싱글톤 인스턴스
# ============================================================================

_sync_manager: Optional[SyncManager] = None


def get_sync_manager(db_manager=None, sheets_manager=None) -> SyncManager:
    """SyncManager 싱글톤 인스턴스 반환"""
    global _sync_manager
    
    if _sync_manager is None:
        if db_manager is None:
            raise ValueError("db_manager is required for initial creation")
        _sync_manager = SyncManager(db_manager, sheets_manager)
    
    # Sheets 매니저 업데이트 (나중에 설정 가능)
    if sheets_manager and _sync_manager.sheets_manager is None:
        _sync_manager.sheets_manager = sheets_manager
        logger.info("Sheets manager attached to sync manager")
    
    return _sync_manager


# ============================================================================
# 테스트 및 디버깅
# ============================================================================

if __name__ == "__main__":
    # 단독 실행 시 기본 테스트
    logging.basicConfig(level=logging.INFO)
    
    print("SyncManager 모듈 로드 완료")
    print(f"동기화 우선순위: {SYNC_CONFIGURATION['sync_priorities']}")
    print(f"자동 동기화 설정: {SYNC_CONFIGURATION['auto_sync']}")
