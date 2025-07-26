# utils/sync_manager.py
"""
온/오프라인 동기화 관리자
SQLite(로컬)와 Google Sheets(클라우드) 간 데이터 동기화를 담당합니다.
오프라인 우선 설계로 로컬 작업을 항상 보장하고, 온라인 시 자동 동기화합니다.
"""

import threading
import time
import logging
import requests
import queue
import hashlib
from typing import Dict, List, Optional, Any, Callable, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import traceback

# Streamlit
import streamlit as st

# 내부 모듈
try:
    from config.offline_config import (
        SYNC_CONFIGURATION, 
        SyncStrategy, 
        get_sync_strategy,
        get_sync_priority
    )
    from config.local_config import LOCAL_CONFIG
except ImportError:
    # 기본값 설정
    SYNC_CONFIGURATION = {
        'auto_sync': {
            'enabled': True,
            'interval': timedelta(minutes=5),
            'on_startup': True,
            'on_connection_restore': True,
            'on_app_close': True
        },
        'sync_priorities': {
            'users': 1,
            'projects': 2,
            'experiments': 3,
            'results': 4,
            'activity_log': 5
        },
        'conflict_resolution': {
            'default_strategy': 'REMOTE_WINS',
            'custom_strategies': {}
        },
        'queue_management': {
            'max_queue_size': 1000,
            'batch_size': 50,
            'max_retries': 3,
            'retry_delay': timedelta(seconds=5)
        }
    }
    
    class SyncStrategy(Enum):
        LOCAL_WINS = "local_wins"
        REMOTE_WINS = "remote_wins"
        MERGE = "merge"
        MANUAL = "manual"
    
    def get_sync_strategy(table_name: str) -> SyncStrategy:
        return SyncStrategy.REMOTE_WINS
    
    def get_sync_priority(table_name: str) -> int:
        return SYNC_CONFIGURATION['sync_priorities'].get(table_name, 10)
    
    LOCAL_CONFIG = {
        'offline_mode': {
            'check_interval': 10  # seconds
        }
    }

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
    CANCELLED = "cancelled"


class SyncDirection(Enum):
    """동기화 방향"""
    LOCAL_TO_REMOTE = "local_to_remote"
    REMOTE_TO_LOCAL = "remote_to_local"
    BIDIRECTIONAL = "bidirectional"


@dataclass
class SyncItem:
    """동기화 항목"""
    id: Optional[int] = None
    table_name: str = ""
    record_id: int = 0
    action: str = ""  # insert, update, delete
    data: Dict[str, Any] = field(default_factory=dict)
    local_timestamp: datetime = field(default_factory=datetime.now)
    sync_status: SyncStatus = SyncStatus.PENDING
    priority: int = 5
    retry_count: int = 0
    error_message: Optional[str] = None
    hash: Optional[str] = None
    
    def __post_init__(self):
        # 우선순위 자동 설정
        if self.table_name:
            self.priority = get_sync_priority(self.table_name)
        
        # 해시 생성 (중복 확인용)
        if not self.hash:
            self.hash = self._generate_hash()
    
    def _generate_hash(self) -> str:
        """동기화 항목의 고유 해시 생성"""
        data_str = f"{self.table_name}:{self.record_id}:{self.action}:{json.dumps(self.data, sort_keys=True)}"
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def __lt__(self, other):
        """우선순위 큐 정렬을 위한 비교"""
        return self.priority < other.priority


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
    conflict_fields: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        # 충돌 필드 자동 감지
        if not self.conflict_fields:
            self.conflict_fields = self._detect_conflicts()
    
    def _detect_conflicts(self) -> List[str]:
        """충돌 필드 감지"""
        conflicts = []
        for key in set(self.local_data.keys()) | set(self.remote_data.keys()):
            if self.local_data.get(key) != self.remote_data.get(key):
                conflicts.append(key)
        return conflicts


@dataclass
class SyncProgress:
    """동기화 진행 상황"""
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    conflicts: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    current_table: Optional[str] = None
    current_action: Optional[str] = None
    
    @property
    def progress_percentage(self) -> float:
        """진행률 계산"""
        if self.total_items == 0:
            return 0.0
        return (self.completed_items / self.total_items) * 100
    
    @property
    def elapsed_time(self) -> timedelta:
        """경과 시간"""
        end = self.end_time or datetime.now()
        return end - self.start_time
    
    @property
    def is_complete(self) -> bool:
        """완료 여부"""
        return self.completed_items + self.failed_items >= self.total_items


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
        self.current_progress = SyncProgress()
        
        # 동기화 큐 (우선순위 큐)
        self.sync_queue = queue.PriorityQueue()
        self.conflict_queue = queue.Queue()
        self.processing_hashes = set()  # 처리 중인 항목 해시
        
        # 스레드 관리
        self.sync_thread = None
        self.monitor_thread = None
        self._stop_event = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # 콜백 함수들
        self.callbacks = {
            'on_online': [],
            'on_offline': [],
            'on_sync_start': [],
            'on_sync_complete': [],
            'on_sync_error': [],
            'on_conflict': [],
            'on_progress': []
        }
        
        # 동기화 통계
        self.stats = {
            'total_synced': 0,
            'failed_syncs': 0,
            'conflicts_resolved': 0,
            'last_successful_sync': None,
            'sync_history': []  # 최근 100개 동기화 기록
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
            logger.warning("동기화 서비스가 이미 실행 중입니다")
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
        
        logger.info("동기화 서비스 시작됨")
    
    def stop(self):
        """동기화 서비스 중지"""
        logger.info("동기화 서비스 중지 중...")
        self._stop_event.set()
        
        # 스레드 종료 대기
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        if self.sync_thread:
            self.sync_thread.join(timeout=10)
        
        # Executor 종료
        self.executor.shutdown(wait=True)
        
        # 종료 시 동기화 처리
        if self.config['auto_sync']['on_app_close']:
            self.sync_all()
        
        logger.info("동기화 서비스 중지됨")
    
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
                logger.error(f"연결 모니터 오류: {str(e)}")
            
            # 대기
            self._stop_event.wait(check_interval)
    
    def _handle_connection_change(self, was_online: bool, is_online: bool):
        """연결 상태 변경 처리"""
        if is_online and not was_online:
            # 오프라인 → 온라인
            logger.info("연결 복원됨")
            self._trigger_callbacks('on_online')
            
            # 자동 동기화 시작
            if self.config['auto_sync']['on_connection_restore']:
                self._schedule_sync(priority=1)
        
        elif not is_online and was_online:
            # 온라인 → 오프라인
            logger.info("연결 끊김")
            self._trigger_callbacks('on_offline')
    
    # ============================================================================
    # 동기화 처리
    # ============================================================================
    
    def _sync_loop(self):
        """동기화 처리 루프"""
        while not self._stop_event.is_set():
            try:
                # 큐에서 항목 가져오기 (우선순위 순)
                sync_item = self.sync_queue.get(timeout=1)
                
                if self.is_online and self.sheets_manager:
                    self._process_sync_item(sync_item)
                else:
                    # 오프라인이면 다시 큐에 추가
                    self.sync_queue.put(sync_item)
                    time.sleep(5)  # 대기
                
            except queue.Empty:
                # 주기적 동기화 확인
                self._check_periodic_sync()
            except Exception as e:
                logger.error(f"동기화 루프 오류: {str(e)}")
                logger.error(traceback.format_exc())
    
    def _process_sync_item(self, item: SyncItem):
        """개별 동기화 항목 처리"""
        # 중복 처리 방지
        if item.hash in self.processing_hashes:
            return
        
        self.processing_hashes.add(item.hash)
        
        try:
            item.sync_status = SyncStatus.IN_PROGRESS
            self._trigger_callbacks('on_sync_start', item)
            
            # 진행률 업데이트
            self.current_progress.current_table = item.table_name
            self.current_progress.current_action = item.action
            
            # 액션별 처리
            if item.action == 'insert':
                success = self._sync_insert(item)
            elif item.action == 'update':
                success = self._sync_update(item)
            elif item.action == 'delete':
                success = self._sync_delete(item)
            else:
                raise ValueError(f"알 수 없는 액션: {item.action}")
            
            if success:
                # 성공
                item.sync_status = SyncStatus.COMPLETED
                self.db_manager.mark_synced(item.id, success=True)
                self.stats['total_synced'] += 1
                self.stats['last_successful_sync'] = datetime.now()
                self.current_progress.completed_items += 1
                self._trigger_callbacks('on_sync_complete', item)
            else:
                # 실패
                self._handle_sync_failure(item)
                
        except Exception as e:
            logger.error(f"동기화 처리 오류: {str(e)}")
            item.error_message = str(e)
            self._handle_sync_failure(item)
        finally:
            self.processing_hashes.discard(item.hash)
            self._trigger_callbacks('on_progress', self.current_progress)
    
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
            logger.error(f"Insert 동기화 실패: {str(e)}")
            return False
    
    def _sync_update(self, item: SyncItem) -> bool:
        """UPDATE 동기화"""
        try:
            # 충돌 확인
            remote_data = self.sheets_manager.get_row(
                sheet_name=item.table_name,
                row_id=item.record_id
            )
            
            if remote_data and self._has_conflict(item.data, remote_data):
                # 충돌 처리
                conflict = SyncConflict(
                    table_name=item.table_name,
                    record_id=item.record_id,
                    local_data=item.data,
                    remote_data=remote_data,
                    local_timestamp=item.local_timestamp,
                    remote_timestamp=remote_data.get('updated_at', datetime.now()),
                    resolution_strategy=get_sync_strategy(item.table_name)
                )
                
                resolved_data = self._resolve_conflict(conflict)
                if resolved_data:
                    item.data = resolved_data
                else:
                    # 수동 해결 필요
                    self.conflict_queue.put(conflict)
                    self.current_progress.conflicts += 1
                    return False
            
            # Sheets 업데이트
            success = self.sheets_manager.update_row(
                sheet_name=item.table_name,
                row_id=item.record_id,
                values=item.data
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Update 동기화 실패: {str(e)}")
            return False
    
    def _sync_delete(self, item: SyncItem) -> bool:
        """DELETE 동기화"""
        try:
            # Sheets에서 삭제
            success = self.sheets_manager.delete_row(
                sheet_name=item.table_name,
                row_id=item.record_id
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Delete 동기화 실패: {str(e)}")
            return False
    
    # ============================================================================
    # 충돌 처리
    # ============================================================================
    
    def _has_conflict(self, local_data: Dict, remote_data: Dict) -> bool:
        """충돌 여부 확인"""
        # 타임스탬프 비교
        local_ts = local_data.get('updated_at')
        remote_ts = remote_data.get('updated_at')
        
        if local_ts and remote_ts:
            # 같은 시간에 업데이트된 경우 내용 비교
            if abs((local_ts - remote_ts).total_seconds()) < 1:
                return False
            
            # 원격이 더 최신인데 내용이 다른 경우
            if remote_ts > local_ts:
                for key in local_data:
                    if key not in ['updated_at', 'sync_status'] and \
                       local_data.get(key) != remote_data.get(key):
                        return True
        
        return False
    
    def _resolve_conflict(self, conflict: SyncConflict) -> Optional[Dict[str, Any]]:
        """충돌 해결"""
        strategy = conflict.resolution_strategy
        
        if strategy == SyncStrategy.LOCAL_WINS:
            return conflict.local_data
        
        elif strategy == SyncStrategy.REMOTE_WINS:
            return conflict.remote_data
        
        elif strategy == SyncStrategy.MERGE:
            # 필드별 병합
            merged = conflict.remote_data.copy()
            
            # 로컬에서 변경된 필드만 업데이트
            for field in conflict.conflict_fields:
                if field not in ['id', 'created_at', 'sync_status']:
                    # 더 최신 데이터 사용
                    local_ts = conflict.local_data.get('updated_at', datetime.min)
                    remote_ts = conflict.remote_data.get('updated_at', datetime.min)
                    
                    if local_ts > remote_ts:
                        merged[field] = conflict.local_data.get(field)
            
            return merged
        
        elif strategy == SyncStrategy.MANUAL:
            # 수동 해결 필요
            self._trigger_callbacks('on_conflict', conflict)
            return None
        
        return None
    
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
        
        self.stats['conflicts_resolved'] += 1
    
    # ============================================================================
    # 일괄 동기화
    # ============================================================================
    
    def sync_all(self, tables: Optional[List[str]] = None):
        """전체 동기화 실행"""
        if not self.is_online or not self.sheets_manager:
            logger.warning("동기화 불가: 오프라인이거나 Sheets 매니저가 없음")
            return
        
        logger.info("전체 동기화 시작...")
        self.sync_in_progress = True
        
        # 새 진행 상황 추적 시작
        self.current_progress = SyncProgress()
        
        try:
            # 대기 중인 모든 변경사항 가져오기
            pending_syncs = self.db_manager.get_pending_sync(limit=1000)
            
            # 테이블 필터링
            if tables:
                pending_syncs = [s for s in pending_syncs if s['table_name'] in tables]
            
            self.current_progress.total_items = len(pending_syncs)
            
            # 우선순위별 정렬 후 큐에 추가
            for sync_data in pending_syncs:
                item = SyncItem(**sync_data)
                self.add_to_queue(item)
            
            logger.info(f"{len(pending_syncs)}개 항목을 동기화 큐에 추가")
            
            # 배치 처리
            self._process_batch()
            
        finally:
            self.sync_in_progress = False
            self.current_progress.end_time = datetime.now()
            self._add_to_history()
    
    def sync_table(self, table_name: str):
        """특정 테이블 동기화"""
        self.sync_all(tables=[table_name])
    
    def _process_batch(self):
        """배치 처리"""
        batch_size = self.config['queue_management']['batch_size']
        batch = []
        
        while not self.sync_queue.empty() and len(batch) < batch_size:
            try:
                item = self.sync_queue.get_nowait()
                batch.append(item)
            except queue.Empty:
                break
        
        # 병렬 처리
        if batch:
            futures = []
            for item in batch:
                future = self.executor.submit(self._process_sync_item, item)
                futures.append(future)
            
            # 결과 대기
            for future in futures:
                try:
                    future.result(timeout=30)
                except Exception as e:
                    logger.error(f"배치 처리 오류: {e}")
    
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
            logger.warning(f"동기화 큐 가득참 ({max_size} 항목)")
            # 오래된 항목 제거
            self._cleanup_queue()
        
        # 우선순위 큐에 추가
        self.sync_queue.put(item)
    
    def _is_duplicate(self, item: SyncItem) -> bool:
        """중복 항목 확인"""
        return item.hash in self.processing_hashes
    
    def _cleanup_queue(self):
        """큐 정리 (오래된 항목 제거)"""
        temp_items = []
        
        # 큐에서 모든 항목 꺼내기
        while not self.sync_queue.empty():
            try:
                temp_items.append(self.sync_queue.get_nowait())
            except queue.Empty:
                break
        
        # 우선순위 높은 항목만 다시 추가
        temp_items.sort(key=lambda x: x.priority)
        max_size = self.config['queue_management']['max_queue_size']
        
        for item in temp_items[:max_size-100]:  # 100개 여유 공간
            self.sync_queue.put(item)
    
    def clear_queue(self):
        """동기화 큐 비우기"""
        while not self.sync_queue.empty():
            try:
                self.sync_queue.get_nowait()
            except queue.Empty:
                break
        
        self.processing_hashes.clear()
    
    def get_queue_status(self) -> Dict[str, Any]:
        """큐 상태 조회"""
        return {
            'size': self.sync_queue.qsize(),
            'is_syncing': self.sync_in_progress,
            'conflicts_pending': self.conflict_queue.qsize(),
            'stats': self.stats.copy(),
            'progress': asdict(self.current_progress),
            'is_online': self.is_online
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
            
            logger.info(f"재시도 예정 {item.retry_count}/{max_retries}: {item.table_name}#{item.record_id}")
        else:
            # 최대 재시도 초과
            logger.error(f"최대 재시도 초과: {item.table_name}#{item.record_id}")
            self.db_manager.mark_synced(item.id, success=False, error_message=item.error_message)
            self.stats['failed_syncs'] += 1
            self.current_progress.failed_items += 1
            self._trigger_callbacks('on_sync_error', item)
    
    def _load_pending_syncs(self):
        """시작 시 대기 중인 동기화 로드"""
        try:
            pending = self.db_manager.get_pending_sync()
            for sync_data in pending:
                item = SyncItem(**sync_data)
                self.add_to_queue(item)
            
            logger.info(f"{len(pending)}개의 대기 중인 동기화 항목 로드됨")
        except Exception as e:
            logger.error(f"대기 동기화 로드 실패: {str(e)}")
    
    # ============================================================================
    # 콜백 관리
    # ============================================================================
    
    def register_callback(self, event: str, callback: Callable):
        """콜백 함수 등록"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
        else:
            logger.warning(f"알 수 없는 이벤트: {event}")
    
    def unregister_callback(self, event: str, callback: Callable):
        """콜백 함수 제거"""
        if event in self.callbacks and callback in self.callbacks[event]:
            self.callbacks[event].remove(callback)
    
    def _trigger_callbacks(self, event: str, *args, **kwargs):
        """콜백 실행"""
        for callback in self.callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"콜백 오류 ({event}): {str(e)}")
    
    # ============================================================================
    # 수동 충돌 해결
    # ============================================================================
    
    def get_pending_conflicts(self) -> List[SyncConflict]:
        """대기 중인 충돌 목록"""
        conflicts = []
        
        # 큐에서 모든 충돌 가져오기 (비파괴적)
        temp_conflicts = []
        while not self.conflict_queue.empty():
            try:
                conflict = self.conflict_queue.get_nowait()
                conflicts.append(conflict)
                temp_conflicts.append(conflict)
            except queue.Empty:
                break
        
        # 다시 큐에 넣기
        for conflict in temp_conflicts:
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
    # 히스토리 관리
    # ============================================================================
    
    def _add_to_history(self):
        """동기화 히스토리에 추가"""
        history_item = {
            'timestamp': datetime.now(),
            'duration': self.current_progress.elapsed_time.total_seconds(),
            'total_items': self.current_progress.total_items,
            'completed': self.current_progress.completed_items,
            'failed': self.current_progress.failed_items,
            'conflicts': self.current_progress.conflicts
        }
        
        self.stats['sync_history'].append(history_item)
        
        # 최근 100개만 유지
        if len(self.stats['sync_history']) > 100:
            self.stats['sync_history'] = self.stats['sync_history'][-100:]
    
    def get_sync_history(self, limit: int = 10) -> List[Dict]:
        """동기화 히스토리 조회"""
        return self.stats['sync_history'][-limit:]
    
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
        logs = {
            'stats': self.stats,
            'queue_status': self.get_queue_status(),
            'last_sync': {k: v.isoformat() for k, v in self.last_sync.items()},
            'current_progress': asdict(self.current_progress) if self.current_progress else None
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2, default=str)
        
        logger.info(f"동기화 로그 내보내기 완료: {filepath}")
    
    def reset_sync_state(self):
        """동기화 상태 초기화"""
        logger.warning("동기화 상태 초기화 중...")
        
        # 큐 비우기
        self.clear_queue()
        
        # 통계 초기화
        self.stats = {
            'total_synced': 0,
            'failed_syncs': 0,
            'conflicts_resolved': 0,
            'last_successful_sync': None,
            'sync_history': []
        }
        
        # 진행 상황 초기화
        self.current_progress = SyncProgress()
        
        # 동기화 로그 초기화
        self.db_manager.clear_sync_log()
        
        logger.info("동기화 상태 초기화 완료")
    
    # ============================================================================
    # UI 헬퍼 메서드
    # ============================================================================
    
    def render_sync_status(self):
        """Streamlit UI에 동기화 상태 표시"""
        status = self.get_queue_status()
        
        # 연결 상태
        if self.is_online:
            st.success("🟢 온라인")
        else:
            st.warning("🔴 오프라인")
        
        # 동기화 진행 상황
        if self.sync_in_progress:
            progress = self.current_progress
            st.progress(progress.progress_percentage / 100)
            st.text(f"동기화 중... {progress.completed_items}/{progress.total_items}")
            
            if progress.current_table:
                st.text(f"현재: {progress.current_table} - {progress.current_action}")
        
        # 통계
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("동기화됨", status['stats']['total_synced'])
        with col2:
            st.metric("실패", status['stats']['failed_syncs'])
        with col3:
            st.metric("충돌", status['conflicts_pending'])
        
        # 큐 상태
        if status['size'] > 0:
            st.info(f"대기 중: {status['size']}개 항목")
    
    def render_conflict_resolver(self):
        """충돌 해결 UI"""
        conflicts = self.get_pending_conflicts()
        
        if not conflicts:
            st.info("해결할 충돌이 없습니다.")
            return
        
        for i, conflict in enumerate(conflicts):
            with st.expander(f"충돌 #{i+1}: {conflict.table_name} - ID {conflict.record_id}"):
                st.write("충돌 필드:", conflict.conflict_fields)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("로컬 데이터")
                    st.json(conflict.local_data)
                
                with col2:
                    st.subheader("원격 데이터")
                    st.json(conflict.remote_data)
                
                resolution_choice = st.radio(
                    "해결 방법 선택",
                    ["로컬 데이터 사용", "원격 데이터 사용", "수동 병합"],
                    key=f"conflict_{i}"
                )
                
                if st.button("해결", key=f"resolve_{i}"):
                    if resolution_choice == "로컬 데이터 사용":
                        self.resolve_conflict(conflict, conflict.local_data)
                    elif resolution_choice == "원격 데이터 사용":
                        self.resolve_conflict(conflict, conflict.remote_data)
                    else:
                        # 수동 병합 UI는 별도 구현 필요
                        st.warning("수동 병합은 아직 구현되지 않았습니다.")
                    
                    st.success("충돌이 해결되었습니다!")
                    st.experimental_rerun()


# ============================================================================
# 싱글톤 인스턴스
# ============================================================================

_sync_manager: Optional[SyncManager] = None


def get_sync_manager(db_manager=None, sheets_manager=None) -> SyncManager:
    """SyncManager 싱글톤 인스턴스 반환"""
    global _sync_manager
    
    if _sync_manager is None:
        if db_manager is None:
            raise ValueError("초기 생성 시 db_manager가 필요합니다")
        _sync_manager = SyncManager(db_manager, sheets_manager)
    
    # Sheets 매니저 업데이트 (나중에 설정 가능)
    if sheets_manager and _sync_manager.sheets_manager is None:
        _sync_manager.sheets_manager = sheets_manager
        logger.info("Sheets 매니저가 동기화 매니저에 연결되었습니다")
    
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
    
    # 연결 테스트
    manager = SyncManager(None)  # DB 매니저 없이 테스트
    print(f"인터넷 연결 상태: {'온라인' if manager.check_connection() else '오프라인'}")
