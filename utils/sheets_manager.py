"""
🌐 Google Sheets Manager - 선택적 클라우드 동기화
===========================================================================
데스크톱 애플리케이션을 위한 Google Sheets 연동 관리자
오프라인 우선 설계, OAuth 2.0/서비스 계정 지원, 실시간 양방향 동기화
===========================================================================
"""

import os
import sys
import json
import time
import logging
import threading
import hashlib
import zlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable, Set
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, PriorityQueue, Empty
from functools import lru_cache, wraps
from dataclasses import dataclass, field
from enum import Enum
import re

# 데이터 처리
import pandas as pd
import numpy as np

# Google API - 선택적 import
try:
    from google.oauth2 import service_account
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False
    logging.warning("Google API 라이브러리가 설치되지 않았습니다. Sheets 기능이 비활성화됩니다.")

# Streamlit
import streamlit as st

# 로컬 모듈
from config.local_config import LOCAL_CONFIG
from config.offline_config import OFFLINE_CONFIG
from config.app_config import GOOGLE_CONFIG

# ===========================================================================
# 🔧 설정 및 상수
# ===========================================================================

logger = logging.getLogger(__name__)

# Google Sheets API 설정
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
API_VERSION = 'v4'
RATE_LIMIT_PER_MINUTE = 60  # Google Sheets API 제한
BATCH_UPDATE_LIMIT = 1000   # 배치당 최대 업데이트 수
MAX_CELL_VALUE_LENGTH = 50000  # 셀당 최대 문자 수

# 시트 구조 정의
SHEET_STRUCTURE = {
    'Projects': {
        'columns': ['id', 'name', 'description', 'type', 'status', 'owner_id', 
                   'created_at', 'updated_at', 'settings', 'is_deleted'],
        'key_column': 'id',
        'json_columns': ['settings'],
        'timestamp_columns': ['created_at', 'updated_at']
    },
    'Experiments': {
        'columns': ['id', 'project_id', 'name', 'type', 'factors', 'responses', 
                   'design_matrix', 'status', 'created_at', 'updated_at'],
        'key_column': 'id',
        'json_columns': ['factors', 'responses', 'design_matrix'],
        'timestamp_columns': ['created_at', 'updated_at']
    },
    'Results': {
        'columns': ['id', 'experiment_id', 'run_number', 'conditions', 'measurements',
                   'notes', 'created_at', 'analyzed'],
        'key_column': 'id',
        'json_columns': ['conditions', 'measurements'],
        'timestamp_columns': ['created_at']
    },
    'Users': {
        'columns': ['id', 'email', 'name', 'role', 'organization', 'settings',
                   'created_at', 'last_login', 'is_active'],
        'key_column': 'id',
        'json_columns': ['settings'],
        'timestamp_columns': ['created_at', 'last_login']
    },
    'SyncLog': {
        'columns': ['id', 'table_name', 'record_id', 'action', 'timestamp',
                   'sync_status', 'error_message', 'checksum'],
        'key_column': 'id',
        'timestamp_columns': ['timestamp']
    }
}

# 캐시 설정
CACHE_TTL = {
    'users': 300,           # 5분
    'projects': 60,         # 1분
    'experiments': 30,      # 30초
    'results': 30,          # 30초
    'static_data': 3600,    # 1시간
}

# 동기화 설정
SYNC_INTERVAL = 30  # 초
CONFLICT_RESOLUTION_STRATEGY = 'last_write_wins'  # 또는 'manual', 'merge'

# ===========================================================================
# 🔧 데이터 클래스
# ===========================================================================

class SyncStatus(Enum):
    """동기화 상태"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CONFLICT = "conflict"

@dataclass
class SyncOperation:
    """동기화 작업"""
    sheet_name: str
    record_id: str
    action: str  # create, update, delete
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 1
    retry_count: int = 0
    
    def __lt__(self, other):
        return self.priority < other.priority

@dataclass
class SyncConflict:
    """동기화 충돌"""
    sheet_name: str
    record_id: str
    local_data: Dict[str, Any]
    remote_data: Dict[str, Any]
    local_checksum: str
    remote_checksum: str
    detected_at: datetime = field(default_factory=datetime.now)

# ===========================================================================
# 🔧 헬퍼 클래스
# ===========================================================================

class RateLimiter:
    """API 호출 속도 제한"""
    
    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.calls = deque()
        self._lock = threading.Lock()
    
    def wait_if_needed(self):
        """필요시 대기"""
        with self._lock:
            now = time.time()
            # 1분 이상 된 호출 제거
            while self.calls and self.calls[0] < now - 60:
                self.calls.popleft()
            
            # 제한 초과시 대기
            if len(self.calls) >= self.calls_per_minute:
                sleep_time = 60 - (now - self.calls[0]) + 0.1
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            self.calls.append(now)

class SheetCache:
    """시트 데이터 캐시"""
    
    def __init__(self):
        self._cache = {}
        self._timestamps = {}
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """캐시에서 가져오기"""
        with self._lock:
            if key in self._cache:
                # TTL 체크
                ttl = CACHE_TTL.get(key.split(':')[0], 60)
                if time.time() - self._timestamps[key] < ttl:
                    return self._cache[key]
                else:
                    del self._cache[key]
                    del self._timestamps[key]
            return None
    
    def set(self, key: str, value: Any):
        """캐시에 저장"""
        with self._lock:
            self._cache[key] = value
            self._timestamps[key] = time.time()
    
    def invalidate(self, pattern: str = None):
        """캐시 무효화"""
        with self._lock:
            if pattern:
                keys_to_delete = [k for k in self._cache if pattern in k]
                for key in keys_to_delete:
                    del self._cache[key]
                    del self._timestamps[key]
            else:
                self._cache.clear()
                self._timestamps.clear()

# ===========================================================================
# 🌐 Google Sheets Manager 클래스
# ===========================================================================

class GoogleSheetsManager:
    """Google Sheets 통합 관리자"""
    
    def __init__(self, spreadsheet_url: Optional[str] = None):
        """
        초기화
        
        Args:
            spreadsheet_url: Google Sheets URL (선택)
        """
        self.initialized = False
        self.service = None
        self.spreadsheet_id = None
        self.spreadsheet_url = spreadsheet_url
        self.auth_type = None  # 'oauth' or 'service_account'
        
        # 컴포넌트 초기화
        self.cache = SheetCache()
        self.rate_limiter = RateLimiter(RATE_LIMIT_PER_MINUTE)
        
        # 동기화 관련
        self.sync_queue = PriorityQueue()
        self.conflict_queue = Queue()
        self.offline_queue = deque()  # 오프라인 작업 큐
        self._sync_thread = None
        self._stop_sync = threading.Event()
        
        # 상태 추적
        self.is_online = False
        self.last_sync = None
        self.sync_errors = deque(maxlen=100)
        self.pending_changes = defaultdict(set)  # sheet_name -> set of record_ids
        
        # 초기화
        if GOOGLE_API_AVAILABLE:
            self._initialize()
    
    def _initialize(self):
        """서비스 초기화"""
        try:
            # 스프레드시트 URL 확인
            if not self.spreadsheet_url:
                self.spreadsheet_url = self._get_spreadsheet_url()
            
            if not self.spreadsheet_url:
                logger.info("Google Sheets URL이 설정되지 않았습니다. 오프라인 모드로 실행됩니다.")
                return
            
            # ID 추출
            self.spreadsheet_id = self._extract_spreadsheet_id(self.spreadsheet_url)
            if not self.spreadsheet_id:
                raise ValueError("유효하지 않은 스프레드시트 URL")
            
            # 인증
            if not self._authenticate():
                raise Exception("Google Sheets 인증 실패")
            
            # 연결 테스트
            if not self._test_connection():
                raise Exception("Google Sheets 연결 실패")
            
            # 시트 구조 확인/생성
            self._ensure_sheet_structure()
            
            # 동기화 스레드 시작
            self._start_sync_thread()
            
            self.initialized = True
            self.is_online = True
            logger.info(f"Google Sheets 연결 성공: {self.spreadsheet_id}")
            
        except Exception as e:
            logger.warning(f"Google Sheets 초기화 실패: {str(e)}")
            self.initialized = False
            self.is_online = False
    
    def _get_spreadsheet_url(self) -> Optional[str]:
        """설정에서 스프레드시트 URL 가져오기"""
        # 1. 세션 상태
        if 'google_sheets_url' in st.session_state:
            return st.session_state.google_sheets_url
        
        # 2. 환경 변수
        url = os.environ.get('GOOGLE_SHEETS_URL')
        if url:
            return url
        
        # 3. Streamlit secrets
        try:
            if 'google_sheets_url' in st.secrets:
                return st.secrets['google_sheets_url']
        except:
            pass
        
        # 4. 설정 파일
        if hasattr(GOOGLE_CONFIG, 'sheets_url'):
            return GOOGLE_CONFIG.sheets_url
        
        return None
    
    def _extract_spreadsheet_id(self, url: str) -> Optional[str]:
        """URL에서 스프레드시트 ID 추출"""
        # 다양한 URL 형식 지원
        patterns = [
            r'/spreadsheets/d/([a-zA-Z0-9-_]+)',
            r'id=([a-zA-Z0-9-_]+)',
            r'^([a-zA-Z0-9-_]+)$'  # ID만 있는 경우
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    def _authenticate(self) -> bool:
        """Google Sheets API 인증"""
        try:
            creds = None
            
            # 1. 서비스 계정 시도
            service_account_file = self._get_service_account_file()
            if service_account_file and os.path.exists(service_account_file):
                creds = service_account.Credentials.from_service_account_file(
                    service_account_file, scopes=SCOPES
                )
                self.auth_type = 'service_account'
                logger.info("서비스 계정으로 인증됨")
            
            # 2. OAuth 2.0 시도
            else:
                token_file = LOCAL_CONFIG['app_data_dir'] / 'auth' / 'google_token.json'
                
                if token_file.exists():
                    creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)
                
                # 토큰 갱신 필요시
                if not creds or not creds.valid:
                    if creds and creds.expired and creds.refresh_token:
                        creds.refresh(Request())
                    else:
                        # OAuth 플로우 실행
                        creds = self._run_oauth_flow()
                        if not creds:
                            return False
                    
                    # 토큰 저장
                    token_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(token_file, 'w') as f:
                        f.write(creds.to_json())
                
                self.auth_type = 'oauth'
                logger.info("OAuth로 인증됨")
            
            # 서비스 빌드
            self.service = build('sheets', API_VERSION, credentials=creds)
            return True
            
        except Exception as e:
            logger.error(f"인증 실패: {str(e)}")
            return False
    
    def _get_service_account_file(self) -> Optional[str]:
        """서비스 계정 파일 경로 가져오기"""
        # 1. 환경 변수
        path = os.environ.get('GOOGLE_SERVICE_ACCOUNT_FILE')
        if path:
            return path
        
        # 2. 기본 경로
        default_path = LOCAL_CONFIG['app_data_dir'] / 'auth' / 'service_account.json'
        if default_path.exists():
            return str(default_path)
        
        return None
    
    def _run_oauth_flow(self) -> Optional[Credentials]:
        """OAuth 2.0 플로우 실행"""
        try:
            # 클라이언트 시크릿 파일 확인
            client_secrets_file = LOCAL_CONFIG['app_data_dir'] / 'auth' / 'client_secrets.json'
            
            if not client_secrets_file.exists():
                # Streamlit에서 OAuth 설정 안내
                st.error("""
                Google Sheets 연동을 위한 OAuth 설정이 필요합니다.
                
                1. Google Cloud Console에서 OAuth 2.0 클라이언트 ID를 생성하세요
                2. 다운로드한 JSON 파일을 다음 경로에 저장하세요:
                   `{}`
                """.format(client_secrets_file))
                return None
            
            flow = InstalledAppFlow.from_client_secrets_file(
                str(client_secrets_file), SCOPES
            )
            
            # 로컬 서버로 인증 (포트 자동 선택)
            creds = flow.run_local_server(port=0)
            
            return creds
            
        except Exception as e:
            logger.error(f"OAuth 플로우 실패: {str(e)}")
            return None
    
    def _test_connection(self) -> bool:
        """연결 테스트"""
        try:
            # 스프레드시트 메타데이터 가져오기
            self.service.spreadsheets().get(
                spreadsheetId=self.spreadsheet_id
            ).execute()
            
            return True
            
        except HttpError as e:
            if e.resp.status == 404:
                logger.error("스프레드시트를 찾을 수 없습니다")
            elif e.resp.status == 403:
                logger.error("스프레드시트에 대한 접근 권한이 없습니다")
            else:
                logger.error(f"연결 테스트 실패: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"연결 테스트 실패: {str(e)}")
            return False
    
    def _ensure_sheet_structure(self):
        """시트 구조 확인 및 생성"""
        try:
            # 현재 시트 목록 가져오기
            spreadsheet = self.service.spreadsheets().get(
                spreadsheetId=self.spreadsheet_id
            ).execute()
            
            existing_sheets = {sheet['properties']['title'] 
                             for sheet in spreadsheet.get('sheets', [])}
            
            # 필요한 시트 생성
            requests = []
            for sheet_name, config in SHEET_STRUCTURE.items():
                if sheet_name not in existing_sheets:
                    requests.append({
                        'addSheet': {
                            'properties': {
                                'title': sheet_name,
                                'gridProperties': {
                                    'rowCount': 1000,
                                    'columnCount': len(config['columns'])
                                }
                            }
                        }
                    })
            
            if requests:
                self.service.spreadsheets().batchUpdate(
                    spreadsheetId=self.spreadsheet_id,
                    body={'requests': requests}
                ).execute()
                logger.info(f"시트 생성됨: {len(requests)}개")
            
            # 헤더 설정
            for sheet_name, config in SHEET_STRUCTURE.items():
                if sheet_name not in existing_sheets or sheet_name in [r['addSheet']['properties']['title'] for r in requests]:
                    self._set_headers(sheet_name, config['columns'])
            
        except Exception as e:
            logger.error(f"시트 구조 생성 실패: {str(e)}")
    
    def _set_headers(self, sheet_name: str, columns: List[str]):
        """시트 헤더 설정"""
        try:
            values = [columns]
            
            self.service.spreadsheets().values().update(
                spreadsheetId=self.spreadsheet_id,
                range=f"{sheet_name}!A1:{self._get_column_letter(len(columns))}1",
                valueInputOption="RAW",
                body={'values': values}
            ).execute()
            
            # 헤더 행 서식 설정 (굵게, 배경색)
            requests = [{
                'repeatCell': {
                    'range': {
                        'sheetId': self._get_sheet_id(sheet_name),
                        'startRowIndex': 0,
                        'endRowIndex': 1
                    },
                    'cell': {
                        'userEnteredFormat': {
                            'textFormat': {'bold': True},
                            'backgroundColor': {'red': 0.9, 'green': 0.9, 'blue': 0.9}
                        }
                    },
                    'fields': 'userEnteredFormat(textFormat,backgroundColor)'
                }
            }]
            
            self.service.spreadsheets().batchUpdate(
                spreadsheetId=self.spreadsheet_id,
                body={'requests': requests}
            ).execute()
            
        except Exception as e:
            logger.error(f"헤더 설정 실패: {str(e)}")
    
    # ===========================================================================
    # 🔍 읽기 작업
    # ===========================================================================
    
    def read(self, sheet_name: str, filters: Optional[Dict] = None,
             columns: Optional[List[str]] = None, 
             use_cache: bool = True) -> pd.DataFrame:
        """
        시트 데이터 읽기
        
        Args:
            sheet_name: 시트 이름
            filters: 필터 조건 {'column': value}
            columns: 읽을 컬럼 목록
            use_cache: 캐시 사용 여부
            
        Returns:
            pandas DataFrame
        """
        if not self.is_online:
            return pd.DataFrame()  # 오프라인시 빈 DataFrame
        
        try:
            # 캐시 확인
            cache_key = f"{sheet_name}:{json.dumps(filters or {})}:{json.dumps(columns or [])}"
            if use_cache:
                cached = self.cache.get(cache_key)
                if cached is not None:
                    return cached
            
            # API 호출
            self.rate_limiter.wait_if_needed()
            
            # 범위 계산
            if columns:
                # 특정 컬럼만
                column_indices = self._get_column_indices(sheet_name, columns)
                ranges = [f"{sheet_name}!{self._get_column_letter(idx)}:{self._get_column_letter(idx)}" 
                         for idx in column_indices]
                
                response = self.service.spreadsheets().values().batchGet(
                    spreadsheetId=self.spreadsheet_id,
                    ranges=ranges
                ).execute()
                
                # 데이터 조합
                all_values = []
                for i, range_data in enumerate(response.get('valueRanges', [])):
                    values = range_data.get('values', [])
                    if i == 0:
                        all_values = values
                    else:
                        # 컬럼 추가
                        for j, row in enumerate(values):
                            if j < len(all_values):
                                all_values[j].extend(row)
            else:
                # 전체 데이터
                response = self.service.spreadsheets().values().get(
                    spreadsheetId=self.spreadsheet_id,
                    range=sheet_name
                ).execute()
                
                all_values = response.get('values', [])
            
            if not all_values:
                return pd.DataFrame()
            
            # DataFrame 생성
            headers = all_values[0]
            data = all_values[1:]
            
            # 빈 행 제거
            data = [row for row in data if any(cell for cell in row)]
            
            df = pd.DataFrame(data, columns=headers)
            
            # 데이터 타입 변환
            df = self._convert_types(df, sheet_name)
            
            # 필터 적용
            if filters:
                for col, value in filters.items():
                    if col in df.columns:
                        df = df[df[col] == value]
            
            # 캐시 저장
            if use_cache:
                self.cache.set(cache_key, df)
            
            return df
            
        except Exception as e:
            logger.error(f"시트 읽기 실패: {str(e)}")
            return pd.DataFrame()
    
    def read_cell(self, sheet_name: str, row: int, column: str) -> Any:
        """단일 셀 읽기"""
        if not self.is_online:
            return None
        
        try:
            self.rate_limiter.wait_if_needed()
            
            response = self.service.spreadsheets().values().get(
                spreadsheetId=self.spreadsheet_id,
                range=f"{sheet_name}!{column}{row}"
            ).execute()
            
            values = response.get('values', [[]])
            return values[0][0] if values and values[0] else None
            
        except Exception as e:
            logger.error(f"셀 읽기 실패: {str(e)}")
            return None
    
    # ===========================================================================
    # ✏️ 쓰기 작업
    # ===========================================================================
    
    def create(self, sheet_name: str, data: Dict[str, Any]) -> Optional[str]:
        """
        새 레코드 생성
        
        Args:
            sheet_name: 시트 이름
            data: 레코드 데이터
            
        Returns:
            생성된 레코드 ID
        """
        try:
            # ID 생성
            if 'id' not in data:
                data['id'] = self._generate_id()
            
            # 타임스탬프 추가
            data['created_at'] = datetime.now().isoformat()
            data['updated_at'] = data['created_at']
            
            # 동기화 큐에 추가
            operation = SyncOperation(
                sheet_name=sheet_name,
                record_id=data['id'],
                action='create',
                data=data,
                priority=2
            )
            
            self.sync_queue.put(operation)
            self.pending_changes[sheet_name].add(data['id'])
            
            # 캐시 무효화
            self.cache.invalidate(sheet_name)
            
            return data['id']
            
        except Exception as e:
            logger.error(f"레코드 생성 실패: {str(e)}")
            # 오프라인 큐에 추가
            self.offline_queue.append(operation)
            return data['id']  # 낙관적 응답
    
    def update(self, sheet_name: str, record_id: str, 
               updates: Dict[str, Any]) -> bool:
        """레코드 업데이트"""
        try:
            # 타임스탬프 업데이트
            updates['updated_at'] = datetime.now().isoformat()
            
            # 동기화 큐에 추가
            operation = SyncOperation(
                sheet_name=sheet_name,
                record_id=record_id,
                action='update',
                data=updates,
                priority=3
            )
            
            self.sync_queue.put(operation)
            self.pending_changes[sheet_name].add(record_id)
            
            # 캐시 무효화
            self.cache.invalidate(f"{sheet_name}:")
            
            return True
            
        except Exception as e:
            logger.error(f"레코드 업데이트 실패: {str(e)}")
            self.offline_queue.append(operation)
            return True  # 낙관적 응답
    
    def delete(self, sheet_name: str, record_id: str, soft: bool = True) -> bool:
        """레코드 삭제"""
        try:
            if soft:
                # 소프트 삭제
                return self.update(sheet_name, record_id, {'is_deleted': True})
            else:
                # 하드 삭제
                operation = SyncOperation(
                    sheet_name=sheet_name,
                    record_id=record_id,
                    action='delete',
                    data={},
                    priority=4
                )
                
                self.sync_queue.put(operation)
                self.pending_changes[sheet_name].add(record_id)
                
                # 캐시 무효화
                self.cache.invalidate(f"{sheet_name}:")
                
                return True
                
        except Exception as e:
            logger.error(f"레코드 삭제 실패: {str(e)}")
            return False
    
    def batch_create(self, sheet_name: str, records: List[Dict[str, Any]]) -> List[str]:
        """배치 생성"""
        ids = []
        for record in records:
            record_id = self.create(sheet_name, record)
            if record_id:
                ids.append(record_id)
        return ids
    
    # ===========================================================================
    # 🔄 동기화 작업
    # ===========================================================================
    
    def _start_sync_thread(self):
        """동기화 스레드 시작"""
        if self._sync_thread and self._sync_thread.is_alive():
            return
        
        self._stop_sync.clear()
        self._sync_thread = threading.Thread(target=self._sync_worker, daemon=True)
        self._sync_thread.start()
        logger.info("동기화 스레드 시작됨")
    
    def _sync_worker(self):
        """동기화 워커"""
        while not self._stop_sync.is_set():
            try:
                # 온라인 체크
                if not self._check_online_status():
                    time.sleep(SYNC_INTERVAL)
                    continue
                
                # 오프라인 큐 처리
                self._process_offline_queue()
                
                # 동기화 큐 처리
                batch = []
                deadline = time.time() + 5  # 5초 동안 배치 수집
                
                while time.time() < deadline and len(batch) < BATCH_UPDATE_LIMIT:
                    try:
                        operation = self.sync_queue.get(timeout=0.1)
                        batch.append(operation)
                    except Empty:
                        break
                
                if batch:
                    self._process_sync_batch(batch)
                
                # 충돌 처리
                self._process_conflicts()
                
                # 대기
                time.sleep(SYNC_INTERVAL)
                
            except Exception as e:
                logger.error(f"동기화 워커 오류: {str(e)}")
                time.sleep(SYNC_INTERVAL)
    
    def _check_online_status(self) -> bool:
        """온라인 상태 확인"""
        try:
            # 간단한 API 호출로 연결 확인
            self.service.spreadsheets().get(
                spreadsheetId=self.spreadsheet_id,
                fields="properties.title"
            ).execute()
            
            self.is_online = True
            return True
            
        except Exception:
            self.is_online = False
            return False
    
    def _process_offline_queue(self):
        """오프라인 큐 처리"""
        if not self.offline_queue:
            return
        
        processed = []
        for operation in list(self.offline_queue):
            try:
                self.sync_queue.put(operation)
                processed.append(operation)
            except Exception as e:
                logger.error(f"오프라인 큐 처리 실패: {str(e)}")
        
        # 처리된 항목 제거
        for op in processed:
            self.offline_queue.remove(op)
        
        if processed:
            logger.info(f"오프라인 큐에서 {len(processed)}개 작업 복원")
    
    def _process_sync_batch(self, batch: List[SyncOperation]):
        """동기화 배치 처리"""
        try:
            # 시트별로 그룹화
            sheet_groups = defaultdict(list)
            for op in batch:
                sheet_groups[op.sheet_name].append(op)
            
            # 시트별 처리
            for sheet_name, operations in sheet_groups.items():
                self._sync_sheet_operations(sheet_name, operations)
            
            # 성공한 작업 제거
            for op in batch:
                self.pending_changes[op.sheet_name].discard(op.record_id)
            
            self.last_sync = datetime.now()
            
        except Exception as e:
            logger.error(f"배치 동기화 실패: {str(e)}")
            # 실패한 작업 재시도 큐에 추가
            for op in batch:
                op.retry_count += 1
                if op.retry_count < 3:
                    self.sync_queue.put(op)
                else:
                    self.sync_errors.append({
                        'operation': op,
                        'error': str(e),
                        'timestamp': datetime.now()
                    })
    
    def _sync_sheet_operations(self, sheet_name: str, operations: List[SyncOperation]):
        """시트별 작업 동기화"""
        # 현재 데이터 읽기
        current_df = self.read(sheet_name, use_cache=False)
        
        # 작업별 처리
        create_rows = []
        update_requests = []
        delete_rows = []
        
        for op in operations:
            if op.action == 'create':
                # 새 행 추가
                row_data = self._prepare_row_data(sheet_name, op.data)
                create_rows.append(row_data)
                
            elif op.action == 'update':
                # 행 찾기
                row_idx = self._find_row_index(current_df, sheet_name, op.record_id)
                if row_idx is not None:
                    # 업데이트 요청 생성
                    for col, value in op.data.items():
                        col_idx = self._get_column_index(sheet_name, col)
                        if col_idx is not None:
                            update_requests.append({
                                'updateCells': {
                                    'range': {
                                        'sheetId': self._get_sheet_id(sheet_name),
                                        'startRowIndex': row_idx + 1,  # 헤더 제외
                                        'endRowIndex': row_idx + 2,
                                        'startColumnIndex': col_idx,
                                        'endColumnIndex': col_idx + 1
                                    },
                                    'rows': [{
                                        'values': [{
                                            'userEnteredValue': self._convert_value(value)
                                        }]
                                    }],
                                    'fields': 'userEnteredValue'
                                }
                            })
            
            elif op.action == 'delete':
                # 삭제할 행 찾기
                row_idx = self._find_row_index(current_df, sheet_name, op.record_id)
                if row_idx is not None:
                    delete_rows.append(row_idx + 2)  # 1-based, 헤더 제외
        
        # 배치 실행
        requests = []
        
        # 생성 작업
        if create_rows:
            self._append_rows(sheet_name, create_rows)
        
        # 업데이트 작업
        if update_requests:
            requests.extend(update_requests)
        
        # 삭제 작업 (역순으로 정렬하여 인덱스 변경 방지)
        if delete_rows:
            for row_idx in sorted(delete_rows, reverse=True):
                requests.append({
                    'deleteDimension': {
                        'range': {
                            'sheetId': self._get_sheet_id(sheet_name),
                            'dimension': 'ROWS',
                            'startIndex': row_idx - 1,
                            'endIndex': row_idx
                        }
                    }
                })
        
        # 배치 업데이트 실행
        if requests:
            self.rate_limiter.wait_if_needed()
            self.service.spreadsheets().batchUpdate(
                spreadsheetId=self.spreadsheet_id,
                body={'requests': requests}
            ).execute()
    
    def _process_conflicts(self):
        """충돌 처리"""
        conflicts = []
        while not self.conflict_queue.empty():
            try:
                conflict = self.conflict_queue.get_nowait()
                conflicts.append(conflict)
            except Empty:
                break
        
        if not conflicts:
            return
        
        # 충돌 해결 전략에 따라 처리
        if CONFLICT_RESOLUTION_STRATEGY == 'last_write_wins':
            # 로컬 데이터로 덮어쓰기
            for conflict in conflicts:
                self.update(conflict.sheet_name, conflict.record_id, 
                           conflict.local_data)
        
        elif CONFLICT_RESOLUTION_STRATEGY == 'manual':
            # 사용자에게 알림
            if 'conflict_handler' in st.session_state:
                st.session_state.conflict_handler(conflicts)
        
        elif CONFLICT_RESOLUTION_STRATEGY == 'merge':
            # 자동 병합 시도
            for conflict in conflicts:
                merged = self._merge_conflicts(conflict)
                if merged:
                    self.update(conflict.sheet_name, conflict.record_id, merged)
    
    # ===========================================================================
    # 🔧 유틸리티 메서드
    # ===========================================================================
    
    def _generate_id(self) -> str:
        """고유 ID 생성"""
        return f"{int(time.time() * 1000)}_{secrets.token_urlsafe(8)}"
    
    def _get_sheet_id(self, sheet_name: str) -> int:
        """시트 ID 가져오기"""
        try:
            spreadsheet = self.service.spreadsheets().get(
                spreadsheetId=self.spreadsheet_id
            ).execute()
            
            for sheet in spreadsheet.get('sheets', []):
                if sheet['properties']['title'] == sheet_name:
                    return sheet['properties']['sheetId']
            
            return 0
        except:
            return 0
    
    def _get_column_letter(self, col_idx: int) -> str:
        """컬럼 인덱스를 문자로 변환 (0-based)"""
        result = ""
        while col_idx >= 0:
            result = chr(col_idx % 26 + ord('A')) + result
            col_idx = col_idx // 26 - 1
        return result
    
    def _get_column_index(self, sheet_name: str, column_name: str) -> Optional[int]:
        """컬럼 이름으로 인덱스 찾기"""
        config = SHEET_STRUCTURE.get(sheet_name, {})
        columns = config.get('columns', [])
        
        try:
            return columns.index(column_name)
        except ValueError:
            return None
    
    def _get_column_indices(self, sheet_name: str, column_names: List[str]) -> List[int]:
        """여러 컬럼의 인덱스 찾기"""
        indices = []
        for name in column_names:
            idx = self._get_column_index(sheet_name, name)
            if idx is not None:
                indices.append(idx)
        return indices
    
    def _find_row_index(self, df: pd.DataFrame, sheet_name: str, record_id: str) -> Optional[int]:
        """레코드 ID로 행 인덱스 찾기"""
        config = SHEET_STRUCTURE.get(sheet_name, {})
        key_column = config.get('key_column', 'id')
        
        if key_column in df.columns:
            matches = df[df[key_column] == record_id].index
            if len(matches) > 0:
                return matches[0]
        
        return None
    
    def _prepare_row_data(self, sheet_name: str, data: Dict[str, Any]) -> List[Any]:
        """행 데이터 준비"""
        config = SHEET_STRUCTURE.get(sheet_name, {})
        columns = config.get('columns', [])
        
        row = []
        for col in columns:
            value = data.get(col, '')
            
            # JSON 컬럼 처리
            if col in config.get('json_columns', []):
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
            
            # 타임스탬프 처리
            elif col in config.get('timestamp_columns', []):
                if isinstance(value, datetime):
                    value = value.isoformat()
            
            row.append(str(value))
        
        return row
    
    def _convert_value(self, value: Any) -> Dict[str, Any]:
        """값을 Google Sheets API 형식으로 변환"""
        if value is None or value == '':
            return {'stringValue': ''}
        elif isinstance(value, bool):
            return {'boolValue': value}
        elif isinstance(value, (int, float)):
            return {'numberValue': value}
        elif isinstance(value, datetime):
            return {'stringValue': value.isoformat()}
        elif isinstance(value, (dict, list)):
            return {'stringValue': json.dumps(value)}
        else:
            return {'stringValue': str(value)}
    
    def _convert_types(self, df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
        """DataFrame 타입 변환"""
        config = SHEET_STRUCTURE.get(sheet_name, {})
        
        # JSON 컬럼
        for col in config.get('json_columns', []):
            if col in df.columns:
                df[col] = df[col].apply(lambda x: json.loads(x) if x and x != '' else {})
        
        # 타임스탬프 컬럼
        for col in config.get('timestamp_columns', []):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # 불린 컬럼
        bool_columns = ['is_active', 'is_deleted', 'analyzed']
        for col in bool_columns:
            if col in df.columns:
                df[col] = df[col].map({'TRUE': True, 'FALSE': False, 'true': True, 'false': False})
        
        return df
    
    def _append_rows(self, sheet_name: str, rows: List[List[Any]]):
        """행 추가"""
        if not rows:
            return
        
        self.rate_limiter.wait_if_needed()
        
        self.service.spreadsheets().values().append(
            spreadsheetId=self.spreadsheet_id,
            range=sheet_name,
            valueInputOption="USER_ENTERED",
            insertDataOption="INSERT_ROWS",
            body={'values': rows}
        ).execute()
    
    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """데이터 체크섬 계산"""
        # 정렬된 JSON 문자열로 변환
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(json_str.encode()).hexdigest()
    
    def _merge_conflicts(self, conflict: SyncConflict) -> Optional[Dict[str, Any]]:
        """충돌 자동 병합"""
        # 간단한 병합 전략: 필드별로 최신 값 선택
        merged = {}
        
        for key in set(conflict.local_data.keys()) | set(conflict.remote_data.keys()):
            local_val = conflict.local_data.get(key)
            remote_val = conflict.remote_data.get(key)
            
            # 타임스탬프가 있는 경우 최신 선택
            if key == 'updated_at':
                if local_val and remote_val:
                    merged[key] = max(local_val, remote_val)
                else:
                    merged[key] = local_val or remote_val
            else:
                # 로컬 우선
                merged[key] = local_val if local_val is not None else remote_val
        
        return merged
    
    # ===========================================================================
    # 📊 상태 및 통계
    # ===========================================================================
    
    def get_sync_status(self) -> Dict[str, Any]:
        """동기화 상태 조회"""
        return {
            'is_online': self.is_online,
            'last_sync': self.last_sync,
            'pending_changes': sum(len(changes) for changes in self.pending_changes.values()),
            'offline_queue_size': len(self.offline_queue),
            'sync_queue_size': self.sync_queue.qsize(),
            'recent_errors': list(self.sync_errors)[-10:],  # 최근 10개
            'auth_type': self.auth_type
        }
    
    def force_sync(self):
        """강제 동기화"""
        if not self.is_online:
            logger.warning("오프라인 상태에서는 동기화할 수 없습니다")
            return False
        
        try:
            # 모든 큐 처리
            self._process_offline_queue()
            
            # 동기화 큐 비우기
            batch = []
            while not self.sync_queue.empty() and len(batch) < BATCH_UPDATE_LIMIT:
                try:
                    batch.append(self.sync_queue.get_nowait())
                except Empty:
                    break
            
            if batch:
                self._process_sync_batch(batch)
            
            return True
            
        except Exception as e:
            logger.error(f"강제 동기화 실패: {str(e)}")
            return False
    
    def export_to_csv(self, sheet_name: str, file_path: str) -> bool:
        """시트를 CSV로 내보내기"""
        try:
            df = self.read(sheet_name, use_cache=False)
            df.to_csv(file_path, index=False)
            return True
        except Exception as e:
            logger.error(f"CSV 내보내기 실패: {str(e)}")
            return False
    
    def import_from_csv(self, sheet_name: str, file_path: str, 
                       replace: bool = False) -> bool:
        """CSV에서 가져오기"""
        try:
            df = pd.read_csv(file_path)
            
            if replace:
                # 기존 데이터 삭제
                self._clear_sheet(sheet_name)
            
            # 데이터 추가
            records = df.to_dict('records')
            self.batch_create(sheet_name, records)
            
            return True
            
        except Exception as e:
            logger.error(f"CSV 가져오기 실패: {str(e)}")
            return False
    
    def _clear_sheet(self, sheet_name: str):
        """시트 데이터 지우기 (헤더 제외)"""
        try:
            # 시트 크기 가져오기
            sheet_metadata = self.service.spreadsheets().get(
                spreadsheetId=self.spreadsheet_id,
                ranges=[sheet_name]
            ).execute()
            
            sheet_props = None
            for sheet in sheet_metadata.get('sheets', []):
                if sheet['properties']['title'] == sheet_name:
                    sheet_props = sheet['properties']
                    break
            
            if not sheet_props:
                return
            
            row_count = sheet_props.get('gridProperties', {}).get('rowCount', 0)
            
            if row_count > 1:
                # 헤더를 제외한 모든 행 삭제
                request = {
                    'deleteDimension': {
                        'range': {
                            'sheetId': sheet_props['sheetId'],
                            'dimension': 'ROWS',
                            'startIndex': 1,
                            'endIndex': row_count
                        }
                    }
                }
                
                self.service.spreadsheets().batchUpdate(
                    spreadsheetId=self.spreadsheet_id,
                    body={'requests': [request]}
                ).execute()
                
        except Exception as e:
            logger.error(f"시트 지우기 실패: {str(e)}")
    
    def close(self):
        """연결 종료 및 정리"""
        # 동기화 스레드 중지
        self._stop_sync.set()
        if self._sync_thread:
            self._sync_thread.join(timeout=5)
        
        # 남은 작업 처리
        if self.is_online:
            self.force_sync()
        
        logger.info("Google Sheets Manager 종료")


# ===========================================================================
# 🔧 싱글톤 인스턴스
# ===========================================================================

_sheets_manager: Optional[GoogleSheetsManager] = None


def get_sheets_manager(spreadsheet_url: Optional[str] = None) -> GoogleSheetsManager:
    """GoogleSheetsManager 싱글톤 인스턴스 반환"""
    global _sheets_manager
    
    if _sheets_manager is None:
        _sheets_manager = GoogleSheetsManager(spreadsheet_url)
    
    return _sheets_manager


# ===========================================================================
# 🔧 간편 함수
# ===========================================================================

def is_sheets_available() -> bool:
    """Google Sheets 사용 가능 여부"""
    return GOOGLE_API_AVAILABLE and get_sheets_manager().initialized


def read_sheet(sheet_name: str, **kwargs) -> pd.DataFrame:
    """시트 읽기 간편 함수"""
    return get_sheets_manager().read(sheet_name, **kwargs)


def create_record(sheet_name: str, data: Dict[str, Any]) -> Optional[str]:
    """레코드 생성 간편 함수"""
    return get_sheets_manager().create(sheet_name, data)


def update_record(sheet_name: str, record_id: str, updates: Dict[str, Any]) -> bool:
    """레코드 업데이트 간편 함수"""
    return get_sheets_manager().update(sheet_name, record_id, updates)


def delete_record(sheet_name: str, record_id: str, soft: bool = True) -> bool:
    """레코드 삭제 간편 함수"""
    return get_sheets_manager().delete(sheet_name, record_id, soft)


def get_sync_status() -> Dict[str, Any]:
    """동기화 상태 조회"""
    return get_sheets_manager().get_sync_status()


# ===========================================================================
# 🧪 테스트 코드
# ===========================================================================

if __name__ == "__main__":
    # 테스트
    print("Google Sheets Manager 테스트")
    
    # 초기화
    manager = GoogleSheetsManager()
    
    if manager.initialized:
        print("✅ Google Sheets 연결 성공")
        
        # 테스트 데이터 생성
        test_data = {
            'name': '테스트 프로젝트',
            'description': '테스트 설명',
            'type': 'polymer',
            'status': 'active'
        }
        
        # 생성 테스트
        project_id = manager.create('Projects', test_data)
        print(f"✅ 프로젝트 생성: {project_id}")
        
        # 읽기 테스트
        df = manager.read('Projects', filters={'id': project_id})
        print(f"✅ 프로젝트 조회: {len(df)}개")
        
        # 업데이트 테스트
        success = manager.update('Projects', project_id, {'status': 'completed'})
        print(f"✅ 프로젝트 업데이트: {success}")
        
        # 동기화 상태
        status = manager.get_sync_status()
        print(f"✅ 동기화 상태: {status}")
        
    else:
        print("❌ Google Sheets 연결 실패 - 오프라인 모드")
