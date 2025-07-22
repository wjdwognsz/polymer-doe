"""
Google Sheets를 선택적 클라우드 저장소로 사용하는 관리자
데스크톱 앱의 SQLite 데이터를 온라인 시 동기화합니다.
"""
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue, Empty, PriorityQueue
import hashlib
import pandas as pd
import numpy as np
from functools import lru_cache, wraps
from pathlib import Path
import os

# Google API - 선택적 import
try:
    from google.oauth2 import service_account, credentials as oauth2_credentials
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False
    logging.warning("Google API 라이브러리가 설치되지 않았습니다. Sheets 기능이 비활성화됩니다.")

from config.app_config import GOOGLE_SHEETS_CONFIG
from config.local_config import LOCAL_CONFIG
from config.secrets_config import GOOGLE_CONFIG

logger = logging.getLogger(__name__)


# ============================================================================
# 상수 정의
# ============================================================================

# Google Sheets API 설정
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
API_VERSION = 'v4'
RATE_LIMIT_PER_MINUTE = 60
BATCH_UPDATE_LIMIT = 1000

# 시트 구조 정의
SHEET_SCHEMAS = {
    'users': {
        'columns': ['id', 'email', 'name', 'password_hash', 'role', 'created_at', 
                   'updated_at', 'last_login', 'settings', 'is_active'],
        'key': 'id',
        'indexes': ['email']
    },
    'projects': {
        'columns': ['id', 'user_id', 'name', 'description', 'field', 'module_id', 
                   'status', 'data', 'created_at', 'updated_at'],
        'key': 'id',
        'indexes': ['user_id', 'status']
    },
    'experiments': {
        'columns': ['id', 'project_id', 'name', 'design_type', 'factors', 
                   'responses', 'design_matrix', 'results', 'status', 
                   'created_at', 'updated_at'],
        'key': 'id',
        'indexes': ['project_id', 'status']
    },
    'results': {
        'columns': ['id', 'experiment_id', 'run_number', 'conditions', 
                   'measurements', 'notes', 'created_at', 'updated_at'],
        'key': 'id',
        'indexes': ['experiment_id']
    },
    'sync_metadata': {
        'columns': ['table_name', 'last_sync', 'sync_version', 'checksum'],
        'key': 'table_name',
        'indexes': []
    }
}

# 캐시 TTL 설정
CACHE_TTL = {
    'users': 300,           # 5분
    'projects': 60,         # 1분  
    'experiments': 30,      # 30초
    'results': 30,          # 30초
    'static_data': 3600,    # 1시간
    'sync_metadata': 10     # 10초
}

# 데이터 타입 변환 맵
TYPE_CONVERTERS = {
    'datetime': lambda x: pd.to_datetime(x).isoformat() if pd.notna(x) else None,
    'json': lambda x: json.dumps(x) if x else '{}',
    'bool': lambda x: str(x).lower() == 'true' if isinstance(x, str) else bool(x),
    'int': lambda x: int(float(x)) if x and str(x).replace('.', '').isdigit() else None,
    'float': lambda x: float(x) if x and str(x).replace('.', '').replace('-', '').isdigit() else None,
    'str': lambda x: str(x) if x is not None else ''
}


# ============================================================================
# 헬퍼 클래스
# ============================================================================

class RateLimiter:
    """API 호출 속도 제한기"""
    
    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.calls = deque()
        self.lock = threading.Lock()
    
    def acquire(self):
        """호출 권한 획득 (필요시 대기)"""
        with self.lock:
            now = time.time()
            # 1분 이상 지난 호출 제거
            while self.calls and self.calls[0] < now - 60:
                self.calls.popleft()
            
            # 제한 확인
            if len(self.calls) >= self.calls_per_minute:
                # 대기 시간 계산
                sleep_time = 60 - (now - self.calls[0]) + 0.1
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            self.calls.append(time.time())


class SheetCache:
    """시트 데이터 캐시"""
    
    def __init__(self):
        self.cache = {}
        self.timestamps = {}
        self.locks = defaultdict(threading.Lock)
    
    def get(self, key: str) -> Optional[Any]:
        """캐시에서 데이터 가져오기"""
        with self.locks[key]:
            if key in self.cache:
                # TTL 확인
                ttl = CACHE_TTL.get(key.split(':')[0], 60)
                if time.time() - self.timestamps[key] < ttl:
                    return self.cache[key]
                else:
                    # 만료된 캐시 제거
                    del self.cache[key]
                    del self.timestamps[key]
        return None
    
    def set(self, key: str, value: Any):
        """캐시에 데이터 저장"""
        with self.locks[key]:
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def invalidate(self, pattern: str = None):
        """캐시 무효화"""
        if pattern:
            # 패턴과 일치하는 키만 제거
            keys_to_remove = [k for k in self.cache.keys() if pattern in k]
            for key in keys_to_remove:
                with self.locks[key]:
                    self.cache.pop(key, None)
                    self.timestamps.pop(key, None)
        else:
            # 전체 캐시 클리어
            self.cache.clear()
            self.timestamps.clear()


# ============================================================================
# 메인 GoogleSheetsManager 클래스
# ============================================================================

class GoogleSheetsManager:
    """Google Sheets API 관리자"""
    
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
        
        # 컴포넌트 초기화
        self.cache = SheetCache()
        self.rate_limiter = RateLimiter(RATE_LIMIT_PER_MINUTE)
        self.batch_queue = PriorityQueue()
        self._batch_processor = None
        self._stop_event = threading.Event()
        
        # 연결 상태
        self.is_connected = False
        self.last_error = None
        
        # 초기화 시도
        if GOOGLE_API_AVAILABLE:
            self._initialize()
    
    def _initialize(self):
        """서비스 초기화"""
        try:
            # 스프레드시트 URL 확인
            if not self.spreadsheet_url:
                self.spreadsheet_url = self._get_spreadsheet_url()
            
            if not self.spreadsheet_url:
                logger.info("Google Sheets URL이 설정되지 않았습니다. Sheets 기능 비활성화.")
                return
            
            # ID 추출
            self.spreadsheet_id = self._extract_spreadsheet_id(self.spreadsheet_url)
            if not self.spreadsheet_id:
                raise ValueError("유효하지 않은 스프레드시트 URL")
            
            # 인증
            self._authenticate()
            
            # 시트 구조 확인/생성
            self._ensure_sheet_structure()
            
            # 배치 프로세서 시작
            self._start_batch_processor()
            
            self.initialized = True
            self.is_connected = True
            logger.info(f"Google Sheets 연결 성공: {self.spreadsheet_id}")
            
        except Exception as e:
            self.last_error = str(e)
            logger.warning(f"Google Sheets 초기화 실패: {str(e)}")
            self.initialized = False
            self.is_connected = False
    
    def _get_spreadsheet_url(self) -> Optional[str]:
        """설정에서 스프레드시트 URL 가져오기"""
        # 1. 환경 변수
        url = os.environ.get('GOOGLE_SHEETS_URL')
        if url:
            return url
        
        # 2. Streamlit secrets
        try:
            import streamlit as st
            if 'google_sheets_url' in st.secrets:
                return st.secrets['google_sheets_url']
        except:
            pass
        
        # 3. 설정 파일
        if hasattr(GOOGLE_SHEETS_CONFIG, 'spreadsheet_url'):
            return GOOGLE_SHEETS_CONFIG.get('spreadsheet_url')
        
        return None
    
    def _extract_spreadsheet_id(self, url: str) -> Optional[str]:
        """URL에서 스프레드시트 ID 추출"""
        if not url:
            return None
        
        # 이미 ID인 경우
        if '/' not in url:
            return url
        
        # URL에서 ID 추출
        import re
        match = re.search(r'/spreadsheets/d/([a-zA-Z0-9-_]+)', url)
        if match:
            return match.group(1)
        
        return None
    
    def _authenticate(self):
        """Google Sheets API 인증"""
        creds = None
        
        # 1. 서비스 계정 시도
        creds = self._try_service_account()
        
        # 2. OAuth2 시도
        if not creds:
            creds = self._try_oauth2()
        
        if not creds:
            raise Exception("Google Sheets 인증 실패: API 키를 확인하세요")
        
        # 서비스 빌드
        self.service = build('sheets', API_VERSION, credentials=creds)
    
    def _try_service_account(self) -> Optional[Any]:
        """서비스 계정 인증 시도"""
        try:
            # 1. 파일 경로
            sa_file = os.environ.get('GOOGLE_SERVICE_ACCOUNT_FILE')
            if sa_file and os.path.exists(sa_file):
                return service_account.Credentials.from_service_account_file(
                    sa_file, scopes=SCOPES
                )
            
            # 2. JSON 문자열
            sa_json = os.environ.get('GOOGLE_SERVICE_ACCOUNT_JSON')
            if sa_json:
                sa_info = json.loads(sa_json)
                return service_account.Credentials.from_service_account_info(
                    sa_info, scopes=SCOPES
                )
            
            # 3. Streamlit secrets
            try:
                import streamlit as st
                if 'google_service_account' in st.secrets:
                    sa_info = dict(st.secrets['google_service_account'])
                    return service_account.Credentials.from_service_account_info(
                        sa_info, scopes=SCOPES
                    )
            except:
                pass
            
        except Exception as e:
            logger.debug(f"서비스 계정 인증 실패: {str(e)}")
        
        return None
    
    def _try_oauth2(self) -> Optional[Any]:
        """OAuth2 인증 시도"""
        try:
            token_file = LOCAL_CONFIG['app_data_dir'] / 'token.json'
            creds = None
            
            # 기존 토큰 로드
            if token_file.exists():
                creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)
            
            # 토큰 갱신 또는 새로 생성
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    # OAuth 플로우 (데스크톱 앱에서는 제한적)
                    logger.warning("OAuth2 인증이 필요합니다. 서비스 계정 사용을 권장합니다.")
                    return None
                
                # 토큰 저장
                token_file.parent.mkdir(parents=True, exist_ok=True)
                with open(token_file, 'w') as token:
                    token.write(creds.to_json())
            
            return creds
            
        except Exception as e:
            logger.debug(f"OAuth2 인증 실패: {str(e)}")
        
        return None
    
    def _ensure_sheet_structure(self):
        """시트 구조 확인 및 생성"""
        try:
            # 현재 시트 목록 가져오기
            spreadsheet = self.service.spreadsheets().get(
                spreadsheetId=self.spreadsheet_id
            ).execute()
            
            existing_sheets = {s['properties']['title'] for s in spreadsheet['sheets']}
            
            # 필요한 시트 생성
            requests = []
            for sheet_name, schema in SHEET_SCHEMAS.items():
                if sheet_name not in existing_sheets:
                    requests.append({
                        'addSheet': {
                            'properties': {
                                'title': sheet_name,
                                'gridProperties': {
                                    'rowCount': 1000,
                                    'columnCount': len(schema['columns'])
                                }
                            }
                        }
                    })
            
            if requests:
                self.service.spreadsheets().batchUpdate(
                    spreadsheetId=self.spreadsheet_id,
                    body={'requests': requests}
                ).execute()
                
                # 헤더 행 추가
                self._initialize_headers()
                
                logger.info(f"시트 구조 생성 완료: {len(requests)}개 시트")
        
        except Exception as e:
            logger.error(f"시트 구조 확인 실패: {str(e)}")
            raise
    
    def _initialize_headers(self):
        """각 시트에 헤더 행 추가"""
        for sheet_name, schema in SHEET_SCHEMAS.items():
            try:
                # 첫 번째 행에 헤더 추가
                self.service.spreadsheets().values().update(
                    spreadsheetId=self.spreadsheet_id,
                    range=f"{sheet_name}!A1:{self._col_num_to_letter(len(schema['columns']))}1",
                    valueInputOption='RAW',
                    body={'values': [schema['columns']]}
                ).execute()
            except:
                pass
    
    # ============================================================================
    # CRUD 작업
    # ============================================================================
    
    def read(self, sheet_name: str, 
             filters: Optional[Dict[str, Any]] = None,
             columns: Optional[List[str]] = None,
             order_by: Optional[str] = None,
             limit: Optional[int] = None,
             use_cache: bool = True) -> pd.DataFrame:
        """
        시트 데이터 읽기
        
        Args:
            sheet_name: 시트 이름
            filters: 필터 조건
            columns: 선택할 컬럼
            order_by: 정렬 기준
            limit: 최대 행 수
            use_cache: 캐시 사용 여부
            
        Returns:
            DataFrame
        """
        if not self.initialized:
            return pd.DataFrame()
        
        # 캐시 키 생성
        cache_key = f"{sheet_name}:{hash(str(filters))}"
        
        # 캐시 확인
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached
        
        try:
            # Rate limiting
            self.rate_limiter.acquire()
            
            # 전체 데이터 읽기
            result = self.service.spreadsheets().values().get(
                spreadsheetId=self.spreadsheet_id,
                range=f"{sheet_name}!A:Z"
            ).execute()
            
            values = result.get('values', [])
            
            if not values:
                return pd.DataFrame()
            
            # DataFrame 생성
            headers = values[0]
            data = values[1:]
            
            # 빈 행 채우기
            max_cols = len(headers)
            for row in data:
                while len(row) < max_cols:
                    row.append('')
            
            df = pd.DataFrame(data, columns=headers)
            
            # 타입 변환
            df = self._convert_types(df, sheet_name)
            
            # 필터링
            if filters:
                for col, value in filters.items():
                    if col in df.columns:
                        df = df[df[col] == value]
            
            # 컬럼 선택
            if columns:
                df = df[[c for c in columns if c in df.columns]]
            
            # 정렬
            if order_by and order_by in df.columns:
                df = df.sort_values(order_by)
            
            # 제한
            if limit:
                df = df.head(limit)
            
            # 캐시 저장
            if use_cache:
                self.cache.set(cache_key, df)
            
            return df
            
        except Exception as e:
            logger.error(f"읽기 실패 ({sheet_name}): {str(e)}")
            self.last_error = str(e)
            return pd.DataFrame()
    
    def create(self, sheet_name: str, data: Dict[str, Any]) -> Optional[str]:
        """
        새 행 추가
        
        Args:
            sheet_name: 시트 이름
            data: 추가할 데이터
            
        Returns:
            생성된 ID
        """
        if not self.initialized:
            return None
        
        try:
            # ID 생성
            if 'id' not in data:
                data['id'] = self._generate_id()
            
            # 타임스탬프 추가
            now = datetime.now().isoformat()
            data['created_at'] = now
            data['updated_at'] = now
            
            # 배치 큐에 추가
            self._add_to_batch({
                'operation': 'append',
                'sheet_name': sheet_name,
                'data': data
            }, priority=2)
            
            # 캐시 무효화
            self.cache.invalidate(sheet_name)
            
            return data['id']
            
        except Exception as e:
            logger.error(f"생성 실패 ({sheet_name}): {str(e)}")
            self.last_error = str(e)
            return None
    
    def update(self, sheet_name: str, record_id: str, 
               updates: Dict[str, Any]) -> bool:
        """
        행 업데이트
        
        Args:
            sheet_name: 시트 이름
            record_id: 레코드 ID
            updates: 업데이트할 데이터
            
        Returns:
            성공 여부
        """
        if not self.initialized:
            return False
        
        try:
            # 현재 데이터 읽기
            df = self.read(sheet_name, filters={'id': record_id}, use_cache=False)
            
            if df.empty:
                logger.warning(f"레코드를 찾을 수 없음: {sheet_name}#{record_id}")
                return False
            
            # 업데이트 데이터 준비
            updates['updated_at'] = datetime.now().isoformat()
            
            # 배치 큐에 추가
            self._add_to_batch({
                'operation': 'update',
                'sheet_name': sheet_name,
                'record_id': record_id,
                'updates': updates
            }, priority=3)
            
            # 캐시 무효화
            self.cache.invalidate(sheet_name)
            
            return True
            
        except Exception as e:
            logger.error(f"업데이트 실패 ({sheet_name}#{record_id}): {str(e)}")
            self.last_error = str(e)
            return False
    
    def delete(self, sheet_name: str, record_id: str, 
               soft: bool = True) -> bool:
        """
        행 삭제
        
        Args:
            sheet_name: 시트 이름
            record_id: 레코드 ID
            soft: 소프트 삭제 여부
            
        Returns:
            성공 여부
        """
        if not self.initialized:
            return False
        
        try:
            if soft:
                # 소프트 삭제
                return self.update(sheet_name, record_id, {
                    'is_deleted': True,
                    'deleted_at': datetime.now().isoformat()
                })
            else:
                # 하드 삭제
                self._add_to_batch({
                    'operation': 'delete',
                    'sheet_name': sheet_name,
                    'record_id': record_id
                }, priority=4)
                
                # 캐시 무효화
                self.cache.invalidate(sheet_name)
                
                return True
                
        except Exception as e:
            logger.error(f"삭제 실패 ({sheet_name}#{record_id}): {str(e)}")
            self.last_error = str(e)
            return False
    
    # ============================================================================
    # 배치 처리
    # ============================================================================
    
    def _add_to_batch(self, operation: Dict[str, Any], priority: int = 5):
        """배치 큐에 작업 추가"""
        self.batch_queue.put((priority, time.time(), operation))
    
    def _start_batch_processor(self):
        """배치 프로세서 시작"""
        self._batch_processor = threading.Thread(
            target=self._batch_processor_loop,
            daemon=True
        )
        self._batch_processor.start()
    
    def _batch_processor_loop(self):
        """배치 처리 루프"""
        batch = []
        last_flush = time.time()
        
        while not self._stop_event.is_set():
            try:
                # 큐에서 작업 가져오기
                try:
                    priority, timestamp, operation = self.batch_queue.get(timeout=1)
                    batch.append(operation)
                except Empty:
                    pass
                
                # 배치 플러시 조건
                should_flush = (
                    len(batch) >= 100 or  # 배치 크기
                    time.time() - last_flush > 5 or  # 시간 제한
                    (batch and self.batch_queue.empty())  # 큐가 비었을 때
                )
                
                if should_flush and batch:
                    self._flush_batch(batch)
                    batch = []
                    last_flush = time.time()
                    
            except Exception as e:
                logger.error(f"배치 프로세서 오류: {str(e)}")
    
    def _flush_batch(self, batch: List[Dict[str, Any]]):
        """배치 실행"""
        try:
            # 작업별로 그룹화
            by_operation = defaultdict(list)
            for op in batch:
                by_operation[op['operation']].append(op)
            
            # append 작업
            for ops in by_operation.get('append', []):
                self._batch_append(ops)
            
            # update 작업
            for ops in by_operation.get('update', []):
                self._batch_update(ops)
            
            # delete 작업
            for ops in by_operation.get('delete', []):
                self._batch_delete(ops)
                
        except Exception as e:
            logger.error(f"배치 플러시 실패: {str(e)}")
    
    def _batch_append(self, operations: List[Dict[str, Any]]):
        """배치 추가 작업"""
        # 시트별로 그룹화
        by_sheet = defaultdict(list)
        for op in operations:
            by_sheet[op['sheet_name']].append(op['data'])
        
        # 시트별 실행
        for sheet_name, data_list in by_sheet.items():
            try:
                # 행 데이터 준비
                schema = SHEET_SCHEMAS[sheet_name]
                rows = []
                
                for data in data_list:
                    row = [data.get(col, '') for col in schema['columns']]
                    rows.append(row)
                
                # API 호출
                self.rate_limiter.acquire()
                self.service.spreadsheets().values().append(
                    spreadsheetId=self.spreadsheet_id,
                    range=f"{sheet_name}!A:A",
                    valueInputOption='RAW',
                    insertDataOption='INSERT_ROWS',
                    body={'values': rows}
                ).execute()
                
                logger.debug(f"배치 추가 완료: {sheet_name} ({len(rows)}행)")
                
            except Exception as e:
                logger.error(f"배치 추가 실패 ({sheet_name}): {str(e)}")
    
    # ============================================================================
    # 유틸리티
    # ============================================================================
    
    def _generate_id(self) -> str:
        """고유 ID 생성"""
        import uuid
        return str(uuid.uuid4())
    
    def _convert_types(self, df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
        """데이터 타입 변환"""
        schema = SHEET_SCHEMAS.get(sheet_name, {})
        
        for col in df.columns:
            if col in ['created_at', 'updated_at', 'last_login', 'deleted_at']:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif col in ['is_active', 'is_deleted']:
                df[col] = df[col].apply(lambda x: str(x).lower() == 'true')
            elif col in ['settings', 'data', 'factors', 'responses', 'design_matrix', 'results']:
                df[col] = df[col].apply(lambda x: json.loads(x) if x else {})
            elif col in ['points', 'run_number']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _col_num_to_letter(self, n: int) -> str:
        """열 번호를 문자로 변환"""
        result = ""
        while n > 0:
            n, remainder = divmod(n - 1, 26)
            result = chr(65 + remainder) + result
        return result
    
    def get_connection_status(self) -> Dict[str, Any]:
        """연결 상태 확인"""
        return {
            'connected': self.is_connected,
            'initialized': self.initialized,
            'spreadsheet_id': self.spreadsheet_id,
            'last_error': self.last_error,
            'cache_size': len(self.cache.cache),
            'queue_size': self.batch_queue.qsize()
        }
    
    def test_connection(self) -> bool:
        """연결 테스트"""
        if not self.initialized:
            return False
        
        try:
            # 간단한 읽기 테스트
            self.service.spreadsheets().values().get(
                spreadsheetId=self.spreadsheet_id,
                range='A1'
            ).execute()
            
            self.is_connected = True
            return True
            
        except Exception as e:
            self.is_connected = False
            self.last_error = str(e)
            return False
    
    def close(self):
        """연결 종료 및 정리"""
        self._stop_event.set()
        
        # 남은 배치 처리
        if not self.batch_queue.empty():
            batch = []
            while not self.batch_queue.empty():
                try:
                    _, _, op = self.batch_queue.get_nowait()
                    batch.append(op)
                except:
                    break
            
            if batch:
                self._flush_batch(batch)
        
        logger.info("Google Sheets Manager 종료")


# ============================================================================
# 싱글톤 인스턴스
# ============================================================================

_sheets_manager: Optional[GoogleSheetsManager] = None


def get_sheets_manager(spreadsheet_url: Optional[str] = None) -> GoogleSheetsManager:
    """GoogleSheetsManager 싱글톤 인스턴스 반환"""
    global _sheets_manager
    
    if _sheets_manager is None:
        _sheets_manager = GoogleSheetsManager(spreadsheet_url)
    
    return _sheets_manager


# ============================================================================
# 간편 함수
# ============================================================================

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
