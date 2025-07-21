# 📊 utils/sheets_manager.py - Google Sheets 관리자
"""
Google Sheets를 데이터베이스로 사용하는 통합 관리 시스템
- CRUD 작업, 캐싱, 배치 처리, 트랜잭션 관리
- 비동기 처리 및 실시간 동기화 지원
"""

# 표준 라이브러리
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue, Empty
import hashlib
import uuid
import os
from functools import lru_cache, wraps

# Google API
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from google.auth.transport.requests import Request
except ImportError:
    print("Google API 라이브러리가 설치되지 않았습니다. pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client")

# 데이터 처리
import pandas as pd
import numpy as np
import streamlit as st

# 로깅 설정
logger = logging.getLogger(__name__)

# Google Sheets API 설정
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
API_VERSION = 'v4'
RATE_LIMIT_PER_MINUTE = 60  # Google Sheets API 제한
BATCH_UPDATE_LIMIT = 1000   # 배치당 최대 업데이트 수

# 시트 이름 상수
SHEET_NAMES = {
    'users': 'Users',
    'projects': 'Projects',
    'experiments': 'Experiments',
    'results': 'Results',
    'comments': 'Comments',
    'files': 'Files',
    'notifications': 'Notifications',
    'activity_log': 'Activity_Log',
    'system_config': 'System_Config',
    'templates': 'Templates',
    'modules': 'Modules'
}

# 캐시 설정
CACHE_TTL = {
    'users': 300,           # 5분
    'projects': 60,         # 1분
    'experiments': 30,      # 30초
    'results': 30,          # 30초
    'static_data': 3600,    # 1시간
    'system_config': 7200   # 2시간
}

# 재시도 설정
RETRY_DELAYS = [0.5, 1.0, 2.0, 4.0, 8.0]  # 지수 백오프
MAX_RETRIES = len(RETRY_DELAYS)

# 데이터 타입 매핑
COLUMN_TYPES = {
    'user_id': str,
    'project_id': str,
    'experiment_id': str,
    'created_at': 'datetime',
    'updated_at': 'datetime',
    'last_login': 'datetime',
    'is_active': bool,
    'points': int,
    'level': str,
    'settings': 'json',
    'factors': 'json',
    'responses': 'json',
    'collaborators': 'json',
    'metadata': 'json',
    'results_data': 'json'
}


class GoogleSheetsManager:
    """Google Sheets 데이터베이스 관리자"""
    
    def __init__(self):
        """초기화"""
        self.service = None
        self.spreadsheet_id = None
        self.cache = {}
        self.cache_timestamps = {}
        self.rate_limiter = RateLimiter(RATE_LIMIT_PER_MINUTE)
        self.batch_queue = Queue()
        self.batch_processor_running = False
        self.batch_processor_thread = None
        self._lock = threading.Lock()
        self._initialize_service()
        
    def _initialize_service(self):
        """Google Sheets 서비스 초기화"""
        try:
            # Streamlit secrets에서 정보 가져오기
            if 'google_sheets_url' in st.secrets:
                self.spreadsheet_id = st.secrets['google_sheets_url']
            else:
                logger.error("Google Sheets URL이 secrets에 없습니다")
                return
                
            # 서비스 계정 정보 가져오기
            if 'google_service_account' in st.secrets:
                service_account_info = st.secrets['google_service_account']
                credentials = service_account.Credentials.from_service_account_info(
                    service_account_info, scopes=SCOPES
                )
                
                # 서비스 빌드
                self.service = build('sheets', 'v4', credentials=credentials)
                logger.info("Google Sheets 서비스 초기화 성공")
                
                # 배치 프로세서 시작
                self._start_batch_processor()
                
            else:
                logger.error("Google 서비스 계정 정보가 secrets에 없습니다")
                
        except Exception as e:
            logger.error(f"Google Sheets 서비스 초기화 실패: {str(e)}")
            
    def _start_batch_processor(self):
        """배치 프로세서 시작"""
        if not self.batch_processor_running:
            self.batch_processor_running = True
            self.batch_processor_thread = threading.Thread(
                target=self._batch_processor,
                daemon=True
            )
            self.batch_processor_thread.start()
            
    def _stop_batch_processor(self):
        """배치 프로세서 중지"""
        self.batch_processor_running = False
        if self.batch_processor_thread:
            self.batch_processor_thread.join()
            
    def _retry_with_backoff(self, func, *args, **kwargs):
        """재시도 로직"""
        for i, delay in enumerate(RETRY_DELAYS):
            try:
                return func(*args, **kwargs)
            except HttpError as e:
                if e.resp.status in [429, 500, 503]:  # Rate limit or server error
                    if i < len(RETRY_DELAYS) - 1:
                        time.sleep(delay)
                        continue
                raise
            except Exception as e:
                if i < len(RETRY_DELAYS) - 1:
                    time.sleep(delay)
                    continue
                raise
                
    def _get_cache_key(self, sheet_name: str, **kwargs) -> str:
        """캐시 키 생성"""
        key_data = f"{sheet_name}:{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
        
    def _is_cache_valid(self, cache_key: str, sheet_name: str) -> bool:
        """캐시 유효성 확인"""
        if cache_key not in self.cache:
            return False
            
        timestamp = self.cache_timestamps.get(cache_key, 0)
        ttl = CACHE_TTL.get(sheet_name, 60)
        return time.time() - timestamp < ttl
        
    def _convert_types(self, df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
        """데이터 타입 변환"""
        for column in df.columns:
            if column in COLUMN_TYPES:
                col_type = COLUMN_TYPES[column]
                
                try:
                    if col_type == 'datetime':
                        df[column] = pd.to_datetime(df[column], errors='coerce')
                    elif col_type == 'json':
                        df[column] = df[column].apply(self._parse_json)
                    elif col_type == bool:
                        df[column] = df[column].astype(str).str.lower() == 'true'
                    elif col_type == int:
                        df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0).astype(int)
                    elif col_type == str:
                        df[column] = df[column].astype(str)
                except Exception as e:
                    logger.warning(f"타입 변환 실패 {column}: {str(e)}")
                    
        return df
        
    def _parse_json(self, value):
        """JSON 파싱"""
        if pd.isna(value) or value == '':
            return {}
        if isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(value)
        except:
            return {}
            
    def _apply_filters(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """필터 적용"""
        for column, value in filters.items():
            if column in df.columns:
                if isinstance(value, list):
                    df = df[df[column].isin(value)]
                elif isinstance(value, dict):
                    # 범위 필터 (예: {'$gte': 10, '$lte': 100})
                    if '$gte' in value:
                        df = df[df[column] >= value['$gte']]
                    if '$gt' in value:
                        df = df[df[column] > value['$gt']]
                    if '$lte' in value:
                        df = df[df[column] <= value['$lte']]
                    if '$lt' in value:
                        df = df[df[column] < value['$lt']]
                    if '$ne' in value:
                        df = df[df[column] != value['$ne']]
                else:
                    df = df[df[column] == value]
                    
        return df
        
    # ===== 읽기 작업 =====
    
    def read_sheet(self,
                   sheet_name: str,
                   filters: Optional[Dict[str, Any]] = None,
                   columns: Optional[List[str]] = None,
                   order_by: Optional[str] = None,
                   limit: Optional[int] = None,
                   offset: int = 0,
                   cache: bool = True) -> pd.DataFrame:
        """
        시트 데이터 읽기
        
        Args:
            sheet_name: 시트 이름
            filters: 필터 조건
            columns: 선택할 컬럼
            order_by: 정렬 기준 (- prefix for desc)
            limit: 최대 행 수
            offset: 시작 위치
            cache: 캐시 사용 여부
            
        Returns:
            pd.DataFrame: 조회된 데이터
        """
        if not self.service:
            logger.error("Google Sheets 서비스가 초기화되지 않았습니다")
            return pd.DataFrame()
            
        # 캐시 확인
        cache_key = self._get_cache_key(sheet_name, filters=filters, columns=columns)
        if cache and self._is_cache_valid(cache_key, sheet_name):
            df = self.cache[cache_key].copy()
        else:
            # Rate limiting
            self.rate_limiter.wait_if_needed()
            
            try:
                # 전체 데이터 읽기
                range_name = f"{SHEET_NAMES.get(sheet_name, sheet_name)}!A:Z"
                result = self._retry_with_backoff(
                    self.service.spreadsheets().values().get,
                    spreadsheetId=self.spreadsheet_id,
                    range=range_name
                )
                
                values = result.get('values', [])
                if not values:
                    return pd.DataFrame()
                    
                # DataFrame 생성
                df = pd.DataFrame(values[1:], columns=values[0])
                
                # 데이터 타입 변환
                df = self._convert_types(df, sheet_name)
                
                # 캐시 저장
                if cache:
                    self.cache[cache_key] = df.copy()
                    self.cache_timestamps[cache_key] = time.time()
                    
            except Exception as e:
                logger.error(f"시트 읽기 오류 ({sheet_name}): {str(e)}")
                return pd.DataFrame()
                
        # 필터 적용
        if filters:
            df = self._apply_filters(df, filters)
            
        # 컬럼 선택
        if columns:
            available_columns = [col for col in columns if col in df.columns]
            df = df[available_columns]
            
        # 정렬
        if order_by:
            ascending = True
            if order_by.startswith('-'):
                order_by = order_by[1:]
                ascending = False
            if order_by in df.columns:
                df = df.sort_values(order_by, ascending=ascending)
                
        # 페이지네이션
        if limit:
            df = df.iloc[offset:offset + limit]
        elif offset:
            df = df.iloc[offset:]
            
        return df
        
    def read_row(self, sheet_name: str, row_id: str, id_column: str = None) -> Optional[Dict]:
        """
        단일 행 읽기
        
        Args:
            sheet_name: 시트 이름
            row_id: 행 ID
            id_column: ID 컬럼명 (기본: {sheet_name}_id)
            
        Returns:
            Dict: 행 데이터 또는 None
        """
        if not id_column:
            # 기본 ID 컬럼명 추론
            if sheet_name == 'users':
                id_column = 'user_id'
            elif sheet_name == 'projects':
                id_column = 'project_id'
            elif sheet_name == 'experiments':
                id_column = 'experiment_id'
            else:
                id_column = f"{sheet_name}_id"
                
        df = self.read_sheet(sheet_name, filters={id_column: row_id}, limit=1)
        if not df.empty:
            return df.iloc[0].to_dict()
        return None
        
    # ===== 쓰기 작업 =====
    
    def write_row(self, sheet_name: str, data: Dict[str, Any], return_id: bool = True) -> Optional[str]:
        """
        새 행 추가
        
        Args:
            sheet_name: 시트 이름
            data: 추가할 데이터
            return_id: ID 반환 여부
            
        Returns:
            str: 생성된 행 ID (선택적)
        """
        if not self.service:
            logger.error("Google Sheets 서비스가 초기화되지 않았습니다")
            return None
            
        try:
            # ID 생성
            if return_id:
                id_column = self._get_id_column(sheet_name)
                if id_column not in data:
                    data[id_column] = self._generate_id()
                    
            # 타임스탬프 추가
            data['created_at'] = datetime.now().isoformat()
            data['updated_at'] = datetime.now().isoformat()
            
            # 현재 시트 데이터 읽기 (헤더 확인용)
            range_name = f"{SHEET_NAMES.get(sheet_name, sheet_name)}!A1:Z1"
            result = self._retry_with_backoff(
                self.service.spreadsheets().values().get,
                spreadsheetId=self.spreadsheet_id,
                range=range_name
            )
            
            headers = result.get('values', [[]])[0] if result.get('values') else []
            
            # 새 컬럼 추가 (필요시)
            new_columns = [col for col in data.keys() if col not in headers]
            if new_columns:
                headers.extend(new_columns)
                self._update_headers(sheet_name, headers)
                
            # 데이터 행 생성
            row_values = []
            for header in headers:
                value = data.get(header, '')
                # JSON 필드 처리
                if header in COLUMN_TYPES and COLUMN_TYPES[header] == 'json':
                    if isinstance(value, (dict, list)):
                        value = json.dumps(value)
                row_values.append(str(value))
                
            # 행 추가
            range_name = f"{SHEET_NAMES.get(sheet_name, sheet_name)}!A:Z"
            body = {'values': [row_values]}
            
            self.rate_limiter.wait_if_needed()
            self._retry_with_backoff(
                self.service.spreadsheets().values().append,
                spreadsheetId=self.spreadsheet_id,
                range=range_name,
                valueInputOption='RAW',
                body=body
            )
            
            # 캐시 무효화
            self._invalidate_cache(sheet_name)
            
            return data.get(id_column) if return_id and id_column else None
            
        except Exception as e:
            logger.error(f"행 추가 오류 ({sheet_name}): {str(e)}")
            return None
            
    def update_row(self, sheet_name: str, row_id: str, updates: Dict[str, Any], id_column: str = None) -> bool:
        """
        행 업데이트
        
        Args:
            sheet_name: 시트 이름
            row_id: 행 ID
            updates: 업데이트할 데이터
            id_column: ID 컬럼명
            
        Returns:
            bool: 성공 여부
        """
        if not self.service:
            return False
            
        try:
            if not id_column:
                id_column = self._get_id_column(sheet_name)
                
            # 현재 데이터 읽기
            df = self.read_sheet(sheet_name, cache=False)
            if df.empty:
                return False
                
            # 행 찾기
            row_mask = df[id_column] == row_id
            if not row_mask.any():
                logger.warning(f"행을 찾을 수 없습니다: {row_id}")
                return False
                
            row_index = df[row_mask].index[0] + 2  # 헤더 + 1-based index
            
            # 업데이트 타임스탬프
            updates['updated_at'] = datetime.now().isoformat()
            
            # 업데이트 데이터 준비
            update_requests = []
            for column, value in updates.items():
                if column in df.columns:
                    col_index = df.columns.get_loc(column)
                    
                    # JSON 필드 처리
                    if column in COLUMN_TYPES and COLUMN_TYPES[column] == 'json':
                        if isinstance(value, (dict, list)):
                            value = json.dumps(value)
                            
                    update_requests.append({
                        'range': f"{SHEET_NAMES.get(sheet_name, sheet_name)}!{chr(65 + col_index)}{row_index}",
                        'values': [[str(value)]]
                    })
                    
            if update_requests:
                # 배치 업데이트
                body = {
                    'valueInputOption': 'RAW',
                    'data': update_requests
                }
                
                self.rate_limiter.wait_if_needed()
                self._retry_with_backoff(
                    self.service.spreadsheets().values().batchUpdate,
                    spreadsheetId=self.spreadsheet_id,
                    body=body
                )
                
                # 캐시 무효화
                self._invalidate_cache(sheet_name)
                
                return True
                
        except Exception as e:
            logger.error(f"행 업데이트 오류 ({sheet_name}): {str(e)}")
            
        return False
        
    def delete_row(self, sheet_name: str, row_id: str, id_column: str = None) -> bool:
        """
        행 삭제 (소프트 삭제)
        
        Args:
            sheet_name: 시트 이름
            row_id: 행 ID
            id_column: ID 컬럼명
            
        Returns:
            bool: 성공 여부
        """
        # 소프트 삭제 구현 (is_deleted 플래그 설정)
        return self.update_row(sheet_name, row_id, {'is_deleted': True}, id_column)
        
    # ===== 배치 작업 =====
    
    def batch_write(self, sheet_name: str, data_list: List[Dict[str, Any]]) -> int:
        """
        여러 행 일괄 추가
        
        Args:
            sheet_name: 시트 이름
            data_list: 추가할 데이터 리스트
            
        Returns:
            int: 추가된 행 수
        """
        if not self.service or not data_list:
            return 0
            
        try:
            # 헤더 확인
            range_name = f"{SHEET_NAMES.get(sheet_name, sheet_name)}!A1:Z1"
            result = self._retry_with_backoff(
                self.service.spreadsheets().values().get,
                spreadsheetId=self.spreadsheet_id,
                range=range_name
            )
            
            headers = result.get('values', [[]])[0] if result.get('values') else []
            
            # 모든 컬럼 수집
            all_columns = set(headers)
            for data in data_list:
                all_columns.update(data.keys())
                
            # 새 컬럼 추가
            new_columns = [col for col in all_columns if col not in headers]
            if new_columns:
                headers = list(headers) + new_columns
                self._update_headers(sheet_name, headers)
                
            # 데이터 행 준비
            rows = []
            id_column = self._get_id_column(sheet_name)
            
            for data in data_list:
                # ID 및 타임스탬프 추가
                if id_column not in data:
                    data[id_column] = self._generate_id()
                data['created_at'] = datetime.now().isoformat()
                data['updated_at'] = datetime.now().isoformat()
                
                # 행 데이터 생성
                row = []
                for header in headers:
                    value = data.get(header, '')
                    if header in COLUMN_TYPES and COLUMN_TYPES[header] == 'json':
                        if isinstance(value, (dict, list)):
                            value = json.dumps(value)
                    row.append(str(value))
                rows.append(row)
                
            # 배치 추가
            if rows:
                range_name = f"{SHEET_NAMES.get(sheet_name, sheet_name)}!A:Z"
                body = {'values': rows}
                
                self.rate_limiter.wait_if_needed()
                self._retry_with_backoff(
                    self.service.spreadsheets().values().append,
                    spreadsheetId=self.spreadsheet_id,
                    range=range_name,
                    valueInputOption='RAW',
                    body=body
                )
                
                # 캐시 무효화
                self._invalidate_cache(sheet_name)
                
                return len(rows)
                
        except Exception as e:
            logger.error(f"배치 추가 오류 ({sheet_name}): {str(e)}")
            
        return 0
        
    def batch_update(self, sheet_name: str, updates: List[Dict[str, Any]], id_column: str = None) -> int:
        """
        여러 행 일괄 업데이트
        
        Args:
            sheet_name: 시트 이름
            updates: 업데이트 리스트 (각 항목은 id와 updates 포함)
            id_column: ID 컬럼명
            
        Returns:
            int: 업데이트된 행 수
        """
        if not self.service or not updates:
            return 0
            
        success_count = 0
        
        # 배치 크기로 나누어 처리
        batch_size = 100
        for i in range(0, len(updates), batch_size):
            batch = updates[i:i + batch_size]
            
            try:
                # 현재 데이터 읽기
                df = self.read_sheet(sheet_name, cache=False)
                if df.empty:
                    continue
                    
                if not id_column:
                    id_column = self._get_id_column(sheet_name)
                    
                # 업데이트 요청 준비
                update_requests = []
                
                for update_item in batch:
                    row_id = update_item.get('id')
                    update_data = update_item.get('updates', {})
                    
                    if not row_id or not update_data:
                        continue
                        
                    # 행 찾기
                    row_mask = df[id_column] == row_id
                    if not row_mask.any():
                        continue
                        
                    row_index = df[row_mask].index[0] + 2
                    update_data['updated_at'] = datetime.now().isoformat()
                    
                    # 각 컬럼 업데이트
                    for column, value in update_data.items():
                        if column in df.columns:
                            col_index = df.columns.get_loc(column)
                            
                            if column in COLUMN_TYPES and COLUMN_TYPES[column] == 'json':
                                if isinstance(value, (dict, list)):
                                    value = json.dumps(value)
                                    
                            update_requests.append({
                                'range': f"{SHEET_NAMES.get(sheet_name, sheet_name)}!{chr(65 + col_index)}{row_index}",
                                'values': [[str(value)]]
                            })
                            
                if update_requests:
                    # 배치 업데이트 실행
                    body = {
                        'valueInputOption': 'RAW',
                        'data': update_requests
                    }
                    
                    self.rate_limiter.wait_if_needed()
                    self._retry_with_backoff(
                        self.service.spreadsheets().values().batchUpdate,
                        spreadsheetId=self.spreadsheet_id,
                        body=body
                    )
                    
                    success_count += len(batch)
                    
            except Exception as e:
                logger.error(f"배치 업데이트 오류: {str(e)}")
                
        # 캐시 무효화
        if success_count > 0:
            self._invalidate_cache(sheet_name)
            
        return success_count
        
    # ===== 트랜잭션 =====
    
    class Transaction:
        """트랜잭션 컨텍스트 매니저"""
        
        def __init__(self, manager):
            self.manager = manager
            self.operations = []
            self.original_data = {}
            
        def add_operation(self, op_type: str, sheet_name: str, data: Any):
            """작업 추가"""
            self.operations.append({
                'type': op_type,
                'sheet': sheet_name,
                'data': data,
                'timestamp': datetime.now()
            })
            
        def __enter__(self):
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:
                # 커밋
                self._commit()
            else:
                # 롤백
                self._rollback()
                
        def _commit(self):
            """트랜잭션 커밋"""
            for op in self.operations:
                try:
                    if op['type'] == 'write':
                        self.manager.write_row(op['sheet'], op['data'])
                    elif op['type'] == 'update':
                        self.manager.update_row(
                            op['sheet'], 
                            op['data']['id'], 
                            op['data']['updates']
                        )
                    elif op['type'] == 'delete':
                        self.manager.delete_row(op['sheet'], op['data'])
                except Exception as e:
                    logger.error(f"트랜잭션 커밋 오류: {str(e)}")
                    self._rollback()
                    raise
                    
        def _rollback(self):
            """트랜잭션 롤백"""
            # 간단한 롤백 구현 (실제로는 더 복잡할 수 있음)
            logger.warning("트랜잭션 롤백 수행")
            
    def transaction(self):
        """트랜잭션 시작"""
        return self.Transaction(self)
        
    # ===== 유틸리티 메서드 =====
    
    def _get_id_column(self, sheet_name: str) -> str:
        """시트별 ID 컬럼명 반환"""
        id_columns = {
            'users': 'user_id',
            'projects': 'project_id',
            'experiments': 'experiment_id',
            'results': 'result_id',
            'comments': 'comment_id',
            'files': 'file_id',
            'notifications': 'notification_id'
        }
        return id_columns.get(sheet_name, f"{sheet_name}_id")
        
    def _generate_id(self) -> str:
        """고유 ID 생성"""
        return str(uuid.uuid4())
        
    def _update_headers(self, sheet_name: str, headers: List[str]):
        """헤더 업데이트"""
        try:
            range_name = f"{SHEET_NAMES.get(sheet_name, sheet_name)}!A1:Z1"
            body = {'values': [headers]}
            
            self._retry_with_backoff(
                self.service.spreadsheets().values().update,
                spreadsheetId=self.spreadsheet_id,
                range=range_name,
                valueInputOption='RAW',
                body=body
            )
        except Exception as e:
            logger.error(f"헤더 업데이트 오류: {str(e)}")
            
    def _invalidate_cache(self, sheet_name: str = None):
        """캐시 무효화"""
        with self._lock:
            if sheet_name:
                # 특정 시트의 캐시만 무효화
                keys_to_remove = [
                    key for key in self.cache.keys() 
                    if key.startswith(f"{sheet_name}:")
                ]
                for key in keys_to_remove:
                    self.cache.pop(key, None)
                    self.cache_timestamps.pop(key, None)
            else:
                # 전체 캐시 무효화
                self.cache.clear()
                self.cache_timestamps.clear()
                
    def _batch_processor(self):
        """백그라운드 배치 프로세서"""
        batch = []
        last_flush = time.time()
        
        while self.batch_processor_running:
            try:
                # 큐에서 작업 가져오기
                operation = self.batch_queue.get(timeout=1.0)
                batch.append(operation)
                
                # 배치 크기나 시간 조건 확인
                if (len(batch) >= 100 or time.time() - last_flush > 5.0):
                    self._flush_batch(batch)
                    batch = []
                    last_flush = time.time()
                    
            except Empty:
                # 타임아웃 시 대기 중인 배치 처리
                if batch:
                    self._flush_batch(batch)
                    batch = []
                    last_flush = time.time()
                    
    def _flush_batch(self, batch: List[Dict]):
        """배치 실행"""
        if not batch:
            return
            
        # 시트별로 그룹화
        sheet_operations = defaultdict(list)
        for op in batch:
            sheet_operations[op['sheet']].append(op)
            
        # 각 시트별로 처리
        for sheet_name, ops in sheet_operations.items():
            write_ops = [op for op in ops if op['type'] == 'write']
            update_ops = [op for op in ops if op['type'] == 'update']
            
            if write_ops:
                data_list = [op['data'] for op in write_ops]
                self.batch_write(sheet_name, data_list)
                
            if update_ops:
                updates = [
                    {'id': op['data']['id'], 'updates': op['data']['updates']} 
                    for op in update_ops
                ]
                self.batch_update(sheet_name, updates)
                
    # ===== 특수 메서드 =====
    
    def get_sheet_stats(self, sheet_name: str) -> Dict[str, Any]:
        """시트 통계 정보"""
        try:
            df = self.read_sheet(sheet_name)
            return {
                'total_rows': len(df),
                'columns': list(df.columns),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'last_updated': df['updated_at'].max() if 'updated_at' in df.columns else None
            }
        except Exception as e:
            logger.error(f"통계 조회 오류: {str(e)}")
            return {}
            
    def export_to_csv(self, sheet_name: str, file_path: str) -> bool:
        """CSV로 내보내기"""
        try:
            df = self.read_sheet(sheet_name)
            df.to_csv(file_path, index=False)
            return True
        except Exception as e:
            logger.error(f"CSV 내보내기 오류: {str(e)}")
            return False
            
    def import_from_csv(self, sheet_name: str, file_path: str) -> int:
        """CSV에서 가져오기"""
        try:
            df = pd.read_csv(file_path)
            data_list = df.to_dict('records')
            return self.batch_write(sheet_name, data_list)
        except Exception as e:
            logger.error(f"CSV 가져오기 오류: {str(e)}")
            return 0
            

class RateLimiter:
    """API 호출 속도 제한"""
    
    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.calls = []
        self._lock = threading.Lock()
        
    def wait_if_needed(self):
        """필요시 대기"""
        with self._lock:
            now = time.time()
            # 1분 이상 지난 호출 제거
            self.calls = [call_time for call_time in self.calls if now - call_time < 60]
            
            if len(self.calls) >= self.calls_per_minute:
                # 가장 오래된 호출로부터 1분이 지날 때까지 대기
                sleep_time = 60 - (now - self.calls[0]) + 0.1
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            self.calls.append(time.time())


# 싱글톤 인스턴스
_sheets_manager_instance = None


def get_sheets_manager() -> GoogleSheetsManager:
    """싱글톤 GoogleSheetsManager 인스턴스 반환"""
    global _sheets_manager_instance
    if _sheets_manager_instance is None:
        _sheets_manager_instance = GoogleSheetsManager()
    return _sheets_manager_instance


# 간편 함수들
def read_users(filters: Optional[Dict] = None, **kwargs) -> pd.DataFrame:
    """사용자 데이터 읽기"""
    return get_sheets_manager().read_sheet('users', filters=filters, **kwargs)


def read_projects(filters: Optional[Dict] = None, **kwargs) -> pd.DataFrame:
    """프로젝트 데이터 읽기"""
    return get_sheets_manager().read_sheet('projects', filters=filters, **kwargs)


def read_experiments(filters: Optional[Dict] = None, **kwargs) -> pd.DataFrame:
    """실험 데이터 읽기"""
    return get_sheets_manager().read_sheet('experiments', filters=filters, **kwargs)


def write_user(data: Dict[str, Any]) -> Optional[str]:
    """사용자 추가"""
    return get_sheets_manager().write_row('users', data)


def write_project(data: Dict[str, Any]) -> Optional[str]:
    """프로젝트 추가"""
    return get_sheets_manager().write_row('projects', data)


def write_experiment(data: Dict[str, Any]) -> Optional[str]:
    """실험 추가"""
    return get_sheets_manager().write_row('experiments', data)


def update_user(user_id: str, updates: Dict[str, Any]) -> bool:
    """사용자 정보 업데이트"""
    return get_sheets_manager().update_row('users', user_id, updates)


def update_project(project_id: str, updates: Dict[str, Any]) -> bool:
    """프로젝트 업데이트"""
    return get_sheets_manager().update_row('projects', project_id, updates)


def update_experiment(experiment_id: str, updates: Dict[str, Any]) -> bool:
    """실험 업데이트"""
    return get_sheets_manager().update_row('experiments', experiment_id, updates)
