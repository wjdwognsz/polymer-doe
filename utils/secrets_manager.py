"""
통합 암호 관리 시스템

API 키와 인증 정보를 안전하게 관리하는 통합 시스템
- 로컬 암호화 저장 (AES-256)
- OS 키체인 통합
- 우선순위 기반 키 획득
- 데스크톱/웹 앱 모두 지원
"""

import streamlit as st
import os
import json
import sqlite3
import platform
import keyring
import logging
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from contextlib import contextmanager
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import tempfile
import zipfile

# 프로젝트 설정 임포트
try:
    from config.secrets_config import (
        API_KEY_STRUCTURE, GOOGLE_CONFIG, SECRET_PRIORITY,
        SECURITY_MESSAGES, VALIDATION_RULES, SECRETS_TOML_TEMPLATE
    )
    from config.app_config import API_CONFIG, FREE_API_DEFAULTS
    from config.local_config import get_app_directory
except ImportError:
    # 기본값 설정 (테스트/독립 실행용)
    API_KEY_STRUCTURE = {}
    GOOGLE_CONFIG = {}
    SECRET_PRIORITY = ['session_state', 'environment', 'default']
    SECURITY_MESSAGES = {}
    VALIDATION_RULES = {}

logger = logging.getLogger(__name__)


class SecureStorage:
    """안전한 로컬 저장소 관리"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._init_encryption()
    
    def _init_db(self):
        """암호화된 키 저장용 DB 초기화"""
        with self._get_connection() as conn:
            # 암호 저장 테이블
            conn.execute('''
                CREATE TABLE IF NOT EXISTS encrypted_secrets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    service TEXT NOT NULL,
                    encrypted_value TEXT NOT NULL,
                    iv TEXT NOT NULL,
                    metadata TEXT,
                    expires_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, service)
                )
            ''')
            
            # 접근 로그 테이블
            conn.execute('''
                CREATE TABLE IF NOT EXISTS access_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    service TEXT NOT NULL,
                    action TEXT NOT NULL,
                    success INTEGER DEFAULT 1,
                    ip_address TEXT,
                    user_agent TEXT,
                    error_message TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 인덱스 생성
            conn.execute('CREATE INDEX IF NOT EXISTS idx_secrets_user_service ON encrypted_secrets(user_id, service)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_secrets_expires ON encrypted_secrets(expires_at)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_log_timestamp ON access_log(timestamp)')
    
    def _init_encryption(self):
        """암호화 초기화"""
        master_key = self._get_or_create_master_key()
        self.cipher = Fernet(master_key)
    
    def _get_or_create_master_key(self) -> bytes:
        """로컬 마스터 키 관리"""
        key_file = self.db_path.parent / ".master.key"
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # 하드웨어 기반 키 생성
            salt = os.urandom(16)
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            
            # 시스템 정보 기반 키 파생
            system_info = f"{platform.node()}-{platform.machine()}-{os.getpid()}"
            key = base64.urlsafe_b64encode(kdf.derive(system_info.encode()))
            
            with open(key_file, 'wb') as f:
                f.write(key)
            
            # 파일 권한 설정 (소유자만 읽기)
            if platform.system() != 'Windows':
                os.chmod(key_file, 0o600)
            
            return key
    
    @contextmanager
    def _get_connection(self):
        """DB 연결 컨텍스트 매니저"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def encrypt_value(self, value: str) -> str:
        """값 암호화"""
        if not value:
            return ""
        
        encrypted = self.cipher.encrypt(value.encode())
        return base64.b64encode(encrypted).decode()
    
    def decrypt_value(self, encrypted: str) -> Optional[str]:
        """값 복호화"""
        if not encrypted:
            return None
        
        try:
            decrypted = self.cipher.decrypt(base64.b64decode(encrypted))
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return None
    
    def store_secret(self, service: str, value: str, user_id: Optional[str] = None,
                    metadata: Optional[Dict] = None, expires_in_days: Optional[int] = None):
        """암호 저장"""
        encrypted = self.encrypt_value(value)
        expires_at = None
        
        if expires_in_days:
            expires_at = (datetime.now() + timedelta(days=expires_in_days)).isoformat()
        
        with self._get_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO encrypted_secrets 
                (user_id, service, encrypted_value, iv, metadata, expires_at, updated_at)
                VALUES (?, ?, ?, '', ?, ?, CURRENT_TIMESTAMP)
            ''', (
                user_id or 'default',
                service,
                encrypted,
                json.dumps(metadata) if metadata else None,
                expires_at
            ))
    
    def get_secret(self, service: str, user_id: Optional[str] = None) -> Optional[str]:
        """암호 조회"""
        with self._get_connection() as conn:
            row = conn.execute('''
                SELECT encrypted_value, expires_at
                FROM encrypted_secrets
                WHERE user_id = ? AND service = ?
                AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            ''', (user_id or 'default', service)).fetchone()
            
            if row:
                return self.decrypt_value(row['encrypted_value'])
        
        return None
    
    def delete_secret(self, service: str, user_id: Optional[str] = None):
        """암호 삭제"""
        with self._get_connection() as conn:
            conn.execute(
                "DELETE FROM encrypted_secrets WHERE user_id = ? AND service = ?",
                (user_id or 'default', service)
            )
    
    def list_secrets(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """저장된 암호 목록"""
        with self._get_connection() as conn:
            rows = conn.execute('''
                SELECT service, metadata, expires_at, created_at, updated_at
                FROM encrypted_secrets
                WHERE user_id = ?
                ORDER BY service
            ''', (user_id or 'default',)).fetchall()
            
            return [dict(row) for row in rows]
    
    def log_access(self, service: str, action: str, success: bool = True,
                  user_id: Optional[str] = None, error_message: Optional[str] = None):
        """접근 로그 기록"""
        with self._get_connection() as conn:
            conn.execute('''
                INSERT INTO access_log (user_id, service, action, success, error_message)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id or 'default', service, action, int(success), error_message))
    
    def get_access_logs(self, service: Optional[str] = None, 
                       user_id: Optional[str] = None, 
                       days: int = 30) -> List[Dict[str, Any]]:
        """접근 로그 조회"""
        with self._get_connection() as conn:
            query = '''
                SELECT * FROM access_log
                WHERE timestamp > datetime('now', '-{} days')
            '''.format(days)
            
            params = []
            if service:
                query += " AND service = ?"
                params.append(service)
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            query += " ORDER BY timestamp DESC LIMIT 1000"
            
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]
    
    def cleanup_expired(self):
        """만료된 암호 정리"""
        with self._get_connection() as conn:
            conn.execute('''
                DELETE FROM encrypted_secrets
                WHERE expires_at IS NOT NULL AND expires_at < CURRENT_TIMESTAMP
            ''')
            
            # 오래된 로그 정리 (90일 이상)
            conn.execute('''
                DELETE FROM access_log
                WHERE timestamp < datetime('now', '-90 days')
            ''')


class SecretsManager:
    """API 키와 인증 정보 통합 관리"""
    
    def __init__(self, db_path: Optional[Path] = None):
        # 저장소 경로 설정
        if db_path is None:
            app_dir = get_app_directory() if 'get_app_directory' in globals() else Path.home() / '.universaldoe'
            db_path = app_dir / 'data' / 'secrets.db'
        
        self.storage = SecureStorage(db_path)
        self.keyring_service = "UniversalDOE"
        self._cache = {}
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """세션 상태 초기화"""
        if 'api_keys' not in st.session_state:
            st.session_state.api_keys = {}
        if 'google_auth' not in st.session_state:
            st.session_state.google_auth = {}
        if 'secrets_loaded' not in st.session_state:
            st.session_state.secrets_loaded = False
            self._load_all_secrets()
    
    def _load_all_secrets(self):
        """모든 저장된 암호 로드"""
        try:
            # API 키 로드
            for service_key in API_KEY_STRUCTURE:
                key = self.get_api_key(service_key)
                if key:
                    st.session_state.api_keys[service_key] = key
            
            # Google 설정 로드
            for config_key in GOOGLE_CONFIG:
                value = self.get_secret(f"google_{config_key}")
                if value:
                    st.session_state.google_auth[config_key] = value
            
            st.session_state.secrets_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load secrets: {str(e)}")
    
    def get_api_key(self, service: str, user_id: Optional[str] = None) -> Optional[str]:
        """우선순위에 따른 API 키 획득"""
        # 캐시 확인
        cache_key = f"{service}:{user_id or 'default'}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        key = None
        
        # 1. 세션 상태 확인
        if service in st.session_state.api_keys:
            key = st.session_state.api_keys[service]
        
        # 2. Streamlit Secrets 확인
        if not key and hasattr(st, 'secrets'):
            config = API_KEY_STRUCTURE.get(service, {})
            secrets_key = config.get('secrets_key')
            if secrets_key and secrets_key in st.secrets:
                key = st.secrets[secrets_key]
        
        # 3. 환경 변수 확인
        if not key:
            config = API_KEY_STRUCTURE.get(service, {})
            env_var = config.get('env_var')
            if env_var:
                key = os.environ.get(env_var)
        
        # 4. OS 키체인 확인
        if not key:
            try:
                key = keyring.get_password(self.keyring_service, service)
            except Exception as e:
                logger.debug(f"Keyring access failed: {str(e)}")
        
        # 5. 로컬 저장소 확인
        if not key:
            key = self.storage.get_secret(service, user_id)
        
        # 6. 무료 API 기본값
        if not key and service in FREE_API_DEFAULTS:
            key = FREE_API_DEFAULTS[service]
        
        # 캐시 저장
        if key:
            self._cache[cache_key] = key
            self.storage.log_access(service, 'get', success=True, user_id=user_id)
        else:
            self.storage.log_access(service, 'get', success=False, user_id=user_id,
                                  error_message="Key not found")
        
        return key
    
    def set_api_key(self, service: str, key: str, user_id: Optional[str] = None,
                   save_to_keyring: bool = True, expires_in_days: Optional[int] = None) -> bool:
        """API 키 저장"""
        try:
            # 유효성 검증
            if not self.validate_api_key(service, key):
                raise ValueError(f"Invalid API key for {service}")
            
            # 세션 상태 저장
            st.session_state.api_keys[service] = key
            
            # OS 키체인 저장
            if save_to_keyring:
                try:
                    keyring.set_password(self.keyring_service, service, key)
                except Exception as e:
                    logger.warning(f"Failed to save to keyring: {str(e)}")
            
            # 로컬 저장소 저장
            metadata = {
                'service': service,
                'saved_at': datetime.now().isoformat(),
                'expires_in_days': expires_in_days
            }
            self.storage.store_secret(service, key, user_id, metadata, expires_in_days)
            
            # 캐시 갱신
            cache_key = f"{service}:{user_id or 'default'}"
            self._cache[cache_key] = key
            
            # 접근 로그
            self.storage.log_access(service, 'set', success=True, user_id=user_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save API key: {str(e)}")
            self.storage.log_access(service, 'set', success=False, user_id=user_id,
                                  error_message=str(e))
            return False
    
    def delete_api_key(self, service: str, user_id: Optional[str] = None) -> bool:
        """API 키 삭제"""
        try:
            # 세션 상태에서 삭제
            if service in st.session_state.api_keys:
                del st.session_state.api_keys[service]
            
            # OS 키체인에서 삭제
            try:
                keyring.delete_password(self.keyring_service, service)
            except Exception:
                pass  # 키체인에 없을 수도 있음
            
            # 로컬 저장소에서 삭제
            self.storage.delete_secret(service, user_id)
            
            # 캐시에서 삭제
            cache_key = f"{service}:{user_id or 'default'}"
            if cache_key in self._cache:
                del self._cache[cache_key]
            
            # 접근 로그
            self.storage.log_access(service, 'delete', success=True, user_id=user_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete API key: {str(e)}")
            self.storage.log_access(service, 'delete', success=False, user_id=user_id,
                                  error_message=str(e))
            return False
    
    def validate_api_key(self, service: str, key: str) -> bool:
        """API 키 유효성 검증"""
        if not key:
            return False
        
        config = API_KEY_STRUCTURE.get(service, {})
        
        # 길이 검증
        min_length = VALIDATION_RULES.get('api_key_min_length', 20)
        max_length = VALIDATION_RULES.get('api_key_max_length', 200)
        if not (min_length <= len(key) <= max_length):
            return False
        
        # 프리픽스 검증
        prefix = config.get('validation_prefix')
        if prefix and not key.startswith(prefix):
            return False
        
        return True
    
    def get_secret(self, key: str, user_id: Optional[str] = None) -> Optional[str]:
        """일반 암호 조회"""
        return self.storage.get_secret(key, user_id)
    
    def set_secret(self, key: str, value: str, user_id: Optional[str] = None,
                  expires_in_days: Optional[int] = None) -> bool:
        """일반 암호 저장"""
        try:
            self.storage.store_secret(key, value, user_id, expires_in_days=expires_in_days)
            return True
        except Exception as e:
            logger.error(f"Failed to save secret: {str(e)}")
            return False
    
    def export_secrets(self, user_id: Optional[str] = None, include_keys: bool = False) -> Dict[str, Any]:
        """암호 내보내기"""
        export_data = {
            'version': '1.0',
            'exported_at': datetime.now().isoformat(),
            'services': {}
        }
        
        # 저장된 서비스 목록
        secrets = self.storage.list_secrets(user_id)
        
        for secret in secrets:
            service = secret['service']
            if include_keys:
                # 실제 키 포함 (보안 주의!)
                key = self.storage.get_secret(service, user_id)
                if key:
                    export_data['services'][service] = {
                        'key': key,
                        'metadata': json.loads(secret.get('metadata', '{}'))
                    }
            else:
                # 메타데이터만 포함
                export_data['services'][service] = {
                    'has_key': True,
                    'metadata': json.loads(secret.get('metadata', '{}'))
                }
        
        return export_data
    
    def import_secrets(self, import_data: Dict[str, Any], user_id: Optional[str] = None) -> Tuple[int, int]:
        """암호 가져오기"""
        success_count = 0
        failure_count = 0
        
        services = import_data.get('services', {})
        
        for service, data in services.items():
            try:
                if 'key' in data:
                    # 실제 키가 포함된 경우
                    if self.set_api_key(service, data['key'], user_id):
                        success_count += 1
                    else:
                        failure_count += 1
                else:
                    # 메타데이터만 있는 경우 (스킵)
                    continue
                    
            except Exception as e:
                logger.error(f"Failed to import {service}: {str(e)}")
                failure_count += 1
        
        return success_count, failure_count
    
    def render_secrets_ui(self):
        """Streamlit UI 렌더링"""
        st.subheader("🔐 API 키 관리")
        
        # 탭 생성
        tabs = st.tabs(["API 키 설정", "Google 인증", "고급 설정", "접근 로그"])
        
        # API 키 설정 탭
        with tabs[0]:
            self._render_api_keys_tab()
        
        # Google 인증 탭
        with tabs[1]:
            self._render_google_auth_tab()
        
        # 고급 설정 탭
        with tabs[2]:
            self._render_advanced_tab()
        
        # 접근 로그 탭
        with tabs[3]:
            self._render_access_logs_tab()
    
    def _render_api_keys_tab(self):
        """API 키 설정 탭"""
        st.markdown("### AI 엔진 API 키")
        
        # API 키 상태 표시
        cols = st.columns(2)
        
        for i, (service, config) in enumerate(API_KEY_STRUCTURE.items()):
            col = cols[i % 2]
            
            with col:
                with st.expander(f"{service.replace('_', ' ').title()}", expanded=False):
                    current_key = self.get_api_key(service)
                    
                    # 상태 표시
                    if current_key:
                        if current_key == FREE_API_DEFAULTS.get(service):
                            st.success("✅ 무료 API 사용 중")
                        else:
                            masked_key = current_key[:10] + "..." + current_key[-4:] if len(current_key) > 14 else "***"
                            st.success(f"✅ 설정됨: {masked_key}")
                    else:
                        if config.get('required'):
                            st.error("❌ 필수 API 키가 설정되지 않았습니다")
                        else:
                            st.warning("⚠️ 선택적 API 키가 설정되지 않았습니다")
                    
                    # 키 입력
                    new_key = st.text_input(
                        "API 키 입력",
                        type="password",
                        key=f"api_key_{service}",
                        help=f"환경변수: {config.get('env_var', 'N/A')}"
                    )
                    
                    # 버튼들
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("저장", key=f"save_{service}"):
                            if new_key:
                                if self.set_api_key(service, new_key):
                                    st.success(SECURITY_MESSAGES.get('api_key_saved', "저장되었습니다"))
                                    st.rerun()
                                else:
                                    st.error(SECURITY_MESSAGES.get('api_key_invalid', "유효하지 않은 키입니다"))
                    
                    with col2:
                        if current_key and st.button("삭제", key=f"delete_{service}"):
                            if self.delete_api_key(service):
                                st.success("삭제되었습니다")
                                st.rerun()
                    
                    with col3:
                        if st.button("가이드", key=f"guide_{service}"):
                            st.info(f"{service} API 키 발급 가이드를 참고하세요")
    
    def _render_google_auth_tab(self):
        """Google 인증 탭"""
        st.markdown("### Google 서비스 설정")
        
        # Google Sheets URL
        with st.expander("Google Sheets URL", expanded=True):
            sheets_url = st.text_input(
                "스프레드시트 URL",
                value=st.session_state.google_auth.get('sheets_url', ''),
                help="https://docs.google.com/spreadsheets/d/... 형식"
            )
            
            if st.button("URL 저장"):
                if sheets_url and sheets_url.startswith('https://docs.google.com/spreadsheets/'):
                    st.session_state.google_auth['sheets_url'] = sheets_url
                    self.set_secret('google_sheets_url', sheets_url)
                    st.success("저장되었습니다")
                else:
                    st.error("유효한 Google Sheets URL을 입력하세요")
        
        # 서비스 계정 / OAuth
        auth_method = st.radio(
            "인증 방법",
            ["개인 Google 계정 (OAuth)", "서비스 계정 (JSON)"],
            index=0
        )
        
        if auth_method == "서비스 계정 (JSON)":
            uploaded_file = st.file_uploader(
                "서비스 계정 JSON 파일",
                type=['json'],
                help="Google Cloud Console에서 다운로드한 서비스 계정 키 파일"
            )
            
            if uploaded_file:
                try:
                    service_account = json.load(uploaded_file)
                    st.session_state.google_auth['service_account'] = service_account
                    self.set_secret('google_service_account', json.dumps(service_account))
                    st.success("서비스 계정이 설정되었습니다")
                except Exception as e:
                    st.error(f"파일 읽기 실패: {str(e)}")
    
    def _render_advanced_tab(self):
        """고급 설정 탭"""
        st.markdown("### 고급 설정")
        
        # 가져오기/내보내기
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 설정 내보내기")
            include_keys = st.checkbox("API 키 포함 (보안 주의!)", value=False)
            
            if st.button("내보내기"):
                export_data = self.export_secrets(include_keys=include_keys)
                
                # JSON 파일로 다운로드
                json_str = json.dumps(export_data, indent=2)
                st.download_button(
                    label="📥 다운로드",
                    data=json_str,
                    file_name=f"universaldoe_secrets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            st.markdown("#### 설정 가져오기")
            uploaded_file = st.file_uploader("JSON 파일 선택", type=['json'])
            
            if uploaded_file:
                try:
                    import_data = json.load(uploaded_file)
                    if st.button("가져오기 실행"):
                        success, failure = self.import_secrets(import_data)
                        st.success(f"성공: {success}개, 실패: {failure}개")
                        st.rerun()
                except Exception as e:
                    st.error(f"가져오기 실패: {str(e)}")
        
        # 정리 작업
        st.markdown("---")
        st.markdown("#### 유지보수")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("만료된 키 정리"):
                self.storage.cleanup_expired()
                st.success("정리되었습니다")
        
        with col2:
            if st.button("캐시 초기화"):
                self._cache.clear()
                st.success("캐시가 초기화되었습니다")
        
        with col3:
            if st.button("모든 키 다시 로드"):
                st.session_state.secrets_loaded = False
                self._load_all_secrets()
                st.success("다시 로드되었습니다")
                st.rerun()
    
    def _render_access_logs_tab(self):
        """접근 로그 탭"""
        st.markdown("### 접근 로그")
        
        # 필터
        col1, col2, col3 = st.columns(3)
        
        with col1:
            service_filter = st.selectbox(
                "서비스",
                ["전체"] + list(API_KEY_STRUCTURE.keys()),
                key="log_service_filter"
            )
        
        with col2:
            days_filter = st.number_input(
                "최근 N일",
                min_value=1,
                max_value=90,
                value=7,
                key="log_days_filter"
            )
        
        with col3:
            if st.button("로그 조회", key="refresh_logs"):
                st.rerun()
        
        # 로그 표시
        logs = self.storage.get_access_logs(
            service=None if service_filter == "전체" else service_filter,
            days=days_filter
        )
        
        if logs:
            # 통계
            total = len(logs)
            success = sum(1 for log in logs if log['success'])
            failure = total - success
            
            col1, col2, col3 = st.columns(3)
            col1.metric("전체 접근", total)
            col2.metric("성공", success)
            col3.metric("실패", failure)
            
            # 로그 테이블
            st.dataframe(
                logs[:100],  # 최대 100개만 표시
                use_container_width=True,
                height=400
            )
        else:
            st.info("접근 로그가 없습니다")
    
    def get_status_summary(self) -> Dict[str, Any]:
        """전체 상태 요약"""
        summary = {
            'total_services': len(API_KEY_STRUCTURE),
            'configured': 0,
            'required_missing': [],
            'optional_missing': [],
            'using_free': []
        }
        
        for service, config in API_KEY_STRUCTURE.items():
            key = self.get_api_key(service)
            
            if key:
                summary['configured'] += 1
                if key == FREE_API_DEFAULTS.get(service):
                    summary['using_free'].append(service)
            else:
                if config.get('required'):
                    summary['required_missing'].append(service)
                else:
                    summary['optional_missing'].append(service)
        
        return summary
    
    def render_status_badge(self):
        """상태 배지 표시"""
        summary = self.get_status_summary()
        
        if summary['required_missing']:
            st.error(f"❌ 필수 API 키 {len(summary['required_missing'])}개 누락")
        elif summary['optional_missing']:
            st.warning(f"⚠️ 선택 API 키 {len(summary['optional_missing'])}개 미설정")
        else:
            st.success(f"✅ 모든 API 키 설정 완료 ({summary['configured']}/{summary['total_services']})")


# 싱글톤 인스턴스
_secrets_manager = None

def get_secrets_manager(db_path: Optional[Path] = None) -> SecretsManager:
    """SecretsManager 싱글톤 인스턴스 반환"""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager(db_path)
    return _secrets_manager


# 유틸리티 함수
def render_api_key_input(service: str, label: Optional[str] = None) -> Optional[str]:
    """단일 API 키 입력 위젯"""
    manager = get_secrets_manager()
    current_key = manager.get_api_key(service)
    
    if label is None:
        label = f"{service.replace('_', ' ').title()} API Key"
    
    # 현재 상태 표시
    if current_key:
        st.success(f"✅ {label} 설정됨")
    else:
        st.warning(f"⚠️ {label} 미설정")
    
    # 키 입력
    new_key = st.text_input(
        label,
        type="password",
        key=f"quick_api_key_{service}"
    )
    
    if new_key and st.button(f"저장", key=f"quick_save_{service}"):
        if manager.set_api_key(service, new_key):
            st.success("저장되었습니다")
            st.rerun()
        else:
            st.error("유효하지 않은 API 키입니다")
    
    return current_key


def check_required_secrets() -> bool:
    """필수 암호 확인"""
    manager = get_secrets_manager()
    summary = manager.get_status_summary()
    
    if summary['required_missing']:
        st.error(f"필수 API 키가 설정되지 않았습니다: {', '.join(summary['required_missing'])}")
        
        with st.expander("API 키 설정하기"):
            manager.render_secrets_ui()
        
        return False
    
    return True


def export_secrets_template():
    """secrets.toml 템플릿 생성"""
    return SECRETS_TOML_TEMPLATE
