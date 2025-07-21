"""통합 암호 관리 시스템"""
import streamlit as st
import os
import json
from typing import Optional, Dict, Any, List
from cryptography.fernet import Fernet
import base64
import logging
from datetime import datetime
from pathlib import Path

from config.secrets_config import (
    API_KEY_STRUCTURE, GOOGLE_CONFIG, SECRET_PRIORITY,
    validate_api_key, SECURITY_MESSAGES, get_secrets_template
)
from config.app_config import API_CONFIG, FREE_API_DEFAULTS

logger = logging.getLogger(__name__)

class SecretsManager:
    """API 키와 인증 정보 통합 관리"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.initialize_session_state()
        self._encryption_key = self._get_or_create_encryption_key()
        self._cipher = Fernet(self._encryption_key)
    
    def initialize_session_state(self):
        """세션 상태 초기화"""
        if 'api_keys' not in st.session_state:
            st.session_state.api_keys = {}
        if 'google_auth' not in st.session_state:
            st.session_state.google_auth = {}
        if 'secrets_loaded' not in st.session_state:
            st.session_state.secrets_loaded = False
            # 초기 로드 시 DB에서 키 복원
            if self.db_manager:
                self._load_keys_from_db()
    
    def _get_or_create_encryption_key(self) -> bytes:
        """암호화 키 획득 또는 생성"""
        key_file = Path.home() / '.universaldoe' / '.encryption_key'
        key_file.parent.mkdir(parents=True, exist_ok=True)
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            # 파일 권한 설정 (소유자만 읽기 가능)
            os.chmod(key_file, 0o600)
            return key
    
    def _encrypt(self, data: str) -> str:
        """데이터 암호화"""
        return self._cipher.encrypt(data.encode()).decode()
    
    def _decrypt(self, encrypted_data: str) -> str:
        """데이터 복호화"""
        try:
            return self._cipher.decrypt(encrypted_data.encode()).decode()
        except Exception as e:
            logger.error(f"Decryption failed: {str(e)}")
            return ""
    
    def _load_keys_from_db(self):
        """DB에서 저장된 API 키 로드"""
        if not self.db_manager:
            return
        
        try:
            # 현재 사용자 ID 가져오기 (인증 전이면 로컬 키 사용)
            user_id = st.session_state.get('user', {}).get('id', 0)
            
            # DB에서 암호화된 키 조회
            api_keys = self.db_manager.get_all(
                'api_keys',
                filters={'user_id': user_id} if user_id else None
            )
            
            for record in api_keys:
                service = record['service']
                encrypted_key = record['encrypted_key']
                
                # 복호화하여 세션에 저장
                try:
                    decrypted_key = self._decrypt(encrypted_key)
                    if decrypted_key:
                        st.session_state.api_keys[service] = decrypted_key
                except Exception as e:
                    logger.error(f"Failed to decrypt key for {service}: {str(e)}")
            
            st.session_state.secrets_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load keys from DB: {str(e)}")
    
    def get_api_key(self, service: str) -> Optional[str]:
        """
        API 키 획득 (우선순위 적용)
        
        Returns:
            API 키 또는 None
        """
        # 1. 세션 상태 확인 (앱 내 입력)
        if service in st.session_state.api_keys:
            key = st.session_state.api_keys[service]
            if key and validate_api_key(service, key):
                return key
        
        # 2. Streamlit Secrets 확인
        try:
            key_name = API_KEY_STRUCTURE.get(service, {}).get('secrets_key')
            if key_name and key_name in st.secrets:
                key = st.secrets[key_name]
                if validate_api_key(service, key):
                    # 세션에 캐시
                    st.session_state.api_keys[service] = key
                    return key
        except:
            pass
        
        # 3. 환경 변수 확인
        env_var = API_KEY_STRUCTURE.get(service, {}).get('env_var')
        if env_var:
            key = os.environ.get(env_var)
            if key and validate_api_key(service, key):
                # 세션에 캐시
                st.session_state.api_keys[service] = key
                return key
        
        # 4. 무료 API 기본값 확인
        if service in FREE_API_DEFAULTS:
            default = FREE_API_DEFAULTS[service].get('default_key')
            if default:
                return default
        
        return None
    
    def set_api_key(self, service: str, key: str) -> bool:
        """API 키 설정"""
        if not validate_api_key(service, key):
            return False
        
        # 세션 상태에 저장
        st.session_state.api_keys[service] = key
        
        # DB에 암호화하여 저장
        if self.db_manager:
            try:
                user_id = st.session_state.get('user', {}).get('id', 0)
                encrypted_key = self._encrypt(key)
                
                # 기존 키가 있으면 업데이트, 없으면 삽입
                existing = self.db_manager.fetchone(
                    "SELECT id FROM api_keys WHERE user_id = ? AND service = ?",
                    (user_id, service)
                )
                
                if existing:
                    self.db_manager.update(
                        'api_keys',
                        existing['id'],
                        {'encrypted_key': encrypted_key}
                    )
                else:
                    self.db_manager.insert(
                        'api_keys',
                        {
                            'user_id': user_id,
                            'service': service,
                            'encrypted_key': encrypted_key
                        }
                    )
                
                logger.info(f"API key for {service} saved to database")
                
            except Exception as e:
                logger.error(f"Failed to save API key to DB: {str(e)}")
        
        return True
    
    def remove_api_key(self, service: str):
        """API 키 삭제"""
        # 세션에서 제거
        if service in st.session_state.api_keys:
            del st.session_state.api_keys[service]
        
        # DB에서 제거
        if self.db_manager:
            try:
                user_id = st.session_state.get('user', {}).get('id', 0)
                self.db_manager.execute(
                    "DELETE FROM api_keys WHERE user_id = ? AND service = ?",
                    (user_id, service)
                )
            except Exception as e:
                logger.error(f"Failed to remove API key from DB: {str(e)}")
    
    def get_google_sheets_url(self) -> Optional[str]:
        """Google Sheets URL 획득"""
        # 세션 상태
        if 'sheets_url' in st.session_state.google_auth:
            return st.session_state.google_auth['sheets_url']
        
        # Streamlit Secrets
        try:
            if 'google_sheets_url' in st.secrets:
                return st.secrets['google_sheets_url']
        except:
            pass
        
        # 환경 변수
        return os.environ.get('GOOGLE_SHEETS_URL')
    
    def render_secrets_ui(self):
        """암호 입력 UI 렌더링"""
        with st.expander("🔑 API 키 및 인증 정보 설정", expanded=False):
            st.info("""
            💡 **API 키 설정 방법:**
            1. 아래에서 직접 입력하거나
            2. `.streamlit/secrets.toml` 파일에 저장하거나
            3. 환경 변수로 설정할 수 있습니다.
            
            입력된 API 키는 로컬에 암호화되어 안전하게 저장됩니다.
            """)
            
            # API 키 입력 탭
            tab1, tab2, tab3 = st.tabs(["필수 API", "선택 API", "Google 설정"])
            
            with tab1:
                self._render_required_apis()
            
            with tab2:
                self._render_optional_apis()
            
            with tab3:
                self._render_google_settings()
            
            # 저장 버튼
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("💾 설정 저장", type="primary", use_container_width=True):
                    self._save_settings()
            with col2:
                if st.button("🔄 새로고침", use_container_width=True):
                    self._load_keys_from_db()
                    st.rerun()
            with col3:
                if st.button("📋 secrets.toml 템플릿", use_container_width=True):
                    self._show_secrets_template()
    
    def _render_required_apis(self):
        """필수 API 입력 UI"""
        st.subheader("필수 API 키")
        
        for service, config in API_CONFIG.items():
            if config.get('required'):
                current_key = self.get_api_key(service)
                has_key = current_key is not None
                
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"**{config['name']}**")
                    st.caption(f"문서: {config.get('docs_url', 'N/A')}")
                with col2:
                    if has_key:
                        st.success("✅ 설정됨")
                    else:
                        st.error("❌ 미설정")
                with col3:
                    if has_key and st.button("삭제", key=f"del_{service}"):
                        self.remove_api_key(service)
                        st.rerun()
                
                if not has_key:
                    key = st.text_input(
                        f"{service} API Key",
                        type="password",
                        key=f"input_{service}_key",
                        help=f"{config['name']} API 키를 입력하세요"
                    )
                    if key:
                        if self.set_api_key(service, key):
                            st.success(f"✅ {service} 키가 유효합니다")
                            st.rerun()
                        else:
                            st.error(f"❌ 유효하지 않은 {service} 키입니다")
    
    def _render_optional_apis(self):
        """선택 API 입력 UI"""
        st.subheader("선택 API 키")
        st.caption("추가 기능을 위한 선택적 API입니다")
        
        # 현재 설정된 선택 API 표시
        optional_services = [s for s, c in API_CONFIG.items() if not c.get('required')]
        configured_optional = [s for s in optional_services if self.get_api_key(s)]
        
        if configured_optional:
            st.write("**설정된 API:**")
            for service in configured_optional:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"✅ {API_CONFIG[service]['name']}")
                with col2:
                    if st.button("삭제", key=f"del_opt_{service}"):
                        self.remove_api_key(service)
                        st.rerun()
        
        # 새 API 추가
        st.write("**API 추가:**")
        available_services = [s for s in optional_services if s not in configured_optional]
        
        if available_services:
            selected_service = st.selectbox(
                "추가할 API 선택",
                available_services,
                format_func=lambda x: API_CONFIG[x]['name']
            )
            
            if selected_service:
                config = API_CONFIG[selected_service]
                st.info(f"{config.get('description', '')}")
                if config.get('free_tier'):
                    st.success("🆓 무료 티어 제공")
                
                key = st.text_input(
                    f"{selected_service} API Key",
                    type="password",
                    key=f"opt_{selected_service}_key"
                )
                
                if st.button(f"➕ {selected_service} 추가", key=f"add_{selected_service}"):
                    if key and self.set_api_key(selected_service, key):
                        st.success(f"✅ {selected_service} 키 저장됨")
                        st.rerun()
                    else:
                        st.error("유효한 API 키를 입력하세요")
        else:
            st.info("모든 선택 API가 설정되었습니다")
    
    def _render_google_settings(self):
        """Google 설정 UI"""
        st.subheader("Google 연동 설정")
        
        # Google Sheets URL
        sheets_url = st.text_input(
            "Google Sheets URL",
            value=self.get_google_sheets_url() or "",
            placeholder="https://docs.google.com/spreadsheets/d/...",
            help="데이터 저장용 Google Sheets URL (선택사항)"
        )
        if sheets_url:
            st.session_state.google_auth['sheets_url'] = sheets_url
        
        # 온라인 동기화 옵션
        sync_enabled = st.checkbox(
            "온라인 시 Google Sheets 동기화 활성화",
            value=st.session_state.google_auth.get('sync_enabled', False)
        )
        st.session_state.google_auth['sync_enabled'] = sync_enabled
        
        if sync_enabled:
            # 인증 방법 선택
            auth_method = st.radio(
                "인증 방법",
                ["개인 Google 계정", "서비스 계정 (고급)"],
                help="개인 사용자는 Google 계정 인증을 추천합니다"
            )
            
            if auth_method == "서비스 계정 (고급)":
                service_account_json = st.text_area(
                    "서비스 계정 JSON",
                    height=200,
                    help="Google Cloud Console에서 발급받은 서비스 계정 키"
                )
                if service_account_json:
                    try:
                        json.loads(service_account_json)
                        st.session_state.google_auth['service_account'] = service_account_json
                        st.success("✅ 유효한 서비스 계정 JSON")
                    except:
                        st.error("❌ 유효하지 않은 JSON 형식")
    
    def _save_settings(self):
        """설정 저장"""
        saved_count = 0
        
        # 세션의 모든 API 키 DB에 저장
        for service, key in st.session_state.api_keys.items():
            if self.set_api_key(service, key):
                saved_count += 1
        
        # Google 설정 저장
        if st.session_state.google_auth:
            # TODO: Google 설정도 DB에 저장
            pass
        
        if saved_count > 0:
            st.success(f"✅ {saved_count}개의 API 키가 저장되었습니다")
        else:
            st.info("저장할 새로운 설정이 없습니다")
    
    def _show_secrets_template(self):
        """secrets.toml 템플릿 표시"""
        with st.expander("📋 secrets.toml 템플릿", expanded=True):
            st.code(get_secrets_template(), language='toml')
            st.info("""
            **사용 방법:**
            1. 프로젝트 루트에 `.streamlit` 폴더 생성
            2. `.streamlit/secrets.toml` 파일 생성
            3. 위 템플릿 복사 후 실제 키 입력
            4. `.gitignore`에 `secrets.toml` 추가
            
            **주의:** 데스크톱 앱에서는 앱 내 입력이 더 안전하고 편리합니다.
            """)
    
    def check_required_keys(self) -> Dict[str, bool]:
        """필수 API 키 확인"""
        status = {}
        for service, config in API_CONFIG.items():
            if config.get('required'):
                status[service] = self.get_api_key(service) is not None
        return status
    
    def get_available_apis(self) -> List[str]:
        """사용 가능한 API 목록"""
        available = []
        for service in API_CONFIG:
            if self.get_api_key(service):
                available.append(service)
        return available
    
    def export_config(self) -> Dict[str, Any]:
        """현재 설정을 딕셔너리로 내보내기 (백업용)"""
        config = {
            'api_keys': {},
            'google_auth': st.session_state.google_auth.copy()
        }
        
        # API 키는 마스킹하여 저장
        for service in st.session_state.api_keys:
            key = st.session_state.api_keys[service]
            if key:
                # 앞 4자리와 뒤 4자리만 보이도록
                masked = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "****"
                config['api_keys'][service] = {
                    'configured': True,
                    'masked_key': masked
                }
        
        return config
    
    def import_config(self, config: Dict[str, Any]):
        """설정 가져오기 (복원용)"""
        # Google 설정 복원
        if 'google_auth' in config:
            st.session_state.google_auth.update(config['google_auth'])
        
        # API 키는 보안상 복원하지 않음 (다시 입력 필요)
        st.warning("API 키는 보안상 다시 입력해야 합니다")


# 싱글톤 인스턴스
_secrets_manager = None

def get_secrets_manager(db_manager=None) -> SecretsManager:
    """SecretsManager 싱글톤 인스턴스 반환"""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager(db_manager)
    return _secrets_manager
