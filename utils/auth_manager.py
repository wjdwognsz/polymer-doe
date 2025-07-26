"""
🔐 Authentication Manager - 통합 인증 시스템
===========================================================================
데스크톱 애플리케이션을 위한 포괄적인 인증 관리자
로컬 인증, OAuth, API 키, 2FA 지원, 오프라인 우선 설계
===========================================================================
"""

import os
import sys
import json
import sqlite3
import logging
import secrets
import threading
import pyotp
import qrcode
import io
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from functools import wraps, lru_cache
from contextlib import contextmanager
import hashlib
import base64
import ipaddress
from urllib.parse import urlencode, parse_qs

# 보안 라이브러리
import bcrypt
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# OAuth 라이브러리
try:
    from authlib.integrations.requests_client import OAuth2Session
    OAUTH_AVAILABLE = True
except ImportError:
    OAUTH_AVAILABLE = False
    logging.warning("OAuth libraries not available. Social login disabled.")

# Streamlit
import streamlit as st

# 로컬 모듈
from config.local_config import LOCAL_CONFIG
from config.offline_config import OFFLINE_CONFIG
from config.app_config import SECURITY_CONFIG, SESSION_CONFIG

# ===========================================================================
# 🔧 설정 및 상수
# ===========================================================================

logger = logging.getLogger(__name__)

# 보안 설정
BCRYPT_ROUNDS = 12
JWT_ALGORITHM = "HS256"
TOKEN_EXPIRY_HOURS = 24
REFRESH_TOKEN_DAYS = 30
SESSION_TIMEOUT_MINUTES = 30
API_KEY_LENGTH = 32
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION_MINUTES = 30

# OAuth 설정
OAUTH_PROVIDERS = {
    'google': {
        'authorize_url': 'https://accounts.google.com/o/oauth2/v2/auth',
        'token_url': 'https://oauth2.googleapis.com/token',
        'userinfo_url': 'https://www.googleapis.com/oauth2/v2/userinfo',
        'scope': 'openid email profile'
    },
    'github': {
        'authorize_url': 'https://github.com/login/oauth/authorize',
        'token_url': 'https://github.com/login/oauth/access_token',
        'userinfo_url': 'https://api.github.com/user',
        'scope': 'user:email'
    }
}

# 권한 레벨
class UserRole:
    GUEST = "guest"
    USER = "user"
    PREMIUM = "premium"
    ADMIN = "admin"

ROLE_HIERARCHY = {
    UserRole.GUEST: 0,
    UserRole.USER: 1,
    UserRole.PREMIUM: 2,
    UserRole.ADMIN: 3
}

# 권한 매트릭스
PERMISSION_MATRIX = {
    'project': {
        'create': [UserRole.USER, UserRole.PREMIUM, UserRole.ADMIN],
        'read': [UserRole.GUEST, UserRole.USER, UserRole.PREMIUM, UserRole.ADMIN],
        'update': [UserRole.USER, UserRole.PREMIUM, UserRole.ADMIN],
        'delete': [UserRole.PREMIUM, UserRole.ADMIN],
        'share': [UserRole.PREMIUM, UserRole.ADMIN]
    },
    'experiment': {
        'create': [UserRole.USER, UserRole.PREMIUM, UserRole.ADMIN],
        'run': [UserRole.USER, UserRole.PREMIUM, UserRole.ADMIN],
        'analyze': [UserRole.USER, UserRole.PREMIUM, UserRole.ADMIN],
        'export': [UserRole.PREMIUM, UserRole.ADMIN]
    },
    'module': {
        'use_basic': [UserRole.GUEST, UserRole.USER, UserRole.PREMIUM, UserRole.ADMIN],
        'use_advanced': [UserRole.PREMIUM, UserRole.ADMIN],
        'create': [UserRole.PREMIUM, UserRole.ADMIN],
        'share': [UserRole.ADMIN]
    },
    'api': {
        'read': [UserRole.USER, UserRole.PREMIUM, UserRole.ADMIN],
        'write': [UserRole.PREMIUM, UserRole.ADMIN],
        'admin': [UserRole.ADMIN]
    }
}

# ===========================================================================
# 🔐 인증 관리자 클래스
# ===========================================================================

class AuthManager:
    """통합 인증 관리자"""
    
    def __init__(self, db_manager=None):
        """
        초기화
        
        Args:
            db_manager: DatabaseManager 인스턴스 (선택적)
        """
        self.db_manager = db_manager
        self._lock = threading.Lock()
        self._failed_attempts = {}  # 로그인 실패 추적
        self._sessions = {}  # 활성 세션 관리
        self._api_keys = {}  # API 키 캐시
        self._ip_whitelist = set()  # IP 화이트리스트
        self._2fa_secrets = {}  # 2FA 시크릿 캐시
        
        # 암호화 키 설정
        self._setup_encryption()
        
        # JWT 시크릿 키
        self.jwt_secret = self._get_or_create_jwt_secret()
        
        # OAuth 클라이언트 설정
        self._setup_oauth_clients()
        
        # 세션 정리 스케줄러 시작
        self._start_session_cleanup()
        
        # IP 화이트리스트 로드
        self._load_ip_whitelist()
        
        logger.info("AuthManager initialized with full feature set")
    
    def _setup_encryption(self):
        """암호화 설정"""
        key_file = LOCAL_CONFIG['app_data_dir'] / '.keys' / 'master.key'
        key_file.parent.mkdir(parents=True, exist_ok=True)
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            # 파일 권한 설정 (소유자만 읽기)
            os.chmod(key_file, 0o600)
        
        self.cipher = Fernet(key)
    
    def _get_or_create_jwt_secret(self) -> str:
        """JWT 시크릿 키 생성 또는 로드"""
        secret_file = LOCAL_CONFIG['app_data_dir'] / '.keys' / 'jwt_secret.key'
        secret_file.parent.mkdir(parents=True, exist_ok=True)
        
        if secret_file.exists():
            with open(secret_file, 'r') as f:
                return f.read().strip()
        else:
            secret = secrets.token_urlsafe(64)
            with open(secret_file, 'w') as f:
                f.write(secret)
            os.chmod(secret_file, 0o600)
            return secret
    
    def _setup_oauth_clients(self):
        """OAuth 클라이언트 설정"""
        self.oauth_clients = {}
        
        if not OAUTH_AVAILABLE:
            logger.warning("OAuth not available - social login disabled")
            return
        
        # Google OAuth 설정
        if os.getenv('GOOGLE_OAUTH_CLIENT_ID'):
            self.oauth_clients['google'] = OAuth2Session(
                client_id=os.getenv('GOOGLE_OAUTH_CLIENT_ID'),
                client_secret=os.getenv('GOOGLE_OAUTH_CLIENT_SECRET'),
                redirect_uri=os.getenv('GOOGLE_OAUTH_REDIRECT_URI', 'http://localhost:8501/auth/callback'),
                scope=OAUTH_PROVIDERS['google']['scope']
            )
        
        # GitHub OAuth 설정
        if os.getenv('GITHUB_OAUTH_CLIENT_ID'):
            self.oauth_clients['github'] = OAuth2Session(
                client_id=os.getenv('GITHUB_OAUTH_CLIENT_ID'),
                client_secret=os.getenv('GITHUB_OAUTH_CLIENT_SECRET'),
                redirect_uri=os.getenv('GITHUB_OAUTH_REDIRECT_URI', 'http://localhost:8501/auth/callback'),
                scope=OAUTH_PROVIDERS['github']['scope']
            )
    
    def _start_session_cleanup(self):
        """세션 정리 스케줄러 시작"""
        def cleanup():
            while True:
                try:
                    self._cleanup_expired_sessions()
                    self._cleanup_locked_accounts()
                except Exception as e:
                    logger.error(f"Session cleanup error: {str(e)}")
                threading.Event().wait(300)  # 5분마다 실행
        
        cleanup_thread = threading.Thread(target=cleanup, daemon=True)
        cleanup_thread.start()
    
    def _load_ip_whitelist(self):
        """IP 화이트리스트 로드"""
        whitelist_file = LOCAL_CONFIG['app_data_dir'] / 'security' / 'ip_whitelist.json'
        if whitelist_file.exists():
            try:
                with open(whitelist_file, 'r') as f:
                    data = json.load(f)
                    self._ip_whitelist = set(data.get('whitelist', []))
            except Exception as e:
                logger.error(f"Failed to load IP whitelist: {str(e)}")
    
    # ===========================================================================
    # 🔐 로컬 인증
    # ===========================================================================
    
    def register_user(self, email: str, password: str, name: str, 
                     organization: Optional[str] = None,
                     role: str = UserRole.USER) -> Tuple[bool, str, Optional[int]]:
        """
        새 사용자 등록
        
        Args:
            email: 이메일 주소
            password: 비밀번호
            name: 사용자 이름
            organization: 소속 기관 (선택)
            role: 사용자 역할
            
        Returns:
            (성공 여부, 메시지, 사용자 ID)
        """
        with self._lock:
            try:
                # 이메일 중복 확인
                if self._check_email_exists(email):
                    return False, "이미 등록된 이메일입니다.", None
                
                # 비밀번호 해시화
                hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(BCRYPT_ROUNDS))
                
                # 사용자 생성
                user_data = {
                    'email': email,
                    'password_hash': hashed.decode('utf-8'),
                    'name': name,
                    'organization': organization,
                    'role': role,
                    'created_at': datetime.now().isoformat(),
                    'is_active': True,
                    'is_locked': False,
                    'failed_login_attempts': 0,
                    'last_login': None,
                    'settings': json.dumps({
                        'theme': 'light',
                        'language': 'ko',
                        'notifications': True
                    })
                }
                
                # DB에 저장
                if self.db_manager:
                    user_id = self.db_manager.create_user(user_data)
                else:
                    user_id = self._save_user_to_file(user_data)
                
                # 권한 설정
                self._set_default_permissions(user_id, role)
                
                # 활동 로그
                self._log_activity(user_id, 'user_registered', {'email': email})
                
                return True, "회원가입이 완료되었습니다.", user_id
                
            except Exception as e:
                logger.error(f"Registration error: {str(e)}")
                return False, "회원가입 중 오류가 발생했습니다.", None
    
    def login(self, email: str, password: str, ip_address: Optional[str] = None) -> Tuple[bool, str, Optional[Dict]]:
        """
        로그인
        
        Args:
            email: 이메일
            password: 비밀번호
            ip_address: 클라이언트 IP (선택)
            
        Returns:
            (성공 여부, 메시지, 사용자 정보)
        """
        with self._lock:
            try:
                # IP 화이트리스트 체크 (설정된 경우)
                if self._ip_whitelist and ip_address and ip_address not in self._ip_whitelist:
                    self._log_activity(None, 'login_blocked_ip', {'email': email, 'ip': ip_address})
                    return False, "접근이 차단된 IP입니다.", None
                
                # 계정 잠금 확인
                if self._is_account_locked(email):
                    return False, "계정이 일시적으로 잠겼습니다. 나중에 다시 시도하세요.", None
                
                # 사용자 조회
                user = self._get_user_by_email(email)
                if not user:
                    self._record_failed_login(email)
                    return False, "이메일 또는 비밀번호가 올바르지 않습니다.", None
                
                # 비밀번호 확인
                if not bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
                    self._record_failed_login(email)
                    return False, "이메일 또는 비밀번호가 올바르지 않습니다.", None
                
                # 계정 활성 상태 확인
                if not user.get('is_active', True):
                    return False, "비활성화된 계정입니다.", None
                
                # 로그인 성공
                self._clear_failed_attempts(email)
                
                # 세션 생성
                session_token = self._create_session(user)
                
                # 사용자 정보 준비
                user_info = {
                    'id': user['id'],
                    'email': user['email'],
                    'name': user['name'],
                    'role': user['role'],
                    'organization': user.get('organization'),
                    'permissions': self._get_user_permissions(user['role']),
                    'session_token': session_token,
                    'requires_2fa': user.get('two_factor_enabled', False)
                }
                
                # 2FA 체크
                if user.get('two_factor_enabled', False):
                    user_info['pending_2fa'] = True
                    return True, "2단계 인증이 필요합니다.", user_info
                
                # 마지막 로그인 시간 업데이트
                self._update_last_login(user['id'])
                
                # 활동 로그
                self._log_activity(user['id'], 'user_login', {'ip': ip_address})
                
                return True, "로그인 성공!", user_info
                
            except Exception as e:
                logger.error(f"Login error: {str(e)}")
                return False, "로그인 중 오류가 발생했습니다.", None

    def social_login(self, provider: str, email: str, name: str,
                    profile_picture: str = None, oauth_id: str = None) -> Tuple[bool, str, Optional[Dict]]:
        """소셜 로그인 처리"""
        try:
            # 기존 사용자 확인
            user = self._get_user_by_email(email)
            
            if user:
                # 기존 사용자 - 소셜 계정 연결
                if provider not in user.get('social_accounts', {}):
                    self._link_social_account(user['id'], provider, oauth_id)
                
                # 로그인 처리
                return self._create_session_for_user(user)
            else:
                # 신규 사용자 - 자동 회원가입
                user_id = str(uuid.uuid4())
                
                # 랜덤 비밀번호 생성 (소셜 로그인 사용자용)
                temp_password = secrets.token_urlsafe(32)
                password_hash = bcrypt.hashpw(temp_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                
                # 사용자 생성
                if self.db_manager:
                    conn = self.db_manager.get_connection()
                    cursor = conn.cursor()
                    
                    cursor.execute("""
                        INSERT INTO users (id, email, password_hash, name,
                                         organization, role, profile_picture,
                                         auth_provider, oauth_id, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (user_id, email, password_hash, name, None, 'user',
                         profile_picture, provider, oauth_id, datetime.now()))
                    
                    conn.commit()
                    conn.close()
                
                # 신규 사용자 세션 생성
                new_user = {
                    'id': user_id,
                    'email': email,
                    'name': name,
                    'role': 'user',
                    'profile_picture': profile_picture
                }
                
                return self._create_session_for_user(new_user)
                
        except Exception as e:
            logger.error(f"Social login error: {str(e)}")
            return (False, "소셜 로그인 중 오류가 발생했습니다.", None)
    
    def _link_social_account(self, user_id: str, provider: str, oauth_id: str):
        """소셜 계정 연결 (헬퍼 메서드)"""
        if self.db_manager:
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            
            # social_accounts 테이블이 있다면 사용, 없으면 users 테이블에 JSON으로 저장
            try:
                cursor.execute("""
                    INSERT INTO social_accounts (user_id, provider, oauth_id, linked_at)
                    VALUES (?, ?, ?, ?)
                """, (user_id, provider, oauth_id, datetime.now()))
            except:
                # 대체 방법: users 테이블의 메타데이터에 저장
                cursor.execute("""
                    UPDATE users 
                    SET auth_provider = ?, oauth_id = ?
                    WHERE id = ?
                """, (provider, oauth_id, user_id))
            
            conn.commit()
            conn.close()
    
    def _create_session_for_user(self, user: Dict) -> Tuple[bool, str, Optional[Dict]]:
        """사용자 세션 생성 (헬퍼 메서드)"""
        # 세션 토큰 생성
        session_token = secrets.token_urlsafe(32)
        
        # 사용자 정보 준비
        user_info = {
            'id': user['id'],
            'email': user['email'],
            'name': user['name'],
            'role': user.get('role', 'user'),
            'profile_picture': user.get('profile_picture'),
            'token': session_token,
            'permissions': self._get_user_permissions(user.get('role', 'user'))
        }
        
        # 마지막 로그인 시간 업데이트
        self._update_last_login(user['id'])
        
        logger.info(f"User logged in via social auth: {user['email']}")
        return (True, "로그인되었습니다.", user_info)
    
    def _get_user_permissions(self, role: str) -> list:
        """역할별 권한 반환"""
        permissions = {
            'admin': ['all'],
            'user': ['read', 'write', 'delete_own'],
            'guest': ['read']
        }
        return permissions.get(role, ['read'])
    
    def verify_2fa(self, user_id: int, totp_code: str) -> Tuple[bool, str]:
        """
        2단계 인증 검증
        
        Args:
            user_id: 사용자 ID
            totp_code: TOTP 코드
            
        Returns:
            (성공 여부, 메시지)
        """
        try:
            user = self._get_user_by_id(user_id)
            if not user or not user.get('two_factor_secret'):
                return False, "2단계 인증이 설정되지 않았습니다."
            
            # TOTP 검증
            totp = pyotp.TOTP(user['two_factor_secret'])
            if totp.verify(totp_code, valid_window=1):
                self._log_activity(user_id, '2fa_verified')
                return True, "2단계 인증 성공!"
            else:
                self._log_activity(user_id, '2fa_failed')
                return False, "잘못된 인증 코드입니다."
                
        except Exception as e:
            logger.error(f"2FA verification error: {str(e)}")
            return False, "인증 중 오류가 발생했습니다."
    
    def enable_2fa(self, user_id: int) -> Tuple[bool, str, Optional[str]]:
        """
        2단계 인증 활성화
        
        Returns:
            (성공 여부, 메시지, QR 코드 URL)
        """
        try:
            user = self._get_user_by_id(user_id)
            if not user:
                return False, "사용자를 찾을 수 없습니다.", None
            
            # TOTP 시크릿 생성
            secret = pyotp.random_base32()
            
            # QR 코드 생성
            totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
                name=user['email'],
                issuer_name='Universal DOE Platform'
            )
            
            # QR 코드 이미지 생성
            qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L)
            qr.add_data(totp_uri)
            qr.make(fit=True)
            
            img = qr.make_image(fill_color="black", back_color="white")
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            qr_data = base64.b64encode(buffer.getvalue()).decode()
            
            # 시크릿 저장 (임시)
            self._2fa_secrets[user_id] = secret
            
            return True, "QR 코드를 스캔하여 2단계 인증을 설정하세요.", f"data:image/png;base64,{qr_data}"
            
        except Exception as e:
            logger.error(f"Enable 2FA error: {str(e)}")
            return False, "2단계 인증 설정 중 오류가 발생했습니다.", None
    
    # ===========================================================================
    # 🔐 OAuth 인증
    # ===========================================================================
    
    def get_oauth_url(self, provider: str) -> Optional[str]:
        """OAuth 인증 URL 생성"""
        if not OAUTH_AVAILABLE or provider not in self.oauth_clients:
            return None
        
        try:
            client = self.oauth_clients[provider]
            authorization_url, state = client.create_authorization_url(
                OAUTH_PROVIDERS[provider]['authorize_url']
            )
            
            # 상태 저장 (CSRF 방지)
            st.session_state[f'oauth_{provider}_state'] = state
            
            return authorization_url
            
        except Exception as e:
            logger.error(f"OAuth URL generation error: {str(e)}")
            return None
    
    def handle_oauth_callback(self, provider: str, code: str, state: str) -> Tuple[bool, str, Optional[Dict]]:
        """OAuth 콜백 처리"""
        if not OAUTH_AVAILABLE or provider not in self.oauth_clients:
            return False, "OAuth가 설정되지 않았습니다.", None
        
        try:
            # CSRF 체크
            saved_state = st.session_state.get(f'oauth_{provider}_state')
            if not saved_state or saved_state != state:
                return False, "잘못된 인증 상태입니다.", None
            
            client = self.oauth_clients[provider]
            
            # 토큰 교환
            token = client.fetch_token(
                OAUTH_PROVIDERS[provider]['token_url'],
                authorization_response=code
            )
            
            # 사용자 정보 가져오기
            resp = client.get(OAUTH_PROVIDERS[provider]['userinfo_url'])
            userinfo = resp.json()
            
            # 사용자 처리
            email = userinfo.get('email')
            name = userinfo.get('name', email.split('@')[0])
            
            # 기존 사용자 확인
            user = self._get_user_by_email(email)
            
            if user:
                # 기존 사용자 로그인
                session_token = self._create_session(user)
                user_info = {
                    'id': user['id'],
                    'email': user['email'],
                    'name': user['name'],
                    'role': user['role'],
                    'permissions': self._get_user_permissions(user['role']),
                    'session_token': session_token
                }
                
                self._log_activity(user['id'], 'oauth_login', {'provider': provider})
                return True, "로그인 성공!", user_info
            else:
                # 새 사용자 생성
                success, msg, user_id = self.register_user(
                    email=email,
                    password=secrets.token_urlsafe(32),  # 랜덤 비밀번호
                    name=name,
                    role=UserRole.USER
                )
                
                if success:
                    user = self._get_user_by_id(user_id)
                    session_token = self._create_session(user)
                    user_info = {
                        'id': user_id,
                        'email': email,
                        'name': name,
                        'role': UserRole.USER,
                        'permissions': self._get_user_permissions(UserRole.USER),
                        'session_token': session_token
                    }
                    
                    self._log_activity(user_id, 'oauth_register', {'provider': provider})
                    return True, "회원가입 및 로그인 성공!", user_info
                else:
                    return False, msg, None
                    
        except Exception as e:
            logger.error(f"OAuth callback error: {str(e)}")
            return False, "인증 처리 중 오류가 발생했습니다.", None
    
    # ===========================================================================
    # 🔑 API 키 관리
    # ===========================================================================
    
    def generate_api_key(self, user_id: int, name: str, permissions: List[str]) -> Tuple[bool, str, Optional[str]]:
        """
        API 키 생성
        
        Args:
            user_id: 사용자 ID
            name: API 키 이름
            permissions: 권한 목록
            
        Returns:
            (성공 여부, 메시지, API 키)
        """
        try:
            # API 키 생성
            api_key = f"udoe_{secrets.token_urlsafe(API_KEY_LENGTH)}"
            
            # 키 정보 저장
            key_data = {
                'user_id': user_id,
                'name': name,
                'key_hash': hashlib.sha256(api_key.encode()).hexdigest(),
                'permissions': json.dumps(permissions),
                'created_at': datetime.now().isoformat(),
                'last_used': None,
                'is_active': True
            }
            
            if self.db_manager:
                key_id = self.db_manager.create_api_key(key_data)
            else:
                key_id = self._save_api_key_to_file(key_data)
            
            # 캐시에 저장
            self._api_keys[api_key] = {
                'id': key_id,
                'user_id': user_id,
                'permissions': permissions
            }
            
            self._log_activity(user_id, 'api_key_created', {'name': name})
            
            return True, "API 키가 생성되었습니다.", api_key
            
        except Exception as e:
            logger.error(f"API key generation error: {str(e)}")
            return False, "API 키 생성 중 오류가 발생했습니다.", None
    
    def verify_api_key(self, api_key: str) -> Tuple[bool, Optional[Dict]]:
        """
        API 키 검증
        
        Returns:
            (유효 여부, 사용자 정보)
        """
        try:
            # 캐시 확인
            if api_key in self._api_keys:
                key_info = self._api_keys[api_key]
                user = self._get_user_by_id(key_info['user_id'])
                if user:
                    return True, {
                        'id': user['id'],
                        'email': user['email'],
                        'name': user['name'],
                        'permissions': key_info['permissions']
                    }
            
            # DB 확인
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            key_data = self._get_api_key_by_hash(key_hash)
            
            if key_data and key_data.get('is_active'):
                user = self._get_user_by_id(key_data['user_id'])
                if user:
                    # 캐시 업데이트
                    self._api_keys[api_key] = {
                        'id': key_data['id'],
                        'user_id': key_data['user_id'],
                        'permissions': json.loads(key_data['permissions'])
                    }
                    
                    # 마지막 사용 시간 업데이트
                    self._update_api_key_last_used(key_data['id'])
                    
                    return True, {
                        'id': user['id'],
                        'email': user['email'],
                        'name': user['name'],
                        'permissions': json.loads(key_data['permissions'])
                    }
            
            return False, None
            
        except Exception as e:
            logger.error(f"API key verification error: {str(e)}")
            return False, None
    
    # ===========================================================================
    # 🔐 세션 관리
    # ===========================================================================
    
    def _create_session(self, user: Dict) -> str:
        """세션 생성"""
        # JWT 페이로드
        payload = {
            'user_id': user['id'],
            'email': user['email'],
            'role': user['role'],
            'exp': datetime.utcnow() + timedelta(hours=TOKEN_EXPIRY_HOURS),
            'iat': datetime.utcnow(),
            'jti': secrets.token_urlsafe(16)  # JWT ID
        }
        
        # 토큰 생성
        token = jwt.encode(payload, self.jwt_secret, algorithm=JWT_ALGORITHM)
        
        # 세션 저장
        self._sessions[token] = {
            'user_id': user['id'],
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'ip_address': None  # 필요시 설정
        }
        
        return token
    
    def verify_session(self, token: str) -> Tuple[bool, Optional[Dict]]:
        """세션 검증"""
        try:
            # JWT 디코드
            payload = jwt.decode(token, self.jwt_secret, algorithms=[JWT_ALGORITHM])
            
            # 만료 체크는 jwt.decode에서 자동으로 수행됨
            
            # 세션 존재 확인
            if token not in self._sessions:
                return False, None
            
            # 사용자 정보 반환
            user = self._get_user_by_id(payload['user_id'])
            if user:
                # 마지막 활동 시간 업데이트
                self._sessions[token]['last_activity'] = datetime.now()
                
                return True, {
                    'id': user['id'],
                    'email': user['email'],
                    'name': user['name'],
                    'role': user['role'],
                    'permissions': self._get_user_permissions(user['role'])
                }
            
            return False, None
            
        except jwt.ExpiredSignatureError:
            # 토큰 만료
            if token in self._sessions:
                del self._sessions[token]
            return False, None
        except jwt.InvalidTokenError:
            return False, None
        except Exception as e:
            logger.error(f"Session verification error: {str(e)}")
            return False, None
    
    def logout(self, token: str):
        """로그아웃"""
        if token in self._sessions:
            user_id = self._sessions[token]['user_id']
            del self._sessions[token]
            self._log_activity(user_id, 'user_logout')
    
    def _cleanup_expired_sessions(self):
        """만료된 세션 정리"""
        current_time = datetime.now()
        expired_tokens = []
        
        for token, session in self._sessions.items():
            # 세션 타임아웃 체크
            if current_time - session['last_activity'] > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
                expired_tokens.append(token)
        
        # 만료된 세션 제거
        for token in expired_tokens:
            del self._sessions[token]
    
    # ===========================================================================
    # 🔐 권한 관리
    # ===========================================================================
    
    def _get_user_permissions(self, role: str) -> Dict[str, List[str]]:
        """역할별 권한 조회"""
        permissions = {}
        
        for resource, actions in PERMISSION_MATRIX.items():
            permissions[resource] = []
            for action, allowed_roles in actions.items():
                if role in allowed_roles:
                    permissions[resource].append(action)
        
        return permissions
    
    def check_permission(self, user_id: int, resource: str, action: str) -> bool:
        """권한 확인"""
        user = self._get_user_by_id(user_id)
        if not user:
            return False
        
        allowed_roles = PERMISSION_MATRIX.get(resource, {}).get(action, [])
        return user['role'] in allowed_roles
    
    @lru_cache(maxsize=1000)
    def check_permission_cached(self, user_id: int, resource: str, action: str) -> bool:
        """권한 확인 (캐시)"""
        return self.check_permission(user_id, resource, action)
    
    # ===========================================================================
    # 🔐 계정 잠금 관리
    # ===========================================================================
    
    def _record_failed_login(self, email: str):
        """로그인 실패 기록"""
        if email not in self._failed_attempts:
            self._failed_attempts[email] = {
                'count': 0,
                'first_attempt': datetime.now(),
                'locked_until': None
            }
        
        self._failed_attempts[email]['count'] += 1
        
        # 최대 시도 횟수 초과 시 잠금
        if self._failed_attempts[email]['count'] >= MAX_LOGIN_ATTEMPTS:
            self._failed_attempts[email]['locked_until'] = datetime.now() + timedelta(minutes=LOCKOUT_DURATION_MINUTES)
            self._log_activity(None, 'account_locked', {'email': email})
    
    def _is_account_locked(self, email: str) -> bool:
        """계정 잠금 상태 확인"""
        if email not in self._failed_attempts:
            return False
        
        locked_until = self._failed_attempts[email].get('locked_until')
        if locked_until and datetime.now() < locked_until:
            return True
        
        return False
    
    def _clear_failed_attempts(self, email: str):
        """로그인 실패 기록 초기화"""
        if email in self._failed_attempts:
            del self._failed_attempts[email]
    
    def _cleanup_locked_accounts(self):
        """잠금 해제된 계정 정리"""
        current_time = datetime.now()
        emails_to_clear = []
        
        for email, attempts in self._failed_attempts.items():
            locked_until = attempts.get('locked_until')
            if locked_until and current_time > locked_until:
                emails_to_clear.append(email)
        
        for email in emails_to_clear:
            del self._failed_attempts[email]
    
    # ===========================================================================
    # 🔐 IP 화이트리스트 관리
    # ===========================================================================
    
    def add_ip_to_whitelist(self, ip_address: str, added_by: int) -> bool:
        """IP 화이트리스트 추가"""
        try:
            # IP 유효성 검사
            ipaddress.ip_address(ip_address)
            
            self._ip_whitelist.add(ip_address)
            self._save_ip_whitelist()
            
            self._log_activity(added_by, 'ip_whitelist_add', {'ip': ip_address})
            return True
            
        except ValueError:
            return False
    
    def remove_ip_from_whitelist(self, ip_address: str, removed_by: int) -> bool:
        """IP 화이트리스트 제거"""
        if ip_address in self._ip_whitelist:
            self._ip_whitelist.remove(ip_address)
            self._save_ip_whitelist()
            
            self._log_activity(removed_by, 'ip_whitelist_remove', {'ip': ip_address})
            return True
        
        return False
    
    def _save_ip_whitelist(self):
        """IP 화이트리스트 저장"""
        whitelist_file = LOCAL_CONFIG['app_data_dir'] / 'security' / 'ip_whitelist.json'
        whitelist_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(whitelist_file, 'w') as f:
            json.dump({'whitelist': list(self._ip_whitelist)}, f)
    
    # ===========================================================================
    # 🔐 사용자 관리
    # ===========================================================================
    
    def update_user_profile(self, user_id: int, updates: Dict) -> Tuple[bool, str]:
        """사용자 프로필 업데이트"""
        try:
            allowed_fields = ['name', 'organization', 'settings']
            filtered_updates = {k: v for k, v in updates.items() if k in allowed_fields}
            
            if self.db_manager:
                success = self.db_manager.update_user(user_id, filtered_updates)
            else:
                success = self._update_user_in_file(user_id, filtered_updates)
            
            if success:
                self._log_activity(user_id, 'profile_updated', filtered_updates)
                return True, "프로필이 업데이트되었습니다."
            else:
                return False, "프로필 업데이트 실패"
                
        except Exception as e:
            logger.error(f"Profile update error: {str(e)}")
            return False, "프로필 업데이트 중 오류가 발생했습니다."
    
    def change_password(self, user_id: int, old_password: str, new_password: str) -> Tuple[bool, str]:
        """비밀번호 변경"""
        try:
            user = self._get_user_by_id(user_id)
            if not user:
                return False, "사용자를 찾을 수 없습니다."
            
            # 현재 비밀번호 확인
            if not bcrypt.checkpw(old_password.encode('utf-8'), user['password_hash'].encode('utf-8')):
                return False, "현재 비밀번호가 올바르지 않습니다."
            
            # 새 비밀번호 해시화
            new_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt(BCRYPT_ROUNDS))
            
            # 업데이트
            if self.db_manager:
                success = self.db_manager.update_user(user_id, {'password_hash': new_hash.decode('utf-8')})
            else:
                success = self._update_user_in_file(user_id, {'password_hash': new_hash.decode('utf-8')})
            
            if success:
                self._log_activity(user_id, 'password_changed')
                return True, "비밀번호가 변경되었습니다."
            else:
                return False, "비밀번호 변경 실패"
                
        except Exception as e:
            logger.error(f"Password change error: {str(e)}")
            return False, "비밀번호 변경 중 오류가 발생했습니다."
    
    def reset_password(self, email: str, new_password: str) -> Tuple[bool, str]:
        """비밀번호 재설정"""
        try:
            user = self._get_user_by_email(email)
            if not user:
                return False, "등록되지 않은 이메일입니다."
            
            # 새 비밀번호 해시화
            new_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt(BCRYPT_ROUNDS))
            
            # 업데이트
            if self.db_manager:
                success = self.db_manager.update_user(user['id'], {'password_hash': new_hash.decode('utf-8')})
            else:
                success = self._update_user_in_file(user['id'], {'password_hash': new_hash.decode('utf-8')})
            
            if success:
                self._log_activity(user['id'], 'password_reset')
                return True, "비밀번호가 재설정되었습니다."
            else:
                return False, "비밀번호 재설정 실패"
                
        except Exception as e:
            logger.error(f"Password reset error: {str(e)}")
            return False, "비밀번호 재설정 중 오류가 발생했습니다."
    
    # ===========================================================================
    # 🔐 활동 로그
    # ===========================================================================
    
    def _log_activity(self, user_id: Optional[int], action: str, details: Optional[Dict] = None):
        """활동 로그 기록"""
        try:
            log_entry = {
                'user_id': user_id,
                'action': action,
                'details': json.dumps(details) if details else None,
                'timestamp': datetime.now().isoformat(),
                'ip_address': None  # 필요시 추가
            }
            
            if self.db_manager:
                self.db_manager.create_activity_log(log_entry)
            else:
                self._save_activity_log_to_file(log_entry)
                
        except Exception as e:
            logger.error(f"Activity logging error: {str(e)}")
    
    def get_user_activities(self, user_id: int, limit: int = 50) -> List[Dict]:
        """사용자 활동 로그 조회"""
        try:
            if self.db_manager:
                return self.db_manager.get_user_activities(user_id, limit)
            else:
                return self._get_activities_from_file(user_id, limit)
        except Exception:
            return []
    
    # ===========================================================================
    # 🔐 헬퍼 메서드 (파일 기반 폴백)
    # ===========================================================================
    
    def _get_users_file(self) -> Path:
        """사용자 파일 경로"""
        users_file = LOCAL_CONFIG['app_data_dir'] / 'auth' / 'users.json'
        users_file.parent.mkdir(parents=True, exist_ok=True)
        if not users_file.exists():
            users_file.write_text('[]')
        return users_file
    
    def _check_email_exists(self, email: str) -> bool:
        """이메일 중복 확인"""
        if self.db_manager:
            return self.db_manager.get_user_by_email(email) is not None
        else:
            users = json.loads(self._get_users_file().read_text())
            return any(u['email'] == email for u in users)
    
    def _get_user_by_email(self, email: str) -> Optional[Dict]:
        """이메일로 사용자 조회"""
        if self.db_manager:
            return self.db_manager.get_user_by_email(email)
        else:
            users = json.loads(self._get_users_file().read_text())
            for user in users:
                if user['email'] == email:
                    return user
            return None
    
    def _get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """ID로 사용자 조회"""
        if self.db_manager:
            return self.db_manager.get_user_by_id(user_id)
        else:
            users = json.loads(self._get_users_file().read_text())
            for user in users:
                if user['id'] == user_id:
                    return user
            return None
    
    def _save_user_to_file(self, user_data: Dict) -> int:
        """파일에 사용자 저장"""
        users_file = self._get_users_file()
        users = json.loads(users_file.read_text())
        
        # ID 생성
        user_data['id'] = max([u.get('id', 0) for u in users], default=0) + 1
        
        users.append(user_data)
        users_file.write_text(json.dumps(users, indent=2, ensure_ascii=False))
        
        return user_data['id']
    
    def _update_user_in_file(self, user_id: int, updates: Dict) -> bool:
        """파일에서 사용자 업데이트"""
        users_file = self._get_users_file()
        users = json.loads(users_file.read_text())
        
        for user in users:
            if user['id'] == user_id:
                user.update(updates)
                users_file.write_text(json.dumps(users, indent=2, ensure_ascii=False))
                return True
        
        return False
    
    def _update_last_login(self, user_id: int):
        """마지막 로그인 시간 업데이트"""
        self._update_user_in_file(user_id, {'last_login': datetime.now().isoformat()})
    
    def _set_default_permissions(self, user_id: int, role: str):
        """기본 권한 설정"""
        # 파일 기반 시스템에서는 역할로 권한 결정
        pass
    
    def _save_activity_log_to_file(self, log_entry: Dict):
        """활동 로그 파일 저장"""
        log_file = LOCAL_CONFIG['app_data_dir'] / 'auth' / 'activity_log.jsonl'
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def _get_activities_from_file(self, user_id: int, limit: int) -> List[Dict]:
        """파일에서 활동 로그 조회"""
        log_file = LOCAL_CONFIG['app_data_dir'] / 'auth' / 'activity_log.jsonl'
        if not log_file.exists():
            return []
        
        activities = []
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    log = json.loads(line.strip())
                    if log.get('user_id') == user_id:
                        activities.append(log)
                except:
                    continue
        
        # 최신순 정렬
        activities.sort(key=lambda x: x['timestamp'], reverse=True)
        return activities[:limit]
    
    def _get_api_key_by_hash(self, key_hash: str) -> Optional[Dict]:
        """API 키 해시로 조회"""
        if self.db_manager:
            return self.db_manager.get_api_key_by_hash(key_hash)
        else:
            # 파일 기반 구현
            keys_file = LOCAL_CONFIG['app_data_dir'] / 'auth' / 'api_keys.json'
            if keys_file.exists():
                keys = json.loads(keys_file.read_text())
                for key in keys:
                    if key['key_hash'] == key_hash:
                        return key
            return None
    
    def _save_api_key_to_file(self, key_data: Dict) -> int:
        """API 키 파일 저장"""
        keys_file = LOCAL_CONFIG['app_data_dir'] / 'auth' / 'api_keys.json'
        keys_file.parent.mkdir(parents=True, exist_ok=True)
        
        if keys_file.exists():
            keys = json.loads(keys_file.read_text())
        else:
            keys = []
        
        # ID 생성
        key_data['id'] = max([k.get('id', 0) for k in keys], default=0) + 1
        
        keys.append(key_data)
        keys_file.write_text(json.dumps(keys, indent=2, ensure_ascii=False))
        
        return key_data['id']
    
    def _update_api_key_last_used(self, key_id: int):
        """API 키 사용 시간 업데이트"""
        if self.db_manager:
            self.db_manager.update_api_key(key_id, {'last_used': datetime.now().isoformat()})
        else:
            # 파일 기반 구현
            keys_file = LOCAL_CONFIG['app_data_dir'] / 'auth' / 'api_keys.json'
            if keys_file.exists():
                keys = json.loads(keys_file.read_text())
                for key in keys:
                    if key['id'] == key_id:
                        key['last_used'] = datetime.now().isoformat()
                        keys_file.write_text(json.dumps(keys, indent=2, ensure_ascii=False))
                        break


# ===========================================================================
# 🔧 헬퍼 함수
# ===========================================================================

def get_current_user() -> Optional[Dict]:
    """현재 로그인한 사용자 정보 반환"""
    if 'user' in st.session_state and st.session_state.user:
        return st.session_state.user
    return None

def is_authenticated() -> bool:
    """인증 상태 확인"""
    return get_current_user() is not None

def has_permission(resource: str, action: str) -> bool:
    """현재 사용자의 권한 확인"""
    user = get_current_user()
    if not user:
        return False
    
    permissions = user.get('permissions', {})
    return action in permissions.get(resource, [])

def require_login():
    """로그인 필요 페이지 보호"""
    if not is_authenticated():
        st.error("🔐 이 페이지를 보려면 로그인이 필요합니다.")
        st.stop()

def require_role(min_role: str):
    """최소 역할 요구"""
    user = get_current_user()
    if not user:
        st.error("🔐 로그인이 필요합니다.")
        st.stop()
    
    user_role = user.get('role', UserRole.GUEST)
    if ROLE_HIERARCHY.get(user_role, 0) < ROLE_HIERARCHY.get(min_role, 0):
        st.error(f"⛔ 이 기능을 사용하려면 {min_role} 이상의 권한이 필요합니다.")
        st.stop()

def check_authentication() -> bool:
    """인증 체크 (세션 타임아웃 포함)"""
    if 'auth_manager' not in st.session_state:
        st.session_state.auth_manager = AuthManager()
    
    # 세션 토큰 확인
    if 'session_token' in st.session_state:
        valid, user_info = st.session_state.auth_manager.verify_session(
            st.session_state.session_token
        )
        
        if valid:
            st.session_state.user = user_info
            return True
        else:
            # 세션 만료
            if 'user' in st.session_state:
                del st.session_state.user
            if 'session_token' in st.session_state:
                del st.session_state.session_token
    
    return False


# ===========================================================================
# 🧪 테스트 코드
# ===========================================================================

if __name__ == "__main__":
    # 테스트용 코드
    print("Enhanced AuthManager 모듈 로드 완료")
    
    # 간단한 테스트
    auth = AuthManager()
    
    # 회원가입 테스트
    success, msg, user_id = auth.register_user(
        email="test@example.com",
        password="Test1234!",
        name="테스트 사용자",
        organization="테스트 기관"
    )
    print(f"회원가입: {success}, {msg}")
    
    # 로그인 테스트
    if success:
        success, msg, user_info = auth.login(
            email="test@example.com",
            password="Test1234!"
        )
        print(f"로그인: {success}, {msg}")
        
        if user_info:
            print(f"사용자 정보: {user_info['name']} ({user_info['role']})")
            
            # API 키 생성 테스트
            success, msg, api_key = auth.generate_api_key(
                user_id=user_info['id'],
                name="테스트 API 키",
                permissions=['read', 'write']
            )
            print(f"API 키 생성: {success}, {msg}")
            
            if api_key:
                # API 키 검증 테스트
                valid, key_user = auth.verify_api_key(api_key)
                print(f"API 키 검증: {valid}")
