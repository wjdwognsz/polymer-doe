"""
🔐 Authentication Manager - 로컬 인증 시스템
===========================================================================
데스크톱 애플리케이션을 위한 오프라인 우선 인증 관리자
SQLite 기반 로컬 인증, JWT 세션 관리, 선택적 클라우드 동기화 지원
===========================================================================
"""

import os
import sys
import json
import sqlite3
import logging
import secrets
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from functools import wraps
from contextlib import contextmanager
import hashlib
import base64

# 보안 라이브러리
import bcrypt
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

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
    }
}

# ===========================================================================
# 🔐 인증 관리자 클래스
# ===========================================================================

class AuthManager:
    """로컬 인증 관리자"""
    
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
        
        # 암호화 키 설정
        self._setup_encryption()
        
        # JWT 시크릿 키
        self.jwt_secret = self._get_or_create_jwt_secret()
        
        # 세션 정리 스케줄러 시작
        self._start_session_cleanup()
        
        logger.info("AuthManager initialized")
    
    def _setup_encryption(self):
        """암호화 설정"""
        # 마스터 키 생성 또는 로드
        key_file = LOCAL_CONFIG['app_data_dir'] / '.keys' / 'master.key'
        key_file.parent.mkdir(parents=True, exist_ok=True)
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                self.master_key = f.read()
        else:
            self.master_key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(self.master_key)
            
            # 파일 권한 설정 (읽기 전용)
            if sys.platform != 'win32':
                os.chmod(key_file, 0o600)
        
        self.cipher = Fernet(self.master_key)
    
    def _get_or_create_jwt_secret(self) -> str:
        """JWT 시크릿 키 생성 또는 로드"""
        secret_file = LOCAL_CONFIG['app_data_dir'] / '.keys' / 'jwt_secret.key'
        
        if secret_file.exists():
            with open(secret_file, 'r') as f:
                return f.read()
        else:
            secret = secrets.token_urlsafe(64)
            secret_file.parent.mkdir(parents=True, exist_ok=True)
            with open(secret_file, 'w') as f:
                f.write(secret)
            return secret
    
    # =========================================================================
    # 🔑 사용자 등록
    # =========================================================================
    
    def register_user(self, 
                     email: str, 
                     password: str,
                     name: str,
                     organization: Optional[str] = None,
                     **kwargs) -> Tuple[bool, str, Optional[str]]:
        """
        새 사용자 등록
        
        Args:
            email: 이메일 주소
            password: 비밀번호
            name: 사용자 이름
            organization: 소속 기관
            **kwargs: 추가 정보
            
        Returns:
            (성공여부, 메시지, user_id)
        """
        try:
            # 이메일 중복 확인
            if self._check_email_exists(email):
                return (False, "이미 등록된 이메일입니다.", None)
            
            # 비밀번호 검증
            is_valid, msg = self._validate_password(password)
            if not is_valid:
                return (False, msg, None)
            
            # 비밀번호 해싱
            password_hash = self._hash_password(password)
            
            # 사용자 ID 생성
            user_id = self._generate_user_id()
            
            # 사용자 데이터
            user_data = {
                'id': user_id,
                'email': email,
                'password_hash': password_hash,
                'name': name,
                'organization': organization,
                'role': UserRole.USER,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'last_login': None,
                'is_active': 1,
                'settings': json.dumps({
                    'theme': 'light',
                    'language': 'ko',
                    'notifications': True
                })
            }
            
            # DB에 저장
            if self.db_manager:
                conn = self.db_manager._get_connection()
                try:
                    conn.execute("""
                        INSERT INTO users (id, email, password_hash, name, 
                                         organization, role, created_at, updated_at,
                                         last_login, is_active, settings)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        user_data['id'], user_data['email'], 
                        user_data['password_hash'], user_data['name'],
                        user_data['organization'], user_data['role'],
                        user_data['created_at'], user_data['updated_at'],
                        user_data['last_login'], user_data['is_active'],
                        user_data['settings']
                    ))
                    conn.commit()
                    
                    logger.info(f"User registered: {email}")
                    return (True, "회원가입이 완료되었습니다.", user_id)
                    
                finally:
                    conn.close()
            else:
                # DB 매니저가 없는 경우 (테스트용)
                return (True, "회원가입이 완료되었습니다. (테스트 모드)", user_id)
                
        except Exception as e:
            logger.error(f"Registration error: {str(e)}")
            return (False, "회원가입 중 오류가 발생했습니다.", None)
    
    # =========================================================================
    # 🔓 로그인/로그아웃
    # =========================================================================
    
    def login(self, 
              email: str, 
              password: str,
              remember_me: bool = False) -> Tuple[bool, str, Optional[Dict]]:
        """
        사용자 로그인
        
        Args:
            email: 이메일
            password: 비밀번호
            remember_me: 로그인 유지 여부
            
        Returns:
            (성공여부, 메시지, 사용자정보)
        """
        try:
            # 계정 잠금 확인
            if self._is_account_locked(email):
                return (False, "너무 많은 로그인 시도로 계정이 일시적으로 잠겼습니다. 15분 후 다시 시도해주세요.", None)
            
            # 사용자 조회
            user = self._get_user_by_email(email)
            if not user:
                self._record_failed_attempt(email)
                return (False, "이메일 또는 비밀번호가 올바르지 않습니다.", None)
            
            # 비밀번호 확인
            if not self._verify_password(password, user['password_hash']):
                self._record_failed_attempt(email)
                return (False, "이메일 또는 비밀번호가 올바르지 않습니다.", None)
            
            # 계정 활성화 확인
            if not user.get('is_active', True):
                return (False, "계정이 비활성화되었습니다.", None)
            
            # 세션 생성
            session_data = self._create_session(user, remember_me)
            
            # 로그인 성공 처리
            self._clear_failed_attempts(email)
            self._update_last_login(user['id'])
            
            # 사용자 정보 반환
            user_info = {
                'id': user['id'],
                'email': user['email'],
                'name': user['name'],
                'organization': user['organization'],
                'role': user['role'],
                'token': session_data['access_token'],
                'refresh_token': session_data['refresh_token'] if remember_me else None,
                'permissions': self._get_user_permissions(user['role'])
            }
            
            logger.info(f"User logged in: {email}")
            return (True, "로그인되었습니다.", user_info)
            
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            return (False, "로그인 중 오류가 발생했습니다.", None)
    
    def logout(self, token: str) -> bool:
        """
        로그아웃
        
        Args:
            token: 액세스 토큰
            
        Returns:
            성공 여부
        """
        try:
            # 토큰에서 사용자 ID 추출
            payload = self._decode_token(token)
            if payload:
                user_id = payload.get('user_id')
                
                # 세션 제거
                with self._lock:
                    if user_id in self._sessions:
                        del self._sessions[user_id]
                
                logger.info(f"User logged out: {user_id}")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Logout error: {str(e)}")
            return False
    
    # =========================================================================
    # 🎫 토큰 관리
    # =========================================================================
    
    def verify_token(self, token: str) -> Optional[Dict]:
        """
        토큰 검증
        
        Args:
            token: JWT 토큰
            
        Returns:
            유효한 경우 사용자 정보, 아니면 None
        """
        try:
            payload = self._decode_token(token)
            if not payload:
                return None
            
            # 만료 시간 확인
            exp = payload.get('exp')
            if exp and datetime.fromtimestamp(exp) < datetime.now():
                return None
            
            # 사용자 조회
            user_id = payload.get('user_id')
            user = self._get_user_by_id(user_id)
            
            if user and user.get('is_active', True):
                return {
                    'id': user['id'],
                    'email': user['email'],
                    'name': user['name'],
                    'role': user['role'],
                    'permissions': self._get_user_permissions(user['role'])
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Token verification error: {str(e)}")
            return None
    
    def refresh_token(self, refresh_token: str) -> Optional[str]:
        """
        리프레시 토큰으로 새 액세스 토큰 발급
        
        Args:
            refresh_token: 리프레시 토큰
            
        Returns:
            새 액세스 토큰 또는 None
        """
        try:
            payload = self._decode_token(refresh_token)
            if not payload or payload.get('type') != 'refresh':
                return None
            
            # 사용자 조회
            user_id = payload.get('user_id')
            user = self._get_user_by_id(user_id)
            
            if user and user.get('is_active', True):
                # 새 액세스 토큰 생성
                new_token = self._create_token(user, token_type='access')
                return new_token
            
            return None
            
        except Exception as e:
            logger.error(f"Token refresh error: {str(e)}")
            return None
    
    # =========================================================================
    # 🔐 권한 관리
    # =========================================================================
    
    def check_permission(self, 
                        user_role: str,
                        resource: str,
                        action: str) -> bool:
        """
        권한 확인
        
        Args:
            user_role: 사용자 역할
            resource: 리소스 타입
            action: 액션
            
        Returns:
            권한 여부
        """
        if resource in PERMISSION_MATRIX:
            if action in PERMISSION_MATRIX[resource]:
                allowed_roles = PERMISSION_MATRIX[resource][action]
                return user_role in allowed_roles
        
        return False
    
    def require_auth(self, min_role: str = UserRole.USER):
        """
        인증 필요 데코레이터
        
        Args:
            min_role: 최소 필요 역할
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Streamlit 세션에서 사용자 정보 확인
                if 'user' not in st.session_state or not st.session_state.user:
                    st.error("로그인이 필요합니다.")
                    st.stop()
                
                user = st.session_state.user
                user_role = user.get('role', UserRole.GUEST)
                
                # 역할 계층 확인
                if ROLE_HIERARCHY.get(user_role, 0) < ROLE_HIERARCHY.get(min_role, 0):
                    st.error(f"이 기능을 사용하려면 {min_role} 이상의 권한이 필요합니다.")
                    st.stop()
                
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    # =========================================================================
    # 🔑 API 키 관리
    # =========================================================================
    
    def save_api_key(self, 
                     user_id: str,
                     service: str,
                     api_key: str) -> bool:
        """
        API 키 암호화 저장
        
        Args:
            user_id: 사용자 ID
            service: 서비스명 (google_gemini, groq 등)
            api_key: API 키
            
        Returns:
            성공 여부
        """
        try:
            # API 키 암호화
            encrypted_key = self.cipher.encrypt(api_key.encode())
            
            if self.db_manager:
                conn = self.db_manager._get_connection()
                try:
                    # 기존 키 확인
                    existing = conn.execute("""
                        SELECT id FROM api_keys 
                        WHERE user_id = ? AND service = ?
                    """, (user_id, service)).fetchone()
                    
                    if existing:
                        # 업데이트
                        conn.execute("""
                            UPDATE api_keys 
                            SET encrypted_key = ?, updated_at = ?
                            WHERE user_id = ? AND service = ?
                        """, (encrypted_key, datetime.now().isoformat(), 
                              user_id, service))
                    else:
                        # 새로 저장
                        conn.execute("""
                            INSERT INTO api_keys (user_id, service, encrypted_key, created_at)
                            VALUES (?, ?, ?, ?)
                        """, (user_id, service, encrypted_key, 
                              datetime.now().isoformat()))
                    
                    conn.commit()
                    logger.info(f"API key saved for user {user_id}, service {service}")
                    return True
                    
                finally:
                    conn.close()
            
            return False
            
        except Exception as e:
            logger.error(f"Save API key error: {str(e)}")
            return False
    
    def get_api_key(self, 
                    user_id: str,
                    service: str) -> Optional[str]:
        """
        API 키 복호화 조회
        
        Args:
            user_id: 사용자 ID
            service: 서비스명
            
        Returns:
            복호화된 API 키 또는 None
        """
        try:
            if self.db_manager:
                conn = self.db_manager._get_connection()
                try:
                    result = conn.execute("""
                        SELECT encrypted_key FROM api_keys
                        WHERE user_id = ? AND service = ?
                    """, (user_id, service)).fetchone()
                    
                    if result:
                        encrypted_key = result['encrypted_key']
                        # 복호화
                        api_key = self.cipher.decrypt(encrypted_key).decode()
                        return api_key
                        
                finally:
                    conn.close()
            
            return None
            
        except Exception as e:
            logger.error(f"Get API key error: {str(e)}")
            return None
    
    # =========================================================================
    # 🛠️ 유틸리티 메서드
    # =========================================================================
    
    def _hash_password(self, password: str) -> str:
        """비밀번호 해싱"""
        salt = bcrypt.gensalt(rounds=BCRYPT_ROUNDS)
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def _verify_password(self, password: str, hashed: str) -> bool:
        """비밀번호 검증"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def _validate_password(self, password: str) -> Tuple[bool, str]:
        """비밀번호 강도 검증"""
        if len(password) < SECURITY_CONFIG['password_min_length']:
            return (False, f"비밀번호는 최소 {SECURITY_CONFIG['password_min_length']}자 이상이어야 합니다.")
        
        if SECURITY_CONFIG['password_require_uppercase'] and not any(c.isupper() for c in password):
            return (False, "비밀번호에 대문자가 포함되어야 합니다.")
        
        if SECURITY_CONFIG['password_require_number'] and not any(c.isdigit() for c in password):
            return (False, "비밀번호에 숫자가 포함되어야 합니다.")
        
        if SECURITY_CONFIG['password_require_special'] and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            return (False, "비밀번호에 특수문자가 포함되어야 합니다.")
        
        return (True, "")
    
    def _generate_user_id(self) -> str:
        """사용자 ID 생성"""
        return f"user_{secrets.token_urlsafe(16)}"
    
    def _create_token(self, user: Dict, token_type: str = 'access') -> str:
        """JWT 토큰 생성"""
        now = datetime.now()
        
        if token_type == 'access':
            exp = now + timedelta(hours=TOKEN_EXPIRY_HOURS)
        else:  # refresh token
            exp = now + timedelta(days=REFRESH_TOKEN_DAYS)
        
        payload = {
            'user_id': user['id'],
            'email': user['email'],
            'role': user['role'],
            'type': token_type,
            'iat': now.timestamp(),
            'exp': exp.timestamp()
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm=JWT_ALGORITHM)
    
    def _decode_token(self, token: str) -> Optional[Dict]:
        """JWT 토큰 디코드"""
        try:
            return jwt.decode(token, self.jwt_secret, algorithms=[JWT_ALGORITHM])
        except jwt.InvalidTokenError:
            return None
    
    def _create_session(self, user: Dict, remember_me: bool) -> Dict:
        """세션 생성"""
        access_token = self._create_token(user, 'access')
        refresh_token = self._create_token(user, 'refresh') if remember_me else None
        
        session_data = {
            'user_id': user['id'],
            'access_token': access_token,
            'refresh_token': refresh_token,
            'created_at': datetime.now(),
            'last_activity': datetime.now()
        }
        
        # 세션 저장
        with self._lock:
            self._sessions[user['id']] = session_data
        
        return session_data
    
    def _check_email_exists(self, email: str) -> bool:
        """이메일 중복 확인"""
        if self.db_manager:
            conn = self.db_manager._get_connection()
            try:
                result = conn.execute(
                    "SELECT COUNT(*) as count FROM users WHERE email = ?",
                    (email,)
                ).fetchone()
                return result['count'] > 0
            finally:
                conn.close()
        return False
    
    def _get_user_by_email(self, email: str) -> Optional[Dict]:
        """이메일로 사용자 조회"""
        if self.db_manager:
            conn = self.db_manager._get_connection()
            try:
                return conn.execute(
                    "SELECT * FROM users WHERE email = ?",
                    (email,)
                ).fetchone()
            finally:
                conn.close()
        return None
    
    def _get_user_by_id(self, user_id: str) -> Optional[Dict]:
        """ID로 사용자 조회"""
        if self.db_manager:
            conn = self.db_manager._get_connection()
            try:
                return conn.execute(
                    "SELECT * FROM users WHERE id = ?",
                    (user_id,)
                ).fetchone()
            finally:
                conn.close()
        return None
    
    def _update_last_login(self, user_id: str):
        """마지막 로그인 시간 업데이트"""
        if self.db_manager:
            conn = self.db_manager._get_connection()
            try:
                conn.execute(
                    "UPDATE users SET last_login = ? WHERE id = ?",
                    (datetime.now().isoformat(), user_id)
                )
                conn.commit()
            finally:
                conn.close()
    
    def _get_user_permissions(self, role: str) -> Dict[str, List[str]]:
        """사용자 권한 목록 조회"""
        permissions = {}
        
        for resource, actions in PERMISSION_MATRIX.items():
            allowed_actions = []
            for action, allowed_roles in actions.items():
                if role in allowed_roles:
                    allowed_actions.append(action)
            
            if allowed_actions:
                permissions[resource] = allowed_actions
        
        return permissions
    
    # =========================================================================
    # 🚫 계정 보안
    # =========================================================================
    
    def _record_failed_attempt(self, email: str):
        """로그인 실패 기록"""
        with self._lock:
            if email not in self._failed_attempts:
                self._failed_attempts[email] = []
            
            self._failed_attempts[email].append(datetime.now())
            
            # 오래된 기록 제거 (15분 이상)
            cutoff = datetime.now() - timedelta(minutes=15)
            self._failed_attempts[email] = [
                attempt for attempt in self._failed_attempts[email]
                if attempt > cutoff
            ]
    
    def _is_account_locked(self, email: str) -> bool:
        """계정 잠금 상태 확인"""
        with self._lock:
            if email not in self._failed_attempts:
                return False
            
            # 최근 15분 내 실패 횟수 확인
            recent_attempts = self._failed_attempts[email]
            return len(recent_attempts) >= SECURITY_CONFIG['max_login_attempts']
    
    def _clear_failed_attempts(self, email: str):
        """로그인 실패 기록 초기화"""
        with self._lock:
            if email in self._failed_attempts:
                del self._failed_attempts[email]
    
    # =========================================================================
    # 🔄 세션 관리
    # =========================================================================
    
    def _start_session_cleanup(self):
        """세션 정리 스케줄러 시작"""
        def cleanup():
            while True:
                try:
                    self._cleanup_expired_sessions()
                    threading.Event().wait(300)  # 5분마다
                except Exception as e:
                    logger.error(f"Session cleanup error: {str(e)}")
        
        thread = threading.Thread(target=cleanup, daemon=True)
        thread.start()
    
    def _cleanup_expired_sessions(self):
        """만료된 세션 정리"""
        with self._lock:
            now = datetime.now()
            expired = []
            
            for user_id, session in self._sessions.items():
                # 세션 타임아웃 확인
                if now - session['last_activity'] > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
                    expired.append(user_id)
            
            for user_id in expired:
                del self._sessions[user_id]
                logger.info(f"Session expired for user: {user_id}")
    
    def update_session_activity(self, user_id: str):
        """세션 활동 시간 업데이트"""
        with self._lock:
            if user_id in self._sessions:
                self._sessions[user_id]['last_activity'] = datetime.now()
    
    # =========================================================================
    # 👤 사용자 프로필 관리
    # =========================================================================
    
    def update_user_profile(self,
                           user_id: str,
                           updates: Dict[str, Any]) -> Tuple[bool, str]:
        """
        사용자 프로필 업데이트
        
        Args:
            user_id: 사용자 ID
            updates: 업데이트할 필드들
            
        Returns:
            (성공여부, 메시지)
        """
        try:
            if self.db_manager:
                conn = self.db_manager._get_connection()
                try:
                    # 업데이트 가능한 필드만 필터링
                    allowed_fields = ['name', 'organization', 'settings']
                    filtered_updates = {
                        k: v for k, v in updates.items() 
                        if k in allowed_fields
                    }
                    
                    if not filtered_updates:
                        return (False, "업데이트할 항목이 없습니다.")
                    
                    # SQL 쿼리 생성
                    set_clause = ", ".join([f"{k} = ?" for k in filtered_updates.keys()])
                    values = list(filtered_updates.values())
                    values.append(datetime.now().isoformat())  # updated_at
                    values.append(user_id)
                    
                    conn.execute(f"""
                        UPDATE users 
                        SET {set_clause}, updated_at = ?
                        WHERE id = ?
                    """, values)
                    
                    conn.commit()
                    logger.info(f"User profile updated: {user_id}")
                    return (True, "프로필이 업데이트되었습니다.")
                    
                finally:
                    conn.close()
            
            return (False, "데이터베이스 연결 오류")
            
        except Exception as e:
            logger.error(f"Update profile error: {str(e)}")
            return (False, "프로필 업데이트 중 오류가 발생했습니다.")
    
    def change_password(self,
                       user_id: str,
                       old_password: str,
                       new_password: str) -> Tuple[bool, str]:
        """
        비밀번호 변경
        
        Args:
            user_id: 사용자 ID
            old_password: 현재 비밀번호
            new_password: 새 비밀번호
            
        Returns:
            (성공여부, 메시지)
        """
        try:
            # 사용자 조회
            user = self._get_user_by_id(user_id)
            if not user:
                return (False, "사용자를 찾을 수 없습니다.")
            
            # 현재 비밀번호 확인
            if not self._verify_password(old_password, user['password_hash']):
                return (False, "현재 비밀번호가 올바르지 않습니다.")
            
            # 새 비밀번호 검증
            is_valid, msg = self._validate_password(new_password)
            if not is_valid:
                return (False, msg)
            
            # 비밀번호 업데이트
            new_hash = self._hash_password(new_password)
            
            if self.db_manager:
                conn = self.db_manager._get_connection()
                try:
                    conn.execute("""
                        UPDATE users 
                        SET password_hash = ?, updated_at = ?
                        WHERE id = ?
                    """, (new_hash, datetime.now().isoformat(), user_id))
                    
                    conn.commit()
                    logger.info(f"Password changed for user: {user_id}")
                    return (True, "비밀번호가 변경되었습니다.")
                    
                finally:
                    conn.close()
            
            return (False, "데이터베이스 연결 오류")
            
        except Exception as e:
            logger.error(f"Change password error: {str(e)}")
            return (False, "비밀번호 변경 중 오류가 발생했습니다.")
    
    # =========================================================================
    # 🔄 동기화 지원
    # =========================================================================
    
    def export_user_data(self, user_id: str) -> Optional[Dict]:
        """
        사용자 데이터 내보내기 (동기화용)
        
        Args:
            user_id: 사용자 ID
            
        Returns:
            사용자 데이터 (민감정보 제외)
        """
        try:
            user = self._get_user_by_id(user_id)
            if user:
                # 민감정보 제외
                export_data = {
                    'id': user['id'],
                    'email': user['email'],
                    'name': user['name'],
                    'organization': user['organization'],
                    'role': user['role'],
                    'created_at': user['created_at'],
                    'updated_at': user['updated_at'],
                    'settings': user['settings']
                }
                return export_data
            
            return None
            
        except Exception as e:
            logger.error(f"Export user data error: {str(e)}")
            return None
    
    def import_user_data(self, user_data: Dict) -> bool:
        """
        사용자 데이터 가져오기 (동기화용)
        
        Args:
            user_data: 사용자 데이터
            
        Returns:
            성공 여부
        """
        try:
            # 기존 사용자 확인
            existing = self._get_user_by_email(user_data['email'])
            
            if existing:
                # 업데이트
                return self.update_user_profile(
                    existing['id'],
                    {
                        'name': user_data.get('name'),
                        'organization': user_data.get('organization'),
                        'settings': user_data.get('settings')
                    }
                )[0]
            else:
                # 새로 생성 (임시 비밀번호)
                temp_password = secrets.token_urlsafe(16)
                success, _, _ = self.register_user(
                    email=user_data['email'],
                    password=temp_password,
                    name=user_data['name'],
                    organization=user_data.get('organization')
                )
                return success
                
        except Exception as e:
            logger.error(f"Import user data error: {str(e)}")
            return False


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


# ===========================================================================
# 🧪 테스트 코드
# ===========================================================================

if __name__ == "__main__":
    # 테스트용 코드
    print("AuthManager 모듈 로드 완료")
    
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
