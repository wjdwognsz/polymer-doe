"""
🔐 로그인 페이지 - Universal DOE Platform
오프라인 우선 인증 시스템 with 선택적 클라우드 동기화
"""
import streamlit as st
import sys
from pathlib import Path
from datetime import datetime, timedelta
import re
import time
from typing import Optional, Dict, Any, Tuple
import bcrypt
import json
import os
import secrets
import hashlib
from enum import Enum

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

# 페이지 설정 (필수 - 최상단)
st.set_page_config(
    page_title="로그인 - Universal DOE Platform",
    page_icon="🔐",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 모듈 임포트
try:
    from utils.auth_manager import get_auth_manager
    from utils.common_ui import get_common_ui
    from utils.database_manager import get_database_manager
    from utils.error_handler import handle_error, ErrorSeverity
    from utils.notification_manager import get_notification_manager
    from config.app_config import SECURITY_CONFIG, APP_INFO
    from config.theme_config import apply_theme
    from config.error_config import ERROR_CODES, USER_FRIENDLY_MESSAGES
except ImportError as e:
    st.error(f"필수 모듈을 찾을 수 없습니다: {e}")
    st.info("프로젝트 루트에서 실행 중인지 확인해주세요.")
    st.stop()

# 전역 인스턴스
auth_manager = get_auth_manager()
ui = get_common_ui()
db_manager = get_database_manager()
notification_manager = get_notification_manager()

# 인증 탭 열거형
class AuthTab(Enum):
    LOGIN = "login"
    SIGNUP = "signup"
    FORGOT = "forgot"
    VERIFY = "verify"

# 상수 정의
MAX_LOGIN_ATTEMPTS = SECURITY_CONFIG.get('max_login_attempts', 5)
LOCKOUT_DURATION = timedelta(minutes=SECURITY_CONFIG.get('lockout_minutes', 30))
PASSWORD_MIN_LENGTH = SECURITY_CONFIG.get('password_min_length', 8)
SESSION_TIMEOUT = timedelta(hours=SECURITY_CONFIG.get('session_timeout_hours', 24))
CSRF_TOKEN_LENGTH = 32

# UI 텍스트 (다국어 지원)
UI_TEXTS = {
    'ko': {
        'app_title': 'Universal DOE Platform',
        'app_subtitle': '모든 과학자를 위한 AI 기반 실험 설계 플랫폼',
        'login_title': '로그인',
        'signup_title': '회원가입', 
        'forgot_title': '비밀번호 찾기',
        'verify_title': '이메일 인증',
        'email_label': '이메일',
        'email_placeholder': '이메일 주소를 입력하세요',
        'password_label': '비밀번호',
        'password_placeholder': '비밀번호를 입력하세요',
        'password_confirm_label': '비밀번호 확인',
        'password_confirm_placeholder': '비밀번호를 다시 입력하세요',
        'name_label': '이름',
        'name_placeholder': '이름을 입력하세요',
        'organization_label': '소속 기관',
        'organization_placeholder': '소속 기관 (선택사항)',
        'security_question_label': '보안 질문',
        'security_answer_label': '보안 답변',
        'login_button': '로그인',
        'signup_button': '회원가입',
        'reset_button': '비밀번호 재설정',
        'guest_button': '게스트로 둘러보기',
        'back_to_login': '← 로그인으로 돌아가기',
        'remember_me': '로그인 상태 유지',
        'terms_agree': '서비스 이용약관에 동의합니다',
        'privacy_agree': '개인정보처리방침에 동의합니다',
        'marketing_agree': '마케팅 정보 수신에 동의합니다 (선택)',
        'forgot_password_link': '비밀번호를 잊으셨나요?',
        'signup_link': '계정이 없으신가요? 회원가입',
        'social_login_title': '간편 로그인',
        'google_login': 'Google로 로그인',
        'github_login': 'GitHub로 로그인',
        'offline_mode_info': '🔌 오프라인 모드에서는 소셜 로그인을 사용할 수 없습니다.',
        'password_strength': '비밀번호 강도',
        'password_requirements': '비밀번호 요구사항',
        'verification_code_label': '인증 코드',
        'verification_code_placeholder': '6자리 인증 코드를 입력하세요',
        'resend_code': '인증 코드 재전송',
        'verify_button': '인증하기'
    },
    'en': {
        'app_title': 'Universal DOE Platform',
        'app_subtitle': 'AI-Powered Experiment Design Platform for All Scientists',
        'login_title': 'Login',
        'signup_title': 'Sign Up',
        'forgot_title': 'Forgot Password',
        'verify_title': 'Email Verification',
        'email_label': 'Email',
        'email_placeholder': 'Enter your email address',
        'password_label': 'Password',
        'password_placeholder': 'Enter your password',
        'password_confirm_label': 'Confirm Password',
        'password_confirm_placeholder': 'Re-enter your password',
        'name_label': 'Name',
        'name_placeholder': 'Enter your name',
        'organization_label': 'Organization',
        'organization_placeholder': 'Organization (optional)',
        'security_question_label': 'Security Question',
        'security_answer_label': 'Security Answer',
        'login_button': 'Login',
        'signup_button': 'Sign Up',
        'reset_button': 'Reset Password',
        'guest_button': 'Browse as Guest',
        'back_to_login': '← Back to Login',
        'remember_me': 'Remember me',
        'terms_agree': 'I agree to the Terms of Service',
        'privacy_agree': 'I agree to the Privacy Policy',
        'marketing_agree': 'I agree to receive marketing emails (optional)',
        'forgot_password_link': 'Forgot your password?',
        'signup_link': "Don't have an account? Sign up",
        'social_login_title': 'Social Login',
        'google_login': 'Login with Google',
        'github_login': 'Login with GitHub',
        'offline_mode_info': '🔌 Social login is not available in offline mode.',
        'password_strength': 'Password Strength',
        'password_requirements': 'Password Requirements',
        'verification_code_label': 'Verification Code',
        'verification_code_placeholder': 'Enter 6-digit verification code',
        'resend_code': 'Resend Code',
        'verify_button': 'Verify'
    }
}

# 보안 질문 목록
SECURITY_QUESTIONS = {
    'ko': [
        "첫 반려동물의 이름은?",
        "어머니의 결혼 전 성함은?",
        "출생 도시는?",
        "첫 학교 이름은?",
        "가장 좋아하는 음식은?"
    ],
    'en': [
        "What was your first pet's name?",
        "What is your mother's maiden name?",
        "What city were you born in?",
        "What was the name of your first school?",
        "What is your favorite food?"
    ]
}

def initialize_session_state():
    """세션 상태 초기화"""
    defaults = {
        'current_tab': AuthTab.LOGIN.value,
        'login_attempts': {},
        'show_password': False,
        'show_password_confirm': False,
        'language': 'ko',
        'theme': 'light',
        'csrf_token': None,
        'oauth_state': None,
        'verification_email': None,
        'last_activity': datetime.now(),
        'is_online': check_online_status(),
        'password_reset_token': None,
        'temp_user_data': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def get_text(key: str) -> str:
    """현재 언어에 맞는 텍스트 반환"""
    lang = st.session_state.get('language', 'ko')
    return UI_TEXTS.get(lang, UI_TEXTS['ko']).get(key, key)

def generate_csrf_token() -> str:
    """CSRF 토큰 생성"""
    token = secrets.token_urlsafe(CSRF_TOKEN_LENGTH)
    st.session_state.csrf_token = token
    return token

def verify_csrf_token(token: str) -> bool:
    """CSRF 토큰 검증"""
    return token == st.session_state.get('csrf_token')

def check_session_timeout():
    """세션 타임아웃 확인"""
    if 'last_activity' in st.session_state:
        elapsed = datetime.now() - st.session_state.last_activity
        if elapsed > SESSION_TIMEOUT:
            st.session_state.clear()
            ui.show_warning("세션이 만료되었습니다. 다시 로그인해주세요.")
            st.rerun()
    
    st.session_state.last_activity = datetime.now()

def check_online_status() -> bool:
    """온라인 상태 확인 (캐시됨)"""
    # 5분마다 온라인 상태 재확인
    if 'last_online_check' in st.session_state:
        if datetime.now() - st.session_state.last_online_check < timedelta(minutes=5):
            return st.session_state.is_online
    
    try:
        import requests
        response = requests.get('https://www.google.com', timeout=3)
        is_online = response.status_code == 200
    except:
        is_online = False
    
    st.session_state.is_online = is_online
    st.session_state.last_online_check = datetime.now()
    return is_online

def validate_email(email: str) -> Tuple[bool, Optional[str]]:
    """이메일 형식 검증"""
    if not email:
        return False, "이메일을 입력해주세요."
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        return False, USER_FRIENDLY_MESSAGES.get('7001', "올바른 이메일 형식이 아닙니다.")
    
    # 이메일 길이 체크
    if len(email) > 254:
        return False, "이메일 주소가 너무 깁니다."
    
    return True, None

def validate_password(password: str) -> Dict[str, Any]:
    """비밀번호 강도 검증"""
    result = {
        'valid': True,
        'score': 0,
        'feedback': [],
        'requirements': {
            'length': False,
            'uppercase': False,
            'lowercase': False,
            'number': False,
            'special': False
        }
    }
    
    # 길이 체크
    if len(password) >= PASSWORD_MIN_LENGTH:
        result['requirements']['length'] = True
        result['score'] += 1
    else:
        result['feedback'].append(f"최소 {PASSWORD_MIN_LENGTH}자 이상")
        result['valid'] = False
    
    # 대문자 체크
    if re.search(r'[A-Z]', password):
        result['requirements']['uppercase'] = True
        result['score'] += 1
    else:
        result['feedback'].append("대문자 포함 필요")
    
    # 소문자 체크
    if re.search(r'[a-z]', password):
        result['requirements']['lowercase'] = True
        result['score'] += 1
    else:
        result['feedback'].append("소문자 포함 필요")
    
    # 숫자 체크
    if re.search(r'\d', password):
        result['requirements']['number'] = True
        result['score'] += 1
    else:
        result['feedback'].append("숫자 포함 필요")
    
    # 특수문자 체크
    if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        result['requirements']['special'] = True
        result['score'] += 1
    else:
        result['feedback'].append("특수문자 포함 필요")
    
    # 추가 보너스
    if len(password) >= 12:
        result['score'] += 1
    if len(password) >= 16:
        result['score'] += 1
    
    # 강도 계산
    if result['score'] <= 2:
        result['strength'] = 'weak'
        result['strength_text'] = '약함'
        result['color'] = '#ff4444'
    elif result['score'] <= 4:
        result['strength'] = 'medium'
        result['strength_text'] = '보통'
        result['color'] = '#ff9944'
    elif result['score'] <= 6:
        result['strength'] = 'strong'
        result['strength_text'] = '강함'
        result['color'] = '#00aa00'
    else:
        result['strength'] = 'very_strong'
        result['strength_text'] = '매우 강함'
        result['color'] = '#00ff00'
    
    return result

def check_account_lockout(email: str) -> Tuple[bool, Optional[str]]:
    """계정 잠금 상태 확인"""
    attempts = st.session_state.login_attempts.get(email, {})
    
    if attempts.get('count', 0) >= MAX_LOGIN_ATTEMPTS:
        last_attempt = attempts.get('last_attempt')
        if last_attempt:
            lockout_end = last_attempt + LOCKOUT_DURATION
            if datetime.now() < lockout_end:
                remaining = int((lockout_end - datetime.now()).total_seconds() / 60)
                return True, f"너무 많은 로그인 시도로 계정이 잠겼습니다. {remaining}분 후 다시 시도해주세요."
            else:
                # 잠금 해제
                st.session_state.login_attempts[email] = {'count': 0}
    
    return False, None

def record_login_attempt(email: str, success: bool):
    """로그인 시도 기록"""
    if success:
        # 성공 시 시도 횟수 초기화
        st.session_state.login_attempts.pop(email, None)
    else:
        # 실패 시 시도 횟수 증가
        if email not in st.session_state.login_attempts:
            st.session_state.login_attempts[email] = {'count': 0}
        
        st.session_state.login_attempts[email]['count'] += 1
        st.session_state.login_attempts[email]['last_attempt'] = datetime.now()
    
    # 데이터베이스에 로그 기록
    try:
        db_manager.log_activity({
            'type': 'login_attempt',
            'email': email,
            'success': success,
            'timestamp': datetime.now(),
            'ip_address': get_client_ip()
        })
    except:
        pass  # 로그 실패는 무시

def get_client_ip() -> Optional[str]:
    """클라이언트 IP 주소 가져오기"""
    try:
        # Streamlit Cloud 환경
        headers = st.context.headers
        if headers:
            return headers.get('X-Forwarded-For', '').split(',')[0].strip()
    except:
        pass
    return None

def render_header():
    """공통 헤더 렌더링"""
    st.markdown(f"""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='font-size: 3rem; font-weight: bold; 
                   background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   -webkit-background-clip: text;
                   -webkit-text-fill-color: transparent;
                   margin-bottom: 0.5rem;'>
            {get_text('app_title')}
        </h1>
        <p style='font-size: 1.2rem; color: #666;'>
            {get_text('app_subtitle')}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 온라인/오프라인 상태 표시
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.session_state.is_online:
            st.success("🌐 온라인 모드")
        else:
            st.warning("🔌 오프라인 모드")

def render_login_form():
    """로그인 폼 렌더링"""
    with st.form("login_form", clear_on_submit=False):
        st.markdown(f"### 🔐 {get_text('login_title')}")
        
        # 이메일 입력
        email = st.text_input(
            get_text('email_label'),
            placeholder=get_text('email_placeholder'),
            key="login_email",
            help="가입하신 이메일 주소를 입력하세요"
        )
        
        # 비밀번호 입력
        col1, col2 = st.columns([5, 1])
        with col1:
            password = st.text_input(
                get_text('password_label'),
                type="password" if not st.session_state.show_password else "text",
                placeholder=get_text('password_placeholder'),
                key="login_password"
            )
        with col2:
            st.write("")  # 간격 맞추기
            st.write("")  # 간격 맞추기
            if st.form_submit_button(
                "👁️" if not st.session_state.show_password else "🙈",
                help="비밀번호 표시/숨기기"
            ):
                st.session_state.show_password = not st.session_state.show_password
        
        # 추가 옵션
        col1, col2 = st.columns(2)
        with col1:
            remember_me = st.checkbox(get_text('remember_me'), value=True)
        with col2:
            st.empty()
        
        # CSRF 토큰
        csrf_token = generate_csrf_token()
        
        # 로그인 버튼
        submitted = st.form_submit_button(
            get_text('login_button'),
            use_container_width=True,
            type="primary"
        )
        
        if submitted:
            handle_login(email, password, remember_me, csrf_token)
    
    # 추가 링크
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button(get_text('forgot_password_link'), use_container_width=True):
            st.session_state.current_tab = AuthTab.FORGOT.value
            st.rerun()
    
    with col2:
        if st.button(get_text('signup_title'), use_container_width=True):
            st.session_state.current_tab = AuthTab.SIGNUP.value
            st.rerun()
    
    with col3:
        if st.button(get_text('guest_button'), use_container_width=True):
            handle_guest_login()

def render_signup_form():
    """회원가입 폼 렌더링"""
    st.markdown(f"### 👤 {get_text('signup_title')}")
    
    with st.form("signup_form", clear_on_submit=False):
        # 기본 정보
        st.markdown("#### 📝 기본 정보")
        
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input(
                f"{get_text('name_label')} *",
                placeholder=get_text('name_placeholder'),
                help="실명을 입력해주세요"
            )
        
        with col2:
            email = st.text_input(
                f"{get_text('email_label')} *",
                placeholder=get_text('email_placeholder'),
                help="로그인에 사용할 이메일 주소"
            )
        
        organization = st.text_input(
            get_text('organization_label'),
            placeholder=get_text('organization_placeholder'),
            help="선택사항입니다"
        )
        
        # 비밀번호
        st.markdown("#### 🔒 비밀번호 설정")
        
        col1, col2 = st.columns(2)
        with col1:
            password = st.text_input(
                f"{get_text('password_label')} *",
                type="password" if not st.session_state.show_password else "text",
                help=f"최소 {PASSWORD_MIN_LENGTH}자 이상, 영문 대/소문자, 숫자, 특수문자 포함"
            )
            
            # 비밀번호 강도 표시
            if password:
                pw_result = validate_password(password)
                progress = min(pw_result['score'] / 7.0, 1.0)
                
                st.progress(progress, text=f"{get_text('password_strength')}: {pw_result['strength_text']}")
                
                # 요구사항 체크리스트
                with st.expander(get_text('password_requirements')):
                    req = pw_result['requirements']
                    st.write(f"{'✅' if req['length'] else '❌'} 최소 {PASSWORD_MIN_LENGTH}자 이상")
                    st.write(f"{'✅' if req['uppercase'] else '❌'} 대문자 포함")
                    st.write(f"{'✅' if req['lowercase'] else '❌'} 소문자 포함")
                    st.write(f"{'✅' if req['number'] else '❌'} 숫자 포함")
                    st.write(f"{'✅' if req['special'] else '❌'} 특수문자 포함")
        
        with col2:
            password_confirm = st.text_input(
                f"{get_text('password_confirm_label')} *",
                type="password" if not st.session_state.show_password_confirm else "text",
                help="동일한 비밀번호를 다시 입력하세요"
            )
        
        # 보안 질문 (오프라인 모드용)
        st.markdown("#### 🔐 보안 설정")
        
        security_questions = SECURITY_QUESTIONS[st.session_state.language]
        selected_question = st.selectbox(
            get_text('security_question_label'),
            security_questions,
            help="비밀번호 찾기에 사용됩니다"
        )
        
        security_answer = st.text_input(
            get_text('security_answer_label'),
            help="대소문자를 구분합니다"
        )
        
        # 약관 동의
        st.markdown("#### 📋 약관 동의")
        
        terms_agree = st.checkbox(f"{get_text('terms_agree')} *")
        privacy_agree = st.checkbox(f"{get_text('privacy_agree')} *")
        marketing_agree = st.checkbox(get_text('marketing_agree'))
        
        col1, col2 = st.columns(2)
        with col1:
            if st.form_submit_button("이용약관 보기"):
                show_terms_modal()
        with col2:
            if st.form_submit_button("개인정보처리방침 보기"):
                show_privacy_modal()
        
        # CSRF 토큰
        csrf_token = generate_csrf_token()
        
        # 가입 버튼
        submitted = st.form_submit_button(
            get_text('signup_button'),
            use_container_width=True,
            type="primary"
        )
        
        if submitted:
            handle_signup({
                'name': name,
                'email': email,
                'organization': organization,
                'password': password,
                'password_confirm': password_confirm,
                'security_question': selected_question,
                'security_answer': security_answer,
                'terms_agree': terms_agree,
                'privacy_agree': privacy_agree,
                'marketing_agree': marketing_agree,
                'csrf_token': csrf_token
            })
    
    # 로그인으로 돌아가기
    if st.button(get_text('back_to_login')):
        st.session_state.current_tab = AuthTab.LOGIN.value
        st.rerun()

def render_forgot_password_form():
    """비밀번호 찾기 폼 렌더링"""
    st.markdown(f"### 🔑 {get_text('forgot_title')}")
    
    if st.session_state.is_online:
        # 온라인 모드 - 이메일로 재설정
        st.info("📧 가입하신 이메일로 비밀번호 재설정 링크를 보내드립니다.")
        
        with st.form("forgot_email_form"):
            email = st.text_input(
                get_text('email_label'),
                placeholder=get_text('email_placeholder'),
                help="가입하신 이메일 주소를 입력하세요"
            )
            
            csrf_token = generate_csrf_token()
            
            submitted = st.form_submit_button(
                "재설정 링크 전송",
                use_container_width=True,
                type="primary"
            )
            
            if submitted:
                handle_password_reset_email(email, csrf_token)
    else:
        # 오프라인 모드 - 보안 질문으로 재설정
        st.warning("🔌 오프라인 모드에서는 보안 질문으로 비밀번호를 재설정합니다.")
        
        with st.form("forgot_security_form"):
            email = st.text_input(
                get_text('email_label'),
                placeholder=get_text('email_placeholder')
            )
            
            security_answer = st.text_input(
                "보안 질문 답변",
                type="password",
                placeholder="가입 시 설정한 보안 답변을 입력하세요",
                help="대소문자를 구분합니다"
            )
            
            csrf_token = generate_csrf_token()
            
            submitted = st.form_submit_button(
                get_text('reset_button'),
                use_container_width=True,
                type="primary"
            )
            
            if submitted:
                handle_password_reset_security(email, security_answer, csrf_token)
    
    # 로그인으로 돌아가기
    if st.button(get_text('back_to_login')):
        st.session_state.current_tab = AuthTab.LOGIN.value
        st.rerun()

def render_email_verification_form():
    """이메일 인증 폼 렌더링"""
    st.markdown(f"### 📧 {get_text('verify_title')}")
    
    if not st.session_state.verification_email:
        st.error("인증이 필요한 이메일이 없습니다.")
        if st.button(get_text('back_to_login')):
            st.session_state.current_tab = AuthTab.LOGIN.value
            st.rerun()
        return
    
    st.info(f"📧 {st.session_state.verification_email}로 전송된 인증 코드를 입력하세요.")
    
    with st.form("verification_form"):
        code = st.text_input(
            get_text('verification_code_label'),
            placeholder=get_text('verification_code_placeholder'),
            max_chars=6,
            help="이메일로 전송된 6자리 코드를 입력하세요"
        )
        
        csrf_token = generate_csrf_token()
        
        col1, col2 = st.columns(2)
        
        with col1:
            submitted = st.form_submit_button(
                get_text('verify_button'),
                use_container_width=True,
                type="primary"
            )
        
        with col2:
            resend = st.form_submit_button(
                get_text('resend_code'),
                use_container_width=True
            )
        
        if submitted:
            handle_email_verification(code, csrf_token)
        
        if resend:
            handle_resend_verification(csrf_token)
    
    # 로그인으로 돌아가기
    if st.button(get_text('back_to_login')):
        st.session_state.current_tab = AuthTab.LOGIN.value
        st.session_state.verification_email = None
        st.rerun()

def render_social_login():
    """소셜 로그인 섹션 렌더링"""
    if not st.session_state.is_online:
        st.info(get_text('offline_mode_info'))
        return
    
    st.markdown("---")
    st.markdown(f"### {get_text('social_login_title')}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(f"🔷 {get_text('google_login')}", use_container_width=True):
            handle_google_oauth()
    
    with col2:
        if st.button(f"🐙 {get_text('github_login')}", use_container_width=True):
            handle_github_oauth()

def handle_login(email: str, password: str, remember_me: bool, csrf_token: str):
    """로그인 처리"""
    # CSRF 검증
    if not verify_csrf_token(csrf_token):
        ui.show_error("보안 토큰이 유효하지 않습니다. 페이지를 새로고침하세요.")
        return
    
    # 입력값 검증
    valid, error_msg = validate_email(email)
    if not valid:
        ui.show_error(error_msg)
        return
    
    if not password:
        ui.show_error("비밀번호를 입력해주세요.")
        return
    
    # 계정 잠금 확인
    locked, lock_msg = check_account_lockout(email)
    if locked:
        ui.show_error(lock_msg)
        return
    
    # 로그인 시도
    with st.spinner("로그인 중..."):
        try:
            success, message, user_info = auth_manager.login(
                email=email,
                password=password,
                remember_me=remember_me
            )
            
            if success:
                # 로그인 성공
                record_login_attempt(email, True)
                
                # 세션 설정
                st.session_state.authenticated = True
                st.session_state.user = user_info
                st.session_state.remember_me = remember_me
                
                # 환영 메시지
                placeholder = st.empty()
                for i in range(3):
                    placeholder.success(f"환영합니다, {user_info['name']}님! {'🎉' * (i+1)}")
                    time.sleep(0.3)
                
                # 알림 생성
                notification_manager.add_notification(
                    user_id=user_info['id'],
                    type='system',
                    title='로그인 성공',
                    message=f'{datetime.now().strftime("%Y-%m-%d %H:%M")}에 로그인하였습니다.',
                    priority='low'
                )
                
                # 대시보드로 이동
                time.sleep(0.5)
                st.switch_page("pages/1_📊_Dashboard.py")
            else:
                # 로그인 실패
                record_login_attempt(email, False)
                attempts = st.session_state.login_attempts.get(email, {})
                attempts_left = MAX_LOGIN_ATTEMPTS - attempts.get('count', 0)
                
                if attempts_left > 0:
                    ui.show_error(f"{message} (남은 시도: {attempts_left}회)")
                else:
                    ui.show_error(f"계정이 잠겼습니다. {LOCKOUT_DURATION.total_seconds()//60:.0f}분 후 다시 시도해주세요.")
                    
        except Exception as e:
            handle_error(e, "로그인 처리 중 오류가 발생했습니다.", ErrorSeverity.ERROR)

def handle_signup(data: Dict[str, Any]):
    """회원가입 처리"""
    # CSRF 검증
    if not verify_csrf_token(data['csrf_token']):
        ui.show_error("보안 토큰이 유효하지 않습니다. 페이지를 새로고침하세요.")
        return
    
    # 입력값 검증
    errors = []
    
    # 이름 검증
    if not data['name']:
        errors.append("이름을 입력해주세요.")
    elif len(data['name']) < 2:
        errors.append("이름은 2자 이상이어야 합니다.")
    
    # 이메일 검증
    valid, error_msg = validate_email(data['email'])
    if not valid:
        errors.append(error_msg)
    
    # 비밀번호 검증
    if not data['password']:
        errors.append("비밀번호를 입력해주세요.")
    else:
        pw_result = validate_password(data['password'])
        if not pw_result['valid']:
            errors.extend(pw_result['feedback'])
    
    # 비밀번호 확인
    if data['password'] != data['password_confirm']:
        errors.append("비밀번호가 일치하지 않습니다.")
    
    # 보안 질문/답변
    if not data['security_answer']:
        errors.append("보안 답변을 입력해주세요.")
    
    # 약관 동의
    if not data['terms_agree']:
        errors.append("서비스 이용약관에 동의해주세요.")
    
    if not data['privacy_agree']:
        errors.append("개인정보처리방침에 동의해주세요.")
    
    # 에러가 있으면 표시
    if errors:
        for error in errors:
            ui.show_error(f"• {error}")
        return
    
    # 회원가입 처리
    with st.spinner("회원가입 처리 중..."):
        try:
            success, message, user_id = auth_manager.register(
                email=data['email'],
                password=data['password'],
                name=data['name'],
                organization=data.get('organization', ''),
                security_question=data['security_question'],
                security_answer=data['security_answer'],
                marketing_agree=data.get('marketing_agree', False)
            )
            
            if success:
                # 회원가입 성공
                st.balloons()
                ui.show_success("🎉 회원가입이 완료되었습니다!")
                
                # 온라인 모드에서는 이메일 인증 필요
                if st.session_state.is_online:
                    st.session_state.verification_email = data['email']
                    st.session_state.current_tab = AuthTab.VERIFY.value
                    ui.show_info("📧 이메일 인증을 완료해주세요.")
                    time.sleep(2)
                    st.rerun()
                else:
                    # 오프라인 모드에서는 바로 로그인 가능
                    st.session_state.current_tab = AuthTab.LOGIN.value
                    time.sleep(2)
                    st.rerun()
            else:
                ui.show_error(message)
                
        except Exception as e:
            handle_error(e, "회원가입 처리 중 오류가 발생했습니다.", ErrorSeverity.ERROR)

def handle_guest_login():
    """게스트 로그인 처리"""
    with st.spinner("게스트 모드로 접속 중..."):
        try:
            # 게스트 세션 생성
            guest_id = f"guest_{int(datetime.now().timestamp())}"
            
            st.session_state.authenticated = True
            st.session_state.guest_mode = True
            st.session_state.user = {
                'id': guest_id,
                'name': '게스트',
                'email': 'guest@local',
                'role': 'guest',
                'organization': '',
                'permissions': ['read_demo', 'local_save_only'],
                'created_at': datetime.now()
            }
            
            # 게스트 세션 데이터베이스에 기록
            db_manager.create_guest_session(guest_id)
            
            ui.show_info("📚 게스트 모드로 접속했습니다. 일부 기능이 제한됩니다.")
            time.sleep(1)
            
            # 대시보드로 이동
            st.switch_page("pages/1_📊_Dashboard.py")
            
        except Exception as e:
            handle_error(e, "게스트 로그인 중 오류가 발생했습니다.", ErrorSeverity.WARNING)

def handle_google_oauth():
    """Google OAuth 시작"""
    try:
        client_id = os.getenv('GOOGLE_OAUTH_CLIENT_ID')
        if not client_id:
            ui.show_error("Google 로그인이 설정되지 않았습니다.")
            return
        
        # OAuth state 생성
        state = secrets.token_urlsafe(32)
        st.session_state.oauth_state = f"google_{state}"
        
        # OAuth URL 생성
        from urllib.parse import urlencode
        params = {
            'client_id': client_id,
            'redirect_uri': 'http://localhost:8501/auth/callback',
            'response_type': 'code',
            'scope': 'openid email profile',
            'state': state,
            'access_type': 'offline',
            'prompt': 'consent'
        }
        
        auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"
        
        # 리다이렉트
        st.components.v1.html(f"""
            <script>
                window.location.href = "{auth_url}";
            </script>
        """, height=0)
        
    except Exception as e:
        handle_error(e, "Google 로그인 초기화 중 오류가 발생했습니다.", ErrorSeverity.WARNING)

def handle_github_oauth():
    """GitHub OAuth 시작"""
    try:
        client_id = os.getenv('GITHUB_CLIENT_ID')
        if not client_id:
            ui.show_error("GitHub 로그인이 설정되지 않았습니다.")
            return
        
        # OAuth state 생성
        state = secrets.token_urlsafe(32)
        st.session_state.oauth_state = f"github_{state}"
        
        # OAuth URL 생성
        from urllib.parse import urlencode
        params = {
            'client_id': client_id,
            'redirect_uri': 'http://localhost:8501/auth/github/callback',
            'scope': 'user:email',
            'state': state
        }
        
        auth_url = f"https://github.com/login/oauth/authorize?{urlencode(params)}"
        
        # 리다이렉트
        st.components.v1.html(f"""
            <script>
                window.location.href = "{auth_url}";
            </script>
        """, height=0)
        
    except Exception as e:
        handle_error(e, "GitHub 로그인 초기화 중 오류가 발생했습니다.", ErrorSeverity.WARNING)

def handle_oauth_callback():
    """OAuth 콜백 처리"""
    params = st.query_params
    
    if 'code' in params:
        code = params['code']
        state = params.get('state', '')
        
        if st.session_state.get('oauth_state'):
            if 'google' in st.session_state.oauth_state:
                process_google_oauth_callback(code, state)
            elif 'github' in st.session_state.oauth_state:
                process_github_oauth_callback(code, state)
        
        # URL 파라미터 정리
        st.query_params.clear()
    
    elif 'error' in params:
        ui.show_error(f"OAuth 로그인 실패: {params.get('error_description', params['error'])}")
        st.session_state.pop('oauth_state', None)
        st.query_params.clear()

def process_google_oauth_callback(code: str, state: str):
    """Google OAuth 콜백 처리"""
    try:
        import requests
        
        # 액세스 토큰 요청
        token_url = "https://oauth2.googleapis.com/token"
        token_data = {
            'code': code,
            'client_id': os.getenv('GOOGLE_OAUTH_CLIENT_ID'),
            'client_secret': os.getenv('GOOGLE_OAUTH_CLIENT_SECRET'),
            'redirect_uri': 'http://localhost:8501/auth/callback',
            'grant_type': 'authorization_code'
        }
        
        token_response = requests.post(token_url, data=token_data)
        tokens = token_response.json()
        
        if 'access_token' in tokens:
            # 사용자 정보 가져오기
            user_info_url = "https://www.googleapis.com/oauth2/v2/userinfo"
            headers = {'Authorization': f"Bearer {tokens['access_token']}"}
            user_response = requests.get(user_info_url, headers=headers)
            user_info = user_response.json()
            
            # 소셜 로그인 처리
            success, message, user_data = auth_manager.social_login(
                provider='google',
                email=user_info.get('email'),
                name=user_info.get('name'),
                profile_picture=user_info.get('picture'),
                oauth_id=user_info.get('id')
            )
            
            if success:
                st.session_state.authenticated = True
                st.session_state.user = user_data
                st.session_state.pop('oauth_state', None)
                
                ui.show_success(f"환영합니다, {user_data['name']}님! 🎉")
                time.sleep(1)
                st.switch_page("pages/1_📊_Dashboard.py")
            else:
                ui.show_error(message)
        else:
            ui.show_error("Google 로그인에 실패했습니다.")
            
    except Exception as e:
        handle_error(e, "Google OAuth 처리 중 오류가 발생했습니다.", ErrorSeverity.WARNING)
    finally:
        st.session_state.pop('oauth_state', None)

def process_github_oauth_callback(code: str, state: str):
    """GitHub OAuth 콜백 처리"""
    try:
        import requests
        
        # State 검증
        expected_state = st.session_state.get('oauth_state', '').replace('github_', '')
        if state != expected_state:
            ui.show_error("보안 검증 실패: 잘못된 상태 토큰")
            return
        
        # 액세스 토큰 요청
        token_url = "https://github.com/login/oauth/access_token"
        token_data = {
            'client_id': os.getenv('GITHUB_CLIENT_ID'),
            'client_secret': os.getenv('GITHUB_CLIENT_SECRET'),
            'code': code,
            'state': state
        }
        headers = {'Accept': 'application/json'}
        
        token_response = requests.post(token_url, data=token_data, headers=headers)
        tokens = token_response.json()
        
        if 'access_token' in tokens:
            # 사용자 정보 가져오기
            user_url = "https://api.github.com/user"
            headers = {'Authorization': f"token {tokens['access_token']}"}
            user_response = requests.get(user_url, headers=headers)
            user_info = user_response.json()
            
            # 이메일 가져오기
            email_url = "https://api.github.com/user/emails"
            email_response = requests.get(email_url, headers=headers)
            emails = email_response.json()
            primary_email = next((e['email'] for e in emails if e['primary']), None)
            
            # 소셜 로그인 처리
            success, message, user_data = auth_manager.social_login(
                provider='github',
                email=primary_email or f"{user_info['login']}@github.local",
                name=user_info.get('name') or user_info['login'],
                profile_picture=user_info.get('avatar_url'),
                oauth_id=str(user_info['id'])
            )
            
            if success:
                st.session_state.authenticated = True
                st.session_state.user = user_data
                st.session_state.pop('oauth_state', None)
                
                ui.show_success(f"환영합니다, {user_data['name']}님! 🐙")
                time.sleep(1)
                st.switch_page("pages/1_📊_Dashboard.py")
            else:
                ui.show_error(message)
        else:
            ui.show_error(f"GitHub 로그인 실패: {tokens.get('error_description', '알 수 없는 오류')}")
            
    except Exception as e:
        handle_error(e, "GitHub OAuth 처리 중 오류가 발생했습니다.", ErrorSeverity.WARNING)
    finally:
        st.session_state.pop('oauth_state', None)

def handle_password_reset_email(email: str, csrf_token: str):
    """이메일로 비밀번호 재설정"""
    # CSRF 검증
    if not verify_csrf_token(csrf_token):
        ui.show_error("보안 토큰이 유효하지 않습니다. 페이지를 새로고침하세요.")
        return
    
    # 이메일 검증
    valid, error_msg = validate_email(email)
    if not valid:
        ui.show_error(error_msg)
        return
    
    with st.spinner("처리 중..."):
        try:
            success, message = auth_manager.send_password_reset_email(email)
            
            if success:
                ui.show_success(message)
                time.sleep(2)
                st.session_state.current_tab = AuthTab.LOGIN.value
                st.rerun()
            else:
                ui.show_error(message)
                
        except Exception as e:
            handle_error(e, "비밀번호 재설정 이메일 발송 중 오류가 발생했습니다.", ErrorSeverity.WARNING)

def handle_password_reset_security(email: str, security_answer: str, csrf_token: str):
    """보안 질문으로 비밀번호 재설정"""
    # CSRF 검증
    if not verify_csrf_token(csrf_token):
        ui.show_error("보안 토큰이 유효하지 않습니다. 페이지를 새로고침하세요.")
        return
    
    # 입력값 검증
    valid, error_msg = validate_email(email)
    if not valid:
        ui.show_error(error_msg)
        return
    
    if not security_answer:
        ui.show_error("보안 답변을 입력해주세요.")
        return
    
    with st.spinner("처리 중..."):
        try:
            success, message = auth_manager.reset_password_with_security(
                email=email,
                security_answer=security_answer
            )
            
            if success:
                # 임시 비밀번호 표시 또는 재설정 폼 표시
                ui.show_success(message)
                st.info("새로운 비밀번호로 로그인해주세요.")
                time.sleep(3)
                st.session_state.current_tab = AuthTab.LOGIN.value
                st.rerun()
            else:
                ui.show_error(message)
                
        except Exception as e:
            handle_error(e, "비밀번호 재설정 중 오류가 발생했습니다.", ErrorSeverity.WARNING)

def handle_email_verification(code: str, csrf_token: str):
    """이메일 인증 처리"""
    # CSRF 검증
    if not verify_csrf_token(csrf_token):
        ui.show_error("보안 토큰이 유효하지 않습니다. 페이지를 새로고침하세요.")
        return
    
    if not code or len(code) != 6:
        ui.show_error("올바른 인증 코드를 입력해주세요.")
        return
    
    with st.spinner("인증 확인 중..."):
        try:
            email = st.session_state.verification_email
            success, message = auth_manager.verify_email(email, code)
            
            if success:
                ui.show_success("✅ 이메일 인증이 완료되었습니다!")
                st.session_state.verification_email = None
                st.session_state.current_tab = AuthTab.LOGIN.value
                time.sleep(2)
                st.rerun()
            else:
                ui.show_error(message)
                
        except Exception as e:
            handle_error(e, "이메일 인증 중 오류가 발생했습니다.", ErrorSeverity.WARNING)

def handle_resend_verification(csrf_token: str):
    """인증 코드 재전송"""
    # CSRF 검증
    if not verify_csrf_token(csrf_token):
        ui.show_error("보안 토큰이 유효하지 않습니다. 페이지를 새로고침하세요.")
        return
    
    with st.spinner("인증 코드 재전송 중..."):
        try:
            email = st.session_state.verification_email
            success, message = auth_manager.resend_verification_email(email)
            
            if success:
                ui.show_success("📧 인증 코드를 다시 전송했습니다.")
            else:
                ui.show_error(message)
                
        except Exception as e:
            handle_error(e, "인증 코드 재전송 중 오류가 발생했습니다.", ErrorSeverity.WARNING)

def show_terms_modal():
    """이용약관 모달"""
    with st.expander("서비스 이용약관", expanded=True):
        st.markdown("""
        ### Universal DOE Platform 이용약관
        
        **제 1조 (목적)**
        이 약관은 Universal DOE Platform(이하 "서비스")의 이용에 관한 조건 및 절차, 
        이용자와 회사의 권리, 의무 및 책임사항을 규정함을 목적으로 합니다.
        
        **제 2조 (정의)**
        1. "서비스"란 회사가 제공하는 AI 기반 실험 설계 플랫폼 및 관련 서비스를 의미합니다.
        2. "회원"이란 이 약관에 동의하고 회원가입을 한 자를 의미합니다.
        3. "게스트"란 회원가입 없이 제한된 기능을 이용하는 자를 의미합니다.
        4. "콘텐츠"란 서비스 내에서 생성, 저장, 공유되는 모든 데이터를 의미합니다.
        
        **제 3조 (약관의 효력 및 변경)**
        1. 이 약관은 서비스를 이용하고자 하는 모든 이용자에게 적용됩니다.
        2. 회사는 필요한 경우 약관을 변경할 수 있으며, 변경사항은 서비스 내 공지합니다.
        3. 변경된 약관은 공지 후 7일 이후부터 효력이 발생합니다.
        
        **제 4조 (회원가입)**
        1. 회원가입은 이용자가 약관에 동의하고 가입 양식을 작성하여 신청합니다.
        2. 회사는 이용자의 신청을 승낙함으로써 회원가입이 완료됩니다.
        3. 다음의 경우 가입을 거절하거나 취소할 수 있습니다:
           - 타인의 정보를 도용한 경우
           - 허위 정보를 기재한 경우
           - 서비스 운영을 방해한 이력이 있는 경우
        
        **제 5조 (서비스 이용)**
        1. 서비스는 연중무휴 24시간 이용이 원칙입니다.
        2. 시스템 점검, 통신 장애 등 불가피한 경우 서비스가 중단될 수 있습니다.
        3. 회원은 서비스를 본래의 목적에 맞게 이용해야 합니다.
        
        **제 6조 (개인정보보호)**
        1. 회사는 개인정보보호법에 따라 회원의 개인정보를 보호합니다.
        2. 개인정보의 수집, 이용, 제공에 관한 사항은 개인정보처리방침에 따릅니다.
        
        **제 7조 (회원의 의무)**
        1. 회원은 다음 행위를 하여서는 안 됩니다:
           - 타인의 정보 도용
           - 서비스의 안정적 운영 방해
           - 저작권 등 지적재산권 침해
           - 기타 법령에 위반되는 행위
        
        **제 8조 (저작권)**
        1. 서비스에 대한 저작권은 회사에 귀속됩니다.
        2. 회원이 생성한 콘텐츠의 저작권은 회원에게 있습니다.
        3. 회원은 서비스 이용을 위해 필요한 범위 내에서 회사에 사용권을 부여합니다.
        
        **제 9조 (면책조항)**
        1. 회사는 천재지변, 전쟁 등 불가항력으로 인한 서비스 중단에 책임지지 않습니다.
        2. 회원 간 또는 회원과 제3자 간의 분쟁에 대해 회사는 책임지지 않습니다.
        
        **제 10조 (분쟁해결)**
        1. 서비스 이용과 관련한 분쟁은 대한민국 법령에 따라 해결합니다.
        2. 소송이 제기될 경우 회사 소재지 관할 법원을 전속 관할로 합니다.
        
        **부칙**
        이 약관은 2024년 1월 1일부터 시행됩니다.
        """)

def show_privacy_modal():
    """개인정보처리방침 모달"""
    with st.expander("개인정보처리방침", expanded=True):
        st.markdown("""
        ### Universal DOE Platform 개인정보처리방침
        
        Universal DOE Platform(이하 "회사")은 이용자의 개인정보를 중요시하며, 
        개인정보보호법 등 관련 법령을 준수하고 있습니다.
        
        **1. 수집하는 개인정보 항목**
        
        가. 필수 항목
        - 회원가입: 이메일, 비밀번호, 이름
        - 소셜 로그인: 이메일, 이름, 프로필 사진(선택)
        
        나. 선택 항목
        - 소속 기관, 연구 분야, 연락처
        
        다. 자동 수집 항목
        - IP 주소, 쿠키, 서비스 이용 기록, 접속 로그
        
        **2. 개인정보의 수집 및 이용 목적**
        
        가. 회원 관리
        - 회원제 서비스 제공, 본인 확인, 불량 회원 방지
        
        나. 서비스 제공
        - 실험 설계 도구 제공, 데이터 분석, 협업 기능
        
        다. 서비스 개선
        - 신규 서비스 개발, 맞춤형 서비스 제공, 통계 분석
        
        **3. 개인정보의 보유 및 이용 기간**
        
        가. 회원 정보
        - 회원 탈퇴 시까지
        - 단, 법령에 따른 보관 의무가 있는 경우 해당 기간 동안 보관
        
        나. 법령에 따른 보관
        - 계약 또는 청약철회 기록: 5년
        - 대금결제 및 재화 공급 기록: 5년
        - 소비자 불만 또는 분쟁 처리 기록: 3년
        
        **4. 개인정보의 파기**
        
        가. 파기 절차
        - 이용 목적 달성 후 내부 방침에 따라 일정 기간 저장 후 파기
        
        나. 파기 방법
        - 전자적 파일: 복구 불가능한 방법으로 영구 삭제
        - 종이 문서: 분쇄기로 분쇄 또는 소각
        
        **5. 개인정보의 제3자 제공**
        
        회사는 원칙적으로 이용자의 개인정보를 외부에 제공하지 않습니다.
        다만, 다음의 경우는 예외로 합니다:
        - 이용자의 사전 동의가 있는 경우
        - 법령의 규정에 의한 경우
        
        **6. 개인정보의 위탁**
        
        회사는 서비스 제공을 위해 다음과 같이 개인정보를 위탁할 수 있습니다:
        - 클라우드 서비스: 데이터 저장 및 처리
        - 이메일 발송 서비스: 회원 인증 및 알림
        
        **7. 이용자의 권리**
        
        가. 개인정보 열람권
        - 자신의 개인정보 처리 현황을 열람할 수 있습니다.
        
        나. 개인정보 정정·삭제권
        - 잘못된 정보의 정정 또는 삭제를 요구할 수 있습니다.
        
        다. 개인정보 처리 정지권
        - 개인정보 처리 정지를 요구할 수 있습니다.
        
        **8. 개인정보의 안전성 확보 조치**
        
        가. 기술적 조치
        - 개인정보 암호화 (bcrypt)
        - 해킹 방지 시스템 운영
        - 정기적인 보안 점검
        
        나. 관리적 조치
        - 개인정보 접근 권한 제한
        - 개인정보보호 교육 실시
        
        **9. 개인정보보호 책임자**
        
        - 성명: [담당자명]
        - 이메일: privacy@universaldoe.com
        - 전화: 02-0000-0000
        
        **10. 개인정보처리방침의 변경**
        
        이 개인정보처리방침은 2024년 1월 1일부터 적용됩니다.
        변경사항은 서비스 내 공지사항을 통해 고지합니다.
        """)

def render_footer():
    """푸터 렌더링"""
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.9rem;'>
        <p>© 2024 Universal DOE Platform. All rights reserved.</p>
        <p>
            <a href='#' style='color: #667eea; text-decoration: none; margin: 0 10px;'>도움말</a> |
            <a href='#' style='color: #667eea; text-decoration: none; margin: 0 10px;'>문의하기</a> |
            <a href='#' style='color: #667eea; text-decoration: none; margin: 0 10px;'>개인정보처리방침</a> |
            <a href='#' style='color: #667eea; text-decoration: none; margin: 0 10px;'>이용약관</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

def main():
    """메인 함수"""
    # 테마 적용
    apply_theme()
    
    # 세션 상태 초기화
    initialize_session_state()
    
    # 세션 타임아웃 체크
    check_session_timeout()
    
    # OAuth 콜백 처리
    handle_oauth_callback()
    
    # 이미 로그인된 경우 대시보드로 리다이렉트
    if st.session_state.get('authenticated', False) and not st.session_state.get('guest_mode', False):
        st.switch_page("pages/1_📊_Dashboard.py")
        return
    
    # 헤더 렌더링
    render_header()
    
    # 현재 탭에 따라 콘텐츠 렌더링
    current_tab = st.session_state.get('current_tab', AuthTab.LOGIN.value)
    
    if current_tab == AuthTab.LOGIN.value:
        render_login_form()
        render_social_login()
    elif current_tab == AuthTab.SIGNUP.value:
        render_signup_form()
    elif current_tab == AuthTab.FORGOT.value:
        render_forgot_password_form()
    elif current_tab == AuthTab.VERIFY.value:
        render_email_verification_form()
    
    # 푸터 렌더링
    render_footer()

# 실행
if __name__ == "__main__":
    main()
