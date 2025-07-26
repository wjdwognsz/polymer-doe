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
from typing import Optional, Dict, Any
import bcrypt
import json
import os

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
    from utils.error_handler import handle_error
    from config.app_config import SECURITY_CONFIG, APP_INFO
    from config.theme_config import apply_theme
    from config.error_config import ERROR_CODES
except ImportError as e:
    st.error(f"필수 모듈을 찾을 수 없습니다: {e}")
    st.info("프로젝트 루트에서 실행 중인지 확인해주세요.")
    st.stop()

# 전역 인스턴스
auth_manager = get_auth_manager()
ui = get_common_ui()
db_manager = get_database_manager()

# 상수 정의
MAX_LOGIN_ATTEMPTS = SECURITY_CONFIG.get('max_login_attempts', 5)
LOCKOUT_DURATION = timedelta(minutes=SECURITY_CONFIG.get('lockout_minutes', 30))
PASSWORD_MIN_LENGTH = SECURITY_CONFIG.get('password_min_length', 8)

# UI 텍스트
UI_TEXTS = {
    'ko': {
        'login_title': '🔐 로그인',
        'signup_title': '👤 회원가입',
        'forgot_password': '🔑 비밀번호 찾기',
        'email_placeholder': '이메일 주소를 입력하세요',
        'password_placeholder': '비밀번호를 입력하세요',
        'name_placeholder': '이름을 입력하세요',
        'organization_placeholder': '소속 기관 (선택)',
        'login_button': '로그인',
        'signup_button': '회원가입',
        'reset_button': '비밀번호 재설정',
        'guest_button': '게스트로 둘러보기',
        'login_keep': '로그인 상태 유지',
        'terms_agree': '이용약관에 동의합니다',
        'privacy_agree': '개인정보처리방침에 동의합니다'
    },
    'en': {
        'login_title': '🔐 Login',
        'signup_title': '👤 Sign Up',
        'forgot_password': '🔑 Forgot Password',
        'email_placeholder': 'Enter your email',
        'password_placeholder': 'Enter your password',
        'name_placeholder': 'Enter your name',
        'organization_placeholder': 'Organization (optional)',
        'login_button': 'Login',
        'signup_button': 'Sign Up',
        'reset_button': 'Reset Password',
        'guest_button': 'Browse as Guest',
        'login_keep': 'Keep me logged in',
        'terms_agree': 'I agree to the Terms of Service',
        'privacy_agree': 'I agree to the Privacy Policy'
    }
}

def initialize_session_state():
    """세션 상태 초기화"""
    defaults = {
        'auth_tab': 'login',
        'login_attempts': {},
        'show_password': False,
        'verification_pending': False,
        'temp_email': None,
        'language': 'ko',
        'theme': 'light'
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def get_ui_text(key: str) -> str:
    """현재 언어에 맞는 UI 텍스트 반환"""
    lang = st.session_state.get('language', 'ko')
    return UI_TEXTS[lang].get(key, key)

def validate_email(email: str) -> bool:
    """이메일 형식 검증"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password: str) -> Dict[str, Any]:
    """비밀번호 강도 검증"""
    result = {
        'valid': True,
        'score': 0,
        'feedback': []
    }
    
    # 최소 길이
    if len(password) < PASSWORD_MIN_LENGTH:
        result['feedback'].append(f"최소 {PASSWORD_MIN_LENGTH}자 이상")
        result['valid'] = False
    else:
        result['score'] += 1
    
    # 대문자
    if not re.search(r'[A-Z]', password):
        result['feedback'].append("대문자 포함 필요")
    else:
        result['score'] += 1
    
    # 소문자
    if not re.search(r'[a-z]', password):
        result['feedback'].append("소문자 포함 필요")
    else:
        result['score'] += 1
    
    # 숫자
    if not re.search(r'\d', password):
        result['feedback'].append("숫자 포함 필요")
    else:
        result['score'] += 1
    
    # 특수문자
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        result['feedback'].append("특수문자 포함 필요")
    else:
        result['score'] += 1
    
    # 길이 보너스
    if len(password) >= 12:
        result['score'] += 1
    
    # 강도 레벨
    if result['score'] <= 2:
        result['strength'] = 'weak'
        result['color'] = 'red'
    elif result['score'] <= 4:
        result['strength'] = 'medium'
        result['color'] = 'orange'
    else:
        result['strength'] = 'strong'
        result['color'] = 'green'
    
    return result

def check_lockout(email: str) -> bool:
    """계정 잠금 상태 확인"""
    attempts = st.session_state.login_attempts.get(email, {})
    if attempts.get('count', 0) >= MAX_LOGIN_ATTEMPTS:
        last_attempt = attempts.get('last_attempt')
        if last_attempt:
            lockout_end = last_attempt + LOCKOUT_DURATION
            if datetime.now() < lockout_end:
                return True
            else:
                # 잠금 해제
                st.session_state.login_attempts[email] = {'count': 0}
    return False

def record_failed_attempt(email: str):
    """실패한 로그인 시도 기록"""
    if email not in st.session_state.login_attempts:
        st.session_state.login_attempts[email] = {'count': 0}
    
    st.session_state.login_attempts[email]['count'] += 1
    st.session_state.login_attempts[email]['last_attempt'] = datetime.now()

def render_login_tab():
    """로그인 탭 렌더링"""
    with st.form("login_form", clear_on_submit=False):
        st.markdown(f"### {get_ui_text('login_title')}")
        
        # 이메일 입력
        email = st.text_input(
            "이메일",
            placeholder=get_ui_text('email_placeholder'),
            key="login_email",
            help="가입하신 이메일 주소를 입력하세요"
        )
        
        # 비밀번호 입력
        col1, col2 = st.columns([5, 1])
        with col1:
            password = st.text_input(
                "비밀번호",
                type="password" if not st.session_state.show_password else "text",
                placeholder=get_ui_text('password_placeholder'),
                key="login_password"
            )
        with col2:
            st.write("")  # 간격 맞추기
            if st.button("👁️" if not st.session_state.show_password else "🙈"):
                st.session_state.show_password = not st.session_state.show_password
                st.rerun()
        
        # 추가 옵션
        col1, col2 = st.columns(2)
        with col1:
            remember_me = st.checkbox(get_ui_text('login_keep'), value=True)
        with col2:
            st.empty()  # 정렬용
        
        # 로그인 버튼
        submitted = st.form_submit_button(
            get_ui_text('login_button'),
            use_container_width=True,
            type="primary"
        )
        
        if submitted:
            handle_login(email, password, remember_me)
    
    # 추가 버튼들
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("비밀번호 찾기", use_container_width=True):
            st.session_state.auth_tab = 'forgot'
            st.rerun()
    
    with col2:
        if st.button("회원가입", use_container_width=True):
            st.session_state.auth_tab = 'signup'
            st.rerun()
    
    with col3:
        if st.button(get_ui_text('guest_button'), use_container_width=True):
            handle_guest_login()
    
    # 소셜 로그인 (온라인 시)
    if check_online_status():
        st.markdown("---")
        st.markdown("### 소셜 로그인")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔷 Google로 로그인", use_container_width=True):
                handle_google_login()
        
        with col2:
            if st.button("🐙 GitHub로 로그인", use_container_width=True):
                handle_github_login()

def render_signup_tab():
    """회원가입 탭 렌더링"""
    st.markdown(f"### {get_ui_text('signup_title')}")
    
    with st.form("signup_form", clear_on_submit=False):
        # 기본 정보
        st.markdown("#### 📝 기본 정보")
        
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input(
                "이름 *",
                placeholder=get_ui_text('name_placeholder'),
                help="실명을 입력해주세요"
            )
        
        with col2:
            email = st.text_input(
                "이메일 *",
                placeholder=get_ui_text('email_placeholder'),
                help="로그인에 사용할 이메일 주소"
            )
        
        organization = st.text_input(
            "소속 기관",
            placeholder=get_ui_text('organization_placeholder'),
            help="선택사항입니다"
        )
        
        # 비밀번호
        st.markdown("#### 🔒 비밀번호 설정")
        
        col1, col2 = st.columns(2)
        with col1:
            password = st.text_input(
                "비밀번호 *",
                type="password",
                help=f"최소 {PASSWORD_MIN_LENGTH}자 이상, 영문/숫자/특수문자 포함"
            )
            
            # 비밀번호 강도 표시
            if password:
                pw_result = validate_password(password)
                progress = pw_result['score'] / 6.0
                st.progress(progress)
                
                color_map = {'weak': '🔴', 'medium': '🟡', 'strong': '🟢'}
                st.write(f"비밀번호 강도: {color_map.get(pw_result['strength'], '')} {pw_result['strength']}")
                
                if pw_result['feedback']:
                    with st.expander("개선 사항"):
                        for feedback in pw_result['feedback']:
                            st.write(f"• {feedback}")
        
        with col2:
            password_confirm = st.text_input(
                "비밀번호 확인 *",
                type="password",
                help="동일한 비밀번호를 다시 입력하세요"
            )
        
        # 약관 동의
        st.markdown("#### 📋 약관 동의")
        
        col1, col2 = st.columns(2)
        with col1:
            terms_agree = st.checkbox(get_ui_text('terms_agree'))
            if st.button("이용약관 보기", key="view_terms"):
                show_terms_modal()
        
        with col2:
            privacy_agree = st.checkbox(get_ui_text('privacy_agree'))
            if st.button("개인정보처리방침 보기", key="view_privacy"):
                show_privacy_modal()
        
        # 가입 버튼
        submitted = st.form_submit_button(
            get_ui_text('signup_button'),
            use_container_width=True,
            type="primary"
        )
        
        if submitted:
            handle_signup(name, email, organization, password, password_confirm, 
                        terms_agree, privacy_agree)
    
    # 로그인으로 돌아가기
    if st.button("← 로그인으로 돌아가기"):
        st.session_state.auth_tab = 'login'
        st.rerun()

def render_forgot_password_tab():
    """비밀번호 찾기 탭 렌더링"""
    st.markdown(f"### {get_ui_text('forgot_password')}")
    
    # 온라인/오프라인 확인
    is_online = check_online_status()
    
    if is_online:
        st.info("📧 가입하신 이메일로 비밀번호 재설정 링크를 보내드립니다.")
        
        with st.form("forgot_form"):
            email = st.text_input(
                "이메일",
                placeholder=get_ui_text('email_placeholder'),
                help="가입하신 이메일 주소를 입력하세요"
            )
            
            submitted = st.form_submit_button(
                "재설정 링크 전송",
                use_container_width=True,
                type="primary"
            )
            
            if submitted:
                handle_password_reset(email, method='email')
    else:
        st.warning("🔌 오프라인 모드에서는 보안 질문으로 비밀번호를 재설정합니다.")
        
        with st.form("security_question_form"):
            email = st.text_input(
                "이메일",
                placeholder=get_ui_text('email_placeholder')
            )
            
            security_answer = st.text_input(
                "가입 시 설정한 보안 질문의 답변",
                placeholder="예: 첫 반려동물의 이름"
            )
            
            submitted = st.form_submit_button(
                "비밀번호 재설정",
                use_container_width=True,
                type="primary"
            )
            
            if submitted:
                handle_password_reset(email, method='security', answer=security_answer)
    
    # 로그인으로 돌아가기
    if st.button("← 로그인으로 돌아가기"):
        st.session_state.auth_tab = 'login'
        st.rerun()

def handle_login(email: str, password: str, remember_me: bool):
    """로그인 처리"""
    # 입력값 검증
    if not email or not password:
        ui.show_error("이메일과 비밀번호를 모두 입력해주세요.")
        return
    
    if not validate_email(email):
        ui.show_error("올바른 이메일 형식이 아닙니다.")
        return
    
    # 계정 잠금 확인
    if check_lockout(email):
        remaining_time = LOCKOUT_DURATION.seconds // 60
        ui.show_error(f"너무 많은 로그인 시도로 계정이 잠겼습니다. {remaining_time}분 후 다시 시도해주세요.")
        return
    
    # 로그인 시도
    with st.spinner("로그인 중..."):
        try:
            success, message, user_info = auth_manager.login(email, password, remember_me)
            
            if success:
                # 세션 설정
                st.session_state.authenticated = True
                st.session_state.user = user_info
                st.session_state.login_attempts.pop(email, None)
                
                # 환영 메시지 (애니메이션 효과)
                placeholder = st.empty()
                for i in range(3):
                    placeholder.success(f"환영합니다, {user_info['name']}님! {'🎉' * (i+1)}")
                    time.sleep(0.3)
                
                # 대시보드로 이동
                time.sleep(0.5)
                st.switch_page("pages/1_📊_Dashboard.py")
            else:
                # 로그인 실패
                record_failed_attempt(email)
                attempts_left = MAX_LOGIN_ATTEMPTS - st.session_state.login_attempts[email]['count']
                
                if attempts_left > 0:
                    ui.show_error(f"{message} (남은 시도: {attempts_left}회)")
                else:
                    ui.show_error(f"계정이 잠겼습니다. {LOCKOUT_DURATION.seconds//60}분 후 다시 시도해주세요.")
                    
        except Exception as e:
            handle_error(e, "로그인 처리 중 오류가 발생했습니다.")

def handle_signup(name: str, email: str, organization: str, password: str, 
                 password_confirm: str, terms_agree: bool, privacy_agree: bool):
    """회원가입 처리"""
    # 입력값 검증
    errors = []
    
    if not name:
        errors.append("이름을 입력해주세요.")
    
    if not email:
        errors.append("이메일을 입력해주세요.")
    elif not validate_email(email):
        errors.append("올바른 이메일 형식이 아닙니다.")
    
    if not password:
        errors.append("비밀번호를 입력해주세요.")
    else:
        pw_result = validate_password(password)
        if not pw_result['valid']:
            errors.extend(pw_result['feedback'])
    
    if password != password_confirm:
        errors.append("비밀번호가 일치하지 않습니다.")
    
    if not terms_agree:
        errors.append("이용약관에 동의해주세요.")
    
    if not privacy_agree:
        errors.append("개인정보처리방침에 동의해주세요.")
    
    if errors:
        for error in errors:
            ui.show_error(f"• {error}")
        return
    
    # 회원가입 처리
    with st.spinner("회원가입 처리 중..."):
        try:
            success, message, user_id = auth_manager.register(
                email=email,
                password=password,
                name=name,
                organization=organization
            )
            
            if success:
                # 성공 메시지
                st.balloons()
                ui.show_success("🎉 회원가입이 완료되었습니다!")
                
                # 이메일 인증 필요 시
                if check_online_status():
                    st.session_state.verification_pending = True
                    st.session_state.temp_email = email
                    ui.show_info("📧 이메일 인증을 위해 메일을 확인해주세요.")
                    time.sleep(2)
                
                # 로그인 탭으로 이동
                st.session_state.auth_tab = 'login'
                st.rerun()
            else:
                ui.show_error(message)
                
        except Exception as e:
            handle_error(e, "회원가입 처리 중 오류가 발생했습니다.")

def handle_guest_login():
    """게스트 로그인 처리"""
    with st.spinner("게스트 모드로 접속 중..."):
        # 게스트 세션 생성
        st.session_state.authenticated = True
        st.session_state.guest_mode = True
        st.session_state.user = {
            'id': f'guest_{datetime.now().timestamp()}',
            'name': '게스트',
            'email': 'guest@local',
            'role': 'guest',
            'permissions': ['read_only', 'local_save']
        }
        
        ui.show_info("📚 게스트 모드로 접속했습니다. 일부 기능이 제한됩니다.")
        time.sleep(1)
        
        # 대시보드로 이동
        st.switch_page("pages/1_📊_Dashboard.py")

def handle_google_login():
    """Google OAuth 로그인 실제 구현"""
    import webbrowser
    from urllib.parse import urlencode
    
    # Google OAuth 설정 (config에서 가져오기)
    client_id = os.getenv('GOOGLE_OAUTH_CLIENT_ID')
    redirect_uri = "http://localhost:8501/auth/callback"
    
    if not client_id:
        ui.show_error("Google OAuth가 설정되지 않았습니다. 관리자에게 문의하세요.")
        return
    
    # OAuth URL 생성
    oauth_params = {
        'client_id': client_id,
        'redirect_uri': redirect_uri,
        'response_type': 'code',
        'scope': 'openid email profile',
        'access_type': 'offline',
        'prompt': 'consent'
    }
    
    auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(oauth_params)}"
    
    # 세션에 상태 저장
    st.session_state.oauth_state = 'google_pending'
    
    # 사용자를 Google 로그인 페이지로 리다이렉트
    st.markdown(f"""
    <meta http-equiv="refresh" content="0; url={auth_url}">
    <script>window.location.href = "{auth_url}";</script>
    """, unsafe_allow_html=True)
    
    ui.show_info("Google 로그인 페이지로 이동합니다...")

def handle_github_login():
    """GitHub OAuth 로그인 실제 구현"""
    from urllib.parse import urlencode
    import secrets
    
    # GitHub OAuth 설정
    client_id = os.getenv('GITHUB_CLIENT_ID')
    
    if not client_id:
        ui.show_error("GitHub OAuth가 설정되지 않았습니다. 관리자에게 문의하세요.")
        return
    
    # 상태 토큰 생성 (CSRF 방지)
    state = secrets.token_urlsafe(32)
    st.session_state.oauth_state = state
    
    # OAuth URL 생성
    oauth_params = {
        'client_id': client_id,
        'redirect_uri': "http://localhost:8501/auth/github/callback",
        'scope': 'user:email',
        'state': state
    }
    
    auth_url = f"https://github.com/login/oauth/authorize?{urlencode(oauth_params)}"
    
    # GitHub 로그인 페이지로 리다이렉트
    st.markdown(f"""
    <meta http-equiv="refresh" content="0; url={auth_url}">
    <script>window.location.href = "{auth_url}";</script>
    """, unsafe_allow_html=True)
    
    ui.show_info("GitHub 로그인 페이지로 이동합니다...")

def handle_oauth_callback():
    """OAuth 콜백 처리 (URL 파라미터 확인)"""
    # URL 파라미터 가져오기
    params = st.experimental_get_query_params()
    
    if 'code' in params and 'state' in params:
        code = params['code'][0]
        state = params['state'][0] if 'state' in params else None
        
        # Google OAuth 콜백
        if st.session_state.get('oauth_state') == 'google_pending':
            process_google_callback(code)
        
        # GitHub OAuth 콜백
        elif st.session_state.get('oauth_state') == state:
            process_github_callback(code, state)
    
    elif 'error' in params:
        ui.show_error(f"OAuth 로그인 실패: {params['error'][0]}")
        st.session_state.pop('oauth_state', None)

def process_google_callback(code: str):
    """Google OAuth 콜백 처리"""
    import requests
    
    try:
        # 액세스 토큰 요청
        token_url = "https://oauth2.googleapis.com/token"
        token_data = {
            'code': code,
            'client_id': os.getenv('GOOGLE_OAUTH_CLIENT_ID'),
            'client_secret': os.getenv('GOOGLE_OAUTH_CLIENT_SECRET'),
            'redirect_uri': "http://localhost:8501/auth/callback",
            'grant_type': 'authorization_code'
        }
        
        token_response = requests.post(token_url, data=token_data)
        token_json = token_response.json()
        
        if 'access_token' in token_json:
            # 사용자 정보 가져오기
            user_info_url = "https://www.googleapis.com/oauth2/v2/userinfo"
            headers = {'Authorization': f"Bearer {token_json['access_token']}"}
            user_response = requests.get(user_info_url, headers=headers)
            user_info = user_response.json()
            
            # 사용자 등록 또는 로그인
            email = user_info.get('email')
            name = user_info.get('name')
            picture = user_info.get('picture')
            
            # auth_manager를 통해 소셜 로그인 처리
            success, message, user_data = auth_manager.social_login(
                provider='google',
                email=email,
                name=name,
                profile_picture=picture,
                oauth_id=user_info.get('id')
            )
            
            if success:
                st.session_state.authenticated = True
                st.session_state.user = user_data
                st.session_state.pop('oauth_state', None)
                
                # URL 파라미터 제거
                st.experimental_set_query_params()
                
                ui.show_success(f"환영합니다, {name}님! 🎉")
                time.sleep(1)
                st.switch_page("pages/1_📊_Dashboard.py")
            else:
                ui.show_error(message)
        else:
            ui.show_error("Google 로그인에 실패했습니다.")
            
    except Exception as e:
        handle_error(e, "Google OAuth 처리 중 오류가 발생했습니다.")
    finally:
        st.session_state.pop('oauth_state', None)

def process_github_callback(code: str, state: str):
    """GitHub OAuth 콜백 처리"""
    import requests
    
    # CSRF 검증
    if state != st.session_state.get('oauth_state'):
        ui.show_error("보안 검증 실패: 잘못된 상태 토큰")
        return
    
    try:
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
        token_json = token_response.json()
        
        if 'access_token' in token_json:
            # 사용자 정보 가져오기
            user_url = "https://api.github.com/user"
            headers = {'Authorization': f"token {token_json['access_token']}"}
            user_response = requests.get(user_url, headers=headers)
            user_info = user_response.json()
            
            # 이메일 가져오기 (별도 요청 필요)
            email_url = "https://api.github.com/user/emails"
            email_response = requests.get(email_url, headers=headers)
            emails = email_response.json()
            primary_email = next((e['email'] for e in emails if e['primary']), None)
            
            # 사용자 등록 또는 로그인
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
                
                # URL 파라미터 제거
                st.experimental_set_query_params()
                
                ui.show_success(f"환영합니다, {user_data['name']}님! 🐙")
                time.sleep(1)
                st.switch_page("pages/1_📊_Dashboard.py")
            else:
                ui.show_error(message)
        else:
            ui.show_error(f"GitHub 로그인 실패: {token_json.get('error_description', 'Unknown error')}")
            
    except Exception as e:
        handle_error(e, "GitHub OAuth 처리 중 오류가 발생했습니다.")
    finally:
        st.session_state.pop('oauth_state', None)

def handle_password_reset(email: str, method: str = 'email', answer: str = None):
    """비밀번호 재설정 처리"""
    if not email:
        ui.show_error("이메일을 입력해주세요.")
        return
    
    if not validate_email(email):
        ui.show_error("올바른 이메일 형식이 아닙니다.")
        return
    
    with st.spinner("처리 중..."):
        try:
            if method == 'email':
                # 이메일로 재설정 링크 전송
                success, message = auth_manager.send_password_reset_email(email)
            else:
                # 보안 질문으로 재설정
                success, message = auth_manager.reset_password_with_security(email, answer)
            
            if success:
                ui.show_success(message)
                time.sleep(2)
                st.session_state.auth_tab = 'login'
                st.rerun()
            else:
                ui.show_error(message)
                
        except Exception as e:
            handle_error(e, "비밀번호 재설정 중 오류가 발생했습니다.")

def check_online_status() -> bool:
    """온라인 상태 확인"""
    try:
        import requests
        response = requests.get('https://www.google.com', timeout=3)
        return response.status_code == 200
    except:
        return False

def show_terms_modal():
    """이용약관 모달"""
    with st.expander("서비스 이용약관", expanded=True):
        st.markdown("""
        ### Universal DOE Platform 이용약관
        
        **제 1조 (목적)**
        이 약관은 Universal DOE Platform(이하 "서비스")의 이용에 관한 조건 및 절차를 규정함을 목적으로 합니다.
        
        **제 2조 (정의)**
        1. "서비스"란 회사가 제공하는 실험 설계 플랫폼을 의미합니다.
        2. "회원"이란 이 약관에 동의하고 회원가입을 한 자를 의미합니다.
        3. "게스트"란 회원가입 없이 제한된 기능을 이용하는 자를 의미합니다.
        
        **제 3조 (약관의 효력 및 변경)**
        1. 이 약관은 서비스를 이용하고자 하는 모든 회원에게 적용됩니다.
        2. 회사는 필요한 경우 약관을 변경할 수 있으며, 변경사항은 서비스 내 공지합니다.
        
        [이하 약관 내용...]
        """)

def show_privacy_modal():
    """개인정보처리방침 모달"""
    with st.expander("개인정보처리방침", expanded=True):
        st.markdown("""
        ### Universal DOE Platform 개인정보처리방침
        
        **1. 수집하는 개인정보**
        - 필수: 이메일, 비밀번호, 이름
        - 선택: 소속 기관, 연구 분야
        
        **2. 개인정보의 이용목적**
        - 회원 관리 및 본인 확인
        - 서비스 제공 및 개선
        - 중요한 공지사항 전달
        
        **3. 개인정보의 보관 및 파기**
        - 회원 탈퇴 시 즉시 파기
        - 단, 법령에 따라 보관이 필요한 경우 해당 기간 동안 보관
        
        **4. 개인정보의 제3자 제공**
        - 원칙적으로 외부에 제공하지 않음
        - 단, 법령에 의한 경우 예외
        
        [이하 개인정보처리방침 내용...]
        """)

def main():
    """메인 함수"""
    # 테마 적용
    apply_theme()
    
    # 세션 상태 초기화
    initialize_session_state()
    
    # OAuth 콜백 처리 (URL에 code 파라미터가 있는 경우)
    handle_oauth_callback()
    
    # 이미 로그인된 경우 대시보드로 리다이렉트
    if st.session_state.get('authenticated', False):
        st.switch_page("pages/1_📊_Dashboard.py")
        return
    
    # 헤더
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='font-size: 3rem; font-weight: bold; 
                   background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   -webkit-background-clip: text;
                   -webkit-text-fill-color: transparent;
                   margin-bottom: 0.5rem;'>
            Universal DOE Platform
        </h1>
        <p style='font-size: 1.2rem; color: #666;'>
            모든 과학자를 위한 AI 기반 실험 설계 플랫폼
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 탭 선택
    tabs = ["로그인", "회원가입", "비밀번호 찾기"]
    tab_map = {'login': 0, 'signup': 1, 'forgot': 2}
    
    selected_tab = st.tabs(tabs)[tab_map.get(st.session_state.auth_tab, 0)]
    
    # 탭 컨테이너
    tab1, tab2, tab3 = st.tabs(tabs)
    
    with tab1:
        if st.session_state.auth_tab == 'login':
            render_login_tab()
    
    with tab2:
        if st.session_state.auth_tab == 'signup':
            render_signup_tab()
    
    with tab3:
        if st.session_state.auth_tab == 'forgot':
            render_forgot_password_tab()
    
    # 푸터
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.9rem;'>
        <p>© 2024 Universal DOE Platform. All rights reserved.</p>
        <p>
            <a href='#' style='color: #667eea; text-decoration: none; margin: 0 10px;'>도움말</a> |
            <a href='#' style='color: #667eea; text-decoration: none; margin: 0 10px;'>문의하기</a> |
            <a href='#' style='color: #667eea; text-decoration: none; margin: 0 10px;'>개인정보처리방침</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

# 실행
if __name__ == "__main__":
    main()
