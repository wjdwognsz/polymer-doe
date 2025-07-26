"""
🔐 Login Page - Universal DOE Platform
===========================================================================
데스크톱 애플리케이션을 위한 인증 페이지
- 오프라인 우선 설계 (로컬 SQLite DB 사용)
- 선택적 클라우드 동기화
- Streamlit Pages 자동 라우팅 활용
===========================================================================
"""

import streamlit as st

# 페이지 설정 (최상단에서 호출)
st.set_page_config(
    page_title="Login - Universal DOE",
    page_icon="🔐",
    layout="centered",
    initial_sidebar_state="collapsed"
)

import re
import time
import json
import bcrypt
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
import logging
from pathlib import Path

# 로컬 모듈 임포트
try:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    
    from utils.database_manager import DatabaseManager
    from utils.auth_manager import AuthManager
    from utils.common_ui import (
        render_header, show_success, show_error, show_warning, show_info,
        render_loading_spinner, render_empty_state
    )
    from config.app_config import SECURITY_CONFIG, SESSION_CONFIG, APP_INFO
    from config.local_config import LOCAL_CONFIG
except ImportError as e:
    st.error(f"모듈 임포트 오류: {e}")
    st.stop()

# ===========================================================================
# 🔧 설정 및 상수
# ===========================================================================

logger = logging.getLogger(__name__)

# 인증 관련 상수
MAX_LOGIN_ATTEMPTS = SECURITY_CONFIG.get('max_login_attempts', 5)
LOCKOUT_DURATION = SECURITY_CONFIG.get('lockout_duration', timedelta(minutes=30))
MIN_PASSWORD_LENGTH = SECURITY_CONFIG.get('password_min_length', 8)

# UI 텍스트
TEXTS = {
    'login': {
        'title': '🔐 로그인',
        'subtitle': 'Universal DOE Platform에 오신 것을 환영합니다',
        'email': '이메일',
        'password': '비밀번호',
        'remember': '로그인 상태 유지',
        'forgot': '비밀번호를 잊으셨나요?',
        'no_account': '계정이 없으신가요?',
        'signup_link': '회원가입',
        'guest': '🔍 게스트로 둘러보기'
    },
    'signup': {
        'title': '👤 회원가입',
        'subtitle': '새 계정을 만들어 모든 기능을 이용하세요',
        'name': '이름',
        'email': '이메일',
        'password': '비밀번호',
        'password_confirm': '비밀번호 확인',
        'organization': '소속 기관 (선택)',
        'agree': '이용약관 및 개인정보처리방침에 동의합니다',
        'already': '이미 계정이 있으신가요?',
        'login_link': '로그인'
    },
    'reset': {
        'title': '🔑 비밀번호 재설정',
        'subtitle': '새로운 비밀번호를 설정하세요',
        'email': '가입하신 이메일을 입력하세요',
        'security_question': '보안 질문',
        'security_answer': '답변',
        'new_password': '새 비밀번호',
        'back': '로그인으로 돌아가기'
    }
}

# 비밀번호 강도 레벨
PASSWORD_STRENGTH = {
    0: {'label': '매우 약함', 'color': 'red', 'progress': 0.2},
    1: {'label': '약함', 'color': 'orange', 'progress': 0.4},
    2: {'label': '보통', 'color': 'yellow', 'progress': 0.6},
    3: {'label': '강함', 'color': 'green', 'progress': 0.8},
    4: {'label': '매우 강함', 'color': 'blue', 'progress': 1.0}
}

# ===========================================================================
# 🔐 인증 관련 함수
# ===========================================================================

def get_auth_manager() -> AuthManager:
    """AuthManager 인스턴스 가져오기"""
    if 'auth_manager' not in st.session_state:
        db_path = LOCAL_CONFIG['database']['path']
        db_manager = DatabaseManager(db_path)
        st.session_state.auth_manager = AuthManager(db_manager)
    return st.session_state.auth_manager

def init_session_state():
    """세션 상태 초기화"""
    defaults = {
        'auth_mode': 'login',  # login, signup, reset
        'authenticated': False,
        'user': None,
        'user_email': None,
        'login_attempts': {},  # {email: {'count': int, 'last_attempt': datetime}}
        'temp_data': {},  # 임시 데이터 저장
        'show_password': False,
        'signup_step': 1,  # 회원가입 단계
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def validate_email(email: str) -> bool:
    """이메일 형식 검증"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def check_password_strength(password: str) -> Dict:
    """비밀번호 강도 확인"""
    score = 0
    feedback = []
    
    # 길이 체크
    if len(password) >= MIN_PASSWORD_LENGTH:
        score += 1
    else:
        feedback.append(f"최소 {MIN_PASSWORD_LENGTH}자 이상")
    
    # 대문자 포함
    if re.search(r'[A-Z]', password):
        score += 1
    else:
        feedback.append("대문자 포함 필요")
    
    # 소문자 포함
    if re.search(r'[a-z]', password):
        score += 1
    else:
        feedback.append("소문자 포함 필요")
    
    # 숫자 포함
    if re.search(r'\d', password):
        score += 1
    else:
        feedback.append("숫자 포함 필요")
    
    # 특수문자 포함
    if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        score += 1
    else:
        feedback.append("특수문자 포함 필요")
    
    # 결과 반환
    strength_level = min(score, 4)
    return {
        'score': strength_level,
        'level': PASSWORD_STRENGTH[strength_level],
        'feedback': feedback,
        'is_valid': score >= 3  # 최소 '보통' 이상
    }

def is_account_locked(email: str) -> Tuple[bool, Optional[int]]:
    """계정 잠금 상태 확인"""
    attempts = st.session_state.login_attempts.get(email, {})
    
    if attempts.get('count', 0) >= MAX_LOGIN_ATTEMPTS:
        last_attempt = attempts.get('last_attempt')
        if last_attempt:
            time_passed = datetime.now() - last_attempt
            if time_passed < LOCKOUT_DURATION:
                remaining = LOCKOUT_DURATION - time_passed
                return True, int(remaining.total_seconds() / 60)
            else:
                # 잠금 해제
                st.session_state.login_attempts[email] = {'count': 0}
    
    return False, None

def record_login_attempt(email: str, success: bool):
    """로그인 시도 기록"""
    if email not in st.session_state.login_attempts:
        st.session_state.login_attempts[email] = {'count': 0}
    
    if success:
        # 성공 시 카운트 초기화
        st.session_state.login_attempts[email] = {'count': 0}
    else:
        # 실패 시 카운트 증가
        st.session_state.login_attempts[email]['count'] += 1
        st.session_state.login_attempts[email]['last_attempt'] = datetime.now()

# ===========================================================================
# 🎨 UI 렌더링 함수
# ===========================================================================

def render_login_form():
    """로그인 폼 렌더링"""
    st.markdown(f"### {TEXTS['login']['title']}")
    st.markdown(f"*{TEXTS['login']['subtitle']}*")
    
    # 로그인 폼
    with st.form("login_form", clear_on_submit=False):
        email = st.text_input(
            TEXTS['login']['email'],
            placeholder="user@example.com",
            help="가입하신 이메일 주소를 입력하세요"
        )
        
        password = st.text_input(
            TEXTS['login']['password'],
            type="password",
            placeholder="••••••••",
            help="비밀번호를 입력하세요"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            remember_me = st.checkbox(TEXTS['login']['remember'], value=True)
        with col2:
            if st.button(TEXTS['login']['forgot'], type="secondary"):
                st.session_state.auth_mode = 'reset'
                st.rerun()
        
        # 로그인 버튼
        login_submitted = st.form_submit_button(
            "🔓 로그인",
            use_container_width=True,
            type="primary"
        )
        
        if login_submitted:
            handle_login(email, password, remember_me)
    
    # 추가 옵션
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{TEXTS['login']['no_account']}**")
        if st.button(TEXTS['login']['signup_link'], use_container_width=True):
            st.session_state.auth_mode = 'signup'
            st.rerun()
    
    with col2:
        st.markdown("**데모 체험**")
        if st.button(TEXTS['login']['guest'], use_container_width=True):
            handle_guest_login()

def render_signup_form():
    """회원가입 폼 렌더링"""
    st.markdown(f"### {TEXTS['signup']['title']}")
    st.markdown(f"*{TEXTS['signup']['subtitle']}*")
    
    # 진행 상태 표시
    progress = st.session_state.signup_step / 3
    st.progress(progress, text=f"단계 {st.session_state.signup_step}/3")
    
    with st.form("signup_form", clear_on_submit=False):
        # Step 1: 기본 정보
        if st.session_state.signup_step == 1:
            st.markdown("#### 📝 기본 정보")
            
            name = st.text_input(
                TEXTS['signup']['name'],
                placeholder="홍길동",
                help="실명을 입력해주세요"
            )
            
            email = st.text_input(
                TEXTS['signup']['email'],
                placeholder="user@example.com",
                help="로그인에 사용할 이메일 주소"
            )
            
            organization = st.text_input(
                TEXTS['signup']['organization'],
                placeholder="○○대학교 / ○○연구소",
                help="선택사항입니다"
            )
            
            if st.form_submit_button("다음 단계 →", use_container_width=True):
                if validate_signup_step1(name, email):
                    st.session_state.temp_data.update({
                        'name': name,
                        'email': email,
                        'organization': organization
                    })
                    st.session_state.signup_step = 2
                    st.rerun()
        
        # Step 2: 비밀번호 설정
        elif st.session_state.signup_step == 2:
            st.markdown("#### 🔒 비밀번호 설정")
            
            password = st.text_input(
                TEXTS['signup']['password'],
                type="password",
                placeholder="••••••••",
                help=f"최소 {MIN_PASSWORD_LENGTH}자, 대소문자/숫자/특수문자 포함"
            )
            
            # 비밀번호 강도 표시
            if password:
                strength = check_password_strength(password)
                st.progress(
                    strength['level']['progress'],
                    text=f"비밀번호 강도: {strength['level']['label']}"
                )
                if strength['feedback']:
                    st.warning("개선사항: " + ", ".join(strength['feedback']))
            
            password_confirm = st.text_input(
                TEXTS['signup']['password_confirm'],
                type="password",
                placeholder="••••••••"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("← 이전", use_container_width=True):
                    st.session_state.signup_step = 1
                    st.rerun()
            with col2:
                if st.form_submit_button("다음 단계 →", use_container_width=True):
                    if validate_signup_step2(password, password_confirm):
                        st.session_state.temp_data['password'] = password
                        st.session_state.signup_step = 3
                        st.rerun()
        
        # Step 3: 약관 동의 및 완료
        else:
            st.markdown("#### ✅ 약관 동의")
            
            # 약관 내용 표시
            with st.expander("이용약관 및 개인정보처리방침"):
                st.markdown(get_terms_and_conditions())
            
            agree = st.checkbox(TEXTS['signup']['agree'])
            
            # 보안 질문 설정 (오프라인 비밀번호 재설정용)
            st.markdown("#### 🔐 보안 질문 (비밀번호 찾기용)")
            security_questions = [
                "가장 좋아하는 음식은?",
                "첫 반려동물의 이름은?",
                "어머니의 성함은?",
                "출신 초등학교는?",
                "가장 좋아하는 영화는?"
            ]
            
            selected_question = st.selectbox(
                "보안 질문 선택",
                security_questions
            )
            
            security_answer = st.text_input(
                "답변",
                type="password",
                help="비밀번호를 잊어버렸을 때 필요합니다"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("← 이전", use_container_width=True):
                    st.session_state.signup_step = 2
                    st.rerun()
            with col2:
                if st.form_submit_button("🎉 가입 완료", use_container_width=True, type="primary"):
                    if validate_signup_step3(agree, security_answer):
                        st.session_state.temp_data.update({
                            'security_question': selected_question,
                            'security_answer': security_answer
                        })
                        handle_signup()
    
    # 로그인 링크
    st.divider()
    st.markdown(f"**{TEXTS['signup']['already']}**")
    if st.button(TEXTS['signup']['login_link'], use_container_width=True):
        st.session_state.auth_mode = 'login'
        st.session_state.signup_step = 1
        st.rerun()

def render_reset_password_form():
    """비밀번호 재설정 폼 렌더링"""
    st.markdown(f"### {TEXTS['reset']['title']}")
    st.markdown(f"*{TEXTS['reset']['subtitle']}*")
    
    with st.form("reset_form"):
        # Step 1: 이메일 입력
        if 'reset_email_verified' not in st.session_state:
            email = st.text_input(
                TEXTS['reset']['email'],
                placeholder="user@example.com"
            )
            
            if st.form_submit_button("다음 →", use_container_width=True):
                auth_manager = get_auth_manager()
                user = auth_manager.get_user_by_email(email)
                
                if user:
                    st.session_state.reset_email = email
                    st.session_state.reset_user = user
                    st.session_state.reset_email_verified = True
                    st.rerun()
                else:
                    show_error("등록되지 않은 이메일입니다.")
        
        # Step 2: 보안 질문 답변 및 새 비밀번호
        else:
            user = st.session_state.reset_user
            
            st.info(f"이메일: {st.session_state.reset_email}")
            
            # 보안 질문 표시
            st.markdown(f"**{TEXTS['reset']['security_question']}**")
            st.markdown(user.get('security_question', '보안 질문이 설정되지 않았습니다'))
            
            security_answer = st.text_input(
                TEXTS['reset']['security_answer'],
                type="password"
            )
            
            st.divider()
            
            new_password = st.text_input(
                TEXTS['reset']['new_password'],
                type="password",
                help=f"최소 {MIN_PASSWORD_LENGTH}자, 대소문자/숫자/특수문자 포함"
            )
            
            new_password_confirm = st.text_input(
                "새 비밀번호 확인",
                type="password"
            )
            
            if st.form_submit_button("비밀번호 재설정", use_container_width=True, type="primary"):
                if validate_reset_password(security_answer, new_password, new_password_confirm):
                    handle_reset_password(new_password)
    
    # 로그인으로 돌아가기
    st.divider()
    if st.button(f"← {TEXTS['reset']['back']}", use_container_width=True):
        st.session_state.auth_mode = 'login'
        # 재설정 관련 세션 정리
        for key in ['reset_email', 'reset_user', 'reset_email_verified']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# ===========================================================================
# 🔧 핸들러 함수
# ===========================================================================

def handle_login(email: str, password: str, remember_me: bool):
    """로그인 처리"""
    # 입력값 검증
    if not email or not password:
        show_error("이메일과 비밀번호를 모두 입력해주세요.")
        return
    
    if not validate_email(email):
        show_error("올바른 이메일 형식이 아닙니다.")
        return
    
    # 계정 잠금 확인
    is_locked, remaining_minutes = is_account_locked(email)
    if is_locked:
        show_error(f"너무 많은 시도로 계정이 잠겼습니다. {remaining_minutes}분 후 다시 시도해주세요.")
        return
    
    # 로그인 시도
    auth_manager = get_auth_manager()
    
    with st.spinner("로그인 중..."):
        result = auth_manager.authenticate(email, password)
    
    if result['success']:
        # 로그인 성공
        record_login_attempt(email, True)
        
        # 세션 설정
        st.session_state.authenticated = True
        st.session_state.user = result['user']['name']
        st.session_state.user_email = email
        st.session_state.user_data = result['user']
        
        # 로그인 유지 설정
        if remember_me:
            st.session_state.remember_token = auth_manager.create_remember_token(email)
        
        show_success(f"환영합니다, {result['user']['name']}님! 🎉")
        time.sleep(1)
        
        # 대시보드로 이동
        st.switch_page("pages/1_📊_Dashboard.py")
    else:
        # 로그인 실패
        record_login_attempt(email, False)
        
        attempts = st.session_state.login_attempts.get(email, {})
        remaining = MAX_LOGIN_ATTEMPTS - attempts.get('count', 0)
        
        if remaining > 0:
            show_error(f"이메일 또는 비밀번호가 올바르지 않습니다. (남은 시도: {remaining}회)")
        else:
            show_error(f"계정이 잠겼습니다. {LOCKOUT_DURATION.total_seconds()//60}분 후 다시 시도해주세요.")

def handle_guest_login():
    """게스트 로그인 처리"""
    st.session_state.authenticated = True
    st.session_state.user = "Guest"
    st.session_state.user_email = "guest@demo.com"
    st.session_state.user_data = {
        'name': 'Guest User',
        'role': 'guest',
        'permissions': ['view_demo', 'use_basic_features']
    }
    
    show_info("게스트 모드로 입장합니다. 일부 기능이 제한될 수 있습니다.")
    time.sleep(1)
    
    st.switch_page("pages/1_📊_Dashboard.py")

def handle_signup():
    """회원가입 처리"""
    auth_manager = get_auth_manager()
    user_data = st.session_state.temp_data
    
    with st.spinner("계정을 생성하고 있습니다..."):
        # 비밀번호 해싱
        hashed_password = bcrypt.hashpw(
            user_data['password'].encode('utf-8'),
            bcrypt.gensalt()
        ).decode('utf-8')
        
        # 보안 답변 해싱
        hashed_answer = bcrypt.hashpw(
            user_data['security_answer'].encode('utf-8'),
            bcrypt.gensalt()
        ).decode('utf-8')
        
        # 사용자 생성
        result = auth_manager.create_user(
            email=user_data['email'],
            password=hashed_password,
            name=user_data['name'],
            organization=user_data.get('organization', ''),
            security_question=user_data['security_question'],
            security_answer=hashed_answer
        )
    
    if result['success']:
        show_success("🎉 회원가입이 완료되었습니다! 로그인해주세요.")
        
        # 임시 데이터 정리
        st.session_state.temp_data = {}
        st.session_state.signup_step = 1
        st.session_state.auth_mode = 'login'
        
        time.sleep(2)
        st.rerun()
    else:
        show_error(f"회원가입 실패: {result.get('message', '알 수 없는 오류')}")

def handle_reset_password(new_password: str):
    """비밀번호 재설정 처리"""
    auth_manager = get_auth_manager()
    email = st.session_state.reset_email
    
    with st.spinner("비밀번호를 재설정하고 있습니다..."):
        # 새 비밀번호 해싱
        hashed_password = bcrypt.hashpw(
            new_password.encode('utf-8'),
            bcrypt.gensalt()
        ).decode('utf-8')
        
        # 비밀번호 업데이트
        result = auth_manager.update_password(email, hashed_password)
    
    if result['success']:
        show_success("비밀번호가 성공적으로 재설정되었습니다. 새 비밀번호로 로그인해주세요.")
        
        # 재설정 관련 세션 정리
        for key in ['reset_email', 'reset_user', 'reset_email_verified']:
            if key in st.session_state:
                del st.session_state[key]
        
        st.session_state.auth_mode = 'login'
        time.sleep(2)
        st.rerun()
    else:
        show_error("비밀번호 재설정에 실패했습니다.")

# ===========================================================================
# 🔍 검증 함수
# ===========================================================================

def validate_signup_step1(name: str, email: str) -> bool:
    """회원가입 1단계 검증"""
    if not name or not name.strip():
        show_error("이름을 입력해주세요.")
        return False
    
    if not email or not validate_email(email):
        show_error("올바른 이메일 주소를 입력해주세요.")
        return False
    
    # 이메일 중복 확인
    auth_manager = get_auth_manager()
    if auth_manager.get_user_by_email(email):
        show_error("이미 사용 중인 이메일입니다.")
        return False
    
    return True

def validate_signup_step2(password: str, password_confirm: str) -> bool:
    """회원가입 2단계 검증"""
    if not password:
        show_error("비밀번호를 입력해주세요.")
        return False
    
    strength = check_password_strength(password)
    if not strength['is_valid']:
        show_error("비밀번호가 보안 요구사항을 충족하지 않습니다.")
        return False
    
    if password != password_confirm:
        show_error("비밀번호가 일치하지 않습니다.")
        return False
    
    return True

def validate_signup_step3(agree: bool, security_answer: str) -> bool:
    """회원가입 3단계 검증"""
    if not agree:
        show_error("이용약관에 동의해주세요.")
        return False
    
    if not security_answer or not security_answer.strip():
        show_error("보안 질문의 답변을 입력해주세요.")
        return False
    
    return True

def validate_reset_password(security_answer: str, new_password: str, confirm: str) -> bool:
    """비밀번호 재설정 검증"""
    user = st.session_state.reset_user
    
    # 보안 답변 확인
    stored_answer = user.get('security_answer', '')
    if not bcrypt.checkpw(security_answer.encode('utf-8'), stored_answer.encode('utf-8')):
        show_error("보안 질문의 답변이 일치하지 않습니다.")
        return False
    
    # 새 비밀번호 검증
    if not new_password:
        show_error("새 비밀번호를 입력해주세요.")
        return False
    
    strength = check_password_strength(new_password)
    if not strength['is_valid']:
        show_error("비밀번호가 보안 요구사항을 충족하지 않습니다.")
        return False
    
    if new_password != confirm:
        show_error("새 비밀번호가 일치하지 않습니다.")
        return False
    
    return True

# ===========================================================================
# 📄 기타 함수
# ===========================================================================

def get_terms_and_conditions() -> str:
    """이용약관 텍스트 반환"""
    return """
    ### Universal DOE Platform 이용약관
    
    **제1조 (목적)**
    이 약관은 Universal DOE Platform(이하 "서비스")의 이용과 관련하여 필요한 사항을 규정함을 목적으로 합니다.
    
    **제2조 (개인정보보호)**
    1. 서비스는 사용자의 개인정보를 안전하게 보호합니다.
    2. 수집된 정보는 서비스 제공 목적으로만 사용됩니다.
    3. 사용자 동의 없이 제3자에게 제공하지 않습니다.
    
    **제3조 (데이터 보안)**
    1. 모든 데이터는 로컬에 암호화되어 저장됩니다.
    2. 클라우드 동기화는 선택사항이며, 사용자가 직접 제어할 수 있습니다.
    
    **제4조 (사용자의 의무)**
    1. 사용자는 본인의 계정 정보를 안전하게 관리해야 합니다.
    2. 타인의 정보를 도용하거나 부정하게 사용해서는 안 됩니다.
    
    [이하 약관 내용...]
    """

# ===========================================================================
# 🎯 메인 함수
# ===========================================================================

def main():
    """메인 실행 함수"""
    # 세션 초기화
    init_session_state()
    
    # 이미 로그인된 경우
    if st.session_state.authenticated:
        st.info("이미 로그인되어 있습니다.")
        if st.button("대시보드로 이동"):
            st.switch_page("pages/1_📊_Dashboard.py")
        if st.button("로그아웃"):
            # 로그아웃 처리
            for key in ['authenticated', 'user', 'user_email', 'user_data']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        return
    
    # 사이드바에 정보 표시
    with st.sidebar:
        st.markdown("### 📌 도움말")
        st.info(
            "**데모 계정**\n"
            "- Email: demo@test.com\n"
            "- Password: Demo1234!\n\n"
            "또는 '게스트로 둘러보기'를 클릭하세요."
        )
        
        st.markdown("### 🔒 보안 정보")
        st.success(
            "✅ 모든 데이터는 로컬에 저장\n"
            "✅ 비밀번호는 bcrypt로 암호화\n"
            "✅ 오프라인에서도 작동"
        )
    
    # 인증 모드에 따라 렌더링
    auth_mode = st.session_state.auth_mode
    
    if auth_mode == 'signup':
        render_signup_form()
    elif auth_mode == 'reset':
        render_reset_password_form()
    else:  # 기본: login
        render_login_form()

if __name__ == "__main__":
    main()
