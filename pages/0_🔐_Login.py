"""
🔐 로그인 페이지 - Universal DOE Platform
=============================================================================
데스크톱 앱용 오프라인 우선 인증 페이지
SQLite 로컬 DB 기반, 선택적 클라우드 동기화 지원
=============================================================================
"""

import streamlit as st
import sys
from pathlib import Path
import logging
import re
import secrets
import string
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
import json
import base64
from io import BytesIO
from PIL import Image

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

# 로컬 모듈
try:
    from utils.database_manager import get_database_manager
    from utils.auth_manager import get_auth_manager, UserRole
    from utils.common_ui import get_common_ui
    from config.app_config import SECURITY_CONFIG, APP_INFO, AI_EXPLANATION_CONFIG
    from config.local_config import LOCAL_CONFIG
    from config.offline_config import OFFLINE_CONFIG
except ImportError as e:
    st.error(f"🚨 필수 모듈을 찾을 수 없습니다: {str(e)}")
    st.info("프로젝트 루트에서 'streamlit run polymer_platform.py'로 실행하세요.")
    st.stop()

# 로깅 설정
logger = logging.getLogger(__name__)

# 페이지 설정
st.set_page_config(
    page_title="로그인 - Universal DOE Platform",
    page_icon="🔐",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# 🎨 커스텀 CSS
# =============================================================================
CUSTOM_CSS = """
<style>
    /* 로그인 폼 스타일 */
    .auth-container {
        max-width: 450px;
        margin: 0 auto;
        padding: 2rem;
        background: white;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* 로고 스타일 */
    .app-logo {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .app-logo h1 {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    /* 탭 스타일 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
    }
    
    /* 입력 필드 스타일 */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e1e4e8;
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* 버튼 스타일 */
    .stButton > button {
        width: 100%;
        padding: 0.75rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* 비밀번호 강도 표시기 */
    .password-strength {
        margin-top: 0.5rem;
        padding: 0.5rem;
        border-radius: 6px;
        font-size: 0.875rem;
        text-align: center;
    }
    
    .password-weak { background: #fee; color: #c33; }
    .password-fair { background: #ffe; color: #a60; }
    .password-good { background: #efe; color: #060; }
    .password-strong { background: #dfd; color: #040; }
    
    /* 알림 스타일 */
    .notification {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 8px;
        border-left: 4px solid;
    }
    
    .notification-info {
        background: #e3f2fd;
        border-color: #2196f3;
        color: #1565c0;
    }
    
    .notification-success {
        background: #e8f5e9;
        border-color: #4caf50;
        color: #2e7d32;
    }
    
    .notification-warning {
        background: #fff3e0;
        border-color: #ff9800;
        color: #e65100;
    }
    
    .notification-error {
        background: #ffebee;
        border-color: #f44336;
        color: #c62828;
    }
    
    /* 오프라인 배지 */
    .offline-badge {
        display: inline-block;
        background: #ff9800;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    
    .online-badge {
        display: inline-block;
        background: #4caf50;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
</style>
"""

# =============================================================================
# 🔧 유틸리티 함수
# =============================================================================

def init_session_state():
    """세션 상태 초기화"""
    defaults = {
        'authenticated': False,
        'user': None,
        'user_id': None,
        'user_role': None,
        'auth_token': None,
        'login_attempts': {},
        'temp_email': None,
        'verification_pending': False,
        'show_ai_details': AI_EXPLANATION_CONFIG.get('default_show', False),
        'online_status': False,
        'last_online_check': datetime.now()
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def check_online_status() -> bool:
    """온라인 상태 확인 (캐시된 결과 사용)"""
    # 마지막 확인으로부터 30초 경과 시 재확인
    if datetime.now() - st.session_state.last_online_check > timedelta(seconds=30):
        import requests
        try:
            response = requests.get('https://www.google.com', timeout=3)
            st.session_state.online_status = response.status_code == 200
        except:
            st.session_state.online_status = False
        st.session_state.last_online_check = datetime.now()
    
    return st.session_state.online_status

def validate_email(email: str) -> bool:
    """이메일 형식 검증"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password_strength(password: str) -> Dict[str, any]:
    """비밀번호 강도 검증"""
    score = 0
    feedback = []
    
    # 길이 체크
    if len(password) >= SECURITY_CONFIG['password']['min_length']:
        score += 1
    else:
        feedback.append(f"최소 {SECURITY_CONFIG['password']['min_length']}자 이상")
    
    # 대문자
    if re.search(r'[A-Z]', password):
        score += 1
    else:
        feedback.append("대문자 포함 필요")
    
    # 소문자
    if re.search(r'[a-z]', password):
        score += 1
    else:
        feedback.append("소문자 포함 필요")
    
    # 숫자
    if re.search(r'\d', password):
        score += 1
    else:
        feedback.append("숫자 포함 필요")
    
    # 특수문자
    if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        score += 1
    else:
        feedback.append("특수문자 포함 필요")
    
    # 추가 보너스
    if len(password) >= 12:
        score += 1
    
    # 강도 결정
    if score <= 2:
        strength = 'weak'
    elif score <= 3:
        strength = 'fair'
    elif score <= 4:
        strength = 'good'
    else:
        strength = 'strong'
    
    return {
        'score': score,
        'strength': strength,
        'feedback': feedback,
        'valid': score >= 4
    }

def generate_verification_code() -> str:
    """6자리 인증 코드 생성"""
    return ''.join(secrets.choice(string.digits) for _ in range(6))

def render_password_strength_indicator(password: str):
    """비밀번호 강도 표시"""
    if not password:
        return
    
    result = validate_password_strength(password)
    strength = result['strength']
    
    # 프로그레스 바
    progress = result['score'] / 6.0
    st.progress(progress)
    
    # 강도 텍스트
    strength_text = {
        'weak': '🔴 매우 약함',
        'fair': '🟠 약함',
        'good': '🟡 보통',
        'strong': '🟢 강함'
    }
    
    st.markdown(f"""
        <div class="password-strength password-{strength}">
            {strength_text[strength]}
        </div>
    """, unsafe_allow_html=True)
    
    # 피드백
    if result['feedback']:
        with st.expander("💡 비밀번호 강화 방법"):
            for feedback in result['feedback']:
                st.write(f"• {feedback}")

# =============================================================================
# 🔐 인증 함수
# =============================================================================

def handle_login(email: str, password: str, remember: bool = False) -> bool:
    """로그인 처리"""
    auth_manager = get_auth_manager()
    
    # 로그인 시도
    result = auth_manager.login(email, password, remember)
    
    if result['success']:
        # 세션 상태 업데이트
        st.session_state.authenticated = True
        st.session_state.user = result['user']
        st.session_state.user_id = result['user']['id']
        st.session_state.user_role = result['user']['role']
        st.session_state.auth_token = result['token']
        
        # 환영 메시지
        st.success(f"🎉 환영합니다, {result['user']['name']}님!")
        
        # 대시보드로 이동
        st.switch_page("pages/1_📊_Dashboard.py")
        return True
    else:
        # 에러 처리
        error_msg = result.get('error', '로그인에 실패했습니다.')
        remaining_attempts = result.get('remaining_attempts')
        
        if remaining_attempts is not None and remaining_attempts > 0:
            error_msg += f" (남은 시도: {remaining_attempts}회)"
        elif remaining_attempts == 0:
            error_msg = "🔒 너무 많은 시도로 계정이 일시적으로 잠겼습니다. 15분 후 다시 시도해주세요."
        
        st.error(error_msg)
        return False

def handle_signup(user_data: Dict[str, any]) -> bool:
    """회원가입 처리"""
    auth_manager = get_auth_manager()
    
    # 회원가입
    result = auth_manager.register(user_data)
    
    if result['success']:
        st.success("🎉 회원가입이 완료되었습니다!")
        
        # 자동 로그인
        if handle_login(user_data['email'], user_data['password'], True):
            return True
        else:
            st.info("회원가입은 완료되었습니다. 로그인해주세요.")
            st.session_state.auth_mode = 'login'
            return False
    else:
        st.error(f"회원가입 실패: {result.get('error', '알 수 없는 오류')}")
        return False

def handle_password_reset(email: str) -> bool:
    """비밀번호 재설정 처리"""
    auth_manager = get_auth_manager()
    
    # 오프라인 모드에서는 보안 질문으로 처리
    if not check_online_status():
        st.info("오프라인 모드: 보안 질문을 통해 비밀번호를 재설정합니다.")
        
        # 보안 질문 확인
        security_question = auth_manager.get_security_question(email)
        if security_question:
            answer = st.text_input(f"보안 질문: {security_question}")
            if st.button("확인"):
                if auth_manager.verify_security_answer(email, answer):
                    # 새 비밀번호 설정
                    new_password = st.text_input("새 비밀번호", type="password")
                    confirm_password = st.text_input("새 비밀번호 확인", type="password")
                    
                    if new_password and new_password == confirm_password:
                        if auth_manager.reset_password(email, new_password):
                            st.success("✅ 비밀번호가 변경되었습니다.")
                            st.session_state.auth_mode = 'login'
                            st.rerun()
                        else:
                            st.error("비밀번호 변경에 실패했습니다.")
                    elif new_password != confirm_password:
                        st.error("비밀번호가 일치하지 않습니다.")
                else:
                    st.error("보안 답변이 올바르지 않습니다.")
        else:
            st.error("등록된 이메일을 찾을 수 없습니다.")
    else:
        # 온라인 모드: 이메일로 재설정 링크 발송
        result = auth_manager.send_password_reset_email(email)
        if result['success']:
            st.success("📧 비밀번호 재설정 링크를 이메일로 발송했습니다.")
            st.info("이메일을 확인하고 링크를 클릭하여 비밀번호를 재설정하세요.")
        else:
            st.error(f"이메일 발송 실패: {result.get('error', '알 수 없는 오류')}")
    
    return True

# =============================================================================
# 🎨 UI 렌더링 함수
# =============================================================================

def render_header():
    """헤더 렌더링"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # 로고 및 제목
    st.markdown("""
        <div class="app-logo">
            <h1>🧬 Universal DOE</h1>
            <p style="color: #666; font-size: 1.1rem;">모든 연구자를 위한 AI 실험 설계 플랫폼</p>
        </div>
    """, unsafe_allow_html=True)
    
    # 온라인 상태 표시
    if check_online_status():
        st.markdown('<span class="online-badge">🟢 온라인</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="offline-badge">🔴 오프라인</span>', unsafe_allow_html=True)

def render_login_form():
    """로그인 폼 렌더링"""
    with st.form("login_form", clear_on_submit=False):
        st.subheader("🔐 로그인")
        
        # 이메일
        email = st.text_input(
            "이메일",
            placeholder="your@email.com",
            help="가입 시 사용한 이메일 주소"
        )
        
        # 비밀번호
        password = st.text_input(
            "비밀번호",
            type="password",
            placeholder="••••••••",
            help="대소문자, 숫자, 특수문자 포함 8자 이상"
        )
        
        # 옵션
        col1, col2 = st.columns(2)
        with col1:
            remember = st.checkbox("로그인 상태 유지", value=True)
        with col2:
            if st.button("비밀번호를 잊으셨나요?", type="secondary"):
                st.session_state.auth_mode = 'forgot'
                st.rerun()
        
        # 로그인 버튼
        submitted = st.form_submit_button(
            "🚀 로그인",
            use_container_width=True,
            type="primary"
        )
        
        if submitted:
            if not email or not password:
                st.error("이메일과 비밀번호를 모두 입력해주세요.")
            elif not validate_email(email):
                st.error("올바른 이메일 형식이 아닙니다.")
            else:
                handle_login(email, password, remember)
    
    # 추가 옵션
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🆕 회원가입", use_container_width=True):
            st.session_state.auth_mode = 'signup'
            st.rerun()
    
    with col2:
        if st.button("👀 게스트로 둘러보기", use_container_width=True):
            st.session_state.authenticated = True
            st.session_state.user = {
                'id': 'guest',
                'name': '게스트',
                'email': 'guest@universaldoe.com',
                'role': UserRole.GUEST
            }
            st.session_state.user_role = UserRole.GUEST
            st.switch_page("pages/1_📊_Dashboard.py")
    
    # 소셜 로그인 (온라인 시)
    if check_online_status():
        st.divider()
        st.markdown("### 🌐 소셜 로그인")
        
        if st.button("🔵 Google로 로그인", use_container_width=True):
            auth_manager = get_auth_manager()
            auth_url = auth_manager.get_google_auth_url()
            st.markdown(f'<a href="{auth_url}" target="_self">Google 계정으로 로그인하기</a>', unsafe_allow_html=True)

def render_signup_form():
    """회원가입 폼 렌더링"""
    st.subheader("👤 회원가입")
    
    with st.form("signup_form", clear_on_submit=False):
        # 기본 정보
        col1, col2 = st.columns(2)
        with col1:
            first_name = st.text_input("이름", placeholder="길동")
        with col2:
            last_name = st.text_input("성", placeholder="홍")
        
        # 이메일
        email = st.text_input(
            "이메일",
            placeholder="your@email.com",
            help="로그인 시 사용할 이메일 주소"
        )
        
        # 비밀번호
        password = st.text_input(
            "비밀번호",
            type="password",
            placeholder="••••••••",
            help="대소문자, 숫자, 특수문자 포함 8자 이상"
        )
        
        # 비밀번호 강도 표시
        if password:
            render_password_strength_indicator(password)
        
        # 비밀번호 확인
        confirm_password = st.text_input(
            "비밀번호 확인",
            type="password",
            placeholder="••••••••"
        )
        
        # 추가 정보
        st.divider()
        st.markdown("### 추가 정보 (선택)")
        
        organization = st.text_input("소속", placeholder="○○대학교")
        field = st.selectbox(
            "연구 분야",
            ["선택하세요", "화학", "재료과학", "생명공학", "약학", "식품공학", "환경공학", "기타"]
        )
        
        # 보안 질문 (오프라인 비밀번호 재설정용)
        st.divider()
        st.markdown("### 🔒 보안 설정")
        security_question = st.selectbox(
            "보안 질문",
            [
                "선택하세요",
                "졸업한 초등학교 이름은?",
                "어머니의 성함은?",
                "첫 애완동물의 이름은?",
                "가장 좋아하는 음식은?",
                "태어난 도시는?"
            ]
        )
        security_answer = st.text_input("보안 답변", type="password")
        
        # 약관 동의
        st.divider()
        terms_accepted = st.checkbox(
            "서비스 이용약관 및 개인정보 처리방침에 동의합니다",
            help="필수 동의 사항입니다"
        )
        
        marketing_accepted = st.checkbox(
            "마케팅 정보 수신에 동의합니다 (선택)",
            help="제품 업데이트, 이벤트 등의 정보를 받아보실 수 있습니다"
        )
        
        # 가입 버튼
        submitted = st.form_submit_button(
            "🎉 가입하기",
            use_container_width=True,
            type="primary"
        )
        
        if submitted:
            # 유효성 검사
            errors = []
            
            if not all([first_name, last_name, email, password, confirm_password]):
                errors.append("필수 정보를 모두 입력해주세요.")
            
            if not validate_email(email):
                errors.append("올바른 이메일 형식이 아닙니다.")
            
            if password != confirm_password:
                errors.append("비밀번호가 일치하지 않습니다.")
            
            password_check = validate_password_strength(password)
            if not password_check['valid']:
                errors.append("비밀번호가 보안 요구사항을 충족하지 않습니다.")
            
            if security_question == "선택하세요" or not security_answer:
                errors.append("보안 질문과 답변을 설정해주세요.")
            
            if not terms_accepted:
                errors.append("서비스 이용약관에 동의해주세요.")
            
            if errors:
                for error in errors:
                    st.error(error)
            else:
                # 회원가입 처리
                user_data = {
                    'email': email,
                    'password': password,
                    'name': f"{last_name}{first_name}",
                    'first_name': first_name,
                    'last_name': last_name,
                    'organization': organization if organization else None,
                    'field': field if field != "선택하세요" else None,
                    'security_question': security_question,
                    'security_answer': security_answer,
                    'marketing_accepted': marketing_accepted,
                    'role': UserRole.USER
                }
                
                handle_signup(user_data)
    
    # 로그인으로 돌아가기
    if st.button("← 로그인으로 돌아가기"):
        st.session_state.auth_mode = 'login'
        st.rerun()

def render_forgot_password_form():
    """비밀번호 찾기 폼 렌더링"""
    st.subheader("🔑 비밀번호 찾기")
    
    with st.form("forgot_password_form"):
        st.info("가입 시 사용한 이메일 주소를 입력하세요.")
        
        email = st.text_input(
            "이메일",
            placeholder="your@email.com"
        )
        
        submitted = st.form_submit_button(
            "비밀번호 재설정",
            use_container_width=True,
            type="primary"
        )
        
        if submitted:
            if not email:
                st.error("이메일을 입력해주세요.")
            elif not validate_email(email):
                st.error("올바른 이메일 형식이 아닙니다.")
            else:
                handle_password_reset(email)
    
    # 로그인으로 돌아가기
    if st.button("← 로그인으로 돌아가기"):
        st.session_state.auth_mode = 'login'
        st.rerun()

def render_profile_form():
    """프로필 관리 폼 렌더링"""
    if not st.session_state.authenticated:
        st.warning("로그인이 필요합니다.")
        st.session_state.auth_mode = 'login'
        st.rerun()
        return
    
    st.subheader("⚙️ 프로필 설정")
    
    user = st.session_state.user
    auth_manager = get_auth_manager()
    
    tabs = st.tabs(["기본 정보", "비밀번호 변경", "프로필 사진", "계정 설정"])
    
    # 기본 정보 탭
    with tabs[0]:
        with st.form("profile_basic_form"):
            col1, col2 = st.columns(2)
            with col1:
                first_name = st.text_input("이름", value=user.get('first_name', ''))
            with col2:
                last_name = st.text_input("성", value=user.get('last_name', ''))
            
            organization = st.text_input("소속", value=user.get('organization', ''))
            field = st.selectbox(
                "연구 분야",
                ["선택하세요", "화학", "재료과학", "생명공학", "약학", "식품공학", "환경공학", "기타"],
                index=["선택하세요", "화학", "재료과학", "생명공학", "약학", "식품공학", "환경공학", "기타"].index(user.get('field', '선택하세요'))
            )
            
            bio = st.text_area("자기소개", value=user.get('bio', ''), height=100)
            
            if st.form_submit_button("저장", type="primary"):
                update_data = {
                    'first_name': first_name,
                    'last_name': last_name,
                    'name': f"{last_name}{first_name}",
                    'organization': organization,
                    'field': field if field != "선택하세요" else None,
                    'bio': bio
                }
                
                result = auth_manager.update_user_profile(user['id'], update_data)
                if result['success']:
                    st.success("✅ 프로필이 업데이트되었습니다.")
                    st.session_state.user.update(update_data)
                    st.rerun()
                else:
                    st.error(f"업데이트 실패: {result.get('error', '알 수 없는 오류')}")
    
    # 비밀번호 변경 탭
    with tabs[1]:
        with st.form("change_password_form"):
            current_password = st.text_input("현재 비밀번호", type="password")
            new_password = st.text_input("새 비밀번호", type="password")
            
            if new_password:
                render_password_strength_indicator(new_password)
            
            confirm_new_password = st.text_input("새 비밀번호 확인", type="password")
            
            if st.form_submit_button("비밀번호 변경", type="primary"):
                if not all([current_password, new_password, confirm_new_password]):
                    st.error("모든 필드를 입력해주세요.")
                elif new_password != confirm_new_password:
                    st.error("새 비밀번호가 일치하지 않습니다.")
                else:
                    result = auth_manager.change_password(
                        user['id'],
                        current_password,
                        new_password
                    )
                    if result['success']:
                        st.success("✅ 비밀번호가 변경되었습니다.")
                        st.info("보안을 위해 다시 로그인해주세요.")
                        st.session_state.authenticated = False
                        st.session_state.auth_mode = 'login'
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error(f"비밀번호 변경 실패: {result.get('error', '현재 비밀번호가 올바르지 않습니다.')}")
    
    # 프로필 사진 탭
    with tabs[2]:
        st.write("현재 프로필 사진:")
        
        # 현재 프로필 사진 표시
        if user.get('profile_image'):
            try:
                # Base64 디코딩
                image_data = base64.b64decode(user['profile_image'])
                image = Image.open(BytesIO(image_data))
                st.image(image, width=150)
            except:
                st.info("프로필 사진이 없습니다.")
        else:
            st.info("프로필 사진이 없습니다.")
        
        # 새 프로필 사진 업로드
        uploaded_file = st.file_uploader(
            "새 프로필 사진 선택",
            type=['png', 'jpg', 'jpeg'],
            help="최대 2MB, PNG/JPG 형식"
        )
        
        if uploaded_file:
            # 파일 크기 체크
            if uploaded_file.size > 2 * 1024 * 1024:
                st.error("파일 크기는 2MB 이하여야 합니다.")
            else:
                # 이미지 처리
                image = Image.open(uploaded_file)
                
                # 이미지 리사이즈 (최대 500x500)
                image.thumbnail((500, 500), Image.Resampling.LANCZOS)
                
                # Base64 인코딩
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                image_base64 = base64.b64encode(buffered.getvalue()).decode()
                
                # 미리보기
                st.image(image, caption="새 프로필 사진 미리보기", width=150)
                
                if st.button("프로필 사진 저장", type="primary"):
                    result = auth_manager.update_user_profile(
                        user['id'],
                        {'profile_image': image_base64}
                    )
                    if result['success']:
                        st.success("✅ 프로필 사진이 업데이트되었습니다.")
                        st.session_state.user['profile_image'] = image_base64
                        st.rerun()
                    else:
                        st.error("프로필 사진 업데이트 실패")
    
    # 계정 설정 탭
    with tabs[3]:
        st.markdown("### 🔔 알림 설정")
        
        notifications = user.get('notification_settings', {})
        
        email_notifications = st.checkbox(
            "이메일 알림 받기",
            value=notifications.get('email', True),
            help="프로젝트 업데이트, 협업 요청 등"
        )
        
        marketing_emails = st.checkbox(
            "마케팅 정보 수신",
            value=notifications.get('marketing', False),
            help="제품 업데이트, 이벤트 정보 등"
        )
        
        if st.button("알림 설정 저장"):
            notification_settings = {
                'email': email_notifications,
                'marketing': marketing_emails
            }
            
            result = auth_manager.update_user_profile(
                user['id'],
                {'notification_settings': notification_settings}
            )
            if result['success']:
                st.success("✅ 알림 설정이 저장되었습니다.")
                st.session_state.user['notification_settings'] = notification_settings
            else:
                st.error("알림 설정 저장 실패")
        
        st.divider()
        
        st.markdown("### 🚪 계정 관리")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚪 로그아웃", use_container_width=True):
                auth_manager.logout()
                st.session_state.clear()
                st.success("로그아웃되었습니다.")
                st.switch_page("polymer_platform.py")
        
        with col2:
            if st.button("🗑️ 계정 삭제", use_container_width=True, type="secondary"):
                st.session_state.show_delete_confirm = True
        
        if st.session_state.get('show_delete_confirm'):
            st.warning("⚠️ 계정을 삭제하면 모든 데이터가 영구적으로 삭제됩니다.")
            confirm_text = st.text_input("확인을 위해 'DELETE'를 입력하세요:")
            
            if confirm_text == "DELETE":
                if st.button("계정 영구 삭제", type="primary"):
                    result = auth_manager.delete_account(user['id'])
                    if result['success']:
                        st.success("계정이 삭제되었습니다.")
                        st.session_state.clear()
                        st.switch_page("polymer_platform.py")
                    else:
                        st.error("계정 삭제 실패")
            
            if st.button("취소"):
                st.session_state.show_delete_confirm = False
                st.rerun()

# =============================================================================
# 🎯 메인 함수
# =============================================================================

def main():
    """메인 함수"""
    # 초기화
    init_session_state()
    
    # 이미 로그인된 경우
    if st.session_state.authenticated and st.session_state.get('auth_mode') != 'profile':
        st.info("이미 로그인되어 있습니다.")
        if st.button("대시보드로 이동"):
            st.switch_page("pages/1_📊_Dashboard.py")
        if st.button("프로필 관리"):
            st.session_state.auth_mode = 'profile'
            st.rerun()
        return
    
    # 헤더 렌더링
    render_header()
    
    # AI 설명 모드 토글 (사이드바)
    with st.sidebar:
        st.markdown("### ⚙️ 설정")
        
        show_details = st.checkbox(
            "🤖 AI 상세 설명 표시",
            value=st.session_state.show_ai_details,
            help="AI의 추론 과정과 설계 근거를 자세히 볼 수 있습니다"
        )
        st.session_state.show_ai_details = show_details
        
        # 언어 설정 (향후 구현)
        # language = st.selectbox("🌐 언어", ["한국어", "English"])
    
    # 인증 모드에 따른 렌더링
    auth_mode = st.session_state.get('auth_mode', 'login')
    
    if auth_mode == 'login':
        render_login_form()
    elif auth_mode == 'signup':
        render_signup_form()
    elif auth_mode == 'forgot':
        render_forgot_password_form()
    elif auth_mode == 'profile':
        render_profile_form()
    
    # 푸터
    st.divider()
    st.markdown("""
        <div style="text-align: center; color: #888; font-size: 0.875rem;">
            <p>Universal DOE Platform v2.0.0 | 
            <a href="https://github.com/universaldoe" target="_blank">GitHub</a> | 
            <a href="mailto:support@universaldoe.com">지원</a></p>
            <p>© 2024 Universal DOE Team. All rights reserved.</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
