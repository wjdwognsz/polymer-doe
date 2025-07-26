"""
0_🔐_Login.py - Universal DOE Platform 인증 페이지
로컬 우선 인증 시스템 with 선택적 클라우드 동기화
"""
import streamlit as st
import sys
from pathlib import Path
from datetime import datetime, timedelta
import re
import time
from typing import Optional, Dict, Any

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

# 페이지 설정 (필수 - 최상단)
st.set_page_config(
    page_title="Login - Universal DOE Platform",
    page_icon="🔐",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 모듈 임포트
try:
    from utils.auth_manager import get_auth_manager
    from utils.common_ui import get_common_ui
    from utils.database_manager import get_database_manager
    from config.app_config import SECURITY_CONFIG, APP_INFO
    from config.theme_config import apply_theme
except ImportError as e:
    st.error(f"필수 모듈을 찾을 수 없습니다: {e}")
    st.info("프로젝트 루트에서 실행 중인지 확인해주세요.")
    st.stop()

# 테마 적용
apply_theme()

# 전역 변수
AUTH_MANAGER = get_auth_manager()
UI = get_common_ui()
DB_MANAGER = get_database_manager()

def initialize_session_state():
    """세션 상태 초기화"""
    defaults = {
        'login_attempts': 0,
        'lockout_until': None,
        'show_password': False,
        'show_password_confirm': False,
        'password_strength': 0,
        'terms_accepted': False,
        'remember_me': False,
        'selected_tab': 0,
        'show_reset_form': False,
        'online_status': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # 온라인 상태 체크 (비동기적으로)
    check_online_status()

def check_online_status():
    """온라인 상태 확인"""
    try:
        import requests
        response = requests.get('https://www.google.com', timeout=2)
        st.session_state.online_status = response.status_code == 200
    except:
        st.session_state.online_status = False

def validate_email(email: str) -> bool:
    """이메일 형식 검증"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def calculate_password_strength(password: str) -> Dict[str, Any]:
    """비밀번호 강도 계산"""
    strength = 0
    feedback = []
    
    # 길이 체크
    if len(password) >= SECURITY_CONFIG['password_min_length']:
        strength += 25
    else:
        feedback.append(f"최소 {SECURITY_CONFIG['password_min_length']}자 이상")
    
    # 대문자 체크
    if re.search(r'[A-Z]', password):
        strength += 25
    elif SECURITY_CONFIG['password_require_uppercase']:
        feedback.append("대문자 포함 필요")
    
    # 숫자 체크
    if re.search(r'\d', password):
        strength += 25
    elif SECURITY_CONFIG['password_require_number']:
        feedback.append("숫자 포함 필요")
    
    # 특수문자 체크
    if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        strength += 25
    elif SECURITY_CONFIG['password_require_special']:
        feedback.append("특수문자 포함 필요")
    
    # 강도별 색상
    if strength < 50:
        color = "red"
        label = "약함"
    elif strength < 75:
        color = "orange"
        label = "보통"
    else:
        color = "green"
        label = "강함"
    
    return {
        'strength': strength,
        'color': color,
        'label': label,
        'feedback': feedback
    }

def render_login_tab():
    """로그인 탭 렌더링"""
    # 잠금 상태 확인
    if st.session_state.lockout_until:
        if datetime.now() < st.session_state.lockout_until:
            remaining = (st.session_state.lockout_until - datetime.now()).seconds
            st.error(f"🔒 너무 많은 로그인 시도가 있었습니다. {remaining//60}분 {remaining%60}초 후에 다시 시도해주세요.")
            return
        else:
            # 잠금 해제
            st.session_state.lockout_until = None
            st.session_state.login_attempts = 0
    
    # 로그인 폼
    with st.form("login_form", clear_on_submit=False):
        st.markdown("### 🔑 로그인")
        
        # 이메일 입력
        email = st.text_input(
            "이메일",
            placeholder="your.email@example.com",
            help="가입 시 사용한 이메일을 입력하세요"
        )
        
        # 비밀번호 입력
        col1, col2 = st.columns([5, 1])
        with col1:
            password = st.text_input(
                "비밀번호",
                type="password" if not st.session_state.show_password else "text",
                placeholder="비밀번호 입력"
            )
        with col2:
            st.write("")  # 간격 맞추기
            if st.button("👁️" if not st.session_state.show_password else "🙈", 
                        help="비밀번호 표시/숨기기"):
                st.session_state.show_password = not st.session_state.show_password
                st.rerun()
        
        # 옵션
        col1, col2 = st.columns(2)
        with col1:
            remember_me = st.checkbox("로그인 상태 유지", value=st.session_state.remember_me)
        with col2:
            if st.button("비밀번호를 잊으셨나요?", type="secondary"):
                st.session_state.show_reset_form = True
                st.rerun()
        
        # 로그인 버튼
        submitted = st.form_submit_button("🚀 로그인", type="primary", use_container_width=True)
        
        if submitted:
            # 입력 검증
            if not email or not password:
                st.error("이메일과 비밀번호를 모두 입력해주세요.")
            elif not validate_email(email):
                st.error("올바른 이메일 형식이 아닙니다.")
            else:
                # 로그인 시도
                with st.spinner("로그인 중..."):
                    result = AUTH_MANAGER.login(email, password, remember_me)
                
                if result['success']:
                    st.success("✅ 로그인 성공!")
                    st.balloons()
                    
                    # 세션 설정
                    st.session_state.authenticated = True
                    st.session_state.user = result['user']
                    st.session_state.user_email = email
                    st.session_state.user_role = result['user'].get('role', 'user')
                    
                    # 온라인이면 동기화 시도
                    if st.session_state.online_status:
                        with st.spinner("클라우드 동기화 중..."):
                            AUTH_MANAGER.sync_user_data(email)
                    
                    time.sleep(1)  # 성공 메시지 표시
                    st.switch_page("pages/1_📊_Dashboard.py")
                else:
                    # 로그인 실패
                    st.session_state.login_attempts += 1
                    
                    if st.session_state.login_attempts >= SECURITY_CONFIG['max_login_attempts']:
                        st.session_state.lockout_until = datetime.now() + SECURITY_CONFIG['lockout_duration']
                        st.error(f"🔒 로그인 시도 횟수를 초과했습니다. {SECURITY_CONFIG['lockout_duration'].seconds//60}분 후에 다시 시도해주세요.")
                    else:
                        remaining = SECURITY_CONFIG['max_login_attempts'] - st.session_state.login_attempts
                        st.error(f"❌ {result.get('message', '로그인에 실패했습니다.')} (남은 시도: {remaining}회)")
    
    # 소셜 로그인 (온라인 시만)
    if st.session_state.online_status:
        st.divider()
        st.markdown("### 또는")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔷 Google로 로그인", use_container_width=True):
                auth_url = AUTH_MANAGER.get_google_auth_url()
                st.markdown(f'<meta http-equiv="refresh" content="0; url={auth_url}">', 
                          unsafe_allow_html=True)
    
    # 게스트 모드
    st.divider()
    if st.button("👀 게스트로 둘러보기", use_container_width=True, type="secondary"):
        st.session_state.authenticated = True
        st.session_state.user = {
            'name': 'Guest User',
            'email': 'guest@universaldoe.com',
            'role': 'guest'
        }
        st.session_state.guest_mode = True
        st.info("게스트 모드로 접속합니다. 일부 기능이 제한됩니다.")
        time.sleep(1)
        st.switch_page("pages/1_📊_Dashboard.py")

def render_signup_tab():
    """회원가입 탭 렌더링"""
    with st.form("signup_form", clear_on_submit=False):
        st.markdown("### 🎉 회원가입")
        
        # 이름 입력
        name = st.text_input(
            "이름",
            placeholder="홍길동",
            help="프로필에 표시될 이름입니다"
        )
        
        # 이메일 입력
        email = st.text_input(
            "이메일",
            placeholder="your.email@example.com",
            help="로그인 시 사용할 이메일입니다"
        )
        
        # 이메일 유효성 표시
        if email:
            if validate_email(email):
                st.success("✅ 사용 가능한 이메일 형식입니다")
                # 중복 체크 (실시간)
                if DB_MANAGER.check_email_exists(email):
                    st.error("❌ 이미 사용 중인 이메일입니다")
            else:
                st.error("❌ 올바른 이메일 형식이 아닙니다")
        
        # 비밀번호 입력
        col1, col2 = st.columns([5, 1])
        with col1:
            password = st.text_input(
                "비밀번호",
                type="password" if not st.session_state.show_password else "text",
                placeholder="안전한 비밀번호 입력",
                help="대소문자, 숫자, 특수문자를 포함해주세요"
            )
        with col2:
            st.write("")
            if st.button("👁️" if not st.session_state.show_password else "🙈",
                        key="pwd_toggle_signup",
                        help="비밀번호 표시/숨기기"):
                st.session_state.show_password = not st.session_state.show_password
                st.rerun()
        
        # 비밀번호 강도 표시
        if password:
            pwd_info = calculate_password_strength(password)
            st.progress(pwd_info['strength'] / 100)
            st.markdown(f"비밀번호 강도: **:{pwd_info['color']}[{pwd_info['label']}]**")
            if pwd_info['feedback']:
                st.warning("개선사항: " + ", ".join(pwd_info['feedback']))
        
        # 비밀번호 확인
        col1, col2 = st.columns([5, 1])
        with col1:
            password_confirm = st.text_input(
                "비밀번호 확인",
                type="password" if not st.session_state.show_password_confirm else "text",
                placeholder="비밀번호 재입력"
            )
        with col2:
            st.write("")
            if st.button("👁️" if not st.session_state.show_password_confirm else "🙈",
                        key="pwd_confirm_toggle",
                        help="비밀번호 확인 표시/숨기기"):
                st.session_state.show_password_confirm = not st.session_state.show_password_confirm
                st.rerun()
        
        # 비밀번호 일치 확인
        if password and password_confirm:
            if password == password_confirm:
                st.success("✅ 비밀번호가 일치합니다")
            else:
                st.error("❌ 비밀번호가 일치하지 않습니다")
        
        # 약관 동의
        st.divider()
        terms_accepted = st.checkbox(
            "서비스 이용약관 및 개인정보 처리방침에 동의합니다",
            value=st.session_state.terms_accepted
        )
        
        if st.button("이용약관 보기", type="secondary"):
            with st.expander("서비스 이용약관", expanded=True):
                st.markdown("""
                **Universal DOE Platform 이용약관**
                
                1. 본 플랫폼은 연구 목적으로 무료로 제공됩니다.
                2. 사용자의 데이터는 안전하게 보호됩니다.
                3. 부적절한 사용 시 서비스 이용이 제한될 수 있습니다.
                4. 자세한 내용은 [GitHub 페이지](https://github.com/universaldoe)를 참조하세요.
                """)
        
        # 가입 버튼
        submitted = st.form_submit_button("🚀 가입하기", type="primary", use_container_width=True)
        
        if submitted:
            # 입력 검증
            errors = []
            
            if not name:
                errors.append("이름을 입력해주세요")
            if not email:
                errors.append("이메일을 입력해주세요")
            elif not validate_email(email):
                errors.append("올바른 이메일 형식이 아닙니다")
            elif DB_MANAGER.check_email_exists(email):
                errors.append("이미 사용 중인 이메일입니다")
            
            if not password:
                errors.append("비밀번호를 입력해주세요")
            else:
                pwd_info = calculate_password_strength(password)
                if pwd_info['strength'] < 50:
                    errors.append("더 안전한 비밀번호를 사용해주세요")
            
            if password != password_confirm:
                errors.append("비밀번호가 일치하지 않습니다")
            
            if not terms_accepted:
                errors.append("이용약관에 동의해주세요")
            
            if errors:
                for error in errors:
                    st.error(f"❌ {error}")
            else:
                # 회원가입 처리
                with st.spinner("계정을 생성하는 중..."):
                    result = AUTH_MANAGER.register(
                        email=email,
                        password=password,
                        name=name
                    )
                
                if result['success']:
                    st.success("✅ 회원가입이 완료되었습니다!")
                    st.info("이제 로그인할 수 있습니다.")
                    
                    # 온라인이면 환영 이메일 발송 시도
                    if st.session_state.online_status:
                        AUTH_MANAGER.send_welcome_email(email, name)
                    
                    # 로그인 탭으로 전환
                    time.sleep(2)
                    st.session_state.selected_tab = 0
                    st.rerun()
                else:
                    st.error(f"❌ 가입 중 오류가 발생했습니다: {result.get('message', '알 수 없는 오류')}")

def render_reset_password():
    """비밀번호 재설정 폼"""
    st.markdown("### 🔓 비밀번호 재설정")
    
    if st.session_state.online_status:
        # 온라인: 이메일로 재설정 링크 발송
        with st.form("reset_form"):
            email = st.text_input(
                "가입한 이메일 주소",
                placeholder="your.email@example.com"
            )
            
            submitted = st.form_submit_button("재설정 링크 받기", use_container_width=True)
            
            if submitted:
                if not email:
                    st.error("이메일을 입력해주세요")
                elif not validate_email(email):
                    st.error("올바른 이메일 형식이 아닙니다")
                else:
                    with st.spinner("처리 중..."):
                        result = AUTH_MANAGER.send_reset_email(email)
                    
                    if result['success']:
                        st.success("✅ 재설정 링크를 이메일로 발송했습니다. 메일함을 확인해주세요.")
                        st.session_state.show_reset_form = False
                    else:
                        st.error(f"❌ {result.get('message', '재설정 링크 발송에 실패했습니다')}")
    else:
        # 오프라인: 보안 질문 방식
        st.info("오프라인 모드에서는 보안 질문을 통해 비밀번호를 재설정합니다.")
        
        with st.form("reset_offline_form"):
            email = st.text_input(
                "가입한 이메일 주소",
                placeholder="your.email@example.com"
            )
            
            # 보안 질문 (실제로는 DB에서 가져와야 함)
            security_answer = st.text_input(
                "보안 질문: 가장 좋아하는 색은?",
                placeholder="답변 입력"
            )
            
            new_password = st.text_input(
                "새 비밀번호",
                type="password",
                placeholder="새로운 비밀번호 입력"
            )
            
            submitted = st.form_submit_button("비밀번호 변경", use_container_width=True)
            
            if submitted:
                # 오프라인 재설정 로직
                st.info("오프라인 비밀번호 재설정 기능은 준비 중입니다.")
    
    if st.button("← 로그인으로 돌아가기", type="secondary"):
        st.session_state.show_reset_form = False
        st.rerun()

def render_login_page():
    """메인 로그인 페이지"""
    # 헤더
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">
            🧬 Universal DOE Platform
        </h1>
        <p style="font-size: 1.2rem; color: #666;">
            모든 연구자를 위한 AI 기반 실험 설계 플랫폼
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 온라인 상태 표시
    if st.session_state.online_status:
        st.success("🟢 온라인 모드 - 모든 기능 사용 가능")
    else:
        st.warning("🟡 오프라인 모드 - 로컬 기능만 사용 가능")
    
    # 비밀번호 재설정 폼 표시
    if st.session_state.show_reset_form:
        render_reset_password()
    else:
        # 탭 선택
        tab1, tab2 = st.tabs(["🔑 로그인", "🎉 회원가입"])
        
        with tab1:
            render_login_tab()
        
        with tab2:
            render_signup_tab()
    
    # 푸터
    st.divider()
    st.markdown(
        f"""
        <div style="text-align: center; color: #888; font-size: 0.9rem;">
            {APP_INFO['name']} v{APP_INFO['version']} | 
            <a href="{APP_INFO['github']}" target="_blank">GitHub</a> | 
            <a href="#" onclick="alert('도움말 준비 중')">도움말</a>
        </div>
        """,
        unsafe_allow_html=True
    )

def main():
    """메인 함수"""
    # 세션 초기화
    initialize_session_state()
    
    # 이미 인증된 경우 대시보드로 리다이렉트
    if st.session_state.get('authenticated', False):
        st.switch_page("pages/1_📊_Dashboard.py")
        return
    
    # Google OAuth 콜백 처리
    query_params = st.query_params
    if 'code' in query_params:
        with st.spinner("Google 로그인 처리 중..."):
            result = AUTH_MANAGER.handle_google_callback(query_params['code'])
            if result['success']:
                st.session_state.authenticated = True
                st.session_state.user = result['user']
                st.success("✅ Google 로그인 성공!")
                st.switch_page("pages/1_📊_Dashboard.py")
            else:
                st.error(f"❌ Google 로그인 실패: {result.get('message')}")
    
    # 로그인 페이지 렌더링
    render_login_page()

if __name__ == "__main__":
    main()
