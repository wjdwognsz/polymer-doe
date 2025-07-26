"""
🔐 Authentication Page - 인증 페이지
===========================================================================
데스크톱 애플리케이션을 위한 로그인/회원가입/프로필 관리 페이지
로컬 인증 시스템과 연동, 오프라인 우선 설계
===========================================================================
"""

import streamlit as st
import re
import secrets
import string
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List, Any
import logging
from PIL import Image
import io
import base64
import json
import time

# 로컬 모듈
from utils.auth_manager import AuthManager, get_current_user, is_authenticated, UserRole
from utils.common_ui import (
    show_success, show_error, show_warning, show_info,
    render_metric_card, render_progress_bar, render_empty_state,
    render_form_input, render_card, validate_email, validate_password,
    format_datetime
)

# 설정
try:
    from config.app_config import APP_INFO, SECURITY_CONFIG, SESSION_CONFIG
    from config.local_config import LOCAL_CONFIG
except ImportError:
    # 기본값
    SECURITY_CONFIG = {
        'password_min_length': 8,
        'max_login_attempts': 5
    }

# ===========================================================================
# 🔧 설정 및 상수
# ===========================================================================

logger = logging.getLogger(__name__)

# UI 텍스트
UI_TEXTS = {
    'ko': {
        'login_title': '🔐 로그인',
        'signup_title': '👤 회원가입',
        'forgot_password': '🔑 비밀번호 찾기',
        'profile_title': '⚙️ 프로필 설정',
        'email_placeholder': '이메일 주소를 입력하세요',
        'password_placeholder': '비밀번호를 입력하세요',
        'name_placeholder': '이름을 입력하세요',
        'organization_placeholder': '소속 기관 (선택)',
        'login_button': '로그인',
        'signup_button': '회원가입',
        'reset_button': '비밀번호 재설정',
        'guest_button': '둘러보기 (게스트)',
        'logout_button': '로그아웃',
        'save_button': '저장',
        'cancel_button': '취소'
    },
    'en': {
        'login_title': '🔐 Login',
        'signup_title': '👤 Sign Up',
        'forgot_password': '🔑 Forgot Password',
        'profile_title': '⚙️ Profile Settings',
        'email_placeholder': 'Enter your email',
        'password_placeholder': 'Enter your password',
        'name_placeholder': 'Enter your name',
        'organization_placeholder': 'Organization (optional)',
        'login_button': 'Login',
        'signup_button': 'Sign Up',
        'reset_button': 'Reset Password',
        'guest_button': 'Browse as Guest',
        'logout_button': 'Logout',
        'save_button': 'Save',
        'cancel_button': 'Cancel'
    }
}

# 비밀번호 강도 레벨
PASSWORD_STRENGTH_LEVELS = {
    0: {'label': '매우 약함', 'color': '#FF0000', 'icon': '🔴'},
    1: {'label': '약함', 'color': '#FF6B00', 'icon': '🟠'},
    2: {'label': '보통', 'color': '#FFD700', 'icon': '🟡'},
    3: {'label': '강함', 'color': '#32CD32', 'icon': '🟢'},
    4: {'label': '매우 강함', 'color': '#006400', 'icon': '🟢'}
}

# ===========================================================================
# 🎯 인증 페이지 클래스
# ===========================================================================

class AuthPage:
    """인증 페이지 클래스"""
    
    def __init__(self):
        """초기화"""
        self.auth_manager = self._get_auth_manager()
        self.language = st.session_state.get('language', 'ko')
        self.texts = UI_TEXTS[self.language]
        
        # 세션 초기화
        self._initialize_session_state()
    
    def _get_auth_manager(self) -> AuthManager:
        """AuthManager 인스턴스 가져오기"""
        if 'auth_manager' not in st.session_state:
            # DatabaseManager가 있다면 전달, 없으면 None
            db_manager = st.session_state.get('db_manager')
            st.session_state.auth_manager = AuthManager(db_manager)
        return st.session_state.auth_manager
    
    def _initialize_session_state(self):
        """세션 상태 초기화"""
        defaults = {
            'auth_mode': 'login',
            'signup_step': 0,
            'login_attempts': {},
            'temp_data': {},
            'show_terms': False,
            'captcha_answer': None,
            'captcha_question': None
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    # =========================================================================
    # 🔐 로그인 기능
    # =========================================================================
    
    def render_login_form(self):
        """로그인 폼 렌더링"""
        st.markdown(f"### {self.texts['login_title']}")
        
        with st.form("login_form", clear_on_submit=False):
            # 이메일 입력
            email = st.text_input(
                "이메일",
                placeholder=self.texts['email_placeholder'],
                key="login_email",
                help="가입하신 이메일 주소를 입력하세요"
            )
            
            # 비밀번호 입력
            password = st.text_input(
                "비밀번호",
                type="password",
                placeholder=self.texts['password_placeholder'],
                key="login_password"
            )
            
            # 추가 옵션
            col1, col2 = st.columns(2)
            with col1:
                remember_me = st.checkbox("로그인 상태 유지", value=True)
            with col2:
                # 연결 상태 표시
                if LOCAL_CONFIG.get('offline_mode', {}).get('default', True):
                    st.caption("🔌 오프라인 모드")
                else:
                    st.caption("🌐 온라인 모드")
            
            # 로그인 버튼
            submitted = st.form_submit_button(
                self.texts['login_button'],
                use_container_width=True,
                type="primary"
            )
            
            if submitted:
                if not email or not password:
                    show_error("이메일과 비밀번호를 모두 입력해주세요.")
                elif not validate_email(email):
                    show_error("올바른 이메일 형식이 아닙니다.")
                else:
                    self._handle_login(email, password, remember_me)
        
        # 추가 액션
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("회원가입", use_container_width=True):
                st.session_state.auth_mode = 'signup'
                st.session_state.signup_step = 0
                st.rerun()
        
        with col2:
            if st.button("비밀번호 찾기", use_container_width=True):
                st.session_state.auth_mode = 'forgot'
                st.rerun()
        
        with col3:
            if st.button("🔍 " + self.texts['guest_button'], use_container_width=True):
                self._enter_guest_mode()
    
    def _handle_login(self, email: str, password: str, remember_me: bool):
        """로그인 처리"""
        try:
            # 로그인 시도
            with st.spinner("로그인 중..."):
                success, message, user_info = self.auth_manager.login(
                    email=email,
                    password=password,
                    remember_me=remember_me
                )
            
            if success:
                # 세션 설정
                st.session_state.authenticated = True
                st.session_state.user = user_info
                st.session_state.login_time = datetime.now()
                
                # 환영 메시지
                show_success(f"환영합니다, {user_info['name']}님! 👋")
                
                # 로그 기록
                logger.info(f"User logged in: {email}")
                
                # 대시보드로 이동
                time.sleep(1)  # 메시지 표시 시간
                st.session_state.current_page = 'dashboard'
                st.rerun()
            else:
                # 로그인 실패
                show_error(message)
                
                # 실패 횟수 기록
                self._record_failed_attempt(email)
                
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            show_error("로그인 중 오류가 발생했습니다.")
    
    def _record_failed_attempt(self, email: str):
        """로그인 실패 기록"""
        if email not in st.session_state.login_attempts:
            st.session_state.login_attempts[email] = []
        
        st.session_state.login_attempts[email].append(datetime.now())
        
        # 최근 15분 내 시도만 유지
        cutoff = datetime.now() - timedelta(minutes=15)
        st.session_state.login_attempts[email] = [
            attempt for attempt in st.session_state.login_attempts[email]
            if attempt > cutoff
        ]
        
        # 시도 횟수 확인
        attempts = len(st.session_state.login_attempts[email])
        max_attempts = SECURITY_CONFIG.get('max_login_attempts', 5)
        
        if attempts >= max_attempts:
            show_warning(f"로그인 시도 횟수 초과 ({attempts}/{max_attempts}). 15분 후 다시 시도해주세요.")
    
    def _enter_guest_mode(self):
        """게스트 모드 진입"""
        st.session_state.authenticated = True
        st.session_state.user = {
            'id': 'guest',
            'email': 'guest@universaldoe.com',
            'name': '게스트',
            'role': UserRole.GUEST,
            'permissions': {
                'project': ['read'],
                'experiment': [],
                'module': ['use_basic']
            }
        }
        st.session_state.guest_mode = True
        
        show_info("게스트 모드로 접속했습니다. 일부 기능이 제한됩니다.")
        
        time.sleep(1)
        st.session_state.current_page = 'dashboard'
        st.rerun()
    
    # =========================================================================
    # 👤 회원가입 기능
    # =========================================================================
    
    def render_signup_form(self):
        """회원가입 폼 렌더링"""
        st.markdown(f"### {self.texts['signup_title']}")
        
        # 진행 상태 표시
        self._render_signup_progress()
        
        # 단계별 렌더링
        step = st.session_state.signup_step
        
        if step == 0:
            self._render_signup_basic_info()
        elif step == 1:
            self._render_signup_password()
        elif step == 2:
            self._render_signup_additional_info()
        elif step == 3:
            self._render_signup_terms()
        elif step == 4:
            self._render_signup_complete()
    
    def _render_signup_progress(self):
        """회원가입 진행 상태 표시"""
        steps = ['기본 정보', '비밀번호', '추가 정보', '약관 동의', '완료']
        current = st.session_state.signup_step
        
        # 프로그레스 바
        progress = (current + 1) / len(steps)
        render_progress_bar(
            value=current + 1,
            max_value=len(steps),
            label="진행 상황",
            format_string=f"{{:.0f}}/{len(steps)} 단계"
        )
        
        # 단계 표시
        cols = st.columns(len(steps))
        for i, (col, step) in enumerate(zip(cols, steps)):
            with col:
                if i < current:
                    st.markdown(f"✅ **{step}**")
                elif i == current:
                    st.markdown(f"📍 **{step}**")
                else:
                    st.markdown(f"⭕ {step}")
    
    def _render_signup_basic_info(self):
        """Step 1: 기본 정보 입력"""
        st.markdown("#### 📝 기본 정보")
        
        with st.form("signup_basic"):
            # 이메일
            email = st.text_input(
                "이메일 *",
                value=st.session_state.temp_data.get('email', ''),
                placeholder=self.texts['email_placeholder'],
                help="로그인에 사용할 이메일 주소입니다"
            )
            
            # 이름
            name = st.text_input(
                "이름 *",
                value=st.session_state.temp_data.get('name', ''),
                placeholder=self.texts['name_placeholder'],
                help="실명을 입력해주세요"
            )
            
            # 소속
            organization = st.text_input(
                "소속 기관",
                value=st.session_state.temp_data.get('organization', ''),
                placeholder=self.texts['organization_placeholder'],
                help="회사, 대학, 연구소 등"
            )
            
            # 버튼
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("다음", type="primary", use_container_width=True):
                    # 검증
                    if not email or not name:
                        show_error("필수 항목을 모두 입력해주세요.")
                    elif not validate_email(email):
                        show_error("올바른 이메일 형식이 아닙니다.")
                    else:
                        # 이메일 중복 확인
                        if self._check_email_exists(email):
                            show_error("이미 사용 중인 이메일입니다.")
                        else:
                            # 임시 저장
                            st.session_state.temp_data.update({
                                'email': email,
                                'name': name,
                                'organization': organization
                            })
                            st.session_state.signup_step = 1
                            st.rerun()
            
            with col2:
                if st.form_submit_button("취소", use_container_width=True):
                    self._cancel_signup()
    
    def _render_signup_password(self):
        """Step 2: 비밀번호 설정"""
        st.markdown("#### 🔒 비밀번호 설정")
        
        with st.form("signup_password"):
            # 비밀번호
            password = st.text_input(
                "비밀번호 *",
                type="password",
                help=f"최소 {SECURITY_CONFIG['password_min_length']}자 이상, 대소문자, 숫자, 특수문자 포함"
            )
            
            # 비밀번호 확인
            password_confirm = st.text_input(
                "비밀번호 확인 *",
                type="password",
                help="비밀번호를 다시 입력해주세요"
            )
            
            # 비밀번호 강도 표시
            if password:
                self._render_password_strength(password)
            
            # 버튼
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.form_submit_button("이전", use_container_width=True):
                    st.session_state.signup_step = 0
                    st.rerun()
            
            with col2:
                if st.form_submit_button("다음", type="primary", use_container_width=True):
                    # 검증
                    if not password or not password_confirm:
                        show_error("비밀번호를 입력해주세요.")
                    elif password != password_confirm:
                        show_error("비밀번호가 일치하지 않습니다.")
                    else:
                        is_valid, message = validate_password(password)
                        if not is_valid:
                            show_error(message)
                        else:
                            # 임시 저장
                            st.session_state.temp_data['password'] = password
                            st.session_state.signup_step = 2
                            st.rerun()
            
            with col3:
                if st.form_submit_button("취소", use_container_width=True):
                    self._cancel_signup()
    
    def _render_password_strength(self, password: str):
        """비밀번호 강도 표시"""
        strength = self._calculate_password_strength(password)
        level_info = PASSWORD_STRENGTH_LEVELS[strength]
        
        # 시각적 표시
        st.markdown(f"""
        <div style="margin: 1rem 0;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span>{level_info['icon']}</span>
                <span style="color: {level_info['color']}; font-weight: 500;">
                    {level_info['label']}
                </span>
            </div>
            <div style="background: #e0e0e0; height: 8px; border-radius: 4px; margin-top: 0.5rem;">
                <div style="
                    background: {level_info['color']}; 
                    width: {(strength + 1) * 20}%; 
                    height: 100%; 
                    border-radius: 4px;
                    transition: width 0.3s ease;
                "></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def _calculate_password_strength(self, password: str) -> int:
        """비밀번호 강도 계산 (0-4)"""
        strength = 0
        
        if len(password) >= 8:
            strength += 1
        if len(password) >= 12:
            strength += 1
        if re.search(r'[a-z]', password) and re.search(r'[A-Z]', password):
            strength += 1
        if re.search(r'\d', password):
            strength += 1
        if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            strength += 1
        
        return min(strength, 4)
    
    def _render_signup_additional_info(self):
        """Step 3: 추가 정보"""
        st.markdown("#### 📋 추가 정보 (선택)")
        
        with st.form("signup_additional"):
            # 연구 분야
            research_field = st.selectbox(
                "주요 연구 분야",
                options=['선택하세요', '화학', '재료과학', '생명공학', '물리학', '기타'],
                index=0,
                help="주로 연구하시는 분야를 선택해주세요"
            )
            
            # 경력
            experience = st.selectbox(
                "연구 경력",
                options=['선택하세요', '학부생', '대학원생', '박사후연구원', '연구원', '교수', '기타'],
                index=0
            )
            
            # 관심사
            interests = st.multiselect(
                "관심 있는 실험 설계 방법",
                options=['완전요인설계', '부분요인설계', '반응표면설계', '혼합물설계', 
                        'Taguchi 설계', '최적설계', 'AI 기반 설계'],
                default=[]
            )
            
            # 버튼
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.form_submit_button("이전", use_container_width=True):
                    st.session_state.signup_step = 1
                    st.rerun()
            
            with col2:
                if st.form_submit_button("다음", type="primary", use_container_width=True):
                    # 임시 저장 (선택사항이므로 검증 없음)
                    st.session_state.temp_data.update({
                        'research_field': research_field if research_field != '선택하세요' else None,
                        'experience': experience if experience != '선택하세요' else None,
                        'interests': interests
                    })
                    st.session_state.signup_step = 3
                    st.rerun()
            
            with col3:
                if st.form_submit_button("취소", use_container_width=True):
                    self._cancel_signup()
    
    def _render_signup_terms(self):
        """Step 4: 약관 동의"""
        st.markdown("#### 📜 이용약관")
        
        # 약관 내용
        with st.expander("서비스 이용약관", expanded=True):
            st.markdown("""
            **Universal DOE Platform 서비스 이용약관**
            
            제1조 (목적)
            이 약관은 Universal DOE Platform(이하 "서비스")의 이용에 관한 조건 및 절차를 규정함을 목적으로 합니다.
            
            제2조 (정의)
            1. "서비스"란 회사가 제공하는 AI 기반 실험 설계 플랫폼을 의미합니다.
            2. "회원"이란 이 약관에 동의하고 회원가입을 한 자를 의미합니다.
            
            [전체 약관 내용...]
            """)
        
        with st.expander("개인정보 처리방침", expanded=False):
            st.markdown("""
            **개인정보 처리방침**
            
            1. 수집하는 개인정보
            - 필수: 이메일, 이름, 비밀번호
            - 선택: 소속기관, 연구분야, 경력
            
            2. 개인정보 이용 목적
            - 서비스 제공 및 운영
            - 이용자 식별 및 인증
            - 서비스 개선
            
            [전체 내용...]
            """)
        
        # 동의 체크박스
        st.markdown("---")
        
        agree_terms = st.checkbox(
            "서비스 이용약관에 동의합니다 (필수)",
            key="agree_terms"
        )
        
        agree_privacy = st.checkbox(
            "개인정보 처리방침에 동의합니다 (필수)",
            key="agree_privacy"
        )
        
        agree_marketing = st.checkbox(
            "마케팅 정보 수신에 동의합니다 (선택)",
            key="agree_marketing"
        )
        
        # 버튼
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("이전", use_container_width=True):
                st.session_state.signup_step = 2
                st.rerun()
        
        with col2:
            if st.button(self.texts['signup_button'], type="primary", use_container_width=True):
                if not agree_terms or not agree_privacy:
                    show_error("필수 약관에 모두 동의해주세요.")
                else:
                    # 마케팅 동의 저장
                    st.session_state.temp_data['marketing_agreed'] = agree_marketing
                    # 회원가입 처리
                    self._process_signup()
        
        with col3:
            if st.button("취소", use_container_width=True):
                self._cancel_signup()
    
    def _process_signup(self):
        """회원가입 처리"""
        try:
            data = st.session_state.temp_data
            
            with st.spinner("회원가입 처리 중..."):
                # 회원가입 요청
                success, message, user_id = self.auth_manager.register_user(
                    email=data['email'],
                    password=data['password'],
                    name=data['name'],
                    organization=data.get('organization'),
                    research_field=data.get('research_field'),
                    experience=data.get('experience'),
                    interests=data.get('interests', []),
                    marketing_agreed=data.get('marketing_agreed', False)
                )
            
            if success:
                st.session_state.signup_step = 4
                st.session_state.signup_user_id = user_id
                st.rerun()
            else:
                show_error(message)
                
        except Exception as e:
            logger.error(f"Signup error: {str(e)}")
            show_error("회원가입 중 오류가 발생했습니다.")
    
    def _render_signup_complete(self):
        """Step 5: 가입 완료"""
        st.balloons()
        
        st.markdown("### 🎉 회원가입 완료!")
        
        st.success("""
        Universal DOE Platform에 가입해주셔서 감사합니다!
        
        이제 다음과 같은 기능을 사용하실 수 있습니다:
        - 🧪 AI 기반 실험 설계
        - 📊 데이터 분석 및 시각화
        - 🔍 문헌 검색 및 요약
        - 👥 팀 협업 기능
        """)
        
        # 프로필 이미지 업로드 옵션
        with st.expander("프로필 이미지 설정 (선택)", expanded=False):
            uploaded_file = st.file_uploader(
                "프로필 이미지를 업로드하세요",
                type=['png', 'jpg', 'jpeg'],
                help="최대 5MB, 정사각형 이미지 권장"
            )
            
            if uploaded_file:
                # 이미지 처리 및 저장 로직
                pass
        
        # 로그인 버튼
        if st.button("로그인하러 가기", type="primary", use_container_width=True):
            # 가입 데이터 정리
            st.session_state.temp_data = {}
            st.session_state.signup_step = 0
            st.session_state.auth_mode = 'login'
            st.rerun()
    
    def _cancel_signup(self):
        """회원가입 취소"""
        st.session_state.temp_data = {}
        st.session_state.signup_step = 0
        st.session_state.auth_mode = 'login'
        st.rerun()
    
    def _check_email_exists(self, email: str) -> bool:
        """이메일 중복 확인"""
        # AuthManager를 통해 확인 (실제 구현 필요)
        return False  # 임시
    
    # =========================================================================
    # 🔑 비밀번호 찾기
    # =========================================================================
    
    def render_forgot_password_form(self):
        """비밀번호 찾기 폼"""
        st.markdown(f"### {self.texts['forgot_password']}")
        
        st.info("""
        가입하신 이메일 주소를 입력하시면 비밀번호 재설정 방법을 안내해드립니다.
        """)
        
        with st.form("forgot_password"):
            # 이메일 입력
            email = st.text_input(
                "이메일",
                placeholder=self.texts['email_placeholder']
            )
            
            # 캡차 (봇 방지)
            if not st.session_state.captcha_question:
                num1, num2 = secrets.randbelow(10), secrets.randbelow(10)
                st.session_state.captcha_question = (num1, num2)
                st.session_state.captcha_answer = num1 + num2
            
            num1, num2 = st.session_state.captcha_question
            captcha_input = st.number_input(
                f"🤖 자동 입력 방지: {num1} + {num2} = ?",
                min_value=0,
                max_value=99,
                step=1
            )
            
            # 버튼
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button(
                    "재설정 요청",
                    type="primary",
                    use_container_width=True
                ):
                    # 검증
                    if not email:
                        show_error("이메일을 입력해주세요.")
                    elif not validate_email(email):
                        show_error("올바른 이메일 형식이 아닙니다.")
                    elif captcha_input != st.session_state.captcha_answer:
                        show_error("자동 입력 방지 답이 틀렸습니다.")
                    else:
                        self._process_password_reset(email)
            
            with col2:
                if st.form_submit_button("취소", use_container_width=True):
                    st.session_state.auth_mode = 'login'
                    st.rerun()
    
    def _process_password_reset(self, email: str):
        """비밀번호 재설정 처리"""
        try:
            # 오프라인 모드에서는 다른 방식으로 처리
            if LOCAL_CONFIG.get('offline_mode', {}).get('default', True):
                show_info("""
                오프라인 모드에서는 이메일을 보낼 수 없습니다.
                
                비밀번호를 재설정하려면:
                1. 앱을 온라인 모드로 전환하거나
                2. 관리자에게 문의하세요
                """)
            else:
                # 온라인 모드: 이메일 발송 (구현 필요)
                with st.spinner("재설정 링크 발송 중..."):
                    time.sleep(1)  # 실제로는 이메일 발송
                
                show_success(f"""
                {email}로 비밀번호 재설정 안내를 발송했습니다.
                이메일을 확인해주세요.
                """)
            
            # 캡차 초기화
            st.session_state.captcha_question = None
            
        except Exception as e:
            logger.error(f"Password reset error: {str(e)}")
            show_error("비밀번호 재설정 요청 중 오류가 발생했습니다.")
    
    # =========================================================================
    # ⚙️ 프로필 관리
    # =========================================================================
    
    def render_profile_management(self):
        """프로필 관리 페이지"""
        if not is_authenticated():
            show_warning("로그인이 필요합니다.")
            st.session_state.auth_mode = 'login'
            st.rerun()
            return
        
        user = get_current_user()
        st.markdown(f"### {self.texts['profile_title']}")
        
        # 프로필 탭
        tab1, tab2, tab3, tab4 = st.tabs([
            "👤 기본 정보",
            "🔒 보안 설정",
            "🔑 API 키 관리",
            "🎨 환경 설정"
        ])
        
        with tab1:
            self._render_basic_info_tab(user)
        
        with tab2:
            self._render_security_tab(user)
        
        with tab3:
            self._render_api_keys_tab(user)
        
        with tab4:
            self._render_preferences_tab(user)
    
    def _render_basic_info_tab(self, user: Dict):
        """기본 정보 탭"""
        with st.form("profile_basic"):
            # 이름
            name = st.text_input(
                "이름",
                value=user.get('name', ''),
                placeholder="이름을 입력하세요"
            )
            
            # 소속
            organization = st.text_input(
                "소속 기관",
                value=user.get('organization', ''),
                placeholder="소속 기관을 입력하세요"
            )
            
            # 연구 분야
            research_field = st.selectbox(
                "주요 연구 분야",
                options=['선택하세요', '화학', '재료과학', '생명공학', '물리학', '기타'],
                index=0
            )
            
            # 자기소개
            bio = st.text_area(
                "자기소개",
                value=user.get('bio', ''),
                placeholder="간단한 자기소개를 작성해주세요",
                max_chars=500
            )
            
            # 저장 버튼
            if st.form_submit_button("저장", type="primary"):
                success, message = self.auth_manager.update_user_profile(
                    user['id'],
                    {
                        'name': name,
                        'organization': organization,
                        'research_field': research_field,
                        'bio': bio
                    }
                )
                
                if success:
                    show_success("프로필이 업데이트되었습니다.")
                    # 세션 업데이트
                    st.session_state.user.update({
                        'name': name,
                        'organization': organization,
                        'research_field': research_field,
                        'bio': bio
                    })
                else:
                    show_error(message)
    
    def _render_security_tab(self, user: Dict):
        """보안 설정 탭"""
        st.markdown("#### 비밀번호 변경")
        
        with st.form("change_password"):
            # 현재 비밀번호
            current_password = st.text_input(
                "현재 비밀번호",
                type="password"
            )
            
            # 새 비밀번호
            new_password = st.text_input(
                "새 비밀번호",
                type="password",
                help=f"최소 {SECURITY_CONFIG['password_min_length']}자 이상"
            )
            
            # 새 비밀번호 확인
            confirm_password = st.text_input(
                "새 비밀번호 확인",
                type="password"
            )
            
            # 비밀번호 강도 표시
            if new_password:
                self._render_password_strength(new_password)
            
            # 변경 버튼
            if st.form_submit_button("비밀번호 변경", type="primary"):
                # 검증
                if not all([current_password, new_password, confirm_password]):
                    show_error("모든 필드를 입력해주세요.")
                elif new_password != confirm_password:
                    show_error("새 비밀번호가 일치하지 않습니다.")
                else:
                    is_valid, message = validate_password(new_password)
                    if not is_valid:
                        show_error(message)
                    else:
                        # 비밀번호 변경 처리
                        success, message = self.auth_manager.change_password(
                            user['id'],
                            current_password,
                            new_password
                        )
                        
                        if success:
                            show_success("비밀번호가 변경되었습니다.")
                        else:
                            show_error(message)
        
        # 보안 설정
        st.markdown("#### 보안 설정")
        
        # 2단계 인증 (향후 구현)
        two_factor = st.checkbox(
            "2단계 인증 사용",
            value=False,
            disabled=True,
            help="향후 업데이트에서 지원 예정"
        )
        
        # 로그인 알림
        login_notification = st.checkbox(
            "로그인 알림 받기",
            value=user.get('settings', {}).get('login_notification', True),
            help="새로운 기기에서 로그인 시 이메일 알림"
        )
        
        if st.button("보안 설정 저장"):
            # 설정 저장 (구현 필요)
            show_success("보안 설정이 저장되었습니다.")
    
    def _render_api_keys_tab(self, user: Dict):
        """API 키 관리 탭"""
        st.markdown("#### API 키 관리")
        st.info("""
        외부 AI 서비스를 사용하려면 API 키를 등록해주세요.
        모든 API 키는 안전하게 암호화되어 저장됩니다.
        """)
        
        # 지원 서비스 목록
        services = [
            ('google_gemini', 'Google Gemini', True),
            ('groq', 'Groq', False),
            ('xai_grok', 'xAI Grok', False),
            ('deepseek', 'DeepSeek', False),
            ('sambanova', 'SambaNova', False),
            ('huggingface', 'HuggingFace', False)
        ]
        
        for service_id, service_name, is_required in services:
            st.markdown(f"##### {service_name} {'(필수)' if is_required else '(선택)'}")
            
            # 현재 키 상태
            has_key = self.auth_manager.get_api_key(user['id'], service_id) is not None
            
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                if has_key:
                    st.success("✅ API 키가 등록되어 있습니다")
                else:
                    st.warning("⚠️ API 키가 등록되지 않았습니다")
            
            with col2:
                if st.button("변경" if has_key else "등록", key=f"api_{service_id}"):
                    st.session_state[f"show_api_input_{service_id}"] = True
            
            with col3:
                if has_key and st.button("삭제", key=f"del_{service_id}"):
                    # API 키 삭제 (구현 필요)
                    show_info(f"{service_name} API 키가 삭제되었습니다.")
            
            # API 키 입력 폼
            if st.session_state.get(f"show_api_input_{service_id}"):
                with st.form(f"api_form_{service_id}"):
                    api_key = st.text_input(
                        "API 키",
                        type="password",
                        placeholder="API 키를 입력하세요"
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.form_submit_button("저장", type="primary"):
                            if api_key:
                                success = self.auth_manager.save_api_key(
                                    user['id'],
                                    service_id,
                                    api_key
                                )
                                if success:
                                    show_success(f"{service_name} API 키가 저장되었습니다.")
                                    st.session_state[f"show_api_input_{service_id}"] = False
                                    st.rerun()
                                else:
                                    show_error("API 키 저장에 실패했습니다.")
                    
                    with col2:
                        if st.form_submit_button("취소"):
                            st.session_state[f"show_api_input_{service_id}"] = False
                            st.rerun()
            
            st.markdown("---")
    
    def _render_preferences_tab(self, user: Dict):
        """환경 설정 탭"""
        st.markdown("#### 환경 설정")
        
        settings = user.get('settings', {})
        
        # 언어 설정
        language = st.selectbox(
            "언어",
            options=['한국어', 'English'],
            index=0 if settings.get('language', 'ko') == 'ko' else 1
        )
        
        # 테마 설정
        theme = st.selectbox(
            "테마",
            options=['라이트', '다크', '시스템 설정 따름'],
            index=['light', 'dark', 'auto'].index(settings.get('theme', 'light'))
        )
        
        # 알림 설정
        st.markdown("##### 알림 설정")
        
        notifications = settings.get('notifications', {})
        
        email_notifications = st.checkbox(
            "이메일 알림",
            value=notifications.get('email', True),
            help="중요한 알림을 이메일로 받습니다"
        )
        
        project_updates = st.checkbox(
            "프로젝트 업데이트",
            value=notifications.get('project_updates', True),
            help="프로젝트 상태 변경 알림"
        )
        
        collaboration_alerts = st.checkbox(
            "협업 알림",
            value=notifications.get('collaboration', True),
            help="팀원의 활동 알림"
        )
        
        # 저장 버튼
        if st.button("환경 설정 저장", type="primary"):
            new_settings = {
                'language': 'ko' if language == '한국어' else 'en',
                'theme': ['light', 'dark', 'auto'][['라이트', '다크', '시스템 설정 따름'].index(theme)],
                'notifications': {
                    'email': email_notifications,
                    'project_updates': project_updates,
                    'collaboration': collaboration_alerts
                }
            }
            
            # 설정 저장 (구현 필요)
            success, message = self.auth_manager.update_user_profile(
                user['id'],
                {'settings': json.dumps(new_settings)}
            )
            
            if success:
                show_success("환경 설정이 저장되었습니다.")
                # 세션 업데이트
                st.session_state.user['settings'] = new_settings
                
                # 언어 변경 시 페이지 새로고침
                if new_settings['language'] != st.session_state.get('language', 'ko'):
                    st.session_state.language = new_settings['language']
                    st.rerun()
            else:
                show_error(message)
    
    # =========================================================================
    # 🎯 메인 렌더링
    # =========================================================================
    
    def render(self):
        """메인 렌더링 함수"""
        # 이미 로그인된 경우
        if is_authenticated() and st.session_state.auth_mode != 'profile':
            # 프로필 관리가 아니면 대시보드로
            st.session_state.current_page = 'dashboard'
            st.rerun()
            return
        
        # 인증 모드에 따른 렌더링
        mode = st.session_state.auth_mode
        
        if mode == 'login':
            self.render_login_form()
        elif mode == 'signup':
            self.render_signup_form()
        elif mode == 'forgot':
            self.render_forgot_password_form()
        elif mode == 'profile':
            self.render_profile_management()
        else:
            # 기본: 로그인
            self.render_login_form()
        
        # 하단 링크
        if mode != 'profile':
            self._render_footer_links()
    
    def _render_footer_links(self):
        """하단 링크"""
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("🏠 [홈페이지](https://universaldoe.com)")
        
        with col2:
            st.markdown("📚 [사용 가이드](https://docs.universaldoe.com)")
        
        with col3:
            st.markdown("💬 [고객 지원](mailto:support@universaldoe.com)")


# ===========================================================================
# 🚀 페이지 진입점
# ===========================================================================

def render():
    """페이지 렌더링 진입점"""
    auth_page = AuthPage()
    auth_page.render()


# 개발 모드에서 직접 실행
if __name__ == "__main__":
    # 페이지 설정
    st.set_page_config(
        page_title="Universal DOE - 로그인",
        page_icon="🔐",
        layout="centered"
    )
    
    # 렌더링
    render()
