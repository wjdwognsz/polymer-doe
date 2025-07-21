#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧬 Universal DOE Platform - Main Application
================================================================================
Version: 2.0.0
Description: AI-powered universal experiment design platform for all science fields
Author: Universal DOE Research Team
License: MIT
================================================================================
"""

# ==================== 표준 라이브러리 ====================
import streamlit as st
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import os
from pathlib import Path
import traceback
import time
import sys

# ==================== 로깅 설정 ====================
# 로그 디렉토리 생성
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'app.log', mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== 앱 메타데이터 ====================
APP_VERSION = "2.0.0"
APP_NAME = "Universal DOE Platform"
APP_DESCRIPTION = "모든 과학 분야를 위한 AI 기반 실험 설계 플랫폼"
MIN_PASSWORD_LENGTH = 8
SESSION_TIMEOUT_MINUTES = 30

# ==================== 페이지 구성 ====================
PAGES = {
    'auth': {
        'title': '🔐 로그인',
        'icon': '🔐',
        'description': '플랫폼 접속'
    },
    'dashboard': {
        'title': '📊 대시보드',
        'icon': '📊',
        'description': '개인 홈',
        'requires_auth': True
    },
    'project_setup': {
        'title': '📝 프로젝트 설정',
        'icon': '📝',
        'description': '새 프로젝트 생성',
        'requires_auth': True
    },
    'experiment_design': {
        'title': '🧪 실험 설계',
        'icon': '🧪',
        'description': 'AI 기반 실험 설계',
        'requires_auth': True
    },
    'data_analysis': {
        'title': '📈 데이터 분석',
        'icon': '📈',
        'description': '결과 분석 및 시각화',
        'requires_auth': True
    },
    'literature_search': {
        'title': '🔍 문헌 검색',
        'icon': '🔍',
        'description': 'AI 문헌 검색 및 요약',
        'requires_auth': True
    },
    'collaboration': {
        'title': '👥 협업',
        'icon': '👥',
        'description': '팀 협업 공간',
        'requires_auth': True
    },
    'visualization': {
        'title': '📊 시각화',
        'icon': '📊',
        'description': '데이터 시각화',
        'requires_auth': True
    },
    'module_marketplace': {
        'title': '🛍️ 모듈 마켓',
        'icon': '🛍️',
        'description': '커뮤니티 모듈',
        'requires_auth': True
    },
    'settings': {
        'title': '⚙️ 설정',
        'icon': '⚙️',
        'description': '개인 설정',
        'requires_auth': True
    }
}

# ==================== 연구 분야 정의 ====================
RESEARCH_FIELDS = {
    'polymer': {
        'name': '🧬 고분자 과학',
        'description': '고분자 합성, 가공, 특성분석',
        'modules': ['polymer_synthesis', 'polymer_processing', 'polymer_characterization']
    },
    'inorganic': {
        'name': '🔷 무기재료',
        'description': '세라믹, 반도체, 금속 재료',
        'modules': ['ceramic_synthesis', 'semiconductor_processing', 'metal_alloys']
    },
    'nano': {
        'name': '🔬 나노재료',
        'description': '나노입자, 나노구조체, 나노복합체',
        'modules': ['nanoparticle_synthesis', 'nanostructure_fabrication']
    },
    'organic': {
        'name': '⚗️ 유기합성',
        'description': '유기 반응, 촉매, 천연물',
        'modules': ['organic_reactions', 'catalysis', 'natural_products']
    },
    'composite': {
        'name': '🔲 복합재료',
        'description': '섬유강화, 입자강화 복합재료',
        'modules': ['fiber_composites', 'particle_composites', 'hybrid_composites']
    },
    'bio': {
        'name': '🧫 바이오재료',
        'description': '생체적합성, 약물전달, 조직공학',
        'modules': ['biocompatibility', 'drug_delivery', 'tissue_engineering']
    },
    'energy': {
        'name': '🔋 에너지재료',
        'description': '배터리, 연료전지, 태양전지',
        'modules': ['batteries', 'fuel_cells', 'solar_cells']
    },
    'environmental': {
        'name': '🌱 환경재료',
        'description': '수처리, 대기정화, 재활용',
        'modules': ['water_treatment', 'air_purification', 'recycling']
    },
    'general': {
        'name': '🔬 일반 실험',
        'description': '범용 실험 설계',
        'modules': ['general_experiment', 'optimization', 'screening']
    }
}

# ==================== 클래스 정의 ====================

class UniversalDOEApp:
    """Universal DOE Platform 메인 애플리케이션 클래스"""
    
    def __init__(self):
        """앱 초기화"""
        self.setup_complete = False
        self.module_registry = None
        self.auth_manager = None
        self.api_manager = None
        self.notification_manager = None
        self.data_processor = None
        
    def initialize_imports(self):
        """동적 임포트 수행"""
        try:
            # 기본 모듈이 없어도 앱이 실행되도록 처리
            modules_to_import = {
                'pages.auth_page': 'AuthPage',
                'pages.dashboard_page': 'DashboardPage',
                'pages.project_setup': 'ProjectSetupPage',
                'pages.experiment_design': 'ExperimentDesignPage',
                'pages.data_analysis': 'DataAnalysisPage',
                'pages.literature_search': 'LiteratureSearchPage',
                'pages.visualization': 'VisualizationPage',
                'pages.collaboration': 'CollaborationPage',
                'pages.module_marketplace': 'ModuleMarketplacePage',
                'utils.auth_manager': 'GoogleSheetsAuthManager',
                'utils.sheets_manager': 'GoogleSheetsManager',
                'utils.api_manager': 'APIManager',
                'utils.common_ui': ['setup_page_config', 'apply_custom_css', 
                                   'render_header', 'render_footer', 'show_notification'],
                'utils.notification_manager': 'NotificationManager',
                'utils.data_processor': 'DataProcessor',
                'modules.module_registry': 'ModuleRegistry',
                'config.app_config': 'APP_CONFIG',
                'config.theme_config': 'THEME_CONFIG'
            }
            
            self.imported_modules = {}
            
            for module_path, class_names in modules_to_import.items():
                try:
                    if isinstance(class_names, list):
                        module = __import__(module_path, fromlist=class_names)
                        for class_name in class_names:
                            self.imported_modules[class_name] = getattr(module, class_name)
                    else:
                        module = __import__(module_path, fromlist=[class_names])
                        self.imported_modules[class_names] = getattr(module, class_names)
                except ImportError as e:
                    logger.warning(f"모듈 임포트 실패 ({module_path}): {e}")
                    # 기본 더미 클래스 제공
                    self.imported_modules[class_names if isinstance(class_names, str) else class_names[0]] = None
                    
            return True
            
        except Exception as e:
            logger.error(f"임포트 초기화 실패: {e}")
            return False
            
    def initialize_session_state(self):
        """세션 상태 초기화"""
        # 기본 세션 상태
        defaults = {
            # 인증 관련
            'authenticated': False,
            'user': None,
            'user_id': None,
            'guest_mode': False,
            'session_id': str(uuid.uuid4()),
            'login_time': None,
            'last_activity': datetime.now(),
            
            # 앱 상태
            'current_page': 'auth',
            'previous_page': None,
            'page_params': {},
            
            # 프로젝트 관련
            'current_project': None,
            'projects': [],
            'selected_field': None,
            'selected_modules': [],
            
            # UI 상태
            'sidebar_state': 'expanded',
            'theme': 'light',
            'language': 'ko',
            
            # API 키
            'api_keys': {},
            'api_keys_validated': {},
            
            # 알림
            'notifications': [],
            'unread_notifications': 0,
            
            # 모듈 관련
            'module_registry_initialized': False,
            'available_modules': {},
            'loaded_modules': {},
            
            # 임시 데이터
            'temp_data': {},
            'form_data': {},
            
            # 에러 상태
            'last_error': None,
            'error_count': 0
        }
        
        # 초기화되지 않은 상태만 설정
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
                
        logger.info(f"세션 상태 초기화 완료 (세션 ID: {st.session_state.session_id})")
        
    def setup_page_config(self):
        """Streamlit 페이지 설정"""
        try:
            st.set_page_config(
                page_title=APP_NAME,
                page_icon="🧬",
                layout="wide",
                initial_sidebar_state="expanded",
                menu_items={
                    'Get Help': 'https://github.com/yourusername/universal-doe-platform',
                    'Report a bug': 'https://github.com/yourusername/universal-doe-platform/issues',
                    'About': f"{APP_NAME} v{APP_VERSION} - {APP_DESCRIPTION}"
                }
            )
            
            # CSS 스타일 적용
            self.apply_custom_styles()
            
        except Exception as e:
            logger.error(f"페이지 설정 실패: {e}")
            
    def apply_custom_styles(self):
        """커스텀 CSS 스타일 적용"""
        st.markdown("""
        <style>
        /* 메인 컨테이너 스타일 */
        .main {
            padding: 0rem 1rem;
        }
        
        /* 사이드바 스타일 */
        .css-1d391kg {
            padding-top: 1rem;
        }
        
        /* 버튼 스타일 */
        .stButton > button {
            width: 100%;
            border-radius: 0.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 10px rgba(0,0,0,0.2);
        }
        
        /* 메트릭 카드 스타일 */
        [data-testid="metric-container"] {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        /* 알림 스타일 */
        .notification-badge {
            background-color: #ff4b4b;
            color: white;
            border-radius: 50%;
            padding: 0.2rem 0.5rem;
            font-size: 0.8rem;
            font-weight: bold;
        }
        
        /* 모듈 카드 스타일 */
        .module-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 1rem;
            color: white;
            margin-bottom: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .module-card:hover {
            transform: scale(1.02);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }
        
        /* 애니메이션 */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .animate-fadeIn {
            animation: fadeIn 0.5s ease-out;
        }
        </style>
        """, unsafe_allow_html=True)
        
    def initialize_components(self):
        """컴포넌트 초기화"""
        try:
            # 모듈 레지스트리 초기화
            if self.imported_modules.get('ModuleRegistry'):
                self.module_registry = self.imported_modules['ModuleRegistry']()
                
                # 사용자 모듈 발견
                if st.session_state.user_id:
                    discovered = self.module_registry.discover_modules(st.session_state.user_id)
                    st.session_state.available_modules = discovered
                    st.session_state.module_registry_initialized = True
                    logger.info(f"모듈 발견 완료: {discovered}")
            
            # 인증 관리자 초기화
            if self.imported_modules.get('GoogleSheetsAuthManager'):
                self.auth_manager = self.imported_modules['GoogleSheetsAuthManager']()
                
            # API 관리자 초기화
            if self.imported_modules.get('APIManager'):
                self.api_manager = self.imported_modules['APIManager']()
                
            # 알림 관리자 초기화
            if self.imported_modules.get('NotificationManager'):
                self.notification_manager = self.imported_modules['NotificationManager']()
                
            # 데이터 프로세서 초기화
            if self.imported_modules.get('DataProcessor'):
                self.data_processor = self.imported_modules['DataProcessor']()
                
            self.setup_complete = True
            logger.info("컴포넌트 초기화 완료")
            
        except Exception as e:
            logger.error(f"컴포넌트 초기화 실패: {e}")
            st.error(f"시스템 초기화 중 오류가 발생했습니다: {str(e)}")
            
    def check_session_timeout(self):
        """세션 타임아웃 확인"""
        if st.session_state.authenticated and st.session_state.login_time:
            elapsed = datetime.now() - st.session_state.login_time
            if elapsed > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
                logger.info(f"세션 타임아웃 (사용자: {st.session_state.user_id})")
                self.logout()
                st.warning("세션이 만료되었습니다. 다시 로그인해주세요.")
                return False
                
        # 활동 시간 업데이트
        st.session_state.last_activity = datetime.now()
        return True
        
    def render_header(self):
        """헤더 렌더링"""
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col1:
            if st.button("🏠 홈", key="home_btn"):
                st.session_state.current_page = 'dashboard' if st.session_state.authenticated else 'auth'
                st.rerun()
                
        with col2:
            st.markdown(f"""
            <h1 style='text-align: center; color: #7C3AED; margin: 0;'>
                {APP_NAME}
            </h1>
            <p style='text-align: center; color: #6B7280; margin: 0; font-size: 0.9em;'>
                {APP_DESCRIPTION}
            </p>
            """, unsafe_allow_html=True)
            
        with col3:
            if st.session_state.authenticated:
                # 알림 아이콘
                notif_count = st.session_state.unread_notifications
                if notif_count > 0:
                    st.markdown(f"""
                    <div style='text-align: right;'>
                        🔔 <span class='notification-badge'>{notif_count}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
    def render_sidebar(self):
        """사이드바 렌더링"""
        with st.sidebar:
            # 로고 및 타이틀
            st.markdown(f"""
            <div style='text-align: center; padding: 1rem 0;'>
                <h2 style='color: #7C3AED; margin: 0;'>🧬 {APP_NAME}</h2>
                <p style='color: #6B7280; font-size: 0.8em; margin: 0;'>v{APP_VERSION}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.divider()
            
            # 사용자 정보 (인증된 경우)
            if st.session_state.authenticated:
                self.render_user_profile()
                st.divider()
                
            # 네비게이션 메뉴
            self.render_navigation_menu()
            
            # 하단 정보
            st.divider()
            self.render_sidebar_footer()
            
    def render_user_profile(self):
        """사용자 프로필 렌더링"""
        user = st.session_state.user
        if not user:
            return
            
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # 아바타
            st.markdown(f"""
            <div style='
                width: 50px; 
                height: 50px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
                font-size: 1.5rem;
            '>
                {user.get('name', '?')[0].upper()}
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"**{user.get('name', '사용자')}**")
            level = user.get('level', 'beginner')
            level_emoji = {'beginner': '🌱', 'intermediate': '🌿', 'advanced': '🌳', 'expert': '🏆'}
            st.caption(f"{level_emoji.get(level, '🌱')} {level.title()}")
            
        # 빠른 통계
        col1, col2 = st.columns(2)
        with col1:
            st.metric("프로젝트", len(st.session_state.projects))
        with col2:
            st.metric("실험", user.get('experiment_count', 0))
            
    def render_navigation_menu(self):
        """네비게이션 메뉴 렌더링"""
        st.markdown("### 📍 메뉴")
        
        # 현재 페이지에 따른 메뉴 항목 필터링
        menu_items = []
        
        # 인증되지 않은 경우 로그인 페이지만
        if not st.session_state.authenticated and not st.session_state.guest_mode:
            menu_items = ['auth']
        else:
            # 인증된 경우 모든 메뉴
            for page_key, page_info in PAGES.items():
                if page_key == 'auth':
                    continue  # 로그인된 상태에서는 auth 페이지 숨김
                    
                # 게스트 모드에서는 일부 기능만
                if st.session_state.guest_mode and page_key not in ['dashboard', 'literature_search', 'visualization']:
                    continue
                    
                menu_items.append(page_key)
            
        # 메뉴 렌더링
        for page_key in menu_items:
            page_info = PAGES[page_key]
            
            # 현재 페이지 하이라이트
            is_current = st.session_state.current_page == page_key
            button_type = "primary" if is_current else "secondary"
            
            if st.button(
                f"{page_info['icon']} {page_info['title']}", 
                key=f"nav_{page_key}",
                use_container_width=True,
                type=button_type
            ):
                st.session_state.previous_page = st.session_state.current_page
                st.session_state.current_page = page_key
                st.rerun()
                
    def render_sidebar_footer(self):
        """사이드바 하단 정보"""
        if st.session_state.authenticated:
            if st.button("🚪 로그아웃", key="logout_btn", use_container_width=True):
                self.logout()
                st.rerun()
        else:
            if st.session_state.guest_mode:
                if st.button("🔐 로그인하기", key="login_btn", use_container_width=True):
                    st.session_state.guest_mode = False
                    st.session_state.current_page = 'auth'
                    st.rerun()
                    
        # 도움말 링크
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; font-size: 0.8em; color: #6B7280;'>
            <a href='#' style='text-decoration: none; color: #6B7280;'>📖 도움말</a> | 
            <a href='#' style='text-decoration: none; color: #6B7280;'>📧 문의</a>
        </div>
        """, unsafe_allow_html=True)
        
    def render_page_router(self):
        """페이지 라우팅"""
        current_page = st.session_state.current_page
        
        # 페이지별 렌더링
        if current_page == 'auth':
            self.render_auth_page()
        elif current_page == 'dashboard':
            self.render_dashboard_page()
        elif current_page == 'project_setup':
            self.render_project_setup_page()
        elif current_page == 'experiment_design':
            self.render_experiment_design_page()
        elif current_page == 'data_analysis':
            self.render_data_analysis_page()
        elif current_page == 'literature_search':
            self.render_literature_search_page()
        elif current_page == 'collaboration':
            self.render_collaboration_page()
        elif current_page == 'visualization':
            self.render_visualization_page()
        elif current_page == 'module_marketplace':
            self.render_module_marketplace_page()
        elif current_page == 'settings':
            self.render_settings_page()
        else:
            st.error(f"알 수 없는 페이지: {current_page}")
            
    def render_auth_page(self):
        """인증 페이지"""
        if self.imported_modules.get('AuthPage'):
            page = self.imported_modules['AuthPage']()
            page.render()
        else:
            # 기본 인증 UI
            st.title("🔐 로그인")
            
            tab1, tab2 = st.tabs(["로그인", "회원가입"])
            
            with tab1:
                self.render_login_form()
                
            with tab2:
                self.render_signup_form()
                
            # 게스트 모드
            st.divider()
            if st.button("🔍 둘러보기 (게스트 모드)", use_container_width=True):
                st.session_state.guest_mode = True
                st.session_state.current_page = 'dashboard'
                st.rerun()
                
    def render_login_form(self):
        """로그인 폼"""
        with st.form("login_form"):
            email = st.text_input("이메일", placeholder="your@email.com")
            password = st.text_input("비밀번호", type="password")
            remember_me = st.checkbox("자동 로그인")
            
            col1, col2 = st.columns(2)
            with col1:
                submit = st.form_submit_button("로그인", use_container_width=True, type="primary")
            with col2:
                google_login = st.form_submit_button("Google 로그인", use_container_width=True)
                
        if submit:
            if email and password:
                # 인증 시도
                if self.auth_manager:
                    success, user_data = self.auth_manager.authenticate(email, password)
                    if success:
                        self.login_success(user_data)
                    else:
                        st.error("이메일 또는 비밀번호가 올바르지 않습니다.")
                else:
                    # 테스트용 로그인
                    if email == "test@example.com" and password == "test123":
                        self.login_success({
                            'id': 'test_user',
                            'email': email,
                            'name': 'Test User',
                            'level': 'intermediate'
                        })
                    else:
                        st.error("인증 시스템을 사용할 수 없습니다.")
                        
    def render_signup_form(self):
        """회원가입 폼"""
        with st.form("signup_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("이름", placeholder="홍길동")
                email = st.text_input("이메일", placeholder="your@email.com")
            with col2:
                password = st.text_input("비밀번호", type="password", help=f"최소 {MIN_PASSWORD_LENGTH}자 이상")
                password_confirm = st.text_input("비밀번호 확인", type="password")
                
            organization = st.text_input("소속 기관", placeholder="○○대학교 △△학과")
            
            research_field = st.selectbox(
                "주요 연구 분야",
                options=list(RESEARCH_FIELDS.keys()),
                format_func=lambda x: RESEARCH_FIELDS[x]['name']
            )
            
            experience_level = st.select_slider(
                "경험 수준",
                options=['beginner', 'intermediate', 'advanced', 'expert'],
                value='beginner',
                format_func=lambda x: {
                    'beginner': '🌱 초급',
                    'intermediate': '🌿 중급', 
                    'advanced': '🌳 고급',
                    'expert': '🏆 전문가'
                }[x]
            )
            
            terms = st.checkbox("서비스 이용약관 및 개인정보 처리방침에 동의합니다.")
            
            submit = st.form_submit_button("회원가입", use_container_width=True, type="primary")
            
        if submit:
            # 입력 검증
            errors = []
            if not all([name, email, password, password_confirm]):
                errors.append("모든 필드를 입력해주세요.")
            if password != password_confirm:
                errors.append("비밀번호가 일치하지 않습니다.")
            if len(password) < MIN_PASSWORD_LENGTH:
                errors.append(f"비밀번호는 최소 {MIN_PASSWORD_LENGTH}자 이상이어야 합니다.")
            if not terms:
                errors.append("이용약관에 동의해주세요.")
                
            if errors:
                for error in errors:
                    st.error(error)
            else:
                # 회원가입 처리
                if self.auth_manager:
                    success, message = self.auth_manager.register(
                        email=email,
                        password=password,
                        name=name,
                        organization=organization,
                        research_field=research_field,
                        experience_level=experience_level
                    )
                    if success:
                        st.success("회원가입이 완료되었습니다! 로그인해주세요.")
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.info("회원가입 시스템 준비 중입니다.")
                    
    def login_success(self, user_data: Dict[str, Any]):
        """로그인 성공 처리"""
        st.session_state.authenticated = True
        st.session_state.user = user_data
        st.session_state.user_id = user_data.get('id')
        st.session_state.login_time = datetime.now()
        st.session_state.current_page = 'dashboard'
        
        # 모듈 레지스트리 초기화
        if self.module_registry and st.session_state.user_id:
            self.module_registry.discover_modules(st.session_state.user_id)
            
        logger.info(f"로그인 성공: {user_data.get('email')}")
        st.success(f"환영합니다, {user_data.get('name')}님!")
        time.sleep(1)
        st.rerun()
        
    def logout(self):
        """로그아웃 처리"""
        # 세션 초기화
        for key in ['authenticated', 'user', 'user_id', 'login_time', 
                   'current_project', 'projects', 'selected_modules']:
            if key in st.session_state:
                del st.session_state[key]
                
        st.session_state.current_page = 'auth'
        logger.info("로그아웃 완료")
        
    def render_dashboard_page(self):
        """대시보드 페이지"""
        if not st.session_state.authenticated and not st.session_state.guest_mode:
            st.warning("로그인이 필요합니다.")
            return
            
        if self.imported_modules.get('DashboardPage'):
            page = self.imported_modules['DashboardPage']()
            page.render()
        else:
            # 기본 대시보드
            st.title("📊 대시보드")
            
            if st.session_state.guest_mode:
                st.info("게스트 모드로 둘러보는 중입니다. 일부 기능이 제한됩니다.")
                
            # 환영 메시지
            user_name = st.session_state.user.get('name', '사용자') if st.session_state.user else '게스트'
            st.markdown(f"### 👋 안녕하세요, {user_name}님!")
            
            # 빠른 시작
            st.markdown("### 🚀 빠른 시작")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📝 새 프로젝트", use_container_width=True):
                    st.session_state.current_page = 'project_setup'
                    st.rerun()
                    
            with col2:
                if st.button("🔍 문헌 검색", use_container_width=True):
                    st.session_state.current_page = 'literature_search'
                    st.rerun()
                    
            with col3:
                if st.button("🛍️ 모듈 탐색", use_container_width=True):
                    st.session_state.current_page = 'module_marketplace'
                    st.rerun()
                    
            # 연구 분야 선택
            st.markdown("### 🔬 연구 분야 선택")
            field_cols = st.columns(3)
            for idx, (field_key, field_info) in enumerate(RESEARCH_FIELDS.items()):
                with field_cols[idx % 3]:
                    if st.button(
                        field_info['name'],
                        key=f"field_{field_key}",
                        use_container_width=True,
                        help=field_info['description']
                    ):
                        st.session_state.selected_field = field_key
                        st.session_state.current_page = 'project_setup'
                        st.rerun()
                        
    def render_project_setup_page(self):
        """프로젝트 설정 페이지"""
        if not st.session_state.authenticated:
            st.warning("로그인이 필요합니다.")
            return
            
        if self.imported_modules.get('ProjectSetupPage'):
            page = self.imported_modules['ProjectSetupPage']()
            page.render()
        else:
            st.title("📝 프로젝트 설정")
            st.info("프로젝트 설정 모듈을 준비 중입니다.")
            
    def render_experiment_design_page(self):
        """실험 설계 페이지"""
        if not st.session_state.authenticated:
            st.warning("로그인이 필요합니다.")
            return
            
        if self.imported_modules.get('ExperimentDesignPage'):
            page = self.imported_modules['ExperimentDesignPage']()
            page.render()
        else:
            st.title("🧪 실험 설계")
            st.info("실험 설계 모듈을 준비 중입니다.")
            
    def render_data_analysis_page(self):
        """데이터 분석 페이지"""
        if not st.session_state.authenticated:
            st.warning("로그인이 필요합니다.")
            return
            
        if self.imported_modules.get('DataAnalysisPage'):
            page = self.imported_modules['DataAnalysisPage']()
            page.render()
        else:
            st.title("📈 데이터 분석")
            st.info("데이터 분석 모듈을 준비 중입니다.")
            
    def render_literature_search_page(self):
        """문헌 검색 페이지"""
        if self.imported_modules.get('LiteratureSearchPage'):
            page = self.imported_modules['LiteratureSearchPage']()
            page.render()
        else:
            st.title("🔍 문헌 검색")
            st.info("AI 기반 문헌 검색 기능을 준비 중입니다.")
            
    def render_collaboration_page(self):
        """협업 페이지"""
        if not st.session_state.authenticated:
            st.warning("로그인이 필요합니다.")
            return
            
        if self.imported_modules.get('CollaborationPage'):
            page = self.imported_modules['CollaborationPage']()
            page.render()
        else:
            st.title("👥 협업")
            st.info("팀 협업 기능을 준비 중입니다.")
            
    def render_visualization_page(self):
        """시각화 페이지"""
        if not st.session_state.authenticated:
            st.warning("로그인이 필요합니다.")
            return
            
        if self.imported_modules.get('VisualizationPage'):
            page = self.imported_modules['VisualizationPage']()
            page.render()
        else:
            st.title("📊 시각화")
            st.info("데이터 시각화 도구를 준비 중입니다.")
            
    def render_module_marketplace_page(self):
        """모듈 마켓플레이스 페이지"""
        if not st.session_state.authenticated:
            st.warning("로그인이 필요합니다.")
            return
            
        if self.imported_modules.get('ModuleMarketplacePage'):
            page = self.imported_modules['ModuleMarketplacePage']()
            page.render()
        else:
            st.title("🛍️ 모듈 마켓플레이스")
            
            # 카테고리별 모듈 표시
            st.markdown("### 📦 사용 가능한 모듈")
            
            if self.module_registry:
                modules = self.module_registry.list_modules()
                
                if modules:
                    for module in modules:
                        with st.expander(f"{module['display_name']} v{module['version']}"):
                            st.write(f"**작성자**: {module['author']}")
                            st.write(f"**카테고리**: {module['category']}")
                            st.write(f"**설명**: {module['description']}")
                            st.write(f"**평점**: {'⭐' * int(module['rating'])}")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("설치", key=f"install_{module['name']}"):
                                    st.success(f"{module['display_name']} 모듈을 설치했습니다!")
                            with col2:
                                if st.button("상세 정보", key=f"info_{module['name']}"):
                                    st.info("상세 정보 페이지 준비 중")
                else:
                    st.info("사용 가능한 모듈이 없습니다.")
            else:
                st.info("모듈 시스템을 초기화 중입니다.")
                
    def render_settings_page(self):
        """설정 페이지"""
        if not st.session_state.authenticated:
            st.warning("로그인이 필요합니다.")
            return
            
        st.title("⚙️ 설정")
        
        # API 키 설정
        with st.expander("🔑 API 키 설정"):
            st.info("Streamlit Secrets에 이미 설정했다면 건너뛰세요.")
            
            api_services = {
                'google_gemini': 'Google Gemini 2.0 Flash',
                'xai_grok': 'xAI Grok 3 Mini',
                'groq': 'Groq (초고속 추론)',
                'deepseek': 'DeepSeek (코드/수식)',
                'sambanova': 'SambaNova (대규모 모델)',
                'huggingface': 'HuggingFace (특수 모델)'
            }
            
            for service_key, service_name in api_services.items():
                current_key = st.session_state.api_keys.get(service_key, '')
                new_key = st.text_input(
                    f"{service_name} API Key",
                    value='*' * 20 if current_key else '',
                    type="password",
                    key=f"api_key_{service_key}"
                )
                if new_key and new_key != '*' * 20:
                    st.session_state.api_keys[service_key] = new_key
                    
            if st.button("API 키 저장", use_container_width=True):
                st.success("API 키가 저장되었습니다!")
                
        # 프로필 설정
        with st.expander("👤 프로필 설정"):
            if st.session_state.user:
                user = st.session_state.user
                
                name = st.text_input("이름", value=user.get('name', ''))
                organization = st.text_input("소속", value=user.get('organization', ''))
                
                research_field = st.selectbox(
                    "주요 연구 분야",
                    options=list(RESEARCH_FIELDS.keys()),
                    format_func=lambda x: RESEARCH_FIELDS[x]['name'],
                    index=list(RESEARCH_FIELDS.keys()).index(user.get('research_field', 'general'))
                )
                
                if st.button("프로필 업데이트", use_container_width=True):
                    # 프로필 업데이트 로직
                    st.success("프로필이 업데이트되었습니다!")
                    
        # UI 설정
        with st.expander("🎨 UI 설정"):
            theme = st.radio("테마", ["light", "dark"], index=0 if st.session_state.theme == 'light' else 1)
            if theme != st.session_state.theme:
                st.session_state.theme = theme
                st.info("테마가 변경되었습니다. 새로고침하면 적용됩니다.")
                
            language = st.selectbox("언어", ["한국어", "English"], index=0)
            
    def run(self):
        """메인 앱 실행"""
        try:
            # 페이지 설정
            self.setup_page_config()
            
            # 세션 상태 초기화
            self.initialize_session_state()
            
            # 컴포넌트 초기화 (한 번만)
            if not self.setup_complete:
                if self.initialize_imports():
                    self.initialize_components()
            
            # 세션 타임아웃 확인
            if not self.check_session_timeout():
                st.stop()
                
            # 헤더 렌더링
            self.render_header()
            
            # 사이드바 렌더링
            self.render_sidebar()
            
            # 메인 컨텐츠 영역
            self.render_page_router()
            
            # 푸터 렌더링
            self.render_footer()
            
        except Exception as e:
            logger.error(f"앱 실행 중 오류: {e}\n{traceback.format_exc()}")
            st.error(f"시스템 오류가 발생했습니다: {str(e)}")
            
            # 에러 리포트
            with st.expander("🐛 오류 상세 정보"):
                st.code(traceback.format_exc())
                
    def render_footer(self):
        """푸터 렌더링"""
        st.divider()
        st.markdown(f"""
        <div style='text-align: center; color: #6B7280; font-size: 0.8em; padding: 2rem 0;'>
            <p>{APP_NAME} v{APP_VERSION} | © 2024 Universal DOE Research Team</p>
            <p>
                <a href='#' style='color: #6B7280; text-decoration: none;'>이용약관</a> | 
                <a href='#' style='color: #6B7280; text-decoration: none;'>개인정보처리방침</a> | 
                <a href='#' style='color: #6B7280; text-decoration: none;'>문의하기</a>
            </p>
        </div>
        """, unsafe_allow_html=True)


# ==================== 메인 실행 ====================

def main():
    """메인 함수"""
    app = UniversalDOEApp()
    app.run()


if __name__ == "__main__":
    main()
