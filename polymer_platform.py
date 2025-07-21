"""
🧬 Universal DOE Platform - 메인 애플리케이션
고분자 연구자를 위한 AI 기반 실험 설계 플랫폼
"""

import streamlit as st
import os
import sys
import logging
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Type
import json
import uuid
import importlib

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 전역 상수
APP_NAME = "Universal DOE Platform"
APP_VERSION = "2.0.0"
APP_DESCRIPTION = "모든 고분자 연구자를 위한 AI 기반 실험 설계 플랫폼"
SESSION_TIMEOUT_MINUTES = 30

# 페이지 정의
PAGES = {
    'auth': {
        'title': '로그인',
        'icon': '🔐',
        'module': 'pages.auth_page',
        'class': 'AuthPage',
        'public': True
    },
    'dashboard': {
        'title': '대시보드',
        'icon': '📊',
        'module': 'pages.dashboard_page',
        'class': 'DashboardPage',
        'public': False
    },
    'project_setup': {
        'title': '프로젝트 설정',
        'icon': '📝',
        'module': 'pages.project_setup',
        'class': 'ProjectSetupPage',
        'public': False
    },
    'experiment_design': {
        'title': '실험 설계',
        'icon': '🧪',
        'module': 'pages.experiment_design',
        'class': 'ExperimentDesignPage',
        'public': False
    },
    'data_analysis': {
        'title': '데이터 분석',
        'icon': '📈',
        'module': 'pages.data_analysis',
        'class': 'DataAnalysisPage',
        'public': False
    },
    'literature_search': {
        'title': '문헌 검색',
        'icon': '🔍',
        'module': 'pages.literature_search',
        'class': 'LiteratureSearchPage',
        'public': True
    },
    'collaboration': {
        'title': '협업',
        'icon': '👥',
        'module': 'pages.collaboration',
        'class': 'CollaborationPage',
        'public': False
    },
    'visualization': {
        'title': '시각화',
        'icon': '📊',
        'module': 'pages.visualization',
        'class': 'VisualizationPage',
        'public': True
    },
    'module_marketplace': {
        'title': '모듈 마켓플레이스',
        'icon': '🛍️',
        'module': 'pages.module_marketplace',
        'class': 'ModuleMarketplacePage',
        'public': False
    },
    'settings': {
        'title': '설정',
        'icon': '⚙️',
        'module': None,  # 내장 페이지
        'class': None,
        'public': False
    }
}

# 연구 분야 정의
RESEARCH_FIELDS = {
    'general': {
        'name': '🔬 일반 고분자',
        'description': '범용 고분자 합성 및 특성 분석'
    },
    'bio': {
        'name': '🧬 바이오 고분자',
        'description': '생체재료, 의료용 고분자'
    },
    'energy': {
        'name': '🔋 에너지 고분자',
        'description': '전지, 태양전지용 고분자'
    },
    'electronic': {
        'name': '💻 전자재료 고분자',
        'description': '반도체, 디스플레이용 고분자'
    },
    'composite': {
        'name': '🏗️ 복합재료',
        'description': '고분자 복합재료 및 나노복합재'
    },
    'sustainable': {
        'name': '♻️ 지속가능 고분자',
        'description': '생분해성, 재활용 고분자'
    }
}

class PolymerDOEApp:
    """메인 애플리케이션 클래스"""
    
    def __init__(self):
        self.imported_modules = {}
        self.module_registry = None
        self._initialize_app()
        
    def _initialize_app(self):
        """앱 초기화"""
        # 필수 디렉토리 생성
        required_dirs = ['data', 'logs', 'temp', 'modules/user_modules']
        for dir_name in required_dirs:
            dir_path = PROJECT_ROOT / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # 동적 모듈 임포트
        self._import_modules()
        
        # 모듈 레지스트리 초기화
        self._initialize_module_registry()
        
    def _import_modules(self):
        """필요한 모듈 동적 임포트"""
        # 유틸리티 모듈
        utils_imports = {
            'utils.common_ui': ['setup_page_config', 'apply_custom_css', 'render_header', 
                               'render_footer', 'show_notification'],
            'utils.auth_manager': 'GoogleSheetsAuthManager',
            'utils.sheets_manager': 'GoogleSheetsManager',
            'utils.api_manager': 'APIManager',
            'utils.notification_manager': 'NotificationManager',
            'utils.data_processor': 'DataProcessor'
        }
        
        # 설정 모듈
        config_imports = {
            'config.app_config': ['APP_CONFIG', 'API_CONFIGS'],
            'config.theme_config': 'THEME_CONFIG'
        }
        
        # 동적 임포트 실행
        for module_path, imports in {**utils_imports, **config_imports}.items():
            try:
                module = importlib.import_module(module_path)
                
                if isinstance(imports, list):
                    for item in imports:
                        if hasattr(module, item):
                            self.imported_modules[item] = getattr(module, item)
                        else:
                            logger.warning(f"{item} not found in {module_path}")
                else:
                    if hasattr(module, imports):
                        self.imported_modules[imports] = getattr(module, imports)
                    else:
                        logger.warning(f"{imports} not found in {module_path}")
                        
            except ImportError as e:
                logger.error(f"Failed to import {module_path}: {e}")
                # 기본값 제공
                if isinstance(imports, list):
                    for item in imports:
                        self.imported_modules[item] = None
                else:
                    self.imported_modules[imports] = None
                    
    def _initialize_module_registry(self):
        """모듈 레지스트리 초기화"""
        try:
            from modules.module_registry import ModuleRegistry
            self.module_registry = ModuleRegistry()
            st.session_state.module_registry_initialized = True
            logger.info("Module registry initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize module registry: {e}")
            st.session_state.module_registry_initialized = False
            
    def run(self):
        """메인 실행 함수"""
        try:
            # Streamlit 페이지 설정
            self.setup_page_config()
            
            # 세션 상태 초기화
            self.initialize_session_state()
            
            # 세션 유효성 검사
            if not self.check_session_validity():
                st.session_state.current_page = 'auth'
                
            # CSS 적용
            self.apply_custom_css()
            
            # 헤더 렌더링
            self.render_header()
            
            # 사이드바 렌더링
            self.render_sidebar()
            
            # 메인 콘텐츠 렌더링
            self.render_main_content()
            
            # 푸터 렌더링
            self.render_footer()
            
            # 백그라운드 작업
            self.run_background_tasks()
            
        except Exception as e:
            logger.error(f"Application error: {e}")
            st.error(f"애플리케이션 오류가 발생했습니다: {str(e)}")
            if st.button("🔄 새로고침"):
                st.rerun()
                
    def setup_page_config(self):
        """Streamlit 페이지 설정"""
        st.set_page_config(
            page_title=APP_NAME,
            page_icon="🧬",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/your-repo/polymer-doe',
                'Report a bug': 'https://github.com/your-repo/polymer-doe/issues',
                'About': APP_DESCRIPTION
            }
        )
        
    def initialize_session_state(self):
        """세션 상태 초기화"""
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
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
                
    def check_session_validity(self):
        """세션 유효성 검사"""
        if not st.session_state.authenticated:
            return True
            
        # 세션 타임아웃 검사
        if st.session_state.last_activity:
            time_since_activity = datetime.now() - st.session_state.last_activity
            if time_since_activity > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
                st.warning("세션이 만료되었습니다. 다시 로그인해주세요.")
                self.logout()
                return False
                
        # 활동 시간 업데이트
        st.session_state.last_activity = datetime.now()
        return True
        
    def apply_custom_css(self):
        """커스텀 CSS 적용"""
        css = """
        <style>
        /* 메인 컨테이너 */
        .main {
            padding-top: 2rem;
        }
        
        /* 헤더 스타일 */
        .app-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        /* 알림 배지 */
        .notification-badge {
            background: #ef4444;
            color: white;
            border-radius: 50%;
            padding: 0.25rem 0.5rem;
            font-size: 0.75rem;
            font-weight: bold;
        }
        
        /* 버튼 스타일 */
        .stButton > button {
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        
        /* 카드 스타일 */
        .info-card {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            margin-bottom: 1rem;
        }
        
        /* 사이드바 스타일 */
        .css-1d391kg {
            background-color: #f8f9fa;
        }
        
        /* 프로필 아바타 */
        .user-avatar {
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
        }
        
        /* 네비게이션 버튼 */
        .nav-button {
            width: 100%;
            text-align: left;
            padding: 0.5rem 1rem;
            margin-bottom: 0.25rem;
            border-radius: 5px;
            transition: all 0.2s ease;
        }
        
        .nav-button:hover {
            background-color: #e5e7eb;
        }
        
        .nav-button.active {
            background-color: #667eea;
            color: white;
        }
        
        /* 페이지 전환 애니메이션 */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .main-content {
            animation: fadeIn 0.3s ease-out;
        }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
        
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
                    if st.button(f"🔔 {notif_count}", key="notif_btn"):
                        st.session_state.show_notifications = not st.session_state.get('show_notifications', False)
                        
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
            elif st.session_state.guest_mode:
                st.info("👤 게스트 모드로 둘러보는 중")
                if st.button("🔐 로그인하기", use_container_width=True):
                    st.session_state.guest_mode = False
                    st.session_state.current_page = 'auth'
                    st.rerun()
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
            <div class='user-avatar'>
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
        
        # 인증되지 않은 경우
        if not st.session_state.authenticated and not st.session_state.guest_mode:
            menu_items = ['auth']
        else:
            # 인증되거나 게스트 모드인 경우
            for page_key, page_info in PAGES.items():
                if page_key == 'auth':
                    continue
                    
                # 게스트 모드 접근 제한
                if st.session_state.guest_mode and not page_info['public']:
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
        """사이드바 푸터 렌더링"""
        if st.session_state.authenticated:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("⚙️ 설정", use_container_width=True):
                    st.session_state.current_page = 'settings'
                    st.rerun()
                    
            with col2:
                if st.button("🚪 로그아웃", use_container_width=True):
                    self.logout()
                    
        # 도움말 링크
        st.markdown("""
        <div style='text-align: center; margin-top: 2rem;'>
            <a href='https://github.com/your-repo/polymer-doe/wiki' target='_blank'>📚 도움말</a> |
            <a href='https://github.com/your-repo/polymer-doe/issues' target='_blank'>🐛 버그 신고</a>
        </div>
        """, unsafe_allow_html=True)
        
    def render_main_content(self):
        """메인 콘텐츠 렌더링"""
        # 알림 표시
        if st.session_state.get('show_notifications', False):
            self.render_notifications()
            
        # 현재 페이지 렌더링
        current_page = st.session_state.current_page
        
        # 페이지별 렌더링 함수 매핑
        page_renderers = {
            'auth': self.render_auth_page,
            'dashboard': self.render_dashboard_page,
            'project_setup': self.render_project_setup_page,
            'experiment_design': self.render_experiment_design_page,
            'data_analysis': self.render_data_analysis_page,
            'literature_search': self.render_literature_search_page,
            'collaboration': self.render_collaboration_page,
            'visualization': self.render_visualization_page,
            'module_marketplace': self.render_module_marketplace_page,
            'settings': self.render_settings_page
        }
        
        # 페이지 렌더링
        st.markdown('<div class="main-content">', unsafe_allow_html=True)
        
        if current_page in page_renderers:
            try:
                page_renderers[current_page]()
            except Exception as e:
                logger.error(f"Error rendering page {current_page}: {e}")
                st.error(f"페이지 로딩 중 오류가 발생했습니다: {str(e)}")
                if st.button("다시 시도"):
                    st.rerun()
        else:
            st.error(f"페이지를 찾을 수 없습니다: {current_page}")
            
        st.markdown('</div>', unsafe_allow_html=True)
        
    def render_auth_page(self):
        """인증 페이지 렌더링"""
        # 동적 모듈 로드 시도
        if self.imported_modules.get('AuthPage'):
            try:
                page = self.imported_modules['AuthPage']()
                page.render()
            except Exception as e:
                logger.error(f"AuthPage render error: {e}")
                self.render_fallback_auth_page()
        else:
            self.render_fallback_auth_page()
            
    def render_fallback_auth_page(self):
        """폴백 인증 페이지"""
        st.title("🔐 로그인")
        
        tab1, tab2 = st.tabs(["로그인", "회원가입"])
        
        with tab1:
            with st.form("login_form"):
                email = st.text_input("이메일", placeholder="your@email.com")
                password = st.text_input("비밀번호", type="password")
                remember = st.checkbox("로그인 상태 유지")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.form_submit_button("로그인", type="primary", use_container_width=True):
                        # 임시 로그인 처리
                        if email and password:
                            st.session_state.authenticated = True
                            st.session_state.user = {
                                'email': email,
                                'name': email.split('@')[0],
                                'level': 'beginner',
                                'experiment_count': 0
                            }
                            st.session_state.current_page = 'dashboard'
                            st.rerun()
                        else:
                            st.error("이메일과 비밀번호를 입력하세요.")
                            
                with col2:
                    if st.form_submit_button("게스트로 둘러보기", use_container_width=True):
                        st.session_state.guest_mode = True
                        st.session_state.current_page = 'dashboard'
                        st.rerun()
                        
        with tab2:
            st.info("회원가입 기능은 준비 중입니다.")
            
    def render_dashboard_page(self):
        """대시보드 페이지 렌더링"""
        if not st.session_state.authenticated and not st.session_state.guest_mode:
            st.warning("로그인이 필요합니다.")
            return
            
        # 동적 모듈 로드 시도
        if self.imported_modules.get('DashboardPage'):
            try:
                page = self.imported_modules['DashboardPage']()
                page.render()
            except Exception as e:
                logger.error(f"DashboardPage render error: {e}")
                self.render_fallback_dashboard()
        else:
            self.render_fallback_dashboard()
            
    def render_fallback_dashboard(self):
        """폴백 대시보드"""
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
                if st.session_state.guest_mode:
                    st.warning("로그인이 필요합니다.")
                else:
                    st.session_state.current_page = 'project_setup'
                    st.rerun()
                    
        with col2:
            if st.button("🔍 문헌 검색", use_container_width=True):
                st.session_state.current_page = 'literature_search'
                st.rerun()
                
        with col3:
            if st.button("🛍️ 모듈 탐색", use_container_width=True):
                if st.session_state.guest_mode:
                    st.warning("로그인이 필요합니다.")
                else:
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
                    if st.session_state.guest_mode:
                        st.warning("로그인이 필요합니다.")
                    else:
                        st.session_state.selected_field = field_key
                        st.session_state.current_page = 'project_setup'
                        st.rerun()
                        
    def render_project_setup_page(self):
        """프로젝트 설정 페이지"""
        if not st.session_state.authenticated:
            st.warning("로그인이 필요합니다.")
            return
            
        # 동적 모듈 로드
        if self.imported_modules.get('ProjectSetupPage'):
            try:
                page = self.imported_modules['ProjectSetupPage']()
                page.render()
            except Exception as e:
                logger.error(f"ProjectSetupPage render error: {e}")
                st.error("프로젝트 설정 모듈 로딩 중 오류가 발생했습니다.")
        else:
            st.title("📝 프로젝트 설정")
            st.info("프로젝트 설정 모듈을 준비 중입니다.")
            
    def render_experiment_design_page(self):
        """실험 설계 페이지"""
        if not st.session_state.authenticated:
            st.warning("로그인이 필요합니다.")
            return
            
        # 동적 모듈 로드
        if self.imported_modules.get('ExperimentDesignPage'):
            try:
                page = self.imported_modules['ExperimentDesignPage']()
                page.render()
            except Exception as e:
                logger.error(f"ExperimentDesignPage render error: {e}")
                st.error("실험 설계 모듈 로딩 중 오류가 발생했습니다.")
        else:
            st.title("🧪 실험 설계")
            st.info("실험 설계 모듈을 준비 중입니다.")
            
    def render_data_analysis_page(self):
        """데이터 분석 페이지"""
        if not st.session_state.authenticated:
            st.warning("로그인이 필요합니다.")
            return
            
        # 동적 모듈 로드
        if self.imported_modules.get('DataAnalysisPage'):
            try:
                page = self.imported_modules['DataAnalysisPage']()
                page.render()
            except Exception as e:
                logger.error(f"DataAnalysisPage render error: {e}")
                st.error("데이터 분석 모듈 로딩 중 오류가 발생했습니다.")
        else:
            st.title("📈 데이터 분석")
            st.info("데이터 분석 모듈을 준비 중입니다.")
            
    def render_literature_search_page(self):
        """문헌 검색 페이지"""
        # 동적 모듈 로드
        if self.imported_modules.get('LiteratureSearchPage'):
            try:
                page = self.imported_modules['LiteratureSearchPage']()
                page.render()
            except Exception as e:
                logger.error(f"LiteratureSearchPage render error: {e}")
                st.error("문헌 검색 모듈 로딩 중 오류가 발생했습니다.")
        else:
            st.title("🔍 문헌 검색")
            st.info("AI 기반 문헌 검색 기능을 준비 중입니다.")
            
    def render_collaboration_page(self):
        """협업 페이지"""
        if not st.session_state.authenticated:
            st.warning("로그인이 필요합니다.")
            return
            
        # 동적 모듈 로드
        if self.imported_modules.get('CollaborationPage'):
            try:
                page = self.imported_modules['CollaborationPage']()
                page.render()
            except Exception as e:
                logger.error(f"CollaborationPage render error: {e}")
                st.error("협업 모듈 로딩 중 오류가 발생했습니다.")
        else:
            st.title("👥 협업")
            st.info("팀 협업 기능을 준비 중입니다.")
            
    def render_visualization_page(self):
        """시각화 페이지"""
        # 동적 모듈 로드
        if self.imported_modules.get('VisualizationPage'):
            try:
                page = self.imported_modules['VisualizationPage']()
                page.render()
            except Exception as e:
                logger.error(f"VisualizationPage render error: {e}")
                st.error("시각화 모듈 로딩 중 오류가 발생했습니다.")
        else:
            st.title("📊 시각화")
            st.info("데이터 시각화 도구를 준비 중입니다.")
            
    def render_module_marketplace_page(self):
        """모듈 마켓플레이스 페이지"""
        if not st.session_state.authenticated:
            st.warning("로그인이 필요합니다.")
            return
            
        # 동적 모듈 로드
        if self.imported_modules.get('ModuleMarketplacePage'):
            try:
                page = self.imported_modules['ModuleMarketplacePage']()
                page.render()
            except Exception as e:
                logger.error(f"ModuleMarketplacePage render error: {e}")
                st.error("마켓플레이스 모듈 로딩 중 오류가 발생했습니다.")
        else:
            st.title("🛍️ 모듈 마켓플레이스")
            
            # 모듈 레지스트리 사용
            if self.module_registry:
                st.markdown("### 📦 사용 가능한 모듈")
                
                modules = self.module_registry.list_modules()
                
                if modules:
                    for module in modules:
                        with st.expander(f"{module['display_name']} v{module['version']}"):
                            st.write(f"**작성자**: {module['author']}")
                            st.write(f"**카테고리**: {module['category']}")
                            st.write(f"**설명**: {module['description']}")
                            
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
        with st.expander("🔑 API 키 설정", expanded=True):
            st.info("API 키를 설정하여 AI 기능을 활성화하세요.")
            
            api_services = {
                'google_gemini': 'Google Gemini 2.0 Flash (필수)',
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
                    st.session_state.user['name'] = name
                    st.session_state.user['organization'] = organization
                    st.session_state.user['research_field'] = research_field
                    st.success("프로필이 업데이트되었습니다!")
                    
        # UI 설정
        with st.expander("🎨 UI 설정"):
            theme = st.radio("테마", ["light", "dark"], index=0 if st.session_state.theme == 'light' else 1)
            if theme != st.session_state.theme:
                st.session_state.theme = theme
                st.info("테마가 변경되었습니다. 새로고침하면 적용됩니다.")
                
            language = st.selectbox("언어", ["한국어", "English"], index=0 if st.session_state.language == 'ko' else 1)
            if language != st.session_state.language:
                st.session_state.language = 'ko' if language == "한국어" else 'en'
                st.info("언어가 변경되었습니다.")
                
    def render_notifications(self):
        """알림 표시"""
        with st.container():
            st.markdown("### 🔔 알림")
            
            if st.session_state.notifications:
                for notif in st.session_state.notifications[-5:]:  # 최근 5개만
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.info(notif.get('message', ''))
                    with col2:
                        if st.button("✓", key=f"notif_{notif.get('id')}"):
                            st.session_state.notifications.remove(notif)
                            st.session_state.unread_notifications = max(0, st.session_state.unread_notifications - 1)
                            st.rerun()
            else:
                st.info("새로운 알림이 없습니다.")
                
            st.divider()
            
    def render_footer(self):
        """푸터 렌더링"""
        st.divider()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Universal DOE Platform**")
            st.caption(f"Version {APP_VERSION}")
            
        with col2:
            st.markdown("**지원**")
            st.caption("📧 support@polymer-doe.com")
            st.caption("📚 [문서](https://docs.polymer-doe.com)")
            
        with col3:
            st.markdown("**커뮤니티**")
            st.caption("💬 [Discord](https://discord.gg/polymer-doe)")
            st.caption("🐦 [Twitter](https://twitter.com/polymer_doe)")
            
        st.markdown("""
        <div style='text-align: center; margin-top: 2rem; color: #6B7280;'>
            <p>Made with ❤️ by Polymer DOE Team © 2024</p>
        </div>
        """, unsafe_allow_html=True)
        
    def logout(self):
        """로그아웃 처리"""
        # 세션 초기화
        for key in ['authenticated', 'user', 'user_id', 'current_project', 'projects']:
            if key in st.session_state:
                st.session_state[key] = None
                
        st.session_state.authenticated = False
        st.session_state.current_page = 'auth'
        st.rerun()
        
    def run_background_tasks(self):
        """백그라운드 작업 실행"""
        # 주기적으로 실행되어야 할 작업들
        # 예: 알림 확인, 세션 갱신, 자동 저장 등
        pass


def main():
    """메인 실행 함수"""
    try:
        app = PolymerDOEApp()
        app.run()
    except Exception as e:
        logger.critical(f"Critical error in main: {e}")
        st.error("치명적인 오류가 발생했습니다. 관리자에게 문의하세요.")
        st.stop()


if __name__ == "__main__":
    main()
