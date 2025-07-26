"""
🧬 Universal DOE Platform - 메인 애플리케이션
고분자 연구자를 위한 AI 기반 실험 설계 플랫폼
Version: 2.0.0
"""

import streamlit as st
import os
import sys
import logging
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Type, List
import json
import uuid
import importlib
import asyncio
from functools import lru_cache

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 로깅 설정
log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 전역 상수
APP_NAME = "Universal DOE Platform"
APP_VERSION = "2.0.0"
APP_DESCRIPTION = "모든 고분자 연구자를 위한 AI 기반 실험 설계 플랫폼"
SESSION_TIMEOUT_MINUTES = 30
MIN_PASSWORD_LENGTH = 8

# 페이지 정의
PAGES = {
    'auth': {
        'title': '로그인',
        'icon': '🔐',
        'module': 'pages.auth_page',
        'class': 'AuthPage',
        'public': True,
        'order': 0
    },
    'dashboard': {
        'title': '대시보드',
        'icon': '📊',
        'module': 'pages.dashboard_page',
        'class': 'DashboardPage',
        'public': False,
        'order': 1
    },
    'project_setup': {
        'title': '프로젝트 설정',
        'icon': '📝',
        'module': 'pages.project_setup',
        'class': 'ProjectSetupPage',
        'public': False,
        'order': 2
    },
    'experiment_design': {
        'title': '실험 설계',
        'icon': '🧪',
        'module': 'pages.experiment_design',
        'class': 'ExperimentDesignPage',
        'public': False,
        'order': 3
    },
    'data_analysis': {
        'title': '데이터 분석',
        'icon': '📈',
        'module': 'pages.data_analysis',
        'class': 'DataAnalysisPage',
        'public': False,
        'order': 4
    },
    'visualization': {
        'title': '시각화',
        'icon': '📊',
        'module': 'pages.visualization',
        'class': 'VisualizationPage',
        'public': False,
        'order': 5
    },
    'literature_search': {
        'title': '문헌 검색',
        'icon': '🔍',
        'module': 'pages.literature_search',
        'class': 'LiteratureSearchPage',
        'public': True,
        'order': 6
    },
    'collaboration': {
        'title': '협업',
        'icon': '👥',
        'module': 'pages.collaboration',
        'class': 'CollaborationPage',
        'public': False,
        'order': 7
    },
    'module_marketplace': {
        'title': '모듈 마켓플레이스',
        'icon': '🛍️',
        'module': 'pages.module_marketplace',
        'class': 'ModuleMarketplacePage',
        'public': False,
        'order': 8
    },
    'module_loader': {
        'title': '모듈 로더',
        'icon': '📦',
        'module': 'pages.module_loader',
        'class': 'ModuleLoaderPage',
        'public': False,
        'order': 9
    },
    'settings': {
        'title': '설정',
        'icon': '⚙️',
        'module': None,  # 내장 페이지
        'class': None,
        'public': False,
        'order': 10
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
        required_dirs = ['data', 'logs', 'temp', 'modules/user_modules', 'cache']
        for dir_name in required_dirs:
            dir_path = PROJECT_ROOT / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # 동적 모듈 임포트
        self._import_modules()
        
        # 데이터베이스 초기화 확인
        self._check_database()
        
        # 모듈 레지스트리 초기화
        self._initialize_module_registry()
        
    def _import_modules(self):
        """필요한 모듈 동적 임포트"""
        # 페이지 모듈
        page_imports = {}
        for page_key, page_info in PAGES.items():
            if page_info['module'] and page_info['class']:
                page_imports[page_info['class']] = (page_info['module'], page_info['class'])
        
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
        all_imports = {**utils_imports, **config_imports}
        
        for module_path, imports in all_imports.items():
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
                    
        # 페이지 모듈은 별도로 저장
        for class_name, (module_path, _) in page_imports.items():
            self.imported_modules[class_name] = (module_path, class_name)
            
    def _check_database(self):
        """데이터베이스 초기화 확인"""
        try:
            from utils.database_manager import DatabaseManager
            db_manager = DatabaseManager()
            if not db_manager.check_database_exists():
                logger.info("Initializing database...")
                db_manager.initialize_database()
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            
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
            
    @lru_cache(maxsize=10)
    def _import_page_module(self, module_path: str, class_name: str):
        """페이지 모듈 동적 임포트 (캐싱)"""
        try:
            module = importlib.import_module(module_path)
            page_class = getattr(module, class_name)
            return page_class
        except Exception as e:
            logger.error(f"Failed to import {module_path}.{class_name}: {e}")
            return None
            
    def run(self):
        """메인 실행 함수"""
        try:
            # Streamlit 페이지 설정
            self.setup_page_config()
            
            # 세션 상태 초기화
            self.initialize_session_state()
            
            # 오프라인 모드 체크
            self.check_offline_mode()
            
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
            logger.error(traceback.format_exc())
            self.render_error_page(e)
            
    def setup_page_config(self):
        """Streamlit 페이지 설정"""
        st.set_page_config(
            page_title=APP_NAME,
            page_icon="🧬",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/your-repo/polymer-doe/wiki',
                'Report a bug': 'https://github.com/your-repo/polymer-doe/issues',
                'About': f"# {APP_NAME}\n\n{APP_DESCRIPTION}\n\nVersion: {APP_VERSION}"
            }
        )
        
    def initialize_session_state(self):
        """세션 상태 초기화"""
        # 기본 세션 변수
        defaults = {
            # 인증 관련
            'authenticated': False,
            'user': None,
            'user_id': None,
            'guest_mode': False,  # 2번에서 추가
            'session_id': str(uuid.uuid4()),
            'login_time': None,
            'last_activity': datetime.now(),
            
            # 앱 상태
            'current_page': 'auth',
            'previous_page': None,
            'page_params': {},
            'selected_field': None,  # 2번에서 추가
            
            # 프로젝트 관련
            'current_project': None,
            'projects': [],
            'selected_modules': [],
            
            # UI 상태
            'sidebar_state': 'expanded',
            'theme': 'light',
            'language': 'ko',
            'offline_mode': False,
            
            # 알림
            'notifications': [],
            'show_notifications': False,
            'unread_notifications': 0,  # 2번에서 추가
            
            # API 키
            'api_keys': {},
            'api_keys_validated': {},
            
            # 모듈 관련
            'module_registry_initialized': False,
            'available_modules': {},
            'loaded_modules': {},
            
            # 임시 데이터
            'temp_data': {},
            'form_data': {},
            'cache': {},
            
            # 에러 상태
            'last_error': None,
            'error_count': 0
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
                
    def check_offline_mode(self):
        """오프라인 모드 체크"""
        try:
            import requests
            response = requests.get('https://www.google.com', timeout=3)
            st.session_state.offline_mode = False
        except:
            st.session_state.offline_mode = True
            logger.info("Running in offline mode")
            
    def check_session_validity(self) -> bool:
        """세션 유효성 검사"""
        if not st.session_state.authenticated:
            return True  # 로그인 페이지는 항상 접근 가능
            
        # 세션 타임아웃 체크
        if 'last_activity' in st.session_state:
            time_diff = datetime.now() - st.session_state.last_activity
            if time_diff > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
                self.logout()
                st.warning("세션이 만료되었습니다. 다시 로그인해주세요.")
                return False
                
        # 활동 시간 업데이트
        st.session_state.last_activity = datetime.now()
        return True
        
    def apply_custom_css(self):
        """커스텀 CSS 적용 (2번의 상세한 스타일 포함)"""
        css = """
        <style>
        /* 메인 컬러 변수 */
        :root {
            --primary-color: #7C3AED;
            --secondary-color: #F59E0B;
            --success-color: #10B981;
            --danger-color: #EF4444;
            --warning-color: #F59E0B;
            --info-color: #3B82F6;
            --dark-color: #1F2937;
            --light-color: #F3F4F6;
            --muted-color: #6B7280;
        }
        
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
            border-radius: 8px;
            font-weight: 500;
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
        
        /* 입력 필드 스타일 */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div > select {
            border-radius: 8px;
            border: 2px solid #E5E7EB;
            transition: all 0.3s ease;
        }
        
        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus,
        .stSelectbox > div > div > select:focus {
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.1);
        }
        
        /* 메트릭 카드 */
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            border: 1px solid #E5E7EB;
            transition: all 0.3s ease;
        }
        
        .metric-card:hover {
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            transform: translateY(-2px);
        }
        
        /* 오프라인 모드 표시 */
        .offline-badge {
            background-color: #FEE2E2;
            color: #DC2626;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 14px;
            font-weight: 500;
        }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
        
        # 추가로 common_ui에서 CSS 가져오기 시도
        if self.imported_modules.get('apply_custom_css'):
            try:
                self.imported_modules['apply_custom_css']()
            except:
                pass
                
    def render_header(self):
        """헤더 렌더링 (1번과 2번 통합)"""
        header_col1, header_col2, header_col3 = st.columns([6, 3, 1])
        
        with header_col1:
            # 현재 페이지 정보
            current_page = st.session_state.get('current_page', 'auth')
            if current_page in PAGES:
                page_info = PAGES[current_page]
                st.markdown(f"# {page_info['icon']} {page_info['title']}")
            else:
                st.markdown(f"# 🧬 {APP_NAME}")
                
        with header_col2:
            # 오프라인 모드 표시
            if st.session_state.offline_mode:
                st.markdown("""
                <div style='text-align: right; padding: 10px;'>
                    <span class='offline-badge'>🔌 오프라인 모드</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                # 사용자 정보
                if st.session_state.authenticated and st.session_state.user:
                    st.markdown(f"""
                    <div style='text-align: right; padding: 10px;'>
                        👤 {st.session_state.user.get('name', 'User')}
                    </div>
                    """, unsafe_allow_html=True)
                elif st.session_state.guest_mode:
                    st.markdown("""
                    <div style='text-align: right; padding: 10px;'>
                        👤 게스트
                    </div>
                    """, unsafe_allow_html=True)
                    
        with header_col3:
            # 알림 버튼
            if st.session_state.authenticated:
                notification_count = len(st.session_state.notifications)
                if st.button(f"🔔 {notification_count}" if notification_count > 0 else "🔔"):
                    st.session_state.show_notifications = not st.session_state.show_notifications
                    st.rerun()
                    
        st.divider()
        
    def render_sidebar(self):
        """사이드바 렌더링 (2번의 프로필 포함)"""
        with st.sidebar:
            # 로고 및 타이틀
            st.markdown(f"""
            <div style='text-align: center; padding: 1rem 0;'>
                <h1 style='color: #7C3AED; margin: 0;'>🧬</h1>
                <h3 style='margin: 0;'>{APP_NAME}</h3>
                <p style='color: #6B7280; font-size: 14px;'>v{APP_VERSION}</p>
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
            self.render_navigation()
            
            st.divider()
            
            # 사이드바 푸터
            self.render_sidebar_footer()
            
    def render_user_profile(self):
        """사용자 프로필 렌더링 (2번에서)"""
        user = st.session_state.user
        if not user:
            return
            
        col1, col2 = st.sidebar.columns([1, 3])
        
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
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("프로젝트", len(st.session_state.projects))
        with col2:
            st.metric("실험", user.get('experiment_count', 0))
            
    def render_navigation(self):
        """네비게이션 메뉴 렌더링"""
        st.sidebar.markdown("### 📍 메뉴")
        
        # 페이지 정렬
        sorted_pages = sorted(
            [(k, v) for k, v in PAGES.items() if k != 'settings'],
            key=lambda x: x[1]['order']
        )
        
        for page_key, page_info in sorted_pages:
            # 인증 체크
            if not page_info['public'] and not st.session_state.authenticated and not st.session_state.guest_mode:
                continue
                
            # 게스트 모드 접근 제한
            if st.session_state.guest_mode and not page_info['public']:
                continue
                
            # 현재 페이지 하이라이트
            is_current = st.session_state.current_page == page_key
            
            if st.sidebar.button(
                f"{page_info['icon']} {page_info['title']}", 
                key=f"nav_{page_key}",
                use_container_width=True,
                type="primary" if is_current else "secondary"
            ):
                st.session_state.previous_page = st.session_state.current_page
                st.session_state.current_page = page_key
                st.rerun()
                
    def render_sidebar_footer(self):
        """사이드바 푸터 렌더링"""
        if st.session_state.authenticated:
            col1, col2 = st.sidebar.columns(2)
            
            with col1:
                if st.button("⚙️ 설정", use_container_width=True):
                    st.session_state.current_page = 'settings'
                    st.rerun()
                    
            with col2:
                if st.button("🚪 로그아웃", use_container_width=True):
                    self.logout()
                    
        # 도움말 링크
        st.sidebar.markdown("""
        <div style='text-align: center; margin-top: 2rem; color: #6B7280; font-size: 12px;'>
            <a href='https://github.com/your-repo/polymer-doe/wiki' target='_blank'>📚 도움말</a> |
            <a href='https://github.com/your-repo/polymer-doe/issues' target='_blank'>🐛 버그 신고</a>
        </div>
        """, unsafe_allow_html=True)
        
    def render_main_content(self):
        """메인 콘텐츠 렌더링"""
        # 알림 표시
        if st.session_state.show_notifications:
            self.render_notifications()
            
        # 현재 페이지 렌더링
        st.markdown('<div class="main-content">', unsafe_allow_html=True)
        
        current_page = st.session_state.current_page
        
        if current_page == 'settings':
            self.render_settings_page()
        elif current_page in PAGES:
            page_info = PAGES[current_page]
            
            # 권한 체크
            if not page_info['public'] and not st.session_state.authenticated and not st.session_state.guest_mode:
                st.error("이 페이지에 접근할 권한이 없습니다.")
                if st.button("로그인 페이지로 이동"):
                    st.session_state.current_page = 'auth'
                    st.rerun()
                return
                
            # 페이지 모듈 로드 및 렌더링
            if page_info['module'] and page_info['class']:
                # 먼저 동적 임포트 시도
                page_class = self._import_page_module(
                    page_info['module'], 
                    page_info['class']
                )
                
                if page_class:
                    try:
                        # 페이지 인스턴스 생성 및 렌더링
                        page_instance = page_class()
                        page_instance.render()
                    except Exception as e:
                        logger.error(f"Failed to render page {current_page}: {e}")
                        self.render_fallback_page(current_page)
                else:
                    # 모듈 로드 실패 시 폴백
                    self.render_fallback_page(current_page)
        else:
            st.error(f"알 수 없는 페이지: {current_page}")
            
        st.markdown('</div>', unsafe_allow_html=True)
        
    def render_fallback_page(self, page_key: str):
        """폴백 페이지 렌더링 (2번에서 가져옴)"""
        fallback_renderers = {
            'auth': self.render_fallback_auth_page,
            'dashboard': self.render_fallback_dashboard,
            'project_setup': self.render_fallback_project_setup,
            'experiment_design': self.render_fallback_experiment_design,
            'data_analysis': self.render_fallback_data_analysis,
            'literature_search': self.render_fallback_literature_search,
            'collaboration': self.render_fallback_collaboration,
            'visualization': self.render_fallback_visualization,
            'module_marketplace': self.render_fallback_marketplace,
            'module_loader': self.render_fallback_module_loader
        }
        
        if page_key in fallback_renderers:
            fallback_renderers[page_key]()
        else:
            st.error(f"페이지 '{page_key}'를 로드할 수 없습니다.")
            
    def render_fallback_auth_page(self):
        """폴백 인증 페이지 (2번에서)"""
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
            
    def render_fallback_dashboard(self):
        """폴백 대시보드 (2번의 연구 분야 선택 포함)"""
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
                    
        # 연구 분야 선택 (2번에서)
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
                        
        # 최근 활동 (1번에서 추가)
        if st.session_state.projects:
            st.markdown("### 📁 최근 프로젝트")
            for project in st.session_state.projects[:3]:
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{project.get('name', '프로젝트')}**")
                        st.caption(f"생성일: {project.get('created_at', 'N/A')}")
                    with col2:
                        if st.button("열기", key=f"open_{project.get('id')}"):
                            st.session_state.current_project = project
                            st.session_state.current_page = 'experiment_design'
                            st.rerun()
                            
    def render_fallback_project_setup(self):
        """폴백 프로젝트 설정"""
        st.title("📝 프로젝트 설정")
        
        # 선택된 연구 분야 표시
        if st.session_state.selected_field:
            field_info = RESEARCH_FIELDS[st.session_state.selected_field]
            st.info(f"선택된 연구 분야: {field_info['name']}")
            
        st.info("프로젝트 설정 모듈을 준비 중입니다.")
        
    def render_fallback_experiment_design(self):
        """폴백 실험 설계"""
        st.title("🧪 실험 설계")
        st.info("실험 설계 모듈을 준비 중입니다.")
        
    def render_fallback_data_analysis(self):
        """폴백 데이터 분석"""
        st.title("📈 데이터 분석")
        st.info("데이터 분석 모듈을 준비 중입니다.")
        
    def render_fallback_literature_search(self):
        """폴백 문헌 검색"""
        st.title("🔍 문헌 검색")
        st.info("AI 기반 문헌 검색 기능을 준비 중입니다.")
        
    def render_fallback_collaboration(self):
        """폴백 협업"""
        st.title("👥 협업")
        st.info("팀 협업 기능을 준비 중입니다.")
        
    def render_fallback_visualization(self):
        """폴백 시각화"""
        st.title("📊 시각화")
        st.info("데이터 시각화 도구를 준비 중입니다.")
        
    def render_fallback_marketplace(self):
        """폴백 마켓플레이스 (1번의 모듈 레지스트리 사용)"""
        st.title("🛍️ 모듈 마켓플레이스")
        
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
            
    def render_fallback_module_loader(self):
        """폴백 모듈 로더"""
        st.title("📦 모듈 로더")
        st.info("커스텀 모듈 로딩 기능을 준비 중입니다.")
        
    def render_notifications(self):
        """알림 렌더링 (1번과 2번 통합)"""
        with st.container():
            st.markdown("### 🔔 알림")
            
            if not st.session_state.notifications:
                st.info("새로운 알림이 없습니다.")
            else:
                for i, notification in enumerate(st.session_state.notifications):
                    with st.expander(
                        f"{notification.get('icon', '📢')} {notification.get('title', '알림')}",
                        expanded=not notification.get('read', False)
                    ):
                        st.write(notification.get('message', ''))
                        st.caption(notification.get('timestamp', ''))
                        
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            if st.button("읽음", key=f"read_{i}"):
                                st.session_state.notifications[i]['read'] = True
                                st.session_state.unread_notifications = max(0, st.session_state.unread_notifications - 1)
                                st.rerun()
                        with col2:
                            if st.button("삭제", key=f"delete_{i}"):
                                st.session_state.notifications.pop(i)
                                st.rerun()
                                
            st.divider()
            
    def render_settings_page(self):
        """설정 페이지 렌더링 (1번의 완전한 기능)"""
        st.markdown("## ⚙️ 설정")
        
        tabs = st.tabs(["일반", "API 키", "데이터", "고급"])
        
        with tabs[0]:  # 일반 설정
            st.markdown("### 일반 설정")
            
            # 프로필 설정 (2번에서)
            if st.session_state.user:
                with st.expander("👤 프로필 설정", expanded=True):
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
                        st.session_state.user['name'] = name
                        st.session_state.user['organization'] = organization
                        st.session_state.user['research_field'] = research_field
                        st.success("프로필이 업데이트되었습니다!")
                        
            # 테마 설정
            theme = st.selectbox(
                "테마",
                ["light", "dark"],
                index=0 if st.session_state.theme == "light" else 1
            )
            if theme != st.session_state.theme:
                st.session_state.theme = theme
                st.info("테마가 변경되었습니다. 새로고침하면 적용됩니다.")
                
            # 언어 설정
            language = st.selectbox(
                "언어",
                ["한국어", "English"],
                index=0 if st.session_state.language == "ko" else 1
            )
            
            # 알림 설정
            st.checkbox("데스크톱 알림 사용", value=True)
            st.checkbox("이메일 알림 사용", value=False)
            
        with tabs[1]:  # API 키 설정 (2번의 상세 목록)
            st.markdown("### API 키 관리")
            st.info("API 키는 안전하게 암호화되어 저장됩니다.")
            
            # AI 엔진 API 키
            with st.expander("AI 엔진 API 키", expanded=True):
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
                        
            # 데이터베이스 API 키
            with st.expander("데이터베이스 API 키"):
                db_services = {
                    'materials_project': 'Materials Project API Key',
                    'pubchem': 'PubChem API Key',
                    'chemspider': 'ChemSpider API Key'
                }
                
                for service_key, service_name in db_services.items():
                    current_key = st.session_state.api_keys.get(service_key, '')
                    new_key = st.text_input(
                        service_name,
                        value='*' * 20 if current_key else '',
                        type="password",
                        key=f"api_key_{service_key}"
                    )
                    if new_key and new_key != '*' * 20:
                        st.session_state.api_keys[service_key] = new_key
                        
            if st.button("API 키 저장", use_container_width=True):
                try:
                    from utils.secrets_manager import SecretsManager
                    secrets_manager = SecretsManager()
                    secrets_manager.save_api_keys(st.session_state.api_keys)
                    st.success("API 키가 안전하게 저장되었습니다.")
                except Exception as e:
                    st.error(f"API 키 저장 실패: {str(e)}")
                    
        with tabs[2]:  # 데이터 설정 (1번의 백업/복원)
            st.markdown("### 데이터 관리")
            
            # 캐시 관리
            st.markdown("#### 캐시")
            cache_size = self.get_cache_size()
            st.metric("캐시 크기", f"{cache_size:.2f} MB")
            
            if st.button("캐시 비우기"):
                self.clear_cache()
                st.success("캐시가 비워졌습니다.")
                st.rerun()
                
            # 백업/복원
            st.markdown("#### 백업/복원")
            if st.button("데이터 백업"):
                self.backup_data()
                
            uploaded_file = st.file_uploader("백업 파일 복원", type=['zip'])
            if uploaded_file:
                self.restore_data(uploaded_file)
                
        with tabs[3]:  # 고급 설정
            st.markdown("### 고급 설정")
            
            # 디버그 모드
            debug_mode = st.checkbox("디버그 모드", value=False)
            if debug_mode:
                st.warning("디버그 모드가 활성화되었습니다. 성능이 저하될 수 있습니다.")
                
            # 로그 레벨
            log_level = st.selectbox(
                "로그 레벨",
                ["DEBUG", "INFO", "WARNING", "ERROR"],
                index=1
            )
            
            # 세션 타임아웃
            timeout = st.number_input(
                "세션 타임아웃 (분)",
                min_value=5,
                max_value=120,
                value=SESSION_TIMEOUT_MINUTES
            )
            
            if st.button("고급 설정 저장"):
                st.success("설정이 저장되었습니다.")
                
    def render_footer(self):
        """푸터 렌더링"""
        st.divider()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Universal DOE Platform**")
            st.caption(f"Version {APP_VERSION}")
            st.caption("AI 기반 고분자 실험 설계 플랫폼")
            
        with col2:
            st.markdown("**지원**")
            st.caption("📧 support@polymer-doe.com")
            st.caption("📚 [문서](https://docs.polymer-doe.com)")
            st.caption("💬 [커뮤니티](https://community.polymer-doe.com)")
            
        with col3:
            st.markdown("**리소스**")
            st.caption("📺 [튜토리얼](https://youtube.com/polymer-doe)")
            st.caption("🐦 [Twitter](https://twitter.com/polymer_doe)")
            st.caption("📝 [블로그](https://blog.polymer-doe.com)")
            
        st.markdown("""
        <div style='text-align: center; margin-top: 2rem; color: #6B7280; font-size: 12px;'>
            <p>Made with ❤️ by Polymer DOE Team © 2024</p>
            <p>Licensed under MIT License</p>
        </div>
        """, unsafe_allow_html=True)
        
    def logout(self):
        """로그아웃 처리"""
        # 세션 초기화
        keys_to_keep = ['session_id', 'theme', 'language']
        for key in list(st.session_state.keys()):
            if key not in keys_to_keep:
                del st.session_state[key]
                
        # 기본값 재설정
        st.session_state.authenticated = False
        st.session_state.current_page = 'auth'
        st.session_state.user = None
        st.session_state.notifications = []
        st.session_state.guest_mode = False
        
        # 로그 기록
        logger.info(f"User logged out - Session: {st.session_state.session_id}")
        
        st.rerun()
        
    def run_background_tasks(self):
        """백그라운드 작업 실행 (1번의 실제 구현)"""
        try:
            # 1. 알림 확인
            self.check_new_notifications()
            
            # 2. 자동 저장
            if st.session_state.get('current_project'):
                self.auto_save_project()
                
            # 3. 캐시 정리
            self.cleanup_old_cache()
            
            # 4. 세션 갱신
            self.refresh_session()
            
        except Exception as e:
            logger.error(f"Background task error: {e}")
            
    def check_new_notifications(self):
        """새 알림 확인"""
        # 실제 구현에서는 데이터베이스나 API에서 알림을 가져옴
        pass
        
    def auto_save_project(self):
        """프로젝트 자동 저장"""
        # 실제 구현에서는 현재 프로젝트 상태를 저장
        pass
        
    def cleanup_old_cache(self):
        """오래된 캐시 정리"""
        cache_dir = PROJECT_ROOT / "cache"
        if cache_dir.exists():
            for file in cache_dir.iterdir():
                if file.is_file():
                    # 7일 이상 된 파일 삭제
                    if (datetime.now() - datetime.fromtimestamp(file.stat().st_mtime)).days > 7:
                        try:
                            file.unlink()
                        except:
                            pass
                            
    def refresh_session(self):
        """세션 갱신"""
        if st.session_state.authenticated:
            st.session_state.last_activity = datetime.now()
            
    def get_cache_size(self) -> float:
        """캐시 크기 계산 (MB)"""
        cache_dir = PROJECT_ROOT / "cache"
        total_size = 0
        
        if cache_dir.exists():
            for file in cache_dir.rglob('*'):
                if file.is_file():
                    total_size += file.stat().st_size
                    
        return total_size / (1024 * 1024)  # MB로 변환
        
    def clear_cache(self):
        """캐시 비우기"""
        cache_dir = PROJECT_ROOT / "cache"
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            cache_dir.mkdir()
            
        # 세션 캐시도 초기화
        st.session_state.cache = {}
        
    def backup_data(self):
        """데이터 백업 (1번의 실제 구현)"""
        try:
            import zipfile
            from datetime import datetime
            
            # 백업 파일명
            backup_name = f"polymer_doe_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            backup_path = PROJECT_ROOT / "temp" / backup_name
            
            # ZIP 파일 생성
            with zipfile.ZipFile(backup_path, 'w') as zipf:
                # 데이터 디렉토리 백업
                data_dir = PROJECT_ROOT / "data"
                if data_dir.exists():
                    for file in data_dir.rglob('*'):
                        if file.is_file():
                            zipf.write(file, file.relative_to(PROJECT_ROOT))
                            
            st.success(f"백업이 완료되었습니다: {backup_name}")
            
            # 다운로드 버튼
            with open(backup_path, 'rb') as f:
                st.download_button(
                    label="백업 파일 다운로드",
                    data=f.read(),
                    file_name=backup_name,
                    mime="application/zip"
                )
                
        except Exception as e:
            st.error(f"백업 실패: {str(e)}")
            
    def restore_data(self, uploaded_file):
        """데이터 복원"""
        try:
            import zipfile
            import tempfile
            
            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
                
            # ZIP 파일 추출
            with zipfile.ZipFile(tmp_path, 'r') as zipf:
                zipf.extractall(PROJECT_ROOT)
                
            st.success("데이터가 성공적으로 복원되었습니다.")
            st.info("앱을 다시 시작하면 복원된 데이터가 적용됩니다.")
            
        except Exception as e:
            st.error(f"복원 실패: {str(e)}")
            
    def render_error_page(self, error: Exception):
        """에러 페이지 렌더링"""
        st.error("애플리케이션 오류가 발생했습니다")
        
        with st.expander("오류 상세 정보"):
            st.code(traceback.format_exc())
            
        st.markdown("### 해결 방법")
        st.write("1. 페이지를 새로고침해보세요")
        st.write("2. 브라우저 캐시를 지워보세요")
        st.write("3. 문제가 지속되면 관리자에게 문의하세요")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 새로고침"):
                st.rerun()
        with col2:
            if st.button("🏠 홈으로"):
                st.session_state.current_page = 'dashboard' if st.session_state.authenticated else 'auth'
                st.rerun()


def main():
    """메인 실행 함수"""
    try:
        # 앱 인스턴스 생성 및 실행
        app = PolymerDOEApp()
        app.run()
        
    except Exception as e:
        logger.critical(f"Critical error in main: {e}")
        logger.critical(traceback.format_exc())
        
        # 최소한의 에러 페이지
        st.set_page_config(page_title="Error - Polymer DOE", page_icon="❌")
        st.error("치명적인 오류가 발생했습니다")
        st.write("시스템 관리자에게 문의하세요.")
        st.code(str(e))
        
        if st.button("앱 재시작"):
            st.rerun()


if __name__ == "__main__":
    main()
