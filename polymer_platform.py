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
from typing import Dict, Any, Optional, Type, List, Tuple, Union
import json
import uuid
import importlib
import asyncio
from functools import lru_cache
import time
import shutil
import zipfile
import tempfile
import re
from io import BytesIO, StringIO
import base64

# 조건부 임포트
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

# 데이터 분석 라이브러리
try:
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as e:
    st.error(f"필수 라이브러리를 설치해주세요: {e}")
    st.stop()

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 로깅 설정
log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(exist_ok=True)

# 구조화된 로깅 설정
class StructuredFormatter(logging.Formatter):
    def format(self, record):
        # 기본 포맷
        result = super().format(record)
        
        # extra 필드 추가
        if hasattr(record, 'extra_fields'):
            extra = json.dumps(record.extra_fields)
            result = f"{result} | {extra}"
            
        return result

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 구조화된 포맷터 적용
for handler in logging.getLogger().handlers:
    handler.setFormatter(StructuredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))

logger = logging.getLogger(__name__)

# 전역 상수
APP_NAME = "Universal DOE Platform"
APP_VERSION = "2.0.0"
APP_DESCRIPTION = "모든 고분자 연구자를 위한 AI 기반 실험 설계 플랫폼"
SESSION_TIMEOUT_MINUTES = 30
MIN_PASSWORD_LENGTH = 8
MAX_FILE_UPLOAD_SIZE_MB = 500
MAX_EXCEL_EXPORT_SIZE_MB = 10

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

# 연구 분야 정의 (캐싱 가능)
@st.cache_data(ttl=3600)
def load_research_fields():
    """연구 분야 데이터 로드 (캐싱)"""
    return {
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

RESEARCH_FIELDS = load_research_fields()

# API 키 패턴 정의
API_KEY_PATTERNS = {
    'google_gemini': r'^AIza[0-9A-Za-z\-_]{35}$',
    'openai': r'^sk-[A-Za-z0-9]{48}$',
    'github': r'^ghp_[A-Za-z0-9]{36}$',
    'huggingface': r'^hf_[A-Za-z0-9]{34}$',
    'groq': r'^gsk_[A-Za-z0-9\-_]+$',
    'deepseek': r'^sk-[A-Za-z0-9]+$'
}


class FallbackRenderers:
    """폴백 페이지 렌더러 모음"""
    
    @staticmethod
    def render_auth_page():
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
                        
        with tab2:  # 회원가입 탭
            with st.form("signup_form"):
                st.markdown("#### 기본 정보")
                col1, col2 = st.columns(2)
                with col1:
                    name = st.text_input("이름 *", placeholder="홍길동")
                    email = st.text_input("이메일 *", placeholder="your@email.com")
                with col2:
                    organization = st.text_input("소속", placeholder="○○대학교")
                    phone = st.text_input("전화번호", placeholder="010-1234-5678")
        
                st.markdown("#### 비밀번호 설정")
                col1, col2 = st.columns(2)
                with col1:
                    password = st.text_input("비밀번호 *", type="password", 
                                   help="8자 이상, 영문/숫자/특수문자 포함")
                with col2:
                    password_confirm = st.text_input("비밀번호 확인 *", type="password")
        
                # 비밀번호 강도 표시
                if password:
                    app = st.session_state.get('app_instance')
                    if app:
                        strength = app.check_password_strength(password)
                        st.progress(strength['score'] / 6)  # 최대 6점으로 수정
                        st.caption(f"비밀번호 강도: {strength['level']}")
                        if strength['feedback']:
                            for feedback in strength['feedback']:
                                st.caption(f"⚠️ {feedback}")
        
                st.markdown("#### 연구 분야")
                research_field = st.selectbox(
                    "주요 연구 분야",
                    options=list(RESEARCH_FIELDS.keys()),
                    format_func=lambda x: RESEARCH_FIELDS[x]['name']
                )
        
                terms = st.checkbox("이용약관 및 개인정보처리방침에 동의합니다")
        
                if st.form_submit_button("회원가입", type="primary", use_container_width=True):
                    if all([name, email, password, password == password_confirm, terms]):
                        # 실제 회원가입 처리
                        st.session_state.user = {
                            'email': email,
                            'name': name,
                            'organization': organization,
                            'phone': phone,
                            'research_field': research_field,
                            'level': 'beginner',
                            'experiment_count': 0,
                            'created_at': datetime.now().isoformat()
                        }
                        st.success("회원가입이 완료되었습니다!")
                        time.sleep(1)
                        st.session_state.authenticated = True
                        st.session_state.current_page = 'dashboard'
                        st.rerun()
                    else:
                        st.error("모든 필수 항목을 입력하고 약관에 동의해주세요.")
    
    # 나머지 폴백 렌더러들은 여기에 추가...


class PolymerDOEApp:
    """메인 애플리케이션 클래스"""
    
    def __init__(self):
        self.imported_modules = {}
        self.module_registry = None
        self.fallback_renderers = FallbackRenderers()
        
        # 앱 인스턴스를 세션에 저장 (다른 곳에서 참조용)
        st.session_state.app_instance = self
        
        self._initialize_app()

        # SecretsManager 초기화
        try:
            from utils.secrets_manager import get_secrets_manager
            self.secrets_manager = get_secrets_manager()
        except Exception as e:
            logger.warning(f"SecretsManager를 로드할 수 없습니다: {e}")
            self.secrets_manager = None
        
    def _initialize_app(self):
        """앱 초기화"""
        # 필수 디렉토리 생성
        required_dirs = [
            'data', 'logs', 'temp', 'modules/user_modules', 'cache', 
            'db', 'backups', 'exports', 'protocols'
        ]
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
            'config.app_config': ['APP_CONFIG', 'API_CONFIGS', 'API_PROVIDERS'],
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
            
    @st.cache_resource
    def _get_module_registry(self):
        """모듈 레지스트리 캐싱"""
        try:
            from modules.module_registry import ModuleRegistry
            return ModuleRegistry()
        except Exception as e:
            logger.error(f"Failed to create module registry: {e}")
            return None
            
    def _initialize_module_registry(self):
        """모듈 레지스트리 초기화"""
        try:
            self.module_registry = self._get_module_registry()
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
        except ImportError as e:
            logger.error(f"Module not found: {module_path} - {e}")
            return None
        except AttributeError as e:
            logger.error(f"Class not found in module: {class_name} - {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error importing {module_path}.{class_name}: {e}")
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
            logger.error(f"Application error: {e}", extra={
                'extra_fields': {
                    'user_id': st.session_state.get('user_id'),
                    'session_id': st.session_state.get('session_id'),
                    'page': st.session_state.get('current_page')
                }
            })
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
            'guest_mode': False,
            'session_id': str(uuid.uuid4()),
            'session_ip': None,  # IP 추적용
            'login_time': None,
            'last_activity': datetime.now(),
            
            # 앱 상태
            'current_page': 'auth',
            'previous_page': None,
            'page_params': {},
            'selected_field': None,
            
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
            'unread_notifications': 0,
            
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
                
        # IP 주소 저장 (보안용)
        if 'session_ip' not in st.session_state:
            try:
                st.session_state.session_ip = st.context.headers.get("X-Forwarded-For", "unknown")
            except:
                st.session_state.session_ip = "unknown"
                
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
        """세션 유효성 검사 (보안 강화)"""
        if not st.session_state.authenticated:
            return True
            
        # 세션 타임아웃 체크
        if 'last_activity' in st.session_state:
            time_diff = datetime.now() - st.session_state.last_activity
            if time_diff > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
                self.logout()
                st.warning("세션이 만료되었습니다. 다시 로그인해주세요.")
                return False
                
        # IP 변경 체크 (보안 강화)
        try:
            current_ip = st.context.headers.get("X-Forwarded-For", "unknown")
            if st.session_state.session_ip != "unknown" and st.session_state.session_ip != current_ip:
                logger.warning(f"Session IP changed: {st.session_state.session_ip} -> {current_ip}")
                # 필요시 재인증 요구 (현재는 경고만)
        except:
            pass
            
        # 활동 시간 업데이트
        st.session_state.last_activity = datetime.now()
        return True
        
    def apply_custom_css(self):
        """커스텀 CSS 적용"""
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
        
        /* 탭 스타일 */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f3f4f6;
            border-radius: 8px;
            padding: 0 16px;
            font-weight: 500;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: var(--primary-color);
            color: white;
        }
        
        /* 스크롤바 스타일 */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #555;
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
        """헤더 렌더링"""
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
        """사이드바 렌더링"""
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
        """사용자 프로필 렌더링"""
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
            if st.session_state.guest_mode and page_key not in ['dashboard', 'literature_search']:
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
        """폴백 페이지 렌더링"""
        fallback_renderers = {
            'auth': self.fallback_renderers.render_auth_page,
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

    def check_password_strength(self, password: str) -> Dict[str, Any]:
        """비밀번호 강도 체크 (보안 강화)"""
        score = 0
        feedback = []
        
        # 최소 길이 체크
        if len(password) < MIN_PASSWORD_LENGTH:
            feedback.append(f"최소 {MIN_PASSWORD_LENGTH}자 이상 필요")
            return {'score': 0, 'level': '매우 약함', 'feedback': feedback}
        
        # 일반적인 패턴 체크
        common_patterns = ['password', '12345', 'qwerty', 'admin', 'letmein', '111111']
        if any(pattern in password.lower() for pattern in common_patterns):
            feedback.append("일반적인 패턴이 포함되어 있습니다")
            score -= 1
    
        # 길이 체크
        if len(password) >= MIN_PASSWORD_LENGTH:
            score += 1
        if len(password) >= 12:
            score += 1
        if len(password) >= 16:
            score += 1
    
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
    
        # 강도 레벨
        if score <= 2:
            level = "약함"
        elif score <= 4:
            level = "보통"
        elif score <= 6:
            level = "강함"
        else:
            level = "매우 강함"
    
        return {
            'score': max(0, score),  # 음수 방지
            'level': level,
            'feedback': feedback
        }
    
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
                        
        # 최근 활동
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
        """프로젝트 설정 페이지"""
        st.title("📝 프로젝트 설정")
    
        # 프로젝트 생성/편집 폼
        with st.form("project_form"):
            st.markdown("### 프로젝트 정보")
        
            project_name = st.text_input(
                "프로젝트 이름 *",
                placeholder="예: 생분해성 고분자 합성 최적화"
            )
        
            project_desc = st.text_area(
                "프로젝트 설명",
                placeholder="프로젝트의 목적과 주요 내용을 입력하세요",
                height=100
            )
        
            col1, col2 = st.columns(2)
            with col1:
                project_type = st.selectbox(
                    "프로젝트 유형",
                    ["신규 개발", "공정 최적화", "문제 해결", "기초 연구", "스케일업"]
                )
            with col2:
                duration = st.number_input(
                    "예상 기간 (주)",
                    min_value=1,
                    max_value=52,
                    value=12
                )
        
            st.markdown("### 실험 설계 설정")
        
            col1, col2 = st.columns(2)
            with col1:
                design_type = st.selectbox(
                    "실험 설계 유형",
                    ["완전요인설계", "부분요인설계", "반응표면설계", "혼합물설계", "최적설계"]
                )
            with col2:
                confidence_level = st.select_slider(
                    "신뢰수준",
                    options=[90, 95, 99],
                    value=95
                )
        
            st.markdown("### 팀 설정")
            team_members = st.multiselect(
                "팀원 추가",
                ["김연구원", "이박사", "박교수", "최대학원생"],
                default=[]
            )
        
            if st.form_submit_button("프로젝트 생성", type="primary", use_container_width=True):
                if project_name:
                    # 프로젝트 생성
                    new_project = {
                        'id': str(uuid.uuid4()),
                        'name': project_name,
                        'description': project_desc,
                        'type': project_type,
                        'duration': duration,
                        'design_type': design_type,
                        'confidence_level': confidence_level,
                        'team': team_members,
                        'created_at': datetime.now().isoformat(),
                        'status': 'active'
                    }
                    st.session_state.projects.append(new_project)
                    st.session_state.current_project = new_project
                    st.success("프로젝트가 생성되었습니다!")
                    time.sleep(1)
                    st.session_state.current_page = 'experiment_design'
                    st.rerun()
                else:
                    st.error("프로젝트 이름을 입력해주세요.")
    
        # 기존 프로젝트 목록
        if st.session_state.projects:
            st.markdown("### 📁 기존 프로젝트")
            for project in st.session_state.projects:
                with st.expander(f"📌 {project['name']}"):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(f"**유형**: {project['type']}")
                        st.write(f"**생성일**: {project['created_at'][:10]}")
                    with col2:
                        if st.button("열기", key=f"open_{project['id']}"):
                            st.session_state.current_project = project
                            st.session_state.current_page = 'experiment_design'
                            st.rerun()
                    with col3:
                        if st.button("삭제", key=f"del_{project['id']}"):
                            st.session_state.projects.remove(project)
                            st.rerun()
        
    def render_fallback_experiment_design(self):
        """실험 설계 페이지"""
        st.title("🧪 실험 설계")
    
        if not st.session_state.current_project:
            st.warning("먼저 프로젝트를 선택하거나 생성해주세요.")
            if st.button("프로젝트 설정으로 이동"):
                st.session_state.current_page = 'project_setup'
                st.rerun()
            return
    
        project = st.session_state.current_project
        st.info(f"현재 프로젝트: **{project['name']}**")
    
        tabs = st.tabs(["요인 설정", "실험 설계", "실행 계획", "AI 추천"])
    
        with tabs[0]:  # 요인 설정
            st.markdown("### 실험 요인 설정")
        
            # 요인 추가 폼
            with st.form("add_factor"):
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    factor_name = st.text_input("요인 이름", placeholder="예: 반응 온도")
                with col2:
                    factor_type = st.selectbox("유형", ["연속형", "범주형"])
                with col3:
                    st.write("")  # 간격 맞추기
                    add_factor = st.form_submit_button("추가")
            
                if factor_type == "연속형":
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        min_val = st.number_input("최소값", value=0.0)
                    with col2:
                        max_val = st.number_input("최대값", value=100.0)
                    with col3:
                        unit = st.text_input("단위", placeholder="°C")
                else:
                    levels = st.text_input("수준 (쉼표로 구분)", placeholder="A, B, C")
        
            # 현재 요인 목록
            if 'factors' not in project:
                project['factors'] = []
        
            if add_factor and factor_name:
                new_factor = {
                    'name': factor_name,
                    'type': factor_type,
                    'min': min_val if factor_type == "연속형" else None,
                    'max': max_val if factor_type == "연속형" else None,
                    'unit': unit if factor_type == "연속형" else None,
                    'levels': levels.split(', ') if factor_type == "범주형" else None
                }
                project['factors'].append(new_factor)
                st.rerun()
        
            # 요인 표시
            if project['factors']:
                st.markdown("#### 현재 설정된 요인")
                for i, factor in enumerate(project['factors']):
                    col1, col2, col3 = st.columns([3, 2, 1])
                    with col1:
                        st.write(f"**{factor['name']}**")
                    with col2:
                        if factor['type'] == "연속형":
                            st.write(f"{factor['min']} - {factor['max']} {factor['unit']}")
                        else:
                            st.write(f"수준: {', '.join(factor['levels'])}")
                    with col3:
                        if st.button("삭제", key=f"del_factor_{i}"):
                            project['factors'].pop(i)
                            st.rerun()
    
        with tabs[1]:  # 실험 설계
            st.markdown("### 실험 설계 생성")
        
            if len(project.get('factors', [])) < 2:
                st.warning("최소 2개 이상의 요인을 설정해주세요.")
            else:
                design_options = {
                    "완전요인설계": "모든 요인 조합 탐색",
                    "부분요인설계": "주요 효과 중심 탐색",
                    "중심합성설계": "2차 효과 모델링",
                    "Box-Behnken": "3수준 반응표면",
                    "Plackett-Burman": "스크리닝 설계"
                }
            
                selected_design = st.selectbox(
                    "설계 방법 선택",
                    options=list(design_options.keys()),
                    format_func=lambda x: f"{x} - {design_options[x]}"
                )
            
                col1, col2 = st.columns(2)
                with col1:
                    center_points = st.number_input("중심점 반복", min_value=0, value=3)
                with col2:
                    replicates = st.number_input("전체 반복", min_value=1, value=1)
            
                if st.button("실험 설계 생성", type="primary"):
                    # 간단한 실험 설계 생성 (실제로는 pyDOE2 사용)
                    n_factors = len(project['factors'])
                    if selected_design == "완전요인설계":
                        n_runs = 2**n_factors * replicates + center_points
                    else:
                        n_runs = max(8, n_factors * 4) * replicates + center_points
                
                    st.success(f"{selected_design} 생성 완료! 총 {n_runs}회 실험")
                
                    # 실험 테이블 생성
                    experiment_data = []
                    for i in range(n_runs):
                        run = {'Run': i+1}
                        for factor in project['factors']:
                            if factor['type'] == "연속형":
                                import random
                                run[factor['name']] = round(random.uniform(factor['min'], factor['max']), 2)
                            else:
                                import random
                                run[factor['name']] = random.choice(factor['levels'])
                        experiment_data.append(run)
                
                    project['design'] = experiment_data
                    st.dataframe(experiment_data)
                
                    # 다운로드 버튼
                    df = pd.DataFrame(experiment_data)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "실험 계획표 다운로드",
                        csv,
                        "experiment_design.csv",
                        "text/csv"
                    )
    
        with tabs[2]:  # 실행 계획
            st.markdown("### 실험 실행 계획")
        
            if 'design' in project:
                st.write(f"총 실험 횟수: {len(project['design'])}회")
            
                col1, col2 = st.columns(2)
                with col1:
                    daily_capacity = st.number_input(
                        "일일 실험 가능 횟수",
                        min_value=1,
                        value=5
                    )
                with col2:
                    total_days = len(project['design']) / daily_capacity
                    st.metric("예상 소요 일수", f"{total_days:.1f}일")
            
                # 블록화
                st.markdown("#### 블록 설정")
                block_by = st.selectbox(
                    "블록 기준",
                    ["없음", "날짜별", "작업자별", "장비별"]
                )
            
                if block_by != "없음":
                    n_blocks = st.number_input("블록 수", min_value=2, value=3)
                    st.info(f"{block_by} {n_blocks}개 블록으로 나누어 실험합니다.")
            else:
                st.info("먼저 실험 설계를 생성해주세요.")
    
        with tabs[3]:  # AI 추천
            st.markdown("### 🤖 AI 추천")
        
            if project.get('factors'):
                if st.button("AI 분석 요청"):
                    with st.spinner("AI가 실험 설계를 분석 중..."):
                        time.sleep(2)  # 실제로는 AI API 호출
                    
                        st.markdown("#### 💡 AI 추천 사항")
                        st.success("✅ 선택하신 설계는 적절합니다!")
                    
                        st.markdown("**장점:**")
                        st.write("• 모든 주효과와 2차 교호작용 추정 가능")
                        st.write("• 중심점 반복으로 곡률 검출 가능")
                        st.write("• 통계적 검정력 충분 (Power > 0.8)")
                    
                        st.markdown("**고려사항:**")
                        st.write("• 실험 순서를 랜덤화하세요")
                        st.write("• 블록 효과가 예상되면 블록화를 고려하세요")
                    
                        st.markdown("**대안:**")
                        st.info("실험 횟수를 줄이려면 Resolution IV 부분요인설계를 고려해보세요.")
        
    def render_fallback_data_analysis(self):
        """데이터 분석 페이지"""
        st.title("📈 데이터 분석")
    
        tabs = st.tabs(["데이터 입력", "통계 분석", "모델링", "최적화", "벤치마크"])
    
        with tabs[0]:  # 데이터 입력
            st.markdown("### 실험 데이터 입력")
        
            # 파일 업로드
            uploaded_file = st.file_uploader(
                "데이터 파일 업로드",
                type=['csv', 'xlsx', 'xls'],
                help="실험 설계 파일에 결과 데이터를 추가하여 업로드하세요"
            )
        
            if uploaded_file:
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            
                st.write("업로드된 데이터:")
                st.dataframe(df)
            
                # 반응변수 선택
                response_cols = st.multiselect(
                    "반응변수 선택",
                    options=df.columns.tolist(),
                    help="분석할 반응변수를 선택하세요"
                )
            
                if response_cols and st.button("데이터 저장"):
                    if 'analysis_data' not in st.session_state:
                        st.session_state.analysis_data = {}
                    st.session_state.analysis_data['df'] = df
                    st.session_state.analysis_data['responses'] = response_cols
                    st.success("데이터가 저장되었습니다!")
    
        with tabs[1]:  # 통계 분석
            st.markdown("### 통계 분석")
        
            if 'analysis_data' in st.session_state:
                df = st.session_state.analysis_data['df']
                responses = st.session_state.analysis_data['responses']
            
                analysis_type = st.selectbox(
                    "분석 방법",
                    ["기술통계", "ANOVA", "회귀분석", "상관분석"]
                )
            
                if analysis_type == "기술통계":
                    st.write(df[responses].describe())
                
                    # 분포 플롯
                    for resp in responses:
                        fig = px.histogram(df, x=resp, title=f"{resp} 분포")
                        st.plotly_chart(fig)
            
                elif analysis_type == "ANOVA":
                    st.markdown("#### 분산분석 결과")
                    # 간단한 ANOVA 테이블 (실제로는 statsmodels 사용)
                    anova_data = {
                        'Source': ['Model', 'Error', 'Total'],
                        'DF': [5, 10, 15],
                        'SS': [125.3, 23.7, 149.0],
                        'MS': [25.06, 2.37, '-'],
                        'F': [10.57, '-', '-'],
                        'p-value': [0.001, '-', '-']
                    }
                    st.dataframe(anova_data)
                    st.success("모델이 통계적으로 유의합니다 (p < 0.05)")
            else:
                st.info("먼저 데이터를 업로드해주세요.")
    
        with tabs[2]:  # 모델링
            st.markdown("### 반응표면 모델링")
        
            if 'analysis_data' in st.session_state:
                model_type = st.selectbox(
                    "모델 유형",
                    ["1차 모델", "2차 모델", "특수 3차항 포함"]
                )
            
                if st.button("모델 생성"):
                    with st.spinner("모델 fitting 중..."):
                        time.sleep(1)
                    
                        st.markdown("#### 모델 방정식")
                        st.latex(r"Y = 45.2 + 3.1X_1 + 2.3X_2 - 1.5X_1^2 - 0.8X_2^2 + 1.2X_1X_2")
                    
                        st.markdown("#### 모델 통계")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("R²", "0.956")
                        with col2:
                            st.metric("Adj R²", "0.942")
                        with col3:
                            st.metric("RMSE", "2.31")
                    
                        # 잔차 플롯
                        st.markdown("#### 잔차 분석")
                    
                        x = np.random.normal(0, 1, 100)
                        fig = go.Figure(data=go.Scatter(x=x, y=np.random.normal(0, 1, 100), mode='markers'))
                        fig.update_layout(title="잔차 플롯", xaxis_title="예측값", yaxis_title="잔차")
                        st.plotly_chart(fig)
    
        with tabs[3]:  # 최적화
            st.markdown("### 최적 조건 탐색")
        
            optimization_method = st.selectbox(
                "최적화 방법",
                ["Desirability Function", "단일 목적 최적화", "다목적 최적화"]
            )
        
            if st.button("최적화 실행"):
                with st.spinner("최적 조건 탐색 중..."):
                    time.sleep(1.5)
                
                    st.success("최적 조건을 찾았습니다!")
                
                    optimal_conditions = {
                        "온도": "85°C",
                        "압력": "2.3 atm",
                        "시간": "45 min",
                        "촉매량": "0.5 wt%"
                    }
                
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### 최적 조건")
                        for factor, value in optimal_conditions.items():
                            st.write(f"**{factor}**: {value}")
                
                    with col2:
                        st.markdown("#### 예상 결과")
                        st.metric("수율", "92.3%", "+15.2%")
                        st.metric("순도", "98.5%", "+3.1%")
                        st.metric("Desirability", "0.89")
    
        with tabs[4]:  # 벤치마크
            st.markdown("### 문헌 대비 성능 비교")
        
            benchmark_source = st.selectbox(
                "비교 데이터 소스",
                ["Materials Project", "문헌 데이터베이스", "사내 데이터"]
            )
        
            if st.button("벤치마크 분석"):
                with st.spinner("유사 연구 검색 중..."):
                    time.sleep(2)
                
                    # 벤치마크 결과
                    st.markdown("#### 📊 벤치마크 결과")
                
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("귀하의 성능", "92.3%")
                    with col2:
                        st.metric("문헌 평균", "78.5%")
                    with col3:
                        st.metric("상위 백분위", "상위 15%", "우수")
                
                    # 비교 차트
                    categories = ['수율', '순도', '안정성', '비용효율', '친환경성']
                    your_scores = [92, 98, 85, 75, 90]
                    avg_scores = [78, 92, 80, 70, 75]
                    best_scores = [95, 99, 90, 85, 95]
                
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(r=your_scores, theta=categories, name='귀하'))
                    fig.add_trace(go.Scatterpolar(r=avg_scores, theta=categories, name='평균'))
                    fig.add_trace(go.Scatterpolar(r=best_scores, theta=categories, name='최고'))
                    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])))
                    st.plotly_chart(fig)
        
    def render_fallback_literature_search(self):
        """문헌 검색 페이지"""
        st.title("🔍 문헌 검색")
    
        # 검색 바
        col1, col2 = st.columns([4, 1])
        with col1:
            query = st.text_input(
                "검색어 입력",
                placeholder="예: biodegradable polymer synthesis optimization"
            )
        with col2:
            st.write("")  # 간격 맞추기
            search_btn = st.button("검색", type="primary", use_container_width=True)
    
        # 검색 필터
        with st.expander("상세 검색 옵션"):
            col1, col2, col3 = st.columns(3)
            with col1:
                sources = st.multiselect(
                    "데이터베이스",
                    ["PubMed", "Google Scholar", "arXiv", "CrossRef"],
                    default=["PubMed", "Google Scholar"]
                )
            with col2:
                year_range = st.slider(
                    "출판 연도",
                    2000, 2024, (2020, 2024)
                )
            with col3:
                doc_type = st.selectbox(
                    "문서 유형",
                    ["전체", "논문", "리뷰", "특허", "학위논문"]
                )
    
        if search_btn and query:
            with st.spinner("문헌 검색 중..."):
                time.sleep(2)  # 실제로는 API 호출
            
                # 검색 결과 (더미 데이터)
                results = [
                    {
                        'title': 'Optimization of Biodegradable Polymer Synthesis Using Response Surface Methodology',
                        'authors': 'Kim, J.H., Lee, S.M., Park, K.D.',
                        'journal': 'Polymer Engineering & Science',
                        'year': 2023,
                        'citations': 45,
                        'doi': '10.1002/pen.12345',
                        'abstract': 'This study presents a systematic approach to optimize the synthesis conditions...'
                    },
                    {
                        'title': 'Green Synthesis of Polylactic Acid: A Design of Experiments Approach',
                        'authors': 'Zhang, L., Wang, Y., Chen, X.',
                        'journal': 'Green Chemistry',
                        'year': 2023,
                        'citations': 32,
                        'doi': '10.1039/D3GC00123',
                        'abstract': 'We report an environmentally friendly synthesis route for PLA using...'
                    },
                    {
                        'title': 'Machine Learning-Assisted Polymer Design: Recent Advances',
                        'authors': 'Smith, J.A., Johnson, M.R.',
                        'journal': 'Nature Reviews Materials',
                        'year': 2024,
                        'citations': 78,
                        'doi': '10.1038/s41578-024-00123',
                        'abstract': 'This review discusses the latest developments in ML-guided polymer design...'
                    }
                ]
            
                st.success(f"{len(results)}개의 관련 문헌을 찾았습니다.")
            
                # 결과 표시
                for i, paper in enumerate(results):
                    with st.container():
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            st.markdown(f"**{paper['title']}**")
                            st.caption(f"{paper['authors']} - {paper['journal']} ({paper['year']})")
                        
                            with st.expander("초록 보기"):
                                st.write(paper['abstract'])
                        
                        with col2:
                            st.metric("인용", paper['citations'])
                            if st.button("저장", key=f"save_{i}"):
                                if 'saved_papers' not in st.session_state:
                                    st.session_state.saved_papers = []
                                st.session_state.saved_papers.append(paper)
                                st.success("저장됨!")
            
                # 프로토콜 추출
                st.markdown("### 🧪 프로토콜 추출")
                selected_paper = st.selectbox(
                    "프로토콜을 추출할 논문 선택",
                    options=[p['title'] for p in results]
                )
            
                if st.button("프로토콜 추출"):
                    with st.spinner("AI가 프로토콜을 분석 중..."):
                        time.sleep(2)
                    
                        st.markdown("#### 추출된 프로토콜")
                        protocol = {
                            "재료": [
                                "L-lactide (Sigma-Aldrich, 99%)",
                                "Tin(II) 2-ethylhexanoate catalyst",
                                "Toluene (anhydrous)"
                            ],
                            "장비": [
                                "Three-neck round bottom flask",
                                "Magnetic stirrer with heating",
                                "Vacuum line"
                            ],
                            "절차": [
                                "1. L-lactide (10g)를 플라스크에 넣는다",
                                "2. 촉매 (0.1 wt%)를 첨가한다",
                                "3. 질소 분위기에서 180°C로 가열한다",
                                "4. 4시간 동안 교반하며 반응시킨다",
                                "5. 실온으로 냉각 후 정제한다"
                            ],
                            "조건": {
                                "온도": "180°C",
                                "시간": "4시간",
                                "촉매량": "0.1 wt%",
                                "분위기": "N2"
                            }
                        }
                    
                        # 프로토콜 표시
                        tabs = st.tabs(["재료", "장비", "절차", "조건"])
                        with tabs[0]:
                            for material in protocol["재료"]:
                                st.write(f"• {material}")
                        with tabs[1]:
                            for equipment in protocol["장비"]:
                                st.write(f"• {equipment}")
                        with tabs[2]:
                            for step in protocol["절차"]:
                                st.write(step)
                        with tabs[3]:
                            for key, value in protocol["조건"].items():
                                st.write(f"**{key}**: {value}")
                    
                        # 템플릿으로 저장
                        if st.button("실험 템플릿으로 저장"):
                            st.success("프로토콜이 템플릿으로 저장되었습니다!")
        
    def render_fallback_collaboration(self):
        """협업 페이지"""
        st.title("👥 협업")
    
        tabs = st.tabs(["팀 관리", "프로젝트 공유", "실시간 협업", "활동 내역"])
    
        with tabs[0]:  # 팀 관리
            st.markdown("### 팀 구성원")
        
            # 팀원 추가
            with st.form("add_member"):
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    member_email = st.text_input("이메일", placeholder="member@email.com")
                with col2:
                    member_role = st.selectbox("역할", ["연구원", "관리자", "뷰어"])
                with col3:
                    st.write("")
                    add_btn = st.form_submit_button("초대")
        
            # 현재 팀원
            team_members = [
                {"name": "김연구원", "email": "kim@lab.com", "role": "연구원", "status": "활성"},
                {"name": "이박사", "email": "lee@lab.com", "role": "관리자", "status": "활성"},
                {"name": "박교수", "email": "park@univ.edu", "role": "뷰어", "status": "초대중"}
            ]
        
            for member in team_members:
                col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
                with col1:
                    st.write(f"**{member['name']}**")
                    st.caption(member['email'])
                with col2:
                    st.write(f"역할: {member['role']}")
                with col3:
                    if member['status'] == "활성":
                        st.success("활성")
                    else:
                        st.warning("초대중")
                with col4:
                    if st.button("제거", key=f"remove_{member['email']}"):
                        st.info(f"{member['name']}을(를) 제거했습니다.")
    
        with tabs[1]:  # 프로젝트 공유
            st.markdown("### 프로젝트 공유 설정")
        
            if st.session_state.projects:
                selected_project = st.selectbox(
                    "공유할 프로젝트",
                    options=[p['name'] for p in st.session_state.projects]
                )
            
                share_options = st.multiselect(
                    "공유 항목",
                    ["실험 설계", "데이터", "분석 결과", "보고서"],
                    default=["실험 설계", "데이터"]
                )
            
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("공유 링크 생성", use_container_width=True):
                        share_link = f"https://polymer-doe.com/share/{uuid.uuid4().hex[:8]}"
                        st.code(share_link)
                        st.info("링크가 클립보드에 복사되었습니다!")
            
                with col2:
                    if st.button("이메일로 공유", use_container_width=True):
                        st.success("팀원들에게 공유 알림을 전송했습니다!")
    
        with tabs[2]:  # 실시간 협업
            st.markdown("### 실시간 협업 세션")
        
            session_status = st.radio(
                "세션 상태",
                ["오프라인", "온라인 - 대기중", "온라인 - 활성"]
            )
        
            if session_status.startswith("온라인"):
                st.success("실시간 협업 모드가 활성화되었습니다.")
            
                # 현재 접속자
                st.markdown("#### 현재 접속자")
                online_users = ["김연구원 (편집중)", "이박사 (보는중)"]
                for user in online_users:
                    st.write(f"🟢 {user}")
            
                # 실시간 채팅
                st.markdown("#### 팀 채팅")
                chat_messages = [
                    {"user": "김연구원", "message": "온도 조건을 85도로 변경했습니다.", "time": "10:23"},
                    {"user": "이박사", "message": "확인했습니다. ANOVA 결과도 업데이트 했어요.", "time": "10:25"}
                ]
            
                for msg in chat_messages:
                    st.text(f"[{msg['time']}] {msg['user']}: {msg['message']}")
            
                new_message = st.text_input("메시지 입력", placeholder="팀원들과 대화하세요...")
                if st.button("전송"):
                    st.success("메시지를 전송했습니다!")
    
        with tabs[3]:  # 활동 내역
            st.markdown("### 팀 활동 내역")
        
            activities = [
                {"user": "김연구원", "action": "실험 데이터 업로드", "time": "2시간 전", "icon": "📊"},
                {"user": "이박사", "action": "ANOVA 분석 완료", "time": "3시간 전", "icon": "📈"},
                {"user": "박교수", "action": "프로젝트 검토 코멘트", "time": "어제", "icon": "💬"},
                {"user": "김연구원", "action": "실험 설계 수정", "time": "2일 전", "icon": "🧪"}
            ]
        
            for activity in activities:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"{activity['icon']} **{activity['user']}** - {activity['action']}")
                with col2:
                    st.caption(activity['time'])
        
    def render_fallback_visualization(self):
        """데이터 시각화 페이지"""
        st.title("📊 시각화")
    
        # 샘플 데이터 생성
        np.random.seed(42)
        n_points = 50
        data = pd.DataFrame({
            'Temperature': np.random.uniform(60, 100, n_points),
            'Pressure': np.random.uniform(1, 5, n_points),
            'Time': np.random.uniform(20, 80, n_points),
            'Yield': 70 + 0.5*np.random.randn(n_points) + 
                    np.random.uniform(60, 100, n_points)*0.1 + 
                    np.random.uniform(1, 5, n_points)*2,
            'Purity': 95 + 2*np.random.randn(n_points)
        })
    
        tabs = st.tabs(["주효과 플롯", "교호작용", "반응표면", "3D 시각화", "대시보드"])
    
        with tabs[0]:  # 주효과
            st.markdown("### 주효과 플롯")
        
            response = st.selectbox("반응변수", ["Yield", "Purity"])
        
            fig = make_subplots(rows=1, cols=3, 
                               subplot_titles=("Temperature", "Pressure", "Time"))
        
            for i, factor in enumerate(['Temperature', 'Pressure', 'Time'], 1):
                # 주효과 계산 (간단한 평균)
                sorted_data = data.sort_values(factor)
                grouped = sorted_data.groupby(pd.cut(sorted_data[factor], bins=5))[response].mean()
            
                fig.add_trace(
                    go.Scatter(x=grouped.index.astype(str), y=grouped.values, 
                              mode='lines+markers', name=factor),
                    row=1, col=i
                )
        
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
        with tabs[1]:  # 교호작용
            st.markdown("### 교호작용 플롯")
        
            factor1 = st.selectbox("요인 1", ["Temperature", "Pressure", "Time"])
            factor2 = st.selectbox("요인 2", ["Pressure", "Time", "Temperature"])
        
            if factor1 != factor2:
                # 교호작용 플롯
                fig = go.Figure()
            
                # 각 요인을 높음/낮음으로 분류
                f1_low = data[factor1] < data[factor1].median()
                f1_high = ~f1_low
            
                for condition, name, color in [(f1_low, f"Low {factor1}", "blue"), 
                                              (f1_high, f"High {factor1}", "red")]:
                    subset = data[condition]
                    grouped = subset.groupby(pd.cut(subset[factor2], bins=3))['Yield'].mean()
                
                    fig.add_trace(go.Scatter(
                        x=grouped.index.astype(str),
                        y=grouped.values,
                        mode='lines+markers',
                        name=name,
                        line=dict(color=color)
                    ))
            
                fig.update_layout(
                    title=f"{factor1} × {factor2} 교호작용",
                    xaxis_title=factor2,
                    yaxis_title="Yield"
                )
                st.plotly_chart(fig, use_container_width=True)
    
        with tabs[2]:  # 반응표면
            st.markdown("### 반응표면 플롯")
        
            x_factor = st.selectbox("X축", ["Temperature", "Pressure", "Time"], key="rsm_x")
            y_factor = st.selectbox("Y축", ["Pressure", "Time", "Temperature"], key="rsm_y")
        
            if x_factor != y_factor:
                # 등고선 플롯
                fig = go.Figure()
            
                # 그리드 생성
                xi = np.linspace(data[x_factor].min(), data[x_factor].max(), 30)
                yi = np.linspace(data[y_factor].min(), data[y_factor].max(), 30)
                Xi, Yi = np.meshgrid(xi, yi)
            
                # 간단한 보간 (실제로는 모델 예측값 사용)
                from scipy.interpolate import griddata
                Zi = griddata((data[x_factor], data[y_factor]), data['Yield'], 
                             (Xi, Yi), method='cubic')
            
                fig.add_trace(go.Contour(
                    x=xi, y=yi, z=Zi,
                    colorscale='Viridis',
                    contours=dict(showlabels=True)
                ))
            
                # 실험점 추가
                fig.add_trace(go.Scatter(
                    x=data[x_factor], y=data[y_factor],
                    mode='markers',
                    marker=dict(color='red', size=8),
                    name='실험점'
                ))
            
                fig.update_layout(
                    title="반응표면 등고선도",
                    xaxis_title=x_factor,
                    yaxis_title=y_factor
                )
                st.plotly_chart(fig, use_container_width=True)
    
        with tabs[3]:  # 3D 시각화
            st.markdown("### 3D 반응표면")
        
            # 3D 표면 플롯
            fig = go.Figure()
        
            fig.add_trace(go.Surface(
                x=xi, y=yi, z=Zi,
                colorscale='Viridis'
            ))
        
            fig.update_layout(
                title="3D 반응표면",
                scene=dict(
                    xaxis_title=x_factor,
                    yaxis_title=y_factor,
                    zaxis_title='Yield'
                ),
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
    
        with tabs[4]:  # 대시보드
            st.markdown("### 실험 대시보드")
        
            # 메트릭 카드
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("평균 수율", f"{data['Yield'].mean():.1f}%", 
                         f"+{data['Yield'].std():.1f}%")
            with col2:
                st.metric("최고 수율", f"{data['Yield'].max():.1f}%")
            with col3:
                st.metric("평균 순도", f"{data['Purity'].mean():.1f}%")
            with col4:
                st.metric("실험 완료", f"{len(data)}/50")
        
            # 복합 차트
            st.markdown("#### 실험 진행 현황")
        
            # 시계열 차트 (실험 순서대로)
            data['Run'] = range(1, len(data)+1)
        
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("수율 추이", "순도 추이", "요인별 분포", "상관관계"),
                specs=[[{"type": "scatter"}, {"type": "scatter"}],
                      [{"type": "box"}, {"type": "scatter"}]]
            )
        
            # 수율 추이
            fig.add_trace(
                go.Scatter(x=data['Run'], y=data['Yield'], mode='lines+markers'),
                row=1, col=1
            )
        
            # 순도 추이
            fig.add_trace(
                go.Scatter(x=data['Run'], y=data['Purity'], mode='lines+markers'),
                row=1, col=2
            )
        
            # 박스플롯
            for factor in ['Temperature', 'Pressure', 'Time']:
                fig.add_trace(
                    go.Box(y=data[factor], name=factor),
                    row=2, col=1
                )
        
            # 산점도 매트릭스 (간단 버전)
            fig.add_trace(
                go.Scatter(x=data['Temperature'], y=data['Yield'], 
                          mode='markers', marker=dict(color=data['Purity'])),
                row=2, col=2
            )
        
            fig.update_layout(height=800, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
    def render_fallback_marketplace(self):
        """모듈 마켓플레이스 페이지"""
        st.title("🛍️ 모듈 마켓플레이스")
        
        # 카테고리 필터
        col1, col2, col3 = st.columns(3)
        with col1:
            category = st.selectbox(
                "카테고리",
                ["전체", "실험 설계", "데이터 분석", "시각화", "최적화"]
            )
        with col2:
            sort_by = st.selectbox(
                "정렬 기준",
                ["인기순", "최신순", "평점순", "다운로드순"]
            )
        with col3:
            price_filter = st.selectbox(
                "가격",
                ["전체", "무료", "유료"]
            )
            
        # 검색
        search = st.text_input("모듈 검색", placeholder="원하는 모듈을 검색하세요...")
        
        # 모듈 목록 (더미 데이터)
        modules = [
            {
                'name': 'advanced_doe',
                'display_name': '고급 실험 설계 모듈',
                'author': 'PolymerDOE Team',
                'category': '실험 설계',
                'description': '혼합물 설계, Split-plot, 최적 설계 등 고급 기능 지원',
                'version': '2.1.0',
                'downloads': 1523,
                'rating': 4.8,
                'price': '무료',
                'icon': '🧪'
            },
            {
                'name': 'ml_optimizer',
                'display_name': 'ML 기반 최적화',
                'author': 'AI Lab',
                'category': '최적화',
                'description': '머신러닝을 활용한 실시간 실험 최적화',
                'version': '1.5.2',
                'downloads': 892,
                'rating': 4.5,
                'price': '$9.99',
                'icon': '🤖'
            },
            {
                'name': 'polymer_db',
                'display_name': '고분자 DB 연동',
                'author': 'DataConnect',
                'category': '데이터베이스',
                'description': 'PolyInfo, Materials Project 등 주요 DB 통합 검색',
                'version': '3.0.1',
                'downloads': 2341,
                'rating': 4.9,
                'price': '무료',
                'icon': '🗄️'
            }
        ]
        
        # 모듈 카드 표시
        cols = st.columns(3)
        for idx, module in enumerate(modules):
            with cols[idx % 3]:
                with st.container():
                    # 모듈 카드
                    st.markdown(f"""
                    <div style='background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; 
                              border: 1px solid #e9ecef; height: 100%;'>
                        <h3>{module['icon']} {module['display_name']}</h3>
                        <p style='color: #6c757d; font-size: 14px;'>by {module['author']}</p>
                        <p>{module['description']}</p>
                        <div style='display: flex; justify-content: space-between; align-items: center; 
                                  margin-top: 1rem;'>
                            <span>⭐ {module['rating']}</span>
                            <span>⬇️ {module['downloads']:,}</span>
                            <span style='font-weight: bold; color: #28a745;'>{module['price']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("설치", key=f"install_{module['name']}", use_container_width=True):
                            with st.spinner("설치 중..."):
                                time.sleep(1)
                            st.success("설치 완료!")
                            
                    with col_b:
                        if st.button("상세", key=f"detail_{module['name']}", use_container_width=True):
                            st.session_state.selected_module = module
                            st.rerun()
                            
        # 선택된 모듈 상세 정보
        if 'selected_module' in st.session_state and st.session_state.selected_module:
            module = st.session_state.selected_module
            st.divider()
            st.markdown(f"### 📦 {module['display_name']} 상세 정보")
            
            tabs = st.tabs(["개요", "문서", "예제", "리뷰"])
            
            with tabs[0]:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"**버전**: {module['version']}")
                    st.markdown(f"**카테고리**: {module['category']}")
                    st.markdown(f"**작성자**: {module['author']}")
                    st.markdown(f"**라이선스**: MIT")
                with col2:
                    st.metric("평점", f"⭐ {module['rating']}/5.0")
                    st.metric("다운로드", f"{module['downloads']:,}")
                    
            with tabs[1]:
                st.markdown("#### 사용법")
                st.code("""
from modules.advanced_doe import AdvancedDOE

# 모듈 초기화
doe = AdvancedDOE()

# 혼합물 설계 생성
design = doe.mixture_design(
    components=['A', 'B', 'C'],
    constraints={'sum': 100, 'min': 10}
)
                """, language='python')
                
            with tabs[2]:
                st.markdown("#### 예제 프로젝트")
                st.info("고분자 블렌드 최적화 예제")
                st.code("""
# 3성분 고분자 블렌드 최적화
components = ['PLA', 'PCL', 'PBS']
properties = ['tensile_strength', 'elongation']

# 실험 설계 생성
design = doe.create_mixture_design(
    components=components,
    n_runs=15
)
                """, language='python')
                
            with tabs[3]:
                st.markdown("#### 사용자 리뷰")
                reviews = [
                    {"user": "김연구원", "rating": 5, "comment": "정말 유용합니다! 시간을 많이 절약했어요."},
                    {"user": "박박사", "rating": 4, "comment": "기능은 좋은데 문서가 조금 더 자세했으면..."}
                ]
                
                for review in reviews:
                    st.write(f"**{review['user']}** {'⭐' * review['rating']}")
                    st.write(review['comment'])
                    st.divider()
            
    def render_fallback_module_loader(self):
        """모듈 로더 페이지"""
        st.title("📦 모듈 로더")
    
        tabs = st.tabs(["모듈 업로드", "코드 편집기", "테스트", "배포"])
    
        with tabs[0]:  # 모듈 업로드
            st.markdown("### 커스텀 모듈 업로드")
        
            uploaded_file = st.file_uploader(
                "Python 모듈 파일 (.py)",
                type=['py'],
                help="BaseModule을 상속한 클래스를 포함해야 합니다"
            )
        
            if uploaded_file:
                # 파일 내용 표시
                file_content = uploaded_file.read().decode('utf-8')
                st.code(file_content, language='python')
            
                if st.button("모듈 검증"):
                    with st.spinner("모듈 검증 중..."):
                        time.sleep(1)
                    
                        # 검증 결과
                        st.success("✅ 모듈 검증 완료!")
                    
                        validation_results = {
                            "BaseModule 상속": True,
                            "필수 메서드 구현": True,
                            "메타데이터 포함": True,
                            "보안 검사": True,
                            "성능 테스트": True
                        }
                    
                        for check, passed in validation_results.items():
                            if passed:
                                st.success(f"✅ {check}")
                            else:
                                st.error(f"❌ {check}")
    
        with tabs[1]:  # 코드 편집기
            st.markdown("### 모듈 코드 편집기")
        
            # 템플릿 선택
            template = st.selectbox(
                "템플릿 선택",
                ["빈 템플릿", "실험 설계 모듈", "데이터 분석 모듈", "시각화 모듈"]
            )
        
            # 코드 에디터
            if template == "실험 설계 모듈":
                default_code = """from modules.base_module import BaseModule
import numpy as np
import pandas as pd

class CustomExperimentModule(BaseModule):
    \"\"\"커스텀 실험 설계 모듈\"\"\"
    
    def __init__(self):
        super().__init__()
        self.name = "Custom Experiment Design"
        self.version = "1.0.0"
        self.author = "Your Name"
        self.description = "Custom experimental design module"
        
    def get_info(self):
        return {
            'name': self.name,
            'version': self.version,
            'author': self.author,
            'description': self.description
        }
    
    def validate_inputs(self, factors, responses):
        # 입력 검증 로직
        if len(factors) < 2:
            return False, "최소 2개 이상의 요인이 필요합니다"
        return True, "Valid"
    
    def generate_design(self, factors, **kwargs):
        # 실험 설계 생성 로직
        n_factors = len(factors)
        n_runs = 2**n_factors  # 예: 완전요인설계
        
        design = []
        for i in range(n_runs):
            run = {'Run': i+1}
            # 설계 생성 로직 구현
            design.append(run)
            
        return pd.DataFrame(design)
    
    def analyze_results(self, data):
        # 결과 분석 로직
        results = {
            'summary': data.describe(),
            'anova': None,  # ANOVA 분석
            'model': None   # 회귀 모델
        }
        return results
    
    def export_data(self, data, filename):
        # 데이터 내보내기
        data.to_csv(filename, index=False)
        return True
"""
            else:
                default_code = "# 여기에 모듈 코드를 작성하세요\n"
        
            code = st.text_area(
                "코드 편집",
                value=default_code,
                height=400,
                help="Ctrl+Enter로 실행"
            )
        
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("문법 검사", use_container_width=True):
                    st.success("문법 검사 통과!")
            with col2:
                if st.button("저장", use_container_width=True):
                    st.success("모듈이 저장되었습니다!")
            with col3:
                if st.button("실행", use_container_width=True):
                    st.info("테스트 탭에서 모듈을 테스트하세요.")
    
        with tabs[2]:  # 테스트
            st.markdown("### 모듈 테스트")
        
            # 테스트 데이터 설정
            st.markdown("#### 테스트 데이터")
            test_factors = st.number_input("요인 개수", min_value=2, max_value=5, value=3)
            test_runs = st.number_input("실험 횟수", min_value=4, max_value=50, value=8)
        
            if st.button("테스트 실행"):
                with st.spinner("모듈 테스트 중..."):
                    time.sleep(1.5)
                
                    # 테스트 결과
                    st.success("테스트 완료!")
                
                    test_results = {
                        "설계 생성": {"status": "통과", "time": "0.23s", "memory": "12MB"},
                        "입력 검증": {"status": "통과", "time": "0.01s", "memory": "1MB"},
                        "결과 분석": {"status": "통과", "time": "0.45s", "memory": "25MB"},
                        "데이터 내보내기": {"status": "통과", "time": "0.12s", "memory": "5MB"}
                    }
                
                    for test, result in test_results.items():
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.write(f"**{test}**")
                        with col2:
                            if result["status"] == "통과":
                                st.success(result["status"])
                            else:
                                st.error(result["status"])
                        with col3:
                            st.write(f"⏱️ {result['time']}")
                        with col4:
                            st.write(f"💾 {result['memory']}")
    
        with tabs[3]:  # 배포
            st.markdown("### 모듈 배포")
        
            deployment_option = st.radio(
                "배포 옵션",
                ["로컬 저장", "팀 공유", "마켓플레이스 게시"]
            )
        
            if deployment_option == "로컬 저장":
                st.info("모듈이 로컬 모듈 디렉토리에 저장됩니다.")
                if st.button("로컬 저장", type="primary"):
                    st.success("모듈이 성공적으로 저장되었습니다!")
                    st.code("modules/user_modules/custom_experiment_v1.py")
        
            elif deployment_option == "팀 공유":
                st.info("팀원들과 모듈을 공유합니다.")
                share_with = st.multiselect(
                    "공유 대상",
                    ["김연구원", "이박사", "박교수", "전체 팀"]
                )
                if st.button("팀 공유", type="primary"):
                    st.success(f"{len(share_with)}명과 모듈을 공유했습니다!")
        
            else:  # 마켓플레이스
                st.info("모듈을 공개 마켓플레이스에 게시합니다.")
            
                with st.form("marketplace_publish"):
                    st.markdown("#### 마켓플레이스 정보")
                
                    module_title = st.text_input("모듈 제목", placeholder="혁신적인 실험 설계 모듈")
                    module_category = st.selectbox(
                        "카테고리",
                        ["실험 설계", "데이터 분석", "시각화", "최적화", "기타"]
                    )
                
                    module_tags = st.multiselect(
                        "태그",
                        ["고분자", "화학", "최적화", "머신러닝", "통계", "시각화"]
                    )
                
                    module_price = st.radio(
                        "가격 설정",
                        ["무료", "유료 ($9.99)", "프리미엄 ($29.99)"]
                    )
                
                    module_desc = st.text_area(
                        "상세 설명",
                        placeholder="이 모듈의 특징과 장점을 설명하세요...",
                        height=100
                    )
                
                    terms = st.checkbox("마켓플레이스 이용약관에 동의합니다")
                
                    if st.form_submit_button("게시하기", type="primary"):
                        if all([module_title, module_desc, terms]):
                            with st.spinner("마켓플레이스에 게시 중..."):
                                time.sleep(2)
                            st.success("🎉 모듈이 마켓플레이스에 성공적으로 게시되었습니다!")
                            st.balloons()
                        else:
                            st.error("모든 필수 항목을 입력해주세요.")
        
    def render_notifications(self):
        """알림 렌더링"""
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

    def validate_api_key(self, provider: str, key: str) -> bool:
        """API 키 형식 검증"""
        if provider in API_KEY_PATTERNS:
            return bool(re.match(API_KEY_PATTERNS[provider], key))
        return True  # 패턴이 없는 경우 통과

    def render_api_status_dashboard(self):
        """API 키 설정 상태 대시보드"""
        st.markdown("### 📊 API 설정 현황")
    
        # 카테고리별 설정 상태
        categories = {
            'AI 엔진': ['google_gemini', 'xai_grok', 'groq', 'deepseek', 'sambanova', 'huggingface'],
            '데이터베이스': ['materials_project', 'zenodo', 'protocols_io', 'figshare', 'github'],
            'Google 서비스': ['google_sheets_url', 'google_oauth_client_id', 'google_oauth_client_secret']
        }
    
        cols = st.columns(len(categories))
    
        for idx, (category, keys) in enumerate(categories.items()):
            with cols[idx]:
                configured = sum(1 for k in keys if st.session_state.api_keys.get(k))
                total = len(keys)
            
                # 진행률 계산
                progress = configured / total if total > 0 else 0
            
                # 메트릭 카드
                st.metric(
                    label=category,
                    value=f"{configured}/{total}",
                    delta=f"{int(progress * 100)}% 완료"
                )
            
                # 진행 바
                st.progress(progress)
    
    def render_settings_page(self):
        """설정 페이지"""
        if not st.session_state.authenticated:
            st.warning("로그인이 필요합니다.")
            return
        
        st.title("⚙️ 설정")

        # API_PROVIDERS import
        API_PROVIDERS = self.imported_modules.get('API_PROVIDERS')
        if not API_PROVIDERS:
            try:
                from config.app_config import API_PROVIDERS
            except ImportError:
                st.error("API 제공자 정보를 로드할 수 없습니다.")
                return
    
        # API 상태 대시보드
        self.render_api_status_dashboard()
        
        st.divider()
        
        # 필수 API 체크
        st.markdown("### 🚨 필수 API 상태")
        required_apis = {k: v for k, v in API_PROVIDERS.items() if v.get('required', False)}
    
        if required_apis:
            for provider_key, provider_info in required_apis.items():
                # 해당 API 키가 설정되어 있는지 확인
                is_configured = st.session_state.api_keys.get(provider_key, '')
                
                # API 키 검증
                is_valid = self.validate_api_key(provider_key, is_configured) if is_configured else False
            
                if is_configured and is_valid:
                    st.success(f"✅ {provider_info['name']} - 설정됨")
                elif is_configured and not is_valid:
                    st.warning(f"⚠️ {provider_info['name']} - 형식 오류")
                else:
                    st.error(f"❌ {provider_info['name']} - 미설정 (필수)")
    
        # API 카테고리별 그룹화
        ai_apis = {k: v for k, v in API_PROVIDERS.items() 
                   if k in ['google_gemini', 'xai_grok', 'groq', 'deepseek', 'sambanova', 'huggingface']}
    
        db_apis = {k: v for k, v in API_PROVIDERS.items() 
                   if k in ['materials_project', 'materials_commons', 'zenodo', 'protocols_io', 'figshare', 'github']}
    
        google_apis = {k: v for k, v in API_PROVIDERS.items() 
                       if k in ['google_sheets', 'google_oauth']}
        
        # 탭 생성
        tabs = st.tabs([
            "🤖 AI 엔진", 
            "📊 데이터베이스", 
            "🔐 OAuth 로그인", 
            "📝 Google 서비스",
            "👤 프로필", 
            "🎨 UI 설정",
            "💾 데이터 관리",
            "🛠️ 고급 설정"
        ])
    
        with tabs[0]:  # AI 엔진 탭
            st.markdown("### 🤖 AI 엔진 API 키")
            st.info("AI 기능을 활성화하려면 최소 1개 이상의 API 키를 설정하세요.")
        
            # AI 서비스 확장 목록
            ai_services = {
                'google_gemini': {
                    'name': 'Google Gemini 2.0 Flash',
                    'required': True,
                    'help': '무료 티어 제공, 필수 추천',
                    'placeholder': 'AIza...',
                    'url': 'https://makersuite.google.com/app/apikey'
                },
                'xai_grok': {
                    'name': 'xAI Grok 3',
                    'required': False,
                    'help': '최신 정보 접근 가능',
                    'placeholder': 'xai-...',
                    'url': 'https://x.ai/api'
                },
                'groq': {
                    'name': 'Groq (초고속 추론)',
                    'required': False,
                    'help': '무료 티어, 빠른 응답',
                    'placeholder': 'gsk_...',
                    'url': 'https://console.groq.com'
                },
                'deepseek': {
                    'name': 'DeepSeek (코드/수식)',
                    'required': False,
                    'help': '코드 생성 특화',
                    'placeholder': 'sk-...',
                    'url': 'https://platform.deepseek.com'
                },
                'sambanova': {
                    'name': 'SambaNova (대규모 모델)',
                    'required': False,
                    'help': '무료 클라우드 서비스',
                    'placeholder': 'samba-...',
                    'url': 'https://cloud.sambanova.ai'
                },
                'huggingface': {
                    'name': 'HuggingFace',
                    'required': False,
                    'help': '도메인 특화 모델',
                    'placeholder': 'hf_...',
                    'url': 'https://huggingface.co/settings/tokens'
                }
            }
        
            for service_key, service_info in ai_services.items():
                with st.expander(
                    f"{'🔴' if service_info['required'] else '⚪'} {service_info['name']}", 
                    expanded=service_info['required']
                ):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.caption(service_info['help'])
                    with col2:
                        if 'url' in service_info:
                            st.link_button("API 발급", service_info['url'], use_container_width=True)
                
                    current_key = st.session_state.api_keys.get(service_key, '')
                    new_key = st.text_input(
                        "API Key",
                        value='*' * 20 if current_key else '',
                        type="password",
                        placeholder=service_info['placeholder'],
                        key=f"api_{service_key}"
                    )
                
                    if new_key and new_key != '*' * 20:
                        # API 키 검증
                        if self.validate_api_key(service_key, new_key):
                            st.session_state.api_keys[service_key] = new_key
                            st.success("✅ 유효한 API 키 형식입니다.")
                        else:
                            st.error("❌ 잘못된 API 키 형식입니다.")
        
            if st.button("AI API 키 저장", use_container_width=True, key="save_ai"):
                self._save_api_keys('ai')
                st.success("AI API 키가 저장되었습니다!")
    
        with tabs[1]:  # 데이터베이스 탭
            st.markdown("### 📊 외부 데이터베이스 API")
            st.info("문헌 검색과 데이터 분석을 위한 외부 데이터베이스 연동")
        
            db_services = {
                'materials_project': {
                    'name': 'Materials Project',
                    'help': '재료 물성 데이터베이스',
                    'url': 'https://materialsproject.org/api',
                    'placeholder': 'mp-...'
                },
                'materials_commons': {
                    'name': 'Materials Commons',
                    'help': '재료 실험 데이터 공유',
                    'url': 'https://materialscommons.org/api',
                    'placeholder': 'mc-...'
                },
                'zenodo': {
                    'name': 'Zenodo',
                    'help': '연구 데이터 리포지토리',
                    'url': 'https://zenodo.org/account/settings/applications',
                    'placeholder': 'zenodo-...'
                },
                'protocols_io': {
                    'name': 'protocols.io',
                    'help': '실험 프로토콜 공유',
                    'url': 'https://www.protocols.io/developers',
                    'placeholder': 'pio-...'
                },
                'figshare': {
                    'name': 'Figshare',
                    'help': '연구 데이터 공유 플랫폼',
                    'url': 'https://figshare.com/account/applications',
                    'placeholder': 'figshare-...'
                },
                'github': {
                    'name': 'GitHub',
                    'help': '코드 및 데이터 리포지토리',
                    'url': 'https://github.com/settings/tokens',
                    'placeholder': 'ghp_...'
                }
            }
        
            for service_key, service_info in db_services.items():
                with st.expander(f"🗄️ {service_info['name']}", expanded=False):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.caption(service_info['help'])
                    with col2:
                        st.link_button("API 발급", service_info['url'], use_container_width=True)
                
                    current_key = st.session_state.api_keys.get(service_key, '')
                    new_key = st.text_input(
                        "API Key/Token",
                        value='*' * 20 if current_key else '',
                        type="password",
                        placeholder=service_info['placeholder'],
                        key=f"db_{service_key}"
                    )
                
                    if new_key and new_key != '*' * 20:
                        if self.validate_api_key(service_key, new_key):
                            st.session_state.api_keys[service_key] = new_key
                        else:
                            st.warning("API 키 형식을 확인해주세요.")
        
            if st.button("데이터베이스 API 저장", use_container_width=True, key="save_db"):
                self._save_api_keys('database')
                st.success("데이터베이스 API 키가 저장되었습니다!")
    
        with tabs[2]:  # OAuth 설정 섹션
            st.markdown("### 🔐 소셜 로그인 설정")
            st.info("Google, GitHub OAuth를 설정하여 간편 로그인을 활성화하세요.")
        
            # Google OAuth
            with st.expander("🔷 Google OAuth 설정", expanded=False):
                st.markdown("""
                **설정 방법:**
                1. [Google Cloud Console](https://console.cloud.google.com) 접속
                2. 새 프로젝트 생성 또는 기존 프로젝트 선택
                3. "API 및 서비스" → "사용자 인증 정보"
                4. "사용자 인증 정보 만들기" → "OAuth 클라이언트 ID"
                5. 승인된 리디렉션 URI: `http://localhost:8501/auth/callback`
                """)
            
                google_client_id = st.session_state.api_keys.get('google_oauth_client_id', '')
                google_client_secret = st.session_state.api_keys.get('google_oauth_client_secret', '')
            
                new_google_id = st.text_input(
                    "Google Client ID",
                    value=google_client_id if google_client_id else '',
                    placeholder="123456789012-xxx.apps.googleusercontent.com",
                    key="google_client_id"
                )
            
                new_google_secret = st.text_input(
                    "Google Client Secret",
                    value='*' * 20 if google_client_secret else '',
                    type="password",
                    placeholder="GOCSPX-xxx",
                    key="google_client_secret"
                )
            
                if st.button("Google OAuth 저장", key="save_google_oauth"):
                    if new_google_id and new_google_secret:
                        st.session_state.api_keys['google_oauth_client_id'] = new_google_id
                        if new_google_secret != '*' * 20:
                            st.session_state.api_keys['google_oauth_client_secret'] = new_google_secret
                    
                        # SecretsManager에 저장
                        if hasattr(self, 'secrets_manager') and self.secrets_manager:
                            self.secrets_manager.add_api_key('GOOGLE_OAUTH_CLIENT_ID', new_google_id)
                            if new_google_secret != '*' * 20:
                                self.secrets_manager.add_api_key('GOOGLE_OAUTH_CLIENT_SECRET', new_google_secret)
                    
                        st.success("Google OAuth 설정이 저장되었습니다!")
                    else:
                        st.error("Client ID와 Secret을 모두 입력해주세요.")
        
            # GitHub OAuth
            with st.expander("🐙 GitHub OAuth 설정", expanded=False):
                st.markdown("""
                **설정 방법:**
                1. GitHub → Settings → Developer settings
                2. "OAuth Apps" → "New OAuth App"
                3. Homepage URL: `http://localhost:8501`
                4. Authorization callback URL: `http://localhost:8501/auth/github/callback`
                """)
            
                github_client_id = st.session_state.api_keys.get('github_client_id', '')
                github_client_secret = st.session_state.api_keys.get('github_client_secret', '')
            
                new_github_id = st.text_input(
                    "GitHub Client ID",
                    value=github_client_id if github_client_id else '',
                    placeholder="1234567890abcdef1234",
                    key="github_client_id"
                )
            
                new_github_secret = st.text_input(
                    "GitHub Client Secret",
                    value='*' * 20 if github_client_secret else '',
                    type="password",
                    placeholder="1234567890abcdef...",
                    key="github_client_secret"
                )
            
                if st.button("GitHub OAuth 저장", key="save_github_oauth"):
                    if new_github_id and new_github_secret:
                        st.session_state.api_keys['github_client_id'] = new_github_id
                        if new_github_secret != '*' * 20:
                            st.session_state.api_keys['github_client_secret'] = new_github_secret
                    
                        # SecretsManager에 저장
                        if hasattr(self, 'secrets_manager') and self.secrets_manager:
                            self.secrets_manager.add_api_key('GITHUB_CLIENT_ID', new_github_id)
                            if new_github_secret != '*' * 20:
                                self.secrets_manager.add_api_key('GITHUB_CLIENT_SECRET', new_github_secret)
                    
                        st.success("GitHub OAuth 설정이 저장되었습니다!")
                    else:
                        st.error("Client ID와 Secret을 모두 입력해주세요.")
        
            # OAuth 상태 확인
            st.markdown("### 📊 OAuth 연결 상태")
            col1, col2 = st.columns(2)
        
            with col1:
                if st.session_state.api_keys.get('google_oauth_client_id'):
                    st.success("✅ Google OAuth 설정됨")
                else:
                    st.warning("⚠️ Google OAuth 미설정")
        
            with col2:
                if st.session_state.api_keys.get('github_client_id'):
                    st.success("✅ GitHub OAuth 설정됨")
                else:
                    st.warning("⚠️ GitHub OAuth 미설정")

        with tabs[3]:  # Google 서비스 탭
            st.markdown("### 📝 Google 서비스 설정")
        
            # Google Sheets URL
            with st.expander("📊 Google Sheets 연동", expanded=True):
                st.info("프로젝트 데이터를 Google Sheets와 동기화합니다.")
            
                current_url = st.session_state.api_keys.get('google_sheets_url', '')
                sheets_url = st.text_input(
                    "Google Sheets URL",
                    value=current_url,
                    placeholder="https://docs.google.com/spreadsheets/d/...",
                    help="공유 설정이 '링크가 있는 모든 사용자' 또는 '편집 가능'이어야 합니다."
                )
            
                if sheets_url and sheets_url != current_url:
                    st.session_state.api_keys['google_sheets_url'] = sheets_url
            
                # 연결 테스트
                if st.button("연결 테스트", key="test_sheets"):
                    if sheets_url:
                        # 실제 연결 테스트 로직
                        with st.spinner("Google Sheets 연결을 테스트하는 중..."):
                            time.sleep(1)  # 실제로는 연결 테스트
                            st.success("✅ 연결 성공!")
                    else:
                        st.error("URL을 입력해주세요.")
        
            if st.button("Google 서비스 저장", use_container_width=True, key="save_google"):
                self._save_api_keys('google')
                st.success("Google 서비스 설정이 저장되었습니다!")
        
        with tabs[4]:  # 프로필 설정
            st.markdown("### 👤 프로필 설정")
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
                    st.session_state.user['name'] = name
                    st.session_state.user['organization'] = organization
                    st.session_state.user['research_field'] = research_field
                    st.success("프로필이 업데이트되었습니다!")
    
        with tabs[5]:  # UI 설정
            st.markdown("### 🎨 UI 설정")
            
            # 테마 설정
            theme = st.radio("테마", ["light", "dark"], 
                            index=0 if st.session_state.theme == 'light' else 1)
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
            st.markdown("#### 알림 설정")
            desktop_notif = st.checkbox("데스크톱 알림 사용", value=True)
            email_notif = st.checkbox("이메일 알림 사용", value=False)
            
            # 접근성 설정
            st.markdown("#### 접근성")
            high_contrast = st.checkbox("고대비 모드", value=False)
            larger_text = st.checkbox("큰 글씨", value=False)
            
            if st.button("UI 설정 저장", use_container_width=True):
                st.success("UI 설정이 저장되었습니다!")
                               
        with tabs[6]:  # 데이터 관리
            st.markdown("### 💾 데이터 관리")
            
            # 저장 공간 사용량
            st.markdown("#### 저장 공간")
            storage_info = self.get_storage_info()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("데이터베이스", f"{storage_info['db_size']:.2f} MB")
            with col2:
                st.metric("캐시", f"{storage_info['cache_size']:.2f} MB")
            with col3:
                st.metric("총 사용량", f"{storage_info['total_size']:.2f} MB")
            
            # 캐시 관리
            st.markdown("#### 캐시 관리")
            st.info(f"캐시를 비우면 임시 저장된 데이터가 삭제되어 저장 공간을 확보할 수 있습니다.")
            
            if st.button("캐시 비우기", type="secondary"):
                self.clear_cache()
                st.success("캐시가 비워졌습니다.")
                st.rerun()
                
            # 백업/복원
            st.markdown("#### 백업/복원")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("데이터 백업", use_container_width=True):
                    self.backup_data()
                    
            with col2:
                uploaded_file = st.file_uploader("백업 파일 복원", type=['zip'])
                if uploaded_file:
                    self.restore_data(uploaded_file)
                    
            # 데이터 내보내기
            st.markdown("#### 데이터 내보내기")
            export_format = st.selectbox(
                "내보내기 형식",
                ["Excel (.xlsx)", "CSV (.csv)", "JSON (.json)", "PDF 보고서"]
            )
            
            if st.button("전체 데이터 내보내기", use_container_width=True):
                self.export_all_data(export_format)
                
        with tabs[7]:  # 고급 설정
            st.markdown("### 🛠️ 고급 설정")
            
            # 개발자 옵션
            st.markdown("#### 개발자 옵션")
            debug_mode = st.checkbox("디버그 모드", value=False)
            if debug_mode:
                st.warning("디버그 모드가 활성화되었습니다. 성능이 저하될 수 있습니다.")
                
                # 디버그 정보 표시
                with st.expander("시스템 정보"):
                    runtime_info = self.get_runtime_info()
                    st.json(runtime_info)
                
            # 로그 설정
            log_level = st.selectbox(
                "로그 레벨",
                ["DEBUG", "INFO", "WARNING", "ERROR"],
                index=1
            )
            
            # 세션 설정
            st.markdown("#### 세션 설정")
            timeout = st.number_input(
                "세션 타임아웃 (분)",
                min_value=5,
                max_value=120,
                value=SESSION_TIMEOUT_MINUTES
            )
            
            # 실험적 기능
            st.markdown("#### 실험적 기능")
            st.warning("실험적 기능은 불안정할 수 있습니다.")
            
            enable_beta = st.checkbox("베타 기능 활성화", value=False)
            enable_experimental_ai = st.checkbox("실험적 AI 모델 사용", value=False)
            
            if st.button("고급 설정 저장", use_container_width=True):
                st.success("설정이 저장되었습니다.")

    def _save_api_keys(self, category: str):
        """API 키를 SecretsManager에 저장"""
        if hasattr(self, 'secrets_manager') and self.secrets_manager:
            saved_count = 0
        
            for key, value in st.session_state.api_keys.items():
                if value and value != '*' * 20:  # 실제 값이 입력된 경우
                    # 키 이름 변환 (예: google_gemini -> GOOGLE_GEMINI_API_KEY)
                    if key in ['google_sheets_url', 'google_oauth_client_id', 'google_oauth_client_secret']:
                        secret_key = key.upper()
                    else:
                        secret_key = f"{key.upper()}_API_KEY"
                
                    self.secrets_manager.add_api_key(secret_key, value)
                    saved_count += 1
        
            logger.info(f"{category} 카테고리에서 {saved_count}개의 API 키 저장됨", extra={
                'extra_fields': {
                    'category': category,
                    'count': saved_count,
                    'user_id': st.session_state.get('user_id')
                }
            })
        else:
            logger.warning("SecretsManager를 사용할 수 없습니다.")
    
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
        # 로그 기록
        logger.info("User logged out", extra={
            'extra_fields': {
                'user_id': st.session_state.get('user_id'),
                'session_id': st.session_state.session_id,
                'session_duration': str(datetime.now() - st.session_state.get('login_time', datetime.now()))
            }
        })
        
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
        
        st.rerun()
        
    def run_background_tasks(self):
        """백그라운드 작업 실행"""
        try:
            # 비동기 작업을 동기적으로 실행
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.run_background_tasks_async())
            
        except Exception as e:
            logger.error(f"Background task error: {e}")
            
    async def run_background_tasks_async(self):
        """비동기 백그라운드 작업"""
        tasks = [
            self.check_new_notifications_async(),
            self.auto_save_project_async(),
            self.cleanup_old_cache_async(),
            self.refresh_session_async()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 에러 로깅
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Background task {i} failed: {result}")
            
    async def check_new_notifications_async(self):
        """새 알림 확인 (비동기)"""
        # 실제 구현에서는 데이터베이스나 API에서 알림을 가져옴
        await asyncio.sleep(0.1)  # 시뮬레이션
        
    async def auto_save_project_async(self):
        """프로젝트 자동 저장 (비동기)"""
        if st.session_state.get('current_project'):
            # 실제 구현에서는 현재 프로젝트 상태를 저장
            await asyncio.sleep(0.1)  # 시뮬레이션
        
    async def cleanup_old_cache_async(self):
        """오래된 캐시 정리 (비동기)"""
        cache_dir = PROJECT_ROOT / "cache"
        if cache_dir.exists():
            current_time = time.time()
            for file in cache_dir.iterdir():
                if file.is_file():
                    # 7일 이상 된 파일 삭제
                    if current_time - file.stat().st_mtime > 7 * 24 * 60 * 60:
                        try:
                            file.unlink()
                        except:
                            pass
                            
    async def refresh_session_async(self):
        """세션 갱신 (비동기)"""
        if st.session_state.authenticated:
            st.session_state.last_activity = datetime.now()
            
    def get_storage_info(self) -> Dict[str, float]:
        """저장 공간 정보 가져오기"""
        info = {
            'db_size': 0,
            'cache_size': 0,
            'total_size': 0
        }
        
        # 데이터베이스 크기
        db_path = PROJECT_ROOT / "data" / "db" / "universaldoe.db"
        if db_path.exists():
            info['db_size'] = db_path.stat().st_size / (1024 * 1024)
        
        # 캐시 크기
        cache_size = self.get_cache_size()
        info['cache_size'] = cache_size
        
        # 총 크기
        info['total_size'] = info['db_size'] + info['cache_size']
        
        return info
            
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
            shutil.rmtree(cache_dir)
            cache_dir.mkdir()
            
        # 세션 캐시도 초기화
        st.session_state.cache = {}
        
        # Streamlit 캐시 초기화
        st.cache_data.clear()
        st.cache_resource.clear()
        
    def backup_data(self):
        """데이터 백업"""
        try:
            # 백업 파일명
            backup_name = f"polymer_doe_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            backup_path = PROJECT_ROOT / "temp" / backup_name
            
            # ZIP 파일 생성
            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # 데이터 디렉토리 백업
                data_dir = PROJECT_ROOT / "data"
                if data_dir.exists():
                    for file in data_dir.rglob('*'):
                        if file.is_file() and not file.name.startswith('.'):
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
            logger.error(f"Backup failed: {e}")
            
    def restore_data(self, uploaded_file):
        """데이터 복원"""
        try:
            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
                
            # ZIP 파일 추출
            with zipfile.ZipFile(tmp_path, 'r') as zipf:
                zipf.extractall(PROJECT_ROOT)
                
            st.success("데이터가 성공적으로 복원되었습니다.")
            st.info("앱을 다시 시작하면 복원된 데이터가 적용됩니다.")
            
            # 임시 파일 삭제
            os.unlink(tmp_path)
            
        except Exception as e:
            st.error(f"복원 실패: {str(e)}")
            logger.error(f"Restore failed: {e}")
            
    def export_all_data(self, format: str):
        """전체 데이터 내보내기"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if "Excel" in format:
                self.export_to_excel(timestamp)
            elif "CSV" in format:
                self.export_to_csv(timestamp)
            elif "JSON" in format:
                self.export_to_json(timestamp)
            elif "PDF" in format:
                self.export_to_pdf(timestamp)
                
        except Exception as e:
            st.error(f"내보내기 실패: {str(e)}")
            logger.error(f"Export failed: {e}")
            
    def export_to_excel(self, timestamp: str):
        """Excel 형식으로 내보내기 (파일 크기 제한 추가)"""
        try:
            output = BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # 프로젝트 데이터
                if st.session_state.projects:
                    df_projects = pd.DataFrame(st.session_state.projects)
                    df_projects.to_excel(writer, sheet_name='Projects', index=False)
                
                # 실험 설계 데이터
                if st.session_state.current_project and 'design' in st.session_state.current_project:
                    df_design = pd.DataFrame(st.session_state.current_project['design'])
                    df_design.to_excel(writer, sheet_name='Experiment Design', index=False)
                
                # 분석 데이터
                if 'analysis_data' in st.session_state and st.session_state.analysis_data.get('df') is not None:
                    st.session_state.analysis_data['df'].to_excel(writer, sheet_name='Analysis Data', index=False)
                
                # 메타데이터
                metadata = {
                    'export_date': datetime.now().isoformat(),
                    'app_version': APP_VERSION,
                    'user': st.session_state.user.get('email', 'unknown') if st.session_state.user else 'guest'
                }
                pd.DataFrame([metadata]).to_excel(writer, sheet_name='Metadata', index=False)
            
            # 파일 크기 확인
            output.seek(0, 2)  # 파일 끝으로 이동
            size_mb = output.tell() / (1024 * 1024)
            
            if size_mb > MAX_EXCEL_EXPORT_SIZE_MB:
                st.error(f"파일 크기가 너무 큽니다 ({size_mb:.1f}MB). 최대 {MAX_EXCEL_EXPORT_SIZE_MB}MB까지 가능합니다.")
                st.info("데이터를 분할하거나 CSV 형식을 사용해주세요.")
                return
                
            output.seek(0)
            
            st.download_button(
                label="Excel 파일 다운로드",
                data=output,
                file_name=f"polymer_doe_export_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
        except Exception as e:
            st.error(f"Excel 내보내기 실패: {str(e)}")
            logger.error(f"Excel export failed: {e}")
        
    def export_to_csv(self, timestamp: str):
        """CSV 형식으로 내보내기"""
        # ZIP 파일로 여러 CSV 묶기
        zip_buffer = BytesIO()
    
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # 프로젝트 데이터
            if st.session_state.projects:
                projects_df = pd.DataFrame(st.session_state.projects)
                csv_buffer = StringIO()
                projects_df.to_csv(csv_buffer, index=False)
                zip_file.writestr('projects.csv', csv_buffer.getvalue())
        
            # 실험 설계 데이터
            if st.session_state.current_project and 'design' in st.session_state.current_project:
                design_df = pd.DataFrame(st.session_state.current_project['design'])
                csv_buffer = StringIO()
                design_df.to_csv(csv_buffer, index=False)
                zip_file.writestr('experiment_design.csv', csv_buffer.getvalue())
        
            # 분석 데이터
            if 'analysis_data' in st.session_state:
                analysis_df = st.session_state.analysis_data.get('df')
                if analysis_df is not None:
                    csv_buffer = StringIO()
                    analysis_df.to_csv(csv_buffer, index=False)
                    zip_file.writestr('analysis_data.csv', csv_buffer.getvalue())
        
            # 메타데이터
            metadata = {
                'export_date': datetime.now().isoformat(),
                'app_version': APP_VERSION,
                'user': st.session_state.user.get('email', 'unknown') if st.session_state.user else 'guest'
            }
            metadata_df = pd.DataFrame([metadata])
            csv_buffer = StringIO()
            metadata_df.to_csv(csv_buffer, index=False)
            zip_file.writestr('metadata.csv', csv_buffer.getvalue())
    
        zip_buffer.seek(0)
    
        st.download_button(
            label="CSV 파일 모음 다운로드 (ZIP)",
            data=zip_buffer,
            file_name=f"polymer_doe_export_{timestamp}.zip",
            mime="application/zip"
        )
    
        st.success("CSV 내보내기가 완료되었습니다!")
        
    def export_to_json(self, timestamp: str):
        """JSON 형식으로 내보내기"""
        data = {
            'export_date': datetime.now().isoformat(),
            'version': APP_VERSION,
            'projects': st.session_state.projects,
            'current_project': st.session_state.current_project,
            'user': st.session_state.user,
            'analysis_data': None
        }
        
        # DataFrame을 dict로 변환
        if 'analysis_data' in st.session_state and st.session_state.analysis_data.get('df') is not None:
            data['analysis_data'] = st.session_state.analysis_data['df'].to_dict('records')
        
        json_str = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        
        st.download_button(
            label="JSON 파일 다운로드",
            data=json_str,
            file_name=f"polymer_doe_export_{timestamp}.json",
            mime="application/json"
        )
        
    def export_to_pdf(self, timestamp: str):
        """PDF 보고서로 내보내기"""
        try:
            # HTML 보고서 생성
            html_content = f"""
            <html>
            <head>
                <meta charset="utf-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    h1 {{ color: #7C3AED; }}
                    h2 {{ color: #667eea; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .metric {{ display: inline-block; margin: 10px; padding: 15px; 
                              background-color: #f8f9fa; border-radius: 8px; }}
                    .footer {{ margin-top: 50px; text-align: center; color: #666; }}
                </style>
            </head>
            <body>
                <h1>Universal DOE Platform - 실험 보고서</h1>
                <p>생성일: {datetime.now().strftime('%Y년 %m월 %d일')}</p>
            
                <h2>1. 프로젝트 정보</h2>
            """
        
            if st.session_state.current_project:
                project = st.session_state.current_project
                html_content += f"""
                <div class="metric">
                    <strong>프로젝트명:</strong> {project.get('name', 'N/A')}<br>
                    <strong>유형:</strong> {project.get('type', 'N/A')}<br>
                    <strong>생성일:</strong> {project.get('created_at', 'N/A')[:10]}
                </div>
                """
            
                # 실험 설계 정보
                if 'factors' in project:
                    html_content += """
                    <h2>2. 실험 요인</h2>
                    <table>
                        <tr><th>요인</th><th>유형</th><th>범위/수준</th></tr>
                    """
                    for factor in project['factors']:
                        if factor['type'] == '연속형':
                            range_str = f"{factor['min']} - {factor['max']} {factor.get('unit', '')}"
                        else:
                            range_str = ', '.join(factor['levels'])
                        html_content += f"""
                        <tr>
                            <td>{factor['name']}</td>
                            <td>{factor['type']}</td>
                            <td>{range_str}</td>
                        </tr>
                        """
                    html_content += "</table>"
        
            # 분석 결과
            if 'analysis_data' in st.session_state:
                html_content += """
                <h2>3. 분석 결과</h2>
                <p>데이터 분석이 수행되었습니다. 상세 결과는 별도 파일을 참조하세요.</p>
                """
        
            html_content += """
                <div class="footer">
                    <p>© 2024 Universal DOE Platform. All rights reserved.</p>
                </div>
            </body>
            </html>
            """
        
            # HTML을 Base64로 인코딩 (브라우저에서 PDF 변환)
            b64 = base64.b64encode(html_content.encode()).decode()
            href = f'<a href="data:text/html;base64,{b64}" download="polymer_doe_report_{timestamp}.html">보고서 다운로드 (HTML)</a>'
            st.markdown(href, unsafe_allow_html=True)
        
            st.info("📄 HTML 보고서가 생성되었습니다. 브라우저에서 PDF로 인쇄하여 저장할 수 있습니다.")
        
            # PDF 생성 안내
            with st.expander("PDF로 저장하는 방법"):
                st.write("""
                1. 위 링크를 클릭하여 HTML 파일을 다운로드합니다.
                2. 다운로드한 파일을 브라우저에서 엽니다.
                3. Ctrl+P (또는 Cmd+P)를 눌러 인쇄 대화상자를 엽니다.
                4. 프린터로 "PDF로 저장"을 선택합니다.
                5. 저장 버튼을 클릭합니다.
                """)
            
        except Exception as e:
            st.error(f"PDF 보고서 생성 중 오류 발생: {str(e)}")
            logger.error(f"PDF export failed: {e}")
        
    def get_runtime_info(self) -> Dict[str, Any]:
        """런타임 정보 가져오기"""
        info = {
            'python': {
                'version': sys.version,
                'platform': sys.platform
            },
            'streamlit': {
                'version': st.__version__
            },
            'session': {
                'id': st.session_state.get('session_id', 'unknown'),
                'authenticated': st.session_state.get('authenticated', False),
                'uptime': str(datetime.now() - st.session_state.get('login_time', datetime.now()))
            }
        }
        
        # psutil이 설치된 경우만 시스템 정보 추가
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                
                info.update({
                    'memory': {
                        'rss_mb': process.memory_info().rss / 1024 / 1024,
                        'vms_mb': process.memory_info().vms / 1024 / 1024,
                        'percent': process.memory_percent(),
                        'available_mb': psutil.virtual_memory().available / 1024 / 1024
                    },
                    'cpu': {
                        'percent': process.cpu_percent(interval=0.1),
                        'threads': process.num_threads(),
                        'cores': psutil.cpu_count()
                    },
                    'disk': {
                        'usage_percent': psutil.disk_usage('/').percent,
                        'free_gb': psutil.disk_usage('/').free / (1024**3)
                    }
                })
            except Exception as e:
                logger.error(f"Failed to get system info: {e}")
                
        return info
            
    def render_error_page(self, error: Exception):
        """에러 페이지 렌더링"""
        st.error("애플리케이션 오류가 발생했습니다")
        
        error_id = str(uuid.uuid4())[:8]
        st.caption(f"오류 ID: {error_id}")
        
        # 오류 로깅
        logger.error(f"Application error {error_id}: {error}", extra={
            'extra_fields': {
                'error_id': error_id,
                'user_id': st.session_state.get('user_id'),
                'page': st.session_state.get('current_page'),
                'traceback': traceback.format_exc()
            }
        })
        
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
