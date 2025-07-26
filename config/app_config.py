"""
⚙️ Universal DOE Platform - 앱 전역 설정
================================================================================
모든 설정을 중앙에서 관리하는 핵심 설정 파일
환경별 설정, AI 엔진, 보안, 기능 플래그 등 통합 관리
================================================================================
"""

import os
import sys
from pathlib import Path
from datetime import timedelta
from typing import Dict, Any, Optional, List, Tuple
import platform
import json

# ============================================================================
# 🌍 환경 설정
# ============================================================================

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent

# 데이터 디렉토리
DATA_DIR = PROJECT_ROOT / 'data'
LOGS_DIR = DATA_DIR / 'logs'
TEMP_DIR = DATA_DIR / 'temp'
CACHE_DIR = DATA_DIR / 'cache'
DB_DIR = DATA_DIR / 'db'
MODULES_DIR = DATA_DIR / 'modules'
BACKUP_DIR = DATA_DIR / 'backups'

# 환경 변수
ENV = os.getenv('STREAMLIT_ENV', 'development')
IS_PRODUCTION = ENV == 'production'
IS_STAGING = ENV == 'staging'
IS_DEVELOPMENT = ENV == 'development'
IS_TEST = ENV == 'test'

# 디버그 모드
DEBUG = os.getenv('DEBUG', str(not IS_PRODUCTION)).lower() in ('true', '1', 'yes')

# 빌드 환경 감지
IS_FROZEN = getattr(sys, 'frozen', False)
IS_DESKTOP = IS_FROZEN or os.getenv('DESKTOP_MODE', 'false').lower() == 'true'

# 시스템 정보
SYSTEM_INFO = {
    'platform': platform.system(),
    'platform_version': platform.version(),
    'python_version': sys.version,
    'is_windows': platform.system() == 'Windows',
    'is_macos': platform.system() == 'Darwin',
    'is_linux': platform.system() == 'Linux'
}

# ============================================================================
# 📱 앱 기본 정보
# ============================================================================

APP_INFO = {
    'name': 'Universal DOE Platform',
    'short_name': 'UniversalDOE',
    'version': '2.0.0',
    'description': '모든 연구자를 위한 AI 기반 실험 설계 플랫폼',
    'author': 'DOE Team',
    'email': 'support@universaldoe.com',
    'website': 'https://universaldoe.com',
    'github': 'https://github.com/universaldoe/platform',
    'license': 'MIT',
    'copyright': '© 2024 DOE Team. All rights reserved.',
    'build_date': '2024-12-01',
    'python_required': '>=3.8'
}

# ============================================================================
# 🤖 AI 엔진 설정 (6개 통합)
# ============================================================================

AI_ENGINES = {
    'google_gemini': {
        'name': 'Google Gemini 2.0 Flash',
        'provider': 'Google',
        'model': 'gemini-2.0-flash-exp',
        'api_key_env': 'GOOGLE_GEMINI_API_KEY',
        'api_key_secret': 'google_gemini_key',
        'required': True,
        'free_tier': True,
        'rate_limit': 60,  # requests per minute
        'max_tokens': 8192,
        'temperature': 0.7,
        'capabilities': ['text', 'code', 'analysis', 'vision'],
        'best_for': ['general', 'multimodal', 'reasoning'],
        'docs_url': 'https://makersuite.google.com/app/apikey'
    },
    'xai_grok': {
        'name': 'xAI Grok',
        'provider': 'xAI',
        'model': 'grok-beta',
        'api_key_env': 'XAI_API_KEY',
        'api_key_secret': 'xai_api_key',
        'base_url': 'https://api.x.ai/v1',
        'required': False,
        'free_tier': False,
        'rate_limit': 30,
        'max_tokens': 4096,
        'temperature': 0.7,
        'capabilities': ['text', 'code', 'humor'],
        'best_for': ['creative', 'unconventional'],
        'docs_url': 'https://x.ai/api'
    },
    'groq': {
        'name': 'Groq LPU',
        'provider': 'Groq',
        'model': 'mixtral-8x7b-32768',
        'api_key_env': 'GROQ_API_KEY',
        'api_key_secret': 'groq_api_key',
        'base_url': 'https://api.groq.com/openai/v1',
        'required': False,
        'free_tier': True,
        'rate_limit': 100,
        'max_tokens': 32768,
        'temperature': 0.7,
        'capabilities': ['text', 'code', 'speed'],
        'best_for': ['fast_inference', 'real_time'],
        'docs_url': 'https://console.groq.com'
    },
    'deepseek': {
        'name': 'DeepSeek Coder',
        'provider': 'DeepSeek',
        'model': 'deepseek-coder',
        'api_key_env': 'DEEPSEEK_API_KEY',
        'api_key_secret': 'deepseek_api_key',
        'base_url': 'https://api.deepseek.com/v1',
        'required': False,
        'free_tier': False,
        'rate_limit': 60,
        'max_tokens': 16384,
        'temperature': 0.3,
        'capabilities': ['code', 'math', 'technical'],
        'best_for': ['code_generation', 'algorithms', 'formulas'],
        'docs_url': 'https://platform.deepseek.com'
    },
    'sambanova': {
        'name': 'SambaNova',
        'provider': 'SambaNova',
        'model': 'llama-3.1-405b',
        'api_key_env': 'SAMBANOVA_API_KEY',
        'api_key_secret': 'sambanova_api_key',
        'base_url': 'https://api.sambanova.ai/v1',
        'required': False,
        'free_tier': True,
        'rate_limit': 10,
        'max_tokens': 4096,
        'temperature': 0.7,
        'capabilities': ['text', 'reasoning', 'large_context'],
        'best_for': ['complex_analysis', 'research'],
        'docs_url': 'https://cloud.sambanova.ai'
    },
    'huggingface': {
        'name': 'HuggingFace Hub',
        'provider': 'HuggingFace',
        'models': {
            'general': 'meta-llama/Llama-3.2-11B-Vision-Instruct',
            'chemistry': 'laituan245/ChemLLM-7B-Chat',
            'materials': 'm3rg-iitd/MatSciBERT',
            'bio': 'microsoft/BioGPT-Large'
        },
        'api_key_env': 'HUGGINGFACE_TOKEN',
        'api_key_secret': 'huggingface_token',
        'required': False,
        'free_tier': True,
        'rate_limit': 100,
        'max_tokens': 2048,
        'temperature': 0.7,
        'capabilities': ['specialized', 'domain_specific'],
        'best_for': ['chemistry', 'materials', 'biology'],
        'docs_url': 'https://huggingface.co/settings/tokens'
    }
}

# ============================================================================
# 🧠 AI 설명 상세도 제어 (필수 구현)
# ============================================================================

AI_EXPLANATION_CONFIG = {
    'modes': {
        'auto': '사용자 레벨에 따라 자동 조정',
        'always_detailed': '항상 상세 설명 표시',
        'always_simple': '항상 간단한 설명만',
        'custom': '사용자 맞춤 설정'
    },
    'default_mode': 'auto',
    'components': {
        'reasoning': {
            'label': '추론 과정',
            'default': True,
            'description': 'AI가 왜 이런 결론에 도달했는지'
        },
        'alternatives': {
            'label': '대안 검토',
            'default': True,
            'description': '다른 가능한 옵션들과 장단점'
        },
        'theory': {
            'label': '이론적 배경',
            'default': True,
            'description': '과학적 원리와 근거'
        },
        'confidence': {
            'label': '신뢰도',
            'default': True,
            'description': '추천의 확실성 정도와 한계점'
        },
        'limitations': {
            'label': '주의사항',
            'default': True,
            'description': '제약사항과 주의해야 할 점'
        }
    },
    'toggle_shortcut': 'Ctrl+D',
    'session_persistent': True,
    'ui_position': 'top_right',  # 토글 버튼 위치
    'animation': True,
    'auto_level_detection': {
        'beginner_indicators': ['처음', '초보', '기초', '쉽게'],
        'expert_indicators': ['전문', '고급', '상세', '깊이']
    }
}

# ============================================================================
# 📂 파일 처리 설정 (다중 형식 지원)
# ============================================================================

FILE_CONFIG = {
    'upload': {
        'max_size_mb': 50,
        'max_files': 10,
        'chunk_size': 1024 * 1024,  # 1MB chunks
        'timeout': 300  # 5분
    },
    'supported_formats': {
        'documents': {
            'pdf': {'mime': 'application/pdf', 'priority': 1},
            'docx': {'mime': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'priority': 2},
            'doc': {'mime': 'application/msword', 'priority': 3},
            'txt': {'mime': 'text/plain', 'priority': 4},
            'rtf': {'mime': 'application/rtf', 'priority': 5},
            'odt': {'mime': 'application/vnd.oasis.opendocument.text', 'priority': 6}
        },
        'markup': {
            'html': {'mime': 'text/html', 'priority': 1},
            'htm': {'mime': 'text/html', 'priority': 1},
            'md': {'mime': 'text/markdown', 'priority': 2},
            'xml': {'mime': 'application/xml', 'priority': 3}
        },
        'data': {
            'csv': {'mime': 'text/csv', 'priority': 1},
            'xlsx': {'mime': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'priority': 2},
            'xls': {'mime': 'application/vnd.ms-excel', 'priority': 3},
            'json': {'mime': 'application/json', 'priority': 4},
            'parquet': {'mime': 'application/octet-stream', 'priority': 5}
        },
        'images': {
            'png': {'mime': 'image/png', 'priority': 1},
            'jpg': {'mime': 'image/jpeg', 'priority': 2},
            'jpeg': {'mime': 'image/jpeg', 'priority': 2},
            'gif': {'mime': 'image/gif', 'priority': 3},
            'webp': {'mime': 'image/webp', 'priority': 4},
            'svg': {'mime': 'image/svg+xml', 'priority': 5}
        }
    },
    'encoding': {
        'auto_detect': True,
        'fallback': 'utf-8',
        'supported': ['utf-8', 'utf-16', 'latin-1', 'cp1252', 'gb2312', 'shift-jis', 'euc-kr'],
        'bom_handling': 'remove'
    },
    'processing': {
        'parallel': True,
        'max_workers': 4,
        'batch_size': 5,
        'memory_limit_mb': 500
    }
}

# ============================================================================
# 🔍 프로토콜 추출 설정
# ============================================================================

PROTOCOL_EXTRACTION_CONFIG = {
    'methods': {
        'rule_based': {
            'enabled': True,
            'patterns': ['methods', 'experimental', 'procedure', 'protocol'],
            'confidence_weight': 0.3
        },
        'ml_based': {
            'enabled': True,
            'model': 'spacy',
            'confidence_weight': 0.4
        },
        'ai_based': {
            'enabled': True,
            'engine': 'google_gemini',
            'confidence_weight': 0.3
        }
    },
    'extraction': {
        'min_confidence': 0.7,
        'max_length': 500000,  # 문자 수
        'timeout': 60,  # 초
        'languages': ['en', 'ko', 'zh', 'ja'],
        'cache_results': True,
        'cache_ttl': timedelta(days=30)
    },
    'output_formats': {
        'json': {'structured': True, 'schema_version': '2.0'},
        'yaml': {'human_readable': True},
        'csv': {'tabular': True},
        'template': {'fillable': True}
    },
    'ocr': {
        'enabled': True,
        'engine': 'tesseract',
        'languages': ['eng', 'kor', 'chi_sim', 'jpn'],
        'dpi': 300,
        'preprocessing': ['deskew', 'denoise', 'contrast']
    }
}

# ============================================================================
# 💾 데이터베이스 설정
# ============================================================================

# SQLite (로컬/데스크톱)
SQLITE_CONFIG = {
    'database_path': DB_DIR / 'app.db',
    'backup_path': BACKUP_DIR / 'db',
    'connection': {
        'check_same_thread': False,
        'timeout': 30,
        'isolation_level': 'DEFERRED',
        'journal_mode': 'WAL'  # Write-Ahead Logging
    },
    'pool': {
        'size': 5,
        'max_overflow': 10,
        'timeout': 30,
        'recycle': 3600
    },
    'backup': {
        'enabled': True,
        'interval': timedelta(hours=6),
        'keep_last': 5,
        'compress': True
    }
}

# Google Sheets (온라인/협업)
GOOGLE_SHEETS_CONFIG = {
    'enabled': not IS_DESKTOP or os.getenv('ENABLE_CLOUD_SYNC', 'false').lower() == 'true',
    'scope': [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive.file'
    ],
    'credentials': {
        'type': 'service_account',  # or 'oauth2'
        'file': PROJECT_ROOT / '.streamlit' / 'google_credentials.json',
        'env_var': 'GOOGLE_APPLICATION_CREDENTIALS'
    },
    'spreadsheets': {
        'users': {
            'name': 'UniversalDOE_Users',
            'key_env': 'GOOGLE_SHEETS_USERS_KEY',
            'key_secret': 'google_sheets_users_key'
        },
        'projects': {
            'name': 'UniversalDOE_Projects',
            'key_env': 'GOOGLE_SHEETS_PROJECTS_KEY',
            'key_secret': 'google_sheets_projects_key'
        },
        'shared': {
            'name': 'UniversalDOE_SharedData',
            'key_env': 'GOOGLE_SHEETS_SHARED_KEY',
            'key_secret': 'google_sheets_shared_key'
        }
    },
    'sync': {
        'interval': timedelta(minutes=5),
        'batch_size': 100,
        'conflict_resolution': 'latest_wins'
    }
}

# ============================================================================
# 🔒 보안 설정
# ============================================================================

SECURITY_CONFIG = {
    'password': {
        'min_length': 8,
        'require_uppercase': True,
        'require_lowercase': True,
        'require_numbers': True,
        'require_special': True,
        'special_chars': '!@#$%^&*()_+-=[]{}|;:,.<>?',
        'bcrypt_rounds': 12,
        'history_check': 3  # 최근 3개 비밀번호 재사용 금지
    },
    'session': {
        'timeout': timedelta(hours=24),
        'remember_me_duration': timedelta(days=30),
        'max_concurrent': 3,
        'secret_key': os.getenv('SESSION_SECRET_KEY', 'dev-secret-key-change-in-production'),
        'cookie_name': 'universaldoe_session',
        'secure': IS_PRODUCTION,
        'httponly': True,
        'samesite': 'Lax'
    },
    'jwt': {
        'secret_key': os.getenv('JWT_SECRET_KEY', 'dev-jwt-secret'),
        'algorithm': 'HS256',
        'access_token_expire': timedelta(hours=1),
        'refresh_token_expire': timedelta(days=7)
    },
    'auth': {
        'max_login_attempts': 5,
        'lockout_duration': timedelta(minutes=30),
        'enable_2fa': False,  # 향후 구현
        'oauth_providers': ['google'],  # 향후 확장
        'api_key_length': 32
    },
    'encryption': {
        'algorithm': 'AES-256-GCM',
        'key_derivation': 'PBKDF2',
        'iterations': 100000
    },
    'api': {
        'rate_limit': {
            'enabled': True,
            'requests_per_minute': 60,
            'requests_per_hour': 1000
        },
        'cors': {
            'enabled': False,
            'origins': ['http://localhost:8501']
        }
    }
}

# ============================================================================
# 🎨 UI/UX 설정
# ============================================================================

UI_CONFIG = {
    'theme': {
        'default': 'light',
        'allow_switching': True,
        'auto_detect': True
    },
    'layout': {
        'sidebar_default': 'expanded',
        'max_width': 1200,
        'padding': 20,
        'animation_speed': 300  # ms
    },
    'components': {
        'data_editor': {
            'num_rows': 'dynamic',
            'column_config': 'auto',
            'hide_index': True
        },
        'charts': {
            'default_renderer': 'plotly',
            'interactive': True,
            'export_formats': ['png', 'svg', 'pdf']
        },
        'tables': {
            'pagination': True,
            'page_size': 25,
            'sortable': True,
            'filterable': True
        }
    },
    'accessibility': {
        'high_contrast': False,
        'keyboard_navigation': True,
        'screen_reader_support': True,
        'font_size_adjustment': True
    },
    'notifications': {
        'position': 'top-right',
        'duration': 5000,  # ms
        'max_stack': 3
    }
}

# ============================================================================
# 🧪 실험 설계 설정
# ============================================================================

EXPERIMENT_CONFIG = {
    'design_types': {
        'factorial': {
            'name': '완전/부분 요인설계',
            'min_factors': 2,
            'max_factors': 10,
            'levels': [2, 3, 4, 5]
        },
        'response_surface': {
            'name': '반응표면설계',
            'types': ['ccd', 'box-behnken', 'face-centered'],
            'min_factors': 2,
            'max_factors': 6
        },
        'mixture': {
            'name': '혼합물 설계',
            'types': ['simplex-lattice', 'simplex-centroid', 'extreme-vertices'],
            'min_components': 3,
            'max_components': 10
        },
        'optimal': {
            'name': '최적 설계',
            'criteria': ['D-optimal', 'I-optimal', 'A-optimal', 'G-optimal'],
            'custom_allowed': True
        },
        'screening': {
            'name': '스크리닝 설계',
            'types': ['plackett-burman', 'definitive-screening'],
            'min_factors': 3
        }
    },
    'constraints': {
        'min_runs': 3,
        'max_runs': 1000,
        'max_center_points': 20,
        'replication_allowed': True,
        'blocking_allowed': True
    },
    'analysis': {
        'confidence_level': 0.95,
        'power': 0.8,
        'alpha': 0.05,
        'multiple_comparison': 'bonferroni'
    },
    'optimization': {
        'methods': ['desirability', 'pareto', 'genetic_algorithm'],
        'multi_objective': True,
        'constraints_allowed': True
    }
}

# ============================================================================
# 📦 모듈 시스템 설정
# ============================================================================

MODULE_CONFIG = {
    'paths': {
        'core_modules': PROJECT_ROOT / 'modules' / 'core',
        'user_modules': MODULES_DIR / 'user_modules',
        'marketplace': MODULES_DIR / 'marketplace',
        'templates': PROJECT_ROOT / 'modules' / 'templates'
    },
    'validation': {
        'required_methods': ['get_info', 'validate', 'generate_design', 'analyze'],
        'max_size_mb': 10,
        'sandbox_execution': True,
        'timeout': 30
    },
    'marketplace': {
        'enabled': True,
        'api_endpoint': 'https://api.universaldoe.com/modules',
        'cache_duration': timedelta(hours=24),
        'featured_refresh': timedelta(hours=6)
    },
    'development': {
        'hot_reload': IS_DEVELOPMENT,
        'debug_mode': DEBUG,
        'template_available': True
    }
}

# ============================================================================
# 🔄 동기화 설정
# ============================================================================

SYNC_CONFIG = {
    'enabled': not IS_DESKTOP,
    'mode': 'manual',  # 'manual', 'auto', 'scheduled'
    'providers': {
        'google_drive': {
            'enabled': True,
            'folder': 'UniversalDOE_Sync',
            'file_types': ['projects', 'data', 'reports']
        },
        'dropbox': {
            'enabled': False,
            'folder': 'UniversalDOE'
        },
        'onedrive': {
            'enabled': False,
            'folder': 'UniversalDOE'
        }
    },
    'conflict_resolution': {
        'strategy': 'latest_wins',  # 'latest_wins', 'manual', 'merge'
        'backup_conflicts': True
    },
    'scheduling': {
        'interval': timedelta(hours=1),
        'wifi_only': True,
        'battery_threshold': 20  # %
    }
}

# ============================================================================
# 🔄 업데이트 설정
# ============================================================================

UPDATE_CONFIG = {
    'enabled': True,
    'check_on_startup': True,
    'check_interval': timedelta(days=1),
    'auto_download': False,
    'auto_install': False,
    'channels': {
        'stable': 'https://api.universaldoe.com/updates/stable',
        'beta': 'https://api.universaldoe.com/updates/beta',
        'nightly': 'https://api.universaldoe.com/updates/nightly'
    },
    'current_channel': 'stable' if IS_PRODUCTION else 'beta',
    'show_changelog': True,
    'backup_before_update': True
}

# ============================================================================
# 🌐 지역화 설정
# ============================================================================

LOCALIZATION_CONFIG = {
    'default_language': 'ko_KR',
    'supported_languages': {
        'ko_KR': {'name': '한국어', 'flag': '🇰🇷', 'rtl': False},
        'en_US': {'name': 'English', 'flag': '🇺🇸', 'rtl': False},
        'zh_CN': {'name': '简体中文', 'flag': '🇨🇳', 'rtl': False},
        'ja_JP': {'name': '日本語', 'flag': '🇯🇵', 'rtl': False}
    },
    'auto_detect': True,
    'fallback_language': 'en_US',
    'date_format': {
        'ko_KR': 'YYYY년 MM월 DD일',
        'en_US': 'MM/DD/YYYY',
        'zh_CN': 'YYYY年MM月DD日',
        'ja_JP': 'YYYY年MM月DD日'
    },
    'number_format': {
        'decimal_separator': '.',
        'thousands_separator': ',',
        'decimal_places': 2
    },
    'currency': {
        'ko_KR': 'KRW',
        'en_US': 'USD',
        'zh_CN': 'CNY',
        'ja_JP': 'JPY'
    }
}

# ============================================================================
# 📊 성능 설정
# ============================================================================

PERFORMANCE_CONFIG = {
    'caching': {
        'enabled': True,
        'ttl': {
            'user_data': timedelta(minutes=5),
            'project_data': timedelta(minutes=10),
            'ai_responses': timedelta(hours=1),
            'analysis_results': timedelta(hours=24),
            'file_previews': timedelta(hours=12)
        },
        'max_size_mb': 500,
        'eviction_policy': 'lru'
    },
    'threading': {
        'max_workers': 8,
        'queue_size': 100,
        'timeout': 30
    },
    'resource_limits': {
        'max_memory_mb': 2048,
        'max_cpu_percent': 80,
        'max_file_handles': 1000
    },
    'optimization': {
        'lazy_loading': True,
        'compression': True,
        'minification': IS_PRODUCTION
    }
}

# ============================================================================
# 🛠️ 개발자 설정
# ============================================================================

DEVELOPER_CONFIG = {
    'debug': {
        'enabled': DEBUG,
        'verbose': IS_DEVELOPMENT,
        'show_errors': not IS_PRODUCTION,
        'profiling': False,
        'timing': True
    },
    'logging': {
        'level': 'DEBUG' if DEBUG else 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file': LOGS_DIR / f'app_{ENV}.log',
        'max_size_mb': 10,
        'backup_count': 5,
        'console': True
    },
    'testing': {
        'mock_data': IS_DEVELOPMENT,
        'test_accounts': IS_DEVELOPMENT,
        'bypass_auth': False,
        'fixtures_dir': PROJECT_ROOT / 'tests' / 'fixtures'
    }
}

# ============================================================================
# 🎯 기능 플래그
# ============================================================================

FEATURE_FLAGS = {
    # 핵심 기능
    'ai_chat': True,
    'multi_ai_engines': True,
    'protocol_extraction': True,
    'collaboration': True,
    'marketplace': True,
    'advanced_analytics': True,
    'custom_modules': True,
    
    # 실험적 기능
    'experimental': {
        'voice_commands': False,
        'ar_visualization': False,
        'ai_autopilot': False,
        'blockchain_verification': False
    },
    
    # 베타 기능
    'beta': {
        'new_ui': IS_DEVELOPMENT or IS_STAGING,
        'cloud_compute': False,
        'mobile_sync': False,
        'api_v2': IS_DEVELOPMENT
    },
    
    # 플랫폼별 기능
    'platform': {
        'desktop': {
            'system_tray': True,
            'auto_update': True,
            'offline_mode': True
        },
        'web': {
            'pwa': True,
            'push_notifications': False,
            'webgl': True
        }
    }
}

# ============================================================================
# 🔧 유틸리티 함수
# ============================================================================

def get_config(key: str, default: Any = None) -> Any:
    """
    점 표기법으로 중첩된 설정값 가져오기
    
    Args:
        key: 설정 키 (예: 'ai_engines.google_gemini.model')
        default: 기본값
        
    Returns:
        설정값 또는 기본값
    """
    # 환경변수 우선 확인
    env_key = f"DOE_{key.upper().replace('.', '_')}"
    env_value = os.getenv(env_key)
    
    if env_value is not None:
        # 타입 변환
        if isinstance(default, bool):
            return env_value.lower() in ('true', '1', 'yes', 'on')
        elif isinstance(default, int):
            try:
                return int(env_value)
            except ValueError:
                pass
        elif isinstance(default, float):
            try:
                return float(env_value)
            except ValueError:
                pass
        return env_value
    
    # 설정 딕셔너리에서 찾기
    keys = key.split('.')
    config = globals()
    
    for k in keys:
        if isinstance(config, dict) and k in config:
            config = config[k]
        else:
            return default
    
    return config

def validate_config() -> List[Tuple[str, str]]:
    """
    설정 검증 및 경고 메시지 반환
    
    Returns:
        [(레벨, 메시지), ...] 형태의 경고 리스트
    """
    warnings = []
    
    # 필수 디렉토리 생성
    for dir_path in [DATA_DIR, LOGS_DIR, TEMP_DIR, CACHE_DIR, DB_DIR, MODULES_DIR, BACKUP_DIR]:
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            warnings.append(('ERROR', f'{dir_path} 디렉토리 생성 실패: {e}'))
    
    # 보안 키 확인
    if IS_PRODUCTION:
        if SECURITY_CONFIG['session']['secret_key'] == 'dev-secret-key-change-in-production':
            warnings.append(('CRITICAL', '프로덕션 환경에서 기본 세션 키 사용 중!'))
        if SECURITY_CONFIG['jwt']['secret_key'] == 'dev-jwt-secret':
            warnings.append(('CRITICAL', '프로덕션 환경에서 기본 JWT 키 사용 중!'))
    
    # 필수 AI 엔진 확인
    required_engines = [k for k, v in AI_ENGINES.items() if v.get('required')]
    if not required_engines:
        warnings.append(('WARNING', '필수 AI 엔진이 설정되지 않음'))
    
    # Python 버전 확인
    current_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    required_version = APP_INFO['python_required'].replace('>=', '')
    if current_version < required_version:
        warnings.append(('ERROR', f'Python {required_version} 이상 필요 (현재: {current_version})'))
    
    return warnings

def save_config_snapshot(filename: Optional[str] = None) -> Path:
    """
    현재 설정을 JSON 파일로 저장
    
    Args:
        filename: 파일명 (기본값: config_snapshot_TIMESTAMP.json)
        
    Returns:
        저장된 파일 경로
    """
    from datetime import datetime
    
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'config_snapshot_{timestamp}.json'
    
    snapshot_path = BACKUP_DIR / 'configs' / filename
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 민감한 정보 제외하고 설정 수집
    config_data = {
        'timestamp': datetime.now().isoformat(),
        'version': APP_INFO['version'],
        'environment': ENV,
        'system': SYSTEM_INFO,
        'settings': {
            'app_info': APP_INFO,
            'ai_engines': {k: {sk: sv for sk, sv in v.items() if 'key' not in sk} 
                         for k, v in AI_ENGINES.items()},
            'file_config': FILE_CONFIG,
            'ui_config': UI_CONFIG,
            'experiment_config': EXPERIMENT_CONFIG,
            'feature_flags': FEATURE_FLAGS
        }
    }
    
    with open(snapshot_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False, default=str)
    
    return snapshot_path

# ============================================================================
# 🚀 초기화 검증
# ============================================================================

# 앱 시작 시 자동 실행
if __name__ != "__main__":
    # 설정 검증
    config_warnings = validate_config()
    
    # 경고 로깅
    if config_warnings:
        import logging
        logger = logging.getLogger(__name__)
        
        for level, message in config_warnings:
            if level == 'CRITICAL':
                logger.critical(message)
            elif level == 'ERROR':
                logger.error(message)
            elif level == 'WARNING':
                logger.warning(message)
            else:
                logger.info(message)
    
    # 개발 환경에서 설정 스냅샷 저장
    if IS_DEVELOPMENT and not IS_FROZEN:
        try:
            snapshot_path = save_config_snapshot()
            if DEBUG:
                print(f"Config snapshot saved: {snapshot_path}")
        except Exception as e:
            if DEBUG:
                print(f"Failed to save config snapshot: {e}")

# ============================================================================
# 📤 Public API
# ============================================================================

__all__ = [
    # 환경 정보
    'PROJECT_ROOT', 'DATA_DIR', 'ENV', 'IS_PRODUCTION', 'IS_DEVELOPMENT',
    'IS_DESKTOP', 'DEBUG', 'SYSTEM_INFO',
    
    # 앱 정보
    'APP_INFO',
    
    # 주요 설정
    'AI_ENGINES', 'AI_EXPLANATION_CONFIG', 'FILE_CONFIG', 
    'PROTOCOL_EXTRACTION_CONFIG', 'SQLITE_CONFIG', 'GOOGLE_SHEETS_CONFIG',
    'SECURITY_CONFIG', 'UI_CONFIG', 'EXPERIMENT_CONFIG', 'MODULE_CONFIG',
    'SYNC_CONFIG', 'UPDATE_CONFIG', 'LOCALIZATION_CONFIG', 
    'PERFORMANCE_CONFIG', 'DEVELOPER_CONFIG', 'FEATURE_FLAGS',
    
    # 유틸리티 함수
    'get_config', 'validate_config', 'save_config_snapshot'
]
