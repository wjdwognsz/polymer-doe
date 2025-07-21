"""
⚙️ Universal DOE Platform - 앱 전역 설정
================================================================================
데스크톱 애플리케이션에 최적화된 중앙 설정 관리 시스템
오프라인 우선 설계, 확장 가능한 구조, 타입 안정성 보장
================================================================================
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import timedelta
from enum import Enum
import platform

# ============================================================================
# 🔧 환경 설정
# ============================================================================

# 프로젝트 경로
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "config"
LOGS_DIR = DATA_DIR / "logs"
TEMP_DIR = DATA_DIR / "temp"
CACHE_DIR = DATA_DIR / "cache"

# 환경 변수
ENV = os.getenv('APP_ENV', 'development')
IS_PRODUCTION = ENV == 'production'
IS_DEVELOPMENT = ENV == 'development'
DEBUG = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

# ============================================================================
# 📱 앱 기본 정보
# ============================================================================

APP_INFO = {
    'name': 'Universal DOE Platform',
    'version': '2.0.0',
    'description': '모든 연구자를 위한 AI 기반 실험 설계 데스크톱 플랫폼',
    'author': 'DOE Team',
    'email': 'support@universaldoe.com',
    'website': 'https://universaldoe.com',
    'github': 'https://github.com/universaldoe/platform',
    'license': 'MIT'
}

# ============================================================================
# 🤖 AI 엔진 설정
# ============================================================================

AI_ENGINES = {
    'google_gemini': {
        'name': 'Google Gemini 2.0 Flash',
        'model': 'gemini-2.0-flash-latest',
        'description': '가장 빠르고 효율적인 범용 AI',
        'provider': 'google',
        'api_base': 'https://generativelanguage.googleapis.com',
        'features': ['text', 'code', 'analysis', 'vision'],
        'rate_limit': 60,  # requests per minute
        'max_tokens': 8192,
        'free_tier': True,
        'required': True,
        'priority': 1
    },
    'xai_grok': {
        'name': 'xAI Grok 3 Mini',
        'model': 'grok-3-mini',
        'description': '실시간 정보와 유머를 갖춘 AI',
        'provider': 'xai',
        'api_base': 'https://api.x.ai/v1',
        'features': ['text', 'realtime', 'humor'],
        'rate_limit': 30,
        'max_tokens': 4096,
        'free_tier': False,
        'required': False,
        'priority': 2
    },
    'groq': {
        'name': 'Groq (초고속 추론)',
        'model': 'mixtral-8x7b-32768',
        'description': '가장 빠른 응답 속도',
        'provider': 'groq',
        'api_base': 'https://api.groq.com/openai/v1',
        'features': ['text', 'code', 'speed'],
        'rate_limit': 100,
        'max_tokens': 32768,
        'free_tier': True,
        'required': False,
        'priority': 3
    },
    'deepseek': {
        'name': 'DeepSeek (코드/수식 특화)',
        'model': 'deepseek-chat',
        'description': '과학 계산과 코드 생성 전문',
        'provider': 'deepseek',
        'api_base': 'https://api.deepseek.com/v1',
        'features': ['code', 'math', 'science'],
        'rate_limit': 60,
        'max_tokens': 16384,
        'free_tier': False,
        'required': False,
        'priority': 4
    },
    'sambanova': {
        'name': 'SambaNova (대규모 모델)',
        'model': 'llama-3.1-405b',
        'description': '복잡한 분석과 추론',
        'provider': 'sambanova',
        'api_base': 'https://api.sambanova.ai/v1',
        'features': ['text', 'analysis', 'reasoning'],
        'rate_limit': 10,
        'max_tokens': 4096,
        'free_tier': True,
        'required': False,
        'priority': 5
    },
    'huggingface': {
        'name': 'HuggingFace (특수 모델)',
        'models': ['ChemBERTa', 'MatSciBERT', 'BioBERT'],
        'description': '도메인 특화 모델',
        'provider': 'huggingface',
        'api_base': 'https://api-inference.huggingface.co',
        'features': ['specialized', 'domain-specific'],
        'rate_limit': 100,
        'max_tokens': 512,
        'free_tier': True,
        'required': False,
        'priority': 6
    }
}

# ============================================================================
# 💾 데이터베이스 설정
# ============================================================================

# SQLite 설정 (기본)
SQLITE_CONFIG = {
    'database_path': DATA_DIR / 'db' / 'app.db',
    'backup_enabled': True,
    'backup_interval': timedelta(hours=1),
    'backup_retention': 5,  # 최대 백업 파일 수
    'wal_mode': True,  # Write-Ahead Logging
    'foreign_keys': True,
    'journal_mode': 'WAL',
    'synchronous': 'NORMAL',
    'cache_size': -64000,  # 64MB
    'temp_store': 'MEMORY'
}

# Google Sheets 설정 (선택적 동기화)
GOOGLE_SHEETS_CONFIG = {
    'enabled': os.getenv('GOOGLE_SHEETS_ENABLED', 'false').lower() == 'true',
    'spreadsheet_url': os.getenv('GOOGLE_SHEETS_URL', ''),
    'sync_interval': timedelta(minutes=5),
    'batch_size': 100,
    'rate_limit': 60,  # requests per minute
    'scopes': [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive.file'
    ],
    'sheet_names': {
        'users': 'Users',
        'projects': 'Projects',
        'experiments': 'Experiments',
        'results': 'Results',
        'shared_data': 'SharedData',
        'templates': 'Templates'
    }
}

# ============================================================================
# 🔐 보안 설정
# ============================================================================

SECURITY_CONFIG = {
    'session': {
        'secret_key': os.getenv('SESSION_SECRET_KEY', 'dev-secret-key-change-in-production'),
        'timeout': timedelta(hours=24),
        'remember_me_duration': timedelta(days=30),
        'max_concurrent_sessions': 3,
        'cookie_secure': IS_PRODUCTION,
        'cookie_httponly': True,
        'cookie_samesite': 'Lax'
    },
    'password': {
        'min_length': 8,
        'require_uppercase': True,
        'require_lowercase': True,
        'require_digits': True,
        'require_special': True,
        'bcrypt_rounds': 12,
        'reset_token_expiry': timedelta(hours=24),
        'max_attempts': 5,
        'lockout_duration': timedelta(minutes=30)
    },
    'api_keys': {
        'encryption_key': os.getenv('ENCRYPTION_KEY', 'dev-encryption-key'),
        'rotation_days': 90,
        'audit_log': True
    },
    'jwt': {
        'secret_key': os.getenv('JWT_SECRET_KEY', 'dev-jwt-secret'),
        'algorithm': 'HS256',
        'expiry': timedelta(hours=24)
    }
}

# ============================================================================
# 🎨 UI/UX 설정
# ============================================================================

UI_CONFIG = {
    'theme': {
        'default': 'light',
        'allow_dark_mode': True,
        'primary_color': '#a880ed',  # theme_config.py와 동기화
        'auto_detect_system': True
    },
    'layout': {
        'sidebar_default': 'expanded',
        'wide_mode_default': True,
        'show_footer': True,
        'show_header': True
    },
    'language': {
        'default': 'ko',
        'supported': ['ko', 'en'],
        'auto_detect': True
    },
    'notifications': {
        'position': 'top-right',
        'duration': 5000,  # milliseconds
        'max_stack': 3
    }
}

# ============================================================================
# 📁 파일 업로드 설정
# ============================================================================

FILE_UPLOAD_CONFIG = {
    'max_file_size': 200 * 1024 * 1024,  # 200MB
    'max_files_per_upload': 10,
    'allowed_extensions': {
        'data': ['.csv', '.xlsx', '.xls', '.json', '.parquet', '.txt'],
        'document': ['.pdf', '.docx', '.doc', '.pptx', '.md'],
        'image': ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp'],
        'code': ['.py', '.ipynb', '.r', '.m', '.jl'],
        'module': ['.py', '.json', '.yaml']
    },
    'temp_dir': TEMP_DIR,
    'cleanup_interval': timedelta(hours=24),
    'virus_scan': False  # 로컬 앱이므로 비활성화
}

# ============================================================================
# ⚡ 성능 설정
# ============================================================================

PERFORMANCE_CONFIG = {
    'cache': {
        'enabled': True,
        'backend': 'memory',  # memory, redis, file
        'ttl': {
            'api_response': timedelta(minutes=30),
            'analysis_result': timedelta(hours=1),
            'user_data': timedelta(minutes=5),
            'static_data': timedelta(hours=24)
        },
        'max_size_mb': 500,
        'eviction_policy': 'LRU'
    },
    'parallel_processing': {
        'enabled': True,
        'max_workers': min(4, os.cpu_count() or 1),
        'chunk_size': 1000
    },
    'batch_processing': {
        'default_batch_size': 100,
        'max_batch_size': 1000,
        'timeout': timedelta(seconds=30)
    },
    'rate_limiting': {
        'enabled': True,
        'default_limit': 60,  # requests per minute
        'burst_size': 100
    }
}

# ============================================================================
# 🧪 실험 설계 기본값
# ============================================================================

EXPERIMENT_DEFAULTS = {
    'design_types': {
        'screening': ['Plackett-Burman', 'Fractional Factorial'],
        'optimization': ['Central Composite', 'Box-Behnken', 'D-Optimal'],
        'mixture': ['Simplex Lattice', 'Simplex Centroid'],
        'robust': ['Taguchi', 'Split-Plot'],
        'custom': ['Custom Design']
    },
    'constraints': {
        'min_runs': 3,
        'max_runs': 1000,
        'max_factors': 50,
        'max_responses': 20
    },
    'statistics': {
        'confidence_level': 0.95,
        'power': 0.8,
        'alpha': 0.05,
        'replicates_min': 2
    },
    'optimization': {
        'methods': ['gradient', 'genetic', 'bayesian', 'grid'],
        'max_iterations': 1000,
        'tolerance': 1e-6
    }
}

# ============================================================================
# 🌐 오프라인 모드 설정
# ============================================================================

OFFLINE_CONFIG = {
    'default_mode': True,  # 기본적으로 오프라인
    'features': {
        'ai_chat': 'limited',  # limited, disabled
        'collaboration': False,
        'cloud_sync': False,
        'marketplace': 'cached',
        'literature_search': False,
        'updates': 'manual'
    },
    'cache_policy': {
        'ai_responses': True,
        'analysis_results': True,
        'templates': True,
        'modules': True
    },
    'sync_on_connect': True,
    'offline_duration_limit': None  # 무제한
}

# ============================================================================
# 📊 분석 설정
# ============================================================================

ANALYSIS_CONFIG = {
    'statistical_tests': {
        'normality': ['shapiro', 'anderson', 'kstest'],
        'variance': ['levene', 'bartlett', 'fligner'],
        'correlation': ['pearson', 'spearman', 'kendall'],
        'regression': ['linear', 'polynomial', 'stepwise']
    },
    'visualization': {
        'default_backend': 'plotly',
        'themes': ['default', 'publication', 'presentation'],
        'export_formats': ['png', 'svg', 'pdf', 'html'],
        'dpi': 300
    },
    'reporting': {
        'formats': ['html', 'pdf', 'docx', 'pptx'],
        'templates': ['academic', 'industry', 'summary'],
        'include_code': False,
        'include_raw_data': False
    }
}

# ============================================================================
# 🔄 동기화 설정
# ============================================================================

SYNC_CONFIG = {
    'enabled': False,  # 기본적으로 비활성화
    'interval': timedelta(minutes=5),
    'conflict_resolution': 'local_first',  # local_first, remote_first, newest
    'retry_attempts': 3,
    'retry_delay': timedelta(seconds=5),
    'batch_size': 50,
    'compression': True
}

# ============================================================================
# 📦 모듈 시스템 설정
# ============================================================================

MODULE_CONFIG = {
    'enabled': True,
    'auto_discovery': True,
    'module_dirs': [
        PROJECT_ROOT / 'modules' / 'core',
        DATA_DIR / 'modules' / 'user',
        DATA_DIR / 'modules' / 'community'
    ],
    'validation': {
        'strict': True,
        'sandbox': True,
        'timeout': timedelta(seconds=30)
    },
    'marketplace': {
        'enabled': True,
        'api_endpoint': 'https://api.universaldoe.com/modules',
        'cache_duration': timedelta(days=1)
    }
}

# ============================================================================
# 🚀 자동 업데이트 설정
# ============================================================================

UPDATE_CONFIG = {
    'enabled': os.getenv('AUTO_UPDATE_ENABLED', 'true').lower() == 'true',
    'check_interval': timedelta(days=1),
    'channel': 'stable',  # stable, beta, nightly
    'server_url': 'https://api.universaldoe.com/updates',
    'download_timeout': timedelta(minutes=30),
    'install_on_exit': True,
    'show_release_notes': True
}

# ============================================================================
# 📍 지역화 설정
# ============================================================================

LOCALIZATION_CONFIG = {
    'default_locale': 'ko_KR',
    'fallback_locale': 'en_US',
    'timezone': os.getenv('TIMEZONE', 'Asia/Seoul'),
    'date_format': '%Y-%m-%d',
    'time_format': '%H:%M:%S',
    'datetime_format': '%Y-%m-%d %H:%M:%S',
    'number_format': {
        'decimal_separator': '.',
        'thousands_separator': ',',
        'decimal_places': 2
    }
}

# ============================================================================
# 🛠️ 개발자 설정
# ============================================================================

if IS_DEVELOPMENT:
    # 개발 환경 전용 설정
    DEBUG_CONFIG = {
        'show_debug_toolbar': True,
        'log_level': 'DEBUG',
        'profile_performance': False,
        'show_error_details': True,
        'hot_reload': True,
        'mock_data': True,
        'bypass_auth': False
    }
else:
    DEBUG_CONFIG = {
        'show_debug_toolbar': False,
        'log_level': 'INFO',
        'profile_performance': False,
        'show_error_details': False,
        'hot_reload': False,
        'mock_data': False,
        'bypass_auth': False
    }

# ============================================================================
# 🎯 통합 설정 내보내기
# ============================================================================

def get_config(key: str, default: Any = None) -> Any:
    """설정값 가져오기 (환경변수 우선)"""
    # 환경변수에서 먼저 찾기
    env_key = key.upper().replace('.', '_')
    env_value = os.getenv(env_key)
    
    if env_value is not None:
        # 타입 변환 시도
        if isinstance(default, bool):
            return env_value.lower() in ('true', '1', 'yes', 'on')
        elif isinstance(default, int):
            try:
                return int(env_value)
            except ValueError:
                return default
        elif isinstance(default, float):
            try:
                return float(env_value)
            except ValueError:
                return default
        else:
            return env_value
    
    # 설정에서 찾기
    config_dict = globals()
    keys = key.split('.')
    value = config_dict
    
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    
    return value

def update_config(key: str, value: Any) -> None:
    """설정값 업데이트 (런타임 전용)"""
    config_dict = globals()
    keys = key.split('.')
    
    # 마지막 키 전까지 탐색
    for k in keys[:-1]:
        if k not in config_dict:
            config_dict[k] = {}
        config_dict = config_dict[k]
    
    # 값 설정
    config_dict[keys[-1]] = value

def validate_config() -> List[str]:
    """설정 검증 및 경고 반환"""
    warnings = []
    
    # 필수 디렉토리 생성
    for dir_path in [DATA_DIR, LOGS_DIR, TEMP_DIR, CACHE_DIR]:
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                warnings.append(f"Failed to create directory {dir_path}: {e}")
    
    # SQLite 파일 경로 확인
    db_path = SQLITE_CONFIG['database_path']
    if not db_path.parent.exists():
        try:
            db_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            warnings.append(f"Failed to create database directory: {e}")
    
    # 보안 키 확인
    if IS_PRODUCTION:
        if SECURITY_CONFIG['session']['secret_key'] == 'dev-secret-key-change-in-production':
            warnings.append("Using default session secret key in production!")
        if SECURITY_CONFIG['jwt']['secret_key'] == 'dev-jwt-secret':
            warnings.append("Using default JWT secret key in production!")
    
    return warnings

# 시작 시 설정 검증
if __name__ != "__main__":
    validation_warnings = validate_config()
    if validation_warnings:
        import logging
        logger = logging.getLogger(__name__)
        for warning in validation_warnings:
            logger.warning(warning)
