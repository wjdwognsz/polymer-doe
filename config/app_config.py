"""
⚙️ Universal DOE Platform - 앱 전역 설정
================================================================================
데스크톱 애플리케이션에 최적화된 중앙 설정 관리 시스템
오프라인 우선 설계, 확장 가능한 구조, 타입 안정성 보장
================================================================================
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import timedelta
from enum import Enum
import platform
import json

# ============================================================================
# 🔧 환경 설정
# ============================================================================

# PyInstaller 빌드 대응 경로 처리
if getattr(sys, 'frozen', False):
    # PyInstaller로 패키징된 경우
    PROJECT_ROOT = Path(sys._MEIPASS)
    DATA_DIR = Path(sys.executable).parent / 'data'
else:
    # 개발 환경
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / 'data'

# 주요 디렉토리 설정
CONFIG_DIR = PROJECT_ROOT / "config"
LOGS_DIR = DATA_DIR / "logs"
TEMP_DIR = DATA_DIR / "temp"
CACHE_DIR = DATA_DIR / "cache"
DB_DIR = DATA_DIR / "db"
MODULES_DIR = DATA_DIR / "modules"
BACKUP_DIR = DATA_DIR / "backups"

# 환경 변수
ENV = os.getenv('APP_ENV', 'development')
IS_PRODUCTION = ENV == 'production'
IS_DEVELOPMENT = ENV == 'development'
IS_FROZEN = getattr(sys, 'frozen', False)
DEBUG = os.getenv('DEBUG_MODE', 'false').lower() == 'true' and not IS_PRODUCTION

# 시스템 정보
SYSTEM_INFO = {
    'platform': platform.system(),  # Windows, Darwin, Linux
    'platform_version': platform.version(),
    'python_version': sys.version,
    'is_64bit': sys.maxsize > 2**32,
    'cpu_count': os.cpu_count() or 1
}

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
    'license': 'MIT',
    'copyright': '© 2024 DOE Team. All rights reserved.'
}

# ============================================================================
# 🤖 AI 엔진 설정
# ============================================================================

class AIProvider(Enum):
    """AI 제공자 열거형"""
    GOOGLE_GEMINI = "google_gemini"
    XAI_GROK = "xai_grok"
    GROQ = "groq"
    DEEPSEEK = "deepseek"
    SAMBANOVA = "sambanova"
    HUGGINGFACE = "huggingface"

AI_ENGINES = {
    AIProvider.GOOGLE_GEMINI: {
        'name': 'Google Gemini',
        'models': ['gemini-2.0-flash-exp', 'gemini-1.5-pro', 'gemini-1.5-flash'],
        'default_model': 'gemini-2.0-flash-exp',
        'max_tokens': 8192,
        'temperature': 0.7,
        'top_p': 0.95,
        'api_base': 'https://generativelanguage.googleapis.com/v1beta',
        'features': ['text', 'code', 'vision', 'function_calling'],
        'rate_limit': 60,  # requests per minute
        'free_tier': True,
        'required': True,  # 최소 하나는 필수
        'description': '빠르고 정확한 범용 AI, 무료 티어 제공'
    },
    AIProvider.XAI_GROK: {
        'name': 'xAI Grok',
        'models': ['grok-2-latest', 'grok-2-mini'],
        'default_model': 'grok-2-latest',
        'max_tokens': 131072,
        'temperature': 0.7,
        'api_base': 'https://api.x.ai/v1',
        'features': ['text', 'code', 'real_time_info'],
        'rate_limit': 60,
        'free_tier': False,
        'required': False,
        'description': '실시간 정보 접근, 대용량 컨텍스트'
    },
    AIProvider.GROQ: {
        'name': 'Groq',
        'models': ['llama-3.1-70b-versatile', 'llama-3.1-8b-instant', 'mixtral-8x7b-32768'],
        'default_model': 'llama-3.1-70b-versatile',
        'max_tokens': 8192,
        'temperature': 0.7,
        'api_base': 'https://api.groq.com/openai/v1',
        'features': ['text', 'code', 'ultra_fast'],
        'rate_limit': 30,
        'free_tier': True,
        'required': False,
        'description': '초고속 추론, 무료 티어 제공'
    },
    AIProvider.DEEPSEEK: {
        'name': 'DeepSeek',
        'models': ['deepseek-chat', 'deepseek-coder'],
        'default_model': 'deepseek-chat',
        'max_tokens': 16384,
        'temperature': 0.7,
        'api_base': 'https://api.deepseek.com/v1',
        'features': ['text', 'code', 'math', 'reasoning'],
        'rate_limit': 60,
        'free_tier': False,
        'required': False,
        'description': '코드와 수학에 특화, 추론 능력 우수'
    },
    AIProvider.SAMBANOVA: {
        'name': 'SambaNova',
        'models': ['Meta-Llama-3.1-405B-Instruct', 'Meta-Llama-3.1-70B-Instruct'],
        'default_model': 'Meta-Llama-3.1-70B-Instruct',
        'max_tokens': 4096,
        'temperature': 0.7,
        'api_base': 'https://api.sambanova.ai/v1',
        'features': ['text', 'code', 'enterprise'],
        'rate_limit': 100,
        'free_tier': True,
        'required': False,
        'description': '엔터프라이즈급 성능, 무료 클라우드'
    },
    AIProvider.HUGGINGFACE: {
        'name': 'HuggingFace',
        'models': ['microsoft/Phi-3-mini-4k-instruct', 'google/flan-t5-xxl', 'bigscience/bloom'],
        'default_model': 'microsoft/Phi-3-mini-4k-instruct',
        'max_tokens': 2048,
        'temperature': 0.7,
        'api_base': 'https://api-inference.huggingface.co/models',
        'features': ['text', 'specialized_models', 'fine_tuning'],
        'rate_limit': 100,
        'free_tier': True,
        'required': False,
        'description': '다양한 특화 모델, 커스터마이징 가능'
    }
}

# AI 설명 상세도 설정
AI_EXPLANATION_LEVELS = {
    'beginner': {
        'detail': 'simple',
        'technical_terms': False,
        'examples': True,
        'length': 'short'
    },
    'intermediate': {
        'detail': 'moderate',
        'technical_terms': True,
        'examples': True,
        'length': 'medium'
    },
    'advanced': {
        'detail': 'comprehensive',
        'technical_terms': True,
        'examples': False,
        'length': 'detailed'
    },
    'expert': {
        'detail': 'technical',
        'technical_terms': True,
        'examples': False,
        'length': 'concise'
    }
}

# ============================================================================
# 💾 데이터베이스 설정
# ============================================================================

# SQLite 설정
SQLITE_CONFIG = {
    'database_path': DB_DIR / 'universaldoe.db',
    'backup_path': BACKUP_DIR / 'db_backups',
    'backup_interval_hours': 24,
    'max_backups': 7,
    'pool_size': 5,
    'max_overflow': 10,
    'pool_timeout': 30,
    'echo': DEBUG,
    'foreign_keys': True,
    'journal_mode': 'WAL',  # Write-Ahead Logging for better concurrency
    'cache_size': -64000,  # 64MB cache
    'temp_store': 'MEMORY'
}

# Google Sheets 설정 (선택적 동기화)
GOOGLE_SHEETS_CONFIG = {
    'scopes': [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive.file'
    ],
    'api_version': 'v4',
    'batch_size': 1000,
    'rate_limit': 100,  # requests per minute
    'retry_count': 3,
    'retry_delay': 1.0,
    'cache_ttl': 300,  # 5 minutes
    'sync_enabled': False,  # 기본적으로 비활성화
    'sync_interval_minutes': 30,
    'conflict_resolution': 'local_first'  # local_first, remote_first, newest
}

# ============================================================================
# 🔐 보안 설정
# ============================================================================

SECURITY_CONFIG = {
    'session': {
        'secret_key': os.getenv('SESSION_SECRET_KEY', 'dev-secret-key-change-in-production'),
        'timeout_minutes': 30,
        'remember_me_days': 30,
        'max_sessions_per_user': 3,
        'secure_cookie': IS_PRODUCTION,
        'http_only': True,
        'same_site': 'Lax'
    },
    'password': {
        'min_length': 8,
        'require_uppercase': True,
        'require_lowercase': True,
        'require_numbers': True,
        'require_special': True,
        'bcrypt_rounds': 12,
        'reset_token_hours': 24,
        'max_attempts': 5,
        'lockout_minutes': 30
    },
    'api': {
        'rate_limit_per_minute': 60,
        'rate_limit_per_hour': 1000,
        'api_key_length': 32,
        'api_key_prefix': 'udoe_',
        'token_expiry_days': 90
    },
    'encryption': {
        'algorithm': 'AES-256-GCM',
        'key_derivation': 'PBKDF2',
        'iterations': 100000,
        'salt_length': 32
    },
    'jwt': {
        'secret_key': os.getenv('JWT_SECRET_KEY', 'dev-jwt-secret-change-in-production'),
        'algorithm': 'HS256',
        'access_token_expire_minutes': 60,
        'refresh_token_expire_days': 30
    }
}

# ============================================================================
# 🎨 UI/UX 설정
# ============================================================================

UI_CONFIG = {
    'theme': {
        'default': 'light',
        'available': ['light', 'dark', 'auto'],
        'primary_color': '#a880ed',  # 보라색
        'font_family': 'Inter, system-ui, sans-serif'
    },
    'layout': {
        'max_width': '1200px',
        'sidebar_width': '300px',
        'mobile_breakpoint': '768px',
        'default_page': 'dashboard'
    },
    'animations': {
        'enabled': True,
        'duration': '0.3s',
        'easing': 'ease-out'
    },
    'notifications': {
        'position': 'top-right',
        'duration': 5000,
        'max_stack': 3
    }
}

# ============================================================================
# 🧪 실험 설정
# ============================================================================

EXPERIMENT_CONFIG = {
    'design_types': {
        'factorial': '완전요인설계',
        'fractional': '부분요인설계',
        'rsm': '반응표면설계',
        'mixture': '혼합물설계',
        'custom': '사용자정의설계'
    },
    'max_factors': 20,
    'max_levels': 10,
    'max_runs': 10000,
    'confidence_levels': [0.90, 0.95, 0.99],
    'default_replicates': 3,
    'randomization': True,
    'blocking_enabled': True,
    'center_points': {
        'default': 3,
        'max': 10
    }
}

# 실험 프로젝트 타입
PROJECT_TYPES = {
    'polymer_synthesis': '고분자 합성',
    'polymer_processing': '고분자 가공',
    'polymer_characterization': '고분자 특성분석',
    'formulation': '배합 최적화',
    'material_testing': '재료 시험',
    'custom': '사용자 정의'
}

# 실험 기본값
EXPERIMENT_DEFAULTS = {
    'temperature_range': [20, 200],  # °C
    'time_range': [0.1, 24],  # hours
    'pressure_range': [0.1, 10],  # MPa
    'concentration_range': [0, 100],  # %
    'ph_range': [0, 14],
    'rpm_range': [0, 5000]
}

# ============================================================================
# 📦 파일 처리 설정
# ============================================================================

FILE_CONFIG = {
    'upload': {
        'max_size_mb': 200,
        'allowed_extensions': [
            # 데이터 파일
            '.csv', '.xlsx', '.xls', '.json', '.txt', '.tsv',
            # 이미지 파일
            '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg',
            # 문서 파일
            '.pdf', '.doc', '.docx', '.ppt', '.pptx',
            # 과학 데이터
            '.cif', '.mol', '.sdf', '.pdb', '.xyz',
            # 압축 파일
            '.zip', '.rar', '.7z', '.tar', '.gz'
        ],
        'temp_path': TEMP_DIR / 'uploads',
        'scan_viruses': IS_PRODUCTION
    },
    'export': {
        'formats': ['excel', 'csv', 'json', 'pdf', 'html'],
        'include_metadata': True,
        'compression': 'zip',
        'temp_path': TEMP_DIR / 'exports'
    },
    'templates': {
        'path': PROJECT_ROOT / 'templates',
        'categories': ['polymer', 'general', 'custom']
    }
}

# ============================================================================
# ⚡ 성능 설정
# ============================================================================

PERFORMANCE_CONFIG = {
    'max_workers': min(4, SYSTEM_INFO['cpu_count']),
    'chunk_size': 1000,
    'batch_size': 100,
    'timeout_seconds': 30,
    'memory_limit_mb': 2048,
    'gc_threshold': 1000,
    'profiling_enabled': DEBUG
}

# 캐시 설정
CACHE_CONFIG = {
    'enabled': True,
    'backend': 'disk',  # memory, disk, redis
    'max_size_mb': 500,
    'ttl_default': 3600,  # 1 hour
    'ttl_ai_response': 86400,  # 24 hours
    'ttl_api_call': 300,  # 5 minutes
    'ttl_computation': 7200,  # 2 hours
    'cleanup_interval': 3600,
    'redis_url': os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
    'redis_enabled': False  # 기본적으로 로컬 캐시 사용
}

# ============================================================================
# 🔧 모듈 시스템 설정
# ============================================================================

MODULE_CONFIG = {
    'core_modules': [
        'general_experiment',
        'polymer_experiment',
        'mixture_design',
        'optimization',
        'screening'
    ],
    'user_modules_path': MODULES_DIR / 'user',
    'marketplace_url': 'https://marketplace.universaldoe.com/api/v1',
    'auto_update': True,
    'validation_strict': True,
    'sandbox_enabled': True,
    'max_module_size_mb': 10
}

# ============================================================================
# 🔄 동기화 설정
# ============================================================================

SYNC_CONFIG = {
    'enabled': False,  # 기본적으로 비활성화
    'providers': ['google_drive', 'dropbox', 'onedrive', 'github'],
    'interval_minutes': 15,
    'conflict_strategy': 'manual',  # manual, local_wins, remote_wins
    'excluded_files': ['*.tmp', '*.log', '.DS_Store', 'Thumbs.db'],
    'bandwidth_limit_mbps': 10,
    'compress_before_sync': True
}

# ============================================================================
# 🔔 알림 설정
# ============================================================================

NOTIFICATION_CONFIG = {
    'channels': ['in_app', 'email', 'desktop'],
    'default_channel': 'in_app',
    'email': {
        'smtp_host': os.getenv('SMTP_HOST', 'smtp.gmail.com'),
        'smtp_port': int(os.getenv('SMTP_PORT', '587')),
        'use_tls': True,
        'username': os.getenv('SMTP_USERNAME', ''),
        'from_address': 'noreply@universaldoe.com',
        'from_name': 'Universal DOE Platform'
    },
    'desktop': {
        'enabled': SYSTEM_INFO['platform'] in ['Windows', 'Darwin'],
        'duration': 5000,
        'sound': True
    },
    'retention_days': 30,
    'batch_size': 100
}

# ============================================================================
# 🌍 지역화 설정
# ============================================================================

LOCALIZATION_CONFIG = {
    'default_language': 'ko',
    'supported_languages': {
        'ko': '한국어',
        'en': 'English',
        'ja': '日本語',
        'zh': '中文'
    },
    'date_format': '%Y-%m-%d',
    'time_format': '%H:%M:%S',
    'timezone': 'Asia/Seoul',
    'number_format': {
        'decimal_separator': '.',
        'thousands_separator': ',',
        'decimal_places': 2
    }
}

# ============================================================================
# 📊 분석 설정
# ============================================================================

ANALYTICS_CONFIG = {
    'enabled': not IS_PRODUCTION,  # 프로덕션에서는 프라이버시 보호
    'track_usage': False,
    'track_errors': True,
    'anonymize_data': True,
    'retention_days': 90,
    'export_format': 'json',
    'metrics': [
        'active_users',
        'experiments_created',
        'ai_usage',
        'module_usage',
        'error_rate'
    ]
}

# ============================================================================
# 🔄 업데이트 설정
# ============================================================================

UPDATE_CONFIG = {
    'check_updates': True,
    'auto_update': False,
    'channel': 'stable',  # stable, beta, dev
    'check_interval_hours': 24,
    'update_url': 'https://api.universaldoe.com/updates',
    'require_admin': SYSTEM_INFO['platform'] == 'Windows',
    'backup_before_update': True,
    'rollback_enabled': True
}

# ============================================================================
# 🛠️ 개발자 설정
# ============================================================================

DEVELOPER_CONFIG = {
    'debug_mode': DEBUG,
    'show_internal_errors': DEBUG,
    'enable_profiler': DEBUG,
    'log_level': 'DEBUG' if DEBUG else 'INFO',
    'sql_echo': DEBUG,
    'api_mock_enabled': IS_DEVELOPMENT,
    'test_data_enabled': IS_DEVELOPMENT,
    'hot_reload': IS_DEVELOPMENT and not IS_FROZEN,
    'dev_tools': {
        'memory_profiler': False,
        'api_explorer': IS_DEVELOPMENT,
        'db_browser': IS_DEVELOPMENT,
        'log_viewer': True
    }
}

# ============================================================================
# 🎯 기능 플래그
# ============================================================================

FEATURE_FLAGS = {
    # 핵심 기능
    'offline_mode': True,
    'ai_assistance': True,
    'collaboration': True,
    'cloud_sync': False,
    
    # 베타 기능
    'beta_features': IS_DEVELOPMENT,
    'new_ui': True,
    'advanced_analytics': True,
    'module_marketplace': True,
    'voice_commands': False,
    
    # 실험적 기능
    'experimental': {
        'ar_visualization': False,
        'ml_predictions': IS_DEVELOPMENT,
        'auto_optimization': False,
        'blockchain_verification': False
    },
    
    # 프리미엄 기능
    'premium': {
        'unlimited_projects': False,
        'priority_support': False,
        'custom_branding': False,
        'api_access': False
    }
}

# ============================================================================
# 🔧 유틸리티 함수
# ============================================================================

def get_config(key: str, default: Any = None) -> Any:
    """
    설정값 가져오기 (환경변수 우선)
    
    Args:
        key: 설정 키 (점 표기법 지원)
        default: 기본값
        
    Returns:
        설정값 또는 기본값
    """
    # 환경변수 확인 (DOE_ 접두사 사용)
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
    try:
        keys = key.split('.')
        value = globals()
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
    except:
        return default

def validate_config() -> Tuple[bool, List[str]]:
    """
    설정 유효성 검증
    
    Returns:
        (성공 여부, 오류 메시지 리스트)
    """
    errors = []
    
    # 디렉토리 생성
    for dir_path in [DATA_DIR, LOGS_DIR, TEMP_DIR, CACHE_DIR, DB_DIR, MODULES_DIR, BACKUP_DIR]:
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"디렉토리 생성 실패 {dir_path}: {e}")
    
    # 보안 키 검증 (프로덕션)
    if IS_PRODUCTION:
        if SECURITY_CONFIG['session']['secret_key'] == 'dev-secret-key-change-in-production':
            errors.append("⚠️ 프로덕션 환경에서 기본 세션 키 사용 중!")
        if SECURITY_CONFIG['jwt']['secret_key'] == 'dev-jwt-secret-change-in-production':
            errors.append("⚠️ 프로덕션 환경에서 기본 JWT 키 사용 중!")
    
    # 필수 AI 엔진 확인
    required_engines = [k for k, v in AI_ENGINES.items() if v.get('required')]
    if not required_engines:
        errors.append("최소 하나의 AI 엔진이 필수로 설정되어야 합니다")
    
    return len(errors) == 0, errors

def save_config_snapshot(filepath: Optional[Path] = None) -> Path:
    """
    현재 설정을 JSON 파일로 저장
    
    Args:
        filepath: 저장 경로 (기본: backups/config_snapshot_TIMESTAMP.json)
        
    Returns:
        저장된 파일 경로
    """
    if filepath is None:
        timestamp = platform.datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = BACKUP_DIR / f'config_snapshot_{timestamp}.json'
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # 직렬화 가능한 설정만 추출
    config_data = {
        'timestamp': platform.datetime.now().isoformat(),
        'environment': ENV,
        'version': APP_INFO['version'],
        'system': SYSTEM_INFO,
        'settings': {
            'app_info': APP_INFO,
            'ai_engines': {k.value: v for k, v in AI_ENGINES.items()},
            'experiment': EXPERIMENT_CONFIG,
            'ui': UI_CONFIG,
            'features': FEATURE_FLAGS
        }
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    return filepath

# ============================================================================
# 🚀 초기화
# ============================================================================

# 설정 검증
if __name__ != "__main__":
    success, errors = validate_config()
    
    if not success and errors:
        import logging
        logger = logging.getLogger(__name__)
        for error in errors:
            logger.error(error)

# ============================================================================
# 📤 외부 노출 API
# ============================================================================

__all__ = [
    # 환경 정보
    'PROJECT_ROOT', 'DATA_DIR', 'ENV', 'IS_PRODUCTION', 'IS_DEVELOPMENT',
    'DEBUG', 'SYSTEM_INFO',
    
    # 앱 정보
    'APP_INFO',
    
    # 주요 설정
    'AI_ENGINES', 'AI_EXPLANATION_LEVELS', 'SQLITE_CONFIG', 'GOOGLE_SHEETS_CONFIG',
    'SECURITY_CONFIG', 'UI_CONFIG', 'EXPERIMENT_CONFIG', 'FILE_CONFIG',
    'PERFORMANCE_CONFIG', 'CACHE_CONFIG', 'MODULE_CONFIG', 'SYNC_CONFIG',
    'NOTIFICATION_CONFIG', 'LOCALIZATION_CONFIG', 'ANALYTICS_CONFIG',
    'UPDATE_CONFIG', 'DEVELOPER_CONFIG', 'FEATURE_FLAGS',
    
    # 상수
    'AIProvider', 'PROJECT_TYPES', 'EXPERIMENT_DEFAULTS',
    
    # 유틸리티 함수
    'get_config', 'validate_config', 'save_config_snapshot'
]
