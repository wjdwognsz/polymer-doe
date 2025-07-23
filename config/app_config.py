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

AI_ENGINES = {
    'google_gemini': {
        'name': 'Google Gemini 2.0 Flash',
        'model': 'gemini-2.0-flash-exp',
        'description': '가장 빠르고 효율적인 범용 AI',
        'provider': 'google',
        'api_base': 'https://generativelanguage.googleapis.com',
        'features': ['text', 'code', 'analysis', 'vision'],
        'rate_limit': 60,  # requests per minute
        'max_tokens': 8192,
        'free_tier': True,
        'required': True,
        'priority': 1,
        'cost_per_1k_tokens': {'input': 0.0, 'output': 0.0},  # 무료
        'best_for': ['general', 'fast_response', 'multimodal']
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
        'priority': 4,
        'cost_per_1k_tokens': {'input': 0.002, 'output': 0.006},
        'best_for': ['realtime_info', 'creative', 'conversational']
    },
    'groq': {
        'name': 'Groq (Mixtral)',
        'model': 'mixtral-8x7b-32768',
        'description': '초고속 추론 엔진',
        'provider': 'groq',
        'api_base': 'https://api.groq.com/openai/v1',
        'features': ['text', 'code', 'fast'],
        'rate_limit': 100,
        'max_tokens': 32768,
        'free_tier': True,
        'required': False,
        'priority': 2,
        'cost_per_1k_tokens': {'input': 0.0, 'output': 0.0},  # 무료
        'best_for': ['speed', 'code_generation', 'large_context']
    },
    'deepseek': {
        'name': 'DeepSeek Chat',
        'model': 'deepseek-chat',
        'description': '코드와 수식에 특화된 AI',
        'provider': 'deepseek',
        'api_base': 'https://api.deepseek.com/v1',
        'features': ['text', 'code', 'math'],
        'rate_limit': 60,
        'max_tokens': 16384,
        'free_tier': False,
        'required': False,
        'priority': 3,
        'cost_per_1k_tokens': {'input': 0.001, 'output': 0.002},
        'best_for': ['code', 'mathematics', 'technical']
    },
    'sambanova': {
        'name': 'SambaNova (Llama 3.1)',
        'model': 'llama-3.1-405b',
        'description': '최대 규모의 오픈소스 모델',
        'provider': 'sambanova',
        'api_base': 'https://api.sambanova.ai/v1',
        'features': ['text', 'analysis', 'reasoning'],
        'rate_limit': 10,
        'max_tokens': 8192,
        'free_tier': True,
        'required': False,
        'priority': 5,
        'cost_per_1k_tokens': {'input': 0.0, 'output': 0.0},  # 무료
        'best_for': ['complex_reasoning', 'analysis', 'research']
    },
    'huggingface': {
        'name': 'HuggingFace Models',
        'models': ['ChemBERTa', 'MatSciBERT', 'BioBERT'],
        'description': '과학 분야 특화 모델들',
        'provider': 'huggingface',
        'api_base': 'https://api-inference.huggingface.co',
        'features': ['specialized', 'embeddings', 'classification'],
        'rate_limit': 100,
        'max_tokens': 512,
        'free_tier': True,
        'required': False,
        'priority': 6,
        'cost_per_1k_tokens': {'input': 0.0, 'output': 0.0},
        'best_for': ['domain_specific', 'embeddings', 'classification']
    }
}

# AI 설명 상세도 제어 (새로 추가된 AI 투명성 원칙)
AI_EXPLANATION_CONFIG = {
    'default_mode': 'auto',  # auto, always_detailed, always_simple, custom
    'auto_mode_rules': {
        'beginner': 'detailed',
        'intermediate': 'balanced',
        'advanced': 'simple',
        'expert': 'minimal'
    },
    'detail_sections': {
        'reasoning': True,      # 추론 과정
        'alternatives': True,   # 대안 검토
        'background': True,     # 이론적 배경
        'confidence': True,     # 신뢰도
        'limitations': True     # 한계점
    },
    'keyboard_shortcut': 'Ctrl+D',
    'toggle_animation': True,
    'remember_preference': True
}

# ============================================================================
# 💾 데이터베이스 설정
# ============================================================================

# SQLite 설정 (기본 로컬 DB)
SQLITE_CONFIG = {
    'database_path': DB_DIR / 'universaldoe.db',
    'backup_enabled': True,
    'backup_interval': timedelta(hours=24),
    'backup_count': 7,  # 최대 백업 파일 수
    'vacuum_on_startup': True,
    'journal_mode': 'WAL',  # Write-Ahead Logging
    'synchronous': 'NORMAL',
    'cache_size': -64000,  # 64MB
    'busy_timeout': 5000  # 5초
}

# Google Sheets 설정 (선택적 클라우드 동기화)
GOOGLE_SHEETS_CONFIG = {
    'enabled': False,  # 기본값: 비활성화
    'scope': [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ],
    'sync_interval': timedelta(minutes=5),
    'conflict_resolution': 'local_first',  # local_first, remote_first, newest, manual
    'sheet_names': {
        'users': 'Users',
        'projects': 'Projects',
        'experiments': 'Experiments',
        'results': 'Results',
        'modules': 'Modules',
        'templates': 'Templates'
    },
    'rate_limit': 100,  # requests per minute
    'batch_size': 500,  # rows per batch
    'retry_config': {
        'max_attempts': 3,
        'initial_delay': 1,
        'exponential_base': 2
    }
}

# ============================================================================
# 🔐 보안 설정
# ============================================================================

SECURITY_CONFIG = {
    'password': {
        'min_length': 8,
        'require_uppercase': True,
        'require_lowercase': True,
        'require_numbers': True,
        'require_special': True,
        'special_chars': '!@#$%^&*()_+-=[]{}|;:,.<>?',
        'bcrypt_rounds': 12
    },
    'session': {
        'timeout': timedelta(minutes=30),
        'refresh_threshold': timedelta(minutes=5),
        'max_concurrent': 3,
        'remember_me_duration': timedelta(days=30),
        'secret_key': os.getenv('SESSION_SECRET', 'dev-secret-key-change-in-production'),
        'cookie_secure': IS_PRODUCTION,
        'cookie_httponly': True
    },
    'jwt': {
        'algorithm': 'HS256',
        'secret_key': os.getenv('JWT_SECRET', 'dev-jwt-secret'),
        'access_token_expire': timedelta(minutes=15),
        'refresh_token_expire': timedelta(days=7)
    },
    'encryption': {
        'algorithm': 'AES-256-GCM',
        'key_derivation': 'PBKDF2',
        'iterations': 100000
    },
    'rate_limiting': {
        'login_attempts': 5,
        'lockout_duration': timedelta(minutes=30),
        'api_calls_per_minute': 60
    },
    'api_keys': {
        'rotation_days': 90,
        'encryption_enabled': True,
        'audit_access': True
    }
}

# ============================================================================
# 🌐 네트워크 및 API 설정
# ============================================================================

NETWORK_CONFIG = {
    'timeout': {
        'connect': 10,  # 초
        'read': 30,
        'write': 30,
        'pool': 5
    },
    'retry': {
        'max_attempts': 3,
        'backoff_factor': 1.5,
        'status_forcelist': [408, 429, 500, 502, 503, 504]
    },
    'proxy': {
        'enabled': False,
        'http': os.getenv('HTTP_PROXY', ''),
        'https': os.getenv('HTTPS_PROXY', ''),
        'no_proxy': ['localhost', '127.0.0.1', '.local']
    },
    'ssl': {
        'verify': True,
        'cert_path': None,
        'key_path': None
    }
}

# ============================================================================
# 🎨 UI/UX 설정
# ============================================================================

UI_CONFIG = {
    'theme': {
        'default': 'light',
        'allow_dark_mode': True,
        'auto_detect_system': True,
        'primary_color': '#1E88E5',  # Material Blue 600
        'secondary_color': '#43A047',  # Material Green 600
        'accent_color': '#E53935',  # Material Red 600
    },
    'layout': {
        'sidebar_state': 'expanded',
        'wide_mode': True,
        'show_footer': True,
        'show_header': True,
        'max_width': 1200
    },
    'language': {
        'default': 'ko',
        'supported': ['ko', 'en'],
        'auto_detect': True,
        'fallback': 'en'
    },
    'notifications': {
        'position': 'top-right',
        'duration': 5000,  # milliseconds
        'max_stack': 3,
        'animation': 'slide',
        'sound_enabled': True
    },
    'charts': {
        'default_height': 400,
        'responsive': True,
        'animation_duration': 1000,
        'color_palette': 'plotly'
    }
}

# ============================================================================
# 📁 파일 처리 설정
# ============================================================================

FILE_CONFIG = {
    'upload': {
        'max_size_mb': 200,
        'max_files': 10,
        'chunk_size_kb': 1024,
        'allowed_extensions': {
            'data': ['.csv', '.xlsx', '.xls', '.json', '.parquet', '.txt', '.tsv'],
            'document': ['.pdf', '.docx', '.doc', '.pptx', '.md', '.rtf'],
            'image': ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.bmp'],
            'code': ['.py', '.ipynb', '.r', '.m', '.jl', '.cpp', '.js'],
            'module': ['.py', '.json', '.yaml', '.yml']
        },
        'scan_for_malware': False,  # 로컬 앱이므로 비활성화
        'auto_cleanup_hours': 24
    },
    'export': {
        'formats': ['xlsx', 'csv', 'json', 'pdf', 'html'],
        'compression': True,
        'include_metadata': True,
        'timestamp_format': '%Y%m%d_%H%M%S'
    },
    'temp': {
        'dir': TEMP_DIR,
        'cleanup_on_exit': True,
        'max_age_hours': 48
    }
}

# ============================================================================
# ⚡ 성능 설정
# ============================================================================

PERFORMANCE_CONFIG = {
    'cache': {
        'enabled': True,
        'backend': 'memory',  # memory, file, redis (future)
        'memory_limit_mb': 500,
        'file_cache_dir': CACHE_DIR,
        'ttl': {
            'api_response': timedelta(minutes=30),
            'analysis_result': timedelta(hours=1),
            'user_data': timedelta(minutes=5),
            'static_data': timedelta(hours=24),
            'module_list': timedelta(hours=6)
        },
        'compression': True,
        'eviction_policy': 'LRU'
    },
    'parallel': {
        'enabled': True,
        'max_workers': min(4, SYSTEM_INFO['cpu_count']),
        'thread_name_prefix': 'DOE-Worker',
        'queue_size': 100
    },
    'batch': {
        'default_size': 100,
        'max_size': 1000,
        'timeout_seconds': 30
    },
    'memory': {
        'gc_threshold': 80,  # percentage
        'monitor_interval': timedelta(minutes=5),
        'log_usage': IS_DEVELOPMENT
    }
}

# ============================================================================
# 📊 실험 설계 설정
# ============================================================================

EXPERIMENT_CONFIG = {
    'design_types': [
        'Full Factorial',
        'Fractional Factorial',
        'Central Composite',
        'Box-Behnken',
        'Plackett-Burman',
        'Latin Hypercube',
        'D-Optimal',
        'Custom'
    ],
    'constraints': {
        'min_runs': 3,
        'max_runs': 10000,
        'max_factors': 50,
        'max_responses': 20
    },
    'optimization': {
        'algorithms': ['gradient', 'genetic', 'bayesian', 'grid'],
        'max_iterations': 1000,
        'convergence_tolerance': 1e-6,
        'parallel_trials': True
    },
    'validation': {
        'cross_validation_folds': 5,
        'test_size_ratio': 0.2,
        'random_state': 42
    },
    'templates': {
        'enabled': True,
        'categories': ['chemistry', 'materials', 'biology', 'engineering'],
        'custom_allowed': True,
        'share_enabled': True
    }
}

# ============================================================================
# 📦 모듈 시스템 설정
# ============================================================================

MODULE_CONFIG = {
    'discovery': {
        'enabled': True,
        'scan_on_startup': True,
        'watch_directories': True,
        'auto_reload': IS_DEVELOPMENT
    },
    'directories': {
        'core': PROJECT_ROOT / 'modules' / 'core',
        'user': MODULES_DIR / 'user',
        'community': MODULES_DIR / 'community',
        'temp': MODULES_DIR / 'temp'
    },
    'validation': {
        'strict_mode': True,
        'sandbox_enabled': True,
        'timeout_seconds': 30,
        'memory_limit_mb': 512,
        'allowed_imports': [
            'numpy', 'pandas', 'scipy', 'sklearn',
            'matplotlib', 'plotly', 'streamlit'
        ]
    },
    'marketplace': {
        'enabled': True,
        'api_endpoint': 'https://api.universaldoe.com/modules',
        'cache_duration': timedelta(hours=6),
        'featured_count': 10,
        'reviews_enabled': True
    },
    'development': {
        'template_repo': 'https://github.com/universaldoe/module-template',
        'docs_url': 'https://docs.universaldoe.com/modules',
        'debug_mode': IS_DEVELOPMENT
    }
}

# ============================================================================
# 🔄 동기화 설정
# ============================================================================

SYNC_CONFIG = {
    'enabled': False,  # 기본적으로 오프라인
    'mode': 'manual',  # manual, auto, scheduled
    'interval': timedelta(minutes=5),
    'conflict_resolution': 'local_first',
    'strategies': {
        'projects': 'merge',
        'experiments': 'latest',
        'results': 'append',
        'modules': 'version'
    },
    'compression': True,
    'encryption': True,
    'batch_size': 50,
    'queue': {
        'max_size': 1000,
        'priority_levels': 3,
        'retry_failed': True
    }
}

# ============================================================================
# 🚀 자동 업데이트 설정
# ============================================================================

UPDATE_CONFIG = {
    'enabled': True,
    'check_on_startup': True,
    'check_interval': timedelta(days=1),
    'channel': 'stable',  # stable, beta, nightly
    'server': {
        'url': 'https://api.universaldoe.com/updates',
        'timeout': 30,
        'verify_signature': True
    },
    'download': {
        'chunk_size_kb': 1024,
        'resume_enabled': True,
        'verify_checksum': True,
        'temp_dir': TEMP_DIR / 'updates'
    },
    'install': {
        'mode': 'on_restart',  # immediate, on_restart, scheduled
        'backup_current': True,
        'rollback_enabled': True,
        'silent_mode': False
    },
    'notifications': {
        'show_available': True,
        'show_progress': True,
        'show_release_notes': True
    }
}

# ============================================================================
# 📍 지역화 설정
# ============================================================================

LOCALIZATION_CONFIG = {
    'default_locale': 'ko_KR',
    'fallback_locale': 'en_US',
    'supported_locales': ['ko_KR', 'en_US'],
    'timezone': 'Asia/Seoul',
    'date_format': {
        'ko_KR': '%Y년 %m월 %d일',
        'en_US': '%B %d, %Y'
    },
    'time_format': {
        'ko_KR': '%H시 %M분',
        'en_US': '%I:%M %p'
    },
    'number_format': {
        'decimal_separator': '.',
        'thousands_separator': ',',
        'decimal_places': 2
    },
    'currency': {
        'ko_KR': 'KRW',
        'en_US': 'USD'
    }
}

# ============================================================================
# 📧 알림 설정
# ============================================================================

NOTIFICATION_CONFIG = {
    'channels': {
        'in_app': {
            'enabled': True,
            'max_queue': 100,
            'persist': True
        },
        'email': {
            'enabled': False,  # 향후 구현
            'smtp_server': os.getenv('SMTP_SERVER', ''),
            'smtp_port': 587,
            'use_tls': True
        },
        'desktop': {
            'enabled': True,
            'permission_required': True,
            'sound': True,
            'icon': PROJECT_ROOT / 'assets' / 'icon.png'
        }
    },
    'types': {
        'system': {'priority': 'high', 'persistent': True},
        'experiment': {'priority': 'medium', 'persistent': True},
        'collaboration': {'priority': 'medium', 'persistent': False},
        'update': {'priority': 'low', 'persistent': True},
        'tip': {'priority': 'low', 'persistent': False}
    },
    'preferences': {
        'do_not_disturb': False,
        'quiet_hours': {'enabled': False, 'start': '22:00', 'end': '08:00'},
        'batch_notifications': True,
        'batch_interval': timedelta(minutes=5)
    }
}

# ============================================================================
# 📈 분석 및 리포팅 설정
# ============================================================================

ANALYTICS_CONFIG = {
    'tracking': {
        'enabled': False,  # 프라이버시 우선
        'anonymous': True,
        'events': ['app_start', 'experiment_created', 'analysis_completed'],
        'exclude_sensitive': True
    },
    'reports': {
        'formats': ['pdf', 'html', 'docx', 'pptx'],
        'templates': {
            'academic': 'APA 스타일 학술 보고서',
            'industry': '산업체 기술 보고서',
            'summary': '요약 리포트',
            'presentation': '프레젠테이션'
        },
        'include_code': True,
        'include_data': False,  # 보안상 기본값 False
        'watermark': False
    },
    'export': {
        'compression': True,
        'encryption': False,
        'metadata': True,
        'versioning': True
    }
}

# ============================================================================
# 🛠️ 개발자 설정
# ============================================================================

DEVELOPER_CONFIG = {
    'debug': {
        'enabled': DEBUG,
        'show_stats': True,
        'show_queries': False,
        'show_timings': True,
        'save_logs': True
    },
    'logging': {
        'level': 'DEBUG' if DEBUG else 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file': LOGS_DIR / f'app_{ENV}.log',
        'max_size_mb': 10,
        'backup_count': 5,
        'console': True
    },
    'profiling': {
        'enabled': False,
        'cpu': True,
        'memory': True,
        'save_reports': True
    },
    'testing': {
        'mock_data': IS_DEVELOPMENT,
        'test_accounts': IS_DEVELOPMENT,
        'bypass_auth': False,  # 절대 True로 하지 말 것
        'fixtures_dir': PROJECT_ROOT / 'tests' / 'fixtures'
    }
}

# ============================================================================
# 🎯 기능 플래그
# ============================================================================

FEATURE_FLAGS = {
    'new_ui': True,
    'ai_chat': True,
    'collaboration': True,
    'marketplace': True,
    'advanced_analytics': True,
    'custom_modules': True,
    'cloud_sync': False,  # 기본적으로 비활성화
    'beta_features': IS_DEVELOPMENT,
    'experimental': {
        'ar_visualization': False,
        'voice_commands': False,
        'ai_autopilot': False
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
    # 1. 환경변수에서 먼저 확인
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
    
    # 2. 설정 딕셔너리에서 찾기
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
    설정 검증 및 경고 반환
    
    Returns:
        경고 메시지 리스트 [(레벨, 메시지), ...]
    """
    warnings = []
    
    # 디렉토리 생성 시도
    for dir_name, dir_path in [
        ('Data', DATA_DIR),
        ('Logs', LOGS_DIR),
        ('Temp', TEMP_DIR),
        ('Cache', CACHE_DIR),
        ('Database', DB_DIR),
        ('Modules', MODULES_DIR),
        ('Backup', BACKUP_DIR)
    ]:
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            warnings.append(('ERROR', f'{dir_name} 디렉토리 생성 실패: {e}'))
    
    # 보안 키 확인
    if IS_PRODUCTION:
        if SECURITY_CONFIG['session']['secret_key'] == 'dev-secret-key-change-in-production':
            warnings.append(('CRITICAL', '프로덕션 환경에서 기본 세션 키 사용 중!'))
        if SECURITY_CONFIG['jwt']['secret_key'] == 'dev-jwt-secret':
            warnings.append(('CRITICAL', '프로덕션 환경에서 기본 JWT 키 사용 중!'))
    
    # 파일 크기 제한 확인
    if FILE_CONFIG['upload']['max_size_mb'] > 500:
        warnings.append(('WARNING', '파일 업로드 크기 제한이 500MB 초과'))
    
    # AI 엔진 확인
    required_engines = [k for k, v in AI_ENGINES.items() if v.get('required')]
    if not required_engines:
        warnings.append(('WARNING', '필수 AI 엔진이 설정되지 않음'))
    
    return warnings

def save_config_snapshot(filename: Optional[str] = None) -> Path:
    """
    현재 설정 스냅샷 저장
    
    Args:
        filename: 저장할 파일명 (기본값: config_snapshot_TIMESTAMP.json)
        
    Returns:
        저장된 파일 경로
    """
    import json
    from datetime import datetime
    
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'config_snapshot_{timestamp}.json'
    
    snapshot_path = BACKUP_DIR / 'configs' / filename
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 설정 수집
    config_data = {
        'timestamp': datetime.now().isoformat(),
        'version': APP_INFO['version'],
        'environment': ENV,
        'system': SYSTEM_INFO,
        'settings': {
            'app_info': APP_INFO,
            'ai_engines': AI_ENGINES,
            'security': {k: v for k, v in SECURITY_CONFIG.items() if k != 'jwt'},
            'ui': UI_CONFIG,
            'experiment': EXPERIMENT_CONFIG,
            'features': FEATURE_FLAGS
        }
    }
    
    with open(snapshot_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    return snapshot_path

# ============================================================================
# 🚀 초기화 및 검증
# ============================================================================

# 앱 시작 시 자동 실행
if __name__ != "__main__":
    # 설정 검증
    config_warnings = validate_config()
    
    # 경고 출력
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
    
    # 개발 환경에서 설정 스냅샷 자동 저장
    if IS_DEVELOPMENT and not IS_FROZEN:
        try:
            snapshot_path = save_config_snapshot()
            print(f"설정 스냅샷 저장: {snapshot_path}")
        except Exception as e:
            print(f"설정 스냅샷 저장 실패: {e}")

# ============================================================================
# 📤 Public API
# ============================================================================

__all__ = [
    # 환경 정보
    'PROJECT_ROOT', 'DATA_DIR', 'ENV', 'IS_PRODUCTION', 'IS_DEVELOPMENT',
    'DEBUG', 'SYSTEM_INFO',
    
    # 앱 정보
    'APP_INFO',
    
    # 주요 설정
    'AI_ENGINES', 'AI_EXPLANATION_CONFIG', 'SQLITE_CONFIG', 'GOOGLE_SHEETS_CONFIG',
    'SECURITY_CONFIG', 'UI_CONFIG', 'FILE_CONFIG', 'PERFORMANCE_CONFIG',
    'EXPERIMENT_CONFIG', 'MODULE_CONFIG', 'SYNC_CONFIG', 'UPDATE_CONFIG',
    'LOCALIZATION_CONFIG', 'NOTIFICATION_CONFIG', 'ANALYTICS_CONFIG',
    'DEVELOPER_CONFIG', 'FEATURE_FLAGS',
    
    # 유틸리티 함수
    'get_config', 'validate_config', 'save_config_snapshot'
]
