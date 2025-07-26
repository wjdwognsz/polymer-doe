"""
⚙️ Universal DOE Platform - 앱 전역 설정
================================================================================
데스크톱 애플리케이션에 최적화된 중앙 설정 관리 시스템
오프라인 우선 설계, 6개 AI 엔진 통합, 다중 형식 프로토콜 추출 지원
================================================================================
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Literal, Set
from datetime import timedelta
from enum import Enum
import platform
import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache

# ============================================================================
# 🔧 환경 설정
# ============================================================================

# PyInstaller 빌드 대응 경로 처리
if getattr(sys, 'frozen', False):
    # PyInstaller로 패키징된 경우
    PROJECT_ROOT = Path(sys._MEIPASS)
    DATA_DIR = Path(sys.executable).parent / 'data'
    IS_FROZEN = True
else:
    # 개발 환경
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / 'data'
    IS_FROZEN = False

# 주요 디렉토리 설정
CONFIG_DIR = PROJECT_ROOT / "config"
LOGS_DIR = DATA_DIR / "logs"
TEMP_DIR = DATA_DIR / "temp"
CACHE_DIR = DATA_DIR / "cache"
DB_DIR = DATA_DIR / "db"
MODULES_DIR = DATA_DIR / "modules"
BACKUP_DIR = DATA_DIR / "backups"
EXPORTS_DIR = DATA_DIR / "exports"
PROTOCOLS_DIR = DATA_DIR / "protocols"

# 환경 변수
ENV = os.getenv('STREAMLIT_ENV', 'development')
IS_PRODUCTION = ENV == 'production'
IS_STAGING = ENV == 'staging'
IS_DEVELOPMENT = ENV == 'development'
IS_TEST = ENV == 'test'
IS_DESKTOP = IS_FROZEN or os.getenv('DESKTOP_MODE', 'false').lower() == 'true'

# 디버그 모드
DEBUG = os.getenv('DEBUG', str(not IS_PRODUCTION)).lower() in ('true', '1', 'yes')

# 시스템 정보
SYSTEM_INFO = {
    'platform': platform.system(),  # Windows, Darwin, Linux
    'platform_version': platform.version(),
    'python_version': sys.version,
    'python_version_info': sys.version_info,
    'is_windows': platform.system() == 'Windows',
    'is_macos': platform.system() == 'Darwin',
    'is_linux': platform.system() == 'Linux',
    'is_64bit': sys.maxsize > 2**32,
    'cpu_count': os.cpu_count() or 1,
    'machine': platform.machine(),
    'processor': platform.processor()
}

# ============================================================================
# 📱 앱 기본 정보
# ============================================================================

APP_INFO = {
    'name': 'Universal DOE Platform',
    'short_name': 'UniversalDOE',
    'version': '2.0.0',
    'build': '2024.12.01',
    'description': '모든 연구자를 위한 AI 기반 실험 설계 플랫폼',
    'tagline': 'Design, Analyze, Optimize - All in One',
    'author': 'DOE Team',
    'email': 'support@universaldoe.com',
    'website': 'https://universaldoe.com',
    'github': 'https://github.com/universaldoe/platform',
    'documentation': 'https://docs.universaldoe.com',
    'license': 'MIT',
    'copyright': '© 2024 DOE Team. All rights reserved.',
    'python_required': '>=3.8',
    'update_check_url': 'https://api.universaldoe.com/updates/check',
    'telemetry_enabled': False  # 프라이버시 우선
}

# ============================================================================
# 🤖 AI 엔진 설정 (6개 통합)
# ============================================================================

class AIProvider(Enum):
    """AI 제공자 열거형"""
    GOOGLE_GEMINI = "google_gemini"
    XAI_GROK = "xai_grok"
    GROQ = "groq"
    DEEPSEEK = "deepseek"
    SAMBANOVA = "sambanova"
    HUGGINGFACE = "huggingface"

@dataclass
class AIEngineConfig:
    """AI 엔진 설정 데이터 클래스"""
    name: str
    provider: str
    models: Union[List[str], Dict[str, str]]
    default_model: str
    api_key_env: str
    api_key_secret: str
    base_url: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: Optional[int] = None
    rate_limit: int = 60
    free_tier: bool = False
    required: bool = False
    capabilities: List[str] = None
    best_for: List[str] = None
    docs_url: str = ""
    description: str = ""
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = ['text']
        if self.best_for is None:
            self.best_for = ['general']

AI_ENGINES = {
    AIProvider.GOOGLE_GEMINI: AIEngineConfig(
        name='Google Gemini 2.0 Flash',
        provider='Google',
        models=['gemini-2.0-flash-exp', 'gemini-1.5-pro', 'gemini-1.5-flash'],
        default_model='gemini-2.0-flash-exp',
        api_key_env='GOOGLE_GEMINI_API_KEY',
        api_key_secret='google_gemini_key',
        base_url='https://generativelanguage.googleapis.com/v1beta',
        max_tokens=8192,
        temperature=0.7,
        rate_limit=60,
        free_tier=True,
        required=True,
        capabilities=['text', 'code', 'vision', 'function_calling', 'multimodal'],
        best_for=['general', 'multimodal', 'reasoning', 'creative'],
        docs_url='https://makersuite.google.com/app/apikey',
        description='가장 빠르고 다재다능한 AI, 무료 티어 제공'
    ),
    
    AIProvider.XAI_GROK: AIEngineConfig(
        name='xAI Grok',
        provider='xAI',
        models=['grok-beta', 'grok-2-mini'],
        default_model='grok-beta',
        api_key_env='XAI_API_KEY',
        api_key_secret='xai_api_key',
        base_url='https://api.x.ai/v1',
        max_tokens=4096,
        temperature=0.7,
        rate_limit=30,
        free_tier=False,
        required=False,
        capabilities=['text', 'code', 'humor', 'real_time'],
        best_for=['creative', 'unconventional', 'current_events'],
        docs_url='https://x.ai/api',
        description='실시간 정보와 창의적 사고, 유머러스한 응답'
    ),
    
    AIProvider.GROQ: AIEngineConfig(
        name='Groq (초고속 추론)',
        provider='Groq',
        models=['llama-3.1-70b-versatile', 'llama-3.1-8b-instant', 'mixtral-8x7b-32768'],
        default_model='llama-3.1-70b-versatile',
        api_key_env='GROQ_API_KEY',
        api_key_secret='groq_key',
        base_url='https://api.groq.com/openai/v1',
        max_tokens=32768,
        temperature=0.7,
        rate_limit=100,
        free_tier=True,
        required=False,
        capabilities=['text', 'code', 'speed', 'analysis'],
        best_for=['fast_inference', 'real_time', 'large_context'],
        docs_url='https://console.groq.com',
        description='LPU 기반 초고속 추론, 무료 티어 제공'
    ),
    
    AIProvider.DEEPSEEK: AIEngineConfig(
        name='DeepSeek (코드/수식)',
        provider='DeepSeek',
        models=['deepseek-chat', 'deepseek-coder'],
        default_model='deepseek-chat',
        api_key_env='DEEPSEEK_API_KEY',
        api_key_secret='deepseek_key',
        base_url='https://api.deepseek.com/v1',
        max_tokens=16384,
        temperature=0.3,
        rate_limit=60,
        free_tier=False,
        required=False,
        capabilities=['code', 'math', 'technical', 'reasoning'],
        best_for=['code_generation', 'algorithms', 'formulas', 'technical_docs'],
        docs_url='https://platform.deepseek.com',
        description='코드와 수학적 추론에 특화된 전문 AI'
    ),
    
    AIProvider.SAMBANOVA: AIEngineConfig(
        name='SambaNova (대규모 모델)',
        provider='SambaNova',
        models=['Meta-Llama-3.1-405B-Instruct', 'Meta-Llama-3.1-70B-Instruct'],
        default_model='Meta-Llama-3.1-70B-Instruct',
        api_key_env='SAMBANOVA_API_KEY',
        api_key_secret='sambanova_key',
        base_url='https://api.sambanova.ai/v1',
        max_tokens=4096,
        temperature=0.7,
        rate_limit=10,
        free_tier=True,
        required=False,
        capabilities=['text', 'reasoning', 'large_context', 'enterprise'],
        best_for=['complex_analysis', 'research', 'long_documents'],
        docs_url='https://cloud.sambanova.ai',
        description='405B 파라미터 대규모 모델, 무료 클라우드 제공'
    ),
    
    AIProvider.HUGGINGFACE: AIEngineConfig(
        name='HuggingFace (특수 모델)',
        provider='HuggingFace',
        models={
            'general': 'meta-llama/Llama-3.2-11B-Vision-Instruct',
            'chemistry': 'laituan245/ChemLLM-7B-Chat',
            'materials': 'm3rg-iitd/MatSciBERT',
            'biology': 'microsoft/BioGPT-Large',
            'polymer': 'NREL/PolymerGPT'
        },
        default_model='meta-llama/Llama-3.2-11B-Vision-Instruct',
        api_key_env='HUGGINGFACE_TOKEN',
        api_key_secret='huggingface_token',
        base_url='https://api-inference.huggingface.co/models',
        max_tokens=2048,
        temperature=0.7,
        rate_limit=100,
        free_tier=True,
        required=False,
        capabilities=['specialized', 'domain_specific', 'fine_tuning', 'custom'],
        best_for=['chemistry', 'materials', 'biology', 'specialized_tasks'],
        docs_url='https://huggingface.co/settings/tokens',
        description='도메인 특화 모델 허브, 커스터마이징 가능'
    )
}

# ============================================================================
# 🧠 AI 설명 상세도 제어 (필수 구현)
# ============================================================================

AI_EXPLANATION_CONFIG = {
    'modes': {
        'auto': {
            'name': '자동 조정',
            'description': '사용자 레벨에 따라 자동으로 상세도 조정',
            'icon': '🤖'
        },
        'always_detailed': {
            'name': '항상 상세히',
            'description': '모든 AI 응답에 상세한 설명 포함',
            'icon': '📚'
        },
        'always_simple': {
            'name': '항상 간단히',
            'description': '핵심 내용만 간결하게 제공',
            'icon': '📝'
        },
        'custom': {
            'name': '사용자 맞춤',
            'description': '세부 항목별로 직접 설정',
            'icon': '⚙️'
        }
    },
    'default_mode': 'auto',
    'components': {
        'reasoning': {
            'label': '추론 과정',
            'default': True,
            'description': 'AI가 왜 이런 결론에 도달했는지 단계별 설명',
            'icon': '🔍',
            'example': '온도가 높을수록 반응속도가 빨라지는 이유는...'
        },
        'alternatives': {
            'label': '대안 검토',
            'default': True,
            'description': '다른 가능한 옵션들과 각각의 장단점 비교',
            'icon': '🔄',
            'example': '다른 설계 방법으로는 Box-Behnken, CCD가 있으며...'
        },
        'theory': {
            'label': '이론적 배경',
            'default': True,
            'description': '과학적 원리와 학술적 근거',
            'icon': '📖',
            'example': 'Arrhenius 방정식에 따르면...'
        },
        'confidence': {
            'label': '신뢰도 평가',
            'default': True,
            'description': '추천의 확실성 정도와 불확실성 요인',
            'icon': '📊',
            'example': '이 추천의 신뢰도는 85%이며, 주의점은...'
        },
        'limitations': {
            'label': '제약사항',
            'default': True,
            'description': '한계점과 주의해야 할 사항',
            'icon': '⚠️',
            'example': '이 방법은 비선형성이 강한 경우 정확도가 떨어질 수 있습니다'
        },
        'references': {
            'label': '참고문헌',
            'default': False,
            'description': '관련 논문이나 자료 출처',
            'icon': '📑',
            'example': '[1] Montgomery, D.C. (2017). Design and Analysis of Experiments...'
        }
    },
    'ui_settings': {
        'toggle_position': 'top_right',
        'toggle_shortcut': 'Ctrl+D',
        'animation_enabled': True,
        'animation_duration': 300,  # ms
        'persistent_state': True,
        'show_examples': True
    },
    'auto_detection': {
        'beginner_keywords': ['처음', '초보', '기초', '쉽게', '간단히', '뭔가요'],
        'expert_keywords': ['전문', '고급', '상세', '깊이', '구체적', '정확히'],
        'context_analysis': True,
        'learning_enabled': True  # 사용자 선호도 학습
    },
    'session_settings': {
        'remember_preference': True,
        'sync_across_pages': True,
        'export_with_reports': True
    }
}

# ============================================================================
# 📂 파일 처리 설정 (다중 형식 지원 - v9.1 확장)
# ============================================================================

FILE_CONFIG = {
    'upload': {
        'max_size_mb': 50,
        'max_files': 10,
        'chunk_size': 1024 * 1024,  # 1MB chunks
        'timeout': 300,  # 5분
        'parallel_upload': True,
        'resume_enabled': True
    },
    'supported_formats': {
        'documents': {
            'pdf': {
                'mime': ['application/pdf'],
                'priority': 1,
                'parser': 'pdfplumber',
                'ocr_enabled': True
            },
            'docx': {
                'mime': ['application/vnd.openxmlformats-officedocument.wordprocessingml.document'],
                'priority': 2,
                'parser': 'python-docx'
            },
            'doc': {
                'mime': ['application/msword'],
                'priority': 3,
                'parser': 'python-docx',
                'convert_to': 'docx'
            },
            'txt': {
                'mime': ['text/plain'],
                'priority': 4,
                'parser': 'native',
                'encoding_detection': True
            },
            'rtf': {
                'mime': ['application/rtf', 'text/rtf'],
                'priority': 5,
                'parser': 'striprtf'
            },
            'odt': {
                'mime': ['application/vnd.oasis.opendocument.text'],
                'priority': 6,
                'parser': 'odfpy'
            }
        },
        'markup': {
            'html': {
                'mime': ['text/html'],
                'priority': 1,
                'parser': 'beautifulsoup4',
                'extract_method': 'readability'
            },
            'xml': {
                'mime': ['application/xml', 'text/xml'],
                'priority': 2,
                'parser': 'lxml',
                'schema_validation': True
            },
            'md': {
                'mime': ['text/markdown', 'text/x-markdown'],
                'priority': 3,
                'parser': 'markdown',
                'extensions': ['tables', 'fenced_code']
            }
        },
        'data': {
            'csv': {
                'mime': ['text/csv'],
                'priority': 1,
                'parser': 'pandas',
                'encoding_detection': True
            },
            'xlsx': {
                'mime': ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'],
                'priority': 2,
                'parser': 'openpyxl'
            },
            'json': {
                'mime': ['application/json'],
                'priority': 3,
                'parser': 'native',
                'schema_validation': True
            },
            'parquet': {
                'mime': ['application/octet-stream'],
                'priority': 4,
                'parser': 'pyarrow'
            }
        },
        'images': {
            'png': {'mime': ['image/png'], 'ocr_enabled': True},
            'jpg': {'mime': ['image/jpeg'], 'ocr_enabled': True},
            'tiff': {'mime': ['image/tiff'], 'ocr_enabled': True},
            'webp': {'mime': ['image/webp'], 'ocr_enabled': True}
        }
    },
    'encoding': {
        'auto_detect': True,
        'detection_library': 'chardet',
        'fallback': 'utf-8',
        'supported': [
            'utf-8', 'utf-16', 'utf-32',
            'latin-1', 'iso-8859-1', 'windows-1252',
            'gb2312', 'gbk', 'gb18030',  # 중국어
            'shift-jis', 'euc-jp', 'iso-2022-jp',  # 일본어
            'euc-kr', 'iso-2022-kr'  # 한국어
        ],
        'bom_handling': 'remove',
        'normalization': 'NFC'  # Unicode normalization
    },
    'processing': {
        'parallel': True,
        'max_workers': min(4, SYSTEM_INFO['cpu_count']),
        'batch_size': 5,
        'memory_limit_mb': 500,
        'temp_cleanup': True,
        'cache_processed': True
    },
    'text_extraction': {
        'min_confidence': 0.8,
        'language_detection': True,
        'preserve_formatting': True,
        'extract_metadata': True,
        'clean_text': True,
        'remove_headers_footers': True
    }
}

# ============================================================================
# 🔍 프로토콜 추출 설정 (v9.1 확장)
# ============================================================================

PROTOCOL_EXTRACTION_CONFIG = {
    'methods': {
        'rule_based': {
            'enabled': True,
            'patterns': {
                'en': ['methods', 'experimental', 'procedure', 'protocol', 'materials and methods'],
                'ko': ['실험방법', '실험절차', '실험과정', '재료 및 방법', '프로토콜'],
                'ja': ['実験方法', '実験手順', 'プロトコル', '材料と方法'],
                'zh': ['实验方法', '实验步骤', '实验程序', '材料与方法']
            },
            'section_markers': ['\\d+\\.\\d+', '\\([a-z]\\)', '\\d+\\)', 'Step \\d+'],
            'confidence_weight': 0.3,
            'min_section_length': 100  # characters
        },
        'ml_based': {
            'enabled': True,
            'models': {
                'spacy': 'en_core_sci_lg',  # SciBERT based
                'transformers': 'allenai/scibert_scivocab_uncased'
            },
            'confidence_weight': 0.4,
            'use_context_window': True,
            'context_size': 512
        },
        'ai_based': {
            'enabled': True,
            'primary_engine': AIProvider.GOOGLE_GEMINI,
            'fallback_engines': [AIProvider.GROQ, AIProvider.DEEPSEEK],
            'confidence_weight': 0.3,
            'prompt_templates': 'config/prompts/protocol_extraction.yaml',
            'max_retries': 3
        }
    },
    'extraction': {
        'min_confidence': 0.7,
        'confidence_threshold_by_type': {
            'materials': 0.8,
            'conditions': 0.7,
            'procedure': 0.75,
            'measurements': 0.8
        },
        'max_text_length': 500000,  # characters
        'processing_timeout': 60,  # seconds
        'languages': ['en', 'ko', 'zh', 'ja'],
        'auto_translate': True,
        'cache_enabled': True,
        'cache_ttl': timedelta(days=30)
    },
    'output': {
        'formats': {
            'json': {
                'schema_version': '2.0',
                'include_confidence': True,
                'pretty_print': True
            },
            'yaml': {
                'include_comments': True,
                'preserve_order': True
            },
            'template': {
                'format': 'jinja2',
                'custom_templates': True
            },
            'csv': {
                'flatten_nested': True,
                'include_metadata': True
            }
        },
        'structure': {
            'title': str,
            'materials': List[Dict[str, Any]],
            'equipment': List[str],
            'conditions': Dict[str, Any],
            'procedure': List[Dict[str, Any]],
            'measurements': List[Dict[str, Any]],
            'safety': List[str],
            'notes': List[str],
            'references': List[str],
            'metadata': Dict[str, Any]
        }
    },
    'validation': {
        'required_fields': ['materials', 'procedure'],
        'material_validation': {
            'check_cas_numbers': True,
            'verify_units': True,
            'standard_names': True
        },
        'procedure_validation': {
            'check_completeness': True,
            'verify_sequence': True,
            'time_consistency': True
        }
    },
    'ocr': {
        'enabled': True,
        'engines': {
            'primary': 'tesseract',
            'fallback': ['easyocr', 'paddleocr']
        },
        'languages': ['eng', 'kor', 'chi_sim', 'jpn'],
        'preprocessing': {
            'deskew': True,
            'denoise': True,
            'contrast_enhancement': True,
            'resolution_upscale': True
        },
        'confidence_threshold': 0.8,
        'layout_analysis': True
    },
    'web_extraction': {
        'enabled': True,
        'timeout': 30,
        'user_agent': 'UniversalDOE/2.0 (Protocol Extractor)',
        'respect_robots_txt': True,
        'javascript_rendering': True,
        'ad_blocking': True,
        'cookie_handling': 'reject_all'
    }
}

# ============================================================================
# 💾 데이터베이스 설정
# ============================================================================

# SQLite 설정 (로컬/데스크톱)
SQLITE_CONFIG = {
    'database': {
        'path': DB_DIR / 'universaldoe.db',
        'backup_path': BACKUP_DIR / 'db',
        'schema_version': '2.0.0'
    },
    'connection': {
        'check_same_thread': False,
        'timeout': 30,
        'isolation_level': 'DEFERRED',
        'journal_mode': 'WAL',  # Write-Ahead Logging
        'synchronous': 'NORMAL',
        'cache_size': -64000,  # 64MB
        'temp_store': 'MEMORY',
        'mmap_size': 268435456,  # 256MB
        'foreign_keys': True
    },
    'pool': {
        'size': 5,
        'max_overflow': 10,
        'timeout': 30,
        'recycle': 3600,
        'pre_ping': True
    },
    'backup': {
        'enabled': True,
        'auto_backup': True,
        'interval': timedelta(hours=6),
        'keep_count': 7,
        'compress': True,
        'encrypt': IS_PRODUCTION,
        'incremental': True
    },
    'optimization': {
        'auto_vacuum': 'INCREMENTAL',
        'analyze_on_startup': True,
        'optimize_interval': timedelta(days=7),
        'index_stats': True
    }
}

# Google Sheets 설정 (온라인/협업)
GOOGLE_SHEETS_CONFIG = {
    'enabled': not IS_DESKTOP or os.getenv('ENABLE_CLOUD_SYNC', 'false').lower() == 'true',
    'authentication': {
        'method': 'oauth2',  # 'oauth2' or 'service_account'
        'scopes': [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive.file'
        ],
        'credentials_file': CONFIG_DIR / 'credentials' / 'google_service_account.json',
        'token_file': DATA_DIR / 'tokens' / 'google_token.json'
    },
    'spreadsheets': {
        'users': {
            'name': 'UniversalDOE_Users',
            'id_env': 'GOOGLE_SHEETS_USERS_ID',
            'structure': ['id', 'email', 'name', 'role', 'created_at', 'settings']
        },
        'projects': {
            'name': 'UniversalDOE_Projects',
            'id_env': 'GOOGLE_SHEETS_PROJECTS_ID',
            'structure': ['id', 'user_id', 'name', 'type', 'status', 'data', 'created_at']
        },
        'experiments': {
            'name': 'UniversalDOE_Experiments',
            'id_env': 'GOOGLE_SHEETS_EXPERIMENTS_ID',
            'structure': ['id', 'project_id', 'design_type', 'factors', 'results', 'analysis']
        },
        'shared': {
            'name': 'UniversalDOE_SharedData',
            'id_env': 'GOOGLE_SHEETS_SHARED_ID',
            'structure': ['id', 'type', 'data', 'permissions', 'created_by', 'shared_at']
        }
    },
    'sync': {
        'mode': 'manual',  # 'manual', 'auto', 'scheduled'
        'interval': timedelta(minutes=15),
        'batch_size': 100,
        'conflict_resolution': 'local_first',  # 'local_first', 'remote_first', 'newest', 'manual'
        'retry_count': 3,
        'retry_delay': 2.0
    },
    'performance': {
        'cache_enabled': True,
        'cache_ttl': 300,  # 5 minutes
        'batch_updates': True,
        'compression': True
    }
}

# ============================================================================
# 🔒 보안 설정
# ============================================================================

SECURITY_CONFIG = {
    'password': {
        'min_length': 8,
        'max_length': 128,
        'require_uppercase': True,
        'require_lowercase': True,
        'require_numbers': True,
        'require_special': True,
        'special_chars': '!@#$%^&*()_+-=[]{}|;:,.<>?',
        'prohibited_words': ['password', '12345678', 'qwerty', 'admin'],
        'history_check': 3,  # 최근 3개 비밀번호 재사용 금지
        'expiry_days': 90 if IS_PRODUCTION else 0,
        'complexity_score': 3  # 1-5 scale
    },
    'hashing': {
        'algorithm': 'bcrypt',
        'bcrypt_rounds': 12,
        'pepper': os.getenv('PASSWORD_PEPPER', '')  # Application-wide salt
    },
    'session': {
        'secret_key': os.getenv('SESSION_SECRET_KEY', 'dev-secret-key-change-in-production'),
        'algorithm': 'HS256',
        'timeout': timedelta(hours=24),
        'remember_me_duration': timedelta(days=30),
        'max_concurrent': 3,
        'regenerate_id': True,
        'cookie': {
            'name': 'universaldoe_session',
            'secure': IS_PRODUCTION,
            'httponly': True,
            'samesite': 'Lax',
            'max_age': 86400  # 24 hours
        }
    },
    'jwt': {
        'secret_key': os.getenv('JWT_SECRET_KEY', 'dev-jwt-secret'),
        'algorithm': 'HS256',
        'access_token_expire': timedelta(hours=1),
        'refresh_token_expire': timedelta(days=7),
        'issuer': 'universaldoe.com',
        'audience': 'universaldoe-api'
    },
    'auth': {
        'max_login_attempts': 5,
        'lockout_duration': timedelta(minutes=30),
        'captcha_after_attempts': 3,
        'enable_2fa': False,  # 향후 구현
        'oauth_providers': ['google'],  # 향후 확장
        'api_key_length': 32,
        'api_key_prefix': 'udoe_'
    },
    'encryption': {
        'algorithm': 'AES-256-GCM',
        'key_derivation': 'PBKDF2',
        'iterations': 100000,
        'salt_length': 32,
        'key_rotation_days': 90
    },
    'api': {
        'rate_limit': {
            'enabled': True,
            'requests_per_minute': 60,
            'requests_per_hour': 1000,
            'requests_per_day': 10000,
            'burst_size': 10
        },
        'cors': {
            'enabled': False,
            'origins': ['http://localhost:8501'],
            'methods': ['GET', 'POST'],
            'headers': ['Content-Type', 'Authorization']
        },
        'authentication': {
            'type': 'bearer',
            'header': 'Authorization',
            'scheme': 'Bearer'
        }
    },
    'audit': {
        'enabled': True,
        'log_login_attempts': True,
        'log_data_access': IS_PRODUCTION,
        'log_api_calls': True,
        'retention_days': 90
    }
}

# ============================================================================
# 🎨 UI/UX 설정
# ============================================================================

UI_CONFIG = {
    'theme': {
        'default': 'light',
        'available': ['light', 'dark', 'auto'],
        'allow_user_switching': True,
        'auto_detect_system': True,
        'colors': {
            'primary': '#7C3AED',  # 보라색
            'secondary': '#10B981',  # 녹색
            'accent': '#F59E0B',  # 주황색
            'success': '#10B981',
            'warning': '#F59E0B',
            'error': '#EF4444',
            'info': '#3B82F6'
        },
        'fonts': {
            'heading': 'Pretendard, Inter, system-ui, sans-serif',
            'body': 'Pretendard, Inter, system-ui, sans-serif',
            'code': 'JetBrains Mono, Consolas, monospace'
        }
    },
    'layout': {
        'max_width': 1200,
        'sidebar': {
            'default_state': 'expanded',
            'width': 300,
            'collapsible': True
        },
        'padding': 20,
        'spacing': {
            'xs': 4,
            'sm': 8,
            'md': 16,
            'lg': 24,
            'xl': 32
        },
        'breakpoints': {
            'mobile': 640,
            'tablet': 768,
            'desktop': 1024,
            'wide': 1280
        }
    },
    'components': {
        'buttons': {
            'size': 'medium',
            'rounded': 'md',
            'animation': True
        },
        'inputs': {
            'size': 'medium',
            'validation_feedback': 'immediate'
        },
        'tables': {
            'striped': True,
            'hover': True,
            'pagination': True,
            'page_size': 20
        },
        'charts': {
            'engine': 'plotly',
            'theme': 'streamlit',
            'interactive': True
        }
    },
    'animations': {
        'enabled': True,
        'duration': {
            'fast': 150,
            'normal': 300,
            'slow': 500
        },
        'easing': 'ease-out'
    },
    'accessibility': {
        'high_contrast': False,
        'reduce_motion': False,
        'keyboard_navigation': True,
        'screen_reader_support': True,
        'focus_indicators': True,
        'alt_text_required': True
    },
    'notifications': {
        'position': 'top-right',
        'duration': 5000,
        'max_stack': 3,
        'animations': True,
        'sounds': False
    }
}

# ============================================================================
# 🧪 실험 설계 설정
# ============================================================================

EXPERIMENT_CONFIG = {
    'design_types': {
        'factorial': {
            'name': '완전/부분 요인설계',
            'description': '모든 요인 조합을 체계적으로 탐색',
            'min_factors': 2,
            'max_factors': 10,
            'levels': [2, 3, 4, 5],
            'aliases_allowed': True
        },
        'response_surface': {
            'name': '반응표면설계',
            'description': '2차 모델 fitting을 위한 설계',
            'types': {
                'ccd': '중심합성설계',
                'box-behnken': 'Box-Behnken 설계',
                'face-centered': '면중심 설계'
            },
            'min_factors': 2,
            'max_factors': 6,
            'alpha_options': ['rotatable', 'orthogonal', 'face-centered', 'custom']
        },
        'mixture': {
            'name': '혼합물 설계',
            'description': '성분 비율 최적화',
            'types': {
                'simplex-lattice': '심플렉스 격자',
                'simplex-centroid': '심플렉스 중심',
                'extreme-vertices': '극점 설계',
                'optimal': '최적 혼합물 설계'
            },
            'min_components': 3,
            'max_components': 10,
            'constraints_allowed': True
        },
        'optimal': {
            'name': '최적 설계',
            'description': '통계적 기준에 따른 최적 설계',
            'criteria': {
                'D-optimal': '행렬식 최대화',
                'I-optimal': '예측 분산 최소화',
                'A-optimal': '평균 분산 최소화',
                'G-optimal': '최대 예측 분산 최소화'
            },
            'custom_model': True,
            'constraints_allowed': True
        },
        'screening': {
            'name': '스크리닝 설계',
            'description': '중요 요인 선별',
            'types': {
                'plackett-burman': 'Plackett-Burman',
                'definitive-screening': '확정적 스크리닝',
                'fractional-factorial': '부분요인설계'
            },
            'min_factors': 3,
            'max_factors': 50
        },
        'custom': {
            'name': '사용자 정의',
            'description': '직접 설계 입력',
            'import_formats': ['csv', 'excel', 'json'],
            'validation': True
        }
    },
    'factor_settings': {
        'types': ['continuous', 'discrete', 'categorical'],
        'max_factors': 20,
        'max_levels': 10,
        'transformations': ['none', 'log', 'sqrt', 'inverse', 'box-cox'],
        'coding': ['orthogonal', 'normalized', 'actual']
    },
    'response_settings': {
        'max_responses': 10,
        'goals': ['maximize', 'minimize', 'target', 'in_range'],
        'weights_allowed': True,
        'transformations': ['none', 'log', 'sqrt', 'inverse', 'box-cox']
    },
    'constraints': {
        'linear': True,
        'nonlinear': True,
        'multivariate': True,
        'max_constraints': 20
    },
    'run_settings': {
        'min_runs': 3,
        'max_runs': 1000,
        'center_points': {
            'default': 3,
            'max': 20
        },
        'replication': {
            'allowed': True,
            'max_replicates': 10
        },
        'blocking': {
            'allowed': True,
            'max_blocks': 10
        },
        'randomization': {
            'default': True,
            'restricted': True
        }
    },
    'analysis': {
        'confidence_level': [0.90, 0.95, 0.99],
        'default_confidence': 0.95,
        'power_analysis': True,
        'default_power': 0.8,
        'alpha': 0.05,
        'multiple_comparison': ['bonferroni', 'tukey', 'scheffe', 'dunnett']
    },
    'optimization': {
        'methods': ['desirability', 'pareto', 'genetic_algorithm', 'response_surface'],
        'multi_objective': True,
        'robust_design': True,
        'monte_carlo_runs': 10000
    }
}

# ============================================================================
# 📦 모듈 시스템 설정
# ============================================================================

MODULE_CONFIG = {
    'paths': {
        'core_modules': PROJECT_ROOT / 'modules' / 'core',
        'user_modules': MODULES_DIR / 'user_modules',
        'marketplace_cache': MODULES_DIR / 'marketplace',
        'templates': PROJECT_ROOT / 'modules' / 'templates',
        'temp': TEMP_DIR / 'modules'
    },
    'core_modules': [
        'general_experiment',
        'polymer_synthesis',
        'polymer_processing',
        'formulation_optimization',
        'material_testing',
        'mixture_design',
        'robust_design'
    ],
    'validation': {
        'required_interface': [
            'get_info',
            'validate_inputs',
            'generate_design',
            'analyze_results',
            'export_data'
        ],
        'code_analysis': True,
        'security_scan': True,
        'performance_test': True,
        'max_size_mb': 10,
        'timeout_seconds': 30
    },
    'execution': {
        'sandbox': True,
        'resource_limits': {
            'cpu_percent': 50,
            'memory_mb': 512,
            'disk_mb': 100,
            'network': False
        },
        'allowed_imports': [
            'numpy', 'pandas', 'scipy', 'sklearn',
            'pyDOE2', 'statsmodels', 'math', 'statistics'
        ]
    },
    'marketplace': {
        'enabled': True,
        'api_endpoint': 'https://api.universaldoe.com/modules',
        'cdn_endpoint': 'https://cdn.universaldoe.com/modules',
        'update_check_interval': timedelta(hours=24),
        'featured_refresh': timedelta(hours=6),
        'categories': [
            'Chemistry',
            'Materials Science',
            'Biology',
            'Engineering',
            'Data Analysis',
            'Visualization',
            'Utilities'
        ],
        'quality_metrics': [
            'downloads',
            'rating',
            'last_updated',
            'compatibility',
            'documentation'
        ]
    },
    'development': {
        'templates_available': True,
        'hot_reload': IS_DEVELOPMENT,
        'debug_mode': DEBUG,
        'documentation_required': True,
        'example_data_required': True,
        'testing_framework': 'pytest'
    }
}

# ============================================================================
# 🔄 동기화 설정
# ============================================================================

SYNC_CONFIG = {
    'enabled': not IS_DESKTOP,
    'providers': {
        'google_drive': {
            'enabled': True,
            'folder_name': 'UniversalDOE_Sync',
            'file_types': ['projects', 'experiments', 'reports', 'modules'],
            'oauth_scopes': ['https://www.googleapis.com/auth/drive.file']
        },
        'dropbox': {
            'enabled': False,
            'app_key': os.getenv('DROPBOX_APP_KEY', ''),
            'folder_name': '/UniversalDOE'
        },
        'onedrive': {
            'enabled': False,
            'client_id': os.getenv('ONEDRIVE_CLIENT_ID', ''),
            'folder_name': 'UniversalDOE'
        },
        'github': {
            'enabled': True,
            'repo_name': 'universaldoe-sync',
            'branch': 'main',
            'token_env': 'GITHUB_TOKEN'
        }
    },
    'settings': {
        'mode': 'manual',  # 'manual', 'auto', 'scheduled'
        'direction': 'bidirectional',  # 'upload', 'download', 'bidirectional'
        'interval': timedelta(minutes=30),
        'bandwidth_limit_mbps': 10,
        'chunk_size_mb': 5,
        'parallel_transfers': 3
    },
    'conflict_resolution': {
        'strategy': 'manual',  # 'local_wins', 'remote_wins', 'newest', 'manual'
        'backup_conflicts': True,
        'merge_capable_types': ['json', 'yaml', 'csv']
    },
    'filters': {
        'include_patterns': ['*.json', '*.csv', '*.xlsx', '*.pdf'],
        'exclude_patterns': ['*.tmp', '*.log', '.DS_Store', 'Thumbs.db', '~*'],
        'max_file_size_mb': 100,
        'ignore_hidden': True
    },
    'compression': {
        'enabled': True,
        'algorithm': 'zlib',
        'level': 6,  # 1-9
        'min_size_kb': 100
    }
}

# ============================================================================
# 🔄 업데이트 설정
# ============================================================================

UPDATE_CONFIG = {
    'enabled': True,
    'check_on_startup': True,
    'check_interval': timedelta(days=1),
    'channels': {
        'stable': {
            'url': 'https://api.universaldoe.com/updates/stable',
            'description': '안정적인 정식 릴리즈'
        },
        'beta': {
            'url': 'https://api.universaldoe.com/updates/beta',
            'description': '새로운 기능 미리보기'
        },
        'nightly': {
            'url': 'https://api.universaldoe.com/updates/nightly',
            'description': '최신 개발 버전 (불안정)'
        }
    },
    'current_channel': 'stable' if IS_PRODUCTION else 'beta',
    'auto_download': False,
    'auto_install': False,
    'require_admin': SYSTEM_INFO['is_windows'],
    'verification': {
        'check_signature': True,
        'check_checksum': True,
        'public_key_url': 'https://api.universaldoe.com/updates/public_key'
    },
    'backup': {
        'before_update': True,
        'keep_count': 3,
        'include_data': True
    },
    'rollback': {
        'enabled': True,
        'max_versions': 3
    }
}

# ============================================================================
# 🌐 지역화 설정
# ============================================================================

LOCALIZATION_CONFIG = {
    'default_language': 'ko_KR',
    'fallback_language': 'en_US',
    'supported_languages': {
        'ko_KR': {
            'name': '한국어',
            'native_name': '한국어',
            'flag': '🇰🇷',
            'rtl': False,
            'date_format': 'YYYY년 MM월 DD일',
            'time_format': 'HH:mm:ss',
            'decimal_separator': '.',
            'thousands_separator': ',',
            'currency': 'KRW',
            'currency_symbol': '₩'
        },
        'en_US': {
            'name': 'English',
            'native_name': 'English',
            'flag': '🇺🇸',
            'rtl': False,
            'date_format': 'MM/DD/YYYY',
            'time_format': 'hh:mm:ss a',
            'decimal_separator': '.',
            'thousands_separator': ',',
            'currency': 'USD',
            'currency_symbol': '$'
        },
        'zh_CN': {
            'name': 'Chinese (Simplified)',
            'native_name': '简体中文',
            'flag': '🇨🇳',
            'rtl': False,
            'date_format': 'YYYY年MM月DD日',
            'time_format': 'HH:mm:ss',
            'decimal_separator': '.',
            'thousands_separator': ',',
            'currency': 'CNY',
            'currency_symbol': '¥'
        },
        'ja_JP': {
            'name': 'Japanese',
            'native_name': '日本語',
            'flag': '🇯🇵',
            'rtl': False,
            'date_format': 'YYYY年MM月DD日',
            'time_format': 'HH:mm:ss',
            'decimal_separator': '.',
            'thousands_separator': ',',
            'currency': 'JPY',
            'currency_symbol': '¥'
        }
    },
    'auto_detect': {
        'enabled': True,
        'sources': ['system', 'browser', 'ip_geolocation'],
        'cache_duration': timedelta(days=30)
    },
    'translation': {
        'provider': 'local',  # 'local', 'google', 'deepl'
        'cache_translations': True,
        'fallback_to_key': True
    },
    'content': {
        'scientific_notation': True,
        'unit_system': 'metric',  # 'metric', 'imperial', 'auto'
        'temperature_scale': 'celsius',  # 'celsius', 'fahrenheit', 'kelvin'
    }
}

# ============================================================================
# 📊 성능 및 모니터링 설정
# ============================================================================

PERFORMANCE_CONFIG = {
    'optimization': {
        'lazy_loading': True,
        'progressive_rendering': True,
        'virtual_scrolling': True,
        'debounce_ms': 300,
        'throttle_ms': 100
    },
    'caching': {
        'strategy': 'lru',  # 'lru', 'lfu', 'fifo'
        'backends': {
            'memory': {
                'enabled': True,
                'max_size_mb': 256,
                'ttl_seconds': 3600
            },
            'disk': {
                'enabled': True,
                'path': CACHE_DIR,
                'max_size_mb': 1024,
                'ttl_days': 7
            },
            'redis': {
                'enabled': False,
                'url': os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
                'key_prefix': 'udoe:'
            }
        },
        'invalidation': {
            'on_update': True,
            'on_logout': True,
            'scheduled': timedelta(hours=24)
        }
    },
    'resource_limits': {
        'max_memory_mb': 2048,
        'max_cpu_percent': 80,
        'max_file_handles': 1000,
        'max_threads': 20
    },
    'monitoring': {
        'enabled': not IS_PRODUCTION,  # 프라이버시 보호
        'metrics': [
            'response_time',
            'memory_usage',
            'cpu_usage',
            'error_rate',
            'active_users'
        ],
        'export_interval': timedelta(minutes=5),
        'retention_days': 30
    },
    'profiling': {
        'enabled': DEBUG,
        'sampling_rate': 0.1,
        'profile_sql': True,
        'profile_memory': True
    }
}

# ============================================================================
# 🛠️ 개발자 설정
# ============================================================================

DEVELOPER_CONFIG = {
    'debug': {
        'enabled': DEBUG,
        'verbose': IS_DEVELOPMENT,
        'show_internal_errors': not IS_PRODUCTION,
        'show_sql_queries': DEBUG,
        'show_api_calls': DEBUG,
        'save_debug_files': IS_DEVELOPMENT
    },
    'logging': {
        'level': 'DEBUG' if DEBUG else 'INFO',
        'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        'date_format': '%Y-%m-%d %H:%M:%S',
        'handlers': {
            'console': {
                'enabled': True,
                'level': 'DEBUG' if DEBUG else 'INFO',
                'colorize': True
            },
            'file': {
                'enabled': True,
                'path': LOGS_DIR / f'app_{ENV}.log',
                'max_size_mb': 10,
                'backup_count': 5,
                'encoding': 'utf-8'
            },
            'syslog': {
                'enabled': IS_PRODUCTION,
                'host': 'localhost',
                'port': 514
            }
        },
        'loggers': {
            'streamlit': 'WARNING',
            'urllib3': 'WARNING',
            'matplotlib': 'WARNING'
        }
    },
    'testing': {
        'fixtures_path': PROJECT_ROOT / 'tests' / 'fixtures',
        'mock_external_apis': True,
        'test_database': ':memory:',
        'coverage_threshold': 80
    },
    'tools': {
        'api_explorer': IS_DEVELOPMENT,
        'db_browser': IS_DEVELOPMENT,
        'log_viewer': True,
        'performance_monitor': DEBUG,
        'memory_profiler': False
    }
}

# ============================================================================
# 🎯 기능 플래그
# ============================================================================

FEATURE_FLAGS = {
    # 핵심 기능
    'core': {
        'offline_mode': True,
        'ai_assistance': True,
        'multi_ai_engines': True,
        'protocol_extraction': True,
        'ai_explanation_control': True,  # 필수 구현
        'collaboration': True,
        'cloud_sync': not IS_DESKTOP,
        'auto_save': True
    },
    
    # 실험적 기능
    'experimental': {
        'voice_commands': False,
        'ar_visualization': False,
        'ai_autopilot': False,
        'blockchain_verification': False,
        'quantum_optimization': False
    },
    
    # 베타 기능
    'beta': {
        'new_ui_2024': True,
        'advanced_ml_analysis': IS_DEVELOPMENT or IS_STAGING,
        'real_time_collaboration': False,
        'jupyter_integration': False,
        'custom_ai_models': False
    },
    
    # 프리미엄 기능
    'premium': {
        'unlimited_projects': False,
        'priority_processing': False,
        'advanced_export_formats': False,
        'white_label': False,
        'dedicated_support': False
    },
    
    # 플랫폼별 기능
    'platform': {
        'desktop': {
            'system_tray': IS_DESKTOP,
            'global_hotkeys': IS_DESKTOP,
            'file_associations': IS_DESKTOP,
            'auto_update': IS_DESKTOP
        },
        'web': {
            'pwa': not IS_DESKTOP,
            'web_share': not IS_DESKTOP,
            'notifications': True,
            'webgl': True
        }
    },
    
    # 개발자 기능
    'developer': {
        'debug_panel': DEBUG,
        'performance_overlay': DEBUG,
        'feature_toggle_ui': IS_DEVELOPMENT,
        'experimental_api': IS_DEVELOPMENT
    }
}

# ============================================================================
# 🔧 유틸리티 함수
# ============================================================================

@lru_cache(maxsize=128)
def get_config(key: str, default: Any = None) -> Any:
    """
    점 표기법으로 중첩된 설정값 가져오기 (캐시 적용)
    
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
        elif hasattr(config, k):
            config = getattr(config, k)
        else:
            return default
    
    return config

def set_config(key: str, value: Any) -> None:
    """
    런타임에 설정값 변경
    
    Args:
        key: 설정 키
        value: 새 값
    """
    keys = key.split('.')
    config = globals()
    
    for k in keys[:-1]:
        if isinstance(config, dict):
            config = config.get(k, {})
        else:
            raise KeyError(f"Invalid config path: {key}")
    
    if isinstance(config, dict):
        config[keys[-1]] = value
        # 캐시 무효화
        get_config.cache_clear()

def validate_config() -> Tuple[bool, List[str]]:
    """
    설정 유효성 검증
    
    Returns:
        (성공 여부, 경고/오류 메시지 리스트)
    """
    messages = []
    is_valid = True
    
    # 필수 디렉토리 생성
    required_dirs = [
        DATA_DIR, LOGS_DIR, TEMP_DIR, CACHE_DIR, 
        DB_DIR, MODULES_DIR, BACKUP_DIR, EXPORTS_DIR, PROTOCOLS_DIR  # PROTOCOLS_DIR 추가
    ]
    
    for dir_path in required_dirs:
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            messages.append(f"ERROR: {dir_path} 디렉토리 생성 실패: {e}")
            is_valid = False
    
    # 보안 키 확인 (프로덕션)
    if IS_PRODUCTION:
        if SECURITY_CONFIG['session']['secret_key'] == 'dev-secret-key-change-in-production':
            messages.append("CRITICAL: 프로덕션 환경에서 기본 세션 키 사용 중!")
            is_valid = False
        if SECURITY_CONFIG['jwt']['secret_key'] == 'dev-jwt-secret':
            messages.append("CRITICAL: 프로덕션 환경에서 기본 JWT 키 사용 중!")
            is_valid = False
    
    # 필수 AI 엔진 확인
    required_engines = [
        engine for engine, config in AI_ENGINES.items() 
        if config.required
    ]
    if not required_engines:
        messages.append("WARNING: 필수 AI 엔진이 설정되지 않음")
    
    # Python 버전 확인
    min_version = tuple(map(int, APP_INFO['python_required'].replace('>=', '').split('.')))
    current_version = sys.version_info[:2]
    if current_version < min_version:
        messages.append(f"ERROR: Python {'.'.join(map(str, min_version))} 이상 필요 (현재: {'.'.join(map(str, current_version))})")
        is_valid = False
    
    # 파일 시스템 권한 확인
    test_file = TEMP_DIR / '.test_write'
    try:
        test_file.write_text('test')
        test_file.unlink()
    except Exception as e:
        messages.append(f"ERROR: 파일 시스템 쓰기 권한 없음: {e}")
        is_valid = False
    
    # 디스크 공간 확인
    try:
        import shutil
        stat = shutil.disk_usage(DATA_DIR)
        free_gb = stat.free / (1024**3)
        if free_gb < 1:
            messages.append(f"WARNING: 디스크 공간 부족 ({free_gb:.1f}GB 남음)")
    except:
        pass
    
    return is_valid, messages

def save_config_snapshot(filename: Optional[str] = None) -> Path:
    """
    현재 설정을 JSON 파일로 저장 (민감 정보 제외)
    
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
    
    # 직렬화 가능하고 민감하지 않은 설정만 수집
    config_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'version': APP_INFO['version'],
            'environment': ENV,
            'system': SYSTEM_INFO
        },
        'settings': {
            'app_info': APP_INFO,
            'ai_engines': {
                k.value: {
                    'name': v.name,
                    'provider': v.provider,
                    'capabilities': v.capabilities,
                    'required': v.required
                } for k, v in AI_ENGINES.items()
            },
            'ai_explanation': AI_EXPLANATION_CONFIG,
            'file_config': FILE_CONFIG,
            'ui_config': UI_CONFIG,
            'experiment_config': EXPERIMENT_CONFIG,
            'module_config': {k: v for k, v in MODULE_CONFIG.items() if k != 'paths'},
            'feature_flags': FEATURE_FLAGS,
            'localization': LOCALIZATION_CONFIG
        }
    }
    
    with open(snapshot_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False, default=str)
    
    return snapshot_path

def get_runtime_info() -> Dict[str, Any]:
    """
    런타임 정보 수집 (디버깅/모니터링용)
    
    Returns:
        런타임 정보 딕셔너리
    """
    import psutil
    import gc
    
    process = psutil.Process()
    
    return {
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
        'gc': {
            'collections': gc.get_count(),
            'objects': len(gc.get_objects()),
            'threshold': gc.get_threshold()
        },
        'files': {
            'open_files': len(process.open_files()),
            'connections': len(process.connections())
        }
    }

# ============================================================================
# 🚀 초기화 및 검증
# ============================================================================

# 앱 시작 시 자동 실행
if __name__ != "__main__":
    # 설정 검증
    is_valid, messages = validate_config()
    
    # 로깅 설정
    logger = logging.getLogger(__name__)
    
    for message in messages:
        if message.startswith('CRITICAL'):
            logger.critical(message)
        elif message.startswith('ERROR'):
            logger.error(message)
        elif message.startswith('WARNING'):
            logger.warning(message)
        else:
            logger.info(message)
    
    # 검증 실패 시 종료 (프로덕션)
    if not is_valid and IS_PRODUCTION:
        raise RuntimeError("설정 검증 실패. 로그를 확인하세요.")
    
    # 개발 환경에서 설정 스냅샷 저장
    if IS_DEVELOPMENT and not IS_FROZEN:
        try:
            snapshot_path = save_config_snapshot()
            logger.debug(f"설정 스냅샷 저장됨: {snapshot_path}")
        except Exception as e:
            logger.warning(f"설정 스냅샷 저장 실패: {e}")

# ============================================================================
# 📤 Public API
# ============================================================================

__all__ = [
    # 환경 정보
    'PROJECT_ROOT', 'DATA_DIR', 'CONFIG_DIR', 'LOGS_DIR', 'TEMP_DIR',
    'CACHE_DIR', 'DB_DIR', 'MODULES_DIR', 'BACKUP_DIR', 'EXPORTS_DIR',
    'ENV', 'IS_PRODUCTION', 'IS_STAGING', 'IS_DEVELOPMENT', 'IS_TEST',
    'IS_DESKTOP', 'IS_FROZEN', 'DEBUG', 'SYSTEM_INFO',
    
    # 앱 정보
    'APP_INFO',
    
    # AI 설정
    'AIProvider', 'AIEngineConfig', 'AI_ENGINES', 'AI_EXPLANATION_CONFIG',
    
    # 주요 설정
    'FILE_CONFIG', 'PROTOCOL_EXTRACTION_CONFIG', 'SQLITE_CONFIG', 
    'GOOGLE_SHEETS_CONFIG', 'SECURITY_CONFIG', 'UI_CONFIG', 
    'EXPERIMENT_CONFIG', 'MODULE_CONFIG', 'SYNC_CONFIG', 'UPDATE_CONFIG', 
    'LOCALIZATION_CONFIG', 'PERFORMANCE_CONFIG', 'DEVELOPER_CONFIG', 
    'FEATURE_FLAGS',
    
    # 유틸리티 함수
    'get_config', 'set_config', 'validate_config', 'save_config_snapshot',
    'get_runtime_info'
]
