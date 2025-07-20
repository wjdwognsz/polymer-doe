"""
app_config.py - Universal DOE Platform 전역 설정

이 파일은 앱의 모든 전역 설정을 관리합니다.
- AI 엔진 설정
- 연구 분야 및 실험 유형 정의
- 데이터베이스 설정
- 사용자 시스템
- 보안 및 성능 설정
"""

import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import timedelta
from dataclasses import dataclass, field
from enum import Enum

# ===== 앱 메타데이터 =====
APP_INFO = {
    'name': 'Universal DOE Platform',
    'version': '2.0.0',
    'description': '모든 화학·재료과학 연구자를 위한 AI 기반 만능 실험 설계 플랫폼',
    'author': 'Universal DOE Team',
    'contact': 'contact@universaldoe.com',
    'github': 'https://github.com/universaldoe/platform',
    'license': 'MIT'
}

# ===== AI 엔진 설정 =====
AI_ENGINES = {
    'google_gemini': {
        'name': 'Google Gemini 2.0 Flash',
        'model': 'gemini-2.0-flash-exp',
        'api_key_name': 'GOOGLE_GEMINI_API_KEY',
        'required': True,  # 필수 엔진
        'free_tier': True,
        'purpose': '주 AI 엔진, 자연어 이해, 실험 설계 생성',
        'features': ['text_generation', 'analysis', 'code_generation'],
        'rate_limit': {
            'requests_per_minute': 60,
            'tokens_per_minute': 1000000
        },
        'docs_url': 'https://makersuite.google.com/app/apikey',
        'sdk': 'google-generativeai'
    },
    'xai_grok': {
        'name': 'xAI Grok 3 Mini',
        'model': 'grok-3-mini',
        'api_key_name': 'XAI_GROK_API_KEY',
        'required': False,
        'free_tier': False,
        'purpose': '실시간 정보, 최신 연구 동향',
        'features': ['real_time_data', 'research_trends'],
        'rate_limit': {
            'requests_per_minute': 30,
            'tokens_per_minute': 100000
        },
        'docs_url': 'https://x.ai/api',
        'sdk': 'requests'  # OpenAI 호환 API
    },
    'groq': {
        'name': 'Groq (초고속 추론)',
        'model': 'mixtral-8x7b-32768',
        'api_key_name': 'GROQ_API_KEY',
        'required': False,
        'free_tier': True,
        'purpose': '초고속 추론, 배치 처리',
        'features': ['fast_inference', 'batch_processing'],
        'rate_limit': {
            'requests_per_minute': 30,
            'tokens_per_minute': 18000
        },
        'base_url': 'https://api.groq.com/openai/v1',
        'docs_url': 'https://console.groq.com',
        'sdk': 'openai'  # OpenAI 호환
    },
    'deepseek': {
        'name': 'DeepSeek (코드/수식)',
        'model': 'deepseek-chat',
        'api_key_name': 'DEEPSEEK_API_KEY',
        'required': False,
        'free_tier': False,
        'purpose': '코드 생성, 수식 계산, 기술 문서',
        'features': ['code_generation', 'math_computation', 'technical_docs'],
        'rate_limit': {
            'requests_per_minute': 60,
            'tokens_per_minute': 500000
        },
        'base_url': 'https://api.deepseek.com/v1',
        'docs_url': 'https://platform.deepseek.com',
        'sdk': 'openai'  # OpenAI 호환
    },
    'sambanova': {
        'name': 'SambaNova (대규모 모델)',
        'model': 'llama3-405b',
        'api_key_name': 'SAMBANOVA_API_KEY',
        'required': False,
        'free_tier': True,
        'purpose': '대규모 추론, 복잡한 분석',
        'features': ['large_scale_analysis', 'complex_reasoning'],
        'rate_limit': {
            'requests_per_minute': 10,
            'tokens_per_minute': 50000
        },
        'docs_url': 'https://cloud.sambanova.ai',
        'sdk': 'openai'  # OpenAI 호환
    },
    'huggingface': {
        'name': 'HuggingFace (특수 모델)',
        'api_key_name': 'HUGGINGFACE_API_KEY',
        'required': False,
        'free_tier': True,
        'purpose': 'ChemBERTa, MatSciBERT 등 도메인 특화 모델',
        'features': ['domain_specific', 'embeddings', 'classification'],
        'models': {
            'chemistry': 'seyonec/ChemBERTa-zinc-base-v1',
            'materials': 'm3rg-iitd/matscibert',
            'general': 'microsoft/deberta-v3-base'
        },
        'docs_url': 'https://huggingface.co/settings/tokens',
        'sdk': 'huggingface_hub'
    }
}

# ===== 연구 분야 및 실험 유형 =====
RESEARCH_FIELDS = {
    'polymer': {
        'name': '고분자 과학',
        'icon': '🧬',
        'description': '고분자 합성, 가공, 특성분석',
        'experiments': {
            'synthesis': {
                'name': '고분자 합성',
                'types': ['라디칼 중합', '이온 중합', '축합 중합', '개환 중합', 
                         '배위 중합', '리빙 중합', 'RAFT', 'ATRP', 'ROP'],
                'common_factors': ['단량체 농도', '개시제 농도', '온도', '시간', 
                                 '용매', 'pH', '교반속도'],
                'common_responses': ['수율', '분자량', 'PDI', '전환율', 'Tg']
            },
            'processing': {
                'name': '가공 공정',
                'types': ['사출성형', '압출', '블로우성형', '3D 프린팅', 
                         '전기방사', '용액캐스팅'],
                'common_factors': ['온도', '압력', '속도', '시간', '첨가제'],
                'common_responses': ['기계적 물성', '표면 특성', '치수 안정성']
            },
            'characterization': {
                'name': '특성 분석',
                'types': ['GPC', 'NMR', 'FTIR', 'DSC', 'TGA', 'DMA', 'UTM'],
                'common_factors': ['샘플 준비', '측정 조건', '용매'],
                'common_responses': ['분자량', '화학구조', '열적특성', '기계적특성']
            }
        }
    },
    'inorganic': {
        'name': '무기재료',
        'icon': '💎',
        'description': '세라믹, 반도체, 금속 재료',
        'experiments': {
            'synthesis': {
                'name': '무기재료 합성',
                'types': ['고상반응', '용액법', '수열합성', '솔젤법', 'CVD', 'PVD'],
                'common_factors': ['전구체', '온도', '압력', '시간', '분위기'],
                'common_responses': ['결정성', '순도', '입자크기', '비표면적']
            },
            'ceramics': {
                'name': '세라믹 공정',
                'types': ['분말제조', '성형', '소결', '열처리'],
                'common_factors': ['소결온도', '승온속도', '유지시간', '압력'],
                'common_responses': ['밀도', '강도', '경도', '인성']
            }
        }
    },
    'nano': {
        'name': '나노재료',
        'icon': '⚛️',
        'description': '나노입자, 나노구조체',
        'experiments': {
            'nanoparticles': {
                'name': '나노입자 합성',
                'types': ['금속 나노입자', '산화물 나노입자', '양자점', '코어-쉘'],
                'common_factors': ['전구체 농도', '환원제', '캡핑제', '온도', 'pH'],
                'common_responses': ['입자크기', '크기분포', '제타전위', '형태']
            }
        }
    },
    'organic': {
        'name': '유기합성',
        'icon': '🧪',
        'description': '유기 반응, 촉매',
        'experiments': {
            'reactions': {
                'name': '유기 반응',
                'types': ['치환반응', '첨가반응', '제거반응', '재배열반응'],
                'common_factors': ['반응물', '촉매', '용매', '온도', '시간'],
                'common_responses': ['수율', '선택성', '순도', '부산물']
            }
        }
    },
    'composite': {
        'name': '복합재료',
        'icon': '🔧',
        'description': '섬유강화, 입자강화 복합재료',
        'experiments': {
            'fabrication': {
                'name': '복합재료 제조',
                'types': ['RTM', 'VARTM', '핸드레이업', '필라멘트와인딩'],
                'common_factors': ['섬유함량', '수지종류', '경화조건', '압력'],
                'common_responses': ['강도', '탄성률', '층간전단강도', '공극률']
            }
        }
    },
    'bio': {
        'name': '바이오재료',
        'icon': '🧬',
        'description': '생체적합성, 약물전달',
        'experiments': {
            'biocompatibility': {
                'name': '생체적합성',
                'types': ['세포독성', '혈액적합성', '조직적합성'],
                'common_factors': ['재료조성', '표면처리', '배양조건'],
                'common_responses': ['세포생존율', '단백질흡착', '염증반응']
            }
        }
    },
    'energy': {
        'name': '에너지재료',
        'icon': '🔋',
        'description': '배터리, 연료전지, 태양전지',
        'experiments': {
            'battery': {
                'name': '배터리 재료',
                'types': ['리튬이온', '전고체', '나트륨이온'],
                'common_factors': ['전극조성', '전해질', '충방전조건'],
                'common_responses': ['용량', '쿨롱효율', '사이클수명', '율특성']
            }
        }
    },
    'environmental': {
        'name': '환경재료',
        'icon': '🌱',
        'description': '수처리, 대기정화',
        'experiments': {
            'water_treatment': {
                'name': '수처리',
                'types': ['흡착제', '멤브레인', '광촉매'],
                'common_factors': ['pH', '농도', '접촉시간', '온도'],
                'common_responses': ['제거효율', '흡착용량', '재생효율']
            }
        }
    },
    'custom': {
        'name': '사용자 정의',
        'icon': '✨',
        'description': '새로운 연구 분야 추가',
        'experiments': {}
    }
}

# ===== DOE 방법론 =====
DOE_METHODS = {
    'screening': {
        'name': '스크리닝 설계',
        'methods': {
            'pb': 'Plackett-Burman',
            'fractional': '부분요인설계',
            'definitive': 'Definitive Screening'
        },
        'purpose': '중요 인자 선별',
        'factors_range': (4, 15),
        'runs_estimate': lambda k: f"{2**(k-4)}~{2**(k-2)} runs"
    },
    'optimization': {
        'name': '최적화 설계',
        'methods': {
            'ccd': '중심합성설계 (CCD)',
            'bb': 'Box-Behnken',
            'optimal': 'D-Optimal'
        },
        'purpose': '최적 조건 탐색',
        'factors_range': (2, 5),
        'runs_estimate': lambda k: f"{2**k + 2*k + 1}~{3**k} runs"
    },
    'factorial': {
        'name': '요인 설계',
        'methods': {
            'full': '완전요인설계',
            'fractional': '부분요인설계',
            'mixed': '혼합수준설계'
        },
        'purpose': '인자 효과 분석',
        'factors_range': (2, 8),
        'runs_estimate': lambda k, levels=2: f"{levels**k} runs"
    },
    'mixture': {
        'name': '혼합물 설계',
        'methods': {
            'simplex': '심플렉스 격자',
            'centroid': '중심 혼합',
            'extreme': '극점 설계'
        },
        'purpose': '조성 최적화',
        'factors_range': (3, 10),
        'constraint': 'sum = 100%'
    },
    'taguchi': {
        'name': 'Taguchi 설계',
        'methods': {
            'l4': 'L4 (2³)',
            'l8': 'L8 (2⁷)',
            'l9': 'L9 (3⁴)',
            'l16': 'L16 (2¹⁵)',
            'l27': 'L27 (3¹³)'
        },
        'purpose': '품질 강건 설계',
        'features': ['신호 대 잡음비', '직교 배열']
    },
    'custom': {
        'name': '사용자 정의',
        'methods': {
            'manual': '수동 설계',
            'imported': '외부 가져오기',
            'ai_generated': 'AI 생성'
        },
        'purpose': '특수 요구사항'
    }
}

# ===== 사용자 레벨 시스템 =====
class UserLevel(Enum):
    BEGINNER = "초보자"
    INTERMEDIATE = "중급자"
    ADVANCED = "고급자"
    EXPERT = "전문가"

USER_LEVELS = {
    UserLevel.BEGINNER: {
        'name': '초보자',
        'icon': '🌱',
        'description': 'DOE를 처음 접하는 사용자',
        'features': {
            'guided_mode': True,
            'ai_assistance': 'maximum',
            'default_designs': ['full_factorial', 'one_factor'],
            'max_factors': 3,
            'tutorials': True,
            'templates': True
        }
    },
    UserLevel.INTERMEDIATE: {
        'name': '중급자',
        'icon': '🌿',
        'description': '기본적인 DOE 경험이 있는 사용자',
        'features': {
            'guided_mode': False,
            'ai_assistance': 'moderate',
            'default_designs': ['fractional', 'ccd', 'bb'],
            'max_factors': 6,
            'advanced_analysis': True
        }
    },
    UserLevel.ADVANCED: {
        'name': '고급자',
        'icon': '🌳',
        'description': '풍부한 DOE 경험을 가진 사용자',
        'features': {
            'guided_mode': False,
            'ai_assistance': 'minimal',
            'all_designs': True,
            'max_factors': 10,
            'custom_designs': True,
            'advanced_optimization': True
        }
    },
    UserLevel.EXPERT: {
        'name': '전문가',
        'icon': '🏆',
        'description': 'DOE 전문가',
        'features': {
            'all_features': True,
            'dev_mode': True,
            'api_access': True,
            'custom_algorithms': True,
            'plugin_development': True
        }
    }
}

# ===== 데이터베이스 설정 (Google Sheets) =====
DATABASE_CONFIG = {
    'google_sheets': {
        'users_sheet': 'Universal_DOE_Users',
        'projects_sheet': 'Universal_DOE_Projects',
        'experiments_sheet': 'Universal_DOE_Experiments',
        'results_sheet': 'Universal_DOE_Results',
        'templates_sheet': 'Universal_DOE_Templates',
        'shared_modules_sheet': 'Universal_DOE_Modules',
        'scopes': [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive.metadata.readonly'
        ],
        'auto_backup': True,
        'backup_interval': timedelta(hours=6),
        'retention_days': 30
    },
    'cache': {
        'enable': True,
        'ttl': {
            'user_data': 3600,  # 1시간
            'project_list': 300,  # 5분
            'experiment_data': 1800,  # 30분
            'static_data': 86400  # 24시간
        }
    }
}

# ===== 파일 업로드 설정 =====
FILE_UPLOAD_CONFIG = {
    'allowed_extensions': {
        'data': ['.csv', '.xlsx', '.xls', '.txt', '.json'],
        'images': ['.png', '.jpg', '.jpeg', '.gif', '.svg'],
        'documents': ['.pdf', '.docx', '.doc'],
        'code': ['.py', '.r', '.m', '.ipynb']
    },
    'max_file_size_mb': 100,
    'max_files_per_upload': 10,
    'temp_storage_hours': 24,
    'virus_scan': True
}

# ===== 보안 설정 =====
SECURITY_CONFIG = {
    'session': {
        'timeout_minutes': 120,
        'max_concurrent_sessions': 3,
        'remember_me_days': 30
    },
    'password': {
        'min_length': 8,
        'require_uppercase': True,
        'require_lowercase': True,
        'require_numbers': True,
        'require_special': True,
        'hash_algorithm': 'bcrypt',
        'reset_token_hours': 24
    },
    'api_keys': {
        'encryption': True,
        'rotation_days': 90,
        'audit_log': True
    },
    'rate_limiting': {
        'requests_per_minute': 60,
        'burst_size': 100
    }
}

# ===== 알림 설정 =====
NOTIFICATION_CONFIG = {
    'channels': {
        'email': {
            'enabled': True,
            'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            'smtp_port': 587,
            'use_tls': True
        },
        'in_app': {
            'enabled': True,
            'retention_days': 30
        },
        'push': {
            'enabled': False,  # 향후 구현
            'service': 'firebase'
        }
    },
    'triggers': {
        'experiment_complete': True,
        'collaboration_invite': True,
        'analysis_ready': True,
        'error_alert': True,
        'weekly_summary': True
    }
}

# ===== 성능 설정 =====
PERFORMANCE_CONFIG = {
    'parallel_processing': {
        'enabled': True,
        'max_workers': 4,
        'chunk_size': 1000
    },
    'optimization': {
        'lazy_loading': True,
        'pagination_size': 50,
        'query_timeout_seconds': 30
    },
    'monitoring': {
        'enabled': True,
        'metrics': ['response_time', 'error_rate', 'user_activity'],
        'alert_thresholds': {
            'response_time_ms': 1000,
            'error_rate_percent': 5
        }
    }
}

# ===== 기능 플래그 =====
FEATURE_FLAGS = {
    'ai_multi_engine': True,
    'custom_modules': True,
    'real_time_collaboration': True,
    'advanced_visualization': True,
    'machine_learning': True,
    'api_access': False,  # 베타
    'mobile_app': False,  # 개발 중
    'offline_mode': False,  # 계획 중
    'blockchain_verification': False,  # 미래 기능
    'ar_visualization': False  # 미래 기능
}

# ===== 분석 설정 =====
ANALYSIS_CONFIG = {
    'statistical': {
        'confidence_level': 0.95,
        'significance_level': 0.05,
        'power': 0.80,
        'multiple_comparison_correction': 'bonferroni'
    },
    'visualization': {
        'default_theme': 'plotly',
        'color_palette': 'viridis',
        'interactive': True,
        'export_formats': ['png', 'svg', 'html', 'pdf']
    },
    'machine_learning': {
        'models': {
            'regression': ['linear', 'polynomial', 'random_forest', 'xgboost'],
            'classification': ['logistic', 'svm', 'neural_network'],
            'optimization': ['gaussian_process', 'bayesian']
        },
        'cross_validation_folds': 5,
        'test_size': 0.2
    }
}

# ===== 협업 설정 =====
COLLABORATION_CONFIG = {
    'project_sharing': {
        'levels': ['view', 'comment', 'edit', 'admin'],
        'default_permission': 'view',
        'require_approval': True
    },
    'team_features': {
        'max_team_size': 50,
        'roles': ['member', 'manager', 'admin'],
        'activity_tracking': True
    },
    'community': {
        'public_templates': True,
        'module_marketplace': True,
        'forum': True,
        'ratings': True,
        'badges': True
    }
}

# ===== 지역화 설정 =====
LOCALIZATION_CONFIG = {
    'default_language': 'ko',
    'supported_languages': ['ko', 'en', 'zh', 'ja'],
    'date_format': 'YYYY-MM-DD',
    'time_format': '24h',
    'timezone': 'Asia/Seoul',
    'currency': 'KRW',
    'units': {
        'temperature': 'celsius',
        'pressure': 'bar',
        'length': 'mm',
        'mass': 'g',
        'volume': 'mL'
    }
}

# ===== 외부 서비스 통합 =====
INTEGRATIONS = {
    'google_scholar': {
        'enabled': True,
        'api_endpoint': 'https://scholar.google.com',
        'rate_limit': 10  # requests per minute
    },
    'pubmed': {
        'enabled': True,
        'api_endpoint': 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/',
        'api_key_name': 'PUBMED_API_KEY'
    },
    'chemspider': {
        'enabled': False,
        'api_key_name': 'CHEMSPIDER_API_KEY'
    },
    'materials_project': {
        'enabled': False,
        'api_key_name': 'MP_API_KEY'
    }
}

# ===== 오류 메시지 =====
ERROR_MESSAGES = {
    'auth': {
        'invalid_credentials': '잘못된 이메일 또는 비밀번호입니다.',
        'account_locked': '계정이 잠겼습니다. 관리자에게 문의하세요.',
        'session_expired': '세션이 만료되었습니다. 다시 로그인해주세요.'
    },
    'api': {
        'missing_key': 'API 키가 설정되지 않았습니다.',
        'rate_limit': 'API 요청 한도를 초과했습니다.',
        'connection_error': 'API 서버에 연결할 수 없습니다.'
    },
    'data': {
        'invalid_format': '잘못된 데이터 형식입니다.',
        'missing_required': '필수 항목이 누락되었습니다.',
        'size_exceeded': '파일 크기가 제한을 초과했습니다.'
    },
    'general': {
        'unexpected': '예기치 않은 오류가 발생했습니다.',
        'permission_denied': '권한이 없습니다.',
        'not_found': '요청한 리소스를 찾을 수 없습니다.'
    }
}

# ===== 도움말 및 문서 =====
HELP_URLS = {
    'getting_started': '/docs/getting-started',
    'doe_basics': '/docs/doe-basics',
    'api_documentation': '/docs/api',
    'video_tutorials': '/tutorials',
    'faq': '/faq',
    'community_forum': '/forum',
    'contact_support': '/support'
}

# ===== 개발/운영 환경 설정 =====
ENVIRONMENT = os.getenv('APP_ENV', 'development')

if ENVIRONMENT == 'production':
    DEBUG = False
    LOG_LEVEL = 'INFO'
    CACHE_ENABLED = True
    ERROR_TRACKING = True
else:
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    CACHE_ENABLED = False
    ERROR_TRACKING = False

# ===== 유틸리티 함수 =====
def get_ai_engine_config(engine_name: str) -> Optional[Dict[str, Any]]:
    """특정 AI 엔진의 설정을 반환"""
    return AI_ENGINES.get(engine_name)

def get_research_field_experiments(field: str) -> Dict[str, Any]:
    """특정 연구 분야의 실험 유형을 반환"""
    return RESEARCH_FIELDS.get(field, {}).get('experiments', {})

def get_doe_method_info(category: str, method: str) -> Optional[Dict[str, Any]]:
    """특정 DOE 방법의 정보를 반환"""
    category_info = DOE_METHODS.get(category, {})
    if 'methods' in category_info:
        method_name = category_info['methods'].get(method)
        if method_name:
            return {
                'name': method_name,
                'category': category,
                'purpose': category_info.get('purpose', ''),
                **category_info
            }
    return None

def validate_file_extension(filename: str, file_type: str = 'data') -> bool:
    """파일 확장자 검증"""
    allowed = FILE_UPLOAD_CONFIG['allowed_extensions'].get(file_type, [])
    return any(filename.lower().endswith(ext) for ext in allowed)

def get_user_level_features(level: UserLevel) -> Dict[str, Any]:
    """사용자 레벨에 따른 기능 제한 반환"""
    return USER_LEVELS.get(level, {}).get('features', {})

# ===== 상수 export =====
__all__ = [
    'APP_INFO',
    'AI_ENGINES',
    'RESEARCH_FIELDS',
    'DOE_METHODS',
    'UserLevel',
    'USER_LEVELS',
    'DATABASE_CONFIG',
    'FILE_UPLOAD_CONFIG',
    'SECURITY_CONFIG',
    'NOTIFICATION_CONFIG',
    'PERFORMANCE_CONFIG',
    'FEATURE_FLAGS',
    'ANALYSIS_CONFIG',
    'COLLABORATION_CONFIG',
    'LOCALIZATION_CONFIG',
    'INTEGRATIONS',
    'ERROR_MESSAGES',
    'HELP_URLS',
    'ENVIRONMENT',
    'DEBUG',
    'get_ai_engine_config',
    'get_research_field_experiments',
    'get_doe_method_info',
    'validate_file_extension',
    'get_user_level_features'
]
