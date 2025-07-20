# config/app_config.py

# 표준 라이브러리
import os
import json
import logging
from pathlib import Path
from datetime import timedelta
from typing import Dict, List, Optional, Any, Tuple
import secrets

# 환경 변수 관리
from dotenv import load_dotenv

# Streamlit
import streamlit as st

# 환경 변수 로드
load_dotenv()

# ===========================
# 1. 기본 경로 및 환경 설정
# ===========================

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
TEMP_DIR = PROJECT_ROOT / "temp"

# 디렉토리 생성
for dir_path in [DATA_DIR, LOGS_DIR, TEMP_DIR]:
    dir_path.mkdir(exist_ok=True)

# 환경 변수
ENV = os.getenv("APP_ENV", "development")  # development, staging, production
DEBUG = ENV == "development"
VERSION = "2.0.0"
APP_NAME = "Polymer DOE Platform"
APP_DESCRIPTION = "AI 기반 고분자 실험 설계 교육 플랫폼"

# 로깅 설정
LOG_LEVEL = logging.DEBUG if DEBUG else logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / f"app_{ENV}.log"

# 기본 URL
if ENV == "production":
    BASE_URL = "https://polymer-doe.streamlit.app"
elif ENV == "staging":
    BASE_URL = "https://polymer-doe-staging.streamlit.app"
else:
    BASE_URL = "http://localhost:8501"

# 세션 설정
SESSION_COOKIE_NAME = "polymer_doe_session"
SESSION_EXPIRY_HOURS = 24
REMEMBER_ME_DAYS = 30

# ===========================
# 2. Google Sheets 데이터베이스 설정
# ===========================

GOOGLE_SHEETS_CONFIG = {
    # 메인 스프레드시트 ID
    'spreadsheet_id': os.getenv('GOOGLE_SHEETS_ID', st.secrets.get("google_sheets_url", "")),
    
    # Service Account 인증 정보
    'service_account_info': st.secrets.get("google_service_account", None),
    'service_account_file': CONFIG_DIR / 'service_account.json',
    
    # 시트 이름 매핑
    'sheet_names': {
        'users': 'Users',
        'projects': 'Projects',
        'experiments': 'Experiments',
        'results': 'Results',
        'comments': 'Comments',
        'files': 'Files',
        'notifications': 'Notifications',
        'activity_log': 'Activity_Log',
        'learning_progress': 'Learning_Progress',  # 학습 진도 추적
        'growth_metrics': 'Growth_Metrics',      # 성장 지표
        'educational_logs': 'Educational_Logs',   # 교육 콘텐츠 로그
        'system_config': 'System_Config'
    },
    
    # API 설정
    'rate_limit': 60,
    'batch_size': 1000,
    'cache_ttl': 300,
    'max_retries': 5,
    'retry_delay': 1.0,
    'retry_backoff': 2.0,
    
    # 권한 범위
    'scopes': [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive.file'
    ]
}

# ===========================
# 3. 교육적 성장 중심 레벨 시스템
# ===========================

LEVEL_CONFIG = {
    'philosophy': '모든 기능은 모든 레벨에서 사용 가능. 레벨은 교육적 지원의 정도만 결정.',
    
    'levels': {
        'beginner': {
            'min_points': 0,
            'display_name': '초급 연구원',
            'badge': '🌱',
            'color': '#10B981',
            
            # 교육 설정
            'educational_mode': {
                'explanations': 'full',          # 전체 설명
                'skip_allowed': False,           # 설명 스킵 불가
                'auto_guide': True,              # 자동 가이드
                'confirmation_required': True,    # 모든 작업 확인
                'show_why': True,                # "왜?" 버튼 항상 표시
                'tooltips': 'always',            # 툴팁 항상 표시
                'examples': 'multiple',          # 다양한 예시 제공
                'pace': 'slow'                   # 느린 진행 속도
            },
            
            # UI 설정
            'ui_complexity': 'simple',
            'wizard_mode': True,
            'step_by_step': True,
            'max_options_shown': 3,
            
            # 피드백 설정
            'feedback': {
                'success_messages': 'detailed',
                'error_guidance': 'step_by_step',
                'hints_enabled': True,
                'auto_suggestions': True
            }
        },
        
        'intermediate': {
            'min_points': 100,
            'display_name': '중급 연구원',
            'badge': '🌿',
            'color': '#3B82F6',
            
            # 교육 설정
            'educational_mode': {
                'explanations': 'balanced',      # 핵심 설명만
                'skip_allowed': True,            # 설명 스킵 가능
                'auto_guide': False,             # 수동 가이드
                'confirmation_required': False,   # 중요 작업만 확인
                'show_why': 'on_demand',         # 요청시 "왜?" 표시
                'tooltips': 'hover',             # 마우스 오버시
                'examples': 'relevant',          # 관련 예시만
                'pace': 'normal'                 # 일반 속도
            },
            
            # UI 설정
            'ui_complexity': 'standard',
            'wizard_mode': False,
            'step_by_step': False,
            'max_options_shown': 5,
            
            # 피드백 설정
            'feedback': {
                'success_messages': 'concise',
                'error_guidance': 'hints',
                'hints_enabled': True,
                'auto_suggestions': False
            }
        },
        
        'advanced': {
            'min_points': 500,
            'display_name': '고급 연구원',
            'badge': '🌳',
            'color': '#8B5CF6',
            
            # 교육 설정
            'educational_mode': {
                'explanations': 'minimal',       # 최소 설명
                'skip_allowed': True,
                'auto_guide': False,
                'confirmation_required': False,
                'show_why': 'hidden',            # "왜?" 숨김
                'tooltips': 'on_demand',         # 요청시만
                'examples': 'none',              # 예시 없음
                'pace': 'fast'                   # 빠른 속도
            },
            
            # UI 설정
            'ui_complexity': 'advanced',
            'wizard_mode': False,
            'step_by_step': False,
            'max_options_shown': 10,
            
            # 피드백 설정
            'feedback': {
                'success_messages': 'minimal',
                'error_guidance': 'code_only',
                'hints_enabled': False,
                'auto_suggestions': False
            }
        },
        
        'expert': {
            'min_points': 1500,
            'display_name': '전문 연구원',
            'badge': '🏆',
            'color': '#F59E0B',
            
            # 교육 설정
            'educational_mode': {
                'explanations': 'off',           # 설명 없음
                'skip_allowed': True,
                'auto_guide': False,
                'confirmation_required': False,
                'show_why': 'off',               # "왜?" 없음
                'tooltips': 'off',               # 툴팁 없음
                'examples': 'none',
                'pace': 'instant'                # 즉시 실행
            },
            
            # UI 설정
            'ui_complexity': 'expert',
            'wizard_mode': False,
            'step_by_step': False,
            'max_options_shown': -1,  # 모두 표시
            'keyboard_shortcuts': True,
            
            # 피드백 설정
            'feedback': {
                'success_messages': 'off',
                'error_guidance': 'none',
                'hints_enabled': False,
                'auto_suggestions': False
            }
        }
    },
    
    # 포인트 시스템 (순수 성취감용)
    'point_rewards': {
        # 일일 활동
        'daily_login': 5,
        'consistent_week': 25,
        'consistent_month': 100,
        
        # 학습 활동
        'read_explanation': 2,
        'complete_tutorial': 10,
        'skip_guide_first_time': 20,  # 가이드 없이 첫 성공
        
        # 프로젝트 활동
        'project_created': 15,
        'project_completed': 30,
        'complex_design_used': 25,
        
        # 성장 지표
        'reduced_error_rate': 20,
        'increased_speed': 15,
        'helped_others': 30,
        
        # 마일스톤
        'first_solo_project': 50,
        'master_technique': 40,
        'innovation': 100
    }
}

# ===========================
# 4. 교육 콘텐츠 설정
# ===========================

EDUCATIONAL_CONTENT = {
    # 설명 레벨별 콘텐츠
    'explanations': {
        'project_setup': {
            'polymer_selection': {
                'beginner': {
                    'content': """
                    🎯 **고분자 선택 가이드**
                    
                    고분자를 선택할 때는 다음을 고려해야 합니다:
                    1. **용도**: 제품이 어디에 사용되나요?
                    2. **물성**: 필요한 강도, 유연성은?
                    3. **가공성**: 어떻게 성형할 예정인가요?
                    4. **비용**: 예산 범위는?
                    
                    💡 **초보자 팁**: PET는 투명하고 강한 플라스틱으로 
                    음료수 병에 많이 사용됩니다. 처음이라면 PET나 PP같은 
                    범용 플라스틱부터 시작해보세요!
                    """,
                    'interactive': True,
                    'quiz': True,
                    'examples': ['PET 병', 'PP 용기', 'PE 필름']
                },
                'intermediate': {
                    'content': """
                    **고분자 선택**: 용도별 주요 고려사항
                    - 기계적 물성 (인장강도, 신율, 탄성률)
                    - 열적 특성 (Tg, Tm, 열변형온도)
                    - 화학적 저항성
                    """,
                    'interactive': False,
                    'quiz': False,
                    'examples': []
                },
                'advanced': {
                    'content': "고분자 구조-물성 관계를 고려한 선택",
                    'interactive': False,
                    'quiz': False,
                    'examples': []
                },
                'expert': None  # 설명 없음
            }
        },
        
        'experiment_design': {
            'design_selection': {
                'beginner': {
                    'content': """
                    📊 **실험 설계 방법 선택하기**
                    
                    **1. 스크리닝 (Screening)**
                    많은 요인 중 중요한 것을 찾을 때
                    → Plackett-Burman 설계
                    
                    **2. 최적화 (Optimization)**
                    중요 요인의 최적 조건을 찾을 때
                    → Box-Behnken, 중심합성설계
                    
                    **3. 견고성 (Robustness)**
                    외부 변동에 강한 조건을 찾을 때
                    → Taguchi 설계
                    
                    🤔 **어떤 걸 선택해야 할까요?**
                    처음이라면 요인이 3개 이하일 때는 완전요인설계,
                    4개 이상이면 부분요인설계를 추천합니다!
                    """,
                    'decision_tree': True,
                    'calculator': True
                },
                'intermediate': {
                    'content': """
                    **설계 선택 기준**
                    - 요인 수와 실험 횟수의 균형
                    - 교호작용 추정 필요성
                    - 곡면성(curvature) 검출 여부
                    """,
                    'decision_tree': False,
                    'calculator': True
                },
                'advanced': {
                    'content': "설계 효율성: D-optimality, I-optimality",
                    'decision_tree': False,
                    'calculator': False
                },
                'expert': None
            }
        }
    },
    
    # 인터랙티브 가이드
    'interactive_guides': {
        'beginner': {
            'show_arrows': True,
            'highlight_next_step': True,
            'auto_scroll': True,
            'voice_guidance': False,  # 향후 기능
            'animation_speed': 'slow'
        },
        'intermediate': {
            'show_arrows': False,
            'highlight_next_step': False,
            'auto_scroll': False,
            'voice_guidance': False,
            'animation_speed': 'normal'
        },
        'advanced': {
            'all_features_off': True
        },
        'expert': {
            'all_features_off': True
        }
    },
    
    # 오류 메시지 상세도
    'error_messages': {
        'beginner': {
            'missing_data': """
            ❌ 데이터가 입력되지 않았습니다.
            
            **해결 방법:**
            1. 위의 입력 필드를 확인하세요
            2. 빨간색으로 표시된 필수 항목(*)을 모두 입력하세요
            3. 숫자는 숫자만, 텍스트는 텍스트만 입력하세요
            
            💡 도움이 필요하면 우측 상단의 ❓ 버튼을 클릭하세요!
            """,
            'show_video_tutorial': True
        },
        'intermediate': {
            'missing_data': "필수 입력 항목을 확인하세요.",
            'show_video_tutorial': False
        },
        'advanced': {
            'missing_data': "Missing required fields",
            'show_video_tutorial': False
        },
        'expert': {
            'missing_data': "ERR_MISSING_DATA",
            'show_video_tutorial': False
        }
    }
}

# ===========================
# 5. 성장 추적 시스템
# ===========================

GROWTH_TRACKING = {
    'enabled': True,
    
    # 성장 지표
    'metrics': {
        'understanding': {
            'factors': [
                'explanation_read_time',      # 설명 읽은 시간
                'help_clicks_reduction',       # 도움말 클릭 감소율
                'error_rate_reduction',        # 오류 발생 감소율
                'correct_first_attempt'        # 첫 시도 성공률
            ],
            'weight': 0.3
        },
        'independence': {
            'factors': [
                'guide_skip_rate',            # 가이드 스킵 비율
                'wizard_abandon_rate',         # 마법사 모드 포기율
                'direct_navigation',           # 직접 네비게이션
                'advanced_features_usage'      # 고급 기능 사용률
            ],
            'weight': 0.3
        },
        'expertise': {
            'factors': [
                'complex_designs_used',        # 복잡한 설계 사용
                'optimization_success',        # 최적화 성공률
                'time_efficiency',            # 작업 시간 효율
                'innovation_score'            # 혁신성 점수
            ],
            'weight': 0.4
        }
    },
    
    # 성장 마일스톤
    'milestones': {
        'first_steps': {
            'completed_tutorial': '튜토리얼 완료',
            'first_project': '첫 프로젝트 생성',
            'first_experiment': '첫 실험 설계'
        },
        'growing_confidence': {
            'skip_guide_success': '가이드 없이 성공',
            'use_advanced_design': '고급 설계 사용',
            'complete_optimization': '최적화 완료'
        },
        'becoming_expert': {
            'mentor_others': '다른 사용자 도움',
            'create_template': '템플릿 생성',
            'publish_results': '결과 발표'
        }
    },
    
    # 적응형 난이도
    'adaptive_difficulty': {
        'enabled': True,
        'factors': {
            'success_rate': 0.4,
            'speed_improvement': 0.3,
            'feature_exploration': 0.3
        },
        'adjustment_threshold': 0.8,  # 80% 성공시 난이도 상승 제안
        'cooldown_days': 7            # 레벨 변경 제안 주기
    }
}

# ===========================
# 6. API 키 설정 (모든 레벨 동일하게 사용)
# ===========================

API_KEYS = {
    # AI APIs
    'gemini': {
        'key': os.getenv('GEMINI_API_KEY', st.secrets.get("google_gemini", "")),
        'endpoint': 'https://generativelanguage.googleapis.com/v1beta',
        'model': 'gemini-pro',
        'rate_limit': 60,
        'timeout': 30,
        'max_tokens': 8192
    },
    'grok': {
        'key': os.getenv('GROK_API_KEY', st.secrets.get("xai_grok", "")),
        'endpoint': 'https://api.x.ai/v1',
        'model': 'grok-beta',
        'rate_limit': 50,
        'timeout': 30,
        'max_tokens': 4096
    },
    'deepseek': {
        'key': os.getenv('DEEPSEEK_API_KEY', st.secrets.get("deepseek", "")),
        'endpoint': 'https://api.deepseek.com/v1',
        'model': 'deepseek-coder',
        'rate_limit': 100,
        'timeout': 30,
        'max_tokens': 16384
    },
    'groq': {
        'key': os.getenv('GROQ_API_KEY', st.secrets.get("groq", "")),
        'endpoint': 'https://api.groq.com/openai/v1',
        'model': 'mixtral-8x7b-32768',
        'rate_limit': 30,
        'timeout': 30,
        'max_tokens': 32768
    },
    'sambanova': {
        'key': os.getenv('SAMBANOVA_API_KEY', st.secrets.get("sambanova", "")),
        'endpoint': 'https://api.sambanova.ai/v1',
        'model': 'Meta-Llama-3.1-405B-Instruct',
        'rate_limit': 20,
        'timeout': 60,
        'max_tokens': 4096
    },
    'huggingface': {
        'key': os.getenv('HUGGINGFACE_API_KEY', st.secrets.get("huggingface", "")),
        'endpoint': 'https://api-inference.huggingface.co/models',
        'model': 'meta-llama/Llama-2-70b-chat-hf',
        'rate_limit': 100,
        'timeout': 120,
        'max_tokens': 4096
    },
    
    # Database APIs (모든 레벨 동일 접근)
    'materials_project': {
        'key': os.getenv('MP_API_KEY', st.secrets.get("materials_project", "")),
        'endpoint': 'https://api.materialsproject.org',
        'rate_limit': 100,
        'timeout': 30
    },
    'materials_commons': {
        'key': os.getenv('MATERIALS_COMMONS_API_KEY', st.secrets.get("materials_commons", "")),
        'endpoint': 'https://materialscommons.org/api',
        'rate_limit': 60,
        'timeout': 30
    },
    'pubchem': {
        'key': None,
        'endpoint': 'https://pubchem.ncbi.nlm.nih.gov/rest/pug',
        'rate_limit': 5,
        'timeout': 30
    },
    'zenodo': {
        'key': os.getenv('ZENODO_API_KEY', st.secrets.get("zenodo", "")),
        'endpoint': 'https://zenodo.org/api',
        'rate_limit': 60,
        'timeout': 30
    },
    'figshare': {
        'key': os.getenv('FIGSHARE_API_KEY', st.secrets.get("figshare", "")),
        'endpoint': 'https://api.figshare.com/v2',
        'rate_limit': 60,
        'timeout': 30
    },
    'protocols_io': {
        'key': os.getenv('PROTOCOLS_IO_KEY', st.secrets.get("protocols_io", "")),
        'endpoint': 'https://www.protocols.io/api/v3',
        'rate_limit': 60,
        'timeout': 30
    },
    'github': {
        'key': os.getenv('GITHUB_TOKEN', st.secrets.get("github", "")),
        'endpoint': 'https://api.github.com',
        'rate_limit': 60,
        'timeout': 30
    }
}

# API 사용 추적 (제한 없음, 통계만)
API_USAGE_TRACKING = {
    'enabled': True,
    'track_per_user': True,
    'track_per_api': True,
    'show_statistics': True,
    'limits': None,  # 모든 레벨 무제한
    'warnings': None  # 경고 없음
}

# ===========================
# 7. UI 적응형 설정
# ===========================

UI_CONFIG = {
    # 기본 설정
    'page': {
        'title': APP_NAME,
        'icon': '🧬',
        'layout': 'wide',
        'initial_sidebar_state': 'expanded'
    },
    
    # 레벨별 UI 적응
    'level_adaptations': {
        'beginner': {
            'layout_complexity': 'simple',
            'menu_depth': 1,  # 단순 메뉴
            'show_advanced_options': False,
            'animations': 'full',
            'transitions': 'slow',
            'confirmation_dialogs': True,
            'undo_redo': True,
            'autosave': True,
            'shortcuts': False
        },
        'intermediate': {
            'layout_complexity': 'standard',
            'menu_depth': 2,
            'show_advanced_options': True,
            'animations': 'reduced',
            'transitions': 'normal',
            'confirmation_dialogs': False,
            'undo_redo': True,
            'autosave': True,
            'shortcuts': True
        },
        'advanced': {
            'layout_complexity': 'advanced',
            'menu_depth': 3,
            'show_advanced_options': True,
            'animations': 'minimal',
            'transitions': 'fast',
            'confirmation_dialogs': False,
            'undo_redo': True,
            'autosave': False,
            'shortcuts': True
        },
        'expert': {
            'layout_complexity': 'expert',
            'menu_depth': -1,  # 모든 메뉴
            'show_advanced_options': True,
            'animations': 'off',
            'transitions': 'instant',
            'confirmation_dialogs': False,
            'undo_redo': False,
            'autosave': False,
            'shortcuts': True,
            'command_palette': True  # Cmd+K 스타일
        }
    },
    
    # 컴포넌트 설정
    'components': {
        'max_file_size_mb': 200,
        'accepted_file_types': ['csv', 'xlsx', 'xls', 'txt', 'pdf', 'json'],
        'data_editor_height': 400,
        'chart_height': 500,
        'table_page_size': 20
    }
}

# ===========================
# 8. 교육적 프롬프트 시스템
# ===========================

EDUCATIONAL_PROMPTS = {
    'ai_explanations': {
        'beginner': {
            'prefix': "초보자를 위해 쉽고 자세하게 설명해주세요. 전문용어는 풀어서 설명하고, 예시를 많이 들어주세요.",
            'suffix': "마지막에 '💡 핵심 정리'를 3줄로 요약해주세요.",
            'style': "friendly",
            'examples': True,
            'analogies': True
        },
        'intermediate': {
            'prefix': "핵심 개념을 중심으로 설명해주세요.",
            'suffix': "",
            'style': "professional",
            'examples': False,
            'analogies': False
        },
        'advanced': {
            'prefix': "기술적 세부사항을 포함해 간단히 설명해주세요.",
            'suffix': "",
            'style': "technical",
            'examples': False,
            'analogies': False
        },
        'expert': {
            'prefix': "",
            'suffix': "",
            'style': "minimal",
            'examples': False,
            'analogies': False
        }
    }
}

# ===========================
# 9. 보안 설정 (모든 레벨 동일)
# ===========================

SECURITY_CONFIG = {
    # JWT 설정
    'jwt_secret_key': os.getenv(
        'JWT_SECRET_KEY', 
        st.secrets.get("security", {}).get("jwt_secret_key", secrets.token_urlsafe(32))
    ),
    'jwt_algorithm': 'HS256',
    'jwt_expiry_hours': 24,
    
    # 암호화 키
    'encryption_key': os.getenv(
        'ENCRYPTION_KEY',
        st.secrets.get("security", {}).get("encryption_key", secrets.token_urlsafe(32))
    ).encode()[:32],
    
    # 비밀번호 정책
    'password': {
        'min_length': 8,
        'require_uppercase': True,
        'require_lowercase': True,
        'require_numbers': True,
        'require_special': True,
        'bcrypt_rounds': 12
    },
    
    # 로그인 보안
    'login': {
        'max_attempts': 5,
        'lockout_duration_minutes': 15,
        'session_timeout_minutes': 30,
        'remember_me_days': 30
    }
}

# ===========================
# 10. 이메일 설정
# ===========================

EMAIL_CONFIG = {
    'smtp_server': os.getenv(
        'SMTP_SERVER', 
        st.secrets.get("email", {}).get("smtp_server", "smtp.gmail.com")
    ),
    'smtp_port': int(os.getenv(
        'SMTP_PORT',
        st.secrets.get("email", {}).get("smtp_port", 587)
    )),
    'username': os.getenv(
        'SMTP_USERNAME',
        st.secrets.get("email", {}).get("username", "")
    ),
    'password': os.getenv(
        'SMTP_PASSWORD',
        st.secrets.get("email", {}).get("password", "")
    ),
    'from_email': os.getenv('FROM_EMAIL', 'noreply@polymer-doe.com'),
    'from_name': 'Polymer DOE Platform',
    'use_tls': True,
    'timeout': 30
}

# ===========================
# 11. 캐시 설정
# ===========================

CACHE_CONFIG = {
    'backend': 'memory',
    'prefix': f'polymer_doe_{ENV}_',
    
    'ttl': {
        'default': 300,
        'user_data': 300,
        'project_data': 60,
        'api_response': 3600,
        'educational_content': 86400,  # 교육 콘텐츠는 24시간
        'growth_metrics': 600
    },
    
    'max_size': {
        'memory': 1000,
        'per_user': 100
    }
}

# ===========================
# 12. 실험 설계 설정
# ===========================

EXPERIMENT_CONFIG = {
    # 설계 유형 (모든 레벨 사용 가능)
    'design_types': {
        'full_factorial': {
            'name': '완전요인설계',
            'complexity': 'basic',
            'beginner_recommended': True
        },
        'fractional_factorial': {
            'name': '부분요인설계',
            'complexity': 'intermediate',
            'beginner_recommended': True
        },
        'plackett_burman': {
            'name': 'Plackett-Burman 설계',
            'complexity': 'intermediate',
            'beginner_recommended': False
        },
        'box_behnken': {
            'name': 'Box-Behnken 설계',
            'complexity': 'advanced',
            'beginner_recommended': False
        },
        'central_composite': {
            'name': '중심합성설계',
            'complexity': 'advanced',
            'beginner_recommended': False
        },
        'mixture': {
            'name': '혼합물 설계',
            'complexity': 'advanced',
            'beginner_recommended': False
        },
        'taguchi': {
            'name': 'Taguchi 설계',
            'complexity': 'advanced',
            'beginner_recommended': False
        }
    },
    
    # 제한사항 (모든 레벨 동일)
    'limits': {
        'max_factors': 20,
        'max_levels': 10,
        'max_runs': 1000,
        'max_responses': 50
    }
}

# ===========================
# 13. 협업 기능 (모든 레벨 사용 가능)
# ===========================

COLLABORATION_CONFIG = {
    'enabled': True,
    'features': {
        'project_sharing': True,
        'real_time_collaboration': True,
        'commenting': True,
        'version_control': True,
        'team_management': True
    },
    
    # 레벨별 가이드만 다름
    'level_guides': {
        'beginner': {
            'show_collaboration_tutorial': True,
            'auto_save_enabled': True,
            'conflict_resolution_help': True
        },
        'expert': {
            'show_collaboration_tutorial': False,
            'auto_save_enabled': False,
            'conflict_resolution_help': False
        }
    }
}

# ===========================
# 14. 기능 플래그
# ===========================

FEATURE_FLAGS = {
    # 모든 기능은 모든 레벨에서 사용 가능
    'all_features_enabled': True,
    'level_restrictions': False,  # 레벨 제한 없음
    
    # 교육적 기능
    'adaptive_education': True,
    'growth_tracking': True,
    'personalized_learning': True,
    'achievement_system': True,
    
    # 핵심 기능
    'ai_consensus': True,
    'advanced_optimization': True,
    'collaboration': True,
    'templates': True,
    'api_access': True,
    'export_features': True,
    
    # 시스템
    'maintenance_mode': False,
    'registration_enabled': True
}

# ===========================
# 15. 유틸리티 함수
# ===========================

def get_user_level_config(user_level: str = 'beginner') -> Dict:
    """사용자 레벨에 맞는 설정 반환"""
    return LEVEL_CONFIG['levels'].get(user_level, LEVEL_CONFIG['levels']['beginner'])

def get_educational_content(feature: str, topic: str, user_level: str) -> Optional[Dict]:
    """교육 콘텐츠 반환"""
    try:
        content = EDUCATIONAL_CONTENT['explanations'][feature][topic][user_level]
        return content
    except KeyError:
        return None

def should_show_explanation(user_level: str, feature: str) -> bool:
    """설명 표시 여부 결정"""
    level_config = get_user_level_config(user_level)
    edu_mode = level_config['educational_mode']
    
    if edu_mode['explanations'] == 'off':
        return False
    elif edu_mode['explanations'] == 'full':
        return True
    else:
        # 사용자 설정에 따라
        return st.session_state.get(f'show_explanation_{feature}', True)

def get_error_message(error_type: str, user_level: str) -> str:
    """레벨별 에러 메시지 반환"""
    return EDUCATIONAL_CONTENT['error_messages'][user_level].get(
        error_type,
        "An error occurred"
    )

def track_growth_metric(user_id: str, metric_type: str, value: float):
    """성장 지표 추적"""
    if GROWTH_TRACKING['enabled']:
        # 실제 구현은 sheets_manager를 통해
        logger.info(f"Growth metric tracked: {user_id} - {metric_type}: {value}")

def get_ai_prompt_style(user_level: str) -> Dict:
    """AI 프롬프트 스타일 반환"""
    return EDUCATIONAL_PROMPTS['ai_explanations'].get(
        user_level,
        EDUCATIONAL_PROMPTS['ai_explanations']['intermediate']
    )

# ===========================
# 16. 설정 검증
# ===========================

def validate_config() -> Tuple[bool, List[str]]:
    """설정 검증"""
    warnings = []
    
    # 필수 설정 확인
    if not GOOGLE_SHEETS_CONFIG.get('spreadsheet_id'):
        warnings.append("Google Sheets ID가 설정되지 않았습니다.")
    
    if not SECURITY_CONFIG.get('jwt_secret_key'):
        warnings.append("JWT Secret Key가 설정되지 않았습니다.")
    
    # API 키 확인 (경고만, 필수 아님)
    for api_name, config in API_KEYS.items():
        if config.get('key') == "":
            warnings.append(f"{api_name} API 키가 비어있습니다. (선택사항)")
    
    success = len([w for w in warnings if "선택사항" not in w]) == 0
    return success, warnings

# 설정 검증 실행
if __name__ != "__main__":
    success, warnings = validate_config()
    if warnings:
        for warning in warnings:
            logger.warning(warning)
    
    logger.info(f"교육적 성장 중심 플랫폼 설정 로드 완료 (환경: {ENV})")
