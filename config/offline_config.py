"""
🌐 Universal DOE Platform - 오프라인 모드 설정
================================================================================
데스크톱 애플리케이션의 오프라인 동작을 제어하는 상세 설정
오프라인 우선 설계로 인터넷 없이도 완전한 기능 제공
고분자 과학 특화 기능 및 문헌/DB 통합 지원
================================================================================
"""

from typing import Dict, List, Any, Optional, Callable, Tuple
from enum import Enum
from datetime import timedelta
from pathlib import Path
import json

# ============================================================================
# 🎯 오프라인 모드 레벨
# ============================================================================

class OfflineMode(Enum):
    """오프라인 모드 수준"""
    FULL_OFFLINE = "full_offline"      # 완전 오프라인 (인터넷 차단)
    OFFLINE_FIRST = "offline_first"    # 오프라인 우선 (기본값)
    ONLINE_PREFERRED = "online_preferred"  # 온라인 선호
    ONLINE_ONLY = "online_only"        # 온라인 전용 (특수 상황)

class SyncStrategy(Enum):
    """동기화 전략"""
    LOCAL_FIRST = "local_first"        # 로컬 우선 (기본값)
    REMOTE_FIRST = "remote_first"      # 원격 우선
    NEWEST_WINS = "newest_wins"        # 최신 우선
    MANUAL = "manual"                  # 수동 해결
    MERGE = "merge"                    # 자동 병합

# ============================================================================
# 🔧 기본 오프라인 설정
# ============================================================================

OFFLINE_CONFIG = {
    'mode': OfflineMode.OFFLINE_FIRST,  # 기본 모드
    'auto_detect_connection': True,      # 자동 연결 감지
    'connection_check_interval': 30,     # 초 단위
    'retry_interval': 60,               # 재시도 간격
    'max_offline_duration': None,       # 무제한
    'queue_online_requests': True,      # 온라인 요청 대기열
    
    # 연결 상태 체크
    'connectivity_check': {
        'method': 'multiple',  # ping, dns, http
        'timeout': 5,          # 초
        'endpoints': [
            'https://www.google.com',
            'https://cloudflare.com',
            'https://api.github.com'
        ],
        'fallback_to_cache': True,
    },
    
    # 오프라인 데이터 요구사항
    'required_data': {
        'core': ['algorithms.json', 'templates.db', 'base_modules.json'],
        'ai': ['ai_cache.db', 'ai_templates.json', 'response_patterns.json'],
        'literature': ['literature_cache.db', 'protocol_templates.json'],
        'polymer': ['polymer_templates.db', 'solvent_database.json', 'hansen_parameters.db'],
        'benchmark': ['benchmark_data.db', 'materials_properties.db'],
    },
}

# ============================================================================
# 🔌 기능별 오프라인 동작
# ============================================================================

FEATURE_OFFLINE_BEHAVIOR = {
    # 프로젝트 관리
    'project_management': {
        'offline_mode': 'full',  # 완전 지원
        'sync_required': False,
        'cache_ttl': None,  # 영구 저장
        'fallback': None,
        'require_online': [],
    },
    
    # 실험 설계
    'experiment_design': {
        'offline_mode': 'full',
        'rule_based': True,  # 규칙 기반 설계
        'ai_cache': True,    # AI 캐시 사용
        'templates': True,   # 템플릿 기반
        'require_online': ['ai_optimization', 'latest_algorithms'],
    },
    
    # 고분자 특화 기능
    'polymer_design': {
        'offline_mode': 'full',
        'hansen_parameters': True,  # 한센 매개변수 (로컬)
        'solvent_database': True,   # 용매 DB (로컬)
        'phase_diagrams': True,     # 상 다이어그램 (계산)
        'processing_optimization': True,  # 가공 최적화 (규칙)
        'require_online': ['latest_polymer_data', 'patent_search'],
    },
    
    # 데이터 분석
    'data_analysis': {
        'offline_mode': 'full',
        'statistical_analysis': True,
        'visualization': True,
        'report_generation': True,
        'require_online': ['cloud_computing', 'collaborative_analysis'],
    },
    
    # AI 기능
    'ai_features': {
        'offline_mode': 'cached',  # 캐시 기반
        'cache_responses': True,
        'rule_based_fallback': True,
        'template_responses': True,
        'local_models': False,  # 기본 비활성 (대용량)
        'max_cache_size_mb': 500,
        'cache_ttl': timedelta(days=30),
        'require_online': ['real_time_ai', 'model_updates'],
    },
    
    # 문헌 검색 (확장)
    'literature_search': {
        'offline_mode': 'cached',
        'cached_papers': 1000,  # 최대 캐시 논문 수
        'cached_protocols': 1000,  # 캐시된 프로토콜
        'local_index': True,  # 로컬 검색 인덱스
        'metadata_only': False,  # 전문 포함
        'sources': {
            'openalex': {'cache_size': 500, 'ttl_days': 30},
            'crossref': {'cache_size': 300, 'ttl_days': 30},
            'pubmed': {'cache_size': 200, 'ttl_days': 30},
            'arxiv': {'cache_size': 200, 'ttl_days': 14},
            'patents': {'cache_size': 100, 'ttl_days': 60},
        },
        'require_online': ['new_search', 'full_text_download', 'citation_network'],
    },
    
    # 프로토콜 추출
    'protocol_extraction': {
        'offline_mode': 'full',
        'pdf_processing': True,  # 로컬 PDF 처리
        'ocr_support': True,     # OCR 지원
        'nlp_extraction': True,  # NLP 추출
        'template_matching': True,  # 템플릿 매칭
        'cached_protocols': 1000,
        'require_online': ['cloud_ocr', 'advanced_nlp'],
    },
    
    # 벤치마크 분석
    'benchmark_analysis': {
        'offline_mode': 'cached',
        'materials_database': 10000,  # 캐시된 물성 데이터
        'comparison_metrics': True,
        'statistical_analysis': True,
        'trend_analysis': 'limited',  # 제한적
        'require_online': ['real_time_comparison', 'global_rankings'],
    },
    
    # 협업
    'collaboration': {
        'offline_mode': 'queued',  # 대기열에 저장
        'queue_actions': True,
        'local_comments': True,
        'sync_on_connect': True,
        'conflict_resolution': SyncStrategy.LOCAL_FIRST,
        'require_online': ['real_time_collaboration', 'video_call'],
    },
    
    # 모듈 마켓플레이스
    'marketplace': {
        'offline_mode': 'cached',
        'cached_modules': True,
        'installed_modules': 'full',  # 설치된 모듈은 완전 지원
        'browse_cached': True,
        'require_online': ['download_new', 'publish', 'reviews'],
    },
}

# ============================================================================
# 💾 오프라인 데이터 정책
# ============================================================================

OFFLINE_DATA_POLICY = {
    'storage': {
        'primary': 'sqlite',  # 주 저장소
        'backup': 'json',     # 백업 형식
        'encryption': True,   # 암호화 여부
        'compression': True,  # 압축 여부
    },
    
    'retention': {
        'user_data': None,           # 무제한
        'project_data': None,        # 무제한
        'experiment_data': None,     # 무제한
        'ai_cache': timedelta(days=90),
        'literature_cache': timedelta(days=180),
        'temp_data': timedelta(days=7),
        'logs': timedelta(days=30),
    },
    
    'size_limits': {
        'total_cache': 5 * 1024,     # 5GB
        'ai_cache': 500,             # 500MB
        'literature_cache': 2 * 1024, # 2GB
        'media_cache': 1024,         # 1GB
        'temp_files': 500,           # 500MB
    },
    
    'cleanup': {
        'auto_cleanup': True,
        'cleanup_interval': timedelta(days=7),
        'preserve_recent': timedelta(days=30),
        'lru_eviction': True,  # Least Recently Used
    },
}

# ============================================================================
# 🤖 AI 오프라인 전략
# ============================================================================

AI_OFFLINE_STRATEGY = {
    'fallback_chain': [
        'cached_response',      # 1. 캐시된 응답
        'template_response',    # 2. 템플릿 기반
        'rule_based',          # 3. 규칙 기반
        'local_model',         # 4. 로컬 모델 (선택적)
        'queued_request',      # 5. 대기열 추가
    ],
    
    'cache_strategy': {
        'hash_prompts': True,
        'fuzzy_matching': True,
        'similarity_threshold': 0.85,
        'cache_variations': True,
        'language_agnostic': False,
    },
    
    'template_responses': {
        'experiment_design': {
            'factorial': "2^k 요인 설계는 {k}개 요인에 대해 {runs}회 실행이 필요합니다.",
            'screening': "Plackett-Burman 설계로 {k}개 요인을 {runs}회 실행으로 스크리닝할 수 있습니다.",
            'optimization': "중심합성설계(CCD)는 {k}개 요인에 대해 {runs}회 실행이 필요합니다.",
        },
        'polymer': {
            'solvent': "한센 용해도 매개변수 기준으로 {polymer}에 적합한 용매는 {solvents}입니다.",
            'processing': "{polymer}의 권장 가공 온도는 {temp}°C, 압력은 {pressure} MPa입니다.",
        },
    },
    
    'rule_based_logic': {
        'experiment_design': [
            {
                'condition': lambda x: x['factors'] <= 3,
                'response': 'full_factorial',
                'reasoning': "요인이 3개 이하일 때는 완전요인설계가 효율적입니다."
            },
            {
                'condition': lambda x: x['factors'] > 7,
                'response': 'plackett_burman',
                'reasoning': "많은 요인의 스크리닝에는 Plackett-Burman이 적합합니다."
            },
        ],
    },
    
    'local_models': {
        'enabled': False,  # 기본 비활성
        'models': {
            'small_llm': {'size': '1GB', 'capability': 'basic'},
            'chemistry_bert': {'size': '500MB', 'capability': 'chemistry'},
            'materials_gpt': {'size': '2GB', 'capability': 'materials'},
        },
    },
}

# ============================================================================
# 🔄 동기화 설정
# ============================================================================

SYNC_CONFIGURATION = {
    'sync_priorities': [
        'user_projects',      # 1순위
        'experiment_results', # 2순위
        'analysis_data',      # 3순위
        'collaboration_queue', # 4순위
        'ai_responses',       # 5순위
        'literature_cache',   # 6순위
        'module_updates',     # 7순위
    ],
    
    'sync_triggers': {
        'on_connect': True,
        'on_disconnect': True,
        'periodic': timedelta(minutes=30),
        'on_idle': True,
        'manual': True,
    },
    
    'conflict_resolution': {
        'default_strategy': SyncStrategy.LOCAL_FIRST,
        'strategies_by_type': {
            'user_projects': SyncStrategy.NEWEST_WINS,
            'shared_data': SyncStrategy.MERGE,
            'system_config': SyncStrategy.REMOTE_FIRST,
        },
        'backup_before_sync': True,
        'user_confirmation': 'major_conflicts',
    },
    
    'bandwidth_management': {
        'limit_bandwidth': False,
        'max_bandwidth_mbps': 10,
        'compress_data': True,
        'delta_sync': True,
        'batch_size': 100,
    },
}

# ============================================================================
# 📊 오프라인 분석 설정
# ============================================================================

OFFLINE_ANALYTICS = {
    'enabled': True,
    'anonymous': True,
    'local_only': True,
    
    'track_events': [
        'feature_usage',
        'error_frequency',
        'performance_metrics',
        'offline_duration',
        'sync_success_rate',
    ],
    
    'storage': {
        'location': 'local',
        'format': 'sqlite',
        'retention': timedelta(days=90),
        'aggregate_only': True,
    },
    
    'reports': {
        'usage_summary': True,
        'error_report': True,
        'performance_report': True,
        'sync_report': True,
    },
}

# ============================================================================
# 🎨 오프라인 UI 설정
# ============================================================================

OFFLINE_UI_CONFIG = {
    'indicators': {
        'show_offline_badge': True,
        'badge_position': 'top-right',
        'badge_color': '#FF9800',
        'badge_text': '오프라인',
    },
    
    'disabled_features': {
        'show_overlay': True,
        'overlay_opacity': 0.6,
        'message': '온라인 연결이 필요합니다',
        'suggest_alternatives': True,
    },
    
    'sync_status': {
        'show_indicator': True,
        'show_queue_count': True,
        'show_last_sync': True,
        'allow_manual_sync': True,
    },
    
    'notifications': {
        'connection_lost': '인터넷 연결이 끊어졌습니다. 오프라인 모드로 전환합니다.',
        'connection_restored': '인터넷 연결이 복구되었습니다. 데이터를 동기화합니다.',
        'sync_complete': '동기화가 완료되었습니다.',
        'sync_failed': '동기화 실패. 나중에 다시 시도합니다.',
    },
}

# ============================================================================
# 🔒 오프라인 보안
# ============================================================================

OFFLINE_SECURITY = {
    'local_encryption': {
        'enabled': True,
        'algorithm': 'AES-256-GCM',
        'key_derivation': 'PBKDF2',
        'iterations': 100000,
        'encrypt_types': ['credentials', 'api_keys', 'personal_data', 'projects'],
    },
    
    'access_control': {
        'require_login': True,
        'session_timeout': timedelta(hours=24),
        'biometric_support': True,  # 지원하는 경우
        'pin_fallback': True,
        'auto_lock': timedelta(minutes=30),
    },
    
    'data_protection': {
        'secure_delete': True,
        'memory_encryption': False,  # 성능 영향
        'anti_tampering': True,
        'integrity_checks': True,
    },
}

# ============================================================================
# 📦 오프라인 데이터 패키지
# ============================================================================

OFFLINE_DATA_PACKAGES = {
    'core': {
        'name': '핵심 데이터',
        'version': '2.0.0',
        'size_mb': 100,
        'files': [
            'algorithms.json',
            'base_templates.db',
            'core_modules.json',
            'statistical_tables.db',
        ],
        'required': True,
        'description': '오프라인 작동에 필수적인 핵심 데이터',
    },
    
    'polymer': {
        'name': '고분자 과학 데이터',
        'version': '1.5.0',
        'size_mb': 250,
        'files': [
            'polymer_templates.db',  # 50+ 템플릿
            'hansen_parameters.db',  # 용해도 매개변수
            'solvent_database.json', # 용매 데이터베이스
            'phase_diagrams.db',     # 상 다이어그램
            'processing_data.json',  # 가공 조건
        ],
        'required': False,
        'description': '고분자 실험 설계를 위한 특화 데이터',
    },
    
    'ai_cache': {
        'name': 'AI 응답 캐시',
        'version': '2.0.0',
        'size_mb': 500,
        'files': [
            'ai_responses.db',
            'prompt_templates.json',
            'response_patterns.json',
            'embedding_cache.db',
        ],
        'required': False,
        'description': 'AI 기능을 위한 캐시 데이터',
    },
    
    'literature': {
        'name': '문헌 데이터베이스',
        'version': '2.0.0',
        'size_mb': 2000,
        'files': [
            'literature_cache.db',    # 1000+ 논문
            'protocol_library.db',    # 1000+ 프로토콜
            'citation_network.json',  # 인용 네트워크
            'abstract_index.db',      # 초록 인덱스
        ],
        'required': False,
        'description': '캐시된 문헌 및 프로토콜 데이터',
    },
    
    'materials': {
        'name': '재료 물성 데이터베이스',
        'version': '1.2.0',
        'size_mb': 500,
        'files': [
            'materials_properties.db',  # 10000+ 물성
            'benchmark_data.db',        # 벤치마크 데이터
            'structure_database.db',    # 구조 데이터
            'performance_metrics.json', # 성능 지표
        ],
        'required': False,
        'description': '재료 물성 및 벤치마크 데이터',
    },
    
    'templates': {
        'name': '실험 템플릿 라이브러리',
        'version': '2.0.0',
        'size_mb': 50,
        'files': [
            'experiment_templates.db',  # 100+ 템플릿
            'analysis_templates.json',  # 분석 템플릿
            'report_templates.db',      # 보고서 템플릿
            'visualization_presets.json', # 시각화 프리셋
        ],
        'required': True,
        'description': '다양한 실험 설계 템플릿',
    },
}

# ============================================================================
# 🛠️ 오프라인 유틸리티
# ============================================================================

class OfflineManager:
    """오프라인 모드 관리 클래스"""
    
    @staticmethod
    def is_feature_available(feature: str, is_online: bool = False) -> bool:
        """특정 기능의 오프라인 사용 가능 여부"""
        if is_online:
            return True
            
        feature_config = FEATURE_OFFLINE_BEHAVIOR.get(feature, {})
        offline_mode = feature_config.get('offline_mode', 'disabled')
        
        return offline_mode in ['full', 'cached', 'limited', 'queued']
    
    @staticmethod
    def get_offline_message(feature: str) -> str:
        """오프라인 상태 메시지"""
        messages = {
            'ai_chat': "AI 채팅은 캐시된 응답과 로컬 모델을 사용합니다.",
            'literature_search': "저장된 문헌만 검색 가능합니다. 새로운 검색은 온라인 연결이 필요합니다.",
            'collaboration': "변경사항은 로컬에 저장되며 온라인 시 동기화됩니다.",
            'marketplace': "설치된 모듈과 캐시된 목록만 사용 가능합니다.",
            'protocol_extraction': "로컬 PDF 파일에서 프로토콜을 추출할 수 있습니다.",
            'benchmark_analysis': "캐시된 데이터베이스와 비교 분석이 가능합니다.",
        }
        return messages.get(feature, "이 기능은 오프라인에서 제한됩니다.")
    
    @staticmethod
    def get_sync_priority(data_type: str) -> int:
        """데이터 타입별 동기화 우선순위"""
        priorities = SYNC_CONFIGURATION['sync_priorities']
        try:
            return priorities.index(data_type)
        except ValueError:
            return len(priorities)  # 최저 우선순위
    
    @staticmethod
    def should_cache_response(feature: str, response_size: int) -> bool:
        """응답 캐싱 여부 결정"""
        feature_config = FEATURE_OFFLINE_BEHAVIOR.get(feature, {})
        if not feature_config.get('cache_responses', False):
            return False
            
        max_size = feature_config.get('max_cache_size_mb', 100) * 1024 * 1024
        return response_size < max_size
    
    @staticmethod
    def validate_offline_readiness(data_dir: Path) -> Tuple[bool, List[str]]:
        """오프라인 준비 상태 검증"""
        missing = []
        
        # 필수 데이터 확인
        for category, files in OFFLINE_CONFIG['required_data'].items():
            for file in files:
                file_path = data_dir / category / file
                if not file_path.exists():
                    missing.append(f"{category}/{file}")
        
        # 데이터 패키지 확인
        for package_id, package in OFFLINE_DATA_PACKAGES.items():
            if package['required']:
                for file in package['files']:
                    file_path = data_dir / 'packages' / package_id / file
                    if not file_path.exists():
                        missing.append(f"packages/{package_id}/{file}")
        
        return len(missing) == 0, missing
    
    @staticmethod
    def get_cache_stats() -> Dict[str, Any]:
        """캐시 통계 정보"""
        stats = {
            'total_size_mb': 0,
            'by_category': {},
            'oldest_entry': None,
            'newest_entry': None,
            'hit_rate': 0,
        }
        
        # 실제 구현은 database_manager와 연동
        return stats

# ============================================================================
# 🎯 오프라인 전략 함수
# ============================================================================

def get_offline_strategy(feature: str) -> Dict[str, Any]:
    """기능별 오프라인 전략 반환"""
    return FEATURE_OFFLINE_BEHAVIOR.get(feature, {
        'offline_mode': 'disabled',
        'require_online': [feature]
    })

def is_offline_first() -> bool:
    """오프라인 우선 모드 여부"""
    return OFFLINE_CONFIG['mode'] in [
        OfflineMode.FULL_OFFLINE,
        OfflineMode.OFFLINE_FIRST
    ]

def get_cache_ttl(feature: str) -> Optional[timedelta]:
    """기능별 캐시 TTL 반환"""
    feature_config = FEATURE_OFFLINE_BEHAVIOR.get(feature, {})
    return feature_config.get('cache_ttl')

def get_sync_strategy(data_type: str) -> SyncStrategy:
    """데이터 타입별 동기화 전략"""
    strategies = SYNC_CONFIGURATION['conflict_resolution']['strategies_by_type']
    default = SYNC_CONFIGURATION['conflict_resolution']['default_strategy']
    return strategies.get(data_type, default)

def get_required_packages(features: List[str]) -> List[str]:
    """기능에 필요한 오프라인 패키지 목록"""
    required = set(['core'])  # 항상 필요
    
    feature_package_map = {
        'polymer_design': 'polymer',
        'ai_features': 'ai_cache',
        'literature_search': 'literature',
        'benchmark_analysis': 'materials',
        'experiment_design': 'templates',
    }
    
    for feature in features:
        if feature in feature_package_map:
            required.add(feature_package_map[feature])
    
    return list(required)

# ============================================================================
# 📤 Export
# ============================================================================

__all__ = [
    # Enums
    'OfflineMode',
    'SyncStrategy',
    
    # 설정 딕셔너리
    'OFFLINE_CONFIG',
    'FEATURE_OFFLINE_BEHAVIOR',
    'OFFLINE_DATA_POLICY',
    'AI_OFFLINE_STRATEGY',
    'SYNC_CONFIGURATION',
    'OFFLINE_ANALYTICS',
    'OFFLINE_UI_CONFIG',
    'OFFLINE_SECURITY',
    'OFFLINE_DATA_PACKAGES',
    
    # 클래스
    'OfflineManager',
    
    # 함수
    'get_offline_strategy',
    'is_offline_first',
    'get_cache_ttl',
    'get_sync_strategy',
    'get_required_packages',
]
