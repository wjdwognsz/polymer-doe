"""
🌐 Universal DOE Platform - 오프라인 모드 설정
================================================================================
데스크톱 애플리케이션의 오프라인 동작을 제어하는 상세 설정
오프라인 우선 설계로 인터넷 없이도 완전한 기능 제공
================================================================================
"""

from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from datetime import timedelta
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
    'show_offline_indicator': True,     # UI에 오프라인 상태 표시
    'enable_offline_analytics': True,   # 오프라인 분석 데이터 수집
}

# ============================================================================
# 🚀 기능별 오프라인 동작
# ============================================================================

FEATURE_OFFLINE_BEHAVIOR = {
    # AI 기능
    'ai_chat': {
        'offline_mode': 'cached',  # cached, limited, disabled
        'cache_responses': True,
        'cache_ttl': timedelta(days=30),
        'fallback_responses': True,
        'local_models': ['basic_doe', 'statistics'],  # 로컬 모델
        'max_cache_size_mb': 100,
        'smart_suggestions': True,  # 캐시 기반 제안
    },
    
    # 실험 설계
    'experiment_design': {
        'offline_mode': 'full',  # 완전 지원
        'local_algorithms': [
            'full_factorial',
            'fractional_factorial', 
            'central_composite',
            'box_behnken',
            'plackett_burman',
            'latin_hypercube',
            'd_optimal'
        ],
        'require_online': [],  # 온라인 필수 기능 없음
        'cache_designs': True,
    },
    
    # 데이터 분석
    'data_analysis': {
        'offline_mode': 'full',
        'local_statistics': True,
        'local_ml_models': True,
        'visualization': 'full',
        'export_formats': ['excel', 'csv', 'pdf', 'html'],
        'require_online': ['cloud_ml_models'],
    },
    
    # 문헌 검색
    'literature_search': {
        'offline_mode': 'cached',
        'cache_papers': True,
        'cache_ttl': timedelta(days=7),
        'offline_search': 'local_index',  # 로컬 인덱스 검색
        'max_cache_papers': 1000,
        'require_online': ['new_search', 'full_text_download'],
    },
    
    # 협업
    'collaboration': {
        'offline_mode': 'queued',  # 대기열에 저장
        'queue_actions': True,
        'local_comments': True,
        'sync_on_connect': True,
        'conflict_resolution': SyncStrategy.LOCAL_FIRST,
        'require_online': ['realtime_collaboration', 'video_call'],
    },
    
    # 모듈 마켓플레이스
    'marketplace': {
        'offline_mode': 'cached',
        'cached_modules': True,
        'installed_modules': 'full',  # 설치된 모듈은 완전 지원
        'browse_cached': True,
        'require_online': ['download_new', 'publish'],
    },
    
    # 업데이트
    'updates': {
        'offline_mode': 'manual',
        'check_on_connect': True,
        'download_in_background': True,
        'install_offline': True,
        'require_online': ['check_updates', 'download'],
    },
}

# ============================================================================
# 💾 오프라인 데이터 관리
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
        'cache_data': timedelta(days=90),
        'temp_data': timedelta(days=7),
        'log_data': timedelta(days=30),
    },
    
    'size_limits': {
        'total_size_gb': 10,
        'cache_size_gb': 2,
        'backup_size_gb': 5,
        'alert_threshold': 0.8,  # 80% 도달 시 경고
    },
    
    'cleanup': {
        'auto_cleanup': True,
        'cleanup_interval': timedelta(days=1),
        'priorities': ['temp', 'cache', 'logs', 'old_backups'],
    },
}

# ============================================================================
# 🤖 AI 오프라인 전략
# ============================================================================

AI_OFFLINE_STRATEGY = {
    'response_cache': {
        'enabled': True,
        'storage_path': 'cache/ai_responses',
        'index_file': 'cache/ai_index.json',
        'max_entries': 10000,
        'eviction_policy': 'LRU',  # Least Recently Used
        'similarity_threshold': 0.85,  # 유사 질문 매칭
    },
    
    'fallback_models': {
        'basic_statistics': {
            'type': 'rule_based',
            'capabilities': ['mean', 'median', 'std', 'correlation'],
        },
        'doe_templates': {
            'type': 'template_based',
            'templates_path': 'resources/doe_templates.json',
        },
        'error_diagnosis': {
            'type': 'decision_tree',
            'model_path': 'resources/error_tree.pkl',
        },
    },
    
    'smart_suggestions': {
        'enabled': True,
        'based_on': ['history', 'similar_projects', 'common_patterns'],
        'max_suggestions': 5,
    },
    
    'offline_prompts': {
        'no_connection': "오프라인 모드입니다. 캐시된 응답과 로컬 분석을 사용합니다.",
        'limited_features': "일부 AI 기능이 제한됩니다. 기본 분석은 가능합니다.",
        'cached_response': "이전에 유사한 질문에 대한 응답입니다.",
    },
}

# ============================================================================
# 🔄 동기화 설정
# ============================================================================

SYNC_CONFIGURATION = {
    'auto_sync': {
        'enabled': True,
        'on_connection_restore': True,
        'on_app_start': False,  # 시작 시 자동 동기화 안 함
        'on_app_close': True,
        'interval': timedelta(minutes=5),
    },
    
    'sync_priorities': [
        'user_settings',      # 1순위
        'project_metadata',   # 2순위
        'experiment_results', # 3순위
        'analysis_data',      # 4순위
        'comments',          # 5순위
        'cache_data',        # 6순위
    ],
    
    'conflict_resolution': {
        'default_strategy': SyncStrategy.LOCAL_FIRST,
        'strategies_by_type': {
            'user_settings': SyncStrategy.NEWEST_WINS,
            'project_data': SyncStrategy.LOCAL_FIRST,
            'shared_data': SyncStrategy.MANUAL,
            'system_config': SyncStrategy.REMOTE_FIRST,
        },
        'auto_backup_before_sync': True,
    },
    
    'queue_management': {
        'max_queue_size': 1000,
        'queue_persistence': True,
        'retry_failed': True,
        'max_retries': 3,
        'retry_delay': timedelta(minutes=1),
    },
}

# ============================================================================
# 📊 오프라인 분석
# ============================================================================

OFFLINE_ANALYTICS = {
    'local_processing': {
        'statistical_tests': [
            't_test', 'anova', 'chi_square', 'correlation',
            'regression', 'normality_test', 'outlier_detection'
        ],
        'ml_algorithms': [
            'linear_regression', 'logistic_regression',
            'decision_tree', 'random_forest', 'kmeans'
        ],
        'optimization': [
            'gradient_descent', 'genetic_algorithm',
            'simulated_annealing', 'grid_search'
        ],
    },
    
    'visualization': {
        'chart_types': 'all',  # 모든 차트 타입 지원
        'export_formats': ['png', 'svg', 'pdf', 'html'],
        'interactive': True,
        'max_data_points': 100000,
    },
    
    'reporting': {
        'templates': 'local',
        'formats': ['html', 'pdf', 'docx', 'pptx'],
        'include_code': True,
        'include_data': True,
    },
}

# ============================================================================
# 🎨 UI 오프라인 표시
# ============================================================================

OFFLINE_UI_CONFIG = {
    'indicators': {
        'show_badge': True,
        'badge_position': 'top-right',
        'badge_color': '#FFA500',  # 주황색
        'badge_text': '오프라인',
        'tooltip': '인터넷 연결 없음. 로컬 기능만 사용 가능.',
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
        'algorithm': 'AES-256',
        'key_derivation': 'PBKDF2',
        'encrypt_types': ['credentials', 'api_keys', 'personal_data'],
    },
    
    'access_control': {
        'require_login': True,
        'session_timeout': timedelta(hours=24),
        'biometric_support': True,  # 지원하는 경우
        'pin_fallback': True,
    },
    
    'data_protection': {
        'secure_delete': True,
        'memory_encryption': False,  # 성능 영향
        'anti_tampering': True,
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
            'literature_search': "저장된 문헌만 검색 가능합니다.",
            'collaboration': "변경사항은 로컬에 저장되며 온라인 시 동기화됩니다.",
            'marketplace': "설치된 모듈과 캐시된 목록만 사용 가능합니다.",
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

# ============================================================================
# 📤 Export
# ============================================================================

__all__ = [
    'OFFLINE_CONFIG',
    'FEATURE_OFFLINE_BEHAVIOR',
    'OFFLINE_DATA_POLICY',
    'AI_OFFLINE_STRATEGY',
    'SYNC_CONFIGURATION',
    'OFFLINE_ANALYTICS',
    'OFFLINE_UI_CONFIG',
    'OFFLINE_SECURITY',
    'OfflineMode',
    'SyncStrategy',
    'OfflineManager',
    'get_offline_strategy',
    'is_offline_first',
    'get_cache_ttl',
    'get_sync_strategy'
]
