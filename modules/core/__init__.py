"""
핵심 내장 모듈 패키지
- 기본 제공 실험 모듈들
- 모든 코어 모듈의 중앙 관리
"""

import logging
from typing import Dict, Type, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# 필수 모듈 임포트
try:
    from .general_experiment import GeneralExperimentModule
    _general_available = True
except ImportError as e:
    logger.warning(f"GeneralExperimentModule 임포트 실패: {e}")
    GeneralExperimentModule = None
    _general_available = False

# 고분자 특화 모듈 임포트 (선택적)
try:
    from .polymer_experiment import PolymerExperimentModule
    _polymer_available = True
except ImportError:
    logger.info("PolymerExperimentModule은 아직 구현되지 않았습니다.")
    PolymerExperimentModule = None
    _polymer_available = False

# 추가 핵심 모듈들 (향후 확장용)
try:
    from .chemical_experiment import ChemicalExperimentModule
    _chemical_available = True
except ImportError:
    ChemicalExperimentModule = None
    _chemical_available = False

try:
    from .material_experiment import MaterialExperimentModule
    _material_available = True
except ImportError:
    MaterialExperimentModule = None
    _material_available = False

try:
    from .mixture_design import MixtureDesignModule
    _mixture_available = True
except ImportError:
    MixtureDesignModule = None
    _mixture_available = False

try:
    from .optimization_experiment import OptimizationExperimentModule
    _optimization_available = True
except ImportError:
    OptimizationExperimentModule = None
    _optimization_available = False


# 모든 핵심 모듈 목록
CORE_MODULES: Dict[str, Type] = {}

# 사용 가능한 모듈만 등록
if _general_available and GeneralExperimentModule:
    CORE_MODULES['general'] = GeneralExperimentModule

if _polymer_available and PolymerExperimentModule:
    CORE_MODULES['polymer'] = PolymerExperimentModule

if _chemical_available and ChemicalExperimentModule:
    CORE_MODULES['chemical'] = ChemicalExperimentModule

if _material_available and MaterialExperimentModule:
    CORE_MODULES['material'] = MaterialExperimentModule

if _mixture_available and MixtureDesignModule:
    CORE_MODULES['mixture'] = MixtureDesignModule

if _optimization_available and OptimizationExperimentModule:
    CORE_MODULES['optimization'] = OptimizationExperimentModule


# 모듈 정보 딕셔너리
MODULE_INFO = {
    'general': {
        'name': '범용 실험 설계',
        'description': '일반적인 실험 설계 (완전요인, 부분요인, RSM 등)',
        'category': 'general',
        'tags': ['factorial', 'rsm', 'screening', 'optimization'],
        'available': _general_available
    },
    'polymer': {
        'name': '고분자 실험 설계',
        'description': '고분자 특화 실험 (용매 시스템, 가공, 나노재료 등)',
        'category': 'materials',
        'tags': ['polymer', 'solvent', 'processing', 'nanofiber'],
        'available': _polymer_available
    },
    'chemical': {
        'name': '화학 실험 설계',
        'description': '화학 합성 및 반응 최적화',
        'category': 'chemistry',
        'tags': ['synthesis', 'reaction', 'catalyst'],
        'available': _chemical_available
    },
    'material': {
        'name': '재료 실험 설계',
        'description': '재료 특성 및 공정 최적화',
        'category': 'materials',
        'tags': ['materials', 'properties', 'processing'],
        'available': _material_available
    },
    'mixture': {
        'name': '혼합물 설계',
        'description': '다성분 혼합물 최적화 설계',
        'category': 'general',
        'tags': ['mixture', 'formulation', 'blend'],
        'available': _mixture_available
    },
    'optimization': {
        'name': '최적화 실험',
        'description': '다목적 최적화 및 강건 설계',
        'category': 'general',
        'tags': ['optimization', 'robust', 'multi-objective'],
        'available': _optimization_available
    }
}


def get_available_modules() -> Dict[str, Type]:
    """사용 가능한 모듈 목록 반환"""
    return {k: v for k, v in CORE_MODULES.items() if v is not None}


def get_module_info(module_id: str) -> Optional[Dict]:
    """모듈 정보 반환"""
    return MODULE_INFO.get(module_id)


def is_module_available(module_id: str) -> bool:
    """모듈 사용 가능 여부 확인"""
    info = MODULE_INFO.get(module_id, {})
    return info.get('available', False)


def list_core_modules() -> Dict[str, Dict]:
    """모든 코어 모듈 정보 목록"""
    return {
        module_id: {
            **info,
            'class': CORE_MODULES.get(module_id)
        }
        for module_id, info in MODULE_INFO.items()
    }


# 코어 모듈 자동 등록 함수
def register_core_modules(registry):
    """레지스트리에 코어 모듈 자동 등록
    
    Args:
        registry: ModuleRegistry 인스턴스
    """
    registered_count = 0
    
    for module_id, module_class in CORE_MODULES.items():
        if module_class:
            try:
                # 모듈 정보 가져오기
                info = MODULE_INFO.get(module_id, {})
                
                # 레지스트리에 등록
                module_path = Path(__file__).parent / f"{module_id}_experiment.py"
                success, error = registry.register_module(
                    module_path,
                    store_type='core',
                    validate=False  # 코어 모듈은 검증 스킵
                )
                
                if success:
                    registered_count += 1
                    logger.info(f"코어 모듈 등록 성공: {module_id}")
                else:
                    logger.error(f"코어 모듈 등록 실패 ({module_id}): {error}")
                    
            except Exception as e:
                logger.error(f"코어 모듈 등록 중 오류 ({module_id}): {e}")
    
    logger.info(f"총 {registered_count}개의 코어 모듈이 등록되었습니다.")
    return registered_count


# 초기화 시 사용 가능한 모듈 로그
logger.info(f"코어 모듈 패키지 초기화 - 사용 가능한 모듈: {list(CORE_MODULES.keys())}")


# 공개 API
__all__ = [
    # 모듈 클래스
    'GeneralExperimentModule',
    'PolymerExperimentModule',
    
    # 상수
    'CORE_MODULES',
    'MODULE_INFO',
    
    # 함수
    'get_available_modules',
    'get_module_info',
    'is_module_available',
    'list_core_modules',
    'register_core_modules'
]
