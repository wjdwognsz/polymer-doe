# modules/base_module.py

"""
Universal DOE Platform - Base Module System
모든 실험 모듈의 기본 추상 클래스와 데이터 타입 정의
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime
import json


# ==================== 데이터 타입 정의 ====================

class FactorType(Enum):
    """실험 요인의 타입"""
    CONTINUOUS = "continuous"      # 연속형 (온도, 압력 등)
    CATEGORICAL = "categorical"    # 범주형 (촉매 종류, 용매 등)
    DISCRETE = "discrete"         # 이산형 (개수, 횟수 등)
    ORDINAL = "ordinal"          # 순서형 (낮음/중간/높음 등)


class ResponseGoal(Enum):
    """반응변수의 목표"""
    MAXIMIZE = "maximize"         # 최대화 (수율, 강도 등)
    MINIMIZE = "minimize"         # 최소화 (비용, 불순물 등)
    TARGET = "target"            # 목표값 (특정 pH, 점도 등)
    CONSTRAINT = "constraint"     # 제약조건 (안전 범위 등)


@dataclass
class Factor:
    """실험 요인 정의"""
    name: str                    # 요인 이름
    type: FactorType            # 요인 타입
    unit: str = ""              # 단위
    min_value: Optional[float] = None      # 최소값 (연속형)
    max_value: Optional[float] = None      # 최대값 (연속형)
    levels: List[Any] = field(default_factory=list)  # 수준 (범주형)
    default_value: Optional[Any] = None    # 기본값
    description: str = ""        # 설명
    constraints: Dict[str, Any] = field(default_factory=dict)  # 제약사항
    
    def validate(self) -> Tuple[bool, Optional[str]]:
        """요인 유효성 검증"""
        if not self.name:
            return False, "요인 이름이 필요합니다"
        
        if self.type == FactorType.CONTINUOUS:
            if self.min_value is None or self.max_value is None:
                return False, f"{self.name}: 연속형 요인은 최소/최대값이 필요합니다"
            if self.min_value >= self.max_value:
                return False, f"{self.name}: 최소값은 최대값보다 작아야 합니다"
        
        elif self.type in [FactorType.CATEGORICAL, FactorType.ORDINAL]:
            if not self.levels:
                return False, f"{self.name}: 범주형 요인은 수준이 필요합니다"
            if len(self.levels) < 2:
                return False, f"{self.name}: 최소 2개 이상의 수준이 필요합니다"
        
        return True, None


@dataclass
class Response:
    """반응변수 정의"""
    name: str                    # 반응변수 이름
    goal: ResponseGoal          # 목표
    unit: str = ""              # 단위
    target_value: Optional[float] = None   # 목표값 (TARGET 타입)
    tolerance: Optional[float] = None      # 허용 오차
    min_acceptable: Optional[float] = None # 최소 허용값
    max_acceptable: Optional[float] = None # 최대 허용값
    weight: float = 1.0         # 가중치 (다목적 최적화)
    description: str = ""        # 설명
    analysis_method: str = ""    # 분석 방법
    
    def validate(self) -> Tuple[bool, Optional[str]]:
        """반응변수 유효성 검증"""
        if not self.name:
            return False, "반응변수 이름이 필요합니다"
        
        if self.goal == ResponseGoal.TARGET and self.target_value is None:
            return False, f"{self.name}: 목표값 타입은 target_value가 필요합니다"
        
        if self.weight <= 0:
            return False, f"{self.name}: 가중치는 양수여야 합니다"
        
        return True, None


@dataclass
class ExperimentDesign:
    """실험 설계 결과"""
    design_matrix: pd.DataFrame  # 설계 행렬
    design_type: str            # 설계 유형 (완전요인, 부분요인 등)
    run_order: List[int]        # 실행 순서
    blocks: Optional[List[int]] = None  # 블록 설계
    center_points: int = 0      # 중심점 개수
    replicates: int = 1         # 반복 횟수
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_runs(self) -> int:
        """총 실험 횟수"""
        return len(self.design_matrix) * self.replicates


@dataclass
class AnalysisResult:
    """분석 결과"""
    summary: Dict[str, Any]     # 요약 통계
    models: Dict[str, Any]      # 통계 모델
    plots: Dict[str, Any]       # 그래프
    recommendations: List[str]   # 권장사항
    optimal_conditions: Optional[Dict[str, Any]] = None
    confidence_intervals: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==================== 기본 모듈 클래스 ====================

class BaseExperimentModule(ABC):
    """모든 실험 모듈의 추상 기본 클래스"""
    
    def __init__(self):
        """모듈 초기화"""
        self.metadata = {
            'module_id': '',
            'name': 'Base Experiment Module',
            'version': '1.0.0',
            'author': '',
            'created_date': datetime.now().isoformat(),
            'last_modified': datetime.now().isoformat(),
            'description': '',
            'category': '',
            'tags': [],
            'dependencies': [],
            'min_platform_version': '1.0.0',
            'max_platform_version': None,
            'icon': '🧪',
            'color': '#1f77b4'
        }
        
        self.config = {}
        self._initialized = False
        
    # ==================== 추상 메서드 (필수 구현) ====================
    
    @abstractmethod
    def get_factors(self) -> List[Factor]:
        """
        실험 요인 목록 반환
        
        Returns:
            List[Factor]: 실험 요인 리스트
        """
        pass
    
    @abstractmethod
    def get_responses(self) -> List[Response]:
        """
        반응변수 목록 반환
        
        Returns:
            List[Response]: 반응변수 리스트
        """
        pass
    
    @abstractmethod
    def validate_input(self, inputs: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        사용자 입력값 검증
        
        Args:
            inputs: 사용자 입력값 딕셔너리
            
        Returns:
            Tuple[bool, Optional[str]]: (유효성, 에러 메시지)
        """
        pass
    
    @abstractmethod
    def generate_design(self, inputs: Dict[str, Any]) -> ExperimentDesign:
        """
        실험 설계 생성
        
        Args:
            inputs: 설계 파라미터
            
        Returns:
            ExperimentDesign: 생성된 실험 설계
        """
        pass
    
    @abstractmethod
    def analyze_results(self, data: pd.DataFrame) -> AnalysisResult:
        """
        실험 결과 분석
        
        Args:
            data: 실험 데이터
            
        Returns:
            AnalysisResult: 분석 결과
        """
        pass
    
    # ==================== 공통 메서드 ====================
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """모듈 초기화"""
        if config:
            self.config.update(config)
        self._initialized = True
        
    def is_initialized(self) -> bool:
        """초기화 상태 확인"""
        return self._initialized
    
    def get_metadata(self) -> Dict[str, Any]:
        """메타데이터 반환"""
        return self.metadata.copy()
    
    def set_metadata(self, key: str, value: Any) -> None:
        """메타데이터 설정"""
        if key in self.metadata:
            self.metadata[key] = value
            self.metadata['last_modified'] = datetime.now().isoformat()
    
    def get_config(self) -> Dict[str, Any]:
        """설정 반환"""
        return self.config.copy()
    
    def set_config(self, key: str, value: Any) -> None:
        """설정 업데이트"""
        self.config[key] = value
        
    def validate_factors(self, factors: List[Factor]) -> Tuple[bool, List[str]]:
        """요인 목록 검증"""
        errors = []
        for factor in factors:
            valid, error = factor.validate()
            if not valid:
                errors.append(error)
        return len(errors) == 0, errors
    
    def validate_responses(self, responses: List[Response]) -> Tuple[bool, List[str]]:
        """반응변수 목록 검증"""
        errors = []
        for response in responses:
            valid, error = response.validate()
            if not valid:
                errors.append(error)
        return len(errors) == 0, errors
    
    def get_default_design_options(self) -> Dict[str, Any]:
        """기본 설계 옵션 반환"""
        return {
            'design_type': 'full_factorial',
            'center_points': 0,
            'replicates': 1,
            'blocks': None,
            'randomize': True,
            'seed': None
        }
    
    def export_metadata(self, filepath: str) -> None:
        """메타데이터를 JSON 파일로 내보내기"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def import_metadata(self, filepath: str) -> None:
        """JSON 파일에서 메타데이터 가져오기"""
        with open(filepath, 'r', encoding='utf-8') as f:
            imported = json.load(f)
            # 버전 호환성 체크
            if 'min_platform_version' in imported:
                self.metadata.update(imported)
                self.metadata['last_modified'] = datetime.now().isoformat()
    
    def __str__(self) -> str:
        """문자열 표현"""
        return f"{self.metadata['name']} v{self.metadata['version']} by {self.metadata['author']}"
    
    def __repr__(self) -> str:
        """개발자용 표현"""
        return f"<{self.__class__.__name__}(name='{self.metadata['name']}', version='{self.metadata['version']}')>"


# ==================== 헬퍼 클래스 ====================

class ModuleValidationError(Exception):
    """모듈 검증 오류"""
    pass


class ModuleCompatibilityError(Exception):
    """모듈 호환성 오류"""
    pass


# ==================== 유틸리티 함수 ====================

def check_module_compatibility(module: BaseExperimentModule, 
                             platform_version: str) -> Tuple[bool, Optional[str]]:
    """
    모듈과 플랫폼 버전 호환성 확인
    
    Args:
        module: 확인할 모듈
        platform_version: 현재 플랫폼 버전
        
    Returns:
        Tuple[bool, Optional[str]]: (호환 여부, 에러 메시지)
    """
    from packaging import version
    
    current = version.parse(platform_version)
    min_ver = version.parse(module.metadata.get('min_platform_version', '0.0.0'))
    
    if current < min_ver:
        return False, f"플랫폼 버전 {min_ver} 이상이 필요합니다"
    
    max_ver = module.metadata.get('max_platform_version')
    if max_ver:
        max_ver = version.parse(max_ver)
        if current > max_ver:
            return False, f"플랫폼 버전 {max_ver} 이하에서만 작동합니다"
    
    return True, None
