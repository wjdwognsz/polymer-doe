"""
모든 실험 모듈의 기본 추상 클래스
플랫폼의 무한 확장성을 보장하는 핵심 아키텍처 컴포넌트
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np
import json
from pathlib import Path
from pydantic import BaseModel, Field, validator
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ==================== Enums ====================

class FactorType(str, Enum):
    """요인 타입"""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"


class ResponseType(str, Enum):
    """반응변수 타입"""
    NUMERIC = "numeric"
    BINARY = "binary"
    ORDINAL = "ordinal"
    CATEGORICAL = "categorical"


class OptimizationGoal(str, Enum):
    """최적화 목표"""
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
    TARGET = "target"
    NONE = "none"


class DesignType(str, Enum):
    """실험 설계 유형"""
    FULL_FACTORIAL = "full_factorial"
    FRACTIONAL_FACTORIAL = "fractional_factorial"
    CCD = "central_composite"
    BOX_BEHNKEN = "box_behnken"
    PLACKETT_BURMAN = "plackett_burman"
    LATIN_HYPERCUBE = "latin_hypercube"
    D_OPTIMAL = "d_optimal"
    CUSTOM = "custom"


# ==================== 데이터 모델 ====================

class Factor(BaseModel):
    """실험 요인 모델"""
    name: str = Field(..., description="요인 이름")
    display_name: Optional[str] = Field(None, description="표시 이름")
    type: FactorType = Field(..., description="요인 타입")
    unit: Optional[str] = Field(None, description="단위")
    
    # 연속형 요인
    min_value: Optional[float] = Field(None, description="최소값 (연속형)")
    max_value: Optional[float] = Field(None, description="최대값 (연속형)")
    
    # 이산형/범주형 요인
    levels: Optional[List[Any]] = Field(None, description="레벨 목록")
    
    # 추가 속성
    description: Optional[str] = Field(None, description="설명")
    importance: Optional[str] = Field("medium", description="중요도: high/medium/low")
    controllability: Optional[str] = Field("full", description="제어가능성: full/partial/noise")
    measurement_precision: Optional[float] = Field(None, description="측정 정밀도")
    cost_per_level: Optional[float] = Field(None, description="레벨당 비용")
    constraints: Optional[Dict[str, Any]] = Field(default_factory=dict, description="제약조건")
    
    @validator('levels')
    def validate_levels(cls, v, values):
        """레벨 검증"""
        if values.get('type') in [FactorType.CATEGORICAL, FactorType.DISCRETE, FactorType.ORDINAL]:
            if not v or len(v) < 2:
                raise ValueError("범주형/이산형 요인은 최소 2개 이상의 레벨이 필요합니다")
        return v
    
    @validator('max_value')
    def validate_range(cls, v, values):
        """범위 검증"""
        if values.get('type') == FactorType.CONTINUOUS:
            min_val = values.get('min_value')
            if min_val is not None and v is not None and v <= min_val:
                raise ValueError("최대값은 최소값보다 커야 합니다")
        return v
    
    def get_coded_levels(self, n_levels: int = 2) -> List[float]:
        """코드화된 레벨 반환"""
        if self.type == FactorType.CONTINUOUS:
            if n_levels == 2:
                return [-1, 1]
            elif n_levels == 3:
                return [-1, 0, 1]
            else:
                return list(np.linspace(-1, 1, n_levels))
        else:
            return list(range(len(self.levels))) if self.levels else []
    
    class Config:
        use_enum_values = True


class Response(BaseModel):
    """반응변수 모델"""
    name: str = Field(..., description="반응변수 이름")
    display_name: Optional[str] = Field(None, description="표시 이름")
    type: ResponseType = Field(ResponseType.NUMERIC, description="반응변수 타입")
    unit: Optional[str] = Field(None, description="단위")
    
    # 최적화 설정
    goal: OptimizationGoal = Field(OptimizationGoal.NONE, description="최적화 목표")
    target_value: Optional[float] = Field(None, description="목표값 (goal=target인 경우)")
    lower_bound: Optional[float] = Field(None, description="하한")
    upper_bound: Optional[float] = Field(None, description="상한")
    weight: float = Field(1.0, ge=0, le=1, description="가중치 (0-1)")
    
    # 측정 정보
    measurement_method: Optional[str] = Field(None, description="측정 방법")
    equipment_required: Optional[List[str]] = Field(default_factory=list, description="필요 장비")
    measurement_time: Optional[float] = Field(None, description="측정 소요 시간 (분)")
    measurement_cost: Optional[float] = Field(None, description="측정 비용")
    
    # 추가 속성
    description: Optional[str] = Field(None, description="설명")
    precision: Optional[float] = Field(None, description="측정 정밀도")
    accuracy: Optional[float] = Field(None, description="측정 정확도")
    
    @validator('target_value')
    def validate_target(cls, v, values):
        """목표값 검증"""
        if values.get('goal') == OptimizationGoal.TARGET and v is None:
            raise ValueError("목표 최적화 시 target_value가 필요합니다")
        return v
    
    class Config:
        use_enum_values = True


class ExperimentDesign(BaseModel):
    """실험 설계 모델"""
    design_type: DesignType = Field(..., description="설계 유형")
    factors: List[Factor] = Field(..., description="요인 목록")
    responses: List[Response] = Field(..., description="반응변수 목록")
    runs: pd.DataFrame = Field(..., description="실험 런 매트릭스")
    
    # 설계 속성
    n_runs: int = Field(..., description="총 실험 횟수")
    n_center_points: int = Field(0, description="중심점 수")
    n_blocks: int = Field(1, description="블록 수")
    randomized: bool = Field(True, description="랜덤화 여부")
    
    # 설계 품질 지표
    d_efficiency: Optional[float] = Field(None, description="D-효율성")
    g_efficiency: Optional[float] = Field(None, description="G-효율성")
    condition_number: Optional[float] = Field(None, description="조건수")
    vif_max: Optional[float] = Field(None, description="최대 VIF")
    
    # 메타데이터
    created_at: datetime = Field(default_factory=datetime.now)
    created_by: Optional[str] = Field(None, description="생성자")
    notes: Optional[str] = Field(None, description="설계 노트")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True
    
    def to_excel(self, filepath: Union[str, Path]) -> None:
        """Excel 파일로 내보내기"""
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # 실험 런
            self.runs.to_excel(writer, sheet_name='Design Matrix', index=True)
            
            # 요인 정보
            factors_df = pd.DataFrame([f.dict() for f in self.factors])
            factors_df.to_excel(writer, sheet_name='Factors', index=False)
            
            # 반응변수 정보
            responses_df = pd.DataFrame([r.dict() for r in self.responses])
            responses_df.to_excel(writer, sheet_name='Responses', index=False)
            
            # 설계 정보
            info_dict = {
                'Design Type': self.design_type,
                'Total Runs': self.n_runs,
                'Center Points': self.n_center_points,
                'Blocks': self.n_blocks,
                'Randomized': self.randomized,
                'D-Efficiency': self.d_efficiency,
                'Created At': self.created_at.isoformat()
            }
            info_df = pd.DataFrame([info_dict])
            info_df.to_excel(writer, sheet_name='Design Info', index=False)


class AnalysisResult(BaseModel):
    """분석 결과 모델"""
    # 기본 통계
    summary_statistics: Dict[str, Dict[str, float]] = Field(
        default_factory=dict, 
        description="요약 통계 (평균, 표준편차 등)"
    )
    
    # 통계 분석
    anova_tables: Optional[Dict[str, pd.DataFrame]] = Field(None, description="ANOVA 테이블")
    regression_models: Optional[Dict[str, Any]] = Field(None, description="회귀 모델")
    main_effects: Optional[Dict[str, Dict[str, float]]] = Field(None, description="주효과")
    interactions: Optional[Dict[str, Dict[str, float]]] = Field(None, description="교호작용")
    
    # 최적화 결과
    optimal_conditions: Optional[Dict[str, Any]] = Field(None, description="최적 조건")
    predicted_responses: Optional[Dict[str, float]] = Field(None, description="예측값")
    desirability: Optional[float] = Field(None, description="바람직성 점수")
    
    # 진단
    diagnostics: Dict[str, Any] = Field(default_factory=dict, description="모델 진단")
    residuals: Optional[pd.DataFrame] = Field(None, description="잔차")
    outliers: Optional[List[int]] = Field(None, description="이상치 인덱스")
    
    # 시각화
    plots: Dict[str, Any] = Field(default_factory=dict, description="생성된 플롯")
    
    # 권장사항
    recommendations: List[str] = Field(default_factory=list, description="분석 권장사항")
    next_experiments: Optional[List[Dict[str, Any]]] = Field(None, description="추천 다음 실험")
    
    # 메타데이터
    analysis_type: str = Field("full", description="분석 유형")
    analysis_time: float = Field(0, description="분석 소요 시간 (초)")
    confidence_level: float = Field(0.95, description="신뢰 수준")
    
    class Config:
        arbitrary_types_allowed = True


class ValidationResult(BaseModel):
    """검증 결과 모델"""
    is_valid: bool = Field(True, description="전체 유효성")
    errors: List[str] = Field(default_factory=list, description="오류 목록")
    warnings: List[str] = Field(default_factory=list, description="경고 목록")
    suggestions: List[str] = Field(default_factory=list, description="개선 제안")
    
    # 상세 검증 결과
    statistical_validity: Optional[Dict[str, Any]] = Field(None)
    practical_validity: Optional[Dict[str, Any]] = Field(None)
    safety_validity: Optional[Dict[str, Any]] = Field(None)
    
    def add_error(self, message: str) -> None:
        """오류 추가"""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """경고 추가"""
        self.warnings.append(message)
    
    def add_suggestion(self, message: str) -> None:
        """제안 추가"""
        self.suggestions.append(message)


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
            data: 실험 결과 데이터프레임
            
        Returns:
            AnalysisResult: 분석 결과
        """
        pass
    
    # ==================== 헬퍼 메서드 ====================
    
    def get_module_info(self) -> Dict[str, Any]:
        """모듈 정보 반환"""
        return self.metadata.copy()
    
    def validate_design(self, design: ExperimentDesign) -> ValidationResult:
        """
        실험 설계 검증
        
        Args:
            design: 검증할 실험 설계
            
        Returns:
            ValidationResult: 검증 결과
        """
        result = ValidationResult()
        
        # 기본 검증
        if design.n_runs < len(design.factors):
            result.add_error(f"실험 횟수({design.n_runs})가 요인 수({len(design.factors)})보다 적습니다")
        
        if not design.factors:
            result.add_error("요인이 정의되지 않았습니다")
        
        if not design.responses:
            result.add_error("반응변수가 정의되지 않았습니다")
        
        # 통계적 검정력 검증
        if design.n_runs < 2 * len(design.factors):
            result.add_warning("통계적 검정력이 낮을 수 있습니다")
        
        # 실용성 검증
        if design.n_runs > 100:
            result.add_warning("실험 횟수가 많습니다. 실행 가능성을 검토하세요")
        
        # D-효율성 검증
        if design.d_efficiency is not None and design.d_efficiency < 0.7:
            result.add_warning(f"D-효율성이 낮습니다 ({design.d_efficiency:.2f})")
        
        return result
    
    def export_design(self, design: ExperimentDesign, format: str = 'excel') -> Any:
        """
        실험 설계 내보내기
        
        Args:
            design: 내보낼 실험 설계
            format: 출력 형식 (excel/csv/json)
            
        Returns:
            내보내기 결과 (파일 경로 또는 데이터)
        """
        if format == 'excel':
            from io import BytesIO
            output = BytesIO()
            design.to_excel(output)
            output.seek(0)
            return output.getvalue()
            
        elif format == 'csv':
            return design.runs.to_csv(index=False)
            
        elif format == 'json':
            return design.dict()
            
        else:
            raise ValueError(f"지원하지 않는 형식: {format}")
    
    def estimate_resources(self, design: ExperimentDesign) -> Dict[str, Any]:
        """
        실험에 필요한 자원 추정
        
        Args:
            design: 실험 설계
            
        Returns:
            자원 추정치
        """
        total_time = design.n_runs * sum(
            r.measurement_time or 60 for r in design.responses
        )
        
        total_cost = design.n_runs * sum(
            r.measurement_cost or 0 for r in design.responses
        )
        
        return {
            'total_runs': design.n_runs,
            'estimated_time_hours': total_time / 60,
            'estimated_cost': total_cost,
            'required_equipment': list(set(
                equip for r in design.responses 
                for equip in (r.equipment_required or [])
            ))
        }
    
    def get_supported_designs(self) -> List[str]:
        """지원하는 실험 설계 유형 목록"""
        return [design.value for design in DesignType]
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        모듈 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        if config:
            self.config.update(config)
        self._initialized = True
        logger.info(f"Module {self.metadata['name']} initialized")
    
    def cleanup(self) -> None:
        """모듈 정리 작업"""
        self._initialized = False
        logger.info(f"Module {self.metadata['name']} cleaned up")
    
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


def validate_factor_levels(factor: Factor) -> Tuple[bool, Optional[str]]:
    """
    요인 레벨 유효성 검증
    
    Args:
        factor: 검증할 요인
        
    Returns:
        Tuple[bool, Optional[str]]: (유효성, 에러 메시지)
    """
    if factor.type == FactorType.CONTINUOUS:
        if factor.min_value is None or factor.max_value is None:
            return False, "연속형 요인은 최소값과 최대값이 필요합니다"
        if factor.min_value >= factor.max_value:
            return False, "최소값은 최대값보다 작아야 합니다"
    else:
        if not factor.levels or len(factor.levels) < 2:
            return False, "범주형/이산형 요인은 최소 2개 이상의 레벨이 필요합니다"
    
    return True, None


def calculate_design_efficiency(design_matrix: np.ndarray) -> Dict[str, float]:
    """
    설계 효율성 계산
    
    Args:
        design_matrix: 설계 매트릭스
        
    Returns:
        효율성 지표 딕셔너리
    """
    try:
        # 정보 행렬
        X = design_matrix
        info_matrix = X.T @ X
        
        # D-효율성
        det = np.linalg.det(info_matrix)
        n, p = X.shape
        d_efficiency = (det ** (1/p)) / n if det > 0 else 0
        
        # 조건수
        condition_number = np.linalg.cond(info_matrix)
        
        # G-효율성 (최대 예측 분산)
        try:
            inv_info = np.linalg.inv(info_matrix)
            leverage = np.diag(X @ inv_info @ X.T)
            g_efficiency = p / np.max(leverage) if np.max(leverage) > 0 else 0
        except:
            g_efficiency = None
        
        return {
            'd_efficiency': d_efficiency,
            'g_efficiency': g_efficiency,
            'condition_number': condition_number
        }
    except Exception as e:
        logger.error(f"효율성 계산 실패: {str(e)}")
        return {
            'd_efficiency': None,
            'g_efficiency': None,
            'condition_number': None
        }
