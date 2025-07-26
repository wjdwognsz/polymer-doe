"""
🧬 Universal DOE Platform - 실험 모듈 기본 클래스
모든 실험 설계 모듈이 상속받는 추상 기본 클래스
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# ==================== 열거형 정의 ====================

class FactorType(str, Enum):
    """실험 요인 타입"""
    CONTINUOUS = "continuous"  # 연속형 (온도, 압력 등)
    CATEGORICAL = "categorical"  # 범주형 (촉매 종류, 용매 등)
    DISCRETE = "discrete"  # 이산형 (정수값)
    MIXTURE = "mixture"  # 혼합물 성분
    ORDINAL = "ordinal"  # 순서형 (낮음, 중간, 높음)


class ResponseGoal(str, Enum):
    """반응변수 목표"""
    MAXIMIZE = "maximize"  # 최대화 (수율, 효율 등)
    MINIMIZE = "minimize"  # 최소화 (비용, 불순물 등)
    TARGET = "target"  # 목표값 (특정 값에 근접)
    IN_RANGE = "in_range"  # 범위 내 (규격 만족)


class DesignType(str, Enum):
    """실험 설계 타입"""
    FULL_FACTORIAL = "full_factorial"  # 완전요인설계
    FRACTIONAL_FACTORIAL = "fractional_factorial"  # 부분요인설계
    CCD = "central_composite"  # 중심합성설계
    BOX_BEHNKEN = "box_behnken"  # Box-Behnken 설계
    PLACKETT_BURMAN = "plackett_burman"  # Plackett-Burman 설계
    OPTIMAL = "optimal"  # 최적 설계 (D, A, I-optimal)
    MIXTURE = "mixture"  # 혼합물 설계
    CUSTOM = "custom"  # 사용자 정의


# ==================== 데이터 모델 ====================

class Factor(BaseModel):
    """실험 요인 모델"""
    name: str = Field(..., description="요인 이름 (영문)")
    display_name: str = Field("", description="표시 이름 (한글 가능)")
    type: FactorType = Field(FactorType.CONTINUOUS, description="요인 타입")
    unit: str = Field("", description="단위")
    description: str = Field("", description="요인 설명")
    
    # 연속형/이산형 요인용
    min_value: Optional[float] = Field(None, description="최소값")
    max_value: Optional[float] = Field(None, description="최대값")
    
    # 범주형/순서형 요인용
    levels: List[Union[str, float]] = Field(default_factory=list, description="수준 목록")
    
    # 고급 설정
    constraints: Dict[str, Any] = Field(default_factory=dict, description="제약조건")
    coding_type: str = Field("coded", description="코딩 타입 (coded/actual)")
    center_point: Optional[float] = Field(None, description="중심점 값")
    
    @validator('display_name', always=True)
    def set_display_name(cls, v, values):
        """표시 이름이 없으면 이름 사용"""
        return v or values.get('name', '')
    
    @validator('levels')
    def validate_levels(cls, v, values):
        """범주형 요인의 수준 검증"""
        factor_type = values.get('type')
        if factor_type in [FactorType.CATEGORICAL, FactorType.ORDINAL]:
            if len(v) < 2:
                raise ValueError("범주형/순서형 요인은 최소 2개 수준이 필요합니다")
        return v
    
    def get_coded_values(self, n_levels: int = 2) -> List[float]:
        """코딩된 값 반환"""
        if self.type == FactorType.CONTINUOUS:
            if n_levels == 2:
                return [-1, 1]
            elif n_levels == 3:
                return [-1, 0, 1]
            else:
                return list(np.linspace(-1, 1, n_levels))
        elif self.type == FactorType.CATEGORICAL:
            return list(range(len(self.levels)))
        return []
    
    def get_actual_values(self, coded_values: List[float]) -> List[float]:
        """실제 값으로 변환"""
        if self.type == FactorType.CONTINUOUS:
            center = (self.max_value + self.min_value) / 2
            scale = (self.max_value - self.min_value) / 2
            return [center + scale * coded for coded in coded_values]
        return coded_values


class Response(BaseModel):
    """반응변수 모델"""
    name: str = Field(..., description="반응변수 이름")
    display_name: str = Field("", description="표시 이름")
    unit: str = Field("", description="단위")
    goal: ResponseGoal = Field(ResponseGoal.MAXIMIZE, description="최적화 목표")
    description: str = Field("", description="반응변수 설명")
    
    # 목표값/범위 설정
    target_value: Optional[float] = Field(None, description="목표값 (goal=target)")
    lower_limit: Optional[float] = Field(None, description="하한값")
    upper_limit: Optional[float] = Field(None, description="상한값")
    
    # 가중치 및 중요도
    weight: float = Field(1.0, ge=0, description="상대적 중요도")
    critical: bool = Field(False, description="필수 달성 여부")
    
    # 측정 관련
    measurement_method: str = Field("", description="측정 방법")
    measurement_error: Optional[float] = Field(None, description="측정 오차")
    replicates_required: int = Field(1, ge=1, description="반복 측정 횟수")
    
    @validator('display_name', always=True)
    def set_display_name(cls, v, values):
        """표시 이름이 없으면 이름 사용"""
        return v or values.get('name', '')
    
    @validator('target_value')
    def validate_target(cls, v, values):
        """목표값 검증"""
        goal = values.get('goal')
        if goal == ResponseGoal.TARGET and v is None:
            raise ValueError("목표값 최적화는 target_value가 필요합니다")
        return v
    
    def is_in_spec(self, value: float) -> bool:
        """규격 만족 여부 확인"""
        if self.lower_limit is not None and value < self.lower_limit:
            return False
        if self.upper_limit is not None and value > self.upper_limit:
            return False
        return True
    
    def calculate_desirability(self, value: float) -> float:
        """만족도(desirability) 계산"""
        if self.goal == ResponseGoal.MAXIMIZE:
            if self.upper_limit is None:
                return 1.0
            return max(0, min(1, (value - self.lower_limit) / (self.upper_limit - self.lower_limit)))
        elif self.goal == ResponseGoal.MINIMIZE:
            if self.lower_limit is None:
                return 1.0
            return max(0, min(1, (self.upper_limit - value) / (self.upper_limit - self.lower_limit)))
        elif self.goal == ResponseGoal.TARGET:
            if self.target_value is None:
                return 0.0
            deviation = abs(value - self.target_value)
            tolerance = abs(self.upper_limit - self.lower_limit) / 2 if self.upper_limit and self.lower_limit else 1.0
            return max(0, 1 - deviation / tolerance)
        return 1.0 if self.is_in_spec(value) else 0.0


class ExperimentDesign(BaseModel):
    """실험 설계 모델"""
    design_id: str = Field("", description="설계 ID")
    design_type: DesignType = Field(DesignType.CUSTOM, description="설계 타입")
    name: str = Field("", description="설계 이름")
    description: str = Field("", description="설계 설명")
    
    # 핵심 데이터
    runs: pd.DataFrame = Field(..., description="실험 런 데이터프레임")
    factors: List[Factor] = Field(..., description="실험 요인 목록")
    responses: List[Response] = Field(..., description="반응변수 목록")
    
    # 설계 속성
    n_runs: int = Field(0, description="총 실험 횟수")
    n_factors: int = Field(0, description="요인 수")
    n_responses: int = Field(0, description="반응변수 수")
    
    # 설계 품질
    resolution: Optional[int] = Field(None, description="해상도 (부분요인설계)")
    d_efficiency: Optional[float] = Field(None, description="D-효율성")
    a_efficiency: Optional[float] = Field(None, description="A-효율성")
    g_efficiency: Optional[float] = Field(None, description="G-효율성")
    
    # 블록화 및 랜덤화
    blocks: Optional[List[int]] = Field(None, description="블록 할당")
    run_order: Optional[List[int]] = Field(None, description="실행 순서")
    
    # 메타데이터
    created_at: datetime = Field(default_factory=datetime.now)
    created_by: str = Field("", description="생성자")
    modified_at: Optional[datetime] = Field(None)
    tags: List[str] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True
    
    @validator('n_runs', 'n_factors', 'n_responses', always=True)
    def set_counts(cls, v, values):
        """자동으로 개수 설정"""
        if 'runs' in values and values['runs'] is not None:
            if 'n_runs' in values:
                return len(values['runs'])
        if 'factors' in values and values['factors'] is not None:
            if 'n_factors' in values:
                return len(values['factors'])
        if 'responses' in values and values['responses'] is not None:
            if 'n_responses' in values:
                return len(values['responses'])
        return v
    
    def to_excel(self, filepath: Path, include_metadata: bool = True) -> None:
        """Excel 파일로 내보내기"""
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # 실험 런
            self.runs.to_excel(writer, sheet_name='Design', index=False)
            
            # 요인 정보
            factors_df = pd.DataFrame([f.dict() for f in self.factors])
            factors_df.to_excel(writer, sheet_name='Factors', index=False)
            
            # 반응변수 정보
            responses_df = pd.DataFrame([r.dict() for r in self.responses])
            responses_df.to_excel(writer, sheet_name='Responses', index=False)
            
            # 메타데이터
            if include_metadata:
                metadata = {
                    'design_type': self.design_type,
                    'n_runs': self.n_runs,
                    'n_factors': self.n_factors,
                    'n_responses': self.n_responses,
                    'created_at': self.created_at.isoformat(),
                    'created_by': self.created_by
                }
                metadata_df = pd.DataFrame([metadata])
                metadata_df.to_excel(writer, sheet_name='Metadata', index=False)


class AnalysisResult(BaseModel):
    """분석 결과 모델"""
    analysis_id: str = Field("", description="분석 ID")
    design_id: str = Field("", description="원본 설계 ID")
    analysis_type: str = Field("", description="분석 타입")
    
    # 데이터
    raw_data: pd.DataFrame = Field(..., description="원시 데이터")
    processed_data: Optional[pd.DataFrame] = Field(None, description="처리된 데이터")
    
    # 통계 분석
    summary_stats: Dict[str, Any] = Field(default_factory=dict, description="기술통계")
    model_results: Dict[str, Any] = Field(default_factory=dict, description="모델 결과")
    anova_results: Optional[pd.DataFrame] = Field(None, description="ANOVA 결과")
    coefficients: Optional[pd.DataFrame] = Field(None, description="회귀 계수")
    
    # 최적화
    optimal_conditions: Optional[Dict[str, float]] = Field(None, description="최적 조건")
    predicted_responses: Optional[Dict[str, float]] = Field(None, description="예측 반응값")
    desirability: Optional[float] = Field(None, description="종합 만족도")
    
    # 시각화
    plots: Dict[str, Any] = Field(default_factory=dict, description="생성된 플롯")
    
    # 권장사항
    recommendations: List[str] = Field(default_factory=list, description="권장사항")
    next_experiments: Optional[pd.DataFrame] = Field(None, description="추천 다음 실험")
    
    # 메타데이터
    analysis_time: float = Field(0, description="분석 소요 시간 (초)")
    confidence_level: float = Field(0.95, description="신뢰 수준")
    performed_at: datetime = Field(default_factory=datetime.now)
    performed_by: str = Field("", description="분석 수행자")
    
    class Config:
        arbitrary_types_allowed = True
    
    def add_recommendation(self, recommendation: str) -> None:
        """권장사항 추가"""
        self.recommendations.append(recommendation)
    
    def get_significant_factors(self, alpha: float = 0.05) -> List[str]:
        """유의한 요인 목록 반환"""
        if self.anova_results is None:
            return []
        
        significant = self.anova_results[self.anova_results['p_value'] < alpha]
        return significant['factor'].tolist()


class ValidationResult(BaseModel):
    """검증 결과 모델"""
    is_valid: bool = Field(True, description="전체 유효성")
    errors: List[str] = Field(default_factory=list, description="오류 목록")
    warnings: List[str] = Field(default_factory=list, description="경고 목록")
    suggestions: List[str] = Field(default_factory=list, description="개선 제안")
    
    # 상세 검증 결과
    statistical_validity: Optional[Dict[str, Any]] = Field(None, description="통계적 유효성")
    practical_validity: Optional[Dict[str, Any]] = Field(None, description="실용적 유효성")
    safety_validity: Optional[Dict[str, Any]] = Field(None, description="안전성 검증")
    
    # 점수
    overall_score: float = Field(100.0, description="종합 점수")
    
    def add_error(self, message: str) -> None:
        """오류 추가"""
        self.errors.append(message)
        self.is_valid = False
        self.overall_score = max(0, self.overall_score - 20)
    
    def add_warning(self, message: str) -> None:
        """경고 추가"""
        self.warnings.append(message)
        self.overall_score = max(0, self.overall_score - 5)
    
    def add_suggestion(self, message: str) -> None:
        """제안 추가"""
        self.suggestions.append(message)
    
    def get_summary(self) -> str:
        """검증 결과 요약"""
        if self.is_valid:
            return f"✅ 검증 통과 (점수: {self.overall_score:.1f}/100)"
        else:
            return f"❌ 검증 실패 (오류: {len(self.errors)}개, 경고: {len(self.warnings)}개)"


# ==================== 기본 모듈 클래스 ====================

class BaseExperimentModule(ABC):
    """모든 실험 모듈의 추상 기본 클래스"""
    
    def __init__(self):
        """모듈 초기화"""
        self.metadata = {
            'module_id': self._generate_module_id(),
            'name': 'Base Experiment Module',
            'version': '1.0.0',
            'author': '',
            'created_date': datetime.now().isoformat(),
            'last_modified': datetime.now().isoformat(),
            'description': '',
            'category': 'general',
            'tags': [],
            'dependencies': [],
            'min_platform_version': '1.0.0',
            'max_platform_version': None,
            'icon': '🧪',
            'color': '#1f77b4',
            'homepage': '',
            'documentation': '',
            'license': 'MIT'
        }
        
        self.config = {
            'allow_custom_factors': True,
            'allow_custom_responses': True,
            'min_runs': 2,
            'max_runs': 10000,
            'supported_designs': [],
            'validation_level': 'standard'  # minimal, standard, strict
        }
        
        self._initialized = False
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 초기화 메서드 호출
        self._initialize()
    
    def _generate_module_id(self) -> str:
        """모듈 ID 생성"""
        import uuid
        return f"module_{uuid.uuid4().hex[:8]}"
    
    def _initialize(self) -> None:
        """서브클래스 초기화 - 오버라이드 가능"""
        pass
    
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
    
    # ==================== 선택적 메서드 (오버라이드 가능) ====================
    
    def get_templates(self) -> List[Dict[str, Any]]:
        """
        사전 정의된 템플릿 목록 반환
        
        Returns:
            List[Dict]: 템플릿 목록
        """
        return []
    
    def get_examples(self) -> List[Dict[str, Any]]:
        """
        예제 데이터셋 반환
        
        Returns:
            List[Dict]: 예제 목록
        """
        return []
    
    def export_design(self, design: ExperimentDesign, format: str = 'excel') -> bytes:
        """
        설계를 특정 형식으로 내보내기
        
        Args:
            design: 실험 설계
            format: 출력 형식 (excel, csv, json)
            
        Returns:
            bytes: 내보낸 데이터
        """
        if format == 'excel':
            import io
            buffer = io.BytesIO()
            design.to_excel(buffer)
            return buffer.getvalue()
        elif format == 'csv':
            return design.runs.to_csv(index=False).encode()
        elif format == 'json':
            return design.json().encode()
        else:
            raise ValueError(f"지원하지 않는 형식: {format}")
    
    def import_data(self, file_data: bytes, format: str = 'excel') -> pd.DataFrame:
        """
        외부 데이터 가져오기
        
        Args:
            file_data: 파일 데이터
            format: 파일 형식
            
        Returns:
            pd.DataFrame: 가져온 데이터
        """
        import io
        
        if format == 'excel':
            return pd.read_excel(io.BytesIO(file_data))
        elif format == 'csv':
            return pd.read_csv(io.BytesIO(file_data))
        else:
            raise ValueError(f"지원하지 않는 형식: {format}")
    
    # ==================== 헬퍼 메서드 ====================
    
    def get_module_info(self) -> Dict[str, Any]:
        """모듈 정보 반환"""
        return self.metadata.copy()
    
    def get_config(self) -> Dict[str, Any]:
        """설정 정보 반환"""
        return self.config.copy()
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """설정 업데이트"""
        self.config.update(config)
        self._logger.info(f"설정 업데이트: {config}")
    
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
        if design.n_runs < self.config.get('min_runs', 2):
            result.add_error(f"실험 횟수가 최소값({self.config['min_runs']})보다 적습니다")
        
        if design.n_runs > self.config.get('max_runs', 10000):
            result.add_error(f"실험 횟수가 최대값({self.config['max_runs']})을 초과합니다")
        
        if not design.factors:
            result.add_error("요인이 정의되지 않았습니다")
        
        if not design.responses:
            result.add_error("반응변수가 정의되지 않았습니다")
        
        # 통계적 검정력 검증
        if design.n_runs < design.n_factors + 1:
            result.add_error("실험 횟수가 요인 수보다 적어 분석이 불가능합니다")
        elif design.n_runs < 2 * design.n_factors:
            result.add_warning("통계적 검정력이 낮을 수 있습니다")
        
        # 실용성 검증
        if design.n_runs > 100:
            result.add_warning("실험 횟수가 많습니다. 실행 가능성을 검토하세요")
        
        # 중복 실험점 확인
        if design.runs.duplicated().any():
            n_duplicates = design.runs.duplicated().sum()
            result.add_warning(f"{n_duplicates}개의 중복 실험점이 있습니다")
        
        # 설계 공간 커버리지
        if self.config.get('validation_level') == 'strict':
            coverage = self._calculate_design_coverage(design)
            if coverage < 0.8:
                result.add_suggestion(f"설계 공간 커버리지가 {coverage:.1%}입니다. 추가 실험점을 고려하세요")
        
        return result
    
    def _calculate_design_coverage(self, design: ExperimentDesign) -> float:
        """설계 공간 커버리지 계산"""
        # 간단한 구현 - 실제로는 더 정교한 계산 필요
        continuous_factors = [f for f in design.factors if f.type == FactorType.CONTINUOUS]
        if not continuous_factors:
            return 1.0
        
        # 각 요인의 범위를 얼마나 커버하는지 확인
        coverage_scores = []
        for factor in continuous_factors:
            if factor.name in design.runs.columns:
                values = design.runs[factor.name]
                range_covered = (values.max() - values.min()) / (factor.max_value - factor.min_value)
                coverage_scores.append(min(1.0, range_covered))
        
        return np.mean(coverage_scores) if coverage_scores else 0.0
    
    def estimate_completion_time(self, design: ExperimentDesign, 
                                time_per_run: float = 1.0) -> Dict[str, float]:
        """
        실험 완료 시간 추정
        
        Args:
            design: 실험 설계
            time_per_run: 실험 1회당 소요 시간 (시간)
            
        Returns:
            Dict: 시간 추정 정보
        """
        total_time = design.n_runs * time_per_run
        
        return {
            'total_hours': total_time,
            'total_days': total_time / 24,
            'with_8hr_days': total_time / 8,
            'parallel_2': total_time / 2,
            'parallel_4': total_time / 4
        }
    
    def calculate_resource_requirements(self, design: ExperimentDesign,
                                      materials: Dict[str, float]) -> Dict[str, float]:
        """
        필요 자원 계산
        
        Args:
            design: 실험 설계
            materials: 재료별 단위 사용량
            
        Returns:
            Dict: 총 필요량
        """
        requirements = {}
        
        for material, unit_amount in materials.items():
            total_amount = unit_amount * design.n_runs
            requirements[material] = total_amount
            
            # 여유분 10% 추가
            requirements[f"{material}_with_buffer"] = total_amount * 1.1
        
        return requirements
    
    def get_version(self) -> str:
        """모듈 버전 반환"""
        return self.metadata.get('version', '1.0.0')
    
    def is_compatible(self, platform_version: str) -> bool:
        """플랫폼 호환성 확인"""
        from packaging import version
        
        min_version = self.metadata.get('min_platform_version', '1.0.0')
        max_version = self.metadata.get('max_platform_version')
        
        current = version.parse(platform_version)
        
        if current < version.parse(min_version):
            return False
        
        if max_version and current > version.parse(max_version):
            return False
        
        return True
    
    def __str__(self) -> str:
        """문자열 표현"""
        return f"{self.metadata['name']} v{self.metadata['version']}"
    
    def __repr__(self) -> str:
        """개발자용 표현"""
        return (f"<{self.__class__.__name__}("
                f"name='{self.metadata['name']}', "
                f"version='{self.metadata['version']}', "
                f"id='{self.metadata['module_id']}')>")


# ==================== 유틸리티 함수 ====================

def validate_factor_data(factor_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """요인 데이터 검증 헬퍼"""
    required_fields = ['name', 'type']
    
    for field in required_fields:
        if field not in factor_data:
            return False, f"필수 필드 누락: {field}"
    
    if factor_data['type'] == 'continuous':
        if 'min_value' not in factor_data or 'max_value' not in factor_data:
            return False, "연속형 요인은 min_value와 max_value가 필요합니다"
        
        if factor_data['min_value'] >= factor_data['max_value']:
            return False, "min_value는 max_value보다 작아야 합니다"
    
    elif factor_data['type'] == 'categorical':
        if 'levels' not in factor_data or len(factor_data['levels']) < 2:
            return False, "범주형 요인은 최소 2개의 수준이 필요합니다"
    
    return True, None


def validate_response_data(response_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """반응변수 데이터 검증 헬퍼"""
    required_fields = ['name', 'goal']
    
    for field in required_fields:
        if field not in response_data:
            return False, f"필수 필드 누락: {field}"
    
    valid_goals = ['maximize', 'minimize', 'target', 'in_range']
    if response_data['goal'] not in valid_goals:
        return False, f"유효하지 않은 목표: {response_data['goal']}"
    
    if response_data['goal'] == 'target' and 'target_value' not in response_data:
        return False, "목표값 최적화는 target_value가 필요합니다"
    
    return True, None


# ==================== 타입 별칭 ====================

DesignMatrix = pd.DataFrame
ResultsMatrix = pd.DataFrame
FactorSettings = Dict[str, Union[float, str]]
OptimalPoint = Dict[str, float]
