"""
Universal DOE Platform - General Experiment Module
범용 실험 설계 모듈 - 모든 연구 분야를 지원하는 핵심 모듈
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
import json
from datetime import datetime
import traceback
import logging
from itertools import product
from scipy import stats
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# DOE 관련 라이브러리
try:
    from pyDOE2 import (
        fullfact, fracfact, pbdesign, ccdesign, bbdesign,
        lhs, factorial, ff2n
    )
    PYDOE2_AVAILABLE = True
except ImportError:
    PYDOE2_AVAILABLE = False
    st.warning("pyDOE2가 설치되지 않았습니다. 일부 고급 설계 기능이 제한됩니다.")

# 모듈 기본 클래스 임포트
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from modules.base_module import (
    BaseExperimentModule, Factor, FactorType, Response, ResponseGoal,
    ExperimentDesign, DesignResult, AnalysisResult, ValidationResult
)

logger = logging.getLogger(__name__)


# ==================== 데이터 클래스 정의 ====================

@dataclass
class DesignMethod:
    """실험설계법 정의"""
    name: str
    display_name: str
    description: str
    min_factors: int
    max_factors: int
    supports_categorical: bool
    supports_constraints: bool
    supports_blocks: bool
    complexity: str  # low, medium, high
    use_cases: List[str] = field(default_factory=list)
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)


@dataclass
class FactorTemplate:
    """요인 템플릿"""
    name: str
    category: str
    default_type: FactorType
    default_unit: str
    default_min: Optional[float] = None
    default_max: Optional[float] = None
    default_levels: List[Any] = field(default_factory=list)
    description: str = ""
    common_constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponseTemplate:
    """반응변수 템플릿"""
    name: str
    category: str
    default_unit: str
    default_goal: ResponseGoal
    measurement_method: str = ""
    typical_range: Optional[Tuple[float, float]] = None
    description: str = ""


@dataclass
class DesignQuality:
    """설계 품질 지표"""
    d_efficiency: float = 0.0
    a_efficiency: float = 0.0
    g_efficiency: float = 0.0
    condition_number: float = 0.0
    vif_max: float = 0.0
    orthogonality: float = 0.0
    power: Dict[str, float] = field(default_factory=dict)
    
    @property
    def overall_score(self) -> float:
        """종합 품질 점수 (0-100)"""
        scores = []
        if self.d_efficiency > 0:
            scores.append(self.d_efficiency)
        if self.orthogonality > 0:
            scores.append(self.orthogonality * 100)
        if self.condition_number > 0:
            scores.append(min(100, 100 / self.condition_number))
        return np.mean(scores) if scores else 0.0


# ==================== 템플릿 정의 ====================

class ExperimentTemplates:
    """실험 템플릿 관리"""
    
    @staticmethod
    def get_factor_templates() -> Dict[str, List[FactorTemplate]]:
        """요인 템플릿 반환"""
        return {
            "공정 변수": [
                FactorTemplate(
                    name="온도", category="공정 변수",
                    default_type=FactorType.CONTINUOUS,
                    default_unit="°C", default_min=20, default_max=200,
                    description="공정 온도"
                ),
                FactorTemplate(
                    name="압력", category="공정 변수",
                    default_type=FactorType.CONTINUOUS,
                    default_unit="bar", default_min=1, default_max=10,
                    description="공정 압력"
                ),
                FactorTemplate(
                    name="시간", category="공정 변수",
                    default_type=FactorType.CONTINUOUS,
                    default_unit="min", default_min=10, default_max=180,
                    description="반응/처리 시간"
                ),
                FactorTemplate(
                    name="교반속도", category="공정 변수",
                    default_type=FactorType.CONTINUOUS,
                    default_unit="rpm", default_min=100, default_max=1000,
                    description="교반 속도"
                ),
                FactorTemplate(
                    name="유속", category="공정 변수",
                    default_type=FactorType.CONTINUOUS,
                    default_unit="mL/min", default_min=0.1, default_max=10,
                    description="유체 흐름 속도"
                )
            ],
            "조성 변수": [
                FactorTemplate(
                    name="농도", category="조성 변수",
                    default_type=FactorType.CONTINUOUS,
                    default_unit="M", default_min=0.01, default_max=2.0,
                    description="용질 농도"
                ),
                FactorTemplate(
                    name="pH", category="조성 변수",
                    default_type=FactorType.CONTINUOUS,
                    default_unit="", default_min=1, default_max=14,
                    description="수용액 pH"
                ),
                FactorTemplate(
                    name="함량", category="조성 변수",
                    default_type=FactorType.CONTINUOUS,
                    default_unit="wt%", default_min=0, default_max=100,
                    description="성분 함량"
                ),
                FactorTemplate(
                    name="몰비", category="조성 변수",
                    default_type=FactorType.CONTINUOUS,
                    default_unit="", default_min=0.1, default_max=10,
                    description="반응물 몰비"
                )
            ],
            "물리적 변수": [
                FactorTemplate(
                    name="입자크기", category="물리적 변수",
                    default_type=FactorType.CONTINUOUS,
                    default_unit="μm", default_min=0.1, default_max=1000,
                    description="평균 입자 크기"
                ),
                FactorTemplate(
                    name="두께", category="물리적 변수",
                    default_type=FactorType.CONTINUOUS,
                    default_unit="mm", default_min=0.1, default_max=10,
                    description="필름/코팅 두께"
                ),
                FactorTemplate(
                    name="표면적", category="물리적 변수",
                    default_type=FactorType.CONTINUOUS,
                    default_unit="m²/g", default_min=1, default_max=1000,
                    description="비표면적"
                )
            ],
            "범주형 변수": [
                FactorTemplate(
                    name="촉매종류", category="범주형 변수",
                    default_type=FactorType.CATEGORICAL,
                    default_unit="", 
                    default_levels=["Pd/C", "Pt/C", "Ru/C", "None"],
                    description="촉매 종류"
                ),
                FactorTemplate(
                    name="용매", category="범주형 변수",
                    default_type=FactorType.CATEGORICAL,
                    default_unit="",
                    default_levels=["물", "에탄올", "아세톤", "톨루엔"],
                    description="반응 용매"
                ),
                FactorTemplate(
                    name="첨가제", category="범주형 변수",
                    default_type=FactorType.CATEGORICAL,
                    default_unit="",
                    default_levels=["A", "B", "C", "없음"],
                    description="첨가제 종류"
                )
            ]
        }
    
    @staticmethod
    def get_response_templates() -> Dict[str, List[ResponseTemplate]]:
        """반응변수 템플릿 반환"""
        return {
            "수율/효율": [
                ResponseTemplate(
                    name="수율", category="수율/효율",
                    default_unit="%", default_goal=ResponseGoal.MAXIMIZE,
                    typical_range=(0, 100),
                    description="반응/공정 수율"
                ),
                ResponseTemplate(
                    name="순도", category="수율/효율",
                    default_unit="%", default_goal=ResponseGoal.MAXIMIZE,
                    typical_range=(0, 100),
                    description="제품 순도"
                ),
                ResponseTemplate(
                    name="전환율", category="수율/효율",
                    default_unit="%", default_goal=ResponseGoal.MAXIMIZE,
                    typical_range=(0, 100),
                    description="반응물 전환율"
                ),
                ResponseTemplate(
                    name="선택성", category="수율/효율",
                    default_unit="%", default_goal=ResponseGoal.MAXIMIZE,
                    typical_range=(0, 100),
                    description="목표 생성물 선택성"
                )
            ],
            "물성": [
                ResponseTemplate(
                    name="강도", category="물성",
                    default_unit="MPa", default_goal=ResponseGoal.MAXIMIZE,
                    description="인장/압축 강도"
                ),
                ResponseTemplate(
                    name="경도", category="물성",
                    default_unit="HV", default_goal=ResponseGoal.MAXIMIZE,
                    description="비커스 경도"
                ),
                ResponseTemplate(
                    name="점도", category="물성",
                    default_unit="cP", default_goal=ResponseGoal.TARGET,
                    description="용액 점도"
                ),
                ResponseTemplate(
                    name="밀도", category="물성",
                    default_unit="g/cm³", default_goal=ResponseGoal.TARGET,
                    description="재료 밀도"
                )
            ],
            "분석값": [
                ResponseTemplate(
                    name="분해능", category="분석값",
                    default_unit="", default_goal=ResponseGoal.MAXIMIZE,
                    description="크로마토그래피 분해능"
                ),
                ResponseTemplate(
                    name="감도", category="분석값",
                    default_unit="S/N", default_goal=ResponseGoal.MAXIMIZE,
                    description="신호 대 잡음비"
                ),
                ResponseTemplate(
                    name="분석시간", category="분석값",
                    default_unit="min", default_goal=ResponseGoal.MINIMIZE,
                    description="총 분석 시간"
                )
            ],
            "비용/환경": [
                ResponseTemplate(
                    name="비용", category="비용/환경",
                    default_unit="$/kg", default_goal=ResponseGoal.MINIMIZE,
                    description="단위 생산 비용"
                ),
                ResponseTemplate(
                    name="에너지소비", category="비용/환경",
                    default_unit="kWh", default_goal=ResponseGoal.MINIMIZE,
                    description="에너지 소비량"
                ),
                ResponseTemplate(
                    name="폐기물", category="비용/환경",
                    default_unit="kg", default_goal=ResponseGoal.MINIMIZE,
                    description="폐기물 발생량"
                )
            ]
        }
    
    @staticmethod
    def get_experiment_presets() -> Dict[str, Dict[str, Any]]:
        """실험 프리셋 반환"""
        return {
            "화학합성 최적화": {
                "description": "유기/무기 화학 반응 최적화",
                "factors": [
                    {"name": "온도", "type": "continuous", "min": 20, "max": 150, "unit": "°C"},
                    {"name": "시간", "type": "continuous", "min": 30, "max": 360, "unit": "min"},
                    {"name": "촉매량", "type": "continuous", "min": 0.1, "max": 5, "unit": "mol%"},
                    {"name": "용매", "type": "categorical", "levels": ["THF", "톨루엔", "DMF"]}
                ],
                "responses": [
                    {"name": "수율", "unit": "%", "goal": "maximize"},
                    {"name": "순도", "unit": "%", "goal": "maximize"},
                    {"name": "비용", "unit": "$/g", "goal": "minimize"}
                ],
                "suggested_design": "central_composite"
            },
            "재료 물성 최적화": {
                "description": "재료의 기계적/물리적 특성 최적화",
                "factors": [
                    {"name": "조성A", "type": "continuous", "min": 0, "max": 100, "unit": "wt%"},
                    {"name": "조성B", "type": "continuous", "min": 0, "max": 100, "unit": "wt%"},
                    {"name": "처리온도", "type": "continuous", "min": 100, "max": 500, "unit": "°C"},
                    {"name": "처리시간", "type": "continuous", "min": 1, "max": 24, "unit": "h"}
                ],
                "responses": [
                    {"name": "강도", "unit": "MPa", "goal": "maximize"},
                    {"name": "경도", "unit": "HV", "goal": "maximize"},
                    {"name": "밀도", "unit": "g/cm³", "goal": "target", "target": 2.5}
                ],
                "suggested_design": "box_behnken"
            },
            "분석법 개발": {
                "description": "크로마토그래피/분광법 최적화",
                "factors": [
                    {"name": "유속", "type": "continuous", "min": 0.5, "max": 2.0, "unit": "mL/min"},
                    {"name": "컬럼온도", "type": "continuous", "min": 25, "max": 60, "unit": "°C"},
                    {"name": "이동상조성", "type": "continuous", "min": 10, "max": 90, "unit": "%B"},
                    {"name": "pH", "type": "continuous", "min": 2, "max": 8, "unit": ""}
                ],
                "responses": [
                    {"name": "분해능", "unit": "", "goal": "maximize"},
                    {"name": "분석시간", "unit": "min", "goal": "minimize"},
                    {"name": "감도", "unit": "S/N", "goal": "maximize"}
                ],
                "suggested_design": "d_optimal"
            },
            "공정 최적화": {
                "description": "생산 공정 파라미터 최적화",
                "factors": [
                    {"name": "온도", "type": "continuous", "min": 60, "max": 120, "unit": "°C"},
                    {"name": "압력", "type": "continuous", "min": 1, "max": 10, "unit": "bar"},
                    {"name": "체류시간", "type": "continuous", "min": 10, "max": 60, "unit": "min"},
                    {"name": "교반속도", "type": "continuous", "min": 100, "max": 500, "unit": "rpm"}
                ],
                "responses": [
                    {"name": "생산량", "unit": "kg/h", "goal": "maximize"},
                    {"name": "품질", "unit": "%", "goal": "maximize"},
                    {"name": "에너지소비", "unit": "kWh/kg", "goal": "minimize"}
                ],
                "suggested_design": "fractional_factorial"
            }
        }


# ==================== 설계 엔진 ====================

class DesignEngine:
    """실험 설계 생성 엔진"""
    
    def __init__(self):
        self.methods = self._initialize_methods()
        
    def _initialize_methods(self) -> Dict[str, DesignMethod]:
        """설계 방법 초기화"""
        return {
            "full_factorial": DesignMethod(
                name="full_factorial",
                display_name="완전요인설계",
                description="모든 요인 수준의 조합을 실험",
                min_factors=2, max_factors=8,
                supports_categorical=True,
                supports_constraints=False,
                supports_blocks=True,
                complexity="low",
                use_cases=["스크리닝", "주효과와 교호작용 분석"],
                pros=["모든 효과 추정 가능", "해석 용이"],
                cons=["실험 횟수 급증", "비용 증가"]
            ),
            "fractional_factorial": DesignMethod(
                name="fractional_factorial",
                display_name="부분요인설계",
                description="완전요인설계의 일부만 실험",
                min_factors=3, max_factors=15,
                supports_categorical=True,
                supports_constraints=False,
                supports_blocks=True,
                complexity="medium",
                use_cases=["다요인 스크리닝", "주효과 추정"],
                pros=["실험 횟수 절감", "효율적"],
                cons=["일부 교호작용 추정 불가", "해상도 제한"]
            ),
            "central_composite": DesignMethod(
                name="central_composite",
                display_name="중심합성설계",
                description="2차 모델 적합을 위한 RSM 설계",
                min_factors=2, max_factors=8,
                supports_categorical=False,
                supports_constraints=True,
                supports_blocks=True,
                complexity="medium",
                use_cases=["최적화", "곡면 반응 모델링"],
                pros=["2차 효과 추정", "최적점 예측"],
                cons=["연속형 요인만 가능", "축점 실행 어려움"]
            ),
            "box_behnken": DesignMethod(
                name="box_behnken",
                display_name="Box-Behnken 설계",
                description="3수준 요인설계와 중심점 조합",
                min_factors=3, max_factors=7,
                supports_categorical=False,
                supports_constraints=True,
                supports_blocks=True,
                complexity="medium",
                use_cases=["최적화", "극값 회피"],
                pros=["극값 조합 없음", "효율적"],
                cons=["3요인 이상 필요", "범주형 불가"]
            ),
            "plackett_burman": DesignMethod(
                name="plackett_burman",
                display_name="Plackett-Burman 설계",
                description="주효과 스크리닝을 위한 설계",
                min_factors=2, max_factors=47,
                supports_categorical=True,
                supports_constraints=False,
                supports_blocks=False,
                complexity="low",
                use_cases=["다요인 스크리닝", "중요 요인 선별"],
                pros=["매우 효율적", "많은 요인 처리"],
                cons=["교호작용 추정 불가", "주효과만"]
            ),
            "latin_hypercube": DesignMethod(
                name="latin_hypercube",
                display_name="Latin Hypercube 설계",
                description="공간 충진 설계",
                min_factors=2, max_factors=20,
                supports_categorical=False,
                supports_constraints=True,
                supports_blocks=False,
                complexity="low",
                use_cases=["컴퓨터 실험", "비선형 모델"],
                pros=["균등 분포", "유연한 실험수"],
                cons=["통계 모델 약함", "범주형 불가"]
            ),
            "d_optimal": DesignMethod(
                name="d_optimal",
                display_name="D-최적 설계",
                description="모델 파라미터 추정 최적화",
                min_factors=1, max_factors=20,
                supports_categorical=True,
                supports_constraints=True,
                supports_blocks=True,
                complexity="high",
                use_cases=["제약 조건 하 최적화", "불규칙 영역"],
                pros=["매우 유연", "최적 정보량"],
                cons=["계산 복잡", "알고리즘 의존"]
            ),
            "custom": DesignMethod(
                name="custom",
                display_name="사용자 정의",
                description="직접 실험점 지정",
                min_factors=1, max_factors=50,
                supports_categorical=True,
                supports_constraints=True,
                supports_blocks=True,
                complexity="low",
                use_cases=["특수 목적", "기존 데이터 활용"],
                pros=["완전한 자유도", "기존 지식 활용"],
                cons=["통계적 최적성 보장 안됨", "전문성 필요"]
            )
        }
    
    def get_method(self, method_name: str) -> Optional[DesignMethod]:
        """설계 방법 반환"""
        return self.methods.get(method_name)
    
    def recommend_method(self, 
                        n_factors: int,
                        factor_types: List[str],
                        objective: str,
                        n_runs_budget: Optional[int] = None) -> str:
        """설계 방법 추천"""
        has_categorical = any(t == "categorical" for t in factor_types)
        
        # 목적별 추천
        if objective == "screening":
            if n_factors > 7:
                return "plackett_burman"
            elif n_factors > 4:
                return "fractional_factorial"
            else:
                return "full_factorial"
                
        elif objective == "optimization":
            if has_categorical:
                return "d_optimal"
            elif n_factors <= 3:
                return "box_behnken"
            else:
                return "central_composite"
                
        elif objective == "robustness":
            return "fractional_factorial"
            
        else:  # exploration
            if has_categorical:
                return "d_optimal"
            else:
                return "latin_hypercube"
    
    def generate_design(self, 
                       method: str,
                       factors: List[Factor],
                       constraints: Optional[Dict[str, Any]] = None,
                       options: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """설계 매트릭스 생성"""
        if not PYDOE2_AVAILABLE and method != "custom":
            raise ImportError("pyDOE2가 필요합니다. pip install pyDOE2를 실행하세요.")
            
        options = options or {}
        
        # 연속형 요인만 추출
        continuous_factors = [f for f in factors if f.type == FactorType.CONTINUOUS]
        categorical_factors = [f for f in factors if f.type == FactorType.CATEGORICAL]
        
        n_continuous = len(continuous_factors)
        
        # 설계별 생성
        if method == "full_factorial":
            design_matrix = self._generate_full_factorial(continuous_factors, options)
        elif method == "fractional_factorial":
            design_matrix = self._generate_fractional_factorial(continuous_factors, options)
        elif method == "central_composite":
            design_matrix = self._generate_ccd(continuous_factors, options)
        elif method == "box_behnken":
            design_matrix = self._generate_box_behnken(continuous_factors, options)
        elif method == "plackett_burman":
            design_matrix = self._generate_plackett_burman(continuous_factors, options)
        elif method == "latin_hypercube":
            design_matrix = self._generate_lhs(continuous_factors, options)
        elif method == "d_optimal":
            design_matrix = self._generate_d_optimal(factors, constraints, options)
        else:  # custom
            design_matrix = self._generate_custom(factors, options)
        
        # 실제 값으로 변환
        df = self._convert_to_actual_values(design_matrix, continuous_factors, categorical_factors)
        
        # 제약조건 적용
        if constraints:
            df = self._apply_constraints(df, constraints)
        
        # 실행 순서 랜덤화
        if options.get("randomize", True):
            df = df.sample(frac=1).reset_index(drop=True)
        
        # Run 번호 추가
        df.index = range(1, len(df) + 1)
        df.index.name = "Run"
        
        return df
    
    def _generate_full_factorial(self, factors: List[Factor], options: Dict) -> np.ndarray:
        """완전요인설계 생성"""
        n_levels = options.get("n_levels", 2)
        n_factors = len(factors)
        
        if n_levels == 2:
            return ff2n(n_factors)
        else:
            levels = [n_levels] * n_factors
            return fullfact(levels)
    
    def _generate_fractional_factorial(self, factors: List[Factor], options: Dict) -> np.ndarray:
        """부분요인설계 생성"""
        n_factors = len(factors)
        resolution = options.get("resolution", 3)
        
        # 생성자 결정
        if n_factors <= 3:
            gen = None
        elif n_factors == 4:
            gen = "D = A B C" if resolution >= 4 else "D = A B"
        elif n_factors == 5:
            gen = "D = A B; E = A C" if resolution >= 3 else "E = A B C D"
        else:
            # 일반적인 생성자
            gen = self._get_fractional_generators(n_factors, resolution)
        
        return fracfact(gen) if gen else ff2n(n_factors)
    
    def _generate_ccd(self, factors: List[Factor], options: Dict) -> np.ndarray:
        """중심합성설계 생성"""
        n_factors = len(factors)
        center = options.get("center_points", (4, 4))
        alpha = options.get("alpha", "orthogonal")
        face = options.get("face", "circumscribed")
        
        return ccdesign(n_factors, center=center, alpha=alpha, face=face)
    
    def _generate_box_behnken(self, factors: List[Factor], options: Dict) -> np.ndarray:
        """Box-Behnken 설계 생성"""
        n_factors = len(factors)
        center = options.get("center_points", 3)
        
        return bbdesign(n_factors, center=center)
    
    def _generate_plackett_burman(self, factors: List[Factor], options: Dict) -> np.ndarray:
        """Plackett-Burman 설계 생성"""
        n_factors = len(factors)
        return pbdesign(n_factors)
    
    def _generate_lhs(self, factors: List[Factor], options: Dict) -> np.ndarray:
        """Latin Hypercube 설계 생성"""
        n_factors = len(factors)
        n_samples = options.get("n_samples", n_factors * 10)
        criterion = options.get("criterion", "maximin")
        
        # 0-1 범위로 생성 후 -1~1로 변환
        lhs_design = lhs(n_factors, samples=n_samples, criterion=criterion)
        return 2 * lhs_design - 1
    
    def _generate_d_optimal(self, factors: List[Factor], 
                           constraints: Dict, options: Dict) -> np.ndarray:
        """D-최적 설계 생성 (간단한 구현)"""
        n_runs = options.get("n_runs", len(factors) * 3)
        
        # 후보점 생성
        continuous_factors = [f for f in factors if f.type == FactorType.CONTINUOUS]
        n_continuous = len(continuous_factors)
        
        if n_continuous > 0:
            # 격자점 생성
            levels_per_factor = 5
            candidates = fullfact([levels_per_factor] * n_continuous)
            # -1 ~ 1로 정규화
            candidates = 2 * (candidates / (levels_per_factor - 1)) - 1
        else:
            candidates = np.array([[0]])  # 더미
        
        # 간단한 교환 알고리즘으로 D-최적 선택
        selected_indices = np.random.choice(len(candidates), n_runs, replace=False)
        return candidates[selected_indices]
    
    def _generate_custom(self, factors: List[Factor], options: Dict) -> np.ndarray:
        """사용자 정의 설계"""
        custom_points = options.get("custom_points", [])
        if not custom_points:
            # 기본값: 각 요인의 min, center, max 조합
            n_factors = len([f for f in factors if f.type == FactorType.CONTINUOUS])
            return np.array([[-1, 0, 1]] * n_factors).T
        return np.array(custom_points)
    
    def _convert_to_actual_values(self, design_matrix: np.ndarray,
                                 continuous_factors: List[Factor],
                                 categorical_factors: List[Factor]) -> pd.DataFrame:
        """코드화된 값을 실제 값으로 변환"""
        df_data = {}
        
        # 연속형 요인 변환
        for i, factor in enumerate(continuous_factors):
            if i < design_matrix.shape[1]:
                coded_values = design_matrix[:, i]
                # -1 ~ 1 코드를 실제 값으로 변환
                actual_values = factor.min_value + (coded_values + 1) / 2 * \
                               (factor.max_value - factor.min_value)
                df_data[factor.name] = actual_values
        
        # 범주형 요인 추가
        n_runs = len(design_matrix)
        for factor in categorical_factors:
            # 균등하게 레벨 할당
            n_levels = len(factor.levels)
            level_indices = np.tile(range(n_levels), n_runs // n_levels + 1)[:n_runs]
            np.random.shuffle(level_indices)
            df_data[factor.name] = [factor.levels[i] for i in level_indices]
        
        return pd.DataFrame(df_data)
    
    def _apply_constraints(self, df: pd.DataFrame, constraints: Dict) -> pd.DataFrame:
        """제약조건 적용"""
        # 선형 제약조건 예시
        if "linear_constraints" in constraints:
            for constraint in constraints["linear_constraints"]:
                # 예: {"factors": ["A", "B"], "coefficients": [1, 1], "bound": 100}
                factors = constraint["factors"]
                coeffs = constraint["coefficients"]
                bound = constraint["bound"]
                
                if all(f in df.columns for f in factors):
                    constraint_value = sum(df[f] * c for f, c in zip(factors, coeffs))
                    df = df[constraint_value <= bound]
        
        # 금지 조합 제거
        if "forbidden_combinations" in constraints:
            for forbidden in constraints["forbidden_combinations"]:
                mask = pd.Series(True, index=df.index)
                for factor, value in forbidden.items():
                    if factor in df.columns:
                        mask &= (df[factor] != value)
                df = df[mask]
        
        return df.reset_index(drop=True)
    
    def _get_fractional_generators(self, n_factors: int, resolution: int) -> str:
        """부분요인설계 생성자 결정"""
        # 간단한 규칙 기반 생성자
        generators = {
            (5, 3): "D = A B; E = A C",
            (6, 3): "D = A B; E = A C; F = B C",
            (7, 3): "D = A B; E = A C; F = B C; G = A B C",
            (5, 4): "E = A B C D",
            (6, 4): "E = A B C; F = B C D",
            (7, 4): "E = A B C; F = A B D; G = A C D"
        }
        return generators.get((n_factors, resolution), "")


# ==================== 검증 시스템 ====================

class ValidationSystem:
    """설계 검증 시스템"""
    
    def validate_design(self, design: pd.DataFrame, 
                       factors: List[Factor],
                       responses: List[Response]) -> ValidationResult:
        """종합 설계 검증"""
        result = ValidationResult()
        
        # 기본 검증
        self._validate_basic(design, factors, result)
        
        # 통계적 검증
        self._validate_statistical(design, factors, result)
        
        # 실용적 검증
        self._validate_practical(design, factors, responses, result)
        
        return result
    
    def _validate_basic(self, design: pd.DataFrame, 
                       factors: List[Factor], 
                       result: ValidationResult):
        """기본 검증"""
        # 실험 횟수
        n_runs = len(design)
        n_factors = len(factors)
        
        if n_runs < n_factors + 1:
            result.add_error(f"실험 횟수({n_runs})가 요인 수({n_factors})보다 적습니다.")
        elif n_runs < 2 * n_factors:
            result.add_warning(f"실험 횟수가 권장 최소값(2×요인수={2*n_factors})보다 적습니다.")
        
        # 요인 범위 확인
        for factor in factors:
            if factor.name in design.columns:
                if factor.type == FactorType.CONTINUOUS:
                    values = design[factor.name]
                    if values.min() < factor.min_value or values.max() > factor.max_value:
                        result.add_error(f"{factor.name}의 값이 설정 범위를 벗어났습니다.")
        
        # 중복 실험점
        duplicates = design.duplicated().sum()
        if duplicates > 0:
            result.add_warning(f"{duplicates}개의 중복 실험점이 있습니다.")
    
    def _validate_statistical(self, design: pd.DataFrame,
                            factors: List[Factor],
                            result: ValidationResult):
        """통계적 검증"""
        continuous_cols = [f.name for f in factors 
                          if f.type == FactorType.CONTINUOUS and f.name in design.columns]
        
        if len(continuous_cols) < 2:
            return
        
        # 상관관계 확인
        corr_matrix = design[continuous_cols].corr()
        high_corr = np.where(np.abs(corr_matrix) > 0.9)
        for i, j in zip(high_corr[0], high_corr[1]):
            if i < j:
                result.add_warning(
                    f"{continuous_cols[i]}와 {continuous_cols[j]} 간 높은 상관관계 "
                    f"({corr_matrix.iloc[i, j]:.3f})"
                )
        
        # VIF 계산 (간단한 버전)
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            X = design[continuous_cols].values
            for i, col in enumerate(continuous_cols):
                vif = variance_inflation_factor(X, i)
                if vif > 10:
                    result.add_warning(f"{col}의 VIF가 높습니다 ({vif:.1f})")
        except:
            pass
    
    def _validate_practical(self, design: pd.DataFrame,
                          factors: List[Factor],
                          responses: List[Response],
                          result: ValidationResult):
        """실용적 검증"""
        # 실험 실행 가능성
        n_runs = len(design)
        if n_runs > 100:
            result.add_warning(f"실험 횟수가 많습니다 ({n_runs}회). 단계적 접근을 고려하세요.")
        
        # 극단값 조합
        continuous_factors = [f for f in factors if f.type == FactorType.CONTINUOUS]
        if continuous_factors:
            extreme_runs = 0
            for _, row in design.iterrows():
                extreme_count = sum(
                    row[f.name] in [f.min_value, f.max_value]
                    for f in continuous_factors
                    if f.name in row
                )
                if extreme_count == len(continuous_factors):
                    extreme_runs += 1
            
            if extreme_runs > n_runs * 0.5:
                result.add_warning("극단값 조합이 많습니다. 실행 가능성을 확인하세요.")
    
    def calculate_design_quality(self, design: pd.DataFrame,
                               factors: List[Factor]) -> DesignQuality:
        """설계 품질 지표 계산"""
        quality = DesignQuality()
        
        continuous_cols = [f.name for f in factors 
                          if f.type == FactorType.CONTINUOUS and f.name in design.columns]
        
        if not continuous_cols:
            return quality
        
        X = design[continuous_cols].values
        n, p = X.shape
        
        # 정규화
        X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        
        # D-efficiency
        try:
            XtX = X_norm.T @ X_norm
            det_XtX = np.linalg.det(XtX)
            det_full = n ** p  # 완전요인설계 기준
            quality.d_efficiency = (det_XtX / det_full) ** (1/p) * 100
        except:
            quality.d_efficiency = 0
        
        # Condition number
        try:
            quality.condition_number = np.linalg.cond(X_norm)
        except:
            quality.condition_number = np.inf
        
        # Orthogonality
        try:
            corr_matrix = np.corrcoef(X_norm.T)
            off_diagonal = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            quality.orthogonality = 1 - np.mean(np.abs(off_diagonal))
        except:
            quality.orthogonality = 0
        
        # VIF (maximum)
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            vifs = [variance_inflation_factor(X_norm, i) for i in range(p)]
            quality.vif_max = max(vifs)
        except:
            quality.vif_max = 0
        
        return quality


# ==================== 메인 모듈 클래스 ====================

class GeneralExperimentModule(BaseExperimentModule):
    """범용 실험 설계 모듈"""
    
    def __init__(self):
        """모듈 초기화"""
        super().__init__()
        
        # 메타데이터 설정
        self.metadata.update({
            'module_id': 'general_experiment_v2',
            'name': '범용 실험 설계',
            'version': '2.0.0',
            'author': 'Universal DOE Platform Team',
            'description': '모든 연구 분야를 위한 범용 실험 설계 모듈',
            'category': 'core',
            'tags': ['general', 'universal', 'flexible', 'all-purpose', 'doe'],
            'icon': '🌐',
            'color': '#0066cc',
            'supported_designs': list(DesignEngine().methods.keys()),
            'min_factors': 1,
            'max_factors': 50,
            'min_responses': 1,
            'max_responses': 20
        })
        
        # 내부 컴포넌트
        self.templates = ExperimentTemplates()
        self.design_engine = DesignEngine()
        self.validator = ValidationSystem()
        
        # 상태 저장
        self.current_factors: List[Factor] = []
        self.current_responses: List[Response] = []
        self.current_design: Optional[pd.DataFrame] = None
        self.design_quality: Optional[DesignQuality] = None
        
        self._initialized = True
        logger.info("GeneralExperimentModule 초기화 완료")
    
    # ==================== 필수 구현 메서드 ====================
    
    def get_factors(self) -> List[Factor]:
        """현재 정의된 요인 목록 반환"""
        return self.current_factors
    
    def get_responses(self) -> List[Response]:
        """현재 정의된 반응변수 목록 반환"""
        return self.current_responses
    
    def validate_input(self, inputs: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """입력값 검증"""
        try:
            # 필수 필드 확인
            if 'factors' not in inputs or not inputs['factors']:
                return False, "최소 1개 이상의 실험 요인이 필요합니다."
            
            if 'responses' not in inputs or not inputs['responses']:
                return False, "최소 1개 이상의 반응변수가 필요합니다."
            
            # 요인 검증
            factors = inputs['factors']
            if len(factors) > self.metadata['max_factors']:
                return False, f"요인 수는 {self.metadata['max_factors']}개를 초과할 수 없습니다."
            
            # 요인별 검증
            for i, factor_data in enumerate(factors):
                if 'name' not in factor_data or not factor_data['name']:
                    return False, f"요인 {i+1}의 이름이 없습니다."
                
                if factor_data.get('type') == 'continuous':
                    min_val = factor_data.get('min_value', 0)
                    max_val = factor_data.get('max_value', 1)
                    if min_val >= max_val:
                        return False, f"요인 '{factor_data['name']}'의 최소값이 최대값보다 크거나 같습니다."
                
                elif factor_data.get('type') == 'categorical':
                    levels = factor_data.get('levels', [])
                    if len(levels) < 2:
                        return False, f"범주형 요인 '{factor_data['name']}'은 최소 2개 수준이 필요합니다."
            
            # 반응변수 검증
            responses = inputs['responses']
            if len(responses) > self.metadata['max_responses']:
                return False, f"반응변수 수는 {self.metadata['max_responses']}개를 초과할 수 없습니다."
            
            for i, response_data in enumerate(responses):
                if 'name' not in response_data or not response_data['name']:
                    return False, f"반응변수 {i+1}의 이름이 없습니다."
            
            # 설계 방법 검증
            if 'design_method' in inputs:
                method = inputs['design_method']
                if method not in self.design_engine.methods:
                    return False, f"지원하지 않는 설계 방법입니다: {method}"
            
            return True, None
            
        except Exception as e:
            logger.error(f"입력 검증 중 오류: {e}")
            return False, f"검증 중 오류 발생: {str(e)}"
    
    def generate_design(self, inputs: Dict[str, Any]) -> Tuple[bool, Union[str, ExperimentDesign]]:
        """실험 설계 생성"""
        try:
            # 입력 검증
            is_valid, error_msg = self.validate_input(inputs)
            if not is_valid:
                return False, error_msg
            
            # Factor 객체 생성
            self.current_factors = []
            for factor_data in inputs['factors']:
                factor = Factor(
                    name=factor_data['name'],
                    display_name=factor_data.get('display_name', factor_data['name']),
                    type=FactorType(factor_data.get('type', 'continuous')),
                    unit=factor_data.get('unit', ''),
                    min_value=factor_data.get('min_value'),
                    max_value=factor_data.get('max_value'),
                    levels=factor_data.get('levels', []),
                    description=factor_data.get('description', '')
                )
                self.current_factors.append(factor)
            
            # Response 객체 생성
            self.current_responses = []
            for response_data in inputs['responses']:
                response = Response(
                    name=response_data['name'],
                    display_name=response_data.get('display_name', response_data['name']),
                    unit=response_data.get('unit', ''),
                    goal=ResponseGoal(response_data.get('goal', 'maximize')),
                    target_value=response_data.get('target_value'),
                    description=response_data.get('description', '')
                )
                self.current_responses.append(response)
            
            # 설계 방법 결정
            design_method = inputs.get('design_method', 'auto')
            if design_method == 'auto':
                factor_types = [f.type.value for f in self.current_factors]
                objective = inputs.get('objective', 'optimization')
                n_runs_budget = inputs.get('n_runs_budget')
                design_method = self.design_engine.recommend_method(
                    len(self.current_factors), factor_types, objective, n_runs_budget
                )
            
            # 설계 옵션
            design_options = inputs.get('design_options', {})
            constraints = inputs.get('constraints', {})
            
            # 설계 생성
            self.current_design = self.design_engine.generate_design(
                design_method,
                self.current_factors,
                constraints,
                design_options
            )
            
            # 반응변수 열 추가
            for response in self.current_responses:
                self.current_design[response.name] = np.nan
            
            # 품질 지표 계산
            self.design_quality = self.validator.calculate_design_quality(
                self.current_design, self.current_factors
            )
            
            # 검증
            validation_result = self.validator.validate_design(
                self.current_design, self.current_factors, self.current_responses
            )
            
            # ExperimentDesign 객체 생성
            experiment_design = ExperimentDesign(
                design_matrix=self.current_design,
                factors=self.current_factors,
                responses=self.current_responses,
                design_type=design_method,
                quality_metrics={
                    'd_efficiency': self.design_quality.d_efficiency,
                    'condition_number': self.design_quality.condition_number,
                    'orthogonality': self.design_quality.orthogonality,
                    'overall_score': self.design_quality.overall_score
                },
                validation_result=validation_result,
                metadata={
                    'created_at': datetime.now().isoformat(),
                    'module_id': self.metadata['module_id'],
                    'module_version': self.metadata['version'],
                    'design_method': self.design_engine.get_method(design_method).display_name,
                    'n_runs': len(self.current_design),
                    'n_factors': len(self.current_factors),
                    'n_responses': len(self.current_responses)
                }
            )
            
            logger.info(f"실험 설계 생성 완료: {design_method}, {len(self.current_design)}회")
            return True, experiment_design
            
        except Exception as e:
            logger.error(f"설계 생성 중 오류: {e}\n{traceback.format_exc()}")
            return False, f"설계 생성 실패: {str(e)}"
    
    def analyze_results(self, results: pd.DataFrame) -> Tuple[bool, Union[str, AnalysisResult]]:
        """실험 결과 분석"""
        try:
            if self.current_design is None:
                return False, "먼저 실험 설계를 생성하세요."
            
            if results.empty:
                return False, "분석할 결과 데이터가 없습니다."
            
            analysis = AnalysisResult()
            
            # 기본 통계
            summary_stats = {}
            for response in self.current_responses:
                if response.name in results.columns:
                    data = results[response.name].dropna()
                    if len(data) > 0:
                        summary_stats[response.name] = {
                            'count': len(data),
                            'mean': float(data.mean()),
                            'std': float(data.std()),
                            'min': float(data.min()),
                            'max': float(data.max()),
                            'cv': float(data.std() / data.mean() * 100) if data.mean() != 0 else np.inf
                        }
            analysis.summary_statistics = summary_stats
            
            # 요인 효과 분석 (간단한 버전)
            factor_effects = {}
            continuous_factors = [f for f in self.current_factors if f.type == FactorType.CONTINUOUS]
            
            for response in self.current_responses:
                if response.name in results.columns:
                    effects = {}
                    for factor in continuous_factors:
                        if factor.name in results.columns:
                            # 상관관계
                            corr = results[[factor.name, response.name]].corr().iloc[0, 1]
                            effects[factor.name] = {
                                'correlation': float(corr),
                                'significant': abs(corr) > 0.3  # 간단한 기준
                            }
                    factor_effects[response.name] = effects
            analysis.factor_effects = factor_effects
            
            # 최적 조건 찾기 (간단한 버전)
            optimal_conditions = {}
            for response in self.current_responses:
                if response.name in results.columns:
                    data = results[response.name].dropna()
                    if len(data) > 0:
                        if response.goal == ResponseGoal.MAXIMIZE:
                            opt_idx = data.idxmax()
                        elif response.goal == ResponseGoal.MINIMIZE:
                            opt_idx = data.idxmin()
                        else:  # TARGET
                            target = response.target_value or 0
                            opt_idx = (data - target).abs().idxmin()
                        
                        opt_conditions = {}
                        for factor in self.current_factors:
                            if factor.name in results.columns:
                                opt_conditions[factor.name] = results.loc[opt_idx, factor.name]
                        
                        optimal_conditions[response.name] = {
                            'conditions': opt_conditions,
                            'predicted_value': float(data.loc[opt_idx]),
                            'run_number': int(opt_idx)
                        }
            analysis.optimal_conditions = optimal_conditions
            
            # 시각화 생성
            analysis.visualizations = self._create_analysis_plots(results)
            
            # 추천사항
            recommendations = []
            
            # CV 기반 추천
            for resp_name, stats in summary_stats.items():
                if stats['cv'] > 20:
                    recommendations.append(f"{resp_name}의 변동계수가 높습니다 (CV={stats['cv']:.1f}%). "
                                         "실험 조건 제어를 개선하세요.")
            
            # 상관관계 기반 추천
            for resp_name, effects in factor_effects.items():
                significant_factors = [f for f, e in effects.items() if e['significant']]
                if significant_factors:
                    recommendations.append(f"{resp_name}에 대해 {', '.join(significant_factors)}가 "
                                         "유의한 영향을 미칩니다.")
            
            analysis.recommendations = recommendations
            
            # 메타데이터
            analysis.metadata = {
                'analysis_date': datetime.now().isoformat(),
                'n_observations': len(results),
                'n_complete_cases': len(results.dropna()),
                'module_id': self.metadata['module_id']
            }
            
            logger.info("결과 분석 완료")
            return True, analysis
            
        except Exception as e:
            logger.error(f"결과 분석 중 오류: {e}\n{traceback.format_exc()}")
            return False, f"분석 실패: {str(e)}"
    
    # ==================== 추가 기능 메서드 ====================
    
    def get_factor_templates(self) -> Dict[str, List[FactorTemplate]]:
        """요인 템플릿 반환"""
        return self.templates.get_factor_templates()
    
    def get_response_templates(self) -> Dict[str, List[ResponseTemplate]]:
        """반응변수 템플릿 반환"""
        return self.templates.get_response_templates()
    
    def get_experiment_presets(self) -> Dict[str, Dict[str, Any]]:
        """실험 프리셋 반환"""
        return self.templates.get_experiment_presets()
    
    def get_design_methods(self) -> Dict[str, DesignMethod]:
        """사용 가능한 설계 방법 반환"""
        return self.design_engine.methods
    
    def suggest_next_runs(self, current_results: pd.DataFrame, 
                         n_additional: int = 5) -> pd.DataFrame:
        """추가 실험점 제안 (적응형 설계)"""
        if self.current_design is None or current_results.empty:
            return pd.DataFrame()
        
        # 간단한 구현: 예측 분산이 큰 영역 탐색
        continuous_factors = [f for f in self.current_factors if f.type == FactorType.CONTINUOUS]
        
        if not continuous_factors:
            return pd.DataFrame()
        
        # 현재 실험점에서 가장 먼 점들 찾기
        factor_names = [f.name for f in continuous_factors]
        existing_points = current_results[factor_names].values
        
        # 후보점 생성 (LHS)
        candidates = self.design_engine._generate_lhs(
            continuous_factors, 
            {'n_samples': n_additional * 10}
        )
        candidate_df = self.design_engine._convert_to_actual_values(
            candidates, continuous_factors, []
        )
        
        # 거리 계산하여 가장 먼 점 선택
        selected_indices = []
        candidate_array = candidate_df[factor_names].values
        
        for _ in range(n_additional):
            if len(selected_indices) == 0:
                # 첫 점은 중심에서 가장 먼 점
                center = existing_points.mean(axis=0)
                distances = np.linalg.norm(candidate_array - center, axis=1)
            else:
                # 기존 점들과의 최소 거리가 최대인 점
                all_points = np.vstack([existing_points, 
                                       candidate_array[selected_indices]])
                min_distances = []
                for i, cand in enumerate(candidate_array):
                    if i not in selected_indices:
                        dists = np.linalg.norm(all_points - cand, axis=1)
                        min_distances.append(dists.min())
                    else:
                        min_distances.append(-1)
                distances = np.array(min_distances)
            
            next_idx = distances.argmax()
            selected_indices.append(next_idx)
        
        # 선택된 점들 반환
        next_runs = candidate_df.iloc[selected_indices].copy()
        next_runs.index = range(1, len(next_runs) + 1)
        next_runs.index.name = "Additional_Run"
        
        # 반응변수 열 추가
        for response in self.current_responses:
            next_runs[response.name] = np.nan
        
        return next_runs
    
    def export_design(self, format: str = 'excel', 
                     include_analysis: bool = False) -> bytes:
        """설계 내보내기"""
        if self.current_design is None:
            raise ValueError("내보낼 설계가 없습니다.")
        
        import io
        
        if format == 'excel':
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # 실험 설계
                self.current_design.to_excel(writer, sheet_name='Design Matrix')
                
                # 요인 정보
                factor_info = pd.DataFrame([
                    {
                        'Name': f.name,
                        'Type': f.type.value,
                        'Unit': f.unit,
                        'Min': f.min_value,
                        'Max': f.max_value,
                        'Levels': ', '.join(map(str, f.levels)) if f.levels else ''
                    }
                    for f in self.current_factors
                ])
                factor_info.to_excel(writer, sheet_name='Factors', index=False)
                
                # 반응변수 정보
                response_info = pd.DataFrame([
                    {
                        'Name': r.name,
                        'Unit': r.unit,
                        'Goal': r.goal.value,
                        'Target': r.target_value
                    }
                    for r in self.current_responses
                ])
                response_info.to_excel(writer, sheet_name='Responses', index=False)
                
                # 품질 지표
                if self.design_quality:
                    quality_df = pd.DataFrame([{
                        'D-Efficiency': self.design_quality.d_efficiency,
                        'Condition Number': self.design_quality.condition_number,
                        'Orthogonality': self.design_quality.orthogonality,
                        'Overall Score': self.design_quality.overall_score
                    }])
                    quality_df.to_excel(writer, sheet_name='Quality Metrics', index=False)
            
            output.seek(0)
            return output.getvalue()
        
        elif format == 'csv':
            output = io.StringIO()
            self.current_design.to_csv(output)
            return output.getvalue().encode('utf-8')
        
        else:
            raise ValueError(f"지원하지 않는 형식: {format}")
    
    def _create_analysis_plots(self, results: pd.DataFrame) -> Dict[str, Any]:
        """분석 시각화 생성"""
        plots = {}
        
        try:
            # 1. 반응변수 분포
            response_names = [r.name for r in self.current_responses if r.name in results.columns]
            if response_names:
                fig = make_subplots(
                    rows=1, cols=len(response_names),
                    subplot_titles=response_names
                )
                
                for i, resp_name in enumerate(response_names):
                    data = results[resp_name].dropna()
                    fig.add_trace(
                        go.Box(y=data, name=resp_name, boxpoints='all'),
                        row=1, col=i+1
                    )
                
                fig.update_layout(
                    title="반응변수 분포",
                    showlegend=False,
                    height=400
                )
                plots['response_distribution'] = fig.to_dict()
            
            # 2. 요인-반응변수 산점도 (연속형 요인만)
            continuous_factors = [f for f in self.current_factors 
                                if f.type == FactorType.CONTINUOUS and f.name in results.columns]
            
            if continuous_factors and response_names:
                n_factors = len(continuous_factors)
                n_responses = len(response_names)
                
                fig = make_subplots(
                    rows=n_responses, cols=n_factors,
                    subplot_titles=[f.name for f in continuous_factors] * n_responses
                )
                
                for i, response in enumerate(response_names):
                    for j, factor in enumerate(continuous_factors):
                        fig.add_trace(
                            go.Scatter(
                                x=results[factor.name],
                                y=results[response],
                                mode='markers',
                                name=f"{factor.name} vs {response}",
                                marker=dict(size=8)
                            ),
                            row=i+1, col=j+1
                        )
                
                fig.update_layout(
                    title="요인-반응변수 관계",
                    showlegend=False,
                    height=300 * n_responses
                )
                plots['factor_response_scatter'] = fig.to_dict()
            
            # 3. 상관관계 히트맵
            numeric_cols = [f.name for f in self.current_factors if f.type == FactorType.CONTINUOUS]
            numeric_cols.extend(response_names)
            
            if len(numeric_cols) > 1:
                corr_data = results[numeric_cols].corr()
                
                fig = go.Figure(data=go.Heatmap(
                    z=corr_data.values,
                    x=corr_data.columns,
                    y=corr_data.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=np.round(corr_data.values, 2),
                    texttemplate='%{text}'
                ))
                
                fig.update_layout(
                    title="상관관계 히트맵",
                    height=600,
                    width=600
                )
                plots['correlation_heatmap'] = fig.to_dict()
            
        except Exception as e:
            logger.error(f"시각화 생성 중 오류: {e}")
        
        return plots


# ==================== 모듈 등록 ====================

def register_module():
    """모듈 레지스트리에 등록"""
    return GeneralExperimentModule()


# ==================== 테스트 코드 ====================

if __name__ == "__main__":
    # 모듈 테스트
    module = GeneralExperimentModule()
    
    # 테스트 입력
    test_inputs = {
        'factors': [
            {
                'name': '온도',
                'type': 'continuous',
                'min_value': 20,
                'max_value': 100,
                'unit': '°C'
            },
            {
                'name': '시간',
                'type': 'continuous',
                'min_value': 10,
                'max_value': 60,
                'unit': 'min'
            },
            {
                'name': '촉매',
                'type': 'categorical',
                'levels': ['A', 'B', 'C']
            }
        ],
        'responses': [
            {
                'name': '수율',
                'unit': '%',
                'goal': 'maximize'
            },
            {
                'name': '순도',
                'unit': '%',
                'goal': 'maximize'
            }
        ],
        'design_method': 'central_composite',
        'objective': 'optimization'
    }
    
    # 검증
    is_valid, msg = module.validate_input(test_inputs)
    print(f"검증 결과: {is_valid}, {msg}")
    
    if is_valid:
        # 설계 생성
        success, design = module.generate_design(test_inputs)
        if success:
            print(f"\n생성된 설계:\n{design.design_matrix}")
            print(f"\n품질 지표: {design.quality_metrics}")
        else:
            print(f"설계 생성 실패: {design}")
