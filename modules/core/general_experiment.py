"""
Universal DOE Platform - General Experiment Module
범용 실험 설계 모듈 - 모든 연구 분야를 지원하는 핵심 모듈
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import json
from datetime import datetime
import traceback
import logging
from itertools import product
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.express as px

# DOE 관련 라이브러리
try:
    from pyDOE2 import (
        fullfact, fracfact, pbdesign, ccdesign, bbdesign,
        lhs, gsd, factorial, ff2n
    )
except ImportError:
    st.error("pyDOE2가 설치되지 않았습니다. pip install pyDOE2를 실행하세요.")

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
    complexity: str  # low, medium, high
    use_cases: List[str] = field(default_factory=list)


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


# ==================== 템플릿 정의 ====================

class ExperimentTemplates:
    """실험 템플릿 관리"""
    
    @staticmethod
    def get_factor_templates() -> Dict[str, List[FactorTemplate]]:
        """요인 템플릿 반환"""
        return {
            "공정 변수": [
                FactorTemplate(
                    name="온도",
                    category="공정 변수",
                    default_type=FactorType.CONTINUOUS,
                    default_unit="°C",
                    default_min=20,
                    default_max=200,
                    description="공정 온도"
                ),
                FactorTemplate(
                    name="압력",
                    category="공정 변수",
                    default_type=FactorType.CONTINUOUS,
                    default_unit="bar",
                    default_min=1,
                    default_max=10,
                    description="공정 압력"
                ),
                FactorTemplate(
                    name="시간",
                    category="공정 변수",
                    default_type=FactorType.CONTINUOUS,
                    default_unit="min",
                    default_min=10,
                    default_max=120,
                    description="반응 시간"
                ),
                FactorTemplate(
                    name="교반 속도",
                    category="공정 변수",
                    default_type=FactorType.CONTINUOUS,
                    default_unit="rpm",
                    default_min=100,
                    default_max=1000,
                    description="교반기 회전 속도"
                ),
            ],
            "조성 변수": [
                FactorTemplate(
                    name="농도",
                    category="조성 변수",
                    default_type=FactorType.CONTINUOUS,
                    default_unit="%",
                    default_min=0,
                    default_max=100,
                    description="물질 농도"
                ),
                FactorTemplate(
                    name="pH",
                    category="조성 변수",
                    default_type=FactorType.CONTINUOUS,
                    default_unit="",
                    default_min=0,
                    default_max=14,
                    description="용액 pH"
                ),
                FactorTemplate(
                    name="첨가제 종류",
                    category="조성 변수",
                    default_type=FactorType.CATEGORICAL,
                    default_unit="",
                    default_levels=["A", "B", "C"],
                    description="첨가제 종류"
                ),
            ],
            "물리적 변수": [
                FactorTemplate(
                    name="입자 크기",
                    category="물리적 변수",
                    default_type=FactorType.CONTINUOUS,
                    default_unit="μm",
                    default_min=0.1,
                    default_max=1000,
                    description="평균 입자 크기"
                ),
                FactorTemplate(
                    name="두께",
                    category="물리적 변수",
                    default_type=FactorType.CONTINUOUS,
                    default_unit="mm",
                    default_min=0.1,
                    default_max=10,
                    description="시료 두께"
                ),
            ],
            "환경 변수": [
                FactorTemplate(
                    name="습도",
                    category="환경 변수",
                    default_type=FactorType.CONTINUOUS,
                    default_unit="%RH",
                    default_min=0,
                    default_max=100,
                    description="상대 습도"
                ),
                FactorTemplate(
                    name="분위기",
                    category="환경 변수",
                    default_type=FactorType.CATEGORICAL,
                    default_unit="",
                    default_levels=["공기", "질소", "아르곤"],
                    description="반응 분위기"
                ),
            ]
        }
    
    @staticmethod
    def get_response_templates() -> Dict[str, List[ResponseTemplate]]:
        """반응변수 템플릿 반환"""
        return {
            "물성": [
                ResponseTemplate(
                    name="수율",
                    category="물성",
                    default_unit="%",
                    default_goal=ResponseGoal.MAXIMIZE,
                    typical_range=(0, 100),
                    description="반응 수율"
                ),
                ResponseTemplate(
                    name="순도",
                    category="물성",
                    default_unit="%",
                    default_goal=ResponseGoal.MAXIMIZE,
                    typical_range=(90, 100),
                    description="제품 순도"
                ),
                ResponseTemplate(
                    name="강도",
                    category="물성",
                    default_unit="MPa",
                    default_goal=ResponseGoal.MAXIMIZE,
                    description="기계적 강도"
                ),
                ResponseTemplate(
                    name="점도",
                    category="물성",
                    default_unit="cP",
                    default_goal=ResponseGoal.TARGET,
                    description="용액 점도"
                ),
            ],
            "성능": [
                ResponseTemplate(
                    name="효율",
                    category="성능",
                    default_unit="%",
                    default_goal=ResponseGoal.MAXIMIZE,
                    typical_range=(0, 100),
                    description="공정 효율"
                ),
                ResponseTemplate(
                    name="선택성",
                    category="성능",
                    default_unit="%",
                    default_goal=ResponseGoal.MAXIMIZE,
                    typical_range=(0, 100),
                    description="반응 선택성"
                ),
            ],
            "품질": [
                ResponseTemplate(
                    name="색상 L*",
                    category="품질",
                    default_unit="",
                    default_goal=ResponseGoal.TARGET,
                    typical_range=(0, 100),
                    description="CIE L*a*b* 명도"
                ),
                ResponseTemplate(
                    name="투명도",
                    category="품질",
                    default_unit="%",
                    default_goal=ResponseGoal.MAXIMIZE,
                    typical_range=(0, 100),
                    description="광학적 투명도"
                ),
            ],
            "경제성": [
                ResponseTemplate(
                    name="비용",
                    category="경제성",
                    default_unit="$/kg",
                    default_goal=ResponseGoal.MINIMIZE,
                    description="단위 생산 비용"
                ),
                ResponseTemplate(
                    name="처리 시간",
                    category="경제성",
                    default_unit="h",
                    default_goal=ResponseGoal.MINIMIZE,
                    description="총 처리 시간"
                ),
            ]
        }
    
    @staticmethod
    def get_design_methods() -> List[DesignMethod]:
        """실험설계법 목록 반환"""
        return [
            DesignMethod(
                name="full_factorial",
                display_name="완전요인설계",
                description="모든 요인 수준의 조합을 실험하는 가장 기본적인 설계",
                min_factors=2,
                max_factors=5,
                supports_categorical=True,
                supports_constraints=False,
                complexity="low",
                use_cases=["스크리닝", "주효과 분석", "교호작용 분석"]
            ),
            DesignMethod(
                name="fractional_factorial",
                display_name="부분요인설계",
                description="완전요인설계의 일부만 실험하여 효율성을 높인 설계",
                min_factors=3,
                max_factors=10,
                supports_categorical=True,
                supports_constraints=False,
                complexity="medium",
                use_cases=["많은 요인 스크리닝", "주효과 중심 분석"]
            ),
            DesignMethod(
                name="ccd",
                display_name="중심합성설계 (CCD)",
                description="2차 모델 적합을 위한 반응표면 설계",
                min_factors=2,
                max_factors=6,
                supports_categorical=False,
                supports_constraints=True,
                complexity="medium",
                use_cases=["최적화", "곡률 효과 분석", "반응표면 모델링"]
            ),
            DesignMethod(
                name="bbd",
                display_name="Box-Behnken 설계",
                description="3수준 요인을 위한 효율적인 반응표면 설계",
                min_factors=3,
                max_factors=5,
                supports_categorical=False,
                supports_constraints=True,
                complexity="medium",
                use_cases=["최적화", "극값 회피", "효율적 실험"]
            ),
            DesignMethod(
                name="plackett_burman",
                display_name="Plackett-Burman 설계",
                description="많은 요인의 주효과를 효율적으로 스크리닝",
                min_factors=4,
                max_factors=47,
                supports_categorical=False,
                supports_constraints=False,
                complexity="low",
                use_cases=["대규모 스크리닝", "중요 요인 식별"]
            ),
            DesignMethod(
                name="d_optimal",
                display_name="D-최적 설계",
                description="제약조건이 있을 때 최적의 실험점을 선택",
                min_factors=2,
                max_factors=10,
                supports_categorical=True,
                supports_constraints=True,
                complexity="high",
                use_cases=["제약조건 처리", "비정형 설계 공간", "맞춤형 설계"]
            ),
            DesignMethod(
                name="latin_hypercube",
                display_name="라틴 하이퍼큐브 샘플링",
                description="설계 공간을 균등하게 탐색하는 공간충진 설계",
                min_factors=2,
                max_factors=20,
                supports_categorical=False,
                supports_constraints=True,
                complexity="low",
                use_cases=["컴퓨터 실험", "시뮬레이션", "초기 탐색"]
            ),
            DesignMethod(
                name="mixture",
                display_name="혼합물 설계",
                description="성분 비율의 합이 1인 혼합물 실험 설계",
                min_factors=3,
                max_factors=10,
                supports_categorical=False,
                supports_constraints=True,
                complexity="high",
                use_cases=["배합 최적화", "조성 연구", "제형 개발"]
            ),
            DesignMethod(
                name="taguchi",
                display_name="다구치 설계",
                description="강건 설계를 위한 직교배열표 기반 설계",
                min_factors=2,
                max_factors=15,
                supports_categorical=True,
                supports_constraints=False,
                complexity="medium",
                use_cases=["품질 개선", "강건 설계", "잡음 요인 제어"]
            ),
            DesignMethod(
                name="custom",
                display_name="사용자 정의 설계",
                description="사용자가 직접 실험점을 지정하는 설계",
                min_factors=1,
                max_factors=50,
                supports_categorical=True,
                supports_constraints=True,
                complexity="low",
                use_cases=["특수 목적", "기존 데이터 활용", "단계적 실험"]
            )
        ]


# ==================== 메인 모듈 클래스 ====================

class GeneralExperimentModule(BaseExperimentModule):
    """범용 실험 설계 모듈"""
    
    def __init__(self):
        """모듈 초기화"""
        super().__init__()
        
        # 메타데이터 업데이트
        self.metadata.update({
            'module_id': 'general_experiment_v2',
            'name': '범용 실험 설계',
            'version': '2.0.0',
            'author': 'Universal DOE Platform Team',
            'description': '모든 연구 분야를 위한 범용 실험 설계 모듈',
            'category': 'core',
            'tags': ['general', 'universal', 'flexible', 'all-purpose'],
            'icon': '🌐',
            'color': '#0066cc'
        })
        
        # 템플릿 매니저
        self.templates = ExperimentTemplates()
        
        # 사용자 정의 요인/반응변수 저장
        self.custom_factors: List[Factor] = []
        self.custom_responses: List[Response] = []
        
        # 설계 엔진
        self.design_engine = DesignEngine()
        
        # 검증 시스템
        self.validator = ValidationSystem()
        
        self._initialized = True
        
    # ==================== 필수 구현 메서드 ====================
    
    def get_factors(self) -> List[Factor]:
        """실험 요인 목록 반환"""
        return self.custom_factors
    
    def get_responses(self) -> List[Response]:
        """반응변수 목록 반환"""
        return self.custom_responses
    
    def validate_input(self, inputs: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """입력값 검증"""
        try:
            # 요인 검증
            if 'factors' not in inputs or not inputs['factors']:
                return False, "최소 1개 이상의 실험 요인이 필요합니다."
            
            # 반응변수 검증
            if 'responses' not in inputs or not inputs['responses']:
                return False, "최소 1개 이상의 반응변수가 필요합니다."
            
            # 설계 방법 검증
            if 'design_method' not in inputs:
                return False, "실험설계법을 선택해주세요."
            
            # 각 요인 검증
            for factor in inputs['factors']:
                valid, msg = self._validate_factor(factor)
                if not valid:
                    return False, msg
            
            # 각 반응변수 검증
            for response in inputs['responses']:
                valid, msg = self._validate_response(response)
                if not valid:
                    return False, msg
            
            # 설계별 특수 검증
            method = inputs['design_method']
            valid, msg = self._validate_design_specific(method, inputs)
            if not valid:
                return False, msg
            
            return True, None
            
        except Exception as e:
            logger.error(f"입력 검증 중 오류: {str(e)}")
            return False, f"검증 중 오류 발생: {str(e)}"
    
    def generate_design(self, inputs: Dict[str, Any]) -> ExperimentDesign:
        """실험 설계 생성"""
        try:
            # 입력 데이터 추출
            self.custom_factors = self._create_factors_from_input(inputs['factors'])
            self.custom_responses = self._create_responses_from_input(inputs['responses'])
            design_method = inputs['design_method']
            design_params = inputs.get('design_params', {})
            
            # 설계 생성
            design_matrix = self.design_engine.generate_design_matrix(
                design_method,
                self.custom_factors,
                design_params
            )
            
            # 설계 평가
            quality_metrics = self.design_engine.evaluate_design(
                design_matrix,
                self.custom_factors
            )
            
            # 실행 순서 생성
            run_order = self._generate_run_order(design_matrix, inputs.get('randomize', True))
            
            # 결과 생성
            design = ExperimentDesign(
                design_id=f"GEN_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                name=inputs.get('name', 'General Experiment'),
                description=inputs.get('description', ''),
                factors=self.custom_factors,
                responses=self.custom_responses,
                design_matrix=design_matrix,
                run_order=run_order,
                metadata={
                    'design_method': design_method,
                    'design_params': design_params,
                    'quality_metrics': quality_metrics,
                    'total_runs': len(design_matrix),
                    'created_at': datetime.now().isoformat()
                }
            )
            
            return design
            
        except Exception as e:
            logger.error(f"설계 생성 중 오류: {str(e)}\n{traceback.format_exc()}")
            raise
    
    def analyze_results(self, data: pd.DataFrame) -> AnalysisResult:
        """실험 결과 분석"""
        try:
            analysis = AnalysisResult()
            
            # 기술 통계
            analysis.summary_statistics = self._calculate_summary_stats(data)
            
            # 주효과 분석
            analysis.main_effects = self._analyze_main_effects(data)
            
            # 교호작용 분석
            if len(self.custom_factors) >= 2:
                analysis.interactions = self._analyze_interactions(data)
            
            # 회귀 모델
            analysis.regression_models = self._fit_regression_models(data)
            
            # 최적 조건 찾기
            analysis.optimal_conditions = self._find_optimal_conditions(data)
            
            # 시각화 생성
            analysis.visualizations = self._create_visualizations(data)
            
            # 추천사항 생성
            analysis.recommendations = self._generate_recommendations(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"결과 분석 중 오류: {str(e)}")
            raise
    
    # ==================== UI 메서드 ====================
    
    def render_design_interface(self) -> Dict[str, Any]:
        """실험 설계 인터페이스 렌더링"""
        st.header("🌐 범용 실험 설계")
        
        inputs = {}
        
        # 기본 정보
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                inputs['name'] = st.text_input("실험 이름", value="새 실험")
            with col2:
                inputs['description'] = st.text_input("설명")
        
        # 탭 인터페이스
        tab1, tab2, tab3, tab4 = st.tabs(["📊 요인 설정", "🎯 반응변수 설정", "🔧 설계 방법", "⚙️ 고급 설정"])
        
        with tab1:
            inputs['factors'] = self._render_factor_interface()
        
        with tab2:
            inputs['responses'] = self._render_response_interface()
        
        with tab3:
            inputs['design_method'], inputs['design_params'] = self._render_design_method_interface()
        
        with tab4:
            inputs.update(self._render_advanced_settings())
        
        # 검증 및 미리보기
        if st.button("🔍 설계 검증 및 미리보기", type="primary"):
            valid, msg = self.validate_input(inputs)
            
            if valid:
                st.success("✅ 입력값 검증 통과!")
                
                # 설계 미리보기
                with st.spinner("설계 생성 중..."):
                    try:
                        design = self.generate_design(inputs)
                        self._render_design_preview(design)
                        
                        # 세션에 저장
                        st.session_state['current_design'] = design
                        st.session_state['design_inputs'] = inputs
                        
                    except Exception as e:
                        st.error(f"설계 생성 실패: {str(e)}")
            else:
                st.error(f"❌ 검증 실패: {msg}")
        
        return inputs
    
    def _render_factor_interface(self) -> List[Dict[str, Any]]:
        """요인 설정 인터페이스"""
        factors = []
        
        # 템플릿에서 추가
        st.subheader("템플릿에서 요인 추가")
        
        templates = self.templates.get_factor_templates()
        
        # 카테고리 선택
        col1, col2 = st.columns([1, 3])
        with col1:
            category = st.selectbox("카테고리", list(templates.keys()))
        
        with col2:
            if category:
                template_options = [t.name for t in templates[category]]
                selected_templates = st.multiselect(
                    "템플릿 선택",
                    template_options,
                    help="여러 개 선택 가능"
                )
        
        if st.button("템플릿 추가"):
            for template_name in selected_templates:
                template = next(t for t in templates[category] if t.name == template_name)
                factors.append(self._template_to_factor_dict(template))
            st.success(f"{len(selected_templates)}개 요인이 추가되었습니다.")
            st.rerun()
        
        # 사용자 정의 요인
        st.subheader("사용자 정의 요인")
        
        if 'custom_factors' not in st.session_state:
            st.session_state.custom_factors = []
        
        # 요인 추가 폼
        with st.expander("➕ 새 요인 추가"):
            new_factor = self._render_factor_form()
            if st.button("요인 추가", key="add_custom_factor"):
                st.session_state.custom_factors.append(new_factor)
                st.success("요인이 추가되었습니다.")
                st.rerun()
        
        # 기존 요인 표시 및 편집
        if st.session_state.custom_factors:
            st.write("**현재 요인 목록:**")
            
            for i, factor in enumerate(st.session_state.custom_factors):
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                    
                    with col1:
                        st.write(f"**{factor['name']}**")
                        st.caption(f"{factor['type']} | {factor.get('unit', '')}")
                    
                    with col2:
                        if factor['type'] == 'continuous':
                            st.write(f"범위: {factor['min_value']} - {factor['max_value']}")
                        else:
                            st.write(f"수준: {', '.join(map(str, factor['levels']))}")
                    
                    with col3:
                        if st.button("✏️ 편집", key=f"edit_factor_{i}"):
                            st.session_state[f'editing_factor_{i}'] = True
                    
                    with col4:
                        if st.button("🗑️", key=f"delete_factor_{i}"):
                            st.session_state.custom_factors.pop(i)
                            st.rerun()
                
                # 편집 모드
                if st.session_state.get(f'editing_factor_{i}', False):
                    edited_factor = self._render_factor_form(factor)
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("저장", key=f"save_factor_{i}"):
                            st.session_state.custom_factors[i] = edited_factor
                            st.session_state[f'editing_factor_{i}'] = False
                            st.rerun()
                    with col2:
                        if st.button("취소", key=f"cancel_factor_{i}"):
                            st.session_state[f'editing_factor_{i}'] = False
                            st.rerun()
        
        return st.session_state.custom_factors
    
    def _render_factor_form(self, existing_factor: Optional[Dict] = None) -> Dict[str, Any]:
        """요인 입력 폼"""
        factor = existing_factor or {}
        
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("요인 이름", value=factor.get('name', ''), key="factor_name_input")
            factor_type = st.selectbox(
                "요인 타입",
                ['continuous', 'categorical', 'discrete'],
                index=['continuous', 'categorical', 'discrete'].index(factor.get('type', 'continuous')),
                key="factor_type_input"
            )
        
        with col2:
            unit = st.text_input("단위", value=factor.get('unit', ''), key="factor_unit_input")
            description = st.text_input("설명", value=factor.get('description', ''), key="factor_desc_input")
        
        if factor_type == 'continuous':
            col1, col2 = st.columns(2)
            with col1:
                min_val = st.number_input("최소값", value=factor.get('min_value', 0.0), key="factor_min_input")
            with col2:
                max_val = st.number_input("최대값", value=factor.get('max_value', 100.0), key="factor_max_input")
            
            levels = None
        else:
            levels_str = st.text_input(
                "수준 (쉼표로 구분)",
                value=', '.join(map(str, factor.get('levels', []))),
                key="factor_levels_input"
            )
            levels = [l.strip() for l in levels_str.split(',') if l.strip()]
            min_val = None
            max_val = None
        
        return {
            'name': name,
            'type': factor_type,
            'unit': unit,
            'description': description,
            'min_value': min_val,
            'max_value': max_val,
            'levels': levels
        }
    
    def _render_response_interface(self) -> List[Dict[str, Any]]:
        """반응변수 설정 인터페이스"""
        responses = []
        
        # 템플릿에서 추가
        st.subheader("템플릿에서 반응변수 추가")
        
        templates = self.templates.get_response_templates()
        
        col1, col2 = st.columns([1, 3])
        with col1:
            category = st.selectbox("카테고리", list(templates.keys()), key="response_category")
        
        with col2:
            if category:
                template_options = [t.name for t in templates[category]]
                selected_templates = st.multiselect(
                    "템플릿 선택",
                    template_options,
                    key="response_templates"
                )
        
        if st.button("템플릿 추가", key="add_response_template"):
            for template_name in selected_templates:
                template = next(t for t in templates[category] if t.name == template_name)
                responses.append(self._template_to_response_dict(template))
            st.success(f"{len(selected_templates)}개 반응변수가 추가되었습니다.")
            st.rerun()
        
        # 사용자 정의 반응변수
        st.subheader("사용자 정의 반응변수")
        
        if 'custom_responses' not in st.session_state:
            st.session_state.custom_responses = []
        
        # 반응변수 추가 폼
        with st.expander("➕ 새 반응변수 추가"):
            new_response = self._render_response_form()
            if st.button("반응변수 추가", key="add_custom_response"):
                st.session_state.custom_responses.append(new_response)
                st.success("반응변수가 추가되었습니다.")
                st.rerun()
        
        # 기존 반응변수 표시
        if st.session_state.custom_responses:
            st.write("**현재 반응변수 목록:**")
            
            for i, response in enumerate(st.session_state.custom_responses):
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                    
                    with col1:
                        st.write(f"**{response['name']}**")
                        st.caption(f"{response['unit']} | {response['goal']}")
                    
                    with col2:
                        if response['goal'] == 'target':
                            st.write(f"목표: {response.get('target_value', 'N/A')}")
                        else:
                            st.write(f"목표: {response['goal']}")
                    
                    with col3:
                        st.write(f"가중치: {response.get('weight', 1.0)}")
                    
                    with col4:
                        if st.button("🗑️", key=f"delete_response_{i}"):
                            st.session_state.custom_responses.pop(i)
                            st.rerun()
        
        return st.session_state.custom_responses
    
    def _render_response_form(self, existing_response: Optional[Dict] = None) -> Dict[str, Any]:
        """반응변수 입력 폼"""
        response = existing_response or {}
        
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("반응변수 이름", value=response.get('name', ''), key="response_name_input")
            unit = st.text_input("단위", value=response.get('unit', ''), key="response_unit_input")
        
        with col2:
            goal = st.selectbox(
                "최적화 목표",
                ['maximize', 'minimize', 'target'],
                index=['maximize', 'minimize', 'target'].index(response.get('goal', 'maximize')),
                key="response_goal_input"
            )
            weight = st.number_input("가중치", min_value=0.0, max_value=10.0, value=response.get('weight', 1.0), key="response_weight_input")
        
        if goal == 'target':
            target_value = st.number_input("목표값", value=response.get('target_value', 0.0), key="response_target_input")
        else:
            target_value = None
        
        measurement_method = st.text_input("측정 방법", value=response.get('measurement_method', ''), key="response_method_input")
        
        return {
            'name': name,
            'unit': unit,
            'goal': goal,
            'weight': weight,
            'target_value': target_value,
            'measurement_method': measurement_method
        }
    
    def _render_design_method_interface(self) -> Tuple[str, Dict[str, Any]]:
        """설계 방법 선택 인터페이스"""
        st.subheader("실험설계법 선택")
        
        # 현재 요인 수 확인
        num_factors = len(st.session_state.get('custom_factors', []))
        
        if num_factors == 0:
            st.warning("먼저 실험 요인을 추가해주세요.")
            return None, {}
        
        # 사용 가능한 설계법 필터링
        available_methods = []
        for method in self.templates.get_design_methods():
            if method.min_factors <= num_factors <= method.max_factors:
                # 범주형 요인 체크
                has_categorical = any(f['type'] == 'categorical' for f in st.session_state.custom_factors)
                if has_categorical and not method.supports_categorical:
                    continue
                available_methods.append(method)
        
        if not available_methods:
            st.error(f"{num_factors}개 요인에 사용 가능한 설계법이 없습니다.")
            return None, {}
        
        # AI 추천
        with st.container():
            st.info("🤖 **AI 추천**: " + self._get_ai_recommendation(num_factors))
        
        # 설계법 선택
        method_names = [m.display_name for m in available_methods]
        selected_name = st.selectbox("설계법 선택", method_names)
        
        selected_method = next(m for m in available_methods if m.display_name == selected_name)
        
        # 설계법 설명
        with st.expander("ℹ️ 설계법 상세 정보"):
            st.write(f"**{selected_method.display_name}**")
            st.write(selected_method.description)
            st.write(f"**복잡도**: {selected_method.complexity}")
            st.write(f"**사용 사례**: {', '.join(selected_method.use_cases)}")
        
        # 설계 파라미터
        design_params = {}
        
        st.subheader("설계 파라미터")
        
        if selected_method.name == "full_factorial":
            # 각 요인의 수준 수 설정
            st.write("각 요인의 수준 수를 설정하세요:")
            levels = []
            for factor in st.session_state.custom_factors:
                if factor['type'] == 'continuous':
                    n_levels = st.number_input(
                        f"{factor['name']} 수준 수",
                        min_value=2,
                        max_value=5,
                        value=3,
                        key=f"levels_{factor['name']}"
                    )
                    levels.append(n_levels)
                else:
                    levels.append(len(factor['levels']))
            design_params['levels'] = levels
            
            # 중심점
            design_params['center_points'] = st.number_input(
                "중심점 개수",
                min_value=0,
                max_value=10,
                value=3
            )
            
        elif selected_method.name == "ccd":
            design_params['alpha'] = st.selectbox(
                "Alpha 값",
                ['rotatable', 'orthogonal', 'face'],
                help="rotatable: 회전가능, orthogonal: 직교, face: 면중심"
            )
            design_params['center_points'] = st.number_input(
                "중심점 개수",
                min_value=1,
                max_value=10,
                value=3
            )
            
        elif selected_method.name == "bbd":
            design_params['center_points'] = st.number_input(
                "중심점 개수",
                min_value=1,
                max_value=10,
                value=3
            )
            
        elif selected_method.name == "d_optimal":
            design_params['n_runs'] = st.number_input(
                "실험 횟수",
                min_value=num_factors + 1,
                max_value=100,
                value=min(20, 2 * num_factors)
            )
            design_params['criterion'] = st.selectbox(
                "최적화 기준",
                ['D', 'A', 'I', 'G'],
                help="D: 결정계수, A: 평균분산, I: 적분분산, G: 최대분산"
            )
            
        elif selected_method.name == "latin_hypercube":
            design_params['n_samples'] = st.number_input(
                "샘플 수",
                min_value=num_factors + 1,
                max_value=1000,
                value=min(50, 10 * num_factors)
            )
            design_params['criterion'] = st.selectbox(
                "샘플링 기준",
                ['maximin', 'centermaximin', 'correlation'],
                help="최적 공간 충진을 위한 기준"
            )
        
        # 반복 실험
        design_params['replicates'] = st.number_input(
            "반복 실험 횟수",
            min_value=1,
            max_value=5,
            value=1,
            help="각 실험점에서의 반복 횟수"
        )
        
        return selected_method.name, design_params
    
    def _render_advanced_settings(self) -> Dict[str, Any]:
        """고급 설정 인터페이스"""
        settings = {}
        
        st.subheader("고급 설정")
        
        # 실행 순서
        col1, col2 = st.columns(2)
        with col1:
            settings['randomize'] = st.checkbox(
                "실행 순서 랜덤화",
                value=True,
                help="실험 순서를 무작위로 배치하여 시간 효과 제거"
            )
        
        with col2:
            if settings['randomize']:
                settings['blocks'] = st.number_input(
                    "블록 수",
                    min_value=1,
                    max_value=10,
                    value=1,
                    help="실험을 여러 블록으로 나누어 실행"
                )
        
        # 제약조건
        st.subheader("제약조건")
        
        settings['constraints'] = []
        
        if st.checkbox("제약조건 추가"):
            constraint_type = st.selectbox(
                "제약조건 유형",
                ["선형 제약", "실행 불가능 조합", "필수 포함 실험점"]
            )
            
            if constraint_type == "선형 제약":
                st.write("예: 2*X1 + 3*X2 <= 100")
                constraint_expr = st.text_input("제약조건 수식")
                if constraint_expr:
                    settings['constraints'].append({
                        'type': 'linear',
                        'expression': constraint_expr
                    })
            
            elif constraint_type == "실행 불가능 조합":
                st.write("특정 요인 조합이 실행 불가능한 경우")
                # TODO: 구현
            
            elif constraint_type == "필수 포함 실험점":
                st.write("반드시 포함해야 하는 실험 조합")
                # TODO: 구현
        
        # 최적화 설정
        st.subheader("최적화 설정")
        
        settings['optimization'] = {
            'max_iterations': st.number_input(
                "최대 반복 횟수",
                min_value=10,
                max_value=1000,
                value=100
            ),
            'convergence_tol': st.number_input(
                "수렴 허용 오차",
                min_value=0.0001,
                max_value=0.1,
                value=0.001,
                format="%.4f"
            )
        }
        
        return settings
    
    def _render_design_preview(self, design: ExperimentDesign):
        """설계 미리보기"""
        st.subheader("📋 실험 설계 미리보기")
        
        # 설계 요약
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("총 실험 횟수", design.metadata['total_runs'])
        with col2:
            st.metric("요인 수", len(design.factors))
        with col3:
            st.metric("반응변수 수", len(design.responses))
        with col4:
            st.metric("예상 소요 시간", f"{design.metadata['total_runs'] * 2}시간")
        
        # 품질 지표
        if 'quality_metrics' in design.metadata:
            st.subheader("품질 지표")
            metrics = design.metadata['quality_metrics']
            
            cols = st.columns(len(metrics))
            for i, (key, value) in enumerate(metrics.items()):
                with cols[i]:
                    st.metric(key, f"{value:.3f}")
        
        # 설계 매트릭스
        st.subheader("실험 설계 매트릭스")
        
        # DataFrame 생성
        df_data = []
        for i, run in enumerate(design.design_matrix):
            row = {'Run': i+1}
            for j, factor in enumerate(design.factors):
                row[factor.name] = run[j]
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # 실행 순서 추가
        if design.run_order:
            df['실행 순서'] = [design.run_order[i] for i in range(len(df))]
        
        st.dataframe(df, use_container_width=True)
        
        # 시각화
        if len(design.factors) >= 2:
            st.subheader("설계 공간 시각화")
            
            if len(design.factors) == 2:
                # 2D 산점도
                fig = px.scatter(
                    df,
                    x=design.factors[0].name,
                    y=design.factors[1].name,
                    title="2D 설계 공간"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                # 평행 좌표 플롯
                factor_names = [f.name for f in design.factors]
                fig = px.parallel_coordinates(
                    df[factor_names],
                    title="다차원 설계 공간 (평행 좌표)"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # 다운로드 옵션
        st.subheader("📥 내보내기")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                "CSV로 다운로드",
                csv,
                "experiment_design.csv",
                "text/csv"
            )
        
        with col2:
            # Excel 다운로드 (openpyxl 필요)
            try:
                import io
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Design', index=False)
                
                st.download_button(
                    "Excel로 다운로드",
                    buffer.getvalue(),
                    "experiment_design.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except ImportError:
                st.info("Excel 내보내기를 위해 openpyxl 설치가 필요합니다.")
        
        with col3:
            # JSON 다운로드
            design_dict = {
                'design_id': design.design_id,
                'name': design.name,
                'factors': [f.__dict__ for f in design.factors],
                'responses': [r.__dict__ for r in design.responses],
                'design_matrix': design.design_matrix.tolist(),
                'metadata': design.metadata
            }
            
            st.download_button(
                "JSON으로 다운로드",
                json.dumps(design_dict, indent=2),
                "experiment_design.json",
                "application/json"
            )
    
    # ==================== 내부 헬퍼 메서드 ====================
    
    def _validate_factor(self, factor: Dict) -> Tuple[bool, Optional[str]]:
        """개별 요인 검증"""
        if not factor.get('name'):
            return False, "요인 이름이 필요합니다."
        
        if factor['type'] == 'continuous':
            if factor.get('min_value') is None or factor.get('max_value') is None:
                return False, f"{factor['name']}: 연속형 요인은 최소/최대값이 필요합니다."
            if factor['min_value'] >= factor['max_value']:
                return False, f"{factor['name']}: 최소값은 최대값보다 작아야 합니다."
        else:
            if not factor.get('levels'):
                return False, f"{factor['name']}: 범주형/이산형 요인은 수준이 필요합니다."
            if len(factor['levels']) < 2:
                return False, f"{factor['name']}: 최소 2개 이상의 수준이 필요합니다."
        
        return True, None
    
    def _validate_response(self, response: Dict) -> Tuple[bool, Optional[str]]:
        """개별 반응변수 검증"""
        if not response.get('name'):
            return False, "반응변수 이름이 필요합니다."
        
        if response.get('goal') not in ['maximize', 'minimize', 'target']:
            return False, f"{response['name']}: 올바른 최적화 목표를 선택하세요."
        
        if response['goal'] == 'target' and response.get('target_value') is None:
            return False, f"{response['name']}: 목표값이 필요합니다."
        
        return True, None
    
    def _validate_design_specific(self, method: str, inputs: Dict) -> Tuple[bool, Optional[str]]:
        """설계법별 특수 검증"""
        num_factors = len(inputs['factors'])
        
        # 설계법별 검증
        if method == 'mixture':
            # 혼합물 설계는 모든 요인의 합이 1이어야 함
            continuous_factors = [f for f in inputs['factors'] if f['type'] == 'continuous']
            if len(continuous_factors) < 3:
                return False, "혼합물 설계는 최소 3개 이상의 연속형 요인이 필요합니다."
        
        return True, None
    
    def _create_factors_from_input(self, factor_dicts: List[Dict]) -> List[Factor]:
        """입력 딕셔너리에서 Factor 객체 생성"""
        factors = []
        
        for f_dict in factor_dicts:
            if f_dict['type'] == 'continuous':
                factor_type = FactorType.CONTINUOUS
            elif f_dict['type'] == 'categorical':
                factor_type = FactorType.CATEGORICAL
            elif f_dict['type'] == 'discrete':
                factor_type = FactorType.DISCRETE
            else:
                factor_type = FactorType.ORDINAL
            
            factor = Factor(
                name=f_dict['name'],
                type=factor_type,
                unit=f_dict.get('unit', ''),
                min_value=f_dict.get('min_value'),
                max_value=f_dict.get('max_value'),
                levels=f_dict.get('levels', []),
                description=f_dict.get('description', '')
            )
            factors.append(factor)
        
        return factors
    
    def _create_responses_from_input(self, response_dicts: List[Dict]) -> List[Response]:
        """입력 딕셔너리에서 Response 객체 생성"""
        responses = []
        
        for r_dict in response_dicts:
            if r_dict['goal'] == 'maximize':
                goal = ResponseGoal.MAXIMIZE
            elif r_dict['goal'] == 'minimize':
                goal = ResponseGoal.MINIMIZE
            else:
                goal = ResponseGoal.TARGET
            
            response = Response(
                name=r_dict['name'],
                unit=r_dict.get('unit', ''),
                goal=goal,
                target_value=r_dict.get('target_value'),
                weight=r_dict.get('weight', 1.0),
                measurement_method=r_dict.get('measurement_method', '')
            )
            responses.append(response)
        
        return responses
    
    def _template_to_factor_dict(self, template: FactorTemplate) -> Dict[str, Any]:
        """요인 템플릿을 딕셔너리로 변환"""
        return {
            'name': template.name,
            'type': template.default_type.value,
            'unit': template.default_unit,
            'min_value': template.default_min,
            'max_value': template.default_max,
            'levels': template.default_levels,
            'description': template.description
        }
    
    def _template_to_response_dict(self, template: ResponseTemplate) -> Dict[str, Any]:
        """반응변수 템플릿을 딕셔너리로 변환"""
        return {
            'name': template.name,
            'unit': template.default_unit,
            'goal': template.default_goal.value,
            'measurement_method': template.measurement_method,
            'description': template.description,
            'weight': 1.0
        }
    
    def _get_ai_recommendation(self, num_factors: int) -> str:
        """AI 기반 설계법 추천"""
        if num_factors <= 3:
            return "요인이 적으므로 완전요인설계를 추천합니다. 모든 조합을 실험하여 정확한 분석이 가능합니다."
        elif num_factors <= 5:
            return "중심합성설계(CCD) 또는 Box-Behnken 설계를 추천합니다. 2차 효과까지 분석 가능합니다."
        elif num_factors <= 10:
            return "부분요인설계 또는 D-최적 설계를 추천합니다. 효율적으로 주요 효과를 파악할 수 있습니다."
        else:
            return "Plackett-Burman 설계로 스크리닝 후, 중요 요인만으로 상세 실험을 진행하세요."
    
    def _generate_run_order(self, design_matrix: np.ndarray, randomize: bool) -> List[int]:
        """실험 실행 순서 생성"""
        n_runs = len(design_matrix)
        
        if randomize:
            run_order = np.random.permutation(n_runs).tolist()
        else:
            run_order = list(range(n_runs))
        
        return run_order
    
    def _calculate_summary_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """기술 통계 계산"""
        stats = {}
        
        # 반응변수별 통계
        for response in self.custom_responses:
            if response.name in data.columns:
                col_data = data[response.name].dropna()
                stats[response.name] = {
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'cv': col_data.std() / col_data.mean() * 100 if col_data.mean() != 0 else 0
                }
        
        return stats
    
    def _analyze_main_effects(self, data: pd.DataFrame) -> Dict[str, Any]:
        """주효과 분석"""
        # TODO: 구현
        return {}
    
    def _analyze_interactions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """교호작용 분석"""
        # TODO: 구현
        return {}
    
    def _fit_regression_models(self, data: pd.DataFrame) -> Dict[str, Any]:
        """회귀 모델 적합"""
        # TODO: 구현
        return {}
    
    def _find_optimal_conditions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """최적 조건 찾기"""
        # TODO: 구현
        return {}
    
    def _create_visualizations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """시각화 생성"""
        # TODO: 구현
        return {}
    
    def _generate_recommendations(self, analysis: AnalysisResult) -> List[str]:
        """분석 기반 추천사항 생성"""
        recommendations = []
        
        # 기본 추천사항
        recommendations.append("실험 설계가 성공적으로 생성되었습니다.")
        
        # TODO: 분석 결과 기반 추천사항 추가
        
        return recommendations


# ==================== 설계 엔진 ====================

class DesignEngine:
    """실험 설계 생성 엔진"""
    
    def generate_design_matrix(self, method: str, factors: List[Factor], params: Dict) -> np.ndarray:
        """설계 매트릭스 생성"""
        
        if method == "full_factorial":
            return self._generate_full_factorial(factors, params)
        elif method == "fractional_factorial":
            return self._generate_fractional_factorial(factors, params)
        elif method == "ccd":
            return self._generate_ccd(factors, params)
        elif method == "bbd":
            return self._generate_bbd(factors, params)
        elif method == "plackett_burman":
            return self._generate_plackett_burman(factors, params)
        elif method == "d_optimal":
            return self._generate_d_optimal(factors, params)
        elif method == "latin_hypercube":
            return self._generate_latin_hypercube(factors, params)
        elif method == "custom":
            return self._generate_custom(factors, params)
        else:
            raise ValueError(f"Unknown design method: {method}")
    
    def evaluate_design(self, design_matrix: np.ndarray, factors: List[Factor]) -> Dict[str, float]:
        """설계 품질 평가"""
        metrics = {}
        
        # 실험 횟수
        metrics['총 실험수'] = len(design_matrix)
        
        # D-efficiency 계산 (간단한 버전)
        try:
            X = design_matrix
            XtX = X.T @ X
            det_XtX = np.linalg.det(XtX)
            n = len(X)
            p = len(factors)
            d_eff = (det_XtX / n**p) ** (1/p)
            metrics['D-efficiency'] = min(d_eff, 1.0)
        except:
            metrics['D-efficiency'] = 0.0
        
        # 균형성
        balance_score = self._calculate_balance(design_matrix)
        metrics['균형성'] = balance_score
        
        # 직교성
        orthogonality = self._calculate_orthogonality(design_matrix)
        metrics['직교성'] = orthogonality
        
        return metrics
    
    def _generate_full_factorial(self, factors: List[Factor], params: Dict) -> np.ndarray:
        """완전요인설계 생성"""
        levels = params.get('levels', [3] * len(factors))
        
        # pyDOE2 사용
        design = fullfact(levels)
        
        # 실제 값으로 변환
        scaled_design = np.zeros_like(design)
        for i, factor in enumerate(factors):
            if factor.type == FactorType.CONTINUOUS:
                # -1 to 1 스케일을 실제 범위로 변환
                min_val = factor.min_value
                max_val = factor.max_value
                scaled_design[:, i] = min_val + (design[:, i] / (levels[i] - 1)) * (max_val - min_val)
            else:
                scaled_design[:, i] = design[:, i]
        
        # 중심점 추가
        n_center = params.get('center_points', 0)
        if n_center > 0:
            center_points = self._generate_center_points(factors, n_center)
            scaled_design = np.vstack([scaled_design, center_points])
        
        return scaled_design
    
    def _generate_ccd(self, factors: List[Factor], params: Dict) -> np.ndarray:
        """중심합성설계 생성"""
        n_factors = len(factors)
        alpha = params.get('alpha', 'rotatable')
        n_center = params.get('center_points', 3)
        
        # pyDOE2의 ccdesign 사용
        design = ccdesign(n_factors, center=(n_center, n_center), alpha=alpha, face='ccc')
        
        # 실제 값으로 변환
        scaled_design = self._scale_design(design, factors)
        
        return scaled_design
    
    def _generate_bbd(self, factors: List[Factor], params: Dict) -> np.ndarray:
        """Box-Behnken 설계 생성"""
        n_factors = len(factors)
        n_center = params.get('center_points', 3)
        
        # pyDOE2의 bbdesign 사용
        design = bbdesign(n_factors, center=n_center)
        
        # 실제 값으로 변환
        scaled_design = self._scale_design(design, factors)
        
        return scaled_design
    
    def _generate_latin_hypercube(self, factors: List[Factor], params: Dict) -> np.ndarray:
        """라틴 하이퍼큐브 샘플링"""
        n_samples = params.get('n_samples', 10)
        n_factors = len(factors)
        
        # pyDOE2의 lhs 사용
        design = lhs(n_factors, samples=n_samples)
        
        # 실제 값으로 변환
        scaled_design = np.zeros_like(design)
        for i, factor in enumerate(factors):
            if factor.type == FactorType.CONTINUOUS:
                min_val = factor.min_value
                max_val = factor.max_value
                scaled_design[:, i] = min_val + design[:, i] * (max_val - min_val)
            else:
                # 범주형은 균등하게 분배
                n_levels = len(factor.levels)
                level_indices = np.floor(design[:, i] * n_levels).astype(int)
                level_indices = np.clip(level_indices, 0, n_levels - 1)
                scaled_design[:, i] = level_indices
        
        return scaled_design
    
    def _generate_fractional_factorial(self, factors: List[Factor], params: Dict) -> np.ndarray:
        """부분요인설계 생성"""
        # 간단한 구현 (2수준만)
        n_factors = len(factors)
        
        # 해상도에 따른 생성기 선택
        if n_factors <= 4:
            design = ff2n(n_factors)
        else:
            # 2^(k-p) 설계
            from pyDOE2 import fracfact
            gen_string = self._get_fractional_generators(n_factors)
            design = fracfact(gen_string)
        
        # -1, 1을 실제 값으로 변환
        scaled_design = self._scale_design(design, factors)
        
        return scaled_design
    
    def _generate_plackett_burman(self, factors: List[Factor], params: Dict) -> np.ndarray:
        """Plackett-Burman 설계 생성"""
        n_factors = len(factors)
        
        # pyDOE2의 pbdesign 사용
        design = pbdesign(n_factors)
        
        # 실제 값으로 변환
        scaled_design = self._scale_design(design, factors)
        
        return scaled_design
    
    def _generate_d_optimal(self, factors: List[Factor], params: Dict) -> np.ndarray:
        """D-최적 설계 생성"""
        # 간단한 구현 - 후보점 생성 후 선택
        n_runs = params.get('n_runs', 20)
        
        # 후보점 생성 (그리드)
        candidates = self._generate_candidate_points(factors, resolution=10)
        
        # 욕심쟁이 알고리즘으로 D-최적 점 선택
        selected_indices = self._select_d_optimal_points(candidates, n_runs)
        
        return candidates[selected_indices]
    
    def _generate_custom(self, factors: List[Factor], params: Dict) -> np.ndarray:
        """사용자 정의 설계"""
        # 사용자가 직접 입력한 설계 매트릭스 반환
        custom_matrix = params.get('custom_matrix', np.array([]))
        return custom_matrix
    
    def _scale_design(self, coded_design: np.ndarray, factors: List[Factor]) -> np.ndarray:
        """코딩된 설계를 실제 값으로 변환"""
        scaled = np.zeros_like(coded_design)
        
        for i, factor in enumerate(factors):
            if factor.type == FactorType.CONTINUOUS:
                # -1 to 1 범위를 실제 범위로 변환
                min_val = factor.min_value
                max_val = factor.max_value
                center = (max_val + min_val) / 2
                half_range = (max_val - min_val) / 2
                scaled[:, i] = center + coded_design[:, i] * half_range
            else:
                # 범주형은 인덱스로 변환
                unique_vals = np.unique(coded_design[:, i])
                n_levels = len(factor.levels)
                for j, val in enumerate(unique_vals):
                    if j < n_levels:
                        scaled[coded_design[:, i] == val, i] = j
        
        return scaled
    
    def _generate_center_points(self, factors: List[Factor], n_points: int) -> np.ndarray:
        """중심점 생성"""
        center = []
        
        for factor in factors:
            if factor.type == FactorType.CONTINUOUS:
                center.append((factor.max_value + factor.min_value) / 2)
            else:
                center.append(0)  # 범주형은 첫 번째 수준
        
        return np.tile(center, (n_points, 1))
    
    def _get_fractional_generators(self, n_factors: int) -> str:
        """부분요인설계 생성기 문자열 반환"""
        # 일반적인 생성기
        generators = {
            5: "a b c d e",
            6: "a b c d e f",
            7: "a b c d e f g",
            8: "a b c d e f g h"
        }
        
        return generators.get(n_factors, "a b c d")
    
    def _generate_candidate_points(self, factors: List[Factor], resolution: int) -> np.ndarray:
        """D-최적을 위한 후보점 생성"""
        grids = []
        
        for factor in factors:
            if factor.type == FactorType.CONTINUOUS:
                grid = np.linspace(factor.min_value, factor.max_value, resolution)
            else:
                grid = np.arange(len(factor.levels))
            grids.append(grid)
        
        # 모든 조합 생성
        mesh = np.meshgrid(*grids)
        candidates = np.column_stack([m.ravel() for m in mesh])
        
        return candidates
    
    def _select_d_optimal_points(self, candidates: np.ndarray, n_select: int) -> np.ndarray:
        """D-최적 점 선택 알고리즘"""
        n_candidates = len(candidates)
        n_factors = candidates.shape[1]
        
        # 초기 선택 (랜덤)
        selected = np.random.choice(n_candidates, n_select, replace=False)
        
        # 간단한 교환 알고리즘
        for _ in range(100):  # 최대 반복
            improved = False
            
            for i in range(n_select):
                current_design = candidates[selected]
                current_det = self._calculate_determinant(current_design)
                
                # 교환 시도
                for j in range(n_candidates):
                    if j not in selected:
                        # i번째를 j로 교환
                        new_selected = selected.copy()
                        new_selected[i] = j
                        new_design = candidates[new_selected]
                        new_det = self._calculate_determinant(new_design)
                        
                        if new_det > current_det:
                            selected = new_selected
                            improved = True
                            break
                
                if improved:
                    break
            
            if not improved:
                break
        
        return selected
    
    def _calculate_determinant(self, design: np.ndarray) -> float:
        """정보 행렬의 행렬식 계산"""
        try:
            X = np.column_stack([np.ones(len(design)), design])
            XtX = X.T @ X
            return np.linalg.det(XtX)
        except:
            return 0.0
    
    def _calculate_balance(self, design: np.ndarray) -> float:
        """설계 균형성 계산"""
        balance_scores = []
        
        for col in design.T:
            unique, counts = np.unique(col, return_counts=True)
            if len(unique) > 1:
                balance = 1 - np.std(counts) / np.mean(counts)
                balance_scores.append(balance)
        
        return np.mean(balance_scores) if balance_scores else 0.0
    
    def _calculate_orthogonality(self, design: np.ndarray) -> float:
        """설계 직교성 계산"""
        n_factors = design.shape[1]
        
        if n_factors < 2:
            return 1.0
        
        # 상관계수 행렬
        corr_matrix = np.corrcoef(design.T)
        
        # 대각선 제외 평균 절대 상관계수
        off_diagonal = np.abs(corr_matrix[np.triu_indices(n_factors, k=1)])
        
        # 직교성 점수 (0에 가까울수록 직교)
        orthogonality = 1 - np.mean(off_diagonal)
        
        return orthogonality


# ==================== 검증 시스템 ====================

class ValidationSystem:
    """실험 설계 검증 시스템"""
    
    def validate_design(self, design: ExperimentDesign) -> ValidationResult:
        """종합적인 설계 검증"""
        result = ValidationResult(passed=True)
        
        # 통계적 검증
        stat_result = self._validate_statistical(design)
        if not stat_result.passed:
            result.passed = False
            result.errors.extend(stat_result.errors)
        result.warnings.extend(stat_result.warnings)
        
        # 실용적 검증
        prac_result = self._validate_practical(design)
        if not prac_result.passed:
            result.passed = False
            result.errors.extend(prac_result.errors)
        result.warnings.extend(prac_result.warnings)
        
        # 품질 평가
        result.quality_metrics = self._assess_quality(design)
        
        return result
    
    def _validate_statistical(self, design: ExperimentDesign) -> ValidationResult:
        """통계적 타당성 검증"""
        result = ValidationResult(passed=True)
        
        n_runs = len(design.design_matrix)
        n_factors = len(design.factors)
        n_responses = len(design.responses)
        
        # 자유도 검사
        min_runs = n_factors + 1
        if n_runs < min_runs:
            result.passed = False
            result.errors.append(f"실험 횟수({n_runs})가 최소 요구사항({min_runs})보다 적습니다.")
        
        # 검정력 검사 (간단한 추정)
        if n_runs < 2 * n_factors:
            result.warnings.append("주효과 검정력이 낮을 수 있습니다.")
        
        if n_runs < n_factors * (n_factors + 1) / 2:
            result.warnings.append("교호작용 검출이 어려울 수 있습니다.")
        
        return result
    
    def _validate_practical(self, design: ExperimentDesign) -> ValidationResult:
        """실용적 타당성 검증"""
        result = ValidationResult(passed=True)
        
        # 실험 횟수 체크
        n_runs = len(design.design_matrix)
        if n_runs > 100:
            result.warnings.append(f"실험 횟수가 많습니다 ({n_runs}회). 단계적 접근을 고려하세요.")
        
        # 극단값 체크
        for i, factor in enumerate(design.factors):
            if factor.type == FactorType.CONTINUOUS:
                values = design.design_matrix[:, i]
                if np.any(values == factor.min_value) or np.any(values == factor.max_value):
                    result.warnings.append(f"{factor.name}의 극단값이 포함되어 있습니다.")
        
        return result
    
    def _assess_quality(self, design: ExperimentDesign) -> Dict[str, float]:
        """설계 품질 평가"""
        metrics = {}
        
        # 기본 메트릭은 이미 계산됨
        if 'quality_metrics' in design.metadata:
            metrics.update(design.metadata['quality_metrics'])
        
        # 추가 메트릭
        metrics['completeness'] = 1.0  # 모든 필수 정보 포함 여부
        
        return metrics
