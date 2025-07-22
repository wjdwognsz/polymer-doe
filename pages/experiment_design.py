"""
pages/experiment_design.py - 실험 설계 페이지
Universal DOE Platform의 핵심 실험 설계 및 관리 페이지
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path
import io
import base64

# 실험설계 라이브러리
try:
    from pyDOE3 import *
except ImportError:
    st.error("pyDOE3가 설치되지 않았습니다. 'pip install pyDOE3'를 실행해주세요.")

# 내부 모듈 임포트
from utils.common_ui import get_common_ui
from utils.database_manager import get_database_manager
from utils.auth_manager import get_auth_manager
from utils.api_manager import get_api_manager
from utils.notification_manager import get_notification_manager
from utils.data_processor import get_data_processor
from modules.module_registry import get_module_registry
from modules.base_module import ExperimentDesign, Factor, Response
from config.app_config import EXPERIMENT_DEFAULTS, API_CONFIG

# 로깅 설정
logger = logging.getLogger(__name__)

# 실험 설계 타입 정의
DESIGN_TYPES = {
    "full_factorial": {
        "name": "완전요인설계",
        "description": "모든 요인 수준 조합을 실험",
        "min_factors": 2,
        "max_factors": 7,
        "supports_center_points": True
    },
    "fractional_factorial": {
        "name": "부분요인설계",
        "description": "일부 조합만 선택하여 실험 횟수 감소",
        "min_factors": 3,
        "max_factors": 15,
        "supports_center_points": True
    },
    "central_composite": {
        "name": "중심합성설계 (CCD)",
        "description": "2차 모델 적합을 위한 RSM 설계",
        "min_factors": 2,
        "max_factors": 10,
        "supports_center_points": True
    },
    "box_behnken": {
        "name": "Box-Behnken 설계",
        "description": "3수준 RSM 설계, 극단값 제외",
        "min_factors": 3,
        "max_factors": 7,
        "supports_center_points": True
    },
    "plackett_burman": {
        "name": "Plackett-Burman 설계",
        "description": "스크리닝용 2수준 설계",
        "min_factors": 2,
        "max_factors": 47,
        "supports_center_points": False
    },
    "latin_hypercube": {
        "name": "Latin Hypercube 설계",
        "description": "공간충진 설계, 컴퓨터 실험용",
        "min_factors": 1,
        "max_factors": 20,
        "supports_center_points": False
    },
    "custom": {
        "name": "사용자 정의",
        "description": "직접 실험 조건 입력",
        "min_factors": 1,
        "max_factors": 50,
        "supports_center_points": False
    }
}

class ExperimentDesignManager:
    """실험 설계 관리 클래스"""
    
    def __init__(self):
        self.ui = get_common_ui()
        self.db = get_database_manager()
        self.auth = get_auth_manager()
        self.api = get_api_manager()
        self.notifier = get_notification_manager()
        self.processor = get_data_processor()
        self.module_registry = get_module_registry()
        self.current_user = self.auth.get_current_user()
        
        # 현재 프로젝트 확인
        self.project_id = st.session_state.get('current_project')
        if self.project_id:
            self.project = self.db.get_project(self.project_id)
            self.module = self._load_project_module()
        else:
            self.project = None
            self.module = None
    
    def _load_project_module(self):
        """프로젝트 모듈 로드"""
        if not self.project:
            return None
            
        module_id = self.project.get('module_id', 'core.general_experiment')
        return self.module_registry.get_module(module_id)
    
    def render_page(self):
        """실험 설계 페이지 메인"""
        # 인증 확인
        if not self.current_user:
            st.warning("로그인이 필요합니다.")
            st.stop()
        
        # 프로젝트 확인
        if not self.project_id:
            st.warning("먼저 프로젝트를 선택해주세요.")
            if st.button("프로젝트 선택하기"):
                st.session_state.current_page = 'project_setup'
                st.rerun()
            st.stop()
        
        # 페이지 헤더
        self.ui.render_header(
            f"실험 설계 - {self.project['name']}",
            f"{self.project.get('field', '일반')} | {self.project.get('description', '')}",
            "🧪"
        )
        
        # 실험 탭
        tabs = st.tabs([
            "새 실험 설계",
            "진행중인 실험",
            "완료된 실험",
            "실험 분석",
            "템플릿"
        ])
        
        with tabs[0]:
            self._render_new_experiment()
            
        with tabs[1]:
            self._render_ongoing_experiments()
            
        with tabs[2]:
            self._render_completed_experiments()
            
        with tabs[3]:
            self._render_analysis()
            
        with tabs[4]:
            self._render_templates()
    
    def _render_new_experiment(self):
        """새 실험 설계 탭"""
        # 실험 설계 방법 선택
        col1, col2 = st.columns([3, 1])
        
        with col1:
            design_method = st.selectbox(
                "실험 설계 방법",
                ["모듈 기반 설계", "표준 실험설계법", "AI 추천 설계", "수동 입력"],
                help="프로젝트에 맞는 실험 설계 방법을 선택하세요"
            )
        
        with col2:
            # AI 설명 상세도 설정 (필수 구현)
            ai_detail = st.select_slider(
                "AI 설명",
                options=["간단", "보통", "상세", "매우상세"],
                value=st.session_state.get('ai_detail_level', '보통'),
                key="experiment_ai_detail"
            )
            st.session_state.ai_detail_level = ai_detail
        
        st.divider()
        
        if design_method == "모듈 기반 설계":
            self._render_module_based_design()
        elif design_method == "표준 실험설계법":
            self._render_standard_design()
        elif design_method == "AI 추천 설계":
            self._render_ai_guided_design()
        else:
            self._render_manual_design()
    
    def _render_module_based_design(self):
        """모듈 기반 실험 설계"""
        if not self.module:
            st.error("이 프로젝트에 실험 모듈이 설정되지 않았습니다.")
            return
        
        st.subheader(f"🔬 {self.module.metadata['name']} 모듈")
        st.caption(self.module.metadata.get('description', ''))
        
        # 실험 유형 선택
        experiment_types = self.module.get_experiment_types()
        if not experiment_types:
            st.warning("이 모듈에서 사용 가능한 실험 유형이 없습니다.")
            return
        
        selected_type = st.selectbox(
            "실험 유형",
            experiment_types,
            help="모듈에서 제공하는 실험 유형을 선택하세요"
        )
        
        # 모듈별 요인 및 반응변수
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 실험 요인")
            module_factors = self.module.get_factors(selected_type)
            
            if module_factors:
                # 기본 요인 표시
                for i, factor in enumerate(module_factors):
                    with st.expander(f"{factor.name} ({factor.unit or 'N/A'})", expanded=i==0):
                        if factor.type == 'continuous':
                            col_min, col_max = st.columns(2)
                            with col_min:
                                min_val = st.number_input(
                                    "최소값",
                                    value=float(factor.min_value or 0),
                                    key=f"factor_min_{i}"
                                )
                            with col_max:
                                max_val = st.number_input(
                                    "최대값",
                                    value=float(factor.max_value or 100),
                                    key=f"factor_max_{i}"
                                )
                            
                            # 수준 수 선택
                            levels = st.slider(
                                "수준 수",
                                2, 5, 3,
                                key=f"factor_levels_{i}"
                            )
                        else:  # categorical
                            levels = st.multiselect(
                                "수준 선택",
                                factor.levels or [],
                                default=factor.levels[:2] if factor.levels else [],
                                key=f"factor_cat_{i}"
                            )
            else:
                st.info("이 실험 유형에 대한 기본 요인이 없습니다.")
            
            # 추가 요인
            if st.checkbox("사용자 정의 요인 추가"):
                self._render_custom_factor_input()
        
        with col2:
            st.markdown("### 📈 반응변수")
            module_responses = self.module.get_responses(selected_type)
            
            if module_responses:
                selected_responses = []
                for i, response in enumerate(module_responses):
                    if st.checkbox(
                        f"{response.name} ({response.unit or 'N/A'})",
                        value=True,
                        key=f"response_{i}"
                    ):
                        selected_responses.append(response)
                        
                        # 목표 설정
                        goal = st.radio(
                            f"{response.name} 목표",
                            ["최대화", "최소화", "목표값"],
                            horizontal=True,
                            key=f"response_goal_{i}"
                        )
                        
                        if goal == "목표값":
                            target = st.number_input(
                                "목표값",
                                value=response.target_value or 0,
                                key=f"response_target_{i}"
                            )
            else:
                st.info("반응변수를 직접 정의해주세요.")
            
            # 추가 반응변수
            if st.checkbox("사용자 정의 반응변수 추가"):
                self._render_custom_response_input()
        
        # 실험 설계 생성
        st.divider()
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            experiment_name = st.text_input(
                "실험명",
                placeholder="예: 최적 조건 탐색 실험"
            )
        
        with col2:
            design_type = st.selectbox(
                "설계법",
                list(DESIGN_TYPES.keys()),
                format_func=lambda x: DESIGN_TYPES[x]['name']
            )
        
        with col3:
            if st.button("설계 생성", type="primary", use_container_width=True):
                if not experiment_name:
                    st.error("실험명을 입력해주세요.")
                else:
                    self._generate_module_design(
                        experiment_name,
                        design_type,
                        selected_type
                    )
    
    def _render_standard_design(self):
        """표준 실험설계법"""
        st.subheader("📐 표준 실험설계법")
        
        # 설계법 선택
        design_type = st.selectbox(
            "실험설계법 선택",
            list(DESIGN_TYPES.keys()),
            format_func=lambda x: DESIGN_TYPES[x]['name'],
            help="프로젝트 목적에 맞는 실험설계법을 선택하세요"
        )
        
        # 설계법 설명
        design_info = DESIGN_TYPES[design_type]
        st.info(f"**{design_info['name']}**: {design_info['description']}")
        
        # 요인 설정
        st.markdown("### 실험 요인 설정")
        
        n_factors = st.number_input(
            "요인 개수",
            min_value=design_info['min_factors'],
            max_value=design_info['max_factors'],
            value=min(3, design_info['max_factors']),
            help=f"이 설계법은 {design_info['min_factors']}~{design_info['max_factors']}개 요인을 지원합니다"
        )
        
        # 요인 정보 입력
        factors = []
        
        for i in range(n_factors):
            with st.expander(f"요인 {i+1}", expanded=i==0):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    name = st.text_input(
                        "요인명",
                        key=f"std_factor_name_{i}",
                        placeholder="예: 온도, 압력, 시간"
                    )
                
                with col2:
                    unit = st.text_input(
                        "단위",
                        key=f"std_factor_unit_{i}",
                        placeholder="예: °C, bar, min"
                    )
                
                with col3:
                    factor_type = st.selectbox(
                        "타입",
                        ["연속형", "범주형"],
                        key=f"std_factor_type_{i}"
                    )
                
                if factor_type == "연속형":
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        min_val = st.number_input(
                            "최소값",
                            key=f"std_factor_min_{i}",
                            value=0.0
                        )
                    
                    with col2:
                        max_val = st.number_input(
                            "최대값",
                            key=f"std_factor_max_{i}",
                            value=100.0
                        )
                    
                    with col3:
                        if design_type in ["full_factorial", "fractional_factorial"]:
                            levels = st.number_input(
                                "수준 수",
                                min_value=2,
                                max_value=5,
                                value=2,
                                key=f"std_factor_levels_{i}"
                            )
                        else:
                            levels = None
                    
                    factors.append({
                        'name': name,
                        'type': 'continuous',
                        'unit': unit,
                        'min_value': min_val,
                        'max_value': max_val,
                        'levels': levels
                    })
                else:
                    # 범주형 요인
                    categories = st.text_area(
                        "범주 (한 줄에 하나씩)",
                        key=f"std_factor_cat_{i}",
                        placeholder="A\nB\nC"
                    )
                    
                    if categories:
                        cat_list = [c.strip() for c in categories.split('\n') if c.strip()]
                        factors.append({
                            'name': name,
                            'type': 'categorical',
                            'unit': unit,
                            'levels': cat_list
                        })
        
        # 반응변수 설정
        st.markdown("### 반응변수 설정")
        
        n_responses = st.number_input(
            "반응변수 개수",
            min_value=1,
            max_value=10,
            value=1
        )
        
        responses = []
        for i in range(n_responses):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                resp_name = st.text_input(
                    "반응변수명",
                    key=f"std_response_name_{i}",
                    placeholder="예: 수율, 순도, 강도"
                )
            
            with col2:
                resp_unit = st.text_input(
                    "단위",
                    key=f"std_response_unit_{i}",
                    placeholder="예: %, MPa"
                )
            
            with col3:
                goal = st.selectbox(
                    "목표",
                    ["최대화", "최소화", "목표값"],
                    key=f"std_response_goal_{i}"
                )
            
            if resp_name:
                responses.append({
                    'name': resp_name,
                    'unit': resp_unit,
                    'goal': goal.lower().replace('화', 'ize')
                })
        
        # 고급 옵션
        with st.expander("고급 옵션"):
            col1, col2 = st.columns(2)
            
            with col1:
                if design_info['supports_center_points']:
                    center_points = st.number_input(
                        "중심점 개수",
                        min_value=0,
                        max_value=10,
                        value=3,
                        help="재현성 확인을 위한 중심점 반복"
                    )
                else:
                    center_points = 0
                
                randomize = st.checkbox(
                    "실험 순서 무작위화",
                    value=True,
                    help="순서 효과를 제거하기 위해 권장"
                )
            
            with col2:
                if design_type == "fractional_factorial":
                    resolution = st.selectbox(
                        "해상도",
                        ["III", "IV", "V"],
                        index=1,
                        help="높은 해상도는 더 많은 실험이 필요합니다"
                    )
                else:
                    resolution = None
                
                blocks = st.number_input(
                    "블록 수",
                    min_value=1,
                    max_value=4,
                    value=1,
                    help="실험을 여러 그룹으로 나누어 진행"
                )
        
        # 설계 생성 버튼
        st.divider()
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            experiment_name = st.text_input(
                "실험명",
                placeholder="실험 이름을 입력하세요"
            )
        
        with col2:
            if st.button("실험 설계 생성", type="primary", use_container_width=True):
                if not experiment_name:
                    st.error("실험명을 입력해주세요")
                elif not all(f.get('name') for f in factors):
                    st.error("모든 요인의 이름을 입력해주세요")
                elif not responses:
                    st.error("최소 하나의 반응변수를 설정해주세요")
                else:
                    design_params = {
                        'design_type': design_type,
                        'factors': factors,
                        'responses': responses,
                        'center_points': center_points,
                        'randomize': randomize,
                        'resolution': resolution,
                        'blocks': blocks
                    }
                    
                    self._generate_standard_design(experiment_name, design_params)
    
    def _render_ai_guided_design(self):
        """AI 가이드 실험 설계"""
        st.subheader("🤖 AI 추천 실험 설계")
        
        # AI 설명 토글
        show_ai_details = st.checkbox(
            "🔍 AI 추론 과정 보기",
            value=st.session_state.get('show_ai_details', True)
        )
        st.session_state.show_ai_details = show_ai_details
        
        # 실험 목표 입력
        st.markdown("### 실험 목표 설명")
        
        experiment_goal = st.text_area(
            "무엇을 달성하고 싶으신가요?",
            placeholder="""예시:
- PET 필름의 투명도를 90% 이상으로 향상시키고 싶습니다
- 신약 후보물질의 최적 합성 조건을 찾고 있습니다
- 배터리 전극 재료의 전기전도도를 최대화하려고 합니다""",
            height=150
        )
        
        # 제약사항 입력
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 실험 제약사항")
            
            max_runs = st.number_input(
                "최대 실험 횟수",
                min_value=5,
                max_value=1000,
                value=20,
                help="예산과 시간을 고려한 최대 실험 횟수"
            )
            
            time_constraint = st.selectbox(
                "시간 제약",
                ["제약 없음", "1주일 이내", "2주일 이내", "1개월 이내"],
                help="실험 완료 목표 기간"
            )
        
        with col2:
            st.markdown("### 보유 장비/재료")
            
            equipment = st.text_area(
                "사용 가능한 장비",
                placeholder="예: HPLC, UV-Vis, 반응기(100mL)",
                height=80
            )
            
            materials = st.text_area(
                "보유 재료/시약",
                placeholder="예: 에탄올, 촉매 A, B, C",
                height=80
            )
        
        # 이전 실험 정보
        st.markdown("### 관련 경험")
        
        previous_exp = st.text_area(
            "이전에 수행한 유사 실험이나 알고 있는 정보가 있다면 설명해주세요",
            placeholder="예: 온도가 80도 이상에서는 부산물이 생성됨",
            height=100
        )
        
        # AI 분석 요청
        if st.button("🤖 AI 실험 설계 추천받기", type="primary", use_container_width=True):
            if not experiment_goal:
                st.error("실험 목표를 입력해주세요")
            else:
                with st.spinner("AI가 최적의 실험 설계를 분석하고 있습니다..."):
                    recommendations = self._get_ai_design_recommendations({
                        'goal': experiment_goal,
                        'max_runs': max_runs,
                        'time_constraint': time_constraint,
                        'equipment': equipment,
                        'materials': materials,
                        'previous_exp': previous_exp
                    })
                    
                    if recommendations:
                        self._render_ai_recommendations(recommendations, show_ai_details)
    
    def _get_ai_design_recommendations(self, inputs: Dict) -> Dict:
        """AI 기반 실험 설계 추천"""
        # 프롬프트 구성
        prompt = f"""
        사용자가 다음과 같은 실험을 계획하고 있습니다:
        
        목표: {inputs['goal']}
        최대 실험 횟수: {inputs['max_runs']}
        시간 제약: {inputs['time_constraint']}
        사용 가능 장비: {inputs['equipment']}
        보유 재료: {inputs['materials']}
        이전 경험: {inputs['previous_exp']}
        
        다음을 추천해주세요:
        1. 적합한 실험설계법과 그 이유
        2. 주요 실험 요인과 권장 범위
        3. 측정해야 할 반응변수
        4. 예상되는 실험 횟수와 설계
        5. 주의사항과 성공 팁
        
        JSON 형식으로 구조화하여 응답하세요.
        """
        
        # AI 호출 (상세도 레벨 포함)
        detail_level = st.session_state.get('ai_detail_level', '보통')
        response = self.api.generate_structured_response(
            prompt,
            detail_level=detail_level,
            include_reasoning=True,
            include_alternatives=True,
            include_confidence=True
        )
        
        if not response:
            # 오프라인 폴백
            return self._get_offline_recommendations(inputs)
        
        return response
    
    def _render_ai_recommendations(self, recommendations: Dict, show_details: bool):
        """AI 추천 결과 렌더링"""
        st.success("✅ AI 분석이 완료되었습니다!")
        
        # 기본 추천 사항
        st.markdown("### 🎯 추천 실험 설계")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "추천 설계법",
                recommendations.get('design_method', '중심합성설계'),
                recommendations.get('design_method_reason', '')[:50] + "..."
            )
            
            st.metric(
                "예상 실험 횟수",
                f"{recommendations.get('estimated_runs', 20)}회",
                f"목표 대비 {recommendations.get('efficiency', '85')}% 효율"
            )
        
        with col2:
            st.metric(
                "예상 소요 시간",
                recommendations.get('estimated_duration', '2주'),
                recommendations.get('time_saving', '')
            )
            
            st.metric(
                "성공 확률",
                f"{recommendations.get('success_probability', 75)}%",
                "AI 예측 신뢰도"
            )
        
        # 상세 설명 (토글)
        if show_details:
            tabs = st.tabs([
                "추론 과정",
                "실험 요인",
                "대안 검토",
                "위험 분석",
                "참고 사례"
            ])
            
            with tabs[0]:
                st.markdown("#### 🧠 AI 추론 과정")
                reasoning = recommendations.get('reasoning', {})
                
                for step, explanation in reasoning.items():
                    with st.expander(f"단계 {step}", expanded=True):
                        st.write(explanation)
            
            with tabs[1]:
                st.markdown("#### 📊 추천 실험 요인")
                factors = recommendations.get('factors', [])
                
                for i, factor in enumerate(factors):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.write(f"**{factor['name']}**")
                        st.caption(factor.get('importance', ''))
                    
                    with col2:
                        st.write(f"범위: {factor['min']} - {factor['max']} {factor['unit']}")
                    
                    with col3:
                        levels = st.number_input(
                            "수준 수",
                            min_value=2,
                            max_value=5,
                            value=factor.get('recommended_levels', 3),
                            key=f"ai_factor_levels_{i}"
                        )
            
            with tabs[2]:
                st.markdown("#### 🔄 대안 검토")
                alternatives = recommendations.get('alternatives', [])
                
                for alt in alternatives:
                    with st.expander(f"{alt['method']} (적합도: {alt['score']}/10)"):
                        st.write(f"**장점**: {alt['pros']}")
                        st.write(f"**단점**: {alt['cons']}")
                        st.write(f"**적용 조건**: {alt['when_to_use']}")
            
            with tabs[3]:
                st.markdown("#### ⚠️ 위험 분석")
                risks = recommendations.get('risks', [])
                
                for risk in risks:
                    severity = risk.get('severity', 'medium')
                    color = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}.get(severity, '⚪')
                    
                    st.warning(f"{color} **{risk['title']}**\n\n{risk['description']}\n\n"
                             f"**대응방안**: {risk['mitigation']}")
            
            with tabs[4]:
                st.markdown("#### 📚 유사 연구 사례")
                references = recommendations.get('similar_studies', [])
                
                for ref in references:
                    with st.expander(f"{ref['title']} (유사도: {ref['similarity']}%)"):
                        st.write(f"**분야**: {ref['field']}")
                        st.write(f"**주요 발견**: {ref['key_findings']}")
                        st.write(f"**배울 점**: {ref['lessons']}")
        
        # 실험 설계 적용 버튼
        st.divider()
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            exp_name = st.text_input(
                "실험명",
                value=recommendations.get('suggested_name', 'AI 추천 실험')
            )
        
        with col2:
            if st.button("이 설계 적용하기", type="primary", use_container_width=True):
                self._apply_ai_recommendations(exp_name, recommendations)
    
    def _generate_standard_design(self, experiment_name: str, params: Dict):
        """표준 실험 설계 생성"""
        try:
            design_type = params['design_type']
            factors = params['factors']
            
            # 설계 매트릭스 생성
            if design_type == 'full_factorial':
                design_matrix = self._generate_full_factorial(factors)
            elif design_type == 'fractional_factorial':
                design_matrix = self._generate_fractional_factorial(factors, params.get('resolution'))
            elif design_type == 'central_composite':
                design_matrix = self._generate_ccd(factors, params.get('center_points', 3))
            elif design_type == 'box_behnken':
                design_matrix = self._generate_box_behnken(factors, params.get('center_points', 3))
            elif design_type == 'plackett_burman':
                design_matrix = self._generate_plackett_burman(factors)
            elif design_type == 'latin_hypercube':
                design_matrix = self._generate_lhs(factors, params.get('n_samples'))
            else:
                st.error(f"지원하지 않는 설계 타입: {design_type}")
                return
            
            # 실험 순서 무작위화
            if params.get('randomize', True):
                np.random.shuffle(design_matrix)
            
            # 실험 데이터 생성
            experiment_data = {
                'id': f"exp_{uuid.uuid4().hex[:8]}",
                'project_id': self.project_id,
                'name': experiment_name,
                'design_type': design_type,
                'factors': factors,
                'responses': params['responses'],
                'design_matrix': design_matrix.tolist(),
                'status': 'planning',
                'created_by': self.current_user['id'],
                'created_at': datetime.now(),
                'settings': {
                    'center_points': params.get('center_points', 0),
                    'randomized': params.get('randomize', True),
                    'blocks': params.get('blocks', 1),
                    'resolution': params.get('resolution')
                }
            }
            
            # 데이터베이스에 저장
            self.db.create_experiment(experiment_data)
            
            # 성공 메시지
            st.success(f"✅ '{experiment_name}' 실험 설계가 생성되었습니다!")
            st.info(f"총 {len(design_matrix)}회의 실험이 계획되었습니다.")
            
            # 설계 표시
            self._display_design_results(experiment_data)
            
        except Exception as e:
            logger.error(f"실험 설계 생성 실패: {e}")
            st.error(f"실험 설계 생성 중 오류가 발생했습니다: {str(e)}")
    
    def _generate_full_factorial(self, factors: List[Dict]) -> np.ndarray:
        """완전요인설계 생성"""
        continuous_factors = [f for f in factors if f['type'] == 'continuous']
        
        if not continuous_factors:
            st.error("연속형 요인이 필요합니다")
            return np.array([])
        
        # 각 요인의 수준 생성
        levels_list = []
        for factor in continuous_factors:
            n_levels = factor.get('levels', 2)
            if n_levels == 2:
                levels = [factor['min_value'], factor['max_value']]
            else:
                levels = np.linspace(
                    factor['min_value'],
                    factor['max_value'],
                    n_levels
                ).tolist()
            levels_list.append(levels)
        
        # 모든 조합 생성
        import itertools
        combinations = list(itertools.product(*levels_list))
        
        return np.array(combinations)
    
    def _generate_ccd(self, factors: List[Dict], center_points: int = 3) -> np.ndarray:
        """중심합성설계 생성"""
        continuous_factors = [f for f in factors if f['type'] == 'continuous']
        n_factors = len(continuous_factors)
        
        if n_factors < 2:
            st.error("CCD는 최소 2개의 연속형 요인이 필요합니다")
            return np.array([])
        
        # pyDOE3의 ccdesign 사용
        design = ccdesign(n_factors, center=(center_points, center_points), 
                         alpha='orthogonal', face='ccf')
        
        # 코드화된 값을 실제 값으로 변환
        scaled_design = np.zeros_like(design)
        for i, factor in enumerate(continuous_factors):
            min_val = factor['min_value']
            max_val = factor['max_value']
            center = (min_val + max_val) / 2
            half_range = (max_val - min_val) / 2
            
            scaled_design[:, i] = center + design[:, i] * half_range
        
        return scaled_design
    
    def _generate_box_behnken(self, factors: List[Dict], center_points: int = 3) -> np.ndarray:
        """Box-Behnken 설계 생성"""
        continuous_factors = [f for f in factors if f['type'] == 'continuous']
        n_factors = len(continuous_factors)
        
        if n_factors < 3:
            st.error("Box-Behnken 설계는 최소 3개의 요인이 필요합니다")
            return np.array([])
        
        # pyDOE3의 bbdesign 사용
        design = bbdesign(n_factors, center=center_points)
        
        # 스케일링
        scaled_design = np.zeros_like(design)
        for i, factor in enumerate(continuous_factors):
            min_val = factor['min_value']
            max_val = factor['max_value']
            center = (min_val + max_val) / 2
            half_range = (max_val - min_val) / 2
            
            scaled_design[:, i] = center + design[:, i] * half_range
        
        return scaled_design
    
    def _generate_lhs(self, factors: List[Dict], n_samples: Optional[int] = None) -> np.ndarray:
        """Latin Hypercube 설계 생성"""
        continuous_factors = [f for f in factors if f['type'] == 'continuous']
        n_factors = len(continuous_factors)
        
        if not n_samples:
            n_samples = max(10, n_factors * 5)  # 기본값
        
        # pyDOE3의 lhs 사용
        design = lhs(n_factors, samples=n_samples, criterion='maximin')
        
        # 0-1 범위를 실제 값으로 변환
        scaled_design = np.zeros_like(design)
        for i, factor in enumerate(continuous_factors):
            min_val = factor['min_value']
            max_val = factor['max_value']
            
            scaled_design[:, i] = min_val + design[:, i] * (max_val - min_val)
        
        return scaled_design
    
    def _display_design_results(self, experiment_data: Dict):
        """실험 설계 결과 표시"""
        st.markdown("### 📊 실험 설계 결과")
        
        # 설계 매트릭스를 DataFrame으로 변환
        factors = experiment_data['factors']
        design_matrix = np.array(experiment_data['design_matrix'])
        
        # 컬럼명 생성
        columns = []
        for factor in factors:
            if factor['type'] == 'continuous':
                columns.append(f"{factor['name']} ({factor.get('unit', '')})")
        
        # DataFrame 생성
        df = pd.DataFrame(design_matrix, columns=columns)
        df.index = np.arange(1, len(df) + 1)
        df.index.name = 'Run'
        
        # 반응변수 컬럼 추가
        for response in experiment_data['responses']:
            df[f"{response['name']} ({response.get('unit', '')})"] = ''
        
        # 설계 정보 표시
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("실험 횟수", len(df))
        with col2:
            st.metric("요인 수", len(factors))
        with col3:
            st.metric("반응변수 수", len(experiment_data['responses']))
        
        # 설계 매트릭스 표시
        st.markdown("#### 실험 계획표")
        
        # 편집 가능한 데이터프레임
        edited_df = st.data_editor(
            df,
            use_container_width=True,
            num_rows="fixed",
            disabled=columns,  # 요인 컬럼은 편집 불가
            key=f"design_matrix_{experiment_data['id']}"
        )
        
        # 시각화
        if len(factors) >= 2 and all(f['type'] == 'continuous' for f in factors[:2]):
            st.markdown("#### 설계 공간 시각화")
            
            fig = px.scatter(
                df,
                x=columns[0],
                y=columns[1],
                title="실험 설계 점",
                labels={'index': 'Run #'}
            )
            
            fig.update_traces(
                marker=dict(size=10, color='blue'),
                text=df.index,
                textposition="top center"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # 다운로드 옵션
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Excel 다운로드
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                edited_df.to_excel(writer, sheet_name='Design Matrix')
                
                # 실험 정보 시트
                info_df = pd.DataFrame({
                    'Property': ['Experiment Name', 'Design Type', 'Created By', 'Created At'],
                    'Value': [
                        experiment_data['name'],
                        DESIGN_TYPES[experiment_data['design_type']]['name'],
                        self.current_user['name'],
                        experiment_data['created_at'].strftime('%Y-%m-%d %H:%M')
                    ]
                })
                info_df.to_excel(writer, sheet_name='Info', index=False)
            
            excel_buffer.seek(0)
            
            st.download_button(
                label="📥 Excel 다운로드",
                data=excel_buffer,
                file_name=f"{experiment_data['name']}_design.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        with col2:
            # CSV 다운로드
            csv = edited_df.to_csv(index=True)
            st.download_button(
                label="📥 CSV 다운로드",
                data=csv,
                file_name=f"{experiment_data['name']}_design.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col3:
            # 실험 시작 버튼
            if st.button("🚀 실험 시작", type="primary", use_container_width=True):
                self.db.update_experiment(
                    experiment_data['id'],
                    {'status': 'running', 'started_at': datetime.now()}
                )
                st.success("실험이 시작되었습니다!")
                st.rerun()
    
    def _render_ongoing_experiments(self):
        """진행중인 실험 탭"""
        st.subheader("🔬 진행중인 실험")
        
        # 진행중인 실험 로드
        experiments = self.db.get_project_experiments(
            self.project_id,
            status='running'
        )
        
        if not experiments:
            self.ui.render_empty_state(
                "진행중인 실험이 없습니다",
                "🧪"
            )
        else:
            # 실험 목록
            for exp in experiments:
                with st.expander(f"📊 {exp['name']}", expanded=True):
                    # 실험 정보
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "진행률",
                            f"{exp.get('progress', 0)}%",
                            f"{exp.get('completed_runs', 0)}/{exp.get('total_runs', 0)}"
                        )
                    
                    with col2:
                        st.metric(
                            "설계법",
                            DESIGN_TYPES.get(exp['design_type'], {}).get('name', exp['design_type'])
                        )
                    
                    with col3:
                        started = datetime.fromisoformat(exp['started_at'])
                        duration = datetime.now() - started
                        st.metric(
                            "경과 시간",
                            f"{duration.days}일 {duration.seconds//3600}시간"
                        )
                    
                    with col4:
                        if st.button("상세보기", key=f"view_exp_{exp['id']}"):
                            self._show_experiment_details(exp['id'])
                    
                    # 빠른 결과 입력
                    st.markdown("#### 빠른 결과 입력")
                    
                    # 미완료 실험 찾기
                    pending_runs = self._get_pending_runs(exp)
                    
                    if pending_runs:
                        selected_run = st.selectbox(
                            "실험 번호",
                            options=pending_runs,
                            format_func=lambda x: f"Run {x['run_number']}: {x['conditions_summary']}",
                            key=f"run_select_{exp['id']}"
                        )
                        
                        # 결과 입력 폼
                        if selected_run:
                            self._render_result_input_form(exp, selected_run)
                    else:
                        st.success("모든 실험이 완료되었습니다!")
                        
                        if st.button("결과 분석하기", key=f"analyze_{exp['id']}"):
                            st.session_state.current_experiment = exp['id']
                            st.session_state.active_tab = 3
                            st.rerun()
    
    def _render_result_input_form(self, experiment: Dict, run: Dict):
        """결과 입력 폼"""
        responses = experiment['responses']
        
        with st.form(f"result_form_{experiment['id']}_{run['run_number']}"):
            # 실험 조건 표시
            st.info(f"**실험 조건**: {run['conditions_summary']}")
            
            # 반응변수 입력
            results = {}
            
            for i, response in enumerate(responses):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    value = st.number_input(
                        f"{response['name']} ({response.get('unit', '')})",
                        key=f"resp_val_{experiment['id']}_{run['run_number']}_{i}",
                        format="%.4f"
                    )
                    results[response['name']] = value
                
                with col2:
                    st.write("")  # 공간 맞추기
                    st.write(f"목표: {response.get('goal', 'N/A')}")
            
            # 실험 노트
            notes = st.text_area(
                "실험 노트 (선택사항)",
                placeholder="특이사항, 관찰 내용 등을 기록하세요",
                height=100
            )
            
            # 제출
            submitted = st.form_submit_button(
                "결과 저장",
                type="primary",
                use_container_width=True
            )
            
            if submitted:
                # 결과 저장
                success = self.db.save_experiment_result(
                    experiment['id'],
                    run['run_number'],
                    {
                        'results': results,
                        'notes': notes,
                        'completed_by': self.current_user['id'],
                        'completed_at': datetime.now()
                    }
                )
                
                if success:
                    st.success("결과가 저장되었습니다!")
                    
                    # 진행률 업데이트
                    self._update_experiment_progress(experiment['id'])
                    
                    # 알림 발송
                    self.notifier.send(
                        "실험 결과 입력",
                        f"{experiment['name']}의 Run {run['run_number']} 완료",
                        "info"
                    )
                    
                    st.rerun()
                else:
                    st.error("결과 저장에 실패했습니다")
    
    def _render_completed_experiments(self):
        """완료된 실험 탭"""
        st.subheader("✅ 완료된 실험")
        
        # 완료된 실험 로드
        experiments = self.db.get_project_experiments(
            self.project_id,
            status='completed'
        )
        
        if not experiments:
            self.ui.render_empty_state(
                "완료된 실험이 없습니다",
                "📊"
            )
        else:
            # 필터링 옵션
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                search = st.text_input(
                    "검색",
                    placeholder="실험명, 설계법...",
                    label_visibility="collapsed"
                )
            
            with col2:
                date_range = st.date_input(
                    "기간",
                    value=[],
                    label_visibility="collapsed"
                )
            
            with col3:
                if st.button("🔄", use_container_width=True):
                    st.rerun()
            
            # 실험 카드 그리드
            for i in range(0, len(experiments), 2):
                cols = st.columns(2)
                
                for j, col in enumerate(cols):
                    if i + j < len(experiments):
                        exp = experiments[i + j]
                        
                        with col:
                            with st.container():
                                st.markdown(
                                    f"""
                                    <div class="custom-card">
                                        <h4>{exp['name']}</h4>
                                        <p>{DESIGN_TYPES.get(exp['design_type'], {}).get('name', '')}</p>
                                        <p>완료: {exp['completed_at'][:10]}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    if st.button("결과", key=f"res_{exp['id']}", use_container_width=True):
                                        self._show_experiment_results(exp['id'])
                                
                                with col2:
                                    if st.button("분석", key=f"ana_{exp['id']}", use_container_width=True):
                                        st.session_state.current_experiment = exp['id']
                                        st.session_state.active_tab = 3
                                        st.rerun()
                                
                                with col3:
                                    if st.button("복제", key=f"clo_{exp['id']}", use_container_width=True):
                                        self._clone_experiment(exp['id'])
    
    def _render_analysis(self):
        """실험 분석 탭"""
        st.subheader("📊 실험 분석")
        
        # 분석할 실험 선택
        completed_experiments = self.db.get_project_experiments(
            self.project_id,
            status='completed'
        )
        
        if not completed_experiments:
            st.warning("분석할 완료된 실험이 없습니다.")
            return
        
        selected_exp_id = st.selectbox(
            "분석할 실험 선택",
            options=[exp['id'] for exp in completed_experiments],
            format_func=lambda x: next(e['name'] for e in completed_experiments if e['id'] == x),
            index=0 if not st.session_state.get('current_experiment') else None
        )
        
        if selected_exp_id:
            experiment = self.db.get_experiment(selected_exp_id)
            
            if experiment:
                # 분석 탭
                analysis_tabs = st.tabs([
                    "기초 통계",
                    "주효과 분석",
                    "상호작용 분석",
                    "최적화",
                    "AI 인사이트"
                ])
                
                with analysis_tabs[0]:
                    self._render_basic_statistics(experiment)
                
                with analysis_tabs[1]:
                    self._render_main_effects(experiment)
                
                with analysis_tabs[2]:
                    self._render_interaction_analysis(experiment)
                
                with analysis_tabs[3]:
                    self._render_optimization(experiment)
                
                with analysis_tabs[4]:
                    self._render_ai_insights(experiment)
    
    def _render_ai_insights(self, experiment: Dict):
        """AI 인사이트 분석"""
        st.markdown("### 🤖 AI 기반 실험 분석")
        
        # AI 상세도 토글
        show_details = st.checkbox(
            "🔍 상세 분석 보기",
            value=st.session_state.get('show_ai_details', True),
            key="analysis_ai_details"
        )
        
        # 분석 요청
        if st.button("AI 분석 시작", type="primary"):
            with st.spinner("AI가 실험 결과를 분석하고 있습니다..."):
                insights = self._get_ai_experiment_analysis(experiment)
                
                if insights:
                    # 기본 인사이트
                    st.markdown("#### 📌 핵심 발견사항")
                    
                    for finding in insights.get('key_findings', []):
                        st.info(f"• {finding}")
                    
                    # 상세 분석 (토글)
                    if show_details:
                        tabs = st.tabs([
                            "분석 과정",
                            "패턴 발견",
                            "최적 조건",
                            "추가 실험 제안",
                            "신뢰도 평가"
                        ])
                        
                        with tabs[0]:
                            st.markdown("**AI 분석 과정**")
                            for step in insights.get('analysis_steps', []):
                                with st.expander(step['title']):
                                    st.write(step['description'])
                                    if 'code' in step:
                                        st.code(step['code'], language='python')
                        
                        with tabs[1]:
                            st.markdown("**발견된 패턴**")
                            patterns = insights.get('patterns', [])
                            
                            for pattern in patterns:
                                st.write(f"**{pattern['type']}**: {pattern['description']}")
                                
                                if 'visualization' in pattern:
                                    st.plotly_chart(
                                        pattern['visualization'],
                                        use_container_width=True
                                    )
                        
                        with tabs[2]:
                            st.markdown("**최적 조건 예측**")
                            optimal = insights.get('optimal_conditions', {})
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**예측 최적 조건**")
                                for factor, value in optimal.get('factors', {}).items():
                                    st.metric(factor, f"{value['value']} {value['unit']}")
                            
                            with col2:
                                st.write("**예상 결과**")
                                for response, prediction in optimal.get('predictions', {}).items():
                                    st.metric(
                                        response,
                                        f"{prediction['value']} ± {prediction['uncertainty']}"
                                    )
                        
                        with tabs[3]:
                            st.markdown("**추가 실험 제안**")
                            suggestions = insights.get('next_experiments', [])
                            
                            for i, suggestion in enumerate(suggestions):
                                with st.expander(f"제안 {i+1}: {suggestion['title']}"):
                                    st.write(f"**목적**: {suggestion['purpose']}")
                                    st.write(f"**방법**: {suggestion['method']}")
                                    st.write(f"**예상 실험수**: {suggestion['n_runs']}")
                                    st.write(f"**기대 효과**: {suggestion['expected_benefit']}")
                        
                        with tabs[4]:
                            st.markdown("**분석 신뢰도**")
                            confidence = insights.get('confidence_assessment', {})
                            
                            # 신뢰도 메트릭
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    "전체 신뢰도",
                                    f"{confidence.get('overall', 0)}%",
                                    help="AI 분석의 전반적인 신뢰도"
                                )
                            
                            with col2:
                                st.metric(
                                    "데이터 품질",
                                    f"{confidence.get('data_quality', 0)}%",
                                    help="입력 데이터의 품질 평가"
                                )
                            
                            with col3:
                                st.metric(
                                    "모델 적합도",
                                    f"{confidence.get('model_fit', 0)}%",
                                    help="통계 모델의 적합도"
                                )
                            
                            # 제한사항
                            if limitations := confidence.get('limitations', []):
                                st.warning("**⚠️ 분석 제한사항**")
                                for limitation in limitations:
                                    st.write(f"• {limitation}")
    
    def _get_ai_experiment_analysis(self, experiment: Dict) -> Dict:
        """AI 기반 실험 분석"""
        # 실험 데이터 준비
        results_df = self._prepare_experiment_data(experiment)
        
        # 프롬프트 구성
        prompt = f"""
        다음 실험 결과를 분석해주세요:
        
        실험 정보:
        - 실험명: {experiment['name']}
        - 설계법: {DESIGN_TYPES.get(experiment['design_type'], {}).get('name', '')}
        - 요인: {[f['name'] for f in experiment['factors']]}
        - 반응변수: {[r['name'] for r in experiment['responses']]}
        
        실험 데이터:
        {results_df.to_string()}
        
        다음을 분석해주세요:
        1. 주요 발견사항 (3-5개)
        2. 요인별 영향도 분석
        3. 최적 조건 예측
        4. 추가 실험 제안
        5. 분석의 신뢰도 평가
        
        분석 과정과 근거를 포함하여 JSON 형식으로 응답하세요.
        """
        
        # AI 호출
        detail_level = st.session_state.get('ai_detail_level', '보통')
        response = self.api.generate_structured_response(
            prompt,
            detail_level=detail_level,
            include_reasoning=True,
            include_visualization_code=True
        )
        
        if not response:
            # 오프라인 폴백 - 기본 통계 분석
            return self._get_offline_analysis(experiment, results_df)
        
        return response
    
    def _get_offline_analysis(self, experiment: Dict, results_df: pd.DataFrame) -> Dict:
        """오프라인 기본 분석 (AI 없이)"""
        analysis = {
            'key_findings': [],
            'patterns': [],
            'optimal_conditions': {},
            'next_experiments': [],
            'confidence_assessment': {
                'overall': 70,
                'data_quality': 80,
                'model_fit': 60,
                'limitations': ['AI 연결 없이 기본 통계 분석만 수행']
            }
        }
        
        # 기본 통계 분석
        for response in experiment['responses']:
            resp_name = response['name']
            if resp_name in results_df.columns:
                mean_val = results_df[resp_name].mean()
                std_val = results_df[resp_name].std()
                
                analysis['key_findings'].append(
                    f"{resp_name}의 평균: {mean_val:.2f} ± {std_val:.2f}"
                )
        
        # 간단한 상관관계 분석
        numeric_cols = results_df.select_dtypes(include=[np.number]).columns
        corr_matrix = results_df[numeric_cols].corr()
        
        # 강한 상관관계 찾기
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    analysis['patterns'].append({
                        'type': '상관관계',
                        'description': f"{corr_matrix.columns[i]}와 {corr_matrix.columns[j]} 간 "
                                     f"{'강한 양' if corr_val > 0 else '강한 음'}의 상관관계 "
                                     f"(r={corr_val:.2f})"
                    })
        
        return analysis
    
    def _prepare_experiment_data(self, experiment: Dict) -> pd.DataFrame:
        """실험 데이터 준비"""
        # 실험 결과 로드
        results = self.db.get_experiment_results(experiment['id'])
        
        if not results:
            return pd.DataFrame()
        
        # DataFrame 구성
        design_matrix = np.array(experiment['design_matrix'])
        factors = experiment['factors']
        responses = experiment['responses']
        
        # 컬럼 생성
        columns = [f['name'] for f in factors if f['type'] == 'continuous']
        df = pd.DataFrame(design_matrix, columns=columns)
        
        # 결과 데이터 추가
        for i, result in enumerate(results):
            if result and 'results' in result:
                for resp_name, value in result['results'].items():
                    df.loc[i, resp_name] = value
        
        return df
    
    def _get_offline_recommendations(self, inputs: Dict) -> Dict:
        """오프라인 AI 추천 (폴백)"""
        # 기본 추천 로직
        goal = inputs['goal'].lower()
        
        # 키워드 기반 간단한 추천
        if any(word in goal for word in ['최적', '최대', '향상']):
            design_method = 'central_composite'
            method_reason = '최적화에 적합한 2차 모델'
        elif any(word in goal for word in ['스크리닝', '탐색', '초기']):
            design_method = 'plackett_burman'
            method_reason = '많은 요인의 빠른 스크리닝'
        else:
            design_method = 'full_factorial'
            method_reason = '기본적이고 해석이 쉬운 설계'
        
        # 기본 추천 구성
        return {
            'design_method': design_method,
            'design_method_reason': method_reason,
            'estimated_runs': min(inputs['max_runs'], 20),
            'estimated_duration': '2주',
            'success_probability': 70,
            'factors': [
                {
                    'name': '요인 1',
                    'min': 0,
                    'max': 100,
                    'unit': '',
                    'importance': '주요 요인으로 예상',
                    'recommended_levels': 3
                },
                {
                    'name': '요인 2',
                    'min': 20,
                    'max': 80,
                    'unit': '',
                    'importance': '보조 요인',
                    'recommended_levels': 2
                }
            ],
            'suggested_name': f"{inputs['goal'][:20]} 실험",
            'reasoning': {
                '1': f"목표 '{inputs['goal']}'를 분석했습니다",
                '2': f"최대 {inputs['max_runs']}회 실험 제약을 고려했습니다",
                '3': f"{design_method} 설계법이 적합하다고 판단했습니다"
            },
            'alternatives': [],
            'risks': [
                {
                    'title': '데이터 부족',
                    'description': 'AI 연결 없이 기본 추천만 제공',
                    'severity': 'medium',
                    'mitigation': '실험 진행하며 조정 필요'
                }
            ],
            'similar_studies': []
        }
    
    def _render_templates(self):
        """실험 템플릿 탭"""
        st.subheader("📋 실험 템플릿")
        
        # 템플릿 소스 선택
        template_source = st.radio(
            "템플릿 소스",
            ["내 템플릿", "팀 템플릿", "공개 템플릿"],
            horizontal=True
        )
        
        # 템플릿 로드
        if template_source == "내 템플릿":
            templates = self.db.get_user_experiment_templates(self.current_user['id'])
        elif template_source == "팀 템플릿":
            templates = self.db.get_project_experiment_templates(self.project_id)
        else:
            templates = self.db.get_public_experiment_templates()
        
        if not templates:
            self.ui.render_empty_state(
                f"{template_source}이 없습니다",
                "📄"
            )
            
            # 현재 실험을 템플릿으로 저장
            if st.button("현재 실험을 템플릿으로 저장"):
                self._save_current_as_template()
        else:
            # 템플릿 그리드
            for template in templates:
                with st.expander(f"📋 {template['name']}", expanded=False):
                    # 템플릿 정보
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**설계법**: {DESIGN_TYPES.get(template['design_type'], {}).get('name', '')}")
                        st.write(f"**요인 수**: {len(template.get('factors', []))}")
                        st.write(f"**실험 횟수**: {template.get('n_runs', 'N/A')}")
                        st.write(f"**생성일**: {template.get('created_at', 'N/A')}")
                        
                        if template.get('description'):
                            st.write(f"**설명**: {template['description']}")
                    
                    with col2:
                        if st.button("사용하기", key=f"use_template_{template['id']}", use_container_width=True):
                            self._use_template(template)
                        
                        if template['creator_id'] == self.current_user['id']:
                            if st.button("삭제", key=f"del_template_{template['id']}", use_container_width=True):
                                self._delete_template(template['id'])
    
    def _update_experiment_progress(self, experiment_id: str):
        """실험 진행률 업데이트"""
        # 전체 결과 확인
        experiment = self.db.get_experiment(experiment_id)
        results = self.db.get_experiment_results(experiment_id)
        
        total_runs = len(experiment['design_matrix'])
        completed_runs = len([r for r in results if r is not None])
        
        progress = int((completed_runs / total_runs) * 100)
        
        # 상태 업데이트
        updates = {'progress': progress}
        
        if progress >= 100:
            updates['status'] = 'completed'
            updates['completed_at'] = datetime.now()
            
            # 완료 알림
            self.notifier.send(
                "실험 완료",
                f"{experiment['name']} 실험이 완료되었습니다!",
                "success"
            )
        
        self.db.update_experiment(experiment_id, updates)
    
    def _get_pending_runs(self, experiment: Dict) -> List[Dict]:
        """미완료 실험 런 조회"""
        results = self.db.get_experiment_results(experiment['id'])
        design_matrix = np.array(experiment['design_matrix'])
        factors = [f for f in experiment['factors'] if f['type'] == 'continuous']
        
        pending = []
        
        for i in range(len(design_matrix)):
            if i >= len(results) or results[i] is None:
                # 조건 요약 생성
                conditions = []
                for j, factor in enumerate(factors):
                    value = design_matrix[i, j]
                    conditions.append(f"{factor['name']}={value:.1f}")
                
                pending.append({
                    'run_number': i + 1,
                    'conditions': design_matrix[i],
                    'conditions_summary': ', '.join(conditions[:3])  # 처음 3개만
                })
        
        return pending
    
    def _render_basic_statistics(self, experiment: Dict):
        """기초 통계 분석"""
        st.markdown("### 📊 기초 통계")
        
        # 데이터 준비
        df = self._prepare_experiment_data(experiment)
        
        if df.empty:
            st.warning("분석할 데이터가 없습니다")
            return
        
        # 반응변수별 통계
        responses = experiment['responses']
        
        for response in responses:
            resp_name = response['name']
            
            if resp_name in df.columns:
                st.markdown(f"#### {resp_name}")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("평균", f"{df[resp_name].mean():.3f}")
                
                with col2:
                    st.metric("표준편차", f"{df[resp_name].std():.3f}")
                
                with col3:
                    st.metric("최소값", f"{df[resp_name].min():.3f}")
                
                with col4:
                    st.metric("최대값", f"{df[resp_name].max():.3f}")
                
                # 히스토그램
                fig = px.histogram(
                    df,
                    x=resp_name,
                    nbins=20,
                    title=f"{resp_name} 분포"
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_main_effects(self, experiment: Dict):
        """주효과 분석"""
        st.markdown("### 📈 주효과 분석")
        
        # 구현 예정
        st.info("주효과 분석 기능은 준비 중입니다")
    
    def _render_interaction_analysis(self, experiment: Dict):
        """상호작용 분석"""
        st.markdown("### 🔄 상호작용 분석")
        
        # 구현 예정
        st.info("상호작용 분석 기능은 준비 중입니다")
    
    def _render_optimization(self, experiment: Dict):
        """최적화 분석"""
        st.markdown("### 🎯 최적화")
        
        # 구현 예정
        st.info("최적화 분석 기능은 준비 중입니다")

def render():
    """페이지 렌더링 함수"""
    manager = ExperimentDesignManager()
    manager.render_page()

if __name__ == "__main__":
    render()
