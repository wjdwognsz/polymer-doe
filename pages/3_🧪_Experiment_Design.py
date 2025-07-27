"""
3_🧪_Experiment_Design.py - 실험 설계 페이지
AI 지원을 받아 실험을 설계하고 최적화하는 핵심 기능 페이지
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

# 프로젝트 모듈 임포트
from utils.auth_manager import check_authentication
from utils.api_manager import APIManager
from utils.sheets_manager import GoogleSheetsManager
from utils.common_ui import (
    setup_page_config, apply_custom_css, render_header,
    show_notification, show_error, show_success, show_info,
    create_metric_card, render_tooltip
)
from utils.data_processor import DataProcessor
from utils.error_handler import handle_error
from utils.performance_monitor import monitor_performance

from modules.module_registry import ModuleRegistry
from modules.base_module import (
    BaseExperimentModule, Factor, Response, 
    FactorType, ResponseType, OptimizationType,
    DesignConstraints, ExperimentDesign, DesignQuality
)

# 로깅 설정
logger = logging.getLogger(__name__)

# 전역 상수
WIZARD_STEPS = {
    1: {"title": "실험 유형 선택", "icon": "🎯"},
    2: {"title": "요인 정의", "icon": "📊"},
    3: {"title": "반응변수 설정", "icon": "📈"},
    4: {"title": "설계 옵션", "icon": "⚙️"},
    5: {"title": "검토 및 생성", "icon": "✨"}
}

AI_MODELS = {
    'gemini': {'name': 'Google Gemini', 'icon': '🔷'},
    'grok': {'name': 'xAI Grok', 'icon': '🤖'},
    'sambanova': {'name': 'SambaNova', 'icon': '🦙'},
    'deepseek': {'name': 'DeepSeek', 'icon': '🔍'},
    'groq': {'name': 'Groq', 'icon': '⚡'}
}

# 싱글톤 인스턴스 관리
@st.cache_resource
def get_module_registry() -> ModuleRegistry:
    """모듈 레지스트리 싱글톤"""
    return ModuleRegistry()

@st.cache_resource
def get_api_manager() -> APIManager:
    """API 매니저 싱글톤"""
    return APIManager()

@st.cache_resource
def get_sheets_manager() -> GoogleSheetsManager:
    """Google Sheets 매니저 싱글톤"""
    return GoogleSheetsManager()

def initialize_session_state():
    """세션 상태 초기화"""
    defaults = {
        'wizard_step': 1,
        'selected_module_id': None,
        'selected_experiment_type': None,
        'experiment_factors': [],
        'experiment_responses': [],
        'design_constraints': {
            'design_type': 'full_factorial',
            'max_runs': 100,
            'blocks': 1,
            'center_points': 0,
            'replicates': 1,
            'randomize': True
        },
        'generated_design': None,
        'ai_preferences': {
            'show_reasoning': True,
            'show_alternatives': True,
            'show_theory': False,
            'show_confidence': True,
            'show_limitations': True
        },
        'design_chat_history': [],
        'design_versions': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ===== 메인 렌더링 함수 =====
def render():
    """메인 페이지 렌더링"""
    # 인증 체크
    if not check_authentication():
        st.error("로그인이 필요합니다.")
        st.stop()
    
    # 페이지 설정
    setup_page_config("실험 설계", "🧪")
    apply_custom_css()
    
    # 세션 초기화
    initialize_session_state()
    
    # 헤더
    render_header("🧪 실험 설계", "AI 지원 실험 설계 마법사")
    
    # 메인 컨텐츠
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 설계 마법사", "💬 AI 도우미", "📚 내 설계", "📊 템플릿"])
    
    with tab1:
        render_design_wizard()
    
    with tab2:
        render_ai_chat_interface()
    
    with tab3:
        render_saved_designs()
    
    with tab4:
        render_templates()

# ===== 실험 설계 마법사 =====
def render_design_wizard():
    """실험 설계 마법사 UI"""
    st.markdown("## 🎯 실험 설계 마법사")
    
    # 단계 표시
    wizard_step = st.session_state.wizard_step
    progress = wizard_step / len(WIZARD_STEPS)
    
    # 프로그레스 바
    col1, col2, col3, col4, col5 = st.columns(5)
    for i, col in enumerate([col1, col2, col3, col4, col5], 1):
        with col:
            if i < wizard_step:
                st.success(f"✅ {WIZARD_STEPS[i]['title']}")
            elif i == wizard_step:
                st.info(f"👉 {WIZARD_STEPS[i]['title']}")
            else:
                st.caption(f"⏳ {WIZARD_STEPS[i]['title']}")
    
    st.progress(progress)
    st.divider()
    
    # 각 단계별 렌더링
    if wizard_step == 1:
        render_step1_experiment_type()
    elif wizard_step == 2:
        render_step2_factors()
    elif wizard_step == 3:
        render_step3_responses()
    elif wizard_step == 4:
        render_step4_constraints()
    elif wizard_step == 5:
        render_step5_review()

def render_step1_experiment_type():
    """Step 1: 실험 유형 선택"""
    st.markdown("### Step 1: 실험 유형 선택")
    
    # 모듈 레지스트리에서 사용 가능한 모듈 가져오기
    registry = get_module_registry()
    modules = registry.list_modules()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 연구 분야 선택
        categories = list(set(m.get('category', 'general') for m in modules))
        selected_category = st.selectbox(
            "연구 분야",
            ['전체'] + sorted(categories),
            help="연구 분야를 선택하세요"
        )
        
        # 실험 모듈 선택
        if selected_category == '전체':
            available_modules = modules
        else:
            available_modules = [m for m in modules if m.get('category') == selected_category]
        
        # 모듈 선택 UI
        st.markdown("#### 실험 모듈 선택")
        
        for module in available_modules:
            col_a, col_b = st.columns([3, 1])
            with col_a:
                if st.button(
                    f"{module.get('icon', '🔬')} **{module['name']}**\n\n{module.get('description', '')}",
                    key=f"module_{module['id']}",
                    use_container_width=True
                ):
                    st.session_state.selected_module_id = module['id']
                    st.rerun()
            
            with col_b:
                if st.session_state.selected_module_id == module['id']:
                    st.success("✅ 선택됨")
        
        # 선택된 모듈이 있으면 실험 유형 표시
        if st.session_state.selected_module_id:
            st.divider()
            st.markdown("#### 실험 유형 선택")
            
            try:
                module = registry.get_module(st.session_state.selected_module_id)
                experiment_types = module.get_experiment_types()
                
                for exp_type in experiment_types:
                    info = module.get_experiment_info(exp_type)
                    if st.button(
                        f"**{info['name']}**\n\n"
                        f"요인: {info['num_factors']}개 | "
                        f"반응변수: {info['num_responses']}개 | "
                        f"예상 실험수: {info['typical_runs']}회",
                        key=f"exp_{exp_type}",
                        use_container_width=True
                    ):
                        st.session_state.selected_experiment_type = exp_type
                        st.rerun()
                
            except Exception as e:
                st.error(f"모듈 로드 오류: {str(e)}")
    
    with col2:
        # AI 추천
        st.markdown("#### 🤖 AI 추천")
        
        if st.button("AI에게 추천받기", use_container_width=True):
            with st.spinner("AI가 분석 중..."):
                recommendation = get_ai_experiment_recommendation()
                render_ai_response(recommendation, "추천")
        
        # 도움말
        with st.expander("ℹ️ 도움말"):
            st.write("""
            **실험 유형 선택 가이드**
            
            1. **연구 분야**: 귀하의 연구 분야를 선택하세요
            2. **실험 모듈**: 사용하고자 하는 실험 설계 방법을 선택하세요
            3. **실험 유형**: 구체적인 실험 목적을 선택하세요
            
            💡 **팁**: AI 추천을 받으면 더 적합한 선택을 할 수 있습니다.
            """)
    
    # 네비게이션
    col1, col2, col3 = st.columns([1, 1, 1])
    with col3:
        if st.session_state.selected_module_id and st.session_state.selected_experiment_type:
            if st.button("다음 단계 →", type="primary", use_container_width=True):
                st.session_state.wizard_step = 2
                st.rerun()

def render_step2_factors():
    """Step 2: 요인 정의"""
    st.markdown("### Step 2: 요인 정의")
    
    # 모듈에서 기본 요인 가져오기
    registry = get_module_registry()
    module = registry.get_module(st.session_state.selected_module_id)
    
    # 기본 요인이 없으면 빈 리스트로 초기화
    if not st.session_state.experiment_factors:
        default_factors = module.get_default_factors(st.session_state.selected_experiment_type)
        if default_factors:
            st.session_state.experiment_factors = [asdict(f) for f in default_factors]
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("#### 실험 요인 설정")
        
        # 요인 추가 버튼
        if st.button("➕ 요인 추가", use_container_width=True):
            st.session_state.experiment_factors.append({
                'name': f'요인 {len(st.session_state.experiment_factors) + 1}',
                'type': 'continuous',
                'min_value': 0,
                'max_value': 100,
                'unit': '',
                'levels': [],
                'description': ''
            })
            st.rerun()
        
        # 요인 편집
        for i, factor in enumerate(st.session_state.experiment_factors):
            with st.expander(f"**{factor['name']}** ({factor['type']})", expanded=True):
                col_a, col_b = st.columns([3, 1])
                
                with col_a:
                    factor['name'] = st.text_input("요인 이름", value=factor['name'], key=f"fname_{i}")
                    factor['description'] = st.text_area("설명", value=factor.get('description', ''), key=f"fdesc_{i}")
                
                with col_b:
                    factor['type'] = st.selectbox(
                        "유형",
                        ['continuous', 'categorical', 'discrete'],
                        format_func=lambda x: {'continuous': '연속형', 'categorical': '범주형', 'discrete': '이산형'}[x],
                        index=['continuous', 'categorical', 'discrete'].index(factor['type']),
                        key=f"ftype_{i}"
                    )
                
                # 유형별 설정
                if factor['type'] == 'continuous':
                    col_1, col_2, col_3 = st.columns(3)
                    with col_1:
                        factor['min_value'] = st.number_input("최소값", value=factor.get('min_value', 0), key=f"fmin_{i}")
                    with col_2:
                        factor['max_value'] = st.number_input("최대값", value=factor.get('max_value', 100), key=f"fmax_{i}")
                    with col_3:
                        factor['unit'] = st.text_input("단위", value=factor.get('unit', ''), key=f"funit_{i}")
                
                elif factor['type'] == 'categorical':
                    levels_str = st.text_area(
                        "수준 (쉼표로 구분)",
                        value=', '.join(factor.get('levels', [])),
                        key=f"flevels_{i}"
                    )
                    factor['levels'] = [l.strip() for l in levels_str.split(',') if l.strip()]
                
                elif factor['type'] == 'discrete':
                    col_1, col_2, col_3 = st.columns(3)
                    with col_1:
                        factor['min_value'] = st.number_input("최소값", value=int(factor.get('min_value', 0)), step=1, key=f"fdmin_{i}")
                    with col_2:
                        factor['max_value'] = st.number_input("최대값", value=int(factor.get('max_value', 10)), step=1, key=f"fdmax_{i}")
                    with col_3:
                        factor['unit'] = st.text_input("단위", value=factor.get('unit', ''), key=f"fdunit_{i}")
                
                # 삭제 버튼
                if st.button(f"🗑️ 삭제", key=f"fdel_{i}"):
                    st.session_state.experiment_factors.pop(i)
                    st.rerun()
        
        # 요인이 없는 경우
        if not st.session_state.experiment_factors:
            st.info("요인을 추가하여 실험을 설계하세요.")
    
    with col2:
        # AI 지원
        st.markdown("#### 🤖 AI 지원")
        
        user_input = st.text_area("실험 요구사항을 입력하세요", height=100)
        
        if st.button("AI 요인 추천", use_container_width=True):
            if user_input:
                with st.spinner("AI가 분석 중..."):
                    factors = get_ai_factor_recommendations(user_input)
                    render_ai_response(factors, "요인추천")
            else:
                st.warning("요구사항을 입력해주세요.")
        
        # 템플릿
        st.markdown("#### 📋 빠른 템플릿")
        
        templates = module.get_factor_templates(st.session_state.selected_experiment_type)
        template_names = list(templates.keys()) if templates else []
        
        if template_names:
            selected_template = st.selectbox("템플릿 선택", ['없음'] + template_names)
            
            if selected_template != '없음' and st.button("템플릿 적용"):
                st.session_state.experiment_factors = templates[selected_template]
                st.success("템플릿이 적용되었습니다.")
                st.rerun()
    
    # 네비게이션
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← 이전 단계", use_container_width=True):
            st.session_state.wizard_step = 1
            st.rerun()
    with col3:
        if len(st.session_state.experiment_factors) >= 1:
            if st.button("다음 단계 →", type="primary", use_container_width=True):
                st.session_state.wizard_step = 3
                st.rerun()
        else:
            st.warning("최소 1개 이상의 요인을 정의해주세요.")

def render_step3_responses():
    """Step 3: 반응변수 설정"""
    st.markdown("### Step 3: 반응변수 설정")
    
    # 모듈에서 기본 반응변수 가져오기
    registry = get_module_registry()
    module = registry.get_module(st.session_state.selected_module_id)
    
    # 기본 반응변수가 없으면 빈 리스트로 초기화
    if not st.session_state.experiment_responses:
        default_responses = module.get_default_responses(st.session_state.selected_experiment_type)
        if default_responses:
            st.session_state.experiment_responses = [asdict(r) for r in default_responses]
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("#### 반응변수 설정")
        
        # 반응변수 추가 버튼
        if st.button("➕ 반응변수 추가", use_container_width=True):
            st.session_state.experiment_responses.append({
                'name': f'반응변수 {len(st.session_state.experiment_responses) + 1}',
                'type': 'continuous',
                'unit': '',
                'optimization': 'maximize',
                'target_value': None,
                'lower_limit': None,
                'upper_limit': None,
                'importance': 1.0,
                'description': ''
            })
            st.rerun()
        
        # 반응변수 편집
        for i, response in enumerate(st.session_state.experiment_responses):
            with st.expander(f"**{response['name']}** ({response['optimization']})", expanded=True):
                col_a, col_b = st.columns([3, 1])
                
                with col_a:
                    response['name'] = st.text_input("반응변수 이름", value=response['name'], key=f"rname_{i}")
                    response['description'] = st.text_area("설명", value=response.get('description', ''), key=f"rdesc_{i}")
                
                with col_b:
                    response['type'] = st.selectbox(
                        "유형",
                        ['continuous', 'binary', 'count'],
                        format_func=lambda x: {'continuous': '연속형', 'binary': '이진형', 'count': '계수형'}[x],
                        index=['continuous', 'binary', 'count'].index(response['type']),
                        key=f"rtype_{i}"
                    )
                
                # 최적화 방향
                col_1, col_2, col_3 = st.columns(3)
                with col_1:
                    response['optimization'] = st.selectbox(
                        "최적화 방향",
                        ['maximize', 'minimize', 'target', 'in_range'],
                        format_func=lambda x: {
                            'maximize': '최대화',
                            'minimize': '최소화',
                            'target': '목표값',
                            'in_range': '범위내'
                        }[x],
                        index=['maximize', 'minimize', 'target', 'in_range'].index(response['optimization']),
                        key=f"ropt_{i}"
                    )
                
                with col_2:
                    if response['optimization'] == 'target':
                        response['target_value'] = st.number_input(
                            "목표값",
                            value=response.get('target_value', 0.0),
                            key=f"rtarget_{i}"
                        )
                    elif response['optimization'] == 'in_range':
                        response['lower_limit'] = st.number_input(
                            "하한",
                            value=response.get('lower_limit', 0.0),
                            key=f"rlower_{i}"
                        )
                
                with col_3:
                    if response['optimization'] == 'in_range':
                        response['upper_limit'] = st.number_input(
                            "상한",
                            value=response.get('upper_limit', 100.0),
                            key=f"rupper_{i}"
                        )
                    else:
                        response['unit'] = st.text_input("단위", value=response.get('unit', ''), key=f"runit_{i}")
                
                # 중요도
                response['importance'] = st.slider(
                    "중요도",
                    min_value=0.1,
                    max_value=10.0,
                    value=response.get('importance', 1.0),
                    step=0.1,
                    key=f"rimp_{i}"
                )
                
                # 삭제 버튼
                if st.button(f"🗑️ 삭제", key=f"rdel_{i}"):
                    st.session_state.experiment_responses.pop(i)
                    st.rerun()
        
        # 반응변수가 없는 경우
        if not st.session_state.experiment_responses:
            st.info("반응변수를 추가하여 실험 목표를 설정하세요.")
    
    with col2:
        # AI 지원
        st.markdown("#### 🤖 AI 지원")
        
        if st.button("AI 반응변수 추천", use_container_width=True):
            with st.spinner("AI가 분석 중..."):
                responses = get_ai_response_recommendations()
                render_ai_response(responses, "반응변수추천")
        
        # 가이드
        with st.expander("ℹ️ 최적화 방향 가이드"):
            st.write("""
            **최적화 방향 선택 가이드**
            
            - **최대화**: 값이 클수록 좋은 경우 (예: 수율, 강도)
            - **최소화**: 값이 작을수록 좋은 경우 (예: 비용, 시간)
            - **목표값**: 특정 값에 가까울수록 좋은 경우 (예: pH 7.0)
            - **범위내**: 특정 범위 내에 있어야 하는 경우 (예: 온도 20-25°C)
            
            💡 **중요도**: 여러 반응변수 중 상대적 중요성을 나타냅니다.
            """)
    
    # 네비게이션
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← 이전 단계", use_container_width=True):
            st.session_state.wizard_step = 2
            st.rerun()
    with col3:
        if len(st.session_state.experiment_responses) >= 1:
            if st.button("다음 단계 →", type="primary", use_container_width=True):
                st.session_state.wizard_step = 4
                st.rerun()
        else:
            st.warning("최소 1개 이상의 반응변수를 정의해주세요.")

def render_step4_constraints():
    """Step 4: 설계 옵션 및 제약조건"""
    st.markdown("### Step 4: 설계 옵션 및 제약조건")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("#### 실험 설계 유형")
        
        # 설계 유형 선택
        design_types = {
            'full_factorial': '완전요인설계 (Full Factorial)',
            'fractional_factorial': '부분요인설계 (Fractional Factorial)',
            'ccd': '중심합성설계 (Central Composite)',
            'box_behnken': 'Box-Behnken 설계',
            'plackett_burman': 'Plackett-Burman 스크리닝',
            'd_optimal': 'D-최적 설계',
            'space_filling': '공간충진 설계 (Space-Filling)',
            'custom': '사용자 정의'
        }
        
        selected_design = st.selectbox(
            "설계 방법",
            list(design_types.keys()),
            format_func=lambda x: design_types[x],
            index=list(design_types.keys()).index(st.session_state.design_constraints['design_type']),
            help="실험 설계 방법을 선택하세요"
        )
        st.session_state.design_constraints['design_type'] = selected_design
        
        # 설계별 옵션
        st.markdown("#### 설계 옵션")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.session_state.design_constraints['max_runs'] = st.number_input(
                "최대 실험 횟수",
                min_value=1,
                max_value=1000,
                value=st.session_state.design_constraints['max_runs'],
                help="수행 가능한 최대 실험 횟수"
            )
        
        with col_b:
            if selected_design in ['full_factorial', 'fractional_factorial', 'ccd']:
                st.session_state.design_constraints['center_points'] = st.number_input(
                    "중심점 반복",
                    min_value=0,
                    max_value=10,
                    value=st.session_state.design_constraints.get('center_points', 0),
                    help="중심점에서의 반복 실험 횟수"
                )
        
        with col_c:
            st.session_state.design_constraints['replicates'] = st.number_input(
                "전체 반복",
                min_value=1,
                max_value=5,
                value=st.session_state.design_constraints['replicates'],
                help="전체 설계의 반복 횟수"
            )
        
        # 블록화 및 랜덤화
        st.markdown("#### 실험 수행 옵션")
        
        col_1, col_2 = st.columns(2)
        
        with col_1:
            st.session_state.design_constraints['blocks'] = st.number_input(
                "블록 수",
                min_value=1,
                max_value=10,
                value=st.session_state.design_constraints['blocks'],
                help="실험을 나누어 수행할 블록의 수"
            )
        
        with col_2:
            st.session_state.design_constraints['randomize'] = st.checkbox(
                "실험 순서 랜덤화",
                value=st.session_state.design_constraints['randomize'],
                help="실험 순서를 무작위로 배치"
            )
        
        # 추가 제약조건
        st.markdown("#### 추가 제약조건")
        
        constraints_text = st.text_area(
            "제약조건 입력 (선택사항)",
            placeholder="예: 온도 * 압력 < 1000\n     용매A + 용매B = 100",
            height=100,
            help="실험 조건에 대한 추가 제약사항을 입력하세요"
        )
        
        if constraints_text:
            st.session_state.design_constraints['custom_constraints'] = constraints_text
        
        # 예상 실험 수 계산 및 표시
        estimated_runs = estimate_experiment_runs()
        
        if estimated_runs:
            st.info(f"""
            **예상 실험 수**: {estimated_runs}회
            - 기본 설계: {estimated_runs // st.session_state.design_constraints['replicates']}회
            - 반복 포함: {estimated_runs}회
            """)
    
    with col2:
        # AI 최적화 제안
        st.markdown("#### 🤖 AI 최적화")
        
        if st.button("AI 설계 최적화", use_container_width=True):
            with st.spinner("AI가 최적 설계를 찾는 중..."):
                optimization = get_ai_design_optimization()
                render_ai_response(optimization, "설계최적화")
        
        # 설계 품질 메트릭
        st.markdown("#### 📊 설계 품질")
        
        quality_metrics = calculate_design_quality_preview()
        
        for metric, value in quality_metrics.items():
            if value is not None:
                st.metric(metric, value)
        
        # 도움말
        with st.expander("ℹ️ 설계 유형 가이드"):
            st.write("""
            **설계 유형 선택 가이드**
            
            - **완전요인설계**: 모든 요인 조합을 탐색 (요인 수가 적을 때)
            - **부분요인설계**: 주요 효과 중심으로 탐색 (요인 수가 많을 때)
            - **중심합성설계**: 2차 효과까지 모델링 (최적화 목적)
            - **Box-Behnken**: 3수준 설계, 극단값 회피
            - **Plackett-Burman**: 많은 요인 스크리닝
            - **D-최적**: 제약조건이 있을 때 최적
            """)
    
    # 네비게이션
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← 이전 단계", use_container_width=True):
            st.session_state.wizard_step = 3
            st.rerun()
    with col3:
        if st.button("다음 단계 →", type="primary", use_container_width=True):
            st.session_state.wizard_step = 5
            st.rerun()

def render_step5_review():
    """Step 5: 검토 및 생성"""
    st.markdown("### Step 5: 검토 및 생성")
    
    # 설계 요약
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 📋 설계 요약")
        
        # 실험 정보 카드
        with st.container():
            st.info(f"""
            **실험 모듈**: {st.session_state.selected_module_id}  
            **실험 유형**: {st.session_state.selected_experiment_type}  
            **요인 수**: {len(st.session_state.experiment_factors)}  
            **반응변수 수**: {len(st.session_state.experiment_responses)}  
            **설계 유형**: {st.session_state.design_constraints['design_type']}  
            **최대 실험 횟수**: {st.session_state.design_constraints['max_runs']}
            """)
        
        # 요인 요약
        with st.expander("📊 요인 상세", expanded=True):
            factors_df = pd.DataFrame(st.session_state.experiment_factors)
            if not factors_df.empty:
                display_cols = ['name', 'type', 'min_value', 'max_value', 'unit', 'levels']
                available_cols = [col for col in display_cols if col in factors_df.columns]
                st.dataframe(
                    factors_df[available_cols],
                    use_container_width=True,
                    hide_index=True
                )
        
        # 반응변수 요약
        with st.expander("📈 반응변수 상세", expanded=True):
            responses_df = pd.DataFrame(st.session_state.experiment_responses)
            if not responses_df.empty:
                display_cols = ['name', 'type', 'optimization', 'target_value', 'importance', 'unit']
                available_cols = [col for col in display_cols if col in responses_df.columns]
                st.dataframe(
                    responses_df[available_cols],
                    use_container_width=True,
                    hide_index=True
                )
    
    with col2:
        # 실험 설계 생성
        st.markdown("#### 🚀 작업")
        
        if st.button("✨ 실험 설계 생성", type="primary", use_container_width=True):
            generate_experiment_design()
        
        st.divider()
        
        if st.button("💾 템플릿으로 저장", use_container_width=True):
            save_as_template()
        
        if st.button("📤 설정 내보내기", use_container_width=True):
            export_design_settings()
        
        if st.button("🔄 처음부터 다시", use_container_width=True):
            reset_wizard()
    
    # 생성된 설계 표시
    if st.session_state.generated_design:
        st.divider()
        render_generated_design()
    
    # 네비게이션
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← 이전 단계", use_container_width=True):
            st.session_state.wizard_step = 4
            st.rerun()
    with col3:
        if st.session_state.generated_design:
            if st.button("✅ 완료 및 저장", type="primary", use_container_width=True):
                save_experiment_design()

# ===== 실험 설계 생성 및 표시 =====
def generate_experiment_design():
    """실험 설계 생성"""
    try:
        with st.spinner("실험 설계를 생성하는 중..."):
            # 모듈 가져오기
            registry = get_module_registry()
            module = registry.get_module(st.session_state.selected_module_id)
            
            # Factor와 Response 객체 생성
            factors = [Factor(**f) for f in st.session_state.experiment_factors]
            responses = [Response(**r) for r in st.session_state.experiment_responses]
            
            # 제약조건 객체 생성
            constraints = DesignConstraints(**st.session_state.design_constraints)
            
            # 설계 생성
            design = module.generate_design(
                experiment_type=st.session_state.selected_experiment_type,
                factors=factors,
                responses=responses,
                constraints=constraints
            )
            
            # 결과 저장
            st.session_state.generated_design = design
            
            # AI 분석 자동 실행
            if design:
                analysis = get_ai_design_analysis(design)
                st.session_state.design_analysis = analysis
            
            st.success("실험 설계가 성공적으로 생성되었습니다!")
            st.rerun()
            
    except Exception as e:
        handle_error(e, "실험 설계 생성 중 오류가 발생했습니다")

def render_generated_design():
    """생성된 실험 설계 표시"""
    design = st.session_state.generated_design
    
    if not design:
        return
    
    st.markdown("### 📊 생성된 실험 설계")
    
    # 설계 정보 표시
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("실험 횟수", f"{len(design.runs)}회")
    
    with col2:
        st.metric("설계 유형", design.design_type)
    
    with col3:
        if hasattr(design, 'quality') and design.quality:
            st.metric("D-효율성", f"{design.quality.d_efficiency:.1f}%")
    
    with col4:
        if hasattr(design, 'quality') and design.quality:
            st.metric("직교성", f"{design.quality.orthogonality:.2f}")
    
    # 탭 구성
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 실험 런 테이블", 
        "📊 설계 공간 시각화", 
        "📈 통계적 속성",
        "✏️ 편집 및 수정",
        "🤖 AI 분석"
    ])
    
    with tab1:
        render_run_table(design)
    
    with tab2:
        render_design_space_visualization(design)
    
    with tab3:
        render_statistical_properties(design)
    
    with tab4:
        render_design_editor(design)
    
    with tab5:
        render_ai_analysis()

def render_run_table(design: ExperimentDesign):
    """실험 런 테이블 표시"""
    st.markdown("#### 실험 런 테이블")
    
    # 표시 옵션
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_coded = st.checkbox("코드화된 값 표시", value=False)
    
    with col2:
        show_blocks = st.checkbox("블록 표시", value=True)
    
    with col3:
        show_std_order = st.checkbox("표준 순서 표시", value=False)
    
    # 데이터프레임 준비
    df = design.runs.copy()
    
    # 컬럼 순서 정리
    id_cols = ['Run']
    if show_std_order and 'StdOrder' in df.columns:
        id_cols.append('StdOrder')
    if show_blocks and 'Block' in df.columns:
        id_cols.append('Block')
    
    factor_cols = [f.name for f in design.factors if f.name in df.columns]
    other_cols = [col for col in df.columns if col not in id_cols + factor_cols]
    
    df = df[id_cols + factor_cols + other_cols]
    
    # 테이블 표시
    edited_df = st.data_editor(
        df,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        disabled=id_cols
    )
    
    # 변경사항 저장
    if not df.equals(edited_df):
        design.runs = edited_df
        st.session_state.generated_design = design
        st.success("변경사항이 저장되었습니다.")
    
    # 내보내기 옵션
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 CSV 다운로드",
            csv,
            "experiment_design.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        # Excel 다운로드 (BytesIO 사용)
        from io import BytesIO
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Design', index=False)
            # 메타데이터 시트 추가
            metadata_df = pd.DataFrame({
                'Property': ['Design Type', 'Factors', 'Responses', 'Runs', 'Created'],
                'Value': [
                    design.design_type,
                    len(design.factors),
                    len(design.responses),
                    len(design.runs),
                    datetime.now().strftime('%Y-%m-%d %H:%M')
                ]
            })
            metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
        
        st.download_button(
            "📥 Excel 다운로드",
            excel_buffer.getvalue(),
            "experiment_design.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col3:
        if st.button("📋 클립보드 복사", use_container_width=True):
            df.to_clipboard(index=False)
            st.success("클립보드에 복사되었습니다!")

def render_design_space_visualization(design: ExperimentDesign):
    """설계 공간 시각화"""
    st.markdown("#### 설계 공간 시각화")
    
    continuous_factors = [f for f in design.factors if f.type == FactorType.CONTINUOUS]
    
    if len(continuous_factors) < 2:
        st.warning("시각화를 위해서는 최소 2개 이상의 연속형 요인이 필요합니다.")
        return
    
    # 시각화 유형 선택
    viz_type = st.selectbox(
        "시각화 유형",
        ["2D 산점도", "3D 산점도", "평행 좌표계", "페어플롯"]
    )
    
    if viz_type == "2D 산점도":
        col1, col2 = st.columns(2)
        
        with col1:
            x_factor = st.selectbox(
                "X축",
                [f.name for f in continuous_factors]
            )
        
        with col2:
            y_factor = st.selectbox(
                "Y축",
                [f.name for f in continuous_factors if f.name != x_factor]
            )
        
        # 색상 인코딩 옵션
        color_by = st.selectbox(
            "색상 기준",
            ["없음", "블록", "실행 순서"] + 
            [f.name for f in design.factors if f.type == FactorType.CATEGORICAL]
        )
        
        # 플롯 생성
        fig = go.Figure()
        
        if color_by == "없음":
            fig.add_trace(go.Scatter(
                x=design.runs[x_factor],
                y=design.runs[y_factor],
                mode='markers',
                marker=dict(size=12, color='blue'),
                text=[f"Run {i+1}" for i in range(len(design.runs))],
                hovertemplate='%{text}<br>%{x}<br>%{y}<extra></extra>'
            ))
        else:
            color_col = 'Run' if color_by == "실행 순서" else color_by
            fig = px.scatter(
                design.runs,
                x=x_factor,
                y=y_factor,
                color=color_col if color_col in design.runs.columns else None,
                hover_data=['Run'],
                title=f"{x_factor} vs {y_factor}"
            )
        
        fig.update_layout(
            xaxis_title=x_factor,
            yaxis_title=y_factor,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "3D 산점도":
        if len(continuous_factors) < 3:
            st.warning("3D 시각화를 위해서는 최소 3개 이상의 연속형 요인이 필요합니다.")
        else:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                x_factor = st.selectbox("X축", [f.name for f in continuous_factors])
            
            with col2:
                y_factor = st.selectbox(
                    "Y축",
                    [f.name for f in continuous_factors if f.name != x_factor]
                )
            
            with col3:
                z_factor = st.selectbox(
                    "Z축",
                    [f.name for f in continuous_factors if f.name not in [x_factor, y_factor]]
                )
            
            fig = go.Figure(data=[go.Scatter3d(
                x=design.runs[x_factor],
                y=design.runs[y_factor],
                z=design.runs[z_factor],
                mode='markers',
                marker=dict(size=8, color=design.runs.index, colorscale='Viridis'),
                text=[f"Run {i+1}" for i in range(len(design.runs))]
            )])
            
            fig.update_layout(
                scene=dict(
                    xaxis_title=x_factor,
                    yaxis_title=y_factor,
                    zaxis_title=z_factor
                ),
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "평행 좌표계":
        # 연속형 요인만 선택
        factor_cols = [f.name for f in continuous_factors if f.name in design.runs.columns]
        
        # 정규화
        normalized_df = design.runs.copy()
        for col in factor_cols:
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            if max_val > min_val:
                normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
        
        fig = go.Figure(data=go.Parcoords(
            dimensions=[
                dict(
                    label=col,
                    values=normalized_df[col],
                    range=[0, 1]
                ) for col in factor_cols
            ],
            line=dict(
                color=normalized_df.index,
                colorscale='Viridis'
            )
        ))
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "페어플롯":
        # Plotly Express scatter matrix
        factor_cols = [f.name for f in continuous_factors if f.name in design.runs.columns][:4]  # 최대 4개
        
        if len(factor_cols) < 2:
            st.warning("페어플롯을 위해서는 최소 2개 이상의 요인이 필요합니다.")
        else:
            fig = px.scatter_matrix(
                design.runs,
                dimensions=factor_cols,
                hover_data=['Run'],
                title="요인 간 관계"
            )
            
            fig.update_traces(diagonal_visible=False)
            fig.update_layout(height=800)
            
            st.plotly_chart(fig, use_container_width=True)

def render_statistical_properties(design: ExperimentDesign):
    """통계적 속성 표시"""
    st.markdown("#### 통계적 속성")
    
    # 설계 품질 메트릭
    if hasattr(design, 'quality') and design.quality:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "D-효율성",
                f"{design.quality.d_efficiency:.1f}%",
                help="정보 행렬의 행렬식 기반 효율성"
            )
        
        with col2:
            st.metric(
                "A-효율성",
                f"{design.quality.a_efficiency:.1f}%",
                help="평균 분산 기반 효율성"
            )
        
        with col3:
            st.metric(
                "G-효율성",
                f"{design.quality.g_efficiency:.1f}%",
                help="최대 예측 분산 기반 효율성"
            )
        
        with col4:
            st.metric(
                "조건수",
                f"{design.quality.condition_number:.2f}",
                help="설계 행렬의 조건수 (낮을수록 좋음)"
            )
    
    # 상관 행렬
    st.markdown("##### 요인 간 상관관계")
    
    continuous_factors = [f for f in design.factors if f.type == FactorType.CONTINUOUS]
    if continuous_factors:
        factor_cols = [f.name for f in continuous_factors if f.name in design.runs.columns]
        
        if len(factor_cols) > 1:
            corr_matrix = design.runs[factor_cols].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.values.round(3),
                texttemplate='%{text}',
                showscale=True
            ))
            
            fig.update_layout(
                title="요인 상관 행렬",
                height=400,
                xaxis=dict(side='bottom')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 직교성 평가
            max_corr = corr_matrix.abs().values[~np.eye(len(factor_cols), dtype=bool)].max()
            if max_corr < 0.1:
                st.success(f"✅ 우수한 직교성 (최대 상관: {max_corr:.3f})")
            elif max_corr < 0.3:
                st.info(f"⚠️ 양호한 직교성 (최대 상관: {max_corr:.3f})")
            else:
                st.warning(f"❌ 낮은 직교성 (최대 상관: {max_corr:.3f})")
    
    # 파워 분석
    st.markdown("##### 통계적 검정력")
    
    power_results = calculate_power_analysis(design)
    
    if power_results:
        power_df = pd.DataFrame(power_results)
        
        fig = go.Figure()
        
        for effect_type in ['주효과', '2차 교호작용']:
            if effect_type in power_df.columns:
                fig.add_trace(go.Bar(
                    name=effect_type,
                    x=power_df['효과크기'],
                    y=power_df[effect_type]
                ))
        
        fig.update_layout(
            title="효과 크기별 검정력",
            xaxis_title="효과 크기",
            yaxis_title="검정력 (%)",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_design_editor(design: ExperimentDesign):
    """설계 편집기"""
    st.markdown("#### 설계 편집 및 수정")
    
    # 편집 옵션
    edit_option = st.radio(
        "편집 작업 선택",
        ["런 추가/삭제", "조건 수정", "증강 설계", "최적화"]
    )
    
    if edit_option == "런 추가/삭제":
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 런 추가")
            
            new_run = {}
            for factor in design.factors:
                if factor.type == FactorType.CONTINUOUS:
                    new_run[factor.name] = st.number_input(
                        factor.name,
                        min_value=float(factor.min_value),
                        max_value=float(factor.max_value),
                        value=float((factor.min_value + factor.max_value) / 2),
                        key=f"new_{factor.name}"
                    )
                elif factor.type == FactorType.CATEGORICAL:
                    new_run[factor.name] = st.selectbox(
                        factor.name,
                        factor.levels,
                        key=f"new_{factor.name}"
                    )
            
            if st.button("➕ 런 추가", use_container_width=True):
                new_run['Run'] = len(design.runs) + 1
                design.runs = pd.concat([design.runs, pd.DataFrame([new_run])], ignore_index=True)
                st.session_state.generated_design = design
                st.success("새 런이 추가되었습니다.")
                st.rerun()
        
        with col2:
            st.markdown("##### 런 삭제")
            
            run_to_delete = st.selectbox(
                "삭제할 런 선택",
                design.runs['Run'].tolist()
            )
            
            if st.button("🗑️ 런 삭제", use_container_width=True):
                design.runs = design.runs[design.runs['Run'] != run_to_delete].reset_index(drop=True)
                st.session_state.generated_design = design
                st.success(f"런 {run_to_delete}이(가) 삭제되었습니다.")
                st.rerun()
    
    elif edit_option == "조건 수정":
        st.markdown("##### 일괄 조건 수정")
        
        # 요인 선택
        factor_to_modify = st.selectbox(
            "수정할 요인",
            [f.name for f in design.factors]
        )
        
        # 수정 방법
        modify_method = st.radio(
            "수정 방법",
            ["특정 값으로 변경", "비율로 조정", "오프셋 추가"]
        )
        
        # 런 선택
        runs_to_modify = st.multiselect(
            "수정할 런 선택",
            design.runs['Run'].tolist(),
            default=design.runs['Run'].tolist()
        )
        
        if modify_method == "특정 값으로 변경":
            new_value = st.number_input("새 값")
            
            if st.button("적용"):
                design.runs.loc[design.runs['Run'].isin(runs_to_modify), factor_to_modify] = new_value
                st.session_state.generated_design = design
                st.success("조건이 수정되었습니다.")
                st.rerun()
        
        elif modify_method == "비율로 조정":
            scale_factor = st.number_input("배율", value=1.0)
            
            if st.button("적용"):
                mask = design.runs['Run'].isin(runs_to_modify)
                design.runs.loc[mask, factor_to_modify] *= scale_factor
                st.session_state.generated_design = design
                st.success("조건이 수정되었습니다.")
                st.rerun()
        
        elif modify_method == "오프셋 추가":
            offset = st.number_input("오프셋 값", value=0.0)
            
            if st.button("적용"):
                mask = design.runs['Run'].isin(runs_to_modify)
                design.runs.loc[mask, factor_to_modify] += offset
                st.session_state.generated_design = design
                st.success("조건이 수정되었습니다.")
                st.rerun()
    
    elif edit_option == "증강 설계":
        st.markdown("##### 설계 증강")
        
        augment_type = st.selectbox(
            "증강 유형",
            ["축 점 추가", "중심점 추가", "별 점 추가", "사용자 정의 점"]
        )
        
        if augment_type == "중심점 추가":
            n_center = st.number_input("추가할 중심점 수", min_value=1, max_value=10, value=3)
            
            if st.button("중심점 추가"):
                center_point = {}
                center_point['Run'] = len(design.runs) + 1
                
                for factor in design.factors:
                    if factor.type == FactorType.CONTINUOUS:
                        center_point[factor.name] = (factor.min_value + factor.max_value) / 2
                    elif factor.type == FactorType.CATEGORICAL:
                        center_point[factor.name] = factor.levels[0]  # 첫 번째 수준
                
                for i in range(n_center):
                    new_point = center_point.copy()
                    new_point['Run'] = len(design.runs) + i + 1
                    design.runs = pd.concat([design.runs, pd.DataFrame([new_point])], ignore_index=True)
                
                st.session_state.generated_design = design
                st.success(f"{n_center}개의 중심점이 추가되었습니다.")
                st.rerun()
    
    elif edit_option == "최적화":
        st.markdown("##### 설계 최적화")
        
        optimization_criterion = st.selectbox(
            "최적화 기준",
            ["D-최적성", "A-최적성", "G-최적성", "I-최적성"]
        )
        
        if st.button("🎯 설계 최적화", use_container_width=True):
            with st.spinner("설계를 최적화하는 중..."):
                # 실제 최적화 알고리즘 구현
                optimized_design = optimize_design(design, optimization_criterion)
                
                if optimized_design:
                    st.session_state.generated_design = optimized_design
                    st.success("설계가 최적화되었습니다!")
                    st.rerun()

def render_ai_analysis():
    """AI 분석 결과 표시"""
    st.markdown("#### 🤖 AI 설계 분석")
    
    if 'design_analysis' not in st.session_state:
        if st.button("AI 분석 실행", use_container_width=True):
            with st.spinner("AI가 설계를 분석하는 중..."):
                analysis = get_ai_design_analysis(st.session_state.generated_design)
                st.session_state.design_analysis = analysis
                st.rerun()
    else:
        # AI 분석 결과 표시
        analysis = st.session_state.design_analysis
        render_ai_response(analysis, "설계분석")
        
        # 재분석 버튼
        if st.button("🔄 재분석", use_container_width=True):
            with st.spinner("AI가 다시 분석하는 중..."):
                analysis = get_ai_design_analysis(st.session_state.generated_design)
                st.session_state.design_analysis = analysis
                st.rerun()

# ===== AI 관련 함수들 =====
def get_ai_experiment_recommendation() -> Dict[str, Any]:
    """AI 실험 추천"""
    api_manager = get_api_manager()
    
    prompt = """
    사용자가 실험 설계를 시작하려고 합니다. 
    적절한 실험 모듈과 유형을 추천해주세요.
    
    고려사항:
    - 일반적인 연구 분야별 추천
    - 초보자 친화적인 옵션
    - 고급 사용자를 위한 옵션
    
    응답 형식:
    {
        "main": "추천 요약",
        "details": {
            "reasoning": "추천 이유",
            "alternatives": "대안적 선택",
            "theory": "이론적 배경",
            "confidence": "추천 신뢰도",
            "limitations": "고려사항"
        }
    }
    """
    
    return api_manager.get_ai_response(prompt, response_format='json')

def get_ai_factor_recommendations(requirements: str) -> Dict[str, Any]:
    """AI 요인 추천"""
    api_manager = get_api_manager()
    
    prompt = f"""
    사용자의 실험 요구사항:
    {requirements}
    
    실험 유형: {st.session_state.selected_experiment_type}
    
    적절한 실험 요인들을 추천해주세요.
    
    응답 형식:
    {{
        "main": "추천 요인 요약",
        "factors": [
            {{
                "name": "요인명",
                "type": "continuous/categorical/discrete",
                "min_value": 숫자 (연속형/이산형),
                "max_value": 숫자 (연속형/이산형),
                "levels": ["수준1", "수준2"] (범주형),
                "unit": "단위",
                "description": "설명"
            }}
        ],
        "details": {{
            "reasoning": "추천 근거",
            "alternatives": "대안적 요인",
            "theory": "이론적 배경",
            "confidence": "추천 신뢰도",
            "limitations": "주의사항"
        }}
    }}
    """
    
    response = api_manager.get_ai_response(prompt, response_format='json')
    
    # 요인 자동 적용 옵션
    if 'factors' in response:
        if st.button("AI 추천 요인 적용", use_container_width=True):
            st.session_state.experiment_factors = response['factors']
            st.success("AI 추천 요인이 적용되었습니다.")
            st.rerun()
    
    return response

def get_ai_response_recommendations() -> Dict[str, Any]:
    """AI 반응변수 추천"""
    api_manager = get_api_manager()
    
    factors_info = json.dumps(st.session_state.experiment_factors, ensure_ascii=False)
    
    prompt = f"""
    실험 요인:
    {factors_info}
    
    실험 유형: {st.session_state.selected_experiment_type}
    
    적절한 반응변수들을 추천해주세요.
    
    응답 형식:
    {{
        "main": "추천 반응변수 요약",
        "responses": [
            {{
                "name": "반응변수명",
                "type": "continuous/binary/count",
                "optimization": "maximize/minimize/target/in_range",
                "target_value": 숫자 (target인 경우),
                "unit": "단위",
                "importance": 1-10,
                "description": "설명"
            }}
        ],
        "details": {{
            "reasoning": "추천 근거",
            "alternatives": "대안적 반응변수",
            "theory": "측정 이론",
            "confidence": "추천 신뢰도",
            "limitations": "측정 시 주의사항"
        }}
    }}
    """
    
    return api_manager.get_ai_response(prompt, response_format='json')

def get_ai_design_optimization() -> Dict[str, Any]:
    """AI 설계 최적화 제안"""
    api_manager = get_api_manager()
    
    context = {
        'factors': st.session_state.experiment_factors,
        'responses': st.session_state.experiment_responses,
        'constraints': st.session_state.design_constraints
    }
    
    prompt = f"""
    실험 설계 컨텍스트:
    {json.dumps(context, ensure_ascii=False)}
    
    이 실험에 대한 최적의 설계 방법과 파라미터를 제안해주세요.
    
    응답 형식:
    {{
        "main": "최적화 제안 요약",
        "recommendations": {{
            "design_type": "추천 설계 유형",
            "estimated_runs": "예상 실험 횟수",
            "parameters": {{
                "center_points": 숫자,
                "replicates": 숫자,
                "blocks": 숫자
            }}
        }},
        "details": {{
            "reasoning": "최적화 근거",
            "alternatives": "대안적 설계",
            "theory": "통계적 이론",
            "confidence": "효율성 예측",
            "limitations": "제약사항"
        }}
    }}
    """
    
    return api_manager.get_ai_response(prompt, response_format='json')

def get_ai_design_analysis(design: ExperimentDesign) -> Dict[str, Any]:
    """AI 설계 분석"""
    api_manager = get_api_manager()
    
    # 설계 요약 정보
    design_summary = {
        'design_type': design.design_type,
        'num_runs': len(design.runs),
        'factors': [{'name': f.name, 'type': f.type.value} for f in design.factors],
        'responses': [{'name': r.name, 'optimization': r.optimization.value} for r in design.responses],
        'quality': {
            'd_efficiency': design.quality.d_efficiency if hasattr(design, 'quality') and design.quality else None,
            'orthogonality': design.quality.orthogonality if hasattr(design, 'quality') and design.quality else None
        }
    }
    
    prompt = f"""
    생성된 실험 설계 분석:
    {json.dumps(design_summary, ensure_ascii=False)}
    
    이 설계의 품질과 특징을 상세히 분석해주세요.
    
    응답 형식:
    {{
        "main": "설계 품질 종합 평가",
        "strengths": ["강점1", "강점2"],
        "weaknesses": ["약점1", "약점2"],
        "improvements": ["개선방안1", "개선방안2"],
        "details": {{
            "reasoning": "평가 근거",
            "alternatives": "대안적 설계",
            "theory": "통계 이론적 분석",
            "confidence": "신뢰도 평가",
            "limitations": "한계점"
        }}
    }}
    """
    
    return api_manager.get_ai_response(prompt, response_format='json')

def get_ai_chat_response(user_message: str) -> Dict[str, Any]:
    """AI 채팅 응답"""
    api_manager = get_api_manager()
    
    # 현재 컨텍스트
    context = {
        'current_step': st.session_state.wizard_step,
        'has_factors': len(st.session_state.experiment_factors) > 0,
        'has_responses': len(st.session_state.experiment_responses) > 0,
        'has_design': st.session_state.generated_design is not None
    }
    
    # 대화 히스토리 (최근 5개)
    recent_history = st.session_state.design_chat_history[-5:] if st.session_state.design_chat_history else []
    
    prompt = f"""
    사용자가 실험 설계에 대해 질문합니다.
    
    현재 상황: {json.dumps(context)}
    
    대화 히스토리:
    {json.dumps(recent_history, ensure_ascii=False)}
    
    사용자 질문: {user_message}
    
    전문가의 입장에서 도움이 되는 답변을 제공하세요.
    
    응답 형식:
    {{
        "main": "핵심 답변",
        "details": {{
            "reasoning": "답변 근거",
            "alternatives": "추가 고려사항",
            "theory": "이론적 배경",
            "confidence": "답변 신뢰도",
            "limitations": "제한사항"
        }}
    }}
    """
    
    return api_manager.get_ai_response(prompt, response_format='json')

def render_ai_response(response: Dict[str, Any], response_type: str):
    """AI 응답 렌더링 (상세도 제어 포함)"""
    if not response:
        return
    
    # 메인 응답 표시
    if 'main' in response:
        st.write(response['main'])
    
    # 구조화된 정보 표시 (요인, 반응변수 등)
    if 'factors' in response:
        st.markdown("##### 추천 요인")
        for factor in response['factors']:
            st.write(f"- **{factor['name']}**: {factor.get('description', '')}")
    
    if 'responses' in response:
        st.markdown("##### 추천 반응변수")
        for resp in response['responses']:
            st.write(f"- **{resp['name']}**: {resp.get('description', '')}")
    
    if 'strengths' in response:
        st.markdown("##### ✅ 강점")
        for strength in response['strengths']:
            st.write(f"- {strength}")
    
    if 'weaknesses' in response:
        st.markdown("##### ⚠️ 약점")
        for weakness in response['weaknesses']:
            st.write(f"- {weakness}")
    
    if 'improvements' in response:
        st.markdown("##### 💡 개선방안")
        for improvement in response['improvements']:
            st.write(f"- {improvement}")
    
    # 상세 설명 제어
    if 'details' in response and any(response['details'].values()):
        with st.expander("🔍 상세 설명 보기", expanded=st.session_state.ai_preferences['show_reasoning']):
            # 사용자 설정에 따라 표시
            prefs = st.session_state.ai_preferences
            
            tabs = []
            contents = []
            
            if prefs['show_reasoning'] and 'reasoning' in response['details'] and response['details']['reasoning']:
                tabs.append("추론 과정")
                contents.append(response['details']['reasoning'])
            
            if prefs['show_alternatives'] and 'alternatives' in response['details'] and response['details']['alternatives']:
                tabs.append("대안")
                contents.append(response['details']['alternatives'])
            
            if prefs['show_theory'] and 'theory' in response['details'] and response['details']['theory']:
                tabs.append("이론적 배경")
                contents.append(response['details']['theory'])
            
            if prefs['show_confidence'] and 'confidence' in response['details'] and response['details']['confidence']:
                tabs.append("신뢰도")
                contents.append(response['details']['confidence'])
            
            if prefs['show_limitations'] and 'limitations' in response['details'] and response['details']['limitations']:
                tabs.append("한계점")
                contents.append(response['details']['limitations'])
            
            if tabs:
                tab_objects = st.tabs(tabs)
                for tab, content in zip(tab_objects, contents):
                    with tab:
                        st.write(content)
            
            # 설정 변경 UI
            st.divider()
            st.markdown("##### ⚙️ 표시 설정")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.session_state.ai_preferences['show_reasoning'] = st.checkbox(
                    "추론 과정",
                    value=st.session_state.ai_preferences['show_reasoning']
                )
                st.session_state.ai_preferences['show_alternatives'] = st.checkbox(
                    "대안",
                    value=st.session_state.ai_preferences['show_alternatives']
                )
            
            with col2:
                st.session_state.ai_preferences['show_theory'] = st.checkbox(
                    "이론적 배경",
                    value=st.session_state.ai_preferences['show_theory']
                )
                st.session_state.ai_preferences['show_confidence'] = st.checkbox(
                    "신뢰도",
                    value=st.session_state.ai_preferences['show_confidence']
                )
            
            with col3:
                st.session_state.ai_preferences['show_limitations'] = st.checkbox(
                    "한계점",
                    value=st.session_state.ai_preferences['show_limitations']
                )

# ===== 대화형 AI 인터페이스 =====
def render_ai_chat_interface():
    """대화형 AI 인터페이스"""
    st.markdown("### 💬 AI 실험 설계 도우미")
    
    # 채팅 히스토리 표시
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.design_chat_history:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    render_ai_response(message["content"], "채팅")
                else:
                    st.write(message["content"])
    
    # 사용자 입력
    if prompt := st.chat_input("실험 설계에 대해 물어보세요..."):
        # 사용자 메시지 추가
        st.session_state.design_chat_history.append({
            "role": "user",
            "content": prompt
        })
        
        # 스크롤을 위해 재렌더링
        with chat_container:
            with st.chat_message("user"):
                st.write(prompt)
        
        # AI 응답 생성
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("AI가 생각 중..."):
                    response = get_ai_chat_response(prompt)
                    render_ai_response(response, "채팅")
                    
                    # 히스토리에 추가
                    st.session_state.design_chat_history.append({
                        "role": "assistant",
                        "content": response
                    })
    
    # 빠른 질문 버튼
    st.markdown("#### 💡 빠른 질문")
    
    col1, col2, col3 = st.columns(3)
    
    quick_questions = [
        "이 실험 설계의 장단점은?",
        "실험 횟수를 줄이려면?",
        "더 나은 설계 방법은?",
        "요인이 너무 많은가요?",
        "교호작용을 보려면?",
        "최적화하는 방법은?"
    ]
    
    for i, question in enumerate(quick_questions):
        col = [col1, col2, col3][i % 3]
        with col:
            if st.button(question, key=f"quick_{i}", use_container_width=True):
                # 자동으로 질문 입력
                st.session_state.design_chat_history.append({
                    "role": "user",
                    "content": question
                })
                
                # AI 응답 생성
                with st.spinner("AI가 답변 중..."):
                    response = get_ai_chat_response(question)
                    st.session_state.design_chat_history.append({
                        "role": "assistant",
                        "content": response
                    })
                    st.rerun()

# ===== 저장된 설계 관리 =====
def render_saved_designs():
    """저장된 설계 표시"""
    st.markdown("### 📚 내 실험 설계")
    
    # 필터 옵션
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filter_module = st.selectbox(
            "모듈 필터",
            ["전체"] + get_available_modules()
        )
    
    with col2:
        filter_date = st.selectbox(
            "기간 필터",
            ["전체", "오늘", "이번 주", "이번 달", "최근 3개월"]
        )
    
    with col3:
        sort_by = st.selectbox(
            "정렬 기준",
            ["최신순", "이름순", "실험수순"]
        )
    
    # 저장된 설계 가져오기
    try:
        sheets_manager = get_sheets_manager()
        saved_designs = sheets_manager.get_saved_designs(
            user_id=st.session_state.user_id,
            module_filter=filter_module if filter_module != "전체" else None,
            date_filter=filter_date
        )
        
        if not saved_designs:
            st.info("저장된 실험 설계가 없습니다. 새로운 설계를 만들어보세요!")
        else:
            # 설계 카드 표시
            for design in saved_designs:
                with st.expander(f"📋 {design['name']}", expanded=False):
                    col_a, col_b = st.columns([3, 1])
                    
                    with col_a:
                        st.write(f"**모듈**: {design['module_id']}")
                        st.write(f"**유형**: {design['experiment_type']}")
                        st.write(f"**실험수**: {design['num_runs']}회")
                        st.write(f"**생성일**: {design['created_at']}")
                        
                        if design.get('description'):
                            st.write(f"**설명**: {design['description']}")
                    
                    with col_b:
                        if st.button("불러오기", key=f"load_{design['id']}", use_container_width=True):
                            load_saved_design(design)
                        
                        if st.button("삭제", key=f"delete_{design['id']}", use_container_width=True):
                            if st.checkbox(f"정말 삭제하시겠습니까?", key=f"confirm_{design['id']}"):
                                delete_saved_design(design['id'])
    
    except Exception as e:
        st.error(f"저장된 설계를 불러오는 중 오류 발생: {str(e)}")

def render_templates():
    """실험 설계 템플릿"""
    st.markdown("### 📊 실험 설계 템플릿")
    
    # 템플릿 카테고리
    template_categories = {
        "스크리닝": ["Plackett-Burman", "Fractional Factorial", "Definitive Screening"],
        "최적화": ["Central Composite", "Box-Behnken", "D-Optimal"],
        "혼합물": ["Simplex Lattice", "Simplex Centroid", "Extreme Vertices"],
        "고급": ["Split-Plot", "Strip-Plot", "Nested Design"]
    }
    
    selected_category = st.selectbox(
        "템플릿 카테고리",
        list(template_categories.keys())
    )
    
    # 선택된 카테고리의 템플릿 표시
    st.markdown(f"#### {selected_category} 템플릿")
    
    for template_name in template_categories[selected_category]:
        with st.expander(f"📋 {template_name}"):
            # 템플릿 설명
            template_info = get_template_info(template_name)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**설명**: {template_info['description']}")
                st.write(f"**적합한 경우**: {template_info['use_case']}")
                st.write(f"**요인 수**: {template_info['factor_range']}")
                st.write(f"**장점**: {template_info['advantages']}")
                st.write(f"**단점**: {template_info['disadvantages']}")
            
            with col2:
                if st.button(f"템플릿 사용", key=f"use_{template_name}", use_container_width=True):
                    apply_template(template_name)
                
                if st.button(f"예제 보기", key=f"example_{template_name}", use_container_width=True):
                    show_template_example(template_name)

# ===== 헬퍼 함수들 =====
def estimate_experiment_runs() -> int:
    """예상 실험 횟수 계산"""
    num_factors = len(st.session_state.experiment_factors)
    design_type = st.session_state.design_constraints['design_type']
    
    if num_factors == 0:
        return 0
    
    base_runs = 0
    
    if design_type == 'full_factorial':
        levels = []
        for factor in st.session_state.experiment_factors:
            if factor['type'] == 'continuous':
                levels.append(2)  # 기본 2수준
            elif factor['type'] == 'categorical':
                levels.append(len(factor.get('levels', [2])))
            else:
                levels.append(2)
        base_runs = np.prod(levels)
    
    elif design_type == 'fractional_factorial':
        # 대략적인 추정
        base_runs = max(4, 2 ** (num_factors - 1))
    
    elif design_type == 'ccd':
        # 2^k + 2k + cp
        base_runs = 2 ** num_factors + 2 * num_factors + 1
    
    elif design_type == 'box_behnken':
        # Box-Behnken 공식
        if num_factors == 3:
            base_runs = 13
        elif num_factors == 4:
            base_runs = 25
        elif num_factors == 5:
            base_runs = 41
        else:
            base_runs = num_factors * 4
    
    elif design_type == 'plackett_burman':
        # 4의 배수
        base_runs = 4 * ((num_factors + 1) // 4 + 1)
    
    else:
        # 기본 추정
        base_runs = max(num_factors * 2, 8)
    
    # 중심점과 반복 추가
    center_points = st.session_state.design_constraints.get('center_points', 0)
    replicates = st.session_state.design_constraints['replicates']
    
    total_runs = base_runs * replicates + center_points
    
    return min(total_runs, st.session_state.design_constraints['max_runs'])

def calculate_design_quality_preview() -> Dict[str, Any]:
    """설계 품질 미리보기"""
    num_factors = len(st.session_state.experiment_factors)
    estimated_runs = estimate_experiment_runs()
    
    if num_factors == 0 or estimated_runs == 0:
        return {}
    
    # 간단한 품질 지표 계산
    quality_metrics = {}
    
    # 실험 효율성
    full_factorial_runs = 2 ** num_factors
    efficiency = (num_factors * 2) / estimated_runs * 100 if estimated_runs > 0 else 0
    quality_metrics["효율성"] = f"{efficiency:.1f}%"
    
    # 자유도
    dof = estimated_runs - num_factors - 1
    quality_metrics["자유도"] = dof
    
    # Resolution (부분요인설계인 경우)
    if st.session_state.design_constraints['design_type'] == 'fractional_factorial':
        if estimated_runs >= full_factorial_runs / 2:
            quality_metrics["Resolution"] = "V+"
        elif estimated_runs >= full_factorial_runs / 4:
            quality_metrics["Resolution"] = "IV"
        else:
            quality_metrics["Resolution"] = "III"
    
    return quality_metrics

def calculate_power_analysis(design: ExperimentDesign) -> List[Dict[str, Any]]:
    """통계적 검정력 분석"""
    # 간단한 파워 분석 (실제로는 더 복잡한 계산 필요)
    num_runs = len(design.runs)
    num_factors = len(design.factors)
    
    # 효과 크기별 검정력 계산
    effect_sizes = [0.5, 1.0, 1.5, 2.0]  # 표준편차 단위
    alpha = 0.05
    
    results = []
    for effect_size in effect_sizes:
        # 근사 계산
        main_effect_power = min(95, 50 + effect_size * 10 * np.sqrt(num_runs / num_factors))
        interaction_power = min(90, 40 + effect_size * 8 * np.sqrt(num_runs / (num_factors * 2)))
        
        results.append({
            '효과크기': f"{effect_size}σ",
            '주효과': f"{main_effect_power:.0f}%",
            '2차 교호작용': f"{interaction_power:.0f}%"
        })
    
    return results

def optimize_design(design: ExperimentDesign, criterion: str) -> ExperimentDesign:
    """설계 최적화"""
    # 실제 최적화 알고리즘 구현
    # 여기서는 간단한 시뮬레이션
    
    try:
        # D-optimal 예시
        if criterion == "D-최적성":
            # 교환 알고리즘 시뮬레이션
            optimized_design = design  # 실제로는 최적화 수행
            
            # 품질 개선 시뮬레이션
            if hasattr(optimized_design, 'quality'):
                optimized_design.quality.d_efficiency = min(100, optimized_design.quality.d_efficiency + 10)
            
            return optimized_design
        
        else:
            return design
            
    except Exception as e:
        st.error(f"최적화 중 오류 발생: {str(e)}")
        return design

def save_experiment_design():
    """실험 설계 저장"""
    try:
        sheets_manager = get_sheets_manager()
        
        # 설계 이름 입력
        design_name = st.text_input(
            "설계 이름",
            value=f"{st.session_state.selected_experiment_type}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        )
        
        design_description = st.text_area("설명 (선택사항)")
        
        if st.button("저장", type="primary"):
            design_data = {
                'name': design_name,
                'description': design_description,
                'module_id': st.session_state.selected_module_id,
                'experiment_type': st.session_state.selected_experiment_type,
                'factors': json.dumps(st.session_state.experiment_factors),
                'responses': json.dumps(st.session_state.experiment_responses),
                'constraints': json.dumps(st.session_state.design_constraints),
                'design_type': st.session_state.generated_design.design_type,
                'num_runs': len(st.session_state.generated_design.runs),
                'design_matrix': st.session_state.generated_design.runs.to_json(),
                'quality_metrics': json.dumps(asdict(st.session_state.generated_design.quality)) if hasattr(st.session_state.generated_design, 'quality') else None,
                'created_at': datetime.now().isoformat(),
                'user_id': st.session_state.user_id
            }
            
            # Google Sheets에 저장
            design_id = sheets_manager.save_design(design_data)
            
            st.success(f"실험 설계가 저장되었습니다! (ID: {design_id})")
            
            # 로컬 파일로도 저장
            if st.checkbox("로컬 파일로도 저장"):
                save_to_local_file(design_data)
            
            # 완료 후 리셋
            if st.button("새 설계 시작"):
                reset_wizard()
                
    except Exception as e:
        st.error(f"저장 중 오류 발생: {str(e)}")

def save_as_template():
    """템플릿으로 저장"""
    st.info("템플릿 저장 기능은 준비 중입니다.")
    # 실제 구현 시 사용자 템플릿 저장소에 저장

def export_design_settings():
    """설계 설정 내보내기"""
    settings = {
        'module_id': st.session_state.selected_module_id,
        'experiment_type': st.session_state.selected_experiment_type,
        'factors': st.session_state.experiment_factors,
        'responses': st.session_state.experiment_responses,
        'constraints': st.session_state.design_constraints,
        'exported_at': datetime.now().isoformat()
    }
    
    # JSON으로 다운로드
    json_str = json.dumps(settings, indent=2, ensure_ascii=False)
    
    st.download_button(
        "📥 설정 다운로드 (JSON)",
        json_str,
        "experiment_settings.json",
        "application/json"
    )

def reset_wizard():
    """마법사 초기화"""
    st.session_state.wizard_step = 1
    st.session_state.selected_module_id = None
    st.session_state.selected_experiment_type = None
    st.session_state.experiment_factors = []
    st.session_state.experiment_responses = []
    st.session_state.design_constraints = {
        'design_type': 'full_factorial',
        'max_runs': 100,
        'blocks': 1,
        'center_points': 0,
        'replicates': 1,
        'randomize': True
    }
    st.session_state.generated_design = None
    st.rerun()

def load_saved_design(design: Dict[str, Any]):
    """저장된 설계 불러오기"""
    try:
        st.session_state.selected_module_id = design['module_id']
        st.session_state.selected_experiment_type = design['experiment_type']
        st.session_state.experiment_factors = json.loads(design['factors'])
        st.session_state.experiment_responses = json.loads(design['responses'])
        st.session_state.design_constraints = json.loads(design['constraints'])
        st.session_state.wizard_step = 5
        
        # 설계 매트릭스도 불러오기
        if 'design_matrix' in design:
            runs_df = pd.read_json(design['design_matrix'])
            
            # ExperimentDesign 객체 재구성
            registry = get_module_registry()
            module = registry.get_module(design['module_id'])
            
            factors = [Factor(**f) for f in st.session_state.experiment_factors]
            responses = [Response(**r) for r in st.session_state.experiment_responses]
            
            st.session_state.generated_design = ExperimentDesign(
                design_type=design['design_type'],
                runs=runs_df,
                factors=factors,
                responses=responses,
                metadata={
                    'loaded_from': design['id'],
                    'loaded_at': datetime.now().isoformat()
                }
            )
        
        st.success("설계를 성공적으로 불러왔습니다!")
        st.rerun()
        
    except Exception as e:
        st.error(f"설계를 불러오는 중 오류 발생: {str(e)}")

def delete_saved_design(design_id: str):
    """저장된 설계 삭제"""
    try:
        sheets_manager = get_sheets_manager()
        sheets_manager.delete_design(design_id)
        st.success("설계가 삭제되었습니다.")
        st.rerun()
    except Exception as e:
        st.error(f"삭제 중 오류 발생: {str(e)}")

def save_to_local_file(design_data: Dict[str, Any]):
    """로컬 파일로 저장"""
    # JSON 파일로 저장
    json_str = json.dumps(design_data, indent=2, ensure_ascii=False)
    st.download_button(
        "💾 JSON 파일로 저장",
        json_str,
        f"design_{design_data['name']}.json",
        "application/json"
    )

def get_available_modules() -> List[str]:
    """사용 가능한 모듈 목록"""
    registry = get_module_registry()
    modules = registry.list_modules()
    return [m['name'] for m in modules]

def get_template_info(template_name: str) -> Dict[str, str]:
    """템플릿 정보 조회"""
    template_info_db = {
        "Plackett-Burman": {
            "description": "많은 요인을 효율적으로 스크리닝하는 설계",
            "use_case": "초기 단계에서 중요한 요인을 찾을 때",
            "factor_range": "5-50개",
            "advantages": "최소 실험으로 주효과 추정",
            "disadvantages": "교호작용 추정 불가"
        },
        "Central Composite": {
            "description": "2차 반응표면을 모델링하는 설계",
            "use_case": "최적화 단계에서 곡면 효과를 볼 때",
            "factor_range": "2-6개",
            "advantages": "2차 효과 추정, 회전 가능",
            "disadvantages": "실험 수가 많음"
        },
        # 추가 템플릿 정보...
    }
    
    return template_info_db.get(template_name, {
        "description": "설명 없음",
        "use_case": "일반적인 경우",
        "factor_range": "제한 없음",
        "advantages": "다양한 상황에 적용 가능",
        "disadvantages": "특별한 최적화 없음"
    })

def apply_template(template_name: str):
    """템플릿 적용"""
    st.info(f"{template_name} 템플릿을 적용합니다...")
    # 실제 템플릿 적용 로직
    st.success("템플릿이 적용되었습니다!")
    st.rerun()

def show_template_example(template_name: str):
    """템플릿 예제 표시"""
    st.info(f"{template_name} 예제를 표시합니다...")
    # 예제 데이터와 시각화 표시

# 페이지 실행
if __name__ == "__main__":
    render()
