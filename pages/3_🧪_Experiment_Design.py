"""
3_🧪_Experiment_Design.py - AI 기반 실험 설계 페이지
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import asyncio
from pathlib import Path
import sys

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

# 필수 모듈 임포트
from modules.base_module import (
    BaseExperimentModule, Factor, Response, 
    ExperimentDesign, ValidationResult
)
from modules.module_registry import get_module_registry
from utils.common_ui import get_common_ui
from utils.api_manager import get_api_manager
from utils.auth_manager import get_auth_manager
from utils.sheets_manager import get_sheets_manager
from utils.data_processor import DataProcessor

# 페이지 설정
st.set_page_config(
    page_title="실험 설계 - Universal DOE",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 인증 확인
if not st.session_state.get('authenticated', False):
    st.warning("로그인이 필요합니다")
    st.stop()

# ===== AI 설명 상세도 제어 시스템 =====
def initialize_ai_settings():
    """AI 설명 설정 초기화"""
    if 'ai_detail_mode' not in st.session_state:
        st.session_state.ai_detail_mode = 'auto'
    if 'show_ai_details' not in st.session_state:
        st.session_state.show_ai_details = {}
    if 'ai_detail_preferences' not in st.session_state:
        st.session_state.ai_detail_preferences = {
            'show_reasoning': True,
            'show_alternatives': True,
            'show_theory': True,
            'show_confidence': True,
            'show_limitations': True
        }

def render_ai_detail_controller():
    """AI 설명 상세도 컨트롤러 렌더링"""
    with st.sidebar.expander("🧠 AI 설명 설정", expanded=False):
        st.session_state.ai_detail_mode = st.radio(
            "AI 설명 모드",
            ['auto', 'detailed', 'simple', 'custom'],
            format_func=lambda x: {
                'auto': '자동 (레벨 기반)',
                'detailed': '항상 상세히',
                'simple': '항상 간단히',
                'custom': '사용자 정의'
            }[x],
            help="AI 응답의 상세도를 설정합니다"
        )
        
        if st.session_state.ai_detail_mode == 'custom':
            st.write("표시할 항목:")
            st.session_state.ai_detail_preferences['show_reasoning'] = st.checkbox(
                "추론 과정", value=True
            )
            st.session_state.ai_detail_preferences['show_alternatives'] = st.checkbox(
                "대안 검토", value=True
            )
            st.session_state.ai_detail_preferences['show_theory'] = st.checkbox(
                "이론적 배경", value=True
            )
            st.session_state.ai_detail_preferences['show_confidence'] = st.checkbox(
                "신뢰도", value=True
            )
            st.session_state.ai_detail_preferences['show_limitations'] = st.checkbox(
                "한계점", value=True
            )

def should_show_details(context: str = 'general') -> bool:
    """상세 설명을 보여줄지 결정"""
    mode = st.session_state.ai_detail_mode
    
    if mode == 'detailed':
        return True
    elif mode == 'simple':
        return False
    elif mode == 'auto':
        # 사용자 레벨에 따라 자동 결정
        user_level = st.session_state.get('user_level', 'intermediate')
        return user_level in ['beginner', 'intermediate']
    else:  # custom
        return st.session_state.show_ai_details.get(context, True)

def render_ai_response(response: Dict[str, Any], response_type: str = "general"):
    """AI 응답 렌더링 (상세도 제어 포함)"""
    # 핵심 답변 (항상 표시)
    st.markdown(f"### 🤖 {response_type}")
    st.write(response['main'])
    
    # 상세 설명 토글
    show_key = f"show_details_{response_type}"
    show_details = st.session_state.show_ai_details.get(
        show_key, should_show_details(response_type)
    )
    
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button(
            "🔍 상세" if not show_details else "📌 간단히",
            key=f"toggle_{response_type}",
            help="AI 설명의 상세도를 조절합니다"
        ):
            st.session_state.show_ai_details[show_key] = not show_details
            st.rerun()
    
    # 상세 설명 (조건부 표시)
    if show_details and 'details' in response:
        prefs = st.session_state.ai_detail_preferences
        tabs = []
        contents = []
        
        if prefs['show_reasoning'] and 'reasoning' in response['details']:
            tabs.append("추론 과정")
            contents.append(response['details']['reasoning'])
        
        if prefs['show_alternatives'] and 'alternatives' in response['details']:
            tabs.append("대안 검토")
            contents.append(response['details']['alternatives'])
        
        if prefs['show_theory'] and 'theory' in response['details']:
            tabs.append("이론적 배경")
            contents.append(response['details']['theory'])
        
        if prefs['show_confidence'] and 'confidence' in response['details']:
            tabs.append("신뢰도")
            contents.append(response['details']['confidence'])
        
        if prefs['show_limitations'] and 'limitations' in response['details']:
            tabs.append("한계점")
            contents.append(response['details']['limitations'])
        
        if tabs:
            tab_objects = st.tabs(tabs)
            for tab, content in zip(tab_objects, contents):
                with tab:
                    st.write(content)

# ===== 실험 설계 마법사 =====
def render_design_wizard():
    """실험 설계 마법사 UI"""
    st.markdown("## 🎯 실험 설계 마법사")
    
    # 단계 표시
    wizard_step = st.session_state.get('wizard_step', 1)
    progress = wizard_step / 5
    st.progress(progress, text=f"단계 {wizard_step}/5")
    
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
        categories = registry.get_categories()
        selected_category = st.selectbox(
            "연구 분야",
            ['전체'] + categories,
            help="연구 분야를 선택하세요"
        )
        
        # 실험 모듈 선택
        if selected_category == '전체':
            available_modules = modules
        else:
            available_modules = [m for m in modules if m['category'] == selected_category]
        
        module_names = [m['name'] for m in available_modules]
        selected_module_name = st.selectbox(
            "실험 모듈",
            module_names,
            help="사용할 실험 모듈을 선택하세요"
        )
        
        # 선택된 모듈 정보 표시
        selected_module = next((m for m in available_modules if m['name'] == selected_module_name), None)
        if selected_module:
            st.info(f"📝 {selected_module['description']}")
            
            # 실험 유형 선택
            module = registry.get_module(selected_module['id'])
            if module:
                experiment_types = module.get_experiment_types()
                selected_type = st.selectbox(
                    "실험 유형",
                    experiment_types,
                    help="수행할 실험 유형을 선택하세요"
                )
                
                # 세션에 저장
                st.session_state.selected_module_id = selected_module['id']
                st.session_state.selected_experiment_type = selected_type
    
    with col2:
        # AI 추천
        if st.button("🤖 AI 추천 받기", use_container_width=True):
            with st.spinner("AI가 분석 중..."):
                ai_recommendation = get_ai_module_recommendation()
                render_ai_response(ai_recommendation, "모듈 추천")
    
    # 다음 단계 버튼
    col1, col2, col3 = st.columns([1, 1, 1])
    with col3:
        if st.button("다음 단계 →", type="primary", use_container_width=True):
            if 'selected_module_id' in st.session_state:
                st.session_state.wizard_step = 2
                st.rerun()
            else:
                st.error("실험 모듈을 선택해주세요")

def render_step2_factors():
    """Step 2: 실험 요인 설정"""
    st.markdown("### Step 2: 실험 요인 설정")
    
    # 선택된 모듈 가져오기
    registry = get_module_registry()
    module = registry.get_module(st.session_state.selected_module_id)
    
    if not module:
        st.error("모듈을 불러올 수 없습니다")
        return
    
    # 기본 요인 가져오기
    default_factors = module.get_factors(st.session_state.selected_experiment_type)
    
    # 요인 편집 UI
    st.write("#### 실험 요인")
    
    # 세션에서 요인 목록 관리
    if 'experiment_factors' not in st.session_state:
        st.session_state.experiment_factors = [f.model_dump() for f in default_factors]
    
    # 요인 테이블 표시
    factor_df = pd.DataFrame(st.session_state.experiment_factors)
    
    # 데이터 에디터로 편집
    edited_factors = st.data_editor(
        factor_df,
        num_rows="dynamic",
        column_config={
            "name": st.column_config.TextColumn("요인명", required=True),
            "type": st.column_config.SelectboxColumn(
                "유형",
                options=["continuous", "categorical"],
                required=True
            ),
            "unit": st.column_config.TextColumn("단위"),
            "min_value": st.column_config.NumberColumn("최소값"),
            "max_value": st.column_config.NumberColumn("최대값"),
            "levels": st.column_config.TextColumn("수준 (콤마 구분)"),
            "description": st.column_config.TextColumn("설명")
        },
        hide_index=True,
        use_container_width=True
    )
    
    st.session_state.experiment_factors = edited_factors.to_dict('records')
    
    # AI 요인 추천
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🤖 AI 요인 추천", use_container_width=True):
            with st.spinner("AI가 분석 중..."):
                ai_factors = get_ai_factor_recommendation()
                render_ai_response(ai_factors, "요인 추천")
    
    # 요인 상관관계 시각화
    if len(st.session_state.experiment_factors) > 1:
        with st.expander("📊 요인 관계 시각화"):
            render_factor_correlation_plot()
    
    # 네비게이션 버튼
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← 이전 단계", use_container_width=True):
            st.session_state.wizard_step = 1
            st.rerun()
    with col3:
        if st.button("다음 단계 →", type="primary", use_container_width=True):
            if validate_factors():
                st.session_state.wizard_step = 3
                st.rerun()

def render_step3_responses():
    """Step 3: 반응변수 설정"""
    st.markdown("### Step 3: 반응변수 설정")
    
    # 선택된 모듈에서 기본 반응변수 가져오기
    registry = get_module_registry()
    module = registry.get_module(st.session_state.selected_module_id)
    default_responses = module.get_responses(st.session_state.selected_experiment_type)
    
    # 세션에서 반응변수 관리
    if 'experiment_responses' not in st.session_state:
        st.session_state.experiment_responses = [r.model_dump() for r in default_responses]
    
    st.write("#### 반응변수")
    
    # 반응변수 편집
    response_df = pd.DataFrame(st.session_state.experiment_responses)
    
    edited_responses = st.data_editor(
        response_df,
        num_rows="dynamic",
        column_config={
            "name": st.column_config.TextColumn("반응변수명", required=True),
            "unit": st.column_config.TextColumn("단위"),
            "goal": st.column_config.SelectboxColumn(
                "목표",
                options=["maximize", "minimize", "target"],
                required=True
            ),
            "target_value": st.column_config.NumberColumn("목표값"),
            "description": st.column_config.TextColumn("설명")
        },
        hide_index=True,
        use_container_width=True
    )
    
    st.session_state.experiment_responses = edited_responses.to_dict('records')
    
    # 반응변수 중요도 설정
    if len(st.session_state.experiment_responses) > 1:
        st.write("#### 반응변수 중요도")
        importance_cols = st.columns(len(st.session_state.experiment_responses))
        
        for i, (col, response) in enumerate(zip(importance_cols, st.session_state.experiment_responses)):
            with col:
                importance = st.slider(
                    response['name'],
                    0.0, 1.0, 0.5,
                    key=f"importance_{i}"
                )
                st.session_state.experiment_responses[i]['importance'] = importance
    
    # AI 반응변수 추천
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🤖 AI 반응변수 추천", use_container_width=True):
            with st.spinner("AI가 분석 중..."):
                ai_responses = get_ai_response_recommendation()
                render_ai_response(ai_responses, "반응변수 추천")
    
    # 네비게이션
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← 이전 단계", use_container_width=True):
            st.session_state.wizard_step = 2
            st.rerun()
    with col3:
        if st.button("다음 단계 →", type="primary", use_container_width=True):
            if validate_responses():
                st.session_state.wizard_step = 4
                st.rerun()

def render_step4_constraints():
    """Step 4: 제약조건 및 설계 옵션"""
    st.markdown("### Step 4: 제약조건 및 설계 옵션")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### 실험 제약조건")
        
        # 실험 횟수 제약
        max_runs = st.number_input(
            "최대 실험 횟수",
            min_value=1,
            max_value=1000,
            value=20,
            help="수행 가능한 최대 실험 횟수"
        )
        
        # 시간 제약
        time_constraint = st.number_input(
            "실험 기간 (일)",
            min_value=1,
            max_value=365,
            value=30,
            help="전체 실험을 완료해야 하는 기간"
        )
        
        # 예산 제약
        budget_constraint = st.number_input(
            "예산 (만원)",
            min_value=0,
            value=1000,
            step=100,
            help="실험에 사용 가능한 총 예산"
        )
        
        # 블록화 설정
        use_blocking = st.checkbox("블록화 사용", help="시간이나 재료 배치에 따른 변동 제어")
        if use_blocking:
            block_factor = st.selectbox(
                "블록 요인",
                ["시간", "재료 배치", "실험자", "장비"],
                help="블록화할 요인 선택"
            )
    
    with col2:
        st.write("#### 설계 옵션")
        
        # 설계 유형
        design_types = {
            'full_factorial': '완전요인설계',
            'fractional_factorial': '부분요인설계',
            'central_composite': '중심합성설계',
            'box_behnken': 'Box-Behnken 설계',
            'plackett_burman': 'Plackett-Burman 설계',
            'latin_hypercube': 'Latin Hypercube 설계',
            'd_optimal': 'D-최적 설계'
        }
        
        selected_design = st.selectbox(
            "설계 유형",
            list(design_types.keys()),
            format_func=lambda x: design_types[x],
            help="실험 설계 방법 선택"
        )
        
        # 중심점
        if selected_design in ['central_composite', 'box_behnken']:
            center_points = st.number_input(
                "중심점 수",
                min_value=0,
                max_value=10,
                value=3,
                help="중심점 반복 횟수"
            )
        
        # 랜덤화
        randomize = st.checkbox("실험 순서 랜덤화", value=True)
        
        # 반복 실험
        replicates = st.number_input(
            "반복 횟수",
            min_value=1,
            max_value=5,
            value=1,
            help="각 실험 조건의 반복 횟수"
        )
    
    # 제약조건 저장
    st.session_state.design_constraints = {
        'max_runs': max_runs,
        'time_constraint': time_constraint,
        'budget_constraint': budget_constraint,
        'use_blocking': use_blocking,
        'block_factor': block_factor if use_blocking else None,
        'design_type': selected_design,
        'center_points': center_points if selected_design in ['central_composite', 'box_behnken'] else 0,
        'randomize': randomize,
        'replicates': replicates
    }
    
    # AI 최적화 추천
    if st.button("🤖 AI 최적화 추천", use_container_width=True):
        with st.spinner("AI가 최적 설계를 찾는 중..."):
            ai_optimization = get_ai_design_optimization()
            render_ai_response(ai_optimization, "설계 최적화")
    
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
        st.write("#### 설계 요약")
        
        # 실험 정보
        st.info(f"""
        **실험 모듈**: {st.session_state.selected_module_id}  
        **실험 유형**: {st.session_state.selected_experiment_type}  
        **요인 수**: {len(st.session_state.experiment_factors)}  
        **반응변수 수**: {len(st.session_state.experiment_responses)}  
        **설계 유형**: {st.session_state.design_constraints['design_type']}  
        **최대 실험 횟수**: {st.session_state.design_constraints['max_runs']}
        """)
        
        # 요인 요약
        with st.expander("📊 요인 상세"):
            factor_summary = pd.DataFrame(st.session_state.experiment_factors)
            st.dataframe(factor_summary, use_container_width=True)
        
        # 반응변수 요약
        with st.expander("📈 반응변수 상세"):
            response_summary = pd.DataFrame(st.session_state.experiment_responses)
            st.dataframe(response_summary, use_container_width=True)
    
    with col2:
        # 실험 설계 생성
        st.write("#### 작업")
        
        if st.button("🚀 실험 설계 생성", type="primary", use_container_width=True):
            generate_experiment_design()
        
        if st.button("💾 템플릿으로 저장", use_container_width=True):
            save_as_template()
        
        if st.button("📤 내보내기", use_container_width=True):
            export_design_settings()
    
    # 생성된 설계 표시
    if 'generated_design' in st.session_state:
        st.divider()
        render_generated_design()
    
    # 네비게이션
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← 이전 단계", use_container_width=True):
            st.session_state.wizard_step = 4
            st.rerun()
    with col3:
        if st.button("✅ 완료", type="primary", use_container_width=True):
            if 'generated_design' in st.session_state:
                save_experiment_design()
                st.success("실험 설계가 저장되었습니다!")
                st.balloons()

# ===== 실험 설계 생성 =====
def generate_experiment_design():
    """실험 설계 생성"""
    with st.spinner("실험 설계 생성 중..."):
        try:
            # 모듈 가져오기
            registry = get_module_registry()
            module = registry.get_module(st.session_state.selected_module_id)
            
            # 입력 데이터 준비
            inputs = {
                'design_type': st.session_state.design_constraints['design_type'],
                'factors': st.session_state.experiment_factors,
                'responses': st.session_state.experiment_responses,
                'constraints': st.session_state.design_constraints,
                'n_levels': 2,  # 기본값
                'n_samples': st.session_state.design_constraints['max_runs']
            }
            
            # 설계 생성
            design = module.generate_design(inputs)
            
            # 검증
            validation = module.validate_design(design)
            
            if validation.is_valid:
                st.session_state.generated_design = design
                st.success("실험 설계가 성공적으로 생성되었습니다!")
                
                # AI 설계 분석
                ai_analysis = analyze_design_with_ai(design)
                render_ai_response(ai_analysis, "설계 분석")
            else:
                st.error("설계 검증 실패:")
                for error in validation.errors:
                    st.error(f"- {error}")
                for warning in validation.warnings:
                    st.warning(f"- {warning}")
                    
        except Exception as e:
            st.error(f"설계 생성 중 오류 발생: {str(e)}")

def render_generated_design():
    """생성된 설계 표시"""
    design = st.session_state.generated_design
    
    st.markdown("### 🎯 생성된 실험 설계")
    
    # 탭 구성
    tabs = st.tabs(["실험 런 테이블", "설계 공간 시각화", "통계적 속성", "파워 분석"])
    
    with tabs[0]:
        render_run_table(design)
    
    with tabs[1]:
        render_design_space_visualization(design)
    
    with tabs[2]:
        render_statistical_properties(design)
    
    with tabs[3]:
        render_power_analysis(design)

def render_run_table(design: ExperimentDesign):
    """실험 런 테이블 표시"""
    st.write("#### 실험 런 테이블")
    
    # 편집 가능한 데이터 에디터
    edited_runs = st.data_editor(
        design.runs,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=False,
        column_config={
            col: st.column_config.NumberColumn(
                col,
                help=f"Factor: {col}",
                format="%.3f"
            ) for col in design.runs.columns if col not in [r.name for r in design.responses]
        }
    )
    
    # 변경사항 저장
    if not design.runs.equals(edited_runs):
        st.session_state.generated_design.runs = edited_runs
        st.info("변경사항이 감지되었습니다. 저장하려면 '완료' 버튼을 클릭하세요.")
    
    # 실험 순서 재정렬
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("🔀 순서 랜덤화"):
            design.runs = design.runs.sample(frac=1).reset_index(drop=True)
            st.session_state.generated_design = design
            st.rerun()
    
    with col2:
        if st.button("📊 블록별 정렬"):
            if 'Block' in design.runs.columns:
                design.runs = design.runs.sort_values('Block').reset_index(drop=True)
                st.session_state.generated_design = design
                st.rerun()
    
    # 다운로드 옵션
    col1, col2 = st.columns(2)
    with col1:
        csv = design.runs.to_csv(index=True)
        st.download_button(
            "📥 CSV 다운로드",
            csv,
            "experiment_design.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        excel_buffer = design.export_design('excel')
        st.download_button(
            "📥 Excel 다운로드",
            excel_buffer,
            "experiment_design.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

def render_design_space_visualization(design: ExperimentDesign):
    """설계 공간 시각화"""
    st.write("#### 설계 공간 시각화")
    
    # 연속형 요인만 추출
    continuous_factors = [f for f in design.factors if f.type == 'continuous']
    
    if len(continuous_factors) < 2:
        st.warning("시각화를 위해서는 최소 2개의 연속형 요인이 필요합니다.")
        return
    
    # 시각화 옵션
    col1, col2 = st.columns([3, 1])
    with col2:
        viz_type = st.selectbox(
            "시각화 유형",
            ["2D 산점도", "3D 산점도", "평행 좌표", "히트맵"]
        )
    
    if viz_type == "2D 산점도":
        # 요인 선택
        factor_names = [f.name for f in continuous_factors]
        x_factor = st.selectbox("X축 요인", factor_names, index=0)
        y_factor = st.selectbox("Y축 요인", factor_names, index=1 if len(factor_names) > 1 else 0)
        
        # 색상 매핑 (반응변수가 있는 경우)
        color_by = None
        if any(col in design.runs.columns for col in [r.name for r in design.responses]):
            available_responses = [r.name for r in design.responses if r.name in design.runs.columns]
            if available_responses:
                color_by = st.selectbox("색상 매핑", [None] + available_responses)
        
        # 2D 산점도 생성
        fig = px.scatter(
            design.runs,
            x=x_factor,
            y=y_factor,
            color=color_by,
            title="실험 설계 공간",
            labels={x_factor: f"{x_factor}", y_factor: f"{y_factor}"},
            hover_data=design.runs.columns
        )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "3D 산점도":
        if len(continuous_factors) < 3:
            st.warning("3D 시각화를 위해서는 최소 3개의 연속형 요인이 필요합니다.")
        else:
            factor_names = [f.name for f in continuous_factors]
            x_factor = st.selectbox("X축 요인", factor_names, index=0)
            y_factor = st.selectbox("Y축 요인", factor_names, index=1)
            z_factor = st.selectbox("Z축 요인", factor_names, index=2)
            
            fig = go.Figure(data=[go.Scatter3d(
                x=design.runs[x_factor],
                y=design.runs[y_factor],
                z=design.runs[z_factor],
                mode='markers',
                marker=dict(
                    size=8,
                    color=design.runs.index,
                    colorscale='Viridis',
                    showscale=True
                ),
                text=[f"Run {i}" for i in design.runs.index],
                hovertemplate='<b>Run %{text}</b><br>' +
                             f'{x_factor}: %{{x:.3f}}<br>' +
                             f'{y_factor}: %{{y:.3f}}<br>' +
                             f'{z_factor}: %{{z:.3f}}<br>' +
                             '<extra></extra>'
            )])
            
            fig.update_layout(
                title="3D 실험 설계 공간",
                scene=dict(
                    xaxis_title=x_factor,
                    yaxis_title=y_factor,
                    zaxis_title=z_factor
                ),
                height=700
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "평행 좌표":
        # 정규화된 데이터 준비
        factor_cols = [f.name for f in continuous_factors]
        normalized_data = design.runs[factor_cols].copy()
        
        for col in factor_cols:
            min_val = normalized_data[col].min()
            max_val = normalized_data[col].max()
            if max_val > min_val:
                normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
        
        # 평행 좌표 플롯
        fig = go.Figure(data=go.Parcoords(
            dimensions=[dict(
                label=col,
                values=normalized_data[col],
                range=[0, 1]
            ) for col in factor_cols],
            line=dict(
                color=design.runs.index,
                colorscale='Viridis'
            )
        ))
        
        fig.update_layout(
            title="평행 좌표 플롯",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:  # 히트맵
        # 상관관계 히트맵
        factor_cols = [f.name for f in continuous_factors]
        corr_matrix = design.runs[factor_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title="요인 상관관계 히트맵",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_statistical_properties(design: ExperimentDesign):
    """통계적 속성 분석"""
    st.write("#### 통계적 속성")
    
    # 설계 매트릭스 속성
    factor_cols = [f.name for f in design.factors if f.type == 'continuous']
    if not factor_cols:
        st.warning("연속형 요인이 없어 통계적 속성을 계산할 수 없습니다.")
        return
    
    X = design.runs[factor_cols].values
    
    # 정규화
    X_norm = (X - X.mean(axis=0)) / X.std(axis=0)
    
    # 정보 행렬
    info_matrix = X_norm.T @ X_norm
    
    # D-효율성
    det_info = np.linalg.det(info_matrix)
    d_efficiency = (det_info / len(X)) ** (1/len(factor_cols))
    
    # A-효율성
    try:
        inv_info = np.linalg.inv(info_matrix)
        a_efficiency = len(factor_cols) / np.trace(inv_info)
    except:
        a_efficiency = 0
    
    # G-효율성
    g_efficiency = calculate_g_efficiency(X_norm)
    
    # 결과 표시
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("D-효율성", f"{d_efficiency:.3f}", 
                 help="설계의 D-최적성 측도 (1에 가까울수록 좋음)")
    
    with col2:
        st.metric("A-효율성", f"{a_efficiency:.3f}",
                 help="평균 분산의 역수 (높을수록 좋음)")
    
    with col3:
        st.metric("G-효율성", f"{g_efficiency:.3f}",
                 help="최대 예측 분산 (낮을수록 좋음)")
    
    # 상세 분석
    with st.expander("상세 통계 분석"):
        # VIF (분산팽창지수)
        st.write("**분산팽창지수 (VIF)**")
        vif_data = calculate_vif(design.runs[factor_cols])
        st.dataframe(vif_data, use_container_width=True)
        
        # 직교성 검사
        st.write("**직교성 검사**")
        orthogonality = check_orthogonality(X_norm)
        if orthogonality:
            st.success("설계가 직교성을 만족합니다.")
        else:
            st.warning("설계가 완전히 직교하지 않습니다.")
        
        # 균형성 검사
        st.write("**균형성 검사**")
        balance = check_balance(design)
        st.info(balance)

def render_power_analysis(design: ExperimentDesign):
    """파워 분석"""
    st.write("#### 파워 분석")
    
    # 파워 분석 설정
    col1, col2 = st.columns(2)
    
    with col1:
        alpha = st.slider("유의수준 (α)", 0.01, 0.10, 0.05, 0.01)
        effect_size = st.slider("효과 크기", 0.1, 2.0, 0.5, 0.1,
                               help="감지하고자 하는 표준화된 효과 크기")
        
    with col2:
        sigma = st.number_input("오차 표준편차 추정치", 
                               min_value=0.1, value=1.0, step=0.1)
        desired_power = st.slider("목표 검정력", 0.70, 0.95, 0.80, 0.05)
    
    # 파워 계산
    n_runs = len(design.runs)
    n_factors = len([f for f in design.factors if f.type == 'continuous'])
    
    # 간단한 파워 근사 (실제로는 더 복잡한 계산 필요)
    import scipy.stats as stats
    
    # 비중심 모수
    lambda_param = n_runs * (effect_size ** 2) / (2 * sigma ** 2)
    
    # F-분포의 임계값
    df1 = n_factors
    df2 = n_runs - n_factors - 1
    f_crit = stats.f.ppf(1 - alpha, df1, df2)
    
    # 검정력 계산
    power = 1 - stats.ncf.cdf(f_crit, df1, df2, lambda_param)
    
    # 결과 표시
    st.metric("계산된 검정력", f"{power:.3f}",
             delta=f"{power - desired_power:.3f}" if power >= desired_power else f"{power - desired_power:.3f}")
    
    if power < desired_power:
        st.warning(f"목표 검정력 {desired_power:.2f}에 도달하지 못했습니다.")
        
        # 필요한 실험 횟수 계산
        required_n = calculate_required_sample_size(
            alpha, desired_power, effect_size, sigma, n_factors
        )
        st.info(f"목표 검정력을 달성하려면 약 {required_n}회의 실험이 필요합니다.")
    else:
        st.success("목표 검정력을 달성했습니다!")
    
    # 파워 곡선
    st.write("**파워 곡선**")
    effect_sizes = np.linspace(0.1, 2.0, 50)
    powers = []
    
    for es in effect_sizes:
        lambda_p = n_runs * (es ** 2) / (2 * sigma ** 2)
        pwr = 1 - stats.ncf.cdf(f_crit, df1, df2, lambda_p)
        powers.append(pwr)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=effect_sizes,
        y=powers,
        mode='lines',
        name='파워 곡선'
    ))
    
    fig.add_hline(y=desired_power, line_dash="dash", 
                  annotation_text=f"목표 검정력 ({desired_power})")
    fig.add_vline(x=effect_size, line_dash="dash",
                  annotation_text=f"현재 효과 크기 ({effect_size})")
    
    fig.update_layout(
        title="효과 크기에 따른 검정력",
        xaxis_title="효과 크기",
        yaxis_title="검정력",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ===== 도우미 함수들 =====
def validate_factors() -> bool:
    """요인 검증"""
    factors = st.session_state.experiment_factors
    
    if not factors:
        st.error("최소 1개 이상의 요인이 필요합니다.")
        return False
    
    for factor in factors:
        if not factor.get('name'):
            st.error("모든 요인은 이름이 있어야 합니다.")
            return False
        
        if factor['type'] == 'continuous':
            if factor.get('min_value', 0) >= factor.get('max_value', 1):
                st.error(f"요인 '{factor['name']}'의 최소값이 최대값보다 크거나 같습니다.")
                return False
        elif factor['type'] == 'categorical':
            if not factor.get('levels'):
                st.error(f"범주형 요인 '{factor['name']}'의 수준이 정의되지 않았습니다.")
                return False
    
    return True

def validate_responses() -> bool:
    """반응변수 검증"""
    responses = st.session_state.experiment_responses
    
    if not responses:
        st.warning("반응변수가 정의되지 않았습니다. 계속하시겠습니까?")
        return True
    
    for response in responses:
        if not response.get('name'):
            st.error("모든 반응변수는 이름이 있어야 합니다.")
            return False
        
        if response['goal'] == 'target' and response.get('target_value') is None:
            st.error(f"반응변수 '{response['name']}'의 목표값이 설정되지 않았습니다.")
            return False
    
    return True

def calculate_vif(df: pd.DataFrame) -> pd.DataFrame:
    """분산팽창지수 계산"""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    vif_data = pd.DataFrame()
    vif_data["요인"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
    vif_data["상태"] = vif_data["VIF"].apply(
        lambda x: "양호" if x < 5 else "주의" if x < 10 else "문제"
    )
    
    return vif_data

def check_orthogonality(X: np.ndarray) -> bool:
    """직교성 검사"""
    corr_matrix = np.corrcoef(X.T)
    np.fill_diagonal(corr_matrix, 0)
    return np.max(np.abs(corr_matrix)) < 0.01

def check_balance(design: ExperimentDesign) -> str:
    """균형성 검사"""
    categorical_factors = [f for f in design.factors if f.type == 'categorical']
    
    if not categorical_factors:
        return "범주형 요인이 없습니다."
    
    balance_info = []
    for factor in categorical_factors:
        if factor.name in design.runs.columns:
            counts = design.runs[factor.name].value_counts()
            if counts.std() / counts.mean() < 0.1:
                balance_info.append(f"{factor.name}: 균형")
            else:
                balance_info.append(f"{factor.name}: 불균형")
    
    return ", ".join(balance_info)

def calculate_g_efficiency(X: np.ndarray) -> float:
    """G-효율성 계산"""
    try:
        H = X @ np.linalg.inv(X.T @ X) @ X.T
        max_leverage = np.max(np.diag(H))
        return X.shape[1] / (X.shape[0] * max_leverage)
    except:
        return 0

def calculate_required_sample_size(alpha: float, power: float, 
                                 effect_size: float, sigma: float, 
                                 n_factors: int) -> int:
    """필요한 표본 크기 계산"""
    from scipy import stats
    
    # 이진 탐색으로 필요한 n 찾기
    n_min, n_max = n_factors + 2, 1000
    
    while n_min < n_max:
        n_mid = (n_min + n_max) // 2
        
        df1 = n_factors
        df2 = n_mid - n_factors - 1
        
        if df2 <= 0:
            n_min = n_mid + 1
            continue
        
        f_crit = stats.f.ppf(1 - alpha, df1, df2)
        lambda_param = n_mid * (effect_size ** 2) / (2 * sigma ** 2)
        calculated_power = 1 - stats.ncf.cdf(f_crit, df1, df2, lambda_param)
        
        if calculated_power < power:
            n_min = n_mid + 1
        else:
            n_max = n_mid
    
    return n_min

def render_factor_correlation_plot():
    """요인 상관관계 플롯"""
    factors = st.session_state.experiment_factors
    continuous_factors = [f for f in factors if f['type'] == 'continuous']
    
    if len(continuous_factors) < 2:
        return
    
    # 가상의 상관관계 매트릭스 생성 (실제로는 과거 데이터나 도메인 지식 기반)
    n = len(continuous_factors)
    corr_matrix = np.eye(n)
    
    # 일부 상관관계 추가 (예시)
    if n > 1:
        corr_matrix[0, 1] = corr_matrix[1, 0] = 0.3
    if n > 2:
        corr_matrix[0, 2] = corr_matrix[2, 0] = -0.2
        corr_matrix[1, 2] = corr_matrix[2, 1] = 0.1
    
    factor_names = [f['name'] for f in continuous_factors]
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=factor_names,
        y=factor_names,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix, 2),
        texttemplate='%{text}',
        textfont={"size": 12}
    ))
    
    fig.update_layout(
        title="예상 요인 상관관계",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def save_as_template():
    """현재 설정을 템플릿으로 저장"""
    template_name = st.text_input("템플릿 이름", "내 실험 템플릿")
    
    if st.button("저장", type="primary"):
        template = {
            'name': template_name,
            'module_id': st.session_state.selected_module_id,
            'experiment_type': st.session_state.selected_experiment_type,
            'factors': st.session_state.experiment_factors,
            'responses': st.session_state.experiment_responses,
            'constraints': st.session_state.design_constraints,
            'created_at': datetime.now().isoformat()
        }
        
        # 템플릿 저장 (실제로는 데이터베이스에)
        if 'saved_templates' not in st.session_state:
            st.session_state.saved_templates = []
        
        st.session_state.saved_templates.append(template)
        st.success(f"템플릿 '{template_name}'이 저장되었습니다!")

def export_design_settings():
    """설계 설정 내보내기"""
    export_data = {
        'module_id': st.session_state.selected_module_id,
        'experiment_type': st.session_state.selected_experiment_type,
        'factors': st.session_state.experiment_factors,
        'responses': st.session_state.experiment_responses,
        'constraints': st.session_state.design_constraints,
        'exported_at': datetime.now().isoformat()
    }
    
    json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
    
    st.download_button(
        "💾 JSON 다운로드",
        json_str,
        "experiment_settings.json",
        "application/json"
    )

def save_experiment_design():
    """실험 설계 저장"""
    if 'current_project' not in st.session_state:
        st.error("프로젝트를 먼저 선택해주세요.")
        return
    
    try:
        sheets_manager = get_sheets_manager()
        design_data = {
            'project_id': st.session_state.current_project['id'],
            'design_type': st.session_state.design_constraints['design_type'],
            'factors': json.dumps(st.session_state.experiment_factors),
            'responses': json.dumps(st.session_state.experiment_responses),
            'design_matrix': st.session_state.generated_design.runs.to_json(),
            'constraints': json.dumps(st.session_state.design_constraints),
            'created_at': datetime.now().isoformat(),
            'created_by': st.session_state.user['email']
        }
        
        sheets_manager.save_experiment_design(design_data)
        
    except Exception as e:
        st.error(f"저장 중 오류 발생: {str(e)}")

# ===== AI 관련 함수들 =====
def get_ai_module_recommendation() -> Dict[str, Any]:
    """AI 모듈 추천"""
    api_manager = get_api_manager()
    
    prompt = f"""
    사용자의 연구 분야와 목적에 맞는 실험 모듈을 추천해주세요.
    
    현재 프로젝트 정보:
    - 분야: {st.session_state.get('project_field', '일반')}
    - 목적: {st.session_state.get('project_goal', '최적화')}
    
    응답 형식:
    {{
        "main": "추천 모듈과 간단한 이유",
        "details": {{
            "reasoning": "왜 이 모듈을 추천하는지 상세 설명",
            "alternatives": "다른 옵션들과 각각의 장단점",
            "theory": "이 추천의 이론적 배경",
            "confidence": "추천 신뢰도와 근거",
            "limitations": "주의사항 및 한계점"
        }}
    }}
    """
    
    response = api_manager.get_ai_response(prompt, response_format='json')
    return response

def get_ai_factor_recommendation() -> Dict[str, Any]:
    """AI 요인 추천"""
    api_manager = get_api_manager()
    
    current_factors = st.session_state.experiment_factors
    
    prompt = f"""
    현재 실험의 요인 설정을 검토하고 개선사항을 제안해주세요.
    
    현재 요인:
    {json.dumps(current_factors, indent=2)}
    
    실험 유형: {st.session_state.selected_experiment_type}
    
    응답 형식:
    {{
        "main": "핵심 제안사항 요약",
        "details": {{
            "reasoning": "각 제안의 이유와 중요성",
            "alternatives": "추가로 고려할 수 있는 요인들",
            "theory": "요인 선택의 이론적 근거",
            "confidence": "제안의 신뢰도 (0-100%)",
            "limitations": "제안의 한계점과 주의사항"
        }}
    }}
    """
    
    response = api_manager.get_ai_response(prompt, response_format='json')
    return response

def get_ai_response_recommendation() -> Dict[str, Any]:
    """AI 반응변수 추천"""
    api_manager = get_api_manager()
    
    prompt = f"""
    현재 실험의 반응변수 설정을 검토하고 개선사항을 제안해주세요.
    
    현재 반응변수:
    {json.dumps(st.session_state.experiment_responses, indent=2)}
    
    실험 목적: {st.session_state.get('experiment_goal', '최적화')}
    
    응답 형식:
    {{
        "main": "핵심 제안사항",
        "details": {{
            "reasoning": "제안 이유",
            "alternatives": "다른 반응변수 옵션",
            "theory": "측정 이론",
            "confidence": "신뢰도",
            "limitations": "측정의 한계"
        }}
    }}
    """
    
    response = api_manager.get_ai_response(prompt, response_format='json')
    return response

def get_ai_design_optimization() -> Dict[str, Any]:
    """AI 설계 최적화"""
    api_manager = get_api_manager()
    
    prompt = f"""
    실험 설계를 최적화하기 위한 제안을 해주세요.
    
    요인 수: {len(st.session_state.experiment_factors)}
    반응변수 수: {len(st.session_state.experiment_responses)}
    제약조건: {json.dumps(st.session_state.design_constraints, indent=2)}
    
    응답 형식:
    {{
        "main": "최적 설계 방법과 예상 실험 횟수",
        "details": {{
            "reasoning": "이 설계를 추천하는 이유",
            "alternatives": "다른 설계 옵션들",
            "theory": "설계의 통계적 배경",
            "confidence": "효율성 예측",
            "limitations": "설계의 한계점"
        }}
    }}
    """
    
    response = api_manager.get_ai_response(prompt, response_format='json')
    return response

def analyze_design_with_ai(design: ExperimentDesign) -> Dict[str, Any]:
    """AI로 생성된 설계 분석"""
    api_manager = get_api_manager()
    
    # 설계 요약 정보
    design_summary = {
        'n_runs': len(design.runs),
        'n_factors': len(design.factors),
        'n_responses': len(design.responses),
        'design_type': design.design_type
    }
    
    prompt = f"""
    생성된 실험 설계를 분석해주세요.
    
    설계 요약:
    {json.dumps(design_summary, indent=2)}
    
    응답 형식:
    {{
        "main": "설계의 주요 특징과 품질 평가",
        "details": {{
            "reasoning": "설계 품질 평가의 근거",
            "alternatives": "개선 가능한 부분",
            "theory": "통계적 최적성 분석",
            "confidence": "설계 효율성 점수",
            "limitations": "주의해야 할 점"
        }}
    }}
    """
    
    response = api_manager.get_ai_response(prompt, response_format='json')
    return response

# ===== 대화형 AI 인터페이스 =====
def render_ai_chat_interface():
    """대화형 AI 인터페이스"""
    st.markdown("### 💬 AI 실험 설계 도우미")
    
    # 채팅 히스토리 초기화
    if 'design_chat_history' not in st.session_state:
        st.session_state.design_chat_history = []
    
    # 채팅 히스토리 표시
    for message in st.session_state.design_chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_ai_response(message["content"], "대화")
            else:
                st.write(message["content"])
    
    # 사용자 입력
    if prompt := st.chat_input("실험 설계에 대해 물어보세요..."):
        # 사용자 메시지 추가
        st.session_state.design_chat_history.append({
            "role": "user",
            "content": prompt
        })
        
        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("AI가 생각 중..."):
                response = get_ai_chat_response(prompt)
                render_ai_response(response, "대화")
                
                # 히스토리에 추가
                st.session_state.design_chat_history.append({
                    "role": "assistant",
                    "content": response
                })

def get_ai_chat_response(prompt: str) -> Dict[str, Any]:
    """채팅 응답 생성"""
    api_manager = get_api_manager()
    
    # 컨텍스트 구성
    context = {
        'current_step': st.session_state.get('wizard_step', 0),
        'factors': st.session_state.get('experiment_factors', []),
        'responses': st.session_state.get('experiment_responses', []),
        'constraints': st.session_state.get('design_constraints', {})
    }
    
    system_prompt = f"""
    당신은 실험 설계 전문가입니다. 현재 사용자는 실험을 설계하고 있습니다.
    
    현재 상황:
    {json.dumps(context, indent=2)}
    
    사용자의 질문에 정확하고 도움이 되는 답변을 제공하세요.
    
    응답 형식:
    {{
        "main": "핵심 답변",
        "details": {{
            "reasoning": "답변의 근거",
            "alternatives": "다른 옵션",
            "theory": "이론적 배경",
            "confidence": "확신도",
            "limitations": "한계점"
        }}
    }}
    """
    
    response = api_manager.get_ai_response(
        prompt, 
        system_prompt=system_prompt,
        response_format='json'
    )
    
    return response

# ===== 메인 렌더 함수 =====
def render():
    """메인 페이지 렌더링"""
    # 초기화
    initialize_ai_settings()
    
    # UI 컴포넌트
    ui = get_common_ui()
    
    # 헤더
    ui.render_header("실험 설계", "AI와 함께 최적의 실험을 설계하세요", "🧪")
    
    # AI 설정 컨트롤러 (사이드바)
    render_ai_detail_controller()
    
    # 메인 컨텐츠
    tab1, tab2, tab3 = st.tabs(["🎯 설계 마법사", "💬 AI 대화", "📚 저장된 설계"])
    
    with tab1:
        render_design_wizard()
    
    with tab2:
        render_ai_chat_interface()
    
    with tab3:
        render_saved_designs()

def render_saved_designs():
    """저장된 설계 목록"""
    st.markdown("### 📚 저장된 실험 설계")
    
    # 필터
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        filter_module = st.selectbox("모듈 필터", ["전체"] + get_available_modules())
    with col2:
        filter_date = st.date_input("날짜 필터", value=None)
    with col3:
        if st.button("🔄 새로고침"):
            st.rerun()
    
    # 저장된 설계 불러오기
    try:
        sheets_manager = get_sheets_manager()
        saved_designs = sheets_manager.get_saved_designs(
            user_id=st.session_state.user['id'],
            module_filter=filter_module if filter_module != "전체" else None,
            date_filter=filter_date
        )
        
        if saved_designs:
            for design in saved_designs:
                with st.expander(f"📋 {design['name']} - {design['created_at']}"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**모듈**: {design['module_id']}")
                        st.write(f"**요인 수**: {design['n_factors']}")
                        st.write(f"**실험 횟수**: {design['n_runs']}")
                    
                    with col2:
                        if st.button("불러오기", key=f"load_{design['id']}"):
                            load_saved_design(design)
                        if st.button("삭제", key=f"delete_{design['id']}"):
                            delete_saved_design(design['id'])
        else:
            ui.render_empty_state("저장된 설계가 없습니다", "📭")
            
    except Exception as e:
        st.error(f"설계 목록을 불러오는 중 오류 발생: {str(e)}")

def get_available_modules() -> List[str]:
    """사용 가능한 모듈 목록"""
    registry = get_module_registry()
    modules = registry.list_modules()
    return [m['name'] for m in modules]

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
            
            st.session_state.generated_design = ExperimentDesign(
                design_type=design['design_type'],
                runs=runs_df,
                factors=[Factor(**f) for f in st.session_state.experiment_factors],
                responses=[Response(**r) for r in st.session_state.experiment_responses],
                metadata={'loaded_from': design['id']}
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

# 페이지 실행
if __name__ == "__main__":
    render()
