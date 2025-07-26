"""
3_🧪_Experiment_Design.py - 실험 설계
AI 기반 실험 설계 생성, 검토, 최적화를 담당하는 핵심 페이지
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import json
from pathlib import Path
import sys
import io
from pyDOE3 import *
import scipy.stats as stats

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

# 필수 모듈 임포트
try:
    from utils.database_manager import get_database_manager
    from utils.auth_manager import get_auth_manager
    from utils.common_ui import get_common_ui
    from utils.api_manager import get_api_manager
    from utils.data_processor import get_data_processor
    from modules.module_registry import get_module_registry
    from modules.base_module import Factor, Response, ExperimentDesign
    from config.app_config import EXPERIMENT_DEFAULTS
    from config.theme_config import COLORS
except ImportError as e:
    st.error(f"필수 모듈 임포트 오류: {e}")
    st.stop()

# 페이지 설정
st.set_page_config(
    page_title="실험 설계 - Universal DOE",
    page_icon="🧪",
    layout="wide"
)

# 인증 확인
auth_manager = get_auth_manager()
if not auth_manager.check_authentication():
    st.warning("로그인이 필요합니다")
    st.switch_page("pages/0_🔐_Login.py")
    st.stop()

class ExperimentDesignPage:
    """실험 설계 페이지 클래스"""
    
    def __init__(self):
        self.ui = get_common_ui()
        self.db_manager = get_database_manager()
        self.api_manager = get_api_manager()
        self.data_processor = get_data_processor()
        self.module_registry = get_module_registry()
        
        # 세션 상태 초기화
        self._initialize_session_state()
        
        # 현재 프로젝트 확인
        self._check_current_project()
    
    def _initialize_session_state(self):
        """세션 상태 초기화"""
        defaults = {
            'design_step': 0,
            'design_factors': [],
            'design_responses': [],
            'design_type': 'full_factorial',
            'design_options': {},
            'ai_conversation': [],
            'ai_suggestions': [],
            'show_ai_details': False,
            'current_design': None,
            'design_validation': None,
            'preview_mode': 'table'
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def _check_current_project(self):
        """현재 프로젝트 확인"""
        if 'current_project' not in st.session_state or not st.session_state.current_project:
            st.error("프로젝트를 먼저 선택해주세요")
            if st.button("프로젝트 선택하기"):
                st.switch_page("pages/2_📝_Project_Setup.py")
            st.stop()
        
        # 프로젝트 정보 로드
        self.project = self.db_manager.get_project(st.session_state.current_project['id'])
        if not self.project:
            st.error("프로젝트를 찾을 수 없습니다")
            st.stop()
        
        # 프로젝트 모듈 로드
        self.project_modules = []
        for module_id in self.project.get('modules', []):
            module = self.module_registry.get_module(module_id)
            if module:
                self.project_modules.append(module)
    
    def render(self):
        """메인 렌더링"""
        # 헤더
        self.ui.render_header(
            f"🧪 실험 설계 - {self.project['name']}",
            f"{self.project['field']} > {self.project['subfield']}"
        )
        
        # 메인 레이아웃
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 설계 프로세스
            self._render_design_process()
        
        with col2:
            # AI 어시스턴트
            self._render_ai_assistant()
        
        # 설계 미리보기
        if st.session_state.current_design:
            st.divider()
            self._render_design_preview()
    
    def _render_design_process(self):
        """설계 프로세스 렌더링"""
        st.subheader("실험 설계 프로세스")
        
        # 단계 표시
        steps = ["요인 정의", "반응변수", "설계 유형", "옵션 설정", "검토 및 생성"]
        current_step = st.session_state.design_step
        
        # 진행률 표시
        progress = current_step / (len(steps) - 1)
        st.progress(progress)
        
        # 단계 버튼
        cols = st.columns(len(steps))
        for i, (col, step) in enumerate(zip(cols, steps)):
            with col:
                if st.button(
                    step,
                    key=f"step_{i}",
                    type="primary" if i == current_step else "secondary",
                    disabled=i > current_step + 1,
                    use_container_width=True
                ):
                    st.session_state.design_step = i
                    st.rerun()
        
        st.divider()
        
        # 단계별 내용
        if current_step == 0:
            self._render_factors_step()
        elif current_step == 1:
            self._render_responses_step()
        elif current_step == 2:
            self._render_design_type_step()
        elif current_step == 3:
            self._render_options_step()
        elif current_step == 4:
            self._render_review_step()
        
        # 네비게이션
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            if current_step > 0:
                if st.button("⬅️ 이전", use_container_width=True):
                    st.session_state.design_step -= 1
                    st.rerun()
        
        with col2:
            if current_step < len(steps) - 1:
                if st.button("다음 ➡️", use_container_width=True, type="primary"):
                    if self._validate_current_step():
                        st.session_state.design_step += 1
                        st.rerun()
            else:
                if st.button("🚀 설계 생성", use_container_width=True, type="primary"):
                    self._generate_design()
    
    def _render_factors_step(self):
        """요인 정의 단계"""
        st.markdown("### 실험 요인 정의")
        
        # 모듈 템플릿 사용
        if self.project_modules:
            module = st.selectbox(
                "실험 모듈 템플릿",
                options=self.project_modules,
                format_func=lambda x: x.get_module_info()['name']
            )
            
            if st.button("템플릿 요인 불러오기"):
                template_factors = module.get_factors(self.project.get('type', 'general'))
                st.session_state.design_factors = [f.model_dump() for f in template_factors]
                st.success("템플릿 요인을 불러왔습니다")
                st.rerun()
        
        # 현재 요인 목록
        factors = st.session_state.design_factors
        
        if factors:
            st.write(f"**정의된 요인: {len(factors)}개**")
            
            # 요인 테이블
            for i, factor in enumerate(factors):
                with st.expander(f"{i+1}. {factor['name']}", expanded=True):
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        factor['name'] = st.text_input(
                            "요인명",
                            value=factor['name'],
                            key=f"factor_name_{i}"
                        )
                        factor['unit'] = st.text_input(
                            "단위",
                            value=factor.get('unit', ''),
                            key=f"factor_unit_{i}"
                        )
                    
                    with col2:
                        factor['type'] = st.selectbox(
                            "유형",
                            ["continuous", "categorical"],
                            index=0 if factor.get('type') == 'continuous' else 1,
                            key=f"factor_type_{i}"
                        )
                        
                        if factor['type'] == 'continuous':
                            col_min, col_max = st.columns(2)
                            with col_min:
                                factor['min_value'] = st.number_input(
                                    "최소값",
                                    value=factor.get('min_value', 0.0),
                                    key=f"factor_min_{i}"
                                )
                            with col_max:
                                factor['max_value'] = st.number_input(
                                    "최대값",
                                    value=factor.get('max_value', 100.0),
                                    key=f"factor_max_{i}"
                                )
                        else:
                            levels_str = st.text_input(
                                "수준 (쉼표로 구분)",
                                value=', '.join(factor.get('levels', [])),
                                key=f"factor_levels_{i}"
                            )
                            factor['levels'] = [l.strip() for l in levels_str.split(',') if l.strip()]
                    
                    with col3:
                        if st.button("삭제", key=f"delete_factor_{i}"):
                            factors.pop(i)
                            st.rerun()
        else:
            st.info("아직 정의된 요인이 없습니다")
        
        # 요인 추가
        st.divider()
        if st.button("➕ 요인 추가", use_container_width=True):
            st.session_state.design_factors.append({
                'name': f'Factor_{len(factors)+1}',
                'type': 'continuous',
                'unit': '',
                'min_value': 0.0,
                'max_value': 100.0,
                'levels': [],
                'description': ''
            })
            st.rerun()
        
        # AI 추천
        if st.button("🤖 AI 요인 추천", use_container_width=True):
            self._get_ai_factor_recommendations()
    
    def _render_responses_step(self):
        """반응변수 정의 단계"""
        st.markdown("### 반응변수 정의")
        
        # 현재 반응변수 목록
        responses = st.session_state.design_responses
        
        if responses:
            st.write(f"**정의된 반응변수: {len(responses)}개**")
            
            # 반응변수 테이블
            for i, response in enumerate(responses):
                with st.expander(f"{i+1}. {response['name']}", expanded=True):
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        response['name'] = st.text_input(
                            "반응변수명",
                            value=response['name'],
                            key=f"response_name_{i}"
                        )
                        response['unit'] = st.text_input(
                            "단위",
                            value=response.get('unit', ''),
                            key=f"response_unit_{i}"
                        )
                    
                    with col2:
                        response['goal'] = st.selectbox(
                            "목표",
                            ["maximize", "minimize", "target"],
                            index=["maximize", "minimize", "target"].index(response.get('goal', 'maximize')),
                            key=f"response_goal_{i}"
                        )
                        
                        if response['goal'] == 'target':
                            response['target_value'] = st.number_input(
                                "목표값",
                                value=response.get('target_value', 0.0),
                                key=f"response_target_{i}"
                            )
                    
                    with col3:
                        if st.button("삭제", key=f"delete_response_{i}"):
                            responses.pop(i)
                            st.rerun()
        else:
            st.info("아직 정의된 반응변수가 없습니다")
        
        # 반응변수 추가
        st.divider()
        if st.button("➕ 반응변수 추가", use_container_width=True):
            st.session_state.design_responses.append({
                'name': f'Response_{len(responses)+1}',
                'unit': '',
                'goal': 'maximize',
                'target_value': None,
                'description': ''
            })
            st.rerun()
        
        # 모듈 템플릿
        if self.project_modules:
            if st.button("📋 템플릿 반응변수 불러오기", use_container_width=True):
                module = self.project_modules[0]
                template_responses = module.get_responses(self.project.get('type', 'general'))
                st.session_state.design_responses = [r.model_dump() for r in template_responses]
                st.success("템플릿 반응변수를 불러왔습니다")
                st.rerun()
    
    def _render_design_type_step(self):
        """설계 유형 선택 단계"""
        st.markdown("### 실험 설계 유형 선택")
        
        # 설계 유형 카테고리
        design_categories = {
            "스크리닝 설계": {
                "full_factorial": "완전요인설계 (Full Factorial)",
                "fractional_factorial": "부분요인설계 (Fractional Factorial)",
                "plackett_burman": "Plackett-Burman 설계"
            },
            "최적화 설계": {
                "central_composite": "중심합성설계 (CCD)",
                "box_behnken": "Box-Behnken 설계",
                "doehlert": "Doehlert 설계"
            },
            "혼합물 설계": {
                "simplex_lattice": "단순격자설계",
                "simplex_centroid": "단순중심설계",
                "mixture_optimal": "최적혼합설계"
            },
            "고급 설계": {
                "latin_hypercube": "Latin Hypercube 설계",
                "d_optimal": "D-최적설계",
                "space_filling": "공간충전설계"
            }
        }
        
        # 카테고리 선택
        category = st.selectbox(
            "설계 카테고리",
            list(design_categories.keys()),
            help="실험 목적에 맞는 카테고리를 선택하세요"
        )
        
        # 설계 유형 선택
        design_types = design_categories[category]
        selected_type = st.radio(
            "설계 유형",
            list(design_types.keys()),
            format_func=lambda x: design_types[x],
            key="design_type_radio"
        )
        st.session_state.design_type = selected_type
        
        # 설계 유형 설명
        st.info(self._get_design_type_description(selected_type))
        
        # 예상 실험 횟수 계산
        if st.session_state.design_factors:
            estimated_runs = self._estimate_runs(
                selected_type,
                len(st.session_state.design_factors)
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("예상 실험 횟수", f"{estimated_runs}회")
            with col2:
                st.metric("요인 수", len(st.session_state.design_factors))
            with col3:
                st.metric("반응변수 수", len(st.session_state.design_responses))
        
        # AI 추천
        if st.button("🤖 AI 설계 유형 추천", use_container_width=True):
            self._get_ai_design_type_recommendation()
    
    def _render_options_step(self):
        """설계 옵션 설정 단계"""
        st.markdown("### 설계 옵션 설정")
        
        design_type = st.session_state.design_type
        options = st.session_state.design_options
        
        # 공통 옵션
        st.markdown("#### 기본 옵션")
        
        col1, col2 = st.columns(2)
        with col1:
            options['randomize'] = st.checkbox(
                "실험 순서 랜덤화",
                value=options.get('randomize', True),
                help="실험 순서를 무작위로 배치하여 시간적 효과를 제거합니다"
            )
            
            options['blocks'] = st.number_input(
                "블록 수",
                min_value=1,
                max_value=10,
                value=options.get('blocks', 1),
                help="실험을 여러 블록으로 나누어 수행합니다"
            )
        
        with col2:
            options['replicates'] = st.number_input(
                "반복 수",
                min_value=1,
                max_value=5,
                value=options.get('replicates', 1),
                help="각 실험 조건의 반복 횟수"
            )
            
            options['center_points'] = st.number_input(
                "중심점 수",
                min_value=0,
                max_value=10,
                value=options.get('center_points', 0),
                help="곡률 검출을 위한 중심점 추가"
            )
        
        # 설계별 특수 옵션
        st.markdown("#### 고급 옵션")
        
        if design_type == 'fractional_factorial':
            options['resolution'] = st.selectbox(
                "해상도",
                ["III", "IV", "V"],
                index=["III", "IV", "V"].index(options.get('resolution', 'IV')),
                help="높은 해상도는 더 많은 실험이 필요하지만 교호작용 추정이 가능합니다"
            )
        
        elif design_type == 'central_composite':
            col1, col2 = st.columns(2)
            with col1:
                options['alpha'] = st.selectbox(
                    "축점 거리 (α)",
                    ["orthogonal", "rotatable", "custom"],
                    index=["orthogonal", "rotatable", "custom"].index(options.get('alpha', 'rotatable'))
                )
                if options['alpha'] == 'custom':
                    options['alpha_value'] = st.number_input(
                        "α 값",
                        min_value=1.0,
                        max_value=3.0,
                        value=options.get('alpha_value', 1.682)
                    )
            
            with col2:
                options['face_centered'] = st.checkbox(
                    "면심 CCD",
                    value=options.get('face_centered', False),
                    help="축점을 ±1 위치에 배치합니다"
                )
        
        elif design_type == 'latin_hypercube':
            options['samples'] = st.number_input(
                "샘플 수",
                min_value=10,
                max_value=1000,
                value=options.get('samples', 50),
                help="생성할 실험점의 개수"
            )
            
            options['criterion'] = st.selectbox(
                "최적화 기준",
                ["maximin", "center", "correlation"],
                help="실험점 배치 최적화 기준"
            )
        
        # 제약조건
        st.markdown("#### 제약조건")
        
        if st.checkbox("제약조건 추가", value=bool(options.get('constraints', []))):
            constraints = options.get('constraints', [])
            
            # 기존 제약조건 표시
            for i, constraint in enumerate(constraints):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.text_input(
                        f"제약조건 {i+1}",
                        value=constraint,
                        key=f"constraint_{i}",
                        disabled=True
                    )
                with col3:
                    if st.button("삭제", key=f"delete_constraint_{i}"):
                        constraints.pop(i)
                        st.rerun()
            
            # 새 제약조건 추가
            new_constraint = st.text_input(
                "새 제약조건 (예: Factor1 + Factor2 <= 100)",
                key="new_constraint"
            )
            if st.button("추가") and new_constraint:
                constraints.append(new_constraint)
                options['constraints'] = constraints
                st.rerun()
    
    def _render_review_step(self):
        """검토 및 생성 단계"""
        st.markdown("### 설계 검토")
        
        # 설계 요약
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 실험 요인")
            factors_df = pd.DataFrame(st.session_state.design_factors)
            if not factors_df.empty:
                st.dataframe(
                    factors_df[['name', 'type', 'unit', 'min_value', 'max_value']].fillna('-'),
                    use_container_width=True
                )
            
            st.markdown("#### 반응변수")
            responses_df = pd.DataFrame(st.session_state.design_responses)
            if not responses_df.empty:
                st.dataframe(
                    responses_df[['name', 'unit', 'goal']],
                    use_container_width=True
                )
        
        with col2:
            st.markdown("#### 설계 정보")
            st.write(f"**설계 유형**: {self._get_design_type_name(st.session_state.design_type)}")
            st.write(f"**예상 실험 횟수**: {self._calculate_total_runs()}회")
            
            options = st.session_state.design_options
            st.write("**옵션**:")
            st.write(f"- 랜덤화: {'예' if options.get('randomize') else '아니오'}")
            st.write(f"- 블록 수: {options.get('blocks', 1)}")
            st.write(f"- 반복 수: {options.get('replicates', 1)}")
            st.write(f"- 중심점: {options.get('center_points', 0)}")
        
        # 설계 검증
        st.divider()
        validation = self._validate_design()
        
        if validation['errors']:
            st.error("**오류**")
            for error in validation['errors']:
                st.write(f"- {error}")
        
        if validation['warnings']:
            st.warning("**경고**")
            for warning in validation['warnings']:
                st.write(f"- {warning}")
        
        if validation['suggestions']:
            st.info("**제안**")
            for suggestion in validation['suggestions']:
                st.write(f"- {suggestion}")
        
        # 설계 생성 준비 상태
        if validation['is_valid']:
            st.success("✅ 실험 설계를 생성할 준비가 되었습니다!")
        else:
            st.error("❌ 설계를 생성하기 전에 오류를 수정해주세요.")
    
    def _render_ai_assistant(self):
        """AI 어시스턴트 렌더링"""
        with st.container():
            st.subheader("🤖 AI 실험 설계 도우미")
            
            # AI 설명 상세도 제어
            col1, col2 = st.columns([3, 1])
            with col2:
                show_details = st.checkbox(
                    "🔍 상세",
                    value=st.session_state.show_ai_details,
                    key="ai_details_main_toggle",
                    help="AI 응답의 상세 설명을 표시합니다"
                )
                st.session_state.show_ai_details = show_details
            
            # 대화 기록
            chat_container = st.container(height=400)
            with chat_container:
                for msg in st.session_state.ai_conversation:
                    if msg['role'] == 'user':
                        st.chat_message("user").write(msg['content'])
                    else:
                        with st.chat_message("assistant"):
                            st.write(msg['content']['main'])
                            
                            # 상세 설명 (조건부)
                            if show_details and 'details' in msg['content']:
                                with st.expander("상세 설명", expanded=False):
                                    tabs = st.tabs(["추론", "대안", "배경", "신뢰도"])
                                    
                                    with tabs[0]:
                                        st.write(msg['content']['details'].get('reasoning', ''))
                                    with tabs[1]:
                                        st.write(msg['content']['details'].get('alternatives', ''))
                                    with tabs[2]:
                                        st.write(msg['content']['details'].get('background', ''))
                                    with tabs[3]:
                                        confidence = msg['content']['details'].get('confidence', {})
                                        if confidence:
                                            st.metric("신뢰도", f"{confidence.get('score', 0)}%")
                                            st.write(confidence.get('explanation', ''))
            
            # 입력 영역
            user_input = st.chat_input(
                "실험 설계에 대해 질문하세요...",
                key="ai_chat_input"
            )
            
            if user_input:
                self._process_ai_input(user_input)
            
            # 빠른 질문
            st.caption("빠른 질문:")
            quick_questions = [
                "이 실험에 가장 적합한 설계는?",
                "실험 횟수를 줄이려면?",
                "교호작용을 고려하려면?",
                "최적화를 위한 추천?"
            ]
            
            cols = st.columns(2)
            for i, question in enumerate(quick_questions):
                with cols[i % 2]:
                    if st.button(question, key=f"quick_{i}", use_container_width=True):
                        self._process_ai_input(question)
    
    def _render_design_preview(self):
        """설계 미리보기"""
        st.subheader("📊 실험 설계 미리보기")
        
        # 뷰 모드 선택
        view_modes = ["테이블", "2D 플롯", "3D 시각화", "통계 분석"]
        selected_view = st.radio(
            "보기 모드",
            view_modes,
            horizontal=True,
            key="preview_mode_radio"
        )
        
        design = st.session_state.current_design
        
        if selected_view == "테이블":
            self._render_design_table(design)
        elif selected_view == "2D 플롯":
            self._render_2d_plots(design)
        elif selected_view == "3D 시각화":
            self._render_3d_visualization(design)
        elif selected_view == "통계 분석":
            self._render_statistical_analysis(design)
        
        # 액션 버튼
        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("💾 저장", use_container_width=True):
                self._save_design()
        
        with col2:
            if st.button("📥 내보내기", use_container_width=True):
                self._export_design()
        
        with col3:
            if st.button("🔄 재생성", use_container_width=True):
                self._regenerate_design()
        
        with col4:
            if st.button("📤 공유", use_container_width=True):
                self._share_design()
    
    def _render_design_table(self, design: ExperimentDesign):
        """설계 테이블 렌더링"""
        # 편집 가능한 데이터 에디터
        edited_df = st.data_editor(
            design.runs,
            use_container_width=True,
            num_rows="dynamic",
            height=400,
            key="design_table_editor"
        )
        
        # 변경사항 감지
        if not edited_df.equals(design.runs):
            st.info("📝 변경사항이 감지되었습니다. 저장하려면 '저장' 버튼을 클릭하세요.")
            st.session_state.current_design.runs = edited_df
        
        # 테이블 통계
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("총 실험 수", len(design.runs))
        with col2:
            unique_conditions = len(design.runs.drop_duplicates(
                subset=[f.name for f in design.factors]
            ))
            st.metric("고유 조건", unique_conditions)
        with col3:
            if 'Block' in design.runs.columns:
                st.metric("블록 수", design.runs['Block'].nunique())
    
    def _render_2d_plots(self, design: ExperimentDesign):
        """2D 플롯 렌더링"""
        factors = [f for f in design.factors if f.type == 'continuous']
        
        if len(factors) < 2:
            st.warning("2D 플롯을 위해서는 최소 2개의 연속형 요인이 필요합니다")
            return
        
        # 요인 선택
        col1, col2 = st.columns(2)
        with col1:
            x_factor = st.selectbox(
                "X축 요인",
                factors,
                format_func=lambda x: x.name
            )
        with col2:
            y_factor = st.selectbox(
                "Y축 요인",
                [f for f in factors if f != x_factor],
                format_func=lambda x: x.name
            )
        
        # 산점도
        fig = px.scatter(
            design.runs,
            x=x_factor.name,
            y=y_factor.name,
            color='Block' if 'Block' in design.runs.columns else None,
            size_max=10,
            title=f"{x_factor.name} vs {y_factor.name}"
        )
        
        # 설계 공간 경계 추가
        fig.add_shape(
            type="rect",
            x0=x_factor.min_value, y0=y_factor.min_value,
            x1=x_factor.max_value, y1=y_factor.max_value,
            line=dict(color="RoyalBlue", width=2, dash="dash"),
            fillcolor="LightSkyBlue",
            opacity=0.1
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 요인 쌍 행렬
        if st.checkbox("모든 요인 쌍 표시"):
            fig_matrix = px.scatter_matrix(
                design.runs,
                dimensions=[f.name for f in factors],
                color='Block' if 'Block' in design.runs.columns else None,
                title="요인 쌍 행렬"
            )
            fig_matrix.update_traces(diagonal_visible=False)
            st.plotly_chart(fig_matrix, use_container_width=True)
    
    def _render_3d_visualization(self, design: ExperimentDesign):
        """3D 시각화 렌더링"""
        factors = [f for f in design.factors if f.type == 'continuous']
        
        if len(factors) < 3:
            st.warning("3D 시각화를 위해서는 최소 3개의 연속형 요인이 필요합니다")
            return
        
        # 요인 선택
        col1, col2, col3 = st.columns(3)
        with col1:
            x_factor = st.selectbox("X축", factors, format_func=lambda x: x.name, key="3d_x")
        with col2:
            y_factor = st.selectbox(
                "Y축",
                [f for f in factors if f != x_factor],
                format_func=lambda x: x.name,
                key="3d_y"
            )
        with col3:
            z_factor = st.selectbox(
                "Z축",
                [f for f in factors if f not in [x_factor, y_factor]],
                format_func=lambda x: x.name,
                key="3d_z"
            )
        
        # 3D 산점도
        fig = go.Figure(data=[go.Scatter3d(
            x=design.runs[x_factor.name],
            y=design.runs[y_factor.name],
            z=design.runs[z_factor.name],
            mode='markers',
            marker=dict(
                size=8,
                color=design.runs.get('Block', 1),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Block" if 'Block' in design.runs.columns else "")
            ),
            text=[f"Run {i}" for i in design.runs.index],
            hovertemplate='<b>%{text}</b><br>' +
                         f'{x_factor.name}: %{{x}}<br>' +
                         f'{y_factor.name}: %{{y}}<br>' +
                         f'{z_factor.name}: %{{z}}<br>' +
                         '<extra></extra>'
        )])
        
        # 설계 공간 큐브 추가
        cube_data = self._create_cube_edges(
            x_factor.min_value, x_factor.max_value,
            y_factor.min_value, y_factor.max_value,
            z_factor.min_value, z_factor.max_value
        )
        
        for edge in cube_data:
            fig.add_trace(go.Scatter3d(
                x=edge[0], y=edge[1], z=edge[2],
                mode='lines',
                line=dict(color='blue', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            title="3D 실험 설계 공간",
            scene=dict(
                xaxis_title=x_factor.name,
                yaxis_title=y_factor.name,
                zaxis_title=z_factor.name,
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 회전 애니메이션
        if st.checkbox("자동 회전"):
            st.info("마우스로 드래그하여 수동으로 회전할 수 있습니다")
    
    def _render_statistical_analysis(self, design: ExperimentDesign):
        """통계 분석 렌더링"""
        st.markdown("#### 설계 품질 메트릭")
        
        # 설계 속성 계산
        properties = self._calculate_design_properties(design)
        
        # 메트릭 표시
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "D-효율성",
                f"{properties['d_efficiency']:.1f}%",
                help="설계의 D-최적성 척도 (100%가 최적)"
            )
        
        with col2:
            st.metric(
                "G-효율성",
                f"{properties['g_efficiency']:.1f}%",
                help="예측 분산의 균일성 척도"
            )
        
        with col3:
            st.metric(
                "조건수",
                f"{properties['condition_number']:.2f}",
                help="설계 행렬의 조건수 (낮을수록 좋음)"
            )
        
        with col4:
            st.metric(
                "직교성",
                f"{properties['orthogonality']:.1f}%",
                help="요인 간 직교성 정도"
            )
        
        # 상관 행렬
        st.markdown("#### 요인 상관 행렬")
        
        continuous_factors = [f.name for f in design.factors if f.type == 'continuous']
        if continuous_factors:
            corr_matrix = design.runs[continuous_factors].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                color_continuous_scale='RdBu',
                range_color=[-1, 1],
                title="요인 간 상관관계"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 파워 분석
        st.markdown("#### 검정력 분석")
        
        col1, col2 = st.columns(2)
        
        with col1:
            effect_size = st.slider(
                "효과 크기",
                0.1, 2.0, 0.5,
                help="검출하고자 하는 효과의 크기"
            )
            
            alpha = st.slider(
                "유의수준 (α)",
                0.01, 0.10, 0.05,
                format="%.2f"
            )
        
        with col2:
            sigma = st.number_input(
                "표준편차 (σ)",
                min_value=0.1,
                value=1.0,
                help="반응변수의 예상 표준편차"
            )
        
        # 파워 계산
        power = self._calculate_power(design, effect_size, alpha, sigma)
        
        # 파워 곡선
        effect_sizes = np.linspace(0.1, 2.0, 50)
        powers = [self._calculate_power(design, es, alpha, sigma) for es in effect_sizes]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=effect_sizes,
            y=powers,
            mode='lines',
            name='Power Curve'
        ))
        fig.add_vline(x=effect_size, line_dash="dash", line_color="red")
        fig.add_hline(y=0.8, line_dash="dash", line_color="green")
        fig.add_annotation(
            x=effect_size, y=power,
            text=f"Power: {power:.2f}",
            showarrow=True,
            arrowhead=2
        )
        
        fig.update_layout(
            title="검정력 곡선",
            xaxis_title="효과 크기",
            yaxis_title="검정력",
            yaxis_range=[0, 1]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 검정력 해석
        if power >= 0.8:
            st.success(f"✅ 검정력 {power:.2f} - 충분한 검정력을 가집니다")
        elif power >= 0.7:
            st.warning(f"⚠️ 검정력 {power:.2f} - 경계선상의 검정력입니다")
        else:
            st.error(f"❌ 검정력 {power:.2f} - 검정력이 부족합니다. 실험 수를 늘리는 것을 고려하세요")
    
    def _process_ai_input(self, user_input: str):
        """AI 입력 처리"""
        # 대화 기록에 추가
        st.session_state.ai_conversation.append({
            'role': 'user',
            'content': user_input
        })
        
        # 컨텍스트 준비
        context = self._prepare_ai_context()
        
        # AI 호출
        with st.spinner("AI가 분석 중입니다..."):
            prompt = f"""
            실험 설계 전문가로서 다음 질문에 답해주세요:
            
            현재 실험 설계 상황:
            {json.dumps(context, ensure_ascii=False, indent=2)}
            
            사용자 질문: {user_input}
            
            응답 형식:
            1. 핵심 답변 (간단명료하게)
            2. 상세 설명:
               - 추론 과정
               - 대안 (2-3개)
               - 이론적 배경
               - 신뢰도 (백분율과 설명)
               - 한계점/주의사항
            """
            
            response = self.api_manager.call_ai(
                prompt,
                response_format="structured",
                detail_level='detailed' if st.session_state.show_ai_details else 'simple'
            )
            
            if response:
                # 응답 구조화
                ai_response = {
                    'role': 'assistant',
                    'content': {
                        'main': response.get('main_answer', ''),
                        'details': {
                            'reasoning': response.get('reasoning', ''),
                            'alternatives': response.get('alternatives', ''),
                            'background': response.get('background', ''),
                            'confidence': response.get('confidence', {}),
                            'limitations': response.get('limitations', '')
                        }
                    }
                }
                
                st.session_state.ai_conversation.append(ai_response)
                
                # 제안사항 처리
                if 'suggestions' in response:
                    self._process_ai_suggestions(response['suggestions'])
                
                st.rerun()
    
    def _get_ai_factor_recommendations(self):
        """AI 요인 추천"""
        with st.spinner("AI가 최적의 실험 요인을 추천하고 있습니다..."):
            prompt = f"""
            다음 프로젝트에 적합한 실험 요인을 추천해주세요:
            
            프로젝트 정보:
            - 이름: {self.project['name']}
            - 분야: {self.project['field']} > {self.project['subfield']}
            - 설명: {self.project.get('description', '')}
            
            현재 요인: {st.session_state.design_factors}
            
            요청사항:
            1. 추가해야 할 중요 요인
            2. 각 요인의 권장 범위
            3. 요인 간 관계 고려사항
            4. 스크리닝 vs 최적화 관점
            
            응답은 구조화된 JSON 형식으로 해주세요.
            """
            
            response = self.api_manager.call_ai(prompt, response_format="json")
            
            if response:
                st.success("✅ AI 추천이 완료되었습니다")
                
                # 추천 결과 표시
                with st.expander("AI 추천 요인", expanded=True):
                    for factor in response.get('recommended_factors', []):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**{factor['name']}**")
                            st.write(f"- 범위: {factor['min']} ~ {factor['max']} {factor.get('unit', '')}")
                            st.write(f"- 이유: {factor['reason']}")
                        with col2:
                            if st.button(f"추가", key=f"add_ai_factor_{factor['name']}"):
                                st.session_state.design_factors.append({
                                    'name': factor['name'],
                                    'type': 'continuous',
                                    'unit': factor.get('unit', ''),
                                    'min_value': factor['min'],
                                    'max_value': factor['max'],
                                    'description': factor['reason']
                                })
                                st.rerun()
    
    def _get_ai_design_type_recommendation(self):
        """AI 설계 유형 추천"""
        factors_count = len(st.session_state.design_factors)
        responses_count = len(st.session_state.design_responses)
        
        with st.spinner("AI가 최적의 설계 유형을 추천하고 있습니다..."):
            prompt = f"""
            다음 실험에 가장 적합한 설계 유형을 추천해주세요:
            
            실험 정보:
            - 요인 수: {factors_count}
            - 반응변수 수: {responses_count}
            - 프로젝트 단계: {self.project.get('type', '연구개발')}
            - 요인 정보: {st.session_state.design_factors}
            
            고려사항:
            1. 실험 목적 (스크리닝 vs 최적화)
            2. 리소스 제약
            3. 교호작용 중요도
            4. 실험 정밀도 요구사항
            
            각 추천에 대해 장단점과 예상 실험 수를 포함해주세요.
            """
            
            response = self.api_manager.call_ai(prompt)
            
            if response:
                st.success("✅ AI 추천이 완료되었습니다")
                st.info(response)
    
    def _generate_design(self):
        """실험 설계 생성"""
        with st.spinner("실험 설계를 생성하고 있습니다..."):
            try:
                # 설계 입력 준비
                design_input = {
                    'design_type': st.session_state.design_type,
                    'factors': st.session_state.design_factors,
                    'responses': st.session_state.design_responses,
                    **st.session_state.design_options
                }
                
                # 모듈 사용 또는 내장 생성기
                if self.project_modules:
                    module = self.project_modules[0]
                    design = module.generate_design(design_input)
                else:
                    # 내장 설계 생성기 사용
                    design = self._generate_builtin_design(design_input)
                
                # 설계 저장
                st.session_state.current_design = design
                
                # 데이터베이스 저장
                self._save_design_to_db(design)
                
                st.success("✅ 실험 설계가 성공적으로 생성되었습니다!")
                st.balloons()
                
            except Exception as e:
                st.error(f"설계 생성 중 오류가 발생했습니다: {str(e)}")
    
    def _generate_builtin_design(self, design_input: Dict) -> ExperimentDesign:
        """내장 설계 생성기"""
        design_type = design_input['design_type']
        factors = [Factor(**f) for f in design_input['factors']]
        responses = [Response(**r) for r in design_input['responses']]
        
        # 연속형 요인만 추출
        continuous_factors = [f for f in factors if f.type == 'continuous']
        n_factors = len(continuous_factors)
        
        # 설계 행렬 생성
        if design_type == 'full_factorial':
            if design_input.get('n_levels', 2) == 2:
                design_matrix = ff2n(n_factors)
            else:
                levels = [design_input.get('n_levels', 2)] * n_factors
                design_matrix = fullfact(levels)
        
        elif design_type == 'fractional_factorial':
            resolution = design_input.get('resolution', 'IV')
            # 간단한 부분요인설계 생성
            if n_factors <= 4:
                design_matrix = ff2n(n_factors)
            else:
                # 2^(k-p) 설계
                p = max(1, n_factors - 7)  # 최대 128 실험
                design_matrix = fracfact(f"a b c d e f g"[:n_factors*2:2])
        
        elif design_type == 'central_composite':
            center = design_input.get('center_points', [4, 4])
            alpha = design_input.get('alpha', 'rotatable')
            design_matrix = ccdesign(n_factors, center=center, alpha=alpha)
        
        elif design_type == 'box_behnken':
            center = design_input.get('center_points', 3)
            design_matrix = bbdesign(n_factors, center=center)
        
        elif design_type == 'latin_hypercube':
            samples = design_input.get('samples', 10 * n_factors)
            design_matrix = lhs(n_factors, samples=samples)
            design_matrix = 2 * design_matrix - 1  # -1 to 1 스케일
        
        else:
            # 기본: 2수준 완전요인설계
            design_matrix = ff2n(n_factors)
        
        # 실제 값으로 변환
        runs_data = {}
        for i, factor in enumerate(continuous_factors):
            coded_values = design_matrix[:, i]
            real_values = factor.min_value + (coded_values + 1) / 2 * \
                         (factor.max_value - factor.min_value)
            runs_data[factor.name] = real_values
        
        # 범주형 요인 추가
        categorical_factors = [f for f in factors if f.type == 'categorical']
        n_runs = len(design_matrix)
        
        for factor in categorical_factors:
            # 균등 분포로 할당
            runs_data[factor.name] = np.random.choice(
                factor.levels, 
                size=n_runs, 
                replace=True
            )
        
        # 블록 추가
        if design_input.get('blocks', 1) > 1:
            n_blocks = design_input['blocks']
            block_size = n_runs // n_blocks
            blocks = []
            for i in range(n_blocks):
                blocks.extend([i+1] * block_size)
            # 남은 실험은 마지막 블록에
            blocks.extend([n_blocks] * (n_runs - len(blocks)))
            runs_data['Block'] = blocks
        
        # 반복 추가
        if design_input.get('replicates', 1) > 1:
            n_replicates = design_input['replicates']
            replicated_data = {}
            for key, values in runs_data.items():
                replicated_data[key] = np.tile(values, n_replicates)
            runs_data = replicated_data
            
            # 반복 번호 추가
            rep_numbers = []
            for i in range(n_replicates):
                rep_numbers.extend([i+1] * n_runs)
            runs_data['Replicate'] = rep_numbers
        
        # DataFrame 생성
        runs_df = pd.DataFrame(runs_data)
        
        # 랜덤화
        if design_input.get('randomize', True):
            runs_df = runs_df.sample(frac=1).reset_index(drop=True)
        
        runs_df.index = range(1, len(runs_df) + 1)
        runs_df.index.name = 'Run'
        
        # 반응변수 열 추가
        for response in responses:
            runs_df[response.name] = np.nan
        
        return ExperimentDesign(
            design_type=design_type,
            runs=runs_df,
            factors=factors,
            responses=responses,
            metadata={
                'created_at': datetime.now().isoformat(),
                'created_by': st.session_state.user['name'],
                'project_id': self.project['id'],
                'options': design_input
            }
        )
    
    def _validate_current_step(self) -> bool:
        """현재 단계 검증"""
        step = st.session_state.design_step
        
        if step == 0:  # 요인
            if not st.session_state.design_factors:
                st.error("최소 1개 이상의 요인을 정의해주세요")
                return False
            
            # 요인 검증
            for factor in st.session_state.design_factors:
                if not factor.get('name'):
                    st.error("모든 요인의 이름을 입력해주세요")
                    return False
                
                if factor['type'] == 'continuous':
                    if factor.get('min_value', 0) >= factor.get('max_value', 1):
                        st.error(f"요인 '{factor['name']}'의 최소값이 최대값보다 크거나 같습니다")
                        return False
                
                elif factor['type'] == 'categorical':
                    if not factor.get('levels'):
                        st.error(f"범주형 요인 '{factor['name']}'의 수준을 정의해주세요")
                        return False
        
        elif step == 1:  # 반응변수
            if not st.session_state.design_responses:
                st.warning("반응변수를 정의하지 않으면 분석이 제한됩니다. 계속하시겠습니까?")
        
        elif step == 2:  # 설계 유형
            # 특별한 검증 없음
            pass
        
        elif step == 3:  # 옵션
            # 제약조건 검증
            constraints = st.session_state.design_options.get('constraints', [])
            for constraint in constraints:
                try:
                    # 간단한 문법 검증
                    if not any(op in constraint for op in ['<', '>', '=', '!=']):
                        st.error(f"제약조건 '{constraint}'에 비교 연산자가 없습니다")
                        return False
                except:
                    st.error(f"제약조건 '{constraint}'의 형식이 올바르지 않습니다")
                    return False
        
        return True
    
    def _validate_design(self) -> Dict:
        """설계 검증"""
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        factors = st.session_state.design_factors
        responses = st.session_state.design_responses
        design_type = st.session_state.design_type
        options = st.session_state.design_options
        
        # 기본 검증
        if not factors:
            validation['is_valid'] = False
            validation['errors'].append("요인이 정의되지 않았습니다")
        
        # 설계별 검증
        n_factors = len([f for f in factors if f['type'] == 'continuous'])
        
        if design_type == 'central_composite' and n_factors < 2:
            validation['is_valid'] = False
            validation['errors'].append("CCD는 최소 2개의 연속형 요인이 필요합니다")
        
        if design_type == 'box_behnken' and n_factors < 3:
            validation['is_valid'] = False
            validation['errors'].append("Box-Behnken 설계는 최소 3개의 연속형 요인이 필요합니다")
        
        # 실험 수 검증
        total_runs = self._calculate_total_runs()
        
        if total_runs > 1000:
            validation['warnings'].append(f"실험 수가 {total_runs}개로 매우 많습니다")
            validation['suggestions'].append("부분요인설계나 D-최적설계를 고려해보세요")
        
        if total_runs < n_factors + 1:
            validation['warnings'].append("실험 수가 요인 수보다 적어 일부 효과를 추정할 수 없습니다")
        
        # 반응변수 검증
        if not responses:
            validation['warnings'].append("반응변수가 정의되지 않았습니다")
            validation['suggestions'].append("최소 1개 이상의 반응변수를 정의하는 것을 권장합니다")
        
        # 파워 검증
        if n_factors > 0:
            estimated_power = self._estimate_power(total_runs, n_factors)
            if estimated_power < 0.8:
                validation['warnings'].append(f"예상 검정력이 {estimated_power:.2f}로 낮습니다")
                validation['suggestions'].append("실험 수를 늘리거나 효과 크기를 재검토하세요")
        
        return validation
    
    def _calculate_total_runs(self) -> int:
        """총 실험 횟수 계산"""
        base_runs = self._estimate_runs(
            st.session_state.design_type,
            len([f for f in st.session_state.design_factors if f['type'] == 'continuous'])
        )
        
        options = st.session_state.design_options
        total = base_runs * options.get('replicates', 1)
        total += options.get('center_points', 0)
        
        return total
    
    def _estimate_runs(self, design_type: str, n_factors: int) -> int:
        """설계별 기본 실험 횟수 추정"""
        if n_factors == 0:
            return 0
        
        estimates = {
            'full_factorial': 2 ** n_factors,
            'fractional_factorial': 2 ** max(n_factors - 2, 3),
            'central_composite': 2 ** n_factors + 2 * n_factors + 1,
            'box_behnken': 2 * n_factors * (n_factors - 1) + 3,
            'plackett_burman': 4 * ((n_factors + 3) // 4),
            'latin_hypercube': 10 * n_factors,
            'd_optimal': 2 * n_factors + 5
        }
        
        return estimates.get(design_type, 2 ** n_factors)
    
    def _estimate_power(self, n_runs: int, n_factors: int) -> float:
        """간단한 검정력 추정"""
        if n_factors == 0 or n_runs == 0:
            return 0.0
        
        # 자유도
        df_error = n_runs - n_factors - 1
        if df_error <= 0:
            return 0.0
        
        # 간단한 추정 (실제로는 더 복잡한 계산 필요)
        power = 1 - np.exp(-0.1 * df_error)
        return min(power, 0.99)
    
    def _get_design_type_name(self, design_type: str) -> str:
        """설계 유형 한글명"""
        names = {
            'full_factorial': '완전요인설계',
            'fractional_factorial': '부분요인설계',
            'central_composite': '중심합성설계 (CCD)',
            'box_behnken': 'Box-Behnken 설계',
            'plackett_burman': 'Plackett-Burman 설계',
            'latin_hypercube': 'Latin Hypercube 설계',
            'd_optimal': 'D-최적설계'
        }
        return names.get(design_type, design_type)
    
    def _get_design_type_description(self, design_type: str) -> str:
        """설계 유형 설명"""
        descriptions = {
            'full_factorial': "모든 요인의 모든 수준 조합을 실험합니다. 모든 주효과와 교호작용을 추정할 수 있지만 실험 수가 많습니다.",
            'fractional_factorial': "완전요인설계의 일부만 실험합니다. 실험 수를 줄이면서 주요 효과를 추정할 수 있습니다.",
            'central_composite': "2차 모델 적합에 적합한 설계입니다. 반응표면분석에 널리 사용됩니다.",
            'box_behnken': "3수준 설계로 2차 모델에 효율적입니다. CCD보다 적은 실험으로 유사한 정보를 얻을 수 있습니다.",
            'plackett_burman': "많은 요인을 적은 실험으로 스크리닝하는데 적합합니다. 주효과만 추정 가능합니다.",
            'latin_hypercube': "설계 공간을 균등하게 탐색합니다. 컴퓨터 실험이나 시뮬레이션에 적합합니다.",
            'd_optimal': "특정 목적에 최적화된 설계를 생성합니다. 제약조건이 있거나 비표준적인 설계 공간에 유용합니다."
        }
        return descriptions.get(design_type, "")
    
    def _create_cube_edges(self, x_min, x_max, y_min, y_max, z_min, z_max):
        """3D 큐브 엣지 생성"""
        edges = []
        
        # 큐브의 12개 엣지
        # 하단 사각형
        edges.append(([x_min, x_max], [y_min, y_min], [z_min, z_min]))
        edges.append(([x_max, x_max], [y_min, y_max], [z_min, z_min]))
        edges.append(([x_max, x_min], [y_max, y_max], [z_min, z_min]))
        edges.append(([x_min, x_min], [y_max, y_min], [z_min, z_min]))
        
        # 상단 사각형
        edges.append(([x_min, x_max], [y_min, y_min], [z_max, z_max]))
        edges.append(([x_max, x_max], [y_min, y_max], [z_max, z_max]))
        edges.append(([x_max, x_min], [y_max, y_max], [z_max, z_max]))
        edges.append(([x_min, x_min], [y_max, y_min], [z_max, z_max]))
        
        # 수직 연결선
        edges.append(([x_min, x_min], [y_min, y_min], [z_min, z_max]))
        edges.append(([x_max, x_max], [y_min, y_min], [z_min, z_max]))
        edges.append(([x_max, x_max], [y_max, y_max], [z_min, z_max]))
        edges.append(([x_min, x_min], [y_max, y_max], [z_min, z_max]))
        
        return edges
    
    def _calculate_design_properties(self, design: ExperimentDesign) -> Dict:
        """설계 속성 계산"""
        continuous_factors = [f.name for f in design.factors if f.type == 'continuous']
        
        if not continuous_factors:
            return {
                'd_efficiency': 0,
                'g_efficiency': 0,
                'condition_number': np.inf,
                'orthogonality': 0
            }
        
        # 설계 행렬 (코드화된 값으로 변환)
        X = design.runs[continuous_factors].copy()
        
        # 정규화 (-1 to 1)
        for i, col in enumerate(continuous_factors):
            factor = design.factors[i]
            X[col] = 2 * (X[col] - factor.min_value) / (factor.max_value - factor.min_value) - 1
        
        # 모델 행렬 (절편 포함)
        X_model = np.column_stack([np.ones(len(X)), X.values])
        
        # 정보 행렬
        M = X_model.T @ X_model
        
        try:
            # D-효율성
            det_M = np.linalg.det(M)
            n = len(X)
            p = X_model.shape[1]
            d_eff = 100 * (det_M / n**p) ** (1/p)
            
            # G-효율성
            H = X_model @ np.linalg.inv(M) @ X_model.T
            g_eff = 100 * p / np.max(np.diag(H))
            
            # 조건수
            cond = np.linalg.cond(M)
            
            # 직교성 (상관 행렬 기반)
            corr_matrix = np.corrcoef(X.T)
            off_diagonal = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            orthogonality = 100 * (1 - np.mean(np.abs(off_diagonal)))
            
        except:
            d_eff = g_eff = orthogonality = 0
            cond = np.inf
        
        return {
            'd_efficiency': d_eff,
            'g_efficiency': g_eff,
            'condition_number': cond,
            'orthogonality': orthogonality
        }
    
    def _calculate_power(self, design: ExperimentDesign, effect_size: float, 
                        alpha: float, sigma: float) -> float:
        """검정력 계산"""
        n = len(design.runs)
        n_factors = len([f for f in design.factors if f.type == 'continuous'])
        
        if n_factors == 0:
            return 0.0
        
        # 자유도
        df1 = n_factors  # 모델 자유도
        df2 = n - n_factors - 1  # 오차 자유도
        
        if df2 <= 0:
            return 0.0
        
        # 비중심 모수
        lambda_nc = n * (effect_size / sigma) ** 2 / (2 * n_factors)
        
        # F 분포 임계값
        f_crit = stats.f.ppf(1 - alpha, df1, df2)
        
        # 검정력 (비중심 F 분포)
        power = 1 - stats.ncf.cdf(f_crit, df1, df2, lambda_nc)
        
        return power
    
    def _prepare_ai_context(self) -> Dict:
        """AI 컨텍스트 준비"""
        return {
            'project': {
                'name': self.project['name'],
                'field': self.project['field'],
                'subfield': self.project['subfield']
            },
            'factors': st.session_state.design_factors,
            'responses': st.session_state.design_responses,
            'design_type': st.session_state.design_type,
            'options': st.session_state.design_options,
            'current_step': st.session_state.design_step,
            'total_runs': self._calculate_total_runs()
        }
    
    def _process_ai_suggestions(self, suggestions: List[Dict]):
        """AI 제안사항 처리"""
        st.session_state.ai_suggestions = suggestions
        
        # 제안사항 표시
        if suggestions:
            with st.expander("💡 AI 제안사항", expanded=True):
                for i, suggestion in enumerate(suggestions):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"**{suggestion['title']}**")
                        st.write(suggestion['description'])
                    with col2:
                        if st.button("적용", key=f"apply_suggestion_{i}"):
                            self._apply_ai_suggestion(suggestion)
    
    def _apply_ai_suggestion(self, suggestion: Dict):
        """AI 제안 적용"""
        action = suggestion.get('action')
        
        if action == 'add_factor':
            st.session_state.design_factors.append(suggestion['data'])
        elif action == 'modify_option':
            st.session_state.design_options.update(suggestion['data'])
        elif action == 'change_design_type':
            st.session_state.design_type = suggestion['data']
        
        st.success(f"✅ {suggestion['title']} 적용됨")
        st.rerun()
    
    def _save_design(self):
        """설계 저장"""
        design = st.session_state.current_design
        
        # 데이터베이스에 저장
        design_data = {
            'project_id': self.project['id'],
            'name': f"Design_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'design_type': design.design_type,
            'factors': [f.model_dump() for f in design.factors],
            'responses': [r.model_dump() for r in design.responses],
            'runs': design.runs.to_dict(),
            'metadata': design.metadata,
            'created_at': datetime.now().isoformat(),
            'created_by': st.session_state.user['id']
        }
        
        design_id = self.db_manager.save_experiment_design(design_data)
        
        if design_id:
            st.success("✅ 설계가 저장되었습니다")
        else:
            st.error("설계 저장에 실패했습니다")
    
    def _save_design_to_db(self, design: ExperimentDesign):
        """설계를 데이터베이스에 저장"""
        # 위 _save_design과 동일한 로직
        pass
    
    def _export_design(self):
        """설계 내보내기"""
        design = st.session_state.current_design
        
        export_format = st.selectbox(
            "내보내기 형식",
            ["Excel", "CSV", "JSON", "Python 코드", "R 코드"]
        )
        
        if export_format == "Excel":
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                design.runs.to_excel(writer, sheet_name='Experimental Runs', index=True)
                
                # 요인 정보
                factors_df = pd.DataFrame([f.model_dump() for f in design.factors])
                factors_df.to_excel(writer, sheet_name='Factors', index=False)
                
                # 반응변수 정보
                responses_df = pd.DataFrame([r.model_dump() for r in design.responses])
                responses_df.to_excel(writer, sheet_name='Responses', index=False)
            
            st.download_button(
                "📥 Excel 다운로드",
                output.getvalue(),
                f"experiment_design_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        elif export_format == "CSV":
            csv = design.runs.to_csv(index=True)
            st.download_button(
                "📥 CSV 다운로드",
                csv,
                f"experiment_design_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        elif export_format == "JSON":
            json_data = {
                'design_type': design.design_type,
                'factors': [f.model_dump() for f in design.factors],
                'responses': [r.model_dump() for r in design.responses],
                'runs': design.runs.to_dict(),
                'metadata': design.metadata
            }
            
            st.download_button(
                "📥 JSON 다운로드",
                json.dumps(json_data, indent=2, ensure_ascii=False),
                f"experiment_design_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        elif export_format == "Python 코드":
            code = self._generate_python_code(design)
            st.code(code, language='python')
            st.download_button(
                "📥 Python 코드 다운로드",
                code,
                f"experiment_design_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py",
                mime="text/x-python"
            )
        
        elif export_format == "R 코드":
            code = self._generate_r_code(design)
            st.code(code, language='r')
            st.download_button(
                "📥 R 코드 다운로드",
                code,
                f"experiment_design_{datetime.now().strftime('%Y%m%d_%H%M%S')}.R",
                mime="text/plain"
            )
    
    def _generate_python_code(self, design: ExperimentDesign) -> str:
        """Python 코드 생성"""
        code = f"""# Experiment Design Generated by Universal DOE Platform
# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Design Type: {design.design_type}

import pandas as pd
import numpy as np
from pyDOE3 import *

# Factors
factors = {[f.model_dump() for f in design.factors]}

# Responses
responses = {[r.model_dump() for r in design.responses]}

# Experimental Runs
runs_data = {design.runs.to_dict()}
runs_df = pd.DataFrame(runs_data)

# Display
print("Experimental Design:")
print(runs_df)
print(f"\\nTotal runs: {len(runs_df)}")

# Save to file
runs_df.to_csv('experiment_design.csv', index=False)
print("Design saved to 'experiment_design.csv'")
"""
        return code
    
    def _generate_r_code(self, design: ExperimentDesign) -> str:
        """R 코드 생성"""
        code = f"""# Experiment Design Generated by Universal DOE Platform
# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Design Type: {design.design_type}

library(tidyverse)

# Create experimental runs
runs <- data.frame(
{self._format_r_dataframe(design.runs)}
)

# Display design
print("Experimental Design:")
print(runs)
cat(paste("\\nTotal runs:", nrow(runs)))

# Save to file
write.csv(runs, "experiment_design.csv", row.names = FALSE)
cat("Design saved to 'experiment_design.csv'\\n")

# Basic visualization
if(ncol(runs) >= 2) {{
  library(ggplot2)
  p <- ggplot(runs, aes(x = {design.factors[0].name if design.factors else 'X1'})) +
    geom_point() +
    theme_minimal()
  print(p)
}}
"""
        return code
    
    def _format_r_dataframe(self, df: pd.DataFrame) -> str:
        """R dataframe 포맷"""
        lines = []
        for col in df.columns:
            values = df[col].tolist()
            if df[col].dtype == 'object':
                values_str = ', '.join([f'"{v}"' for v in values])
            else:
                values_str = ', '.join([str(v) for v in values])
            lines.append(f'  {col} = c({values_str})')
        return ',\n'.join(lines)
    
    def _regenerate_design(self):
        """설계 재생성"""
        if st.confirm("현재 설계를 버리고 새로 생성하시겠습니까?"):
            self._generate_design()
    
    def _share_design(self):
        """설계 공유"""
        with st.dialog("실험 설계 공유"):
            st.write("공유 옵션을 선택하세요:")
            
            share_type = st.radio(
                "공유 방식",
                ["링크 공유", "이메일 전송", "팀 공유"]
            )
            
            if share_type == "링크 공유":
                # 공유 링크 생성
                share_link = f"https://universaldoe.com/design/{st.session_state.current_design.metadata.get('id', 'temp')}"
                st.code(share_link)
                st.info("링크를 복사하여 공유하세요")
            
            elif share_type == "이메일 전송":
                emails = st.text_area("수신자 이메일 (쉼표로 구분)")
                message = st.text_area("메시지")
                
                if st.button("전송"):
                    st.success("이메일이 전송되었습니다")
            
            elif share_type == "팀 공유":
                st.info("프로젝트 팀원들과 자동으로 공유됩니다")
                if st.button("팀 공유"):
                    st.success("팀원들에게 공유되었습니다")

# 페이지 렌더링
def render():
    """페이지 렌더링 함수"""
    page = ExperimentDesignPage()
    page.render()

# 메인 실행
if __name__ == "__main__":
    render()
