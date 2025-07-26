"""
데이터 분석 페이지
- 실험 데이터 수집, 분석, 시각화
- AI 기반 인사이트 도출
- 협업 분석 기능
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import asyncio
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 내부 모듈 임포트
try:
    from utils.sheets_manager import GoogleSheetsManager
    from utils.api_manager import APIManager
    from utils.common_ui import get_common_ui
    from utils.data_processor import DataProcessor
    from utils.notification_manager import NotificationManager
except ImportError as e:
    st.error(f"필요한 모듈을 찾을 수 없습니다: {e}")
    st.stop()

class DataAnalysisPage:
    """데이터 분석 페이지 클래스"""
    
    def __init__(self):
        """초기화"""
        self.sheets_manager = GoogleSheetsManager()
        self.api_manager = APIManager()
        self.ui = get_common_ui()
        self.data_processor = DataProcessor()
        self.notifier = NotificationManager() if 'NotificationManager' in globals() else None
        
        # 세션 상태 초기화
        self._initialize_session_state()
        
    def _initialize_session_state(self):
        """세션 상태 초기화"""
        defaults = {
            'current_analysis': None,
            'analysis_data': None,
            'processed_data': None,
            'analysis_results': {},
            'ai_insights': {},
            'selected_variables': {},
            'active_tab': 'data',
            'show_ai_details': False,  # AI 설명 상세도
            'ai_detail_level': 'auto',
            'annotations': [],
            'shared_analyses': []
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def render(self):
        """페이지 렌더링"""
        # 헤더
        self.ui.render_header(
            "데이터 분석",
            "실험 결과를 분석하고 AI 기반 인사이트를 도출합니다",
            "📈"
        )
        
        # 현재 프로젝트 확인
        if 'current_project' not in st.session_state:
            st.warning("먼저 프로젝트를 선택해주세요.")
            if st.button("프로젝트 선택하기"):
                st.session_state.current_page = 'project_setup'
                st.rerun()
            return
        
        # 메인 탭
        tabs = st.tabs([
            "📊 데이터 관리",
            "📈 통계 분석", 
            "🤖 AI 분석",
            "🎯 최적화",
            "👥 협업"
        ])
        
        with tabs[0]:
            self._render_data_management()
            
        with tabs[1]:
            self._render_statistical_analysis()
            
        with tabs[2]:
            self._render_ai_analysis()
            
        with tabs[3]:
            self._render_optimization()
            
        with tabs[4]:
            self._render_collaboration()
    
    def _render_data_management(self):
        """데이터 관리 탭"""
        st.subheader("📊 데이터 관리")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 데이터 업로드
            uploaded_file = st.file_uploader(
                "실험 데이터 업로드",
                type=['csv', 'xlsx', 'xls'],
                help="CSV 또는 Excel 파일을 업로드하세요"
            )
            
            if uploaded_file:
                try:
                    # 파일 읽기
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    st.session_state.analysis_data = df
                    st.success(f"데이터 로드 완료: {len(df)}개 행, {len(df.columns)}개 열")
                    
                except Exception as e:
                    st.error(f"파일 읽기 오류: {str(e)}")
        
        with col2:
            # 기존 데이터 불러오기
            st.write("**기존 실험 데이터**")
            
            # 프로젝트의 실험 목록
            experiments = self._get_project_experiments()
            
            if experiments:
                selected_exp = st.selectbox(
                    "실험 선택",
                    options=experiments,
                    format_func=lambda x: f"{x['name']} ({x['date']})"
                )
                
                if st.button("데이터 불러오기", type="primary"):
                    self._load_experiment_data(selected_exp['id'])
            else:
                st.info("저장된 실험 데이터가 없습니다.")
        
        # 데이터 미리보기
        if st.session_state.analysis_data is not None:
            self._render_data_preview()
            
            # 데이터 전처리
            with st.expander("🔧 데이터 전처리", expanded=False):
                self._render_data_preprocessing()
    
    def _render_data_preview(self):
        """데이터 미리보기"""
        df = st.session_state.analysis_data
        
        st.subheader("데이터 미리보기")
        
        # 기본 정보
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            self.ui.render_metric_card("행 수", f"{len(df):,}")
        with col2:
            self.ui.render_metric_card("열 수", f"{len(df.columns):,}")
        with col3:
            self.ui.render_metric_card("결측치", f"{df.isna().sum().sum():,}")
        with col4:
            self.ui.render_metric_card("데이터 타입", f"{df.dtypes.nunique()}")
        
        # 데이터 테이블
        st.dataframe(
            df.head(100),
            use_container_width=True,
            height=400
        )
        
        # 기초 통계
        with st.expander("📊 기초 통계", expanded=True):
            st.dataframe(
                df.describe(),
                use_container_width=True
            )
    
    def _render_data_preprocessing(self):
        """데이터 전처리 옵션"""
        df = st.session_state.analysis_data
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 결측치 처리
            st.write("**결측치 처리**")
            missing_method = st.selectbox(
                "처리 방법",
                ["제거", "평균값", "중앙값", "보간법", "그대로 유지"]
            )
            
            # 이상치 처리
            st.write("**이상치 처리**")
            outlier_method = st.selectbox(
                "처리 방법",
                ["없음", "IQR", "Z-score", "분위수"]
            )
        
        with col2:
            # 변수 변환
            st.write("**변수 변환**")
            transformations = st.multiselect(
                "적용할 변환",
                ["정규화", "표준화", "로그 변환", "제곱근 변환"]
            )
            
            # 변수 선택
            st.write("**분석할 변수 선택**")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            selected_cols = st.multiselect(
                "변수 선택",
                numeric_cols,
                default=numeric_cols
            )
        
        if st.button("전처리 적용", type="primary"):
            with st.spinner("데이터 전처리 중..."):
                processed_df = self.data_processor.preprocess_data(
                    df,
                    missing_method=missing_method,
                    outlier_method=outlier_method,
                    transformations=transformations,
                    selected_columns=selected_cols
                )
                
                st.session_state.processed_data = processed_df
                st.success("전처리 완료!")
                
                # 전처리 결과 요약
                st.info(f"""
                전처리 결과:
                - 원본 데이터: {len(df)} 행
                - 처리된 데이터: {len(processed_df)} 행
                - 제거된 행: {len(df) - len(processed_df)} 행
                """)
    
    def _render_statistical_analysis(self):
        """통계 분석 탭"""
        st.subheader("📈 통계 분석")
        
        if st.session_state.processed_data is None:
            st.warning("먼저 데이터를 로드하고 전처리해주세요.")
            return
        
        df = st.session_state.processed_data
        
        # 분석 유형 선택
        analysis_type = st.selectbox(
            "분석 유형",
            ["분산분석 (ANOVA)", "회귀분석", "반응표면분석 (RSM)", "상관분석", "주성분분석 (PCA)"]
        )
        
        if analysis_type == "분산분석 (ANOVA)":
            self._render_anova_analysis(df)
        elif analysis_type == "회귀분석":
            self._render_regression_analysis(df)
        elif analysis_type == "반응표면분석 (RSM)":
            self._render_rsm_analysis(df)
        elif analysis_type == "상관분석":
            self._render_correlation_analysis(df)
        elif analysis_type == "주성분분석 (PCA)":
            self._render_pca_analysis(df)
    
    def _render_anova_analysis(self, df: pd.DataFrame):
        """ANOVA 분석"""
        st.write("### 분산분석 (ANOVA)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 반응변수 선택
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            response = st.selectbox("반응변수", numeric_cols)
            
            # 요인 선택
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            numeric_cols_as_factor = st.multiselect(
                "숫자형 요인 (범주형으로 처리)",
                numeric_cols
            )
            
            all_factors = categorical_cols + numeric_cols_as_factor
            
            if not all_factors:
                st.warning("범주형 요인이 없습니다. 숫자형 변수를 요인으로 선택하세요.")
                return
                
            factors = st.multiselect("요인", all_factors, default=all_factors[:2])
        
        with col2:
            # ANOVA 옵션
            include_interactions = st.checkbox("교호작용 포함", value=True)
            confidence_level = st.slider("신뢰수준", 0.90, 0.99, 0.95, 0.01)
            
        if st.button("ANOVA 실행", type="primary"):
            with st.spinner("분석 중..."):
                try:
                    # ANOVA 수행
                    results = self._perform_anova(
                        df, response, factors, 
                        include_interactions, confidence_level
                    )
                    
                    # 결과 저장
                    st.session_state.analysis_results['anova'] = results
                    
                    # 결과 표시
                    self._display_anova_results(results)
                    
                except Exception as e:
                    st.error(f"분석 오류: {str(e)}")
    
    def _perform_anova(self, df: pd.DataFrame, response: str, 
                      factors: List[str], include_interactions: bool,
                      confidence_level: float) -> Dict:
        """ANOVA 수행"""
        # 모델 수식 생성
        formula = f"{response} ~ " + " + ".join(factors)
        
        if include_interactions and len(factors) >= 2:
            # 2차 교호작용 추가
            for i in range(len(factors)):
                for j in range(i+1, len(factors)):
                    formula += f" + {factors[i]}:{factors[j]}"
        
        # 모델 적합
        model = ols(formula, data=df).fit()
        
        # ANOVA 테이블
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        # 결과 구성
        results = {
            'model': model,
            'anova_table': anova_table,
            'formula': formula,
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'significant_factors': [],
            'post_hoc': {}
        }
        
        # 유의한 요인 찾기
        alpha = 1 - confidence_level
        for factor in anova_table.index[:-1]:  # Residual 제외
            if anova_table.loc[factor, 'PR(>F)'] < alpha:
                results['significant_factors'].append({
                    'factor': factor,
                    'f_value': anova_table.loc[factor, 'F'],
                    'p_value': anova_table.loc[factor, 'PR(>F)']
                })
        
        return results
    
    def _display_anova_results(self, results: Dict):
        """ANOVA 결과 표시"""
        st.success("분석 완료!")
        
        # 모델 요약
        col1, col2, col3 = st.columns(3)
        with col1:
            self.ui.render_metric_card("R²", f"{results['r_squared']:.4f}")
        with col2:
            self.ui.render_metric_card("Adjusted R²", f"{results['adj_r_squared']:.4f}")
        with col3:
            self.ui.render_metric_card("유의한 요인", len(results['significant_factors']))
        
        # ANOVA 테이블
        st.write("**ANOVA 테이블**")
        st.dataframe(
            results['anova_table'].round(4),
            use_container_width=True
        )
        
        # 유의한 요인
        if results['significant_factors']:
            st.write("**유의한 요인**")
            for factor in results['significant_factors']:
                st.write(f"- {factor['factor']}: F={factor['f_value']:.3f}, p={factor['p_value']:.4f}")
        
        # 잔차 플롯
        self._render_residual_plots(results['model'])
    
    def _render_ai_analysis(self):
        """AI 분석 탭"""
        st.subheader("🤖 AI 기반 분석")
        
        # AI 설명 상세도 제어
        self._render_ai_detail_control()
        
        if st.session_state.processed_data is None:
            st.warning("먼저 데이터를 로드하고 전처리해주세요.")
            return
        
        df = st.session_state.processed_data
        
        # AI 분석 유형
        col1, col2 = st.columns([2, 1])
        
        with col1:
            analysis_types = st.multiselect(
                "AI 분석 유형",
                [
                    "패턴 인식",
                    "이상치 탐지",
                    "예측 모델링",
                    "인사이트 도출",
                    "최적화 제안"
                ],
                default=["인사이트 도출"]
            )
        
        with col2:
            # AI 엔진 선택
            available_engines = self.api_manager.get_available_engines()
            selected_engine = st.selectbox(
                "AI 엔진",
                available_engines,
                help="사용 가능한 AI 엔진 중 선택"
            )
        
        # 추가 컨텍스트
        context = st.text_area(
            "추가 정보 (선택사항)",
            placeholder="실험 목적, 특별히 주목할 점 등을 입력하세요",
            height=100
        )
        
        if st.button("AI 분석 시작", type="primary"):
            with st.spinner("AI가 데이터를 분석 중입니다..."):
                try:
                    # AI 분석 수행
                    ai_results = self._perform_ai_analysis(
                        df, analysis_types, selected_engine, context
                    )
                    
                    # 결과 저장
                    st.session_state.ai_insights = ai_results
                    
                    # 결과 표시
                    self._display_ai_results(ai_results)
                    
                except Exception as e:
                    st.error(f"AI 분석 오류: {str(e)}")
                    # 오프라인 폴백
                    if st.checkbox("오프라인 기본 분석 사용"):
                        basic_results = self._perform_basic_analysis(df)
                        self._display_ai_results(basic_results)
    
    def _render_ai_detail_control(self):
        """AI 설명 상세도 제어 UI"""
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.write("**AI 설명 상세도**")
        
        with col2:
            detail_mode = st.selectbox(
                "모드",
                ["자동", "간단히", "상세히", "사용자 정의"],
                key="ai_detail_mode",
                label_visibility="collapsed"
            )
            st.session_state.ai_detail_level = detail_mode
        
        with col3:
            # 즉시 토글 버튼
            if st.button(
                "🔍 상세" if not st.session_state.show_ai_details else "📝 간단",
                key="toggle_ai_details"
            ):
                st.session_state.show_ai_details = not st.session_state.show_ai_details
    
    def _perform_ai_analysis(self, df: pd.DataFrame, analysis_types: List[str],
                           engine: str, context: str) -> Dict:
        """AI 분석 수행"""
        # 데이터 요약 생성
        data_summary = self._create_data_summary(df)
        
        # 프롬프트 구성
        prompt = self._build_ai_prompt(data_summary, analysis_types, context)
        
        # AI API 호출
        response = self.api_manager.get_ai_response(
            prompt=prompt,
            engine=engine,
            detail_level=st.session_state.ai_detail_level,
            include_reasoning=st.session_state.show_ai_details
        )
        
        # 응답 파싱
        results = self._parse_ai_response(response, analysis_types)
        
        return results
    
    def _build_ai_prompt(self, data_summary: Dict, analysis_types: List[str], 
                        context: str) -> str:
        """AI 프롬프트 구성"""
        prompt = f"""
        고분자 실험 데이터 분석 전문가로서 다음 데이터를 분석해주세요.
        
        데이터 개요:
        - 샘플 수: {data_summary['n_samples']}
        - 변수: {', '.join(data_summary['variables'])}
        - 데이터 유형: {data_summary['data_types']}
        
        기초 통계:
        {json.dumps(data_summary['statistics'], indent=2, ensure_ascii=False)}
        
        {f"추가 정보: {context}" if context else ""}
        
        다음 분석을 수행해주세요:
        {chr(10).join(f"- {t}" for t in analysis_types)}
        """
        
        # 상세도에 따른 추가 지시
        if st.session_state.show_ai_details:
            prompt += """
            
            각 분석에 대해:
            1. 핵심 발견사항 (필수)
            2. 분석 과정과 근거 (상세히)
            3. 대안적 해석
            4. 신뢰도와 한계점
            5. 추가 분석 제안
            
            JSON 형식으로 구조화하여 응답하세요.
            """
        else:
            prompt += """
            
            핵심 발견사항만 간단명료하게 제시하세요.
            불필요한 설명은 제외하고 실용적인 정보만 포함하세요.
            """
        
        return prompt
    
    def _display_ai_results(self, results: Dict):
        """AI 분석 결과 표시"""
        st.success("AI 분석 완료!")
        
        # 핵심 인사이트 (항상 표시)
        st.write("### 🎯 핵심 인사이트")
        
        for insight in results.get('key_insights', []):
            self.ui.render_info_card(
                insight['title'],
                insight['description'],
                insight.get('importance', 'info')
            )
        
        # 상세 설명 (조건부 표시)
        if st.session_state.show_ai_details and 'detailed_analysis' in results:
            tabs = st.tabs(["추론 과정", "대안 분석", "신뢰도 평가", "추가 제안"])
            
            with tabs[0]:
                st.write("**분석 과정**")
                for step in results['detailed_analysis'].get('reasoning_steps', []):
                    st.write(f"- {step}")
            
            with tabs[1]:
                st.write("**대안적 해석**")
                for alt in results['detailed_analysis'].get('alternatives', []):
                    st.info(alt)
            
            with tabs[2]:
                st.write("**신뢰도 평가**")
                confidence = results['detailed_analysis'].get('confidence', {})
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("전체 신뢰도", f"{confidence.get('overall', 0)}%")
                with col2:
                    st.write("**제한사항**")
                    for limitation in confidence.get('limitations', []):
                        st.write(f"- {limitation}")
            
            with tabs[3]:
                st.write("**추가 분석 제안**")
                for suggestion in results['detailed_analysis'].get('next_steps', []):
                    st.write(f"✓ {suggestion}")
        
        # AI 시각화
        if 'visualizations' in results:
            st.write("### 📊 AI 생성 시각화")
            for viz in results['visualizations']:
                if viz['type'] == 'plotly':
                    st.plotly_chart(viz['figure'], use_container_width=True)
                elif viz['type'] == 'explanation':
                    st.caption(viz['description'])
    
    def _render_optimization(self):
        """최적화 탭"""
        st.subheader("🎯 최적화")
        
        if not st.session_state.analysis_results:
            st.warning("먼저 통계 분석을 수행해주세요.")
            return
        
        # 최적화 설정
        col1, col2 = st.columns(2)
        
        with col1:
            # 목적함수 설정
            st.write("**최적화 목표**")
            
            # 반응변수 선택
            if 'responses' in st.session_state:
                responses = st.session_state.responses
            else:
                responses = st.session_state.processed_data.select_dtypes(
                    include=[np.number]
                ).columns.tolist()
            
            objective_response = st.selectbox("목표 반응변수", responses)
            objective_type = st.radio(
                "최적화 방향",
                ["최대화", "최소화", "목표값"],
                horizontal=True
            )
            
            if objective_type == "목표값":
                target_value = st.number_input("목표값", value=0.0)
            
        with col2:
            # 제약조건
            st.write("**제약조건**")
            
            constraints = []
            if st.checkbox("제약조건 추가"):
                num_constraints = st.number_input("제약조건 수", 1, 5, 1)
                
                for i in range(num_constraints):
                    with st.expander(f"제약조건 {i+1}"):
                        const_var = st.selectbox(
                            f"변수 {i+1}",
                            responses,
                            key=f"const_var_{i}"
                        )
                        const_type = st.selectbox(
                            f"유형 {i+1}",
                            ["≥", "≤", "="],
                            key=f"const_type_{i}"
                        )
                        const_value = st.number_input(
                            f"값 {i+1}",
                            key=f"const_val_{i}"
                        )
                        
                        constraints.append({
                            'variable': const_var,
                            'type': const_type,
                            'value': const_value
                        })
        
        # 최적화 알고리즘
        algorithm = st.selectbox(
            "최적화 알고리즘",
            ["Sequential Quadratic Programming (SQP)", 
             "Genetic Algorithm (GA)",
             "Particle Swarm Optimization (PSO)",
             "Nelder-Mead Simplex"]
        )
        
        if st.button("최적화 실행", type="primary"):
            with st.spinner("최적 조건을 찾는 중..."):
                try:
                    # 최적화 수행
                    opt_results = self._perform_optimization(
                        objective_response,
                        objective_type,
                        target_value if objective_type == "목표값" else None,
                        constraints,
                        algorithm
                    )
                    
                    # 결과 표시
                    self._display_optimization_results(opt_results)
                    
                except Exception as e:
                    st.error(f"최적화 오류: {str(e)}")
    
    def _perform_optimization(self, objective: str, obj_type: str,
                            target: Optional[float], constraints: List[Dict],
                            algorithm: str) -> Dict:
        """최적화 수행"""
        # 회귀 모델이 있는지 확인
        if 'regression' not in st.session_state.analysis_results:
            raise ValueError("회귀 모델이 필요합니다. 먼저 회귀분석을 수행하세요.")
        
        model = st.session_state.analysis_results['regression']['model']
        
        # 목적함수 정의
        def objective_function(x):
            pred = model.predict(pd.DataFrame([x], columns=model.feature_names_in_))
            
            if obj_type == "최대화":
                return -pred[0]  # 최소화 문제로 변환
            elif obj_type == "최소화":
                return pred[0]
            else:  # 목표값
                return (pred[0] - target) ** 2
        
        # 변수 범위
        bounds = []
        for var in model.feature_names_in_:
            data = st.session_state.processed_data[var]
            bounds.append((data.min(), data.max()))
        
        # 초기값
        x0 = [st.session_state.processed_data[var].mean() 
              for var in model.feature_names_in_]
        
        # 최적화 실행
        result = minimize(
            objective_function,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'disp': True}
        )
        
        # 최적값에서 예측
        optimal_x = result.x
        optimal_pred = model.predict(
            pd.DataFrame([optimal_x], columns=model.feature_names_in_)
        )[0]
        
        return {
            'success': result.success,
            'optimal_conditions': dict(zip(model.feature_names_in_, optimal_x)),
            'optimal_response': optimal_pred,
            'convergence_info': {
                'iterations': result.nit,
                'function_evals': result.nfev,
                'message': result.message
            },
            'objective_type': obj_type,
            'target_value': target
        }
    
    def _display_optimization_results(self, results: Dict):
        """최적화 결과 표시"""
        if results['success']:
            st.success("최적화 성공!")
        else:
            st.warning("최적화가 수렴하지 못했습니다.")
        
        # 최적 조건
        st.write("### 🎯 최적 조건")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**최적 변수값**")
            for var, value in results['optimal_conditions'].items():
                st.write(f"- {var}: {value:.4f}")
        
        with col2:
            st.write("**예측 결과**")
            self.ui.render_metric_card(
                "최적 반응값",
                f"{results['optimal_response']:.4f}"
            )
            
            if results['objective_type'] == "목표값":
                error = abs(results['optimal_response'] - results['target_value'])
                st.metric("목표값과의 차이", f"{error:.4f}")
        
        # 수렴 정보
        with st.expander("🔍 수렴 정보"):
            info = results['convergence_info']
            st.write(f"- 반복 횟수: {info['iterations']}")
            st.write(f"- 함수 평가 횟수: {info['function_evals']}")
            st.write(f"- 메시지: {info['message']}")
        
        # 민감도 분석
        if st.checkbox("민감도 분석 수행"):
            sensitivity = self._perform_sensitivity_analysis(
                results['optimal_conditions']
            )
            self._display_sensitivity_plot(sensitivity)
    
    def _render_collaboration(self):
        """협업 탭"""
        st.subheader("👥 협업 분석")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 분석 공유
            st.write("### 📤 분석 공유")
            
            if st.session_state.analysis_results:
                # 공유할 분석 선택
                analyses_to_share = st.multiselect(
                    "공유할 분석",
                    list(st.session_state.analysis_results.keys()),
                    default=list(st.session_state.analysis_results.keys())
                )
                
                # 공유 대상
                team_members = self._get_team_members()
                share_with = st.multiselect(
                    "공유 대상",
                    team_members,
                    format_func=lambda x: f"{x['name']} ({x['email']})"
                )
                
                # 공유 메시지
                message = st.text_area(
                    "메시지 (선택사항)",
                    placeholder="분석 결과에 대한 설명이나 의견을 남겨주세요"
                )
                
                if st.button("분석 공유", type="primary"):
                    self._share_analysis(analyses_to_share, share_with, message)
            else:
                st.info("공유할 분석 결과가 없습니다.")
        
        with col2:
            # 공유받은 분석
            st.write("### 📥 공유받은 분석")
            
            shared_analyses = self._get_shared_analyses()
            
            if shared_analyses:
                for analysis in shared_analyses:
                    with st.expander(
                        f"{analysis['title']} - {analysis['shared_by']} "
                        f"({analysis['date']})"
                    ):
                        st.write(analysis['message'])
                        if st.button(
                            "분석 보기",
                            key=f"view_{analysis['id']}"
                        ):
                            self._load_shared_analysis(analysis['id'])
            else:
                st.info("공유받은 분석이 없습니다.")
        
        # 팀 인사이트
        st.write("### 💡 팀 인사이트")
        self._render_team_insights()
    
    def _render_team_insights(self):
        """팀 인사이트 섹션"""
        # 인사이트 작성
        with st.form("insight_form"):
            insight_title = st.text_input("인사이트 제목")
            insight_content = st.text_area(
                "내용",
                placeholder="발견한 패턴, 제안사항 등을 공유하세요"
            )
            
            insight_type = st.selectbox(
                "유형",
                ["발견", "제안", "주의", "질문"]
            )
            
            if st.form_submit_button("인사이트 공유"):
                self._share_insight({
                    'title': insight_title,
                    'content': insight_content,
                    'type': insight_type,
                    'author': st.session_state.user_id,
                    'timestamp': datetime.now().isoformat()
                })
        
        # 기존 인사이트 표시
        insights = self._get_team_insights()
        
        for insight in insights:
            with st.container():
                col1, col2, col3 = st.columns([1, 4, 1])
                
                with col1:
                    icon = {
                        '발견': '💡',
                        '제안': '💭',
                        '주의': '⚠️',
                        '질문': '❓'
                    }.get(insight['type'], '📝')
                    st.write(f"### {icon}")
                
                with col2:
                    st.write(f"**{insight['title']}**")
                    st.write(insight['content'])
                    st.caption(
                        f"{insight['author']} - {insight['timestamp']}"
                    )
                
                with col3:
                    # 투표
                    votes = insight.get('votes', {'up': 0, 'down': 0})
                    col_up, col_down = st.columns(2)
                    
                    with col_up:
                        if st.button("👍", key=f"up_{insight['id']}"):
                            self._vote_insight(insight['id'], 'up')
                    with col_down:
                        if st.button("👎", key=f"down_{insight['id']}"):
                            self._vote_insight(insight['id'], 'down')
                    
                    st.caption(f"{votes['up']-votes['down']:+d}")
    
    # === 헬퍼 함수들 ===
    
    def _get_project_experiments(self) -> List[Dict]:
        """프로젝트의 실험 목록 조회"""
        project_id = st.session_state.current_project['id']
        return self.sheets_manager.get_experiments(project_id)
    
    def _load_experiment_data(self, experiment_id: str):
        """실험 데이터 로드"""
        data = self.sheets_manager.get_experiment_results(experiment_id)
        if data:
            df = pd.DataFrame(data)
            st.session_state.analysis_data = df
            st.success("데이터를 불러왔습니다.")
        else:
            st.error("데이터를 불러올 수 없습니다.")
    
    def _create_data_summary(self, df: pd.DataFrame) -> Dict:
        """데이터 요약 생성"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        return {
            'n_samples': len(df),
            'variables': df.columns.tolist(),
            'data_types': df.dtypes.to_dict(),
            'statistics': {
                'mean': numeric_df.mean().to_dict(),
                'std': numeric_df.std().to_dict(),
                'min': numeric_df.min().to_dict(),
                'max': numeric_df.max().to_dict(),
                'correlations': numeric_df.corr().to_dict()
            },
            'missing_values': df.isna().sum().to_dict()
        }
    
    def _parse_ai_response(self, response: str, analysis_types: List[str]) -> Dict:
        """AI 응답 파싱"""
        try:
            # JSON 응답 파싱 시도
            if response.strip().startswith('{'):
                return json.loads(response)
            else:
                # 텍스트 응답을 구조화
                return {
                    'key_insights': [{
                        'title': '분석 결과',
                        'description': response,
                        'importance': 'info'
                    }],
                    'analysis_types': analysis_types
                }
        except:
            return {
                'key_insights': [{
                    'title': '분석 결과',
                    'description': response,
                    'importance': 'info'
                }]
            }
    
    def _perform_basic_analysis(self, df: pd.DataFrame) -> Dict:
        """오프라인 기본 분석"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        insights = []
        
        # 기본 통계
        for col in numeric_df.columns:
            mean_val = numeric_df[col].mean()
            std_val = numeric_df[col].std()
            insights.append({
                'title': f'{col} 분포',
                'description': f'평균: {mean_val:.2f} ± {std_val:.2f}',
                'importance': 'info'
            })
        
        # 상관관계
        corr_matrix = numeric_df.corr()
        high_corr = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'corr': corr_val
                    })
        
        if high_corr:
            insights.append({
                'title': '높은 상관관계 발견',
                'description': f'{len(high_corr)}개의 변수 쌍에서 높은 상관관계 발견',
                'importance': 'high'
            })
        
        return {'key_insights': insights}
    
    def _render_residual_plots(self, model):
        """잔차 플롯"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['잔차 vs 적합값', 'Q-Q Plot', 
                          '표준화 잔차', '잔차 히스토그램']
        )
        
        # 잔차 계산
        residuals = model.resid
        fitted = model.fittedvalues
        standardized_resid = model.get_influence().resid_studentized_internal
        
        # 1. 잔차 vs 적합값
        fig.add_trace(
            go.Scatter(x=fitted, y=residuals, mode='markers', name='잔차'),
            row=1, col=1
        )
        
        # 2. Q-Q Plot
        theoretical_quantiles = stats.probplot(residuals, dist="norm", fit=False)[0]
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=np.sort(residuals), 
                      mode='markers', name='Q-Q'),
            row=1, col=2
        )
        
        # 3. 표준화 잔차
        fig.add_trace(
            go.Scatter(y=standardized_resid, mode='markers', name='표준화 잔차'),
            row=2, col=1
        )
        
        # 4. 잔차 히스토그램
        fig.add_trace(
            go.Histogram(x=residuals, name='분포'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    def _get_team_members(self) -> List[Dict]:
        """팀 멤버 목록 조회"""
        project_id = st.session_state.current_project['id']
        return self.sheets_manager.get_project_collaborators(project_id)
    
    def _share_analysis(self, analyses: List[str], recipients: List[Dict], 
                       message: str):
        """분석 공유"""
        try:
            # 분석 데이터 준비
            shared_data = {
                'project_id': st.session_state.current_project['id'],
                'analyses': {
                    name: st.session_state.analysis_results[name]
                    for name in analyses
                },
                'shared_by': st.session_state.user_id,
                'shared_with': [r['id'] for r in recipients],
                'message': message,
                'timestamp': datetime.now().isoformat()
            }
            
            # 저장
            analysis_id = self.sheets_manager.save_shared_analysis(shared_data)
            
            # 알림 발송
            if self.notifier:
                for recipient in recipients:
                    self.notifier.send_notification(
                        recipient['id'],
                        f"{st.session_state.user_name}님이 분석을 공유했습니다",
                        'analysis_shared',
                        {'analysis_id': analysis_id}
                    )
            
            st.success("분석이 공유되었습니다!")
            
        except Exception as e:
            st.error(f"공유 실패: {str(e)}")
    
    def _get_shared_analyses(self) -> List[Dict]:
        """공유받은 분석 조회"""
        return self.sheets_manager.get_shared_analyses(
            st.session_state.user_id
        )
    
    def _load_shared_analysis(self, analysis_id: str):
        """공유받은 분석 로드"""
        analysis = self.sheets_manager.get_shared_analysis(analysis_id)
        if analysis:
            st.session_state.analysis_results = analysis['analyses']
            st.success("분석을 불러왔습니다.")
            st.rerun()
    
    def _share_insight(self, insight: Dict):
        """인사이트 공유"""
        insight['id'] = str(uuid.uuid4())
        insight['project_id'] = st.session_state.current_project['id']
        
        self.sheets_manager.save_team_insight(insight)
        st.success("인사이트가 공유되었습니다!")
        st.rerun()
    
    def _get_team_insights(self) -> List[Dict]:
        """팀 인사이트 조회"""
        return self.sheets_manager.get_team_insights(
            st.session_state.current_project['id']
        )
    
    def _vote_insight(self, insight_id: str, vote_type: str):
        """인사이트 투표"""
        self.sheets_manager.vote_insight(
            insight_id,
            st.session_state.user_id,
            vote_type
        )
        st.rerun()
    
    def _render_regression_analysis(self, df: pd.DataFrame):
        """회귀분석 렌더링"""
        st.write("### 회귀분석")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 반응변수 선택
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            response = st.selectbox("반응변수", numeric_cols, key="reg_response")
            
            # 독립변수 선택
            predictors = st.multiselect(
                "독립변수",
                [col for col in numeric_cols if col != response],
                key="reg_predictors"
            )
            
        with col2:
            # 회귀 옵션
            model_type = st.selectbox(
                "모델 유형",
                ["선형", "다항식", "로그 변환"]
            )
            
            if model_type == "다항식":
                poly_degree = st.slider("차수", 2, 4, 2)
            
            include_interaction = st.checkbox("교호작용 포함", value=False)
        
        if st.button("회귀분석 실행", type="primary", key="run_regression"):
            if len(predictors) == 0:
                st.warning("최소 하나의 독립변수를 선택하세요.")
                return
                
            with st.spinner("회귀분석 중..."):
                try:
                    # 회귀분석 수행
                    results = self._perform_regression(
                        df, response, predictors,
                        model_type, poly_degree if model_type == "다항식" else None,
                        include_interaction
                    )
                    
                    # 결과 저장
                    st.session_state.analysis_results['regression'] = results
                    
                    # 결과 표시
                    self._display_regression_results(results)
                    
                except Exception as e:
                    st.error(f"회귀분석 오류: {str(e)}")
    
    def _perform_regression(self, df: pd.DataFrame, response: str,
                           predictors: List[str], model_type: str,
                           poly_degree: Optional[int],
                           include_interaction: bool) -> Dict:
        """회귀분석 수행"""
        # 데이터 준비
        X = df[predictors].copy()
        y = df[response]
        
        # 모델 유형에 따른 변환
        if model_type == "다항식":
            from sklearn.preprocessing import PolynomialFeatures
            poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
            X_poly = poly.fit_transform(X)
            X = pd.DataFrame(
                X_poly,
                columns=poly.get_feature_names_out(predictors)
            )
        elif model_type == "로그 변환":
            X = np.log(X + 1)  # log(x+1) to handle zeros
        
        # 상수항 추가
        X = sm.add_constant(X)
        
        # 모델 적합
        model = sm.OLS(y, X).fit()
        
        # 예측값
        predictions = model.predict(X)
        
        # VIF 계산 (다중공선성)
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        vif_data = pd.DataFrame()
        vif_data["변수"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                          for i in range(len(X.columns))]
        
        return {
            'model': model,
            'summary': model.summary(),
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'coefficients': model.params.to_dict(),
            'p_values': model.pvalues.to_dict(),
            'predictions': predictions,
            'residuals': model.resid,
            'vif': vif_data,
            'aic': model.aic,
            'bic': model.bic
        }
    
    def _display_regression_results(self, results: Dict):
        """회귀분석 결과 표시"""
        st.success("회귀분석 완료!")
        
        # 모델 요약
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            self.ui.render_metric_card("R²", f"{results['r_squared']:.4f}")
        with col2:
            self.ui.render_metric_card("Adjusted R²", f"{results['adj_r_squared']:.4f}")
        with col3:
            self.ui.render_metric_card("AIC", f"{results['aic']:.2f}")
        with col4:
            self.ui.render_metric_card("BIC", f"{results['bic']:.2f}")
        
        # 회귀계수
        st.write("**회귀계수**")
        coef_df = pd.DataFrame({
            '계수': results['coefficients'],
            'p-value': results['p_values']
        })
        coef_df['유의성'] = coef_df['p-value'].apply(
            lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        )
        st.dataframe(coef_df, use_container_width=True)
        
        # VIF (다중공선성)
        with st.expander("다중공선성 진단 (VIF)"):
            st.dataframe(results['vif'], use_container_width=True)
            st.caption("VIF > 10인 경우 다중공선성 문제가 있을 수 있습니다.")
        
        # 잔차 플롯
        self._render_residual_plots(results['model'])
    
    def _render_rsm_analysis(self, df: pd.DataFrame):
        """반응표면분석 렌더링"""
        st.write("### 반응표면분석 (RSM)")
        
        st.info("""
        반응표면분석은 실험계획법의 데이터를 바탕으로 
        최적 조건을 찾는 고급 분석 방법입니다.
        """)
        
        # RSM 설정
        col1, col2 = st.columns(2)
        
        with col1:
            # 반응변수
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            response = st.selectbox("반응변수", numeric_cols, key="rsm_response")
            
            # 실험인자
            factors = st.multiselect(
                "실험인자 (2-4개)",
                [col for col in numeric_cols if col != response],
                key="rsm_factors"
            )
            
        with col2:
            # RSM 모델
            rsm_model = st.selectbox(
                "RSM 모델",
                ["2차 모델", "Box-Behnken", "중심합성계획"]
            )
            
            # 최적화 목표
            optimization_goal = st.radio(
                "최적화 목표",
                ["최대화", "최소화", "목표값"],
                horizontal=True
            )
            
            if optimization_goal == "목표값":
                target = st.number_input("목표값", key="rsm_target")
        
        if len(factors) < 2:
            st.warning("최소 2개의 실험인자를 선택하세요.")
            return
            
        if st.button("RSM 분석 실행", type="primary", key="run_rsm"):
            with st.spinner("반응표면 분석 중..."):
                try:
                    # RSM 수행
                    results = self._perform_rsm(
                        df, response, factors,
                        rsm_model, optimization_goal,
                        target if optimization_goal == "목표값" else None
                    )
                    
                    # 결과 저장
                    st.session_state.analysis_results['rsm'] = results
                    
                    # 결과 표시
                    self._display_rsm_results(results)
                    
                except Exception as e:
                    st.error(f"RSM 분석 오류: {str(e)}")
    
    def _perform_rsm(self, df: pd.DataFrame, response: str,
                    factors: List[str], model_type: str,
                    optimization_goal: str,
                    target: Optional[float]) -> Dict:
        """RSM 분석 수행"""
        # 2차 모델 생성
        from sklearn.preprocessing import PolynomialFeatures
        
        X = df[factors]
        y = df[response]
        
        # 2차 다항식 특성 생성
        poly = PolynomialFeatures(degree=2, include_bias=True)
        X_poly = poly.fit_transform(X)
        feature_names = poly.get_feature_names_out(factors)
        
        # 회귀모델 적합
        X_poly_df = pd.DataFrame(X_poly, columns=feature_names)
        model = sm.OLS(y, X_poly_df).fit()
        
        # 최적점 찾기
        from scipy.optimize import minimize
        
        def objective(x):
            x_poly = poly.transform([x])
            pred = model.predict(x_poly)[0]
            
            if optimization_goal == "최대화":
                return -pred
            elif optimization_goal == "최소화":
                return pred
            else:  # 목표값
                return (pred - target) ** 2
        
        # 초기값과 범위
        x0 = X.mean().values
        bounds = [(X[col].min(), X[col].max()) for col in factors]
        
        # 최적화
        opt_result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
        
        # 최적점에서의 예측
        optimal_x = opt_result.x
        optimal_x_poly = poly.transform([optimal_x])
        optimal_y = model.predict(optimal_x_poly)[0]
        
        # 반응표면 데이터 생성 (시각화용)
        if len(factors) >= 2:
            surface_data = self._generate_surface_data(
                model, poly, factors[:2], X[factors[2:]].mean() if len(factors) > 2 else None
            )
        else:
            surface_data = None
        
        return {
            'model': model,
            'polynomial_features': poly,
            'feature_names': feature_names,
            'optimal_conditions': dict(zip(factors, optimal_x)),
            'optimal_response': optimal_y,
            'optimization_goal': optimization_goal,
            'target_value': target,
            'surface_data': surface_data,
            'r_squared': model.rsquared,
            'anova': sm.stats.anova_lm(model, typ=2)
        }
    
    def _generate_surface_data(self, model, poly, factors, fixed_values=None):
        """반응표면 데이터 생성"""
        # 그리드 생성
        x_range = np.linspace(0, 1, 50)
        y_range = np.linspace(0, 1, 50)
        X_grid, Y_grid = np.meshgrid(x_range, y_range)
        
        # 예측값 계산
        Z_grid = np.zeros_like(X_grid)
        
        for i in range(X_grid.shape[0]):
            for j in range(X_grid.shape[1]):
                point = [X_grid[i, j], Y_grid[i, j]]
                
                # 추가 요인이 있으면 고정값 사용
                if fixed_values is not None:
                    point.extend(fixed_values)
                
                point_poly = poly.transform([point])
                Z_grid[i, j] = model.predict(point_poly)[0]
        
        return {
            'X': X_grid,
            'Y': Y_grid,
            'Z': Z_grid,
            'factors': factors
        }
    
    def _display_rsm_results(self, results: Dict):
        """RSM 결과 표시"""
        st.success("반응표면분석 완료!")
        
        # 최적 조건
        st.write("### 🎯 최적 조건")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**최적 인자 수준**")
            for factor, value in results['optimal_conditions'].items():
                st.write(f"- {factor}: {value:.4f}")
        
        with col2:
            st.write("**예측 반응값**")
            self.ui.render_metric_card(
                f"{results['optimization_goal']} 값",
                f"{results['optimal_response']:.4f}"
            )
            
            if results['optimization_goal'] == "목표값":
                error = abs(results['optimal_response'] - results['target_value'])
                pct_error = (error / results['target_value']) * 100
                st.metric("목표값 대비 오차", f"{pct_error:.1f}%")
        
        # 모델 적합도
        st.write("### 📊 모델 적합도")
        col1, col2 = st.columns(2)
        
        with col1:
            self.ui.render_metric_card("R²", f"{results['r_squared']:.4f}")
        
        with col2:
            # ANOVA 테이블
            with st.expander("ANOVA 테이블"):
                st.dataframe(results['anova'], use_container_width=True)
        
        # 반응표면 플롯
        if results['surface_data']:
            st.write("### 🗺️ 반응표면 플롯")
            
            surface = results['surface_data']
            
            # 3D Surface
            fig = go.Figure(data=[
                go.Surface(
                    x=surface['X'],
                    y=surface['Y'],
                    z=surface['Z'],
                    colorscale='Viridis',
                    contours={
                        "z": {"show": True, "usecolormap": True,
                              "project": {"z": True}}
                    }
                )
            ])
            
            # 최적점 표시
            opt_x = results['optimal_conditions'][surface['factors'][0]]
            opt_y = results['optimal_conditions'][surface['factors'][1]]
            opt_z = results['optimal_response']
            
            fig.add_trace(go.Scatter3d(
                x=[opt_x], y=[opt_y], z=[opt_z],
                mode='markers+text',
                marker=dict(size=10, color='red'),
                text=['최적점'],
                textposition='top center',
                name='최적점'
            ))
            
            fig.update_layout(
                title="반응표면 3D 플롯",
                scene=dict(
                    xaxis_title=surface['factors'][0],
                    yaxis_title=surface['factors'][1],
                    zaxis_title="Response"
                ),
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 등고선 플롯
            fig_contour = go.Figure(data=[
                go.Contour(
                    x=surface['X'][0],
                    y=surface['Y'][:, 0],
                    z=surface['Z'],
                    colorscale='Viridis',
                    contours=dict(
                        coloring='heatmap',
                        showlabels=True
                    )
                )
            ])
            
            # 최적점 표시
            fig_contour.add_trace(go.Scatter(
                x=[opt_x], y=[opt_y],
                mode='markers+text',
                marker=dict(size=15, color='red', symbol='x'),
                text=['최적점'],
                textposition='top center',
                name='최적점'
            ))
            
            fig_contour.update_layout(
                title="반응표면 등고선 플롯",
                xaxis_title=surface['factors'][0],
                yaxis_title=surface['factors'][1],
                height=500
            )
            
            st.plotly_chart(fig_contour, use_container_width=True)
    
    def _render_correlation_analysis(self, df: pd.DataFrame):
        """상관분석 렌더링"""
        st.write("### 상관분석")
        
        # 숫자형 변수만 선택
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            st.warning("숫자형 변수가 없습니다.")
            return
        
        # 상관계수 계산
        corr_matrix = numeric_df.corr()
        
        # 히트맵
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate='%{text}',
            textfont={"size": 10},
            reversescale=True
        ))
        
        fig.update_layout(
            title="상관계수 히트맵",
            height=600,
            xaxis={'side': 'bottom'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 높은 상관관계 표시
        st.write("### 📊 주요 상관관계")
        
        threshold = st.slider("상관계수 임계값", 0.5, 0.9, 0.7, 0.05)
        
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    high_corr.append({
                        '변수1': corr_matrix.columns[i],
                        '변수2': corr_matrix.columns[j],
                        '상관계수': corr_val,
                        '관계': '양의 상관' if corr_val > 0 else '음의 상관'
                    })
        
        if high_corr:
            high_corr_df = pd.DataFrame(high_corr)
            high_corr_df = high_corr_df.sort_values('상관계수', 
                                                    key=abs, ascending=False)
            st.dataframe(high_corr_df, use_container_width=True)
        else:
            st.info(f"상관계수가 {threshold} 이상인 변수 쌍이 없습니다.")
    
    def _render_pca_analysis(self, df: pd.DataFrame):
        """주성분분석 렌더링"""
        st.write("### 주성분분석 (PCA)")
        
        # 숫자형 변수만 선택
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("최소 2개의 숫자형 변수가 필요합니다.")
            return
        
        # 변수 선택
        selected_vars = st.multiselect(
            "분석할 변수",
            numeric_cols,
            default=numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols
        )
        
        if len(selected_vars) < 2:
            st.warning("최소 2개의 변수를 선택하세요.")
            return
        
        # 표준화 옵션
        standardize = st.checkbox("데이터 표준화", value=True)
        
        if st.button("PCA 실행", type="primary"):
            with st.spinner("주성분분석 중..."):
                # 데이터 준비
                X = df[selected_vars].dropna()
                
                if standardize:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                else:
                    X_scaled = X.values
                
                # PCA 수행
                pca = PCA()
                X_pca = pca.fit_transform(X_scaled)
                
                # 결과 표시
                st.success("주성분분석 완료!")
                
                # 설명된 분산
                st.write("### 📊 설명된 분산")
                
                explained_var = pca.explained_variance_ratio_
                cumsum_var = np.cumsum(explained_var)
                
                fig = go.Figure()
                
                # 개별 설명 분산
                fig.add_trace(go.Bar(
                    x=[f'PC{i+1}' for i in range(len(explained_var))],
                    y=explained_var,
                    name='개별',
                    marker_color='lightblue'
                ))
                
                # 누적 설명 분산
                fig.add_trace(go.Scatter(
                    x=[f'PC{i+1}' for i in range(len(cumsum_var))],
                    y=cumsum_var,
                    name='누적',
                    mode='lines+markers',
                    marker_color='red',
                    yaxis='y2'
                ))
                
                fig.update_layout(
                    title="주성분별 설명 분산",
                    xaxis_title="주성분",
                    yaxis_title="설명 분산 비율",
                    yaxis2=dict(
                        title="누적 설명 분산",
                        overlaying='y',
                        side='right'
                    ),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 주성분 개수 선택
                n_components = st.selectbox(
                    "사용할 주성분 개수",
                    range(1, min(len(selected_vars), 6)),
                    index=min(2, len(selected_vars)-1)
                )
                
                # 주성분 점수 플롯
                if n_components >= 2:
                    st.write("### 🔍 주성분 점수 플롯")
                    
                    fig_score = go.Figure()
                    
                    fig_score.add_trace(go.Scatter(
                        x=X_pca[:, 0],
                        y=X_pca[:, 1],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=range(len(X_pca)),
                            colorscale='Viridis',
                            showscale=True
                        ),
                        text=[f'Sample {i+1}' for i in range(len(X_pca))],
                        hovertemplate='%{text}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}'
                    ))
                    
                    fig_score.update_layout(
                        title="주성분 점수 플롯 (PC1 vs PC2)",
                        xaxis_title=f'PC1 ({explained_var[0]:.1%})',
                        yaxis_title=f'PC2 ({explained_var[1]:.1%})',
                        height=500
                    )
                    
                    st.plotly_chart(fig_score, use_container_width=True)
                
                # 주성분 적재량
                st.write("### 📋 주성분 적재량")
                
                loadings = pd.DataFrame(
                    pca.components_[:n_components].T,
                    columns=[f'PC{i+1}' for i in range(n_components)],
                    index=selected_vars
                )
                
                # 히트맵으로 표시
                fig_loading = go.Figure(data=go.Heatmap(
                    z=loadings.values,
                    x=loadings.columns,
                    y=loadings.index,
                    colorscale='RdBu',
                    zmid=0,
                    text=loadings.round(2).values,
                    texttemplate='%{text}',
                    textfont={"size": 10}
                ))
                
                fig_loading.update_layout(
                    title="주성분 적재량 히트맵",
                    height=400
                )
                
                st.plotly_chart(fig_loading, use_container_width=True)
                
                # 적재량 테이블
                st.dataframe(
                    loadings.round(3),
                    use_container_width=True
                )
                
                # 기여도가 높은 변수
                st.write("### 🎯 주요 변수")
                
                for i in range(n_components):
                    pc_loadings = loadings[f'PC{i+1}'].abs().sort_values(ascending=False)
                    top_vars = pc_loadings.head(3)
                    
                    st.write(f"**PC{i+1}의 주요 변수:**")
                    for var, loading in top_vars.items():
                        direction = "+" if loadings.loc[var, f'PC{i+1}'] > 0 else "-"
                        st.write(f"- {var}: {direction}{loading:.3f}")
    
    def _perform_sensitivity_analysis(self, optimal_conditions: Dict) -> Dict:
        """민감도 분석"""
        sensitivity = {}
        
        # 각 변수를 ±10% 변경했을 때의 영향 분석
        for var in optimal_conditions:
            base_value = optimal_conditions[var]
            sensitivity[var] = {
                'base': base_value,
                'sensitivity': []
            }
            
            # -10% ~ +10% 범위에서 분석
            for pct in range(-10, 11, 2):
                test_conditions = optimal_conditions.copy()
                test_conditions[var] = base_value * (1 + pct/100)
                
                # 예측값 계산 (간단한 예시)
                # 실제로는 저장된 모델 사용
                pred_change = pct * 0.5  # 임시 값
                
                sensitivity[var]['sensitivity'].append({
                    'change_pct': pct,
                    'response_change': pred_change
                })
        
        return sensitivity
    
    def _display_sensitivity_plot(self, sensitivity: Dict):
        """민감도 플롯"""
        fig = go.Figure()
        
        for var, data in sensitivity.items():
            changes = [s['change_pct'] for s in data['sensitivity']]
            responses = [s['response_change'] for s in data['sensitivity']]
            
            fig.add_trace(go.Scatter(
                x=changes,
                y=responses,
                mode='lines+markers',
                name=var
            ))
        
        fig.update_layout(
            title="민감도 분석",
            xaxis_title="변수 변화율 (%)",
            yaxis_title="반응값 변화율 (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)


# 페이지 렌더링 함수
def render():
    """데이터 분석 페이지 렌더링"""
    page = DataAnalysisPage()
    page.render()
