"""
4_📈_Data_Analysis.py - 데이터 분석
실험 결과를 분석하고 AI 기반 인사이트를 제공하는 핵심 분석 페이지
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import json
import io
from pathlib import Path
import sys
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

# 필수 모듈 임포트
try:
    from utils.database_manager import get_database_manager
    from utils.auth_manager import get_auth_manager
    from utils.common_ui import get_common_ui
    from utils.api_manager import get_api_manager
    from utils.data_processor import get_data_processor
    from config.app_config import EXPERIMENT_DEFAULTS
    from config.theme_config import COLORS
except ImportError as e:
    st.error(f"필수 모듈 임포트 오류: {e}")
    st.stop()

# 페이지 설정
st.set_page_config(
    page_title="데이터 분석 - Universal DOE",
    page_icon="📈",
    layout="wide"
)

# 인증 확인
auth_manager = get_auth_manager()
if not auth_manager.check_authentication():
    st.warning("로그인이 필요합니다")
    st.switch_page("pages/0_🔐_Login.py")
    st.stop()

class DataAnalysisPage:
    """데이터 분석 페이지 클래스"""
    
    def __init__(self):
        self.ui = get_common_ui()
        self.db_manager = get_database_manager()
        self.api_manager = get_api_manager()
        self.data_processor = get_data_processor()
        
        # 세션 상태 초기화
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """세션 상태 초기화"""
        defaults = {
            'analysis_data': None,
            'processed_data': None,
            'analysis_results': {},
            'ai_insights': [],
            'show_ai_details': False,
            'current_analysis': None,
            'selected_factors': [],
            'selected_responses': [],
            'analysis_type': 'descriptive'
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def render(self):
        """메인 렌더링"""
        self.ui.render_header("📈 데이터 분석", "실험 결과를 분석하고 인사이트를 발견합니다")
        
        # 데이터 업로드 섹션
        with st.expander("📤 데이터 업로드", expanded=not bool(st.session_state.analysis_data)):
            self._render_data_upload()
        
        # 데이터가 업로드된 경우에만 분석 섹션 표시
        if st.session_state.analysis_data is not None:
            # 메인 분석 탭
            tabs = st.tabs([
                "📊 기술통계",
                "🔬 통계 분석", 
                "🤖 AI 인사이트",
                "📈 시각화",
                "🎯 최적화",
                "📋 보고서"
            ])
            
            with tabs[0]:
                self._render_descriptive_stats()
            
            with tabs[1]:
                self._render_statistical_analysis()
            
            with tabs[2]:
                self._render_ai_insights()
            
            with tabs[3]:
                self._render_visualizations()
            
            with tabs[4]:
                self._render_optimization()
            
            with tabs[5]:
                self._render_report()
    
    def _render_data_upload(self):
        """데이터 업로드 렌더링"""
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # 파일 업로더
            uploaded_file = st.file_uploader(
                "실험 데이터 파일 선택",
                type=['csv', 'xlsx', 'xls', 'json', 'parquet'],
                help="CSV, Excel, JSON 또는 Parquet 형식의 파일을 업로드하세요"
            )
            
            if uploaded_file is not None:
                try:
                    # 파일 형식에 따른 데이터 로드
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                        df = pd.read_excel(uploaded_file)
                    elif uploaded_file.name.endswith('.json'):
                        df = pd.read_json(uploaded_file)
                    elif uploaded_file.name.endswith('.parquet'):
                        df = pd.read_parquet(uploaded_file)
                    
                    # 데이터 검증
                    if df.empty:
                        st.error("업로드된 파일이 비어있습니다")
                        return
                    
                    # 세션에 저장
                    st.session_state.analysis_data = df
                    st.success(f"✅ 데이터 로드 완료: {len(df)}행 × {len(df.columns)}열")
                    
                    # 데이터 미리보기
                    st.subheader("데이터 미리보기")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    # 컬럼 정보
                    col_info = pd.DataFrame({
                        '컬럼명': df.columns,
                        '데이터 타입': df.dtypes.astype(str),
                        '결측치': df.isnull().sum(),
                        '고유값': df.nunique()
                    })
                    
                    with st.expander("컬럼 정보"):
                        st.dataframe(col_info, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"파일 로드 중 오류 발생: {str(e)}")
        
        with col2:
            # 예제 데이터 로드
            st.subheader("예제 데이터")
            if st.button("샘플 데이터 로드", use_container_width=True):
                self._load_sample_data()
            
            # 프로젝트 데이터 로드
            if st.button("프로젝트 데이터 로드", use_container_width=True):
                self._load_project_data()
        
        # 데이터 전처리 옵션
        if st.session_state.analysis_data is not None:
            st.divider()
            self._render_preprocessing_options()
    
    def _render_preprocessing_options(self):
        """데이터 전처리 옵션"""
        st.subheader("🔧 데이터 전처리")
        
        df = st.session_state.analysis_data
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # 요인 선택
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            st.session_state.selected_factors = st.multiselect(
                "요인 변수 선택",
                numeric_cols,
                default=st.session_state.selected_factors if st.session_state.selected_factors else numeric_cols[:3]
            )
        
        with col2:
            # 반응변수 선택
            st.session_state.selected_responses = st.multiselect(
                "반응 변수 선택",
                numeric_cols,
                default=st.session_state.selected_responses if st.session_state.selected_responses else [numeric_cols[-1]] if numeric_cols else []
            )
        
        with col3:
            # 전처리 옵션
            preprocess_options = st.multiselect(
                "전처리 옵션",
                ["결측치 제거", "이상치 제거", "정규화", "표준화"],
                default=["결측치 제거"]
            )
        
        # 전처리 실행
        if st.button("전처리 실행", type="primary", use_container_width=True):
            processed_df = self._preprocess_data(df, preprocess_options)
            st.session_state.processed_data = processed_df
            st.success("✅ 데이터 전처리 완료")
            
            # 전처리 결과 요약
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("원본 행 수", len(df))
            with col2:
                st.metric("처리 후 행 수", len(processed_df))
            with col3:
                st.metric("제거된 행", len(df) - len(processed_df))
            with col4:
                removal_rate = ((len(df) - len(processed_df)) / len(df) * 100) if len(df) > 0 else 0
                st.metric("제거율", f"{removal_rate:.1f}%")
    
    def _render_descriptive_stats(self):
        """기술통계 렌더링"""
        st.subheader("📊 기술통계")
        
        df = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.analysis_data
        
        if df is None:
            st.warning("데이터를 먼저 업로드해주세요")
            return
        
        # 전체 데이터 요약
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 📈 통계 요약")
            
            # 선택된 변수들의 기술통계
            selected_cols = st.session_state.selected_factors + st.session_state.selected_responses
            if selected_cols:
                stats_df = df[selected_cols].describe().T
                stats_df['CV(%)'] = (stats_df['std'] / stats_df['mean'] * 100).round(2)
                
                st.dataframe(
                    stats_df.style.format({
                        'mean': '{:.3f}',
                        'std': '{:.3f}',
                        'min': '{:.3f}',
                        'max': '{:.3f}',
                        '25%': '{:.3f}',
                        '50%': '{:.3f}',
                        '75%': '{:.3f}',
                        'CV(%)': '{:.1f}'
                    }),
                    use_container_width=True
                )
            else:
                st.info("변수를 선택해주세요")
        
        with col2:
            st.markdown("#### 📊 데이터 품질")
            
            # 데이터 품질 메트릭
            total_cells = df.shape[0] * df.shape[1]
            missing_cells = df.isnull().sum().sum()
            completeness = (1 - missing_cells / total_cells) * 100 if total_cells > 0 else 0
            
            st.metric("완전성", f"{completeness:.1f}%")
            st.metric("결측치", f"{missing_cells:,}")
            
            # 이상치 탐지
            if selected_cols:
                outliers = 0
                for col in selected_cols:
                    if df[col].dtype in [np.float64, np.int64]:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        outliers += ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                
                st.metric("이상치", f"{outliers}")
        
        # 분포 시각화
        st.divider()
        st.markdown("#### 📉 변수 분포")
        
        if selected_cols:
            # 히스토그램
            n_cols = min(3, len(selected_cols))
            cols = st.columns(n_cols)
            
            for i, col in enumerate(selected_cols[:6]):  # 최대 6개까지만 표시
                with cols[i % n_cols]:
                    fig = px.histogram(
                        df, x=col,
                        nbins=30,
                        title=f"{col} 분포",
                        marginal="box"
                    )
                    fig.update_layout(height=300, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
            
            # 상관관계 히트맵
            if len(selected_cols) > 1:
                st.markdown("#### 🔗 상관관계 분석")
                
                corr_matrix = df[selected_cols].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    color_continuous_scale='RdBu',
                    range_color=[-1, 1],
                    title="변수 간 상관관계"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # 높은 상관관계 쌍 표시
                high_corr = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_value = corr_matrix.iloc[i, j]
                        if abs(corr_value) > 0.7:
                            high_corr.append({
                                '변수 1': corr_matrix.columns[i],
                                '변수 2': corr_matrix.columns[j],
                                '상관계수': corr_value
                            })
                
                if high_corr:
                    st.warning("⚠️ 높은 상관관계 발견")
                    st.dataframe(pd.DataFrame(high_corr), use_container_width=True)
    
    def _render_statistical_analysis(self):
        """통계 분석 렌더링"""
        st.subheader("🔬 통계 분석")
        
        df = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.analysis_data
        
        if df is None or not st.session_state.selected_responses:
            st.warning("데이터와 반응변수를 먼저 선택해주세요")
            return
        
        # 분석 유형 선택
        analysis_type = st.selectbox(
            "분석 유형",
            ["분산분석 (ANOVA)", "회귀분석", "반응표면분석 (RSM)", "주성분분석 (PCA)"]
        )
        
        if analysis_type == "분산분석 (ANOVA)":
            self._render_anova_analysis(df)
        elif analysis_type == "회귀분석":
            self._render_regression_analysis(df)
        elif analysis_type == "반응표면분석 (RSM)":
            self._render_rsm_analysis(df)
        elif analysis_type == "주성분분석 (PCA)":
            self._render_pca_analysis(df)
    
    def _render_anova_analysis(self, df: pd.DataFrame):
        """ANOVA 분석"""
        st.markdown("#### 분산분석 (ANOVA)")
        
        # 반응변수 선택
        response = st.selectbox(
            "반응변수",
            st.session_state.selected_responses,
            key="anova_response"
        )
        
        # 요인 선택 (범주형 또는 이산형)
        categorical_factors = []
        for col in st.session_state.selected_factors:
            if df[col].nunique() <= 10:  # 10개 이하의 고유값은 범주형으로 간주
                categorical_factors.append(col)
        
        if not categorical_factors:
            st.warning("범주형 요인이 없습니다. 연속형 변수를 범주화하시겠습니까?")
            if st.button("연속형 변수 범주화"):
                # 연속형 변수를 범주화하는 로직
                pass
            return
        
        selected_factors = st.multiselect(
            "요인 선택",
            categorical_factors,
            default=categorical_factors[:2] if len(categorical_factors) >= 2 else categorical_factors
        )
        
        if selected_factors and response:
            # ANOVA 수행
            formula = f"{response} ~ " + " + ".join(selected_factors)
            if len(selected_factors) > 1:
                formula += " + " + ":".join(selected_factors)  # 교호작용 추가
            
            try:
                model = ols(formula, data=df).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                
                # ANOVA 테이블 표시
                st.markdown("##### ANOVA 테이블")
                anova_display = anova_table.copy()
                anova_display['F-value'] = anova_display['F'].round(3)
                anova_display['p-value'] = anova_display['PR(>F)'].round(4)
                anova_display['유의성'] = anova_display['PR(>F)'].apply(
                    lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                )
                
                st.dataframe(
                    anova_display[['sum_sq', 'df', 'F-value', 'p-value', '유의성']],
                    use_container_width=True
                )
                
                # 모델 요약
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R-squared", f"{model.rsquared:.3f}")
                with col2:
                    st.metric("Adj. R-squared", f"{model.rsquared_adj:.3f}")
                with col3:
                    st.metric("F-statistic", f"{model.fvalue:.3f}")
                
                # 효과 플롯
                st.markdown("##### 주효과 플롯")
                
                for factor in selected_factors:
                    grouped = df.groupby(factor)[response].agg(['mean', 'std', 'count'])
                    grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=grouped.index.astype(str),
                        y=grouped['mean'],
                        error_y=dict(
                            type='data',
                            array=grouped['se'] * 1.96,
                            visible=True
                        ),
                        mode='markers+lines',
                        name=factor
                    ))
                    
                    fig.update_layout(
                        title=f"{factor}의 주효과",
                        xaxis_title=factor,
                        yaxis_title=response,
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # 교호작용 플롯 (2요인인 경우)
                if len(selected_factors) == 2:
                    st.markdown("##### 교호작용 플롯")
                    
                    interaction_data = df.groupby(selected_factors)[response].mean().reset_index()
                    
                    fig = px.line(
                        interaction_data,
                        x=selected_factors[0],
                        y=response,
                        color=selected_factors[1],
                        markers=True,
                        title=f"{selected_factors[0]} × {selected_factors[1]} 교호작용"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # 잔차 분석
                with st.expander("잔차 분석"):
                    residuals = model.resid
                    fitted = model.fittedvalues
                    
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=('잔차 vs 적합값', 'Q-Q 플롯', '잔차 히스토그램', 'Cook의 거리')
                    )
                    
                    # 잔차 vs 적합값
                    fig.add_trace(
                        go.Scatter(x=fitted, y=residuals, mode='markers'),
                        row=1, col=1
                    )
                    
                    # Q-Q 플롯
                    qq = stats.probplot(residuals, dist="norm")
                    fig.add_trace(
                        go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers'),
                        row=1, col=2
                    )
                    fig.add_trace(
                        go.Scatter(x=qq[0][0], y=qq[0][0], mode='lines', line=dict(color='red')),
                        row=1, col=2
                    )
                    
                    # 잔차 히스토그램
                    fig.add_trace(
                        go.Histogram(x=residuals, nbinsx=30),
                        row=2, col=1
                    )
                    
                    # Cook의 거리
                    influence = model.get_influence()
                    cooks_d = influence.cooks_distance[0]
                    fig.add_trace(
                        go.Scatter(y=cooks_d, mode='markers'),
                        row=2, col=2
                    )
                    
                    fig.update_layout(height=800, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"ANOVA 분석 중 오류 발생: {str(e)}")
    
    def _render_regression_analysis(self, df: pd.DataFrame):
        """회귀분석"""
        st.markdown("#### 회귀분석")
        
        # 반응변수 선택
        response = st.selectbox(
            "반응변수",
            st.session_state.selected_responses,
            key="reg_response"
        )
        
        # 독립변수 선택
        predictors = st.multiselect(
            "독립변수",
            st.session_state.selected_factors,
            default=st.session_state.selected_factors
        )
        
        if response and predictors:
            # 회귀 모델 옵션
            col1, col2, col3 = st.columns(3)
            
            with col1:
                model_type = st.selectbox(
                    "모델 유형",
                    ["선형", "다항식", "교호작용 포함"]
                )
            
            with col2:
                if model_type == "다항식":
                    poly_degree = st.number_input("차수", min_value=2, max_value=4, value=2)
            
            with col3:
                include_intercept = st.checkbox("절편 포함", value=True)
            
            # 모델 구축
            X = df[predictors]
            y = df[response]
            
            if model_type == "다항식":
                from sklearn.preprocessing import PolynomialFeatures
                poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
                X_poly = poly.fit_transform(X)
                feature_names = poly.get_feature_names_out(predictors)
                X = pd.DataFrame(X_poly, columns=feature_names, index=X.index)
            
            if include_intercept:
                X = sm.add_constant(X)
            
            # 모델 적합
            model = sm.OLS(y, X).fit()
            
            # 결과 표시
            st.markdown("##### 회귀분석 결과")
            
            # 모델 요약 메트릭
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("R²", f"{model.rsquared:.3f}")
            with col2:
                st.metric("Adj. R²", f"{model.rsquared_adj:.3f}")
            with col3:
                st.metric("F-statistic", f"{model.fvalue:.3f}")
            with col4:
                st.metric("p-value", f"{model.f_pvalue:.4f}")
            
            # 계수 테이블
            coef_df = pd.DataFrame({
                '계수': model.params,
                '표준오차': model.bse,
                't-값': model.tvalues,
                'p-값': model.pvalues,
                '95% CI 하한': model.conf_int()[0],
                '95% CI 상한': model.conf_int()[1]
            })
            
            coef_df['유의성'] = coef_df['p-값'].apply(
                lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            )
            
            st.dataframe(
                coef_df.style.format({
                    '계수': '{:.4f}',
                    '표준오차': '{:.4f}',
                    't-값': '{:.3f}',
                    'p-값': '{:.4f}',
                    '95% CI 하한': '{:.4f}',
                    '95% CI 상한': '{:.4f}'
                }),
                use_container_width=True
            )
            
            # 예측 vs 실제 플롯
            st.markdown("##### 모델 진단")
            
            predictions = model.predict(X)
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('예측 vs 실제', '잔차 플롯', '잔차 분포', 'Scale-Location')
            )
            
            # 예측 vs 실제
            fig.add_trace(
                go.Scatter(x=y, y=predictions, mode='markers', name='데이터'),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=[y.min(), y.max()], y=[y.min(), y.max()], 
                          mode='lines', line=dict(color='red', dash='dash'), name='y=x'),
                row=1, col=1
            )
            
            # 잔차 플롯
            residuals = y - predictions
            fig.add_trace(
                go.Scatter(x=predictions, y=residuals, mode='markers'),
                row=1, col=2
            )
            fig.add_hline(y=0, line_color="red", line_dash="dash", row=1, col=2)
            
            # 잔차 분포
            fig.add_trace(
                go.Histogram(x=residuals, nbinsx=30),
                row=2, col=1
            )
            
            # Scale-Location
            standardized_residuals = residuals / residuals.std()
            fig.add_trace(
                go.Scatter(x=predictions, y=np.sqrt(np.abs(standardized_residuals)), mode='markers'),
                row=2, col=2
            )
            
            fig.update_layout(height=800, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # VIF (다중공선성) 체크
            if len(predictors) > 1:
                with st.expander("다중공선성 진단"):
                    from statsmodels.stats.outliers_influence import variance_inflation_factor
                    
                    vif_data = pd.DataFrame()
                    vif_data["변수"] = X.columns[1:] if include_intercept else X.columns
                    vif_data["VIF"] = [variance_inflation_factor(X.values, i+1 if include_intercept else i) 
                                       for i in range(len(vif_data))]
                    
                    st.dataframe(vif_data, use_container_width=True)
                    
                    high_vif = vif_data[vif_data['VIF'] > 10]
                    if not high_vif.empty:
                        st.warning(f"⚠️ 다중공선성 주의: {', '.join(high_vif['변수'].tolist())}")
    
    def _render_rsm_analysis(self, df: pd.DataFrame):
        """반응표면분석 (RSM)"""
        st.markdown("#### 반응표면분석 (RSM)")
        
        # 반응변수 선택
        response = st.selectbox(
            "반응변수",
            st.session_state.selected_responses,
            key="rsm_response"
        )
        
        # 요인 선택 (최대 3개)
        factors = st.multiselect(
            "요인 선택 (최대 3개)",
            st.session_state.selected_factors,
            default=st.session_state.selected_factors[:2] if len(st.session_state.selected_factors) >= 2 else st.session_state.selected_factors,
            max_selections=3
        )
        
        if response and len(factors) >= 2:
            # 2차 모델 적합
            from sklearn.preprocessing import PolynomialFeatures
            
            X = df[factors]
            y = df[response]
            
            # 2차 다항식 변환
            poly = PolynomialFeatures(degree=2, include_bias=True)
            X_poly = poly.fit_transform(X)
            feature_names = poly.get_feature_names_out(factors)
            
            # 모델 적합
            model = sm.OLS(y, X_poly).fit()
            
            # 모델 요약
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R²", f"{model.rsquared:.3f}")
            with col2:
                st.metric("Adj. R²", f"{model.rsquared_adj:.3f}")
            with col3:
                st.metric("RMSE", f"{np.sqrt(model.mse_resid):.3f}")
            
            # 반응표면 플롯
            if len(factors) == 2:
                st.markdown("##### 3D 반응표면")
                
                # 격자 생성
                x1_range = np.linspace(df[factors[0]].min(), df[factors[0]].max(), 50)
                x2_range = np.linspace(df[factors[1]].min(), df[factors[1]].max(), 50)
                X1, X2 = np.meshgrid(x1_range, x2_range)
                
                # 예측
                X_pred = np.column_stack([X1.ravel(), X2.ravel()])
                X_pred_poly = poly.transform(X_pred)
                Y_pred = model.predict(X_pred_poly).reshape(X1.shape)
                
                # 3D 표면 플롯
                fig = go.Figure(data=[
                    go.Surface(x=x1_range, y=x2_range, z=Y_pred, colorscale='viridis'),
                    go.Scatter3d(x=df[factors[0]], y=df[factors[1]], z=y,
                                mode='markers', marker=dict(size=5, color='red'))
                ])
                
                fig.update_layout(
                    title="반응표면",
                    scene=dict(
                        xaxis_title=factors[0],
                        yaxis_title=factors[1],
                        zaxis_title=response
                    ),
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 등고선 플롯
                st.markdown("##### 등고선 플롯")
                
                fig_contour = go.Figure(data=
                    go.Contour(x=x1_range, y=x2_range, z=Y_pred, 
                              colorscale='viridis', showscale=True)
                )
                
                # 실험점 추가
                fig_contour.add_trace(
                    go.Scatter(x=df[factors[0]], y=df[factors[1]], 
                              mode='markers', marker=dict(size=8, color='red'),
                              name='실험점')
                )
                
                fig_contour.update_layout(
                    xaxis_title=factors[0],
                    yaxis_title=factors[1],
                    height=500
                )
                
                st.plotly_chart(fig_contour, use_container_width=True)
            
            # 최적점 찾기
            st.markdown("##### 최적화")
            
            optimization_goal = st.radio(
                "최적화 목표",
                ["최대화", "최소화", "목표값"],
                horizontal=True
            )
            
            if optimization_goal == "목표값":
                target_value = st.number_input("목표값", value=y.mean())
            
            if st.button("최적점 찾기", type="primary"):
                from scipy.optimize import minimize
                
                def objective(x):
                    x_poly = poly.transform(x.reshape(1, -1))
                    pred = model.predict(x_poly)[0]
                    
                    if optimization_goal == "최대화":
                        return -pred
                    elif optimization_goal == "최소화":
                        return pred
                    else:  # 목표값
                        return (pred - target_value) ** 2
                
                # 초기점
                x0 = X.mean().values
                
                # 경계 설정
                bounds = [(X[col].min(), X[col].max()) for col in factors]
                
                # 최적화
                result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
                
                if result.success:
                    optimal_x = result.x
                    optimal_x_poly = poly.transform(optimal_x.reshape(1, -1))
                    optimal_y = model.predict(optimal_x_poly)[0]
                    
                    st.success("✅ 최적점 발견!")
                    
                    # 최적 조건 표시
                    optimal_df = pd.DataFrame({
                        '요인': factors,
                        '최적값': optimal_x
                    })
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(optimal_df, use_container_width=True)
                    with col2:
                        st.metric(f"예측 {response}", f"{optimal_y:.3f}")
                else:
                    st.error("최적화 실패")
    
    def _render_pca_analysis(self, df: pd.DataFrame):
        """주성분분석 (PCA)"""
        st.markdown("#### 주성분분석 (PCA)")
        
        # 변수 선택
        variables = st.multiselect(
            "분석할 변수",
            st.session_state.selected_factors + st.session_state.selected_responses,
            default=st.session_state.selected_factors
        )
        
        if len(variables) < 2:
            st.warning("최소 2개 이상의 변수를 선택해주세요")
            return
        
        # 데이터 준비
        X = df[variables].dropna()
        
        # 표준화
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA 수행
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # 주성분 개수 선택
        n_components = st.slider(
            "주성분 개수",
            min_value=2,
            max_value=min(len(variables), 10),
            value=min(3, len(variables))
        )
        
        # 설명된 분산
        st.markdown("##### 설명된 분산")
        
        explained_var = pca.explained_variance_ratio_[:n_components]
        cumulative_var = np.cumsum(explained_var)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f'PC{i+1}' for i in range(n_components)],
            y=explained_var * 100,
            name='개별',
            marker_color='lightblue'
        ))
        fig.add_trace(go.Scatter(
            x=[f'PC{i+1}' for i in range(n_components)],
            y=cumulative_var * 100,
            name='누적',
            line=dict(color='red'),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="주성분별 설명된 분산",
            yaxis=dict(title="설명된 분산 (%)"),
            yaxis2=dict(title="누적 설명된 분산 (%)", overlaying='y', side='right'),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 주성분 점수 플롯
        st.markdown("##### 주성분 점수 플롯")
        
        if n_components >= 2:
            fig = px.scatter(
                x=X_pca[:, 0],
                y=X_pca[:, 1],
                labels={'x': f'PC1 ({explained_var[0]*100:.1f}%)', 
                       'y': f'PC2 ({explained_var[1]*100:.1f}%)'},
                title="주성분 점수"
            )
            
            # 원점 추가
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Loading 플롯
        st.markdown("##### Loading 플롯")
        
        loadings = pca.components_[:2].T * np.sqrt(pca.explained_variance_[:2])
        
        fig = go.Figure()
        
        for i, var in enumerate(variables):
            fig.add_trace(go.Scatter(
                x=[0, loadings[i, 0]],
                y=[0, loadings[i, 1]],
                mode='lines+text',
                line=dict(color='blue'),
                text=['', var],
                textposition='top center',
                showlegend=False
            ))
        
        # 원 추가
        theta = np.linspace(0, 2*np.pi, 100)
        fig.add_trace(go.Scatter(
            x=np.cos(theta),
            y=np.sin(theta),
            mode='lines',
            line=dict(color='red', dash='dash'),
            showlegend=False
        ))
        
        fig.update_layout(
            title="변수 Loading",
            xaxis=dict(title=f'PC1 ({explained_var[0]*100:.1f}%)', range=[-1.2, 1.2]),
            yaxis=dict(title=f'PC2 ({explained_var[1]*100:.1f}%)', range=[-1.2, 1.2]),
            height=500,
            width=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 주성분 해석
        st.markdown("##### 주성분 해석")
        
        components_df = pd.DataFrame(
            pca.components_[:n_components].T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=variables
        )
        
        st.dataframe(
            components_df.style.background_gradient(cmap='RdBu', center=0),
            use_container_width=True
        )
    
    def _render_ai_insights(self):
        """AI 인사이트 렌더링"""
        st.subheader("🤖 AI 인사이트")
        
        df = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.analysis_data
        
        if df is None:
            st.warning("데이터를 먼저 업로드해주세요")
            return
        
        # AI 설명 상세도 제어
        col1, col2 = st.columns([4, 1])
        with col2:
            show_details = st.checkbox(
                "🔍 상세 설명",
                value=st.session_state.show_ai_details,
                key="ai_insights_details_toggle"
            )
            st.session_state.show_ai_details = show_details
        
        # AI 분석 유형 선택
        analysis_types = st.multiselect(
            "AI 분석 유형",
            ["패턴 발견", "이상치 탐지", "예측 모델", "최적화 제안", "인과관계 분석"],
            default=["패턴 발견", "최적화 제안"]
        )
        
        # AI 엔진 선택
        available_engines = self.api_manager.get_available_engines()
        selected_engine = st.selectbox(
            "AI 엔진",
            available_engines,
            format_func=lambda x: f"{x} ({'빠름' if x in ['groq', 'sambanova'] else '정밀'})"
        )
        
        # AI 분석 실행
        if st.button("🚀 AI 분석 시작", type="primary", use_container_width=True):
            self._run_ai_analysis(df, analysis_types, selected_engine)
        
        # AI 인사이트 표시
        if st.session_state.ai_insights:
            for i, insight in enumerate(st.session_state.ai_insights):
                with st.container():
                    # 인사이트 헤더
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.markdown(f"### {insight['type']}")
                    with col2:
                        confidence = insight.get('confidence', 0)
                        st.metric("신뢰도", f"{confidence}%")
                    with col3:
                        st.caption(f"by {insight.get('engine', 'AI')}")
                    
                    # 핵심 인사이트
                    st.info(insight['main_insight'])
                    
                    # 상세 설명 (조건부)
                    if show_details and 'details' in insight:
                        with st.expander("상세 분석", expanded=True):
                            tabs = st.tabs(["추론 과정", "근거 데이터", "대안 해석", "제한사항"])
                            
                            with tabs[0]:
                                st.write("**추론 과정**")
                                st.write(insight['details'].get('reasoning', ''))
                            
                            with tabs[1]:
                                st.write("**근거 데이터**")
                                evidence = insight['details'].get('evidence', {})
                                if evidence:
                                    for key, value in evidence.items():
                                        st.write(f"- {key}: {value}")
                            
                            with tabs[2]:
                                st.write("**대안 해석**")
                                alternatives = insight['details'].get('alternatives', [])
                                for alt in alternatives:
                                    st.write(f"- {alt}")
                            
                            with tabs[3]:
                                st.write("**제한사항**")
                                st.write(insight['details'].get('limitations', ''))
                    
                    # 시각화 (있는 경우)
                    if 'visualization' in insight:
                        st.plotly_chart(insight['visualization'], use_container_width=True)
                    
                    # 액션 아이템
                    if 'actions' in insight:
                        st.markdown("**💡 권장 조치**")
                        for action in insight['actions']:
                            st.write(f"- {action}")
                    
                    st.divider()
    
    def _render_visualizations(self):
        """시각화 렌더링"""
        st.subheader("📈 데이터 시각화")
        
        df = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.analysis_data
        
        if df is None:
            st.warning("데이터를 먼저 업로드해주세요")
            return
        
        # 시각화 유형 선택
        viz_type = st.selectbox(
            "시각화 유형",
            ["산점도 행렬", "평행 좌표", "3D 산점도", "히트맵", "시계열", "분포 비교"]
        )
        
        if viz_type == "산점도 행렬":
            self._render_scatter_matrix(df)
        elif viz_type == "평행 좌표":
            self._render_parallel_coordinates(df)
        elif viz_type == "3D 산점도":
            self._render_3d_scatter(df)
        elif viz_type == "히트맵":
            self._render_heatmap(df)
        elif viz_type == "시계열":
            self._render_timeseries(df)
        elif viz_type == "분포 비교":
            self._render_distribution_comparison(df)
    
    def _render_scatter_matrix(self, df: pd.DataFrame):
        """산점도 행렬"""
        variables = st.multiselect(
            "변수 선택",
            st.session_state.selected_factors + st.session_state.selected_responses,
            default=(st.session_state.selected_factors + st.session_state.selected_responses)[:4]
        )
        
        if len(variables) < 2:
            st.warning("최소 2개 이상의 변수를 선택해주세요")
            return
        
        # 색상 변수 (선택사항)
        categorical_cols = [col for col in df.columns if df[col].nunique() <= 10]
        color_var = st.selectbox("색상 변수 (선택사항)", ["없음"] + categorical_cols)
        
        fig = px.scatter_matrix(
            df,
            dimensions=variables,
            color=None if color_var == "없음" else color_var,
            title="산점도 행렬",
            height=800
        )
        
        fig.update_traces(diagonal_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_parallel_coordinates(self, df: pd.DataFrame):
        """평행 좌표"""
        variables = st.multiselect(
            "변수 선택",
            st.session_state.selected_factors + st.session_state.selected_responses,
            default=st.session_state.selected_factors + st.session_state.selected_responses
        )
        
        if len(variables) < 2:
            st.warning("최소 2개 이상의 변수를 선택해주세요")
            return
        
        # 정규화
        df_norm = df[variables].copy()
        for col in variables:
            df_norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        
        # 색상 변수
        color_var = st.selectbox(
            "색상 변수",
            st.session_state.selected_responses,
            index=0 if st.session_state.selected_responses else None
        )
        
        fig = go.Figure(data=
            go.Parcoords(
                line=dict(
                    color=df[color_var] if color_var else None,
                    colorscale='Viridis',
                    showscale=True
                ),
                dimensions=[
                    dict(
                        range=[0, 1],
                        label=var,
                        values=df_norm[var]
                    ) for var in variables
                ]
            )
        )
        
        fig.update_layout(
            title="평행 좌표 플롯",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_3d_scatter(self, df: pd.DataFrame):
        """3D 산점도"""
        numeric_cols = st.session_state.selected_factors + st.session_state.selected_responses
        
        if len(numeric_cols) < 3:
            st.warning("3D 시각화를 위해서는 최소 3개의 숫자형 변수가 필요합니다")
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_var = st.selectbox("X축", numeric_cols)
        with col2:
            y_var = st.selectbox("Y축", [c for c in numeric_cols if c != x_var])
        with col3:
            z_var = st.selectbox("Z축", [c for c in numeric_cols if c not in [x_var, y_var]])
        
        # 색상 및 크기 변수
        col1, col2 = st.columns(2)
        with col1:
            color_var = st.selectbox("색상 변수", ["없음"] + numeric_cols)
        with col2:
            size_var = st.selectbox("크기 변수", ["없음"] + numeric_cols)
        
        fig = px.scatter_3d(
            df,
            x=x_var, y=y_var, z=z_var,
            color=None if color_var == "없음" else color_var,
            size=None if size_var == "없음" else size_var,
            title="3D 산점도"
        )
        
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_heatmap(self, df: pd.DataFrame):
        """히트맵"""
        # 상관관계 유형 선택
        heatmap_type = st.radio(
            "히트맵 유형",
            ["상관관계", "데이터 값"],
            horizontal=True
        )
        
        if heatmap_type == "상관관계":
            variables = st.multiselect(
                "변수 선택",
                st.session_state.selected_factors + st.session_state.selected_responses,
                default=st.session_state.selected_factors + st.session_state.selected_responses
            )
            
            if len(variables) < 2:
                st.warning("최소 2개 이상의 변수를 선택해주세요")
                return
            
            corr_matrix = df[variables].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                color_continuous_scale='RdBu',
                range_color=[-1, 1],
                title="상관관계 히트맵"
            )
            
        else:  # 데이터 값
            # 행/열 변수 선택
            categorical_vars = [col for col in df.columns if df[col].nunique() <= 20]
            
            if len(categorical_vars) < 2:
                st.warning("범주형 변수가 부족합니다")
                return
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                row_var = st.selectbox("행 변수", categorical_vars)
            with col2:
                col_var = st.selectbox("열 변수", [v for v in categorical_vars if v != row_var])
            with col3:
                value_var = st.selectbox("값 변수", st.session_state.selected_responses)
            
            # 피벗 테이블 생성
            pivot_table = df.pivot_table(
                index=row_var,
                columns=col_var,
                values=value_var,
                aggfunc='mean'
            )
            
            fig = px.imshow(
                pivot_table,
                text_auto=True,
                title=f"{value_var} 평균값 히트맵"
            )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_timeseries(self, df: pd.DataFrame):
        """시계열 시각화"""
        # 시간 변수 확인
        time_cols = []
        for col in df.columns:
            try:
                pd.to_datetime(df[col])
                time_cols.append(col)
            except:
                pass
        
        if not time_cols:
            # 인덱스가 시간인지 확인
            try:
                pd.to_datetime(df.index)
                use_index = st.checkbox("인덱스를 시간축으로 사용", value=True)
                if use_index:
                    df = df.copy()
                    df['시간'] = df.index
                    time_cols = ['시간']
            except:
                st.warning("시계열 데이터가 없습니다")
                return
        
        time_var = st.selectbox("시간 변수", time_cols)
        
        # Y축 변수 선택
        y_vars = st.multiselect(
            "Y축 변수",
            st.session_state.selected_responses,
            default=st.session_state.selected_responses[:3] if len(st.session_state.selected_responses) >= 3 else st.session_state.selected_responses
        )
        
        if not y_vars:
            st.warning("Y축 변수를 선택해주세요")
            return
        
        # 시계열 플롯
        fig = go.Figure()
        
        for var in y_vars:
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(df[time_var]),
                y=df[var],
                mode='lines+markers',
                name=var
            ))
        
        fig.update_layout(
            title="시계열 플롯",
            xaxis_title=time_var,
            yaxis_title="값",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 이동평균 추가 옵션
        if st.checkbox("이동평균 추가"):
            window_size = st.slider("윈도우 크기", 3, 30, 7)
            
            fig_ma = go.Figure()
            
            for var in y_vars:
                # 원본 데이터
                fig_ma.add_trace(go.Scatter(
                    x=pd.to_datetime(df[time_var]),
                    y=df[var],
                    mode='lines',
                    name=f"{var} (원본)",
                    line=dict(width=1)
                ))
                
                # 이동평균
                ma = df[var].rolling(window=window_size).mean()
                fig_ma.add_trace(go.Scatter(
                    x=pd.to_datetime(df[time_var]),
                    y=ma,
                    mode='lines',
                    name=f"{var} (MA-{window_size})",
                    line=dict(width=3)
                ))
            
            fig_ma.update_layout(
                title=f"시계열 플롯 (이동평균 포함)",
                xaxis_title=time_var,
                yaxis_title="값",
                height=500
            )
            
            st.plotly_chart(fig_ma, use_container_width=True)
    
    def _render_distribution_comparison(self, df: pd.DataFrame):
        """분포 비교"""
        # 비교할 변수 선택
        compare_var = st.selectbox(
            "비교할 변수",
            st.session_state.selected_responses
        )
        
        # 그룹 변수 선택
        categorical_vars = [col for col in df.columns if df[col].nunique() <= 10]
        group_var = st.selectbox("그룹 변수", categorical_vars)
        
        if compare_var and group_var:
            # 비교 유형
            plot_type = st.radio(
                "플롯 유형",
                ["박스 플롯", "바이올린 플롯", "히스토그램", "밀도 플롯"],
                horizontal=True
            )
            
            if plot_type == "박스 플롯":
                fig = px.box(df, x=group_var, y=compare_var, title=f"{compare_var} 분포 비교")
            elif plot_type == "바이올린 플롯":
                fig = px.violin(df, x=group_var, y=compare_var, box=True, title=f"{compare_var} 분포 비교")
            elif plot_type == "히스토그램":
                fig = px.histogram(df, x=compare_var, color=group_var, marginal="box", title=f"{compare_var} 분포 비교")
            else:  # 밀도 플롯
                fig = go.Figure()
                for group in df[group_var].unique():
                    group_data = df[df[group_var] == group][compare_var]
                    fig.add_trace(go.Violin(
                        x=group_data,
                        name=str(group),
                        side='positive',
                        meanline_visible=True
                    ))
                fig.update_layout(title=f"{compare_var} 밀도 비교", xaxis_title=compare_var)
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # 통계 검정
            with st.expander("통계 검정"):
                groups = [df[df[group_var] == g][compare_var].dropna() for g in df[group_var].unique()]
                
                if len(groups) == 2:
                    # t-검정
                    statistic, p_value = stats.ttest_ind(groups[0], groups[1])
                    st.write(f"**t-검정**")
                    st.write(f"- t-statistic: {statistic:.4f}")
                    st.write(f"- p-value: {p_value:.4f}")
                    st.write(f"- 결론: {'유의한 차이' if p_value < 0.05 else '유의한 차이 없음'} (α=0.05)")
                else:
                    # ANOVA
                    statistic, p_value = stats.f_oneway(*groups)
                    st.write(f"**일원분산분석 (One-way ANOVA)**")
                    st.write(f"- F-statistic: {statistic:.4f}")
                    st.write(f"- p-value: {p_value:.4f}")
                    st.write(f"- 결론: {'유의한 차이' if p_value < 0.05 else '유의한 차이 없음'} (α=0.05)")
    
    def _render_optimization(self):
        """최적화 렌더링"""
        st.subheader("🎯 최적화")
        
        df = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.analysis_data
        
        if df is None:
            st.warning("데이터를 먼저 업로드해주세요")
            return
        
        # 최적화 목표 설정
        st.markdown("#### 최적화 목표 설정")
        
        optimization_targets = []
        
        for response in st.session_state.selected_responses:
            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
            
            with col1:
                st.write(f"**{response}**")
            
            with col2:
                goal = st.selectbox(
                    "목표",
                    ["최대화", "최소화", "목표값", "제외"],
                    key=f"opt_goal_{response}"
                )
            
            with col3:
                if goal == "목표값":
                    target = st.number_input(
                        "목표값",
                        value=df[response].mean(),
                        key=f"opt_target_{response}"
                    )
                else:
                    target = None
            
            with col4:
                weight = st.number_input(
                    "가중치",
                    min_value=0.0,
                    max_value=10.0,
                    value=1.0,
                    step=0.1,
                    key=f"opt_weight_{response}"
                )
            
            if goal != "제외":
                optimization_targets.append({
                    'response': response,
                    'goal': goal,
                    'target': target,
                    'weight': weight
                })
        
        if not optimization_targets:
            st.warning("최소 1개 이상의 최적화 목표를 설정해주세요")
            return
        
        # 제약조건 설정
        st.markdown("#### 제약조건")
        
        constraints = []
        if st.checkbox("제약조건 추가"):
            for factor in st.session_state.selected_factors:
                col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                
                with col1:
                    st.write(f"**{factor}**")
                
                with col2:
                    min_val = st.number_input(
                        "최소값",
                        value=df[factor].min(),
                        key=f"const_min_{factor}"
                    )
                
                with col3:
                    max_val = st.number_input(
                        "최대값",
                        value=df[factor].max(),
                        key=f"const_max_{factor}"
                    )
                
                with col4:
                    if st.checkbox("적용", key=f"const_apply_{factor}"):
                        constraints.append({
                            'factor': factor,
                            'min': min_val,
                            'max': max_val
                        })
        
        # 최적화 방법 선택
        col1, col2 = st.columns(2)
        
        with col1:
            optimization_method = st.selectbox(
                "최적화 방법",
                ["반응표면법", "유전자 알고리즘", "베이지안 최적화", "그리드 탐색"]
            )
        
        with col2:
            n_iterations = st.number_input(
                "반복 횟수",
                min_value=10,
                max_value=1000,
                value=100
            )
        
        # 최적화 실행
        if st.button("🚀 최적화 실행", type="primary", use_container_width=True):
            with st.spinner("최적화 진행 중..."):
                optimal_solution = self._run_optimization(
                    df, 
                    optimization_targets,
                    constraints,
                    optimization_method,
                    n_iterations
                )
                
                if optimal_solution:
                    st.success("✅ 최적화 완료!")
                    
                    # 최적 조건 표시
                    st.markdown("#### 최적 조건")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        optimal_df = pd.DataFrame([optimal_solution['factors']])
                        st.dataframe(optimal_df.T, use_container_width=True)
                    
                    with col2:
                        st.markdown("**예측 결과**")
                        for target in optimization_targets:
                            pred_value = optimal_solution['predictions'][target['response']]
                            st.metric(target['response'], f"{pred_value:.3f}")
                    
                    # 최적화 히스토리
                    if 'history' in optimal_solution:
                        st.markdown("#### 최적화 과정")
                        
                        fig = go.Figure()
                        for i, target in enumerate(optimization_targets):
                            fig.add_trace(go.Scatter(
                                y=optimal_solution['history'][target['response']],
                                mode='lines',
                                name=target['response']
                            ))
                        
                        fig.update_layout(
                            title="최적화 수렴 과정",
                            xaxis_title="반복",
                            yaxis_title="목적함수 값",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # 민감도 분석
                    if st.checkbox("민감도 분석"):
                        self._render_sensitivity_analysis(
                            optimal_solution,
                            df,
                            optimization_targets
                        )
    
    def _render_report(self):
        """보고서 생성"""
        st.subheader("📋 분석 보고서")
        
        # 보고서 옵션
        col1, col2, col3 = st.columns(3)
        
        with col1:
            report_format = st.selectbox(
                "보고서 형식",
                ["대화형 HTML", "PDF", "Word", "PowerPoint"]
            )
        
        with col2:
            report_style = st.selectbox(
                "보고서 스타일",
                ["기술 보고서", "경영진 요약", "학술 논문", "프레젠테이션"]
            )
        
        with col3:
            include_options = st.multiselect(
                "포함 항목",
                ["기술통계", "통계분석", "시각화", "AI 인사이트", "최적화 결과"],
                default=["기술통계", "시각화", "AI 인사이트"]
            )
        
        # 보고서 미리보기
        if st.button("보고서 미리보기", use_container_width=True):
            with st.spinner("보고서 생성 중..."):
                report_content = self._generate_report(
                    report_format,
                    report_style,
                    include_options
                )
                
                if report_content:
                    st.success("✅ 보고서 생성 완료!")
                    
                    # 미리보기 표시
                    with st.expander("보고서 미리보기", expanded=True):
                        if report_format == "대화형 HTML":
                            st.components.v1.html(report_content, height=800)
                        else:
                            st.text_area("보고서 내용", report_content, height=400)
        
        # 보고서 다운로드
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 다운로드", type="primary", use_container_width=True):
                # 다운로드 로직
                st.download_button(
                    "다운로드",
                    data=self._export_report(report_format),
                    file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{report_format.lower()}",
                    mime=self._get_mime_type(report_format)
                )
        
        with col2:
            if st.button("📧 이메일 전송", use_container_width=True):
                # 이메일 전송 로직
                st.info("이메일 전송 기능은 준비 중입니다")
        
        with col3:
            if st.button("☁️ 클라우드 저장", use_container_width=True):
                # 클라우드 저장 로직
                st.info("클라우드 저장 기능은 준비 중입니다")
    
    def _preprocess_data(self, df: pd.DataFrame, options: List[str]) -> pd.DataFrame:
        """데이터 전처리"""
        processed_df = df.copy()
        
        # 결측치 제거
        if "결측치 제거" in options:
            before_rows = len(processed_df)
            processed_df = processed_df.dropna()
            st.info(f"결측치 제거: {before_rows - len(processed_df)}행 제거됨")
        
        # 이상치 제거 (IQR 방법)
        if "이상치 제거" in options:
            numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
            outliers_removed = 0
            
            for col in numeric_cols:
                Q1 = processed_df[col].quantile(0.25)
                Q3 = processed_df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                before_rows = len(processed_df)
                processed_df = processed_df[
                    (processed_df[col] >= lower_bound) & 
                    (processed_df[col] <= upper_bound)
                ]
                outliers_removed += before_rows - len(processed_df)
            
            st.info(f"이상치 제거: 총 {outliers_removed}행 제거됨")
        
        # 정규화
        if "정규화" in options:
            numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                min_val = processed_df[col].min()
                max_val = processed_df[col].max()
                if max_val > min_val:
                    processed_df[col] = (processed_df[col] - min_val) / (max_val - min_val)
            st.info("정규화 완료 (0-1 범위)")
        
        # 표준화
        if "표준화" in options:
            numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
            scaler = StandardScaler()
            processed_df[numeric_cols] = scaler.fit_transform(processed_df[numeric_cols])
            st.info("표준화 완료 (평균=0, 표준편차=1)")
        
        return processed_df
    
    def _load_sample_data(self):
        """샘플 데이터 로드"""
        # 샘플 데이터 생성
        np.random.seed(42)
        n_samples = 100
        
        sample_data = pd.DataFrame({
            'Temperature': np.random.uniform(20, 100, n_samples),
            'Pressure': np.random.uniform(1, 10, n_samples),
            'Time': np.random.uniform(10, 120, n_samples),
            'Catalyst': np.random.choice(['A', 'B', 'C'], n_samples),
            'Yield': np.random.normal(75, 10, n_samples),
            'Purity': np.random.normal(95, 3, n_samples),
            'Cost': np.random.uniform(10, 50, n_samples)
        })
        
        # 상관관계 추가
        sample_data['Yield'] += 0.3 * sample_data['Temperature'] - 0.2 * sample_data['Pressure']
        sample_data['Purity'] += 0.2 * sample_data['Time'] - 0.1 * sample_data['Temperature']
        
        st.session_state.analysis_data = sample_data
        st.success("✅ 샘플 데이터 로드 완료")
        st.rerun()
    
    def _load_project_data(self):
        """프로젝트 데이터 로드"""
        # 현재 프로젝트의 실험 데이터 로드
        if 'current_project' in st.session_state:
            project_id = st.session_state.current_project.get('id')
            experiment_data = self.db_manager.get_experiment_data(project_id)
            
            if experiment_data:
                st.session_state.analysis_data = pd.DataFrame(experiment_data)
                st.success("✅ 프로젝트 데이터 로드 완료")
                st.rerun()
            else:
                st.warning("프로젝트에 실험 데이터가 없습니다")
        else:
            st.warning("먼저 프로젝트를 선택해주세요")
    
    def _run_ai_analysis(self, df: pd.DataFrame, analysis_types: List[str], engine: str):
        """AI 분석 실행"""
        insights = []
        
        for analysis_type in analysis_types:
            with st.spinner(f"{analysis_type} 분석 중..."):
                # 데이터 요약 준비
                data_summary = self._prepare_data_summary(df)
                
                # AI 프롬프트 생성
                prompt = self._create_analysis_prompt(analysis_type, data_summary)
                
                # AI 호출
                response = self.api_manager.call_ai(
                    prompt,
                    engine=engine,
                    response_format="structured",
                    detail_level='detailed' if st.session_state.show_ai_details else 'simple'
                )
                
                if response:
                    insight = {
                        'type': analysis_type,
                        'engine': engine,
                        'main_insight': response.get('main_insight', ''),
                        'confidence': response.get('confidence', 85),
                        'details': {
                            'reasoning': response.get('reasoning', ''),
                            'evidence': response.get('evidence', {}),
                            'alternatives': response.get('alternatives', []),
                            'limitations': response.get('limitations', '')
                        },
                        'actions': response.get('recommended_actions', [])
                    }
                    
                    # 시각화 생성 (가능한 경우)
                    if 'visualization_data' in response:
                        insight['visualization'] = self._create_insight_visualization(
                            response['visualization_data'],
                            analysis_type
                        )
                    
                    insights.append(insight)
        
        st.session_state.ai_insights = insights
        st.success("✅ AI 분석 완료!")
        st.rerun()
    
    def _prepare_data_summary(self, df: pd.DataFrame) -> Dict:
        """AI 분석을 위한 데이터 요약"""
        summary = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'statistics': df.describe().to_dict(),
            'correlations': df.corr().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 1 else {}
        }
        
        # 선택된 요인과 반응변수 정보
        summary['factors'] = st.session_state.selected_factors
        summary['responses'] = st.session_state.selected_responses
        
        return summary
    
    def _create_analysis_prompt(self, analysis_type: str, data_summary: Dict) -> str:
        """AI 분석 프롬프트 생성"""
        base_prompt = f"""
        실험 데이터를 분석하여 {analysis_type}에 대한 인사이트를 제공해주세요.
        
        데이터 요약:
        - 크기: {data_summary['shape']}
        - 요인: {', '.join(data_summary['factors'])}
        - 반응변수: {', '.join(data_summary['responses'])}
        
        통계 요약:
        {json.dumps(data_summary['statistics'], ensure_ascii=False, indent=2)}
        
        상관관계:
        {json.dumps(data_summary['correlations'], ensure_ascii=False, indent=2)}
        """
        
        specific_prompts = {
            "패턴 발견": """
            다음을 분석해주세요:
            1. 요인과 반응변수 간의 주요 관계
            2. 예상치 못한 패턴이나 트렌드
            3. 상호작용 효과
            4. 실험 설계의 효율성
            """,
            
            "이상치 탐지": """
            다음을 분석해주세요:
            1. 통계적 이상치 식별
            2. 이상치의 가능한 원인
            3. 이상치가 결과에 미치는 영향
            4. 처리 방법 제안
            """,
            
            "예측 모델": """
            다음을 분석해주세요:
            1. 최적의 예측 모델 구조
            2. 중요 변수 순위
            3. 모델 성능 예상
            4. 추가 실험 제안
            """,
            
            "최적화 제안": """
            다음을 분석해주세요:
            1. 최적 조건 예측
            2. 개선 가능 영역
            3. 제약조건 고려사항
            4. 실행 가능한 다음 단계
            """,
            
            "인과관계 분석": """
            다음을 분석해주세요:
            1. 인과관계 가능성이 높은 요인
            2. 매개변수 또는 조절변수
            3. 인과관계 검증 방법
            4. 추가 실험 설계
            """
        }
        
        full_prompt = base_prompt + "\n" + specific_prompts.get(analysis_type, "")
        
        if st.session_state.show_ai_details:
            full_prompt += """
            
            응답 형식:
            1. 핵심 인사이트 (간단명료하게)
            2. 상세 분석:
               - 추론 과정: 단계별 분석 과정
               - 근거 데이터: 구체적인 수치와 통계
               - 대안 해석: 다른 가능한 해석 2-3개
               - 제한사항: 분석의 한계와 가정
            3. 권장 조치사항 (구체적이고 실행 가능한)
            4. 신뢰도 (0-100%)와 그 근거
            """
        
        return full_prompt
    
    def _create_insight_visualization(self, viz_data: Dict, analysis_type: str):
        """인사이트 시각화 생성"""
        # 분석 유형에 따른 시각화 생성
        if analysis_type == "패턴 발견":
            # 산점도 또는 상관관계 플롯
            pass
        elif analysis_type == "이상치 탐지":
            # 박스플롯 또는 산점도
            pass
        # ... 기타 시각화
        
        return None  # 임시
    
    def _run_optimization(self, df: pd.DataFrame, targets: List[Dict], 
                         constraints: List[Dict], method: str, n_iter: int) -> Dict:
        """최적화 실행"""
        # 간단한 최적화 예시 (실제로는 더 복잡한 알고리즘 필요)
        
        # 목적함수 정의
        def objective_function(x):
            # 예측 모델 사용 (여기서는 간단한 선형 모델)
            score = 0
            for target in targets:
                # 실제로는 학습된 모델로 예측
                pred = np.random.normal(75, 5)  # 임시
                
                if target['goal'] == '최대화':
                    score += target['weight'] * pred
                elif target['goal'] == '최소화':
                    score -= target['weight'] * pred
                else:  # 목표값
                    score -= target['weight'] * abs(pred - target['target'])
            
            return -score  # 최소화 문제로 변환
        
        # 최적화 실행
        from scipy.optimize import minimize
        
        # 초기값
        x0 = [df[f].mean() for f in st.session_state.selected_factors]
        
        # 경계 설정
        bounds = []
        for factor in st.session_state.selected_factors:
            constraint = next((c for c in constraints if c['factor'] == factor), None)
            if constraint:
                bounds.append((constraint['min'], constraint['max']))
            else:
                bounds.append((df[factor].min(), df[factor].max()))
        
        # 최적화
        result = minimize(objective_function, x0, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            optimal_solution = {
                'factors': {f: v for f, v in zip(st.session_state.selected_factors, result.x)},
                'predictions': {t['response']: np.random.normal(75, 5) for t in targets},  # 임시
                'score': -result.fun
            }
            
            return optimal_solution
        
        return None
    
    def _render_sensitivity_analysis(self, optimal_solution: Dict, 
                                   df: pd.DataFrame, targets: List[Dict]):
        """민감도 분석"""
        st.markdown("#### 민감도 분석")
        
        # 각 요인의 영향도 분석
        base_values = optimal_solution['factors']
        
        sensitivity_data = []
        
        for factor in st.session_state.selected_factors:
            # ±10% 변화
            variations = np.linspace(0.9, 1.1, 21)
            
            effects = {}
            for response in st.session_state.selected_responses:
                effects[response] = []
                
                for var in variations:
                    # 요인 값 변경
                    test_values = base_values.copy()
                    test_values[factor] = base_values[factor] * var
                    
                    # 예측 (실제로는 모델 사용)
                    pred = np.random.normal(75, 2)  # 임시
                    effects[response].append(pred)
            
            sensitivity_data.append({
                'factor': factor,
                'variations': variations,
                'effects': effects
            })
        
        # 시각화
        for response in st.session_state.selected_responses:
            fig = go.Figure()
            
            for sens_data in sensitivity_data:
                fig.add_trace(go.Scatter(
                    x=(sens_data['variations'] - 1) * 100,
                    y=sens_data['effects'][response],
                    mode='lines',
                    name=sens_data['factor']
                ))
            
            fig.update_layout(
                title=f"{response}에 대한 민감도",
                xaxis_title="요인 변화 (%)",
                yaxis_title=response,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _generate_report(self, format: str, style: str, 
                        include_options: List[str]) -> str:
        """보고서 생성"""
        # 보고서 내용 생성
        report_content = f"""
        # 실험 데이터 분석 보고서
        
        생성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        ## 요약
        
        본 보고서는 {len(st.session_state.analysis_data)}개의 실험 데이터를 분석한 결과입니다.
        """
        
        # 각 섹션 추가
        if "기술통계" in include_options:
            report_content += "\n\n## 기술통계\n\n"
            # 통계 요약 추가
        
        if "통계분석" in include_options:
            report_content += "\n\n## 통계분석\n\n"
            # 분석 결과 추가
        
        # ... 기타 섹션
        
        return report_content
    
    def _export_report(self, format: str) -> bytes:
        """보고서 내보내기"""
        # 형식별 내보내기 로직
        report_content = self._generate_report(format, "", [])
        
        if format == "PDF":
            # PDF 생성 로직
            pass
        elif format == "Word":
            # Word 생성 로직
            pass
        
        return report_content.encode()
    
    def _get_mime_type(self, format: str) -> str:
        """MIME 타입 반환"""
        mime_types = {
            "PDF": "application/pdf",
            "Word": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "대화형 HTML": "text/html",
            "PowerPoint": "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        }
        return mime_types.get(format, "application/octet-stream")

# 보조 함수
from plotly.subplots import make_subplots

# 페이지 렌더링
def render():
    """페이지 렌더링 함수"""
    page = DataAnalysisPage()
    page.render()

# 메인 실행
if __name__ == "__main__":
    render()
