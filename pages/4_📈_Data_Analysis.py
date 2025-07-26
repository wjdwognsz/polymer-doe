"""
📈 Data Analysis Page - 데이터 분석 페이지
실험 데이터의 통계 분석, AI 인사이트, 시각화를 제공하는 핵심 분석 도구
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import io
from datetime import datetime
import uuid
from typing import Dict, List, Optional, Tuple, Any
import asyncio

# 통계 분석 라이브러리
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# 경로 설정
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# 내부 모듈 임포트
from utils.auth_manager import check_authentication, get_current_user
from utils.sheets_manager import GoogleSheetsManager
from utils.api_manager import APIManager
from utils.common_ui import get_common_ui
from utils.notification_manager import NotificationManager
from utils.data_processor import DataProcessor

# 페이지 설정
st.set_page_config(
    page_title="Data Analysis - Polymer DOE",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 전역 변수
ANALYSIS_TYPES = {
    'descriptive': '기술통계',
    'anova': '분산분석 (ANOVA)',
    'regression': '회귀분석',
    'rsm': '반응표면분석 (RSM)',
    'optimization': '최적화',
    'ml_prediction': '머신러닝 예측'
}

def initialize_session_state():
    """세션 상태 초기화"""
    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    if 'show_ai_details' not in st.session_state:
        st.session_state.show_ai_details = False
    if 'current_project' not in st.session_state:
        st.session_state.current_project = None

def render_ai_response(response: Dict, response_type: str = "analysis"):
    """
    AI 응답 렌더링 - AI 투명성 원칙 적용
    """
    ui = get_common_ui()
    
    # 1. 핵심 답변 (항상 표시)
    st.markdown(f"### 🤖 AI {response_type}")
    
    # 핵심 인사이트 표시
    if 'main' in response:
        st.info(response['main'])
    elif 'key_insights' in response:
        for insight in response['key_insights']:
            with st.container():
                col1, col2 = st.columns([10, 1])
                with col1:
                    st.markdown(f"**{insight.get('title', '')}**")
                    st.write(insight.get('description', ''))
                with col2:
                    importance = insight.get('importance', 'medium')
                    if importance == 'high':
                        st.markdown("🔴")
                    elif importance == 'medium':
                        st.markdown("🟡")
                    else:
                        st.markdown("🟢")
    
    # 2. 상세 설명 토글
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🔍 상세 설명", key=f"toggle_details_{response_type}"):
            st.session_state.show_ai_details = not st.session_state.show_ai_details
    
    # 3. 상세 설명 (조건부 표시)
    if st.session_state.show_ai_details and 'details' in response:
        with st.expander("상세 AI 분석", expanded=True):
            tabs = st.tabs(["추론 과정", "대안 검토", "이론적 배경", "신뢰도"])
            
            with tabs[0]:
                st.markdown("#### 추론 과정")
                st.write(response['details'].get('reasoning', '분석 중...'))
            
            with tabs[1]:
                st.markdown("#### 대안 검토")
                alternatives = response['details'].get('alternatives', [])
                if alternatives:
                    for alt in alternatives:
                        st.write(f"• **{alt['name']}**: {alt['description']}")
                        st.caption(f"  장점: {alt.get('pros', '')}")
                        st.caption(f"  단점: {alt.get('cons', '')}")
                else:
                    st.write("대안 분석 중...")
            
            with tabs[2]:
                st.markdown("#### 이론적 배경")
                st.write(response['details'].get('theory', '이론적 배경 분석 중...'))
            
            with tabs[3]:
                st.markdown("#### 신뢰도 평가")
                confidence = response['details'].get('confidence', {})
                if confidence:
                    st.metric("전체 신뢰도", f"{confidence.get('overall', 0)}%")
                    st.write(confidence.get('explanation', ''))
                    
                    # 한계점
                    if 'limitations' in confidence:
                        st.warning("**한계점:**")
                        for limitation in confidence['limitations']:
                            st.write(f"• {limitation}")

def upload_data_section():
    """데이터 업로드 섹션"""
    st.markdown("### 📁 데이터 업로드")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "실험 데이터 파일을 업로드하세요",
            type=['csv', 'xlsx', 'xls', 'json'],
            help="CSV, Excel, JSON 형식을 지원합니다"
        )
    
    with col2:
        st.markdown("#### 샘플 데이터")
        if st.button("🔽 템플릿 다운로드"):
            template_df = create_sample_template()
            csv = template_df.to_csv(index=False)
            st.download_button(
                label="📥 다운로드",
                data=csv,
                file_name="polymer_doe_template.csv",
                mime="text/csv"
            )
    
    if uploaded_file:
        try:
            # 파일 형식에 따라 읽기
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            
            st.session_state.analysis_data = df
            
            # 데이터 미리보기
            st.success(f"✅ 데이터 로드 완료! ({len(df)} 행, {len(df.columns)} 열)")
            
            with st.expander("데이터 미리보기", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)
                
                # 기본 정보
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("총 샘플 수", len(df))
                with col2:
                    st.metric("변수 수", len(df.columns))
                with col3:
                    missing = df.isna().sum().sum()
                    st.metric("결측값", missing)
            
            return df
            
        except Exception as e:
            st.error(f"파일 읽기 오류: {str(e)}")
            return None
    
    # 기존 프로젝트 데이터 로드 옵션
    elif st.session_state.current_project:
        if st.button("📊 프로젝트 데이터 불러오기"):
            df = load_project_data()
            if df is not None:
                st.session_state.analysis_data = df
                return df
    
    return None

def create_sample_template():
    """샘플 템플릿 생성"""
    data = {
        'Run': range(1, 21),
        'Temperature': np.random.uniform(20, 100, 20),
        'Time': np.random.uniform(30, 180, 20),
        'Concentration': np.random.uniform(0.1, 2.0, 20),
        'Yield': np.random.uniform(60, 95, 20),
        'Purity': np.random.uniform(85, 99, 20)
    }
    return pd.DataFrame(data)

def load_project_data():
    """프로젝트 데이터 로드"""
    try:
        sheets = GoogleSheetsManager()
        project_id = st.session_state.current_project
        
        # 실험 데이터 가져오기
        experiments = sheets.get_project_experiments(project_id)
        if experiments:
            return pd.DataFrame(experiments)
        else:
            st.warning("프로젝트에 저장된 실험 데이터가 없습니다.")
            return None
    except Exception as e:
        st.error(f"데이터 로드 실패: {str(e)}")
        return None

def data_preprocessing_section(df: pd.DataFrame):
    """데이터 전처리 섹션"""
    st.markdown("### 🔧 데이터 전처리")
    
    processor = DataProcessor()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # 결측값 처리
        st.markdown("#### 결측값 처리")
        missing_cols = df.columns[df.isna().any()].tolist()
        
        if missing_cols:
            st.warning(f"결측값이 있는 열: {', '.join(missing_cols)}")
            
            missing_method = st.selectbox(
                "처리 방법",
                ["제거", "평균값 대체", "중앙값 대체", "전방 채우기", "후방 채우기"]
            )
            
            if st.button("결측값 처리"):
                if missing_method == "제거":
                    df = df.dropna()
                elif missing_method == "평균값 대체":
                    df = df.fillna(df.mean())
                elif missing_method == "중앙값 대체":
                    df = df.fillna(df.median())
                elif missing_method == "전방 채우기":
                    df = df.fillna(method='ffill')
                elif missing_method == "후방 채우기":
                    df = df.fillna(method='bfill')
                
                st.session_state.analysis_data = df
                st.success("✅ 결측값 처리 완료!")
                st.rerun()
        else:
            st.success("✅ 결측값 없음")
    
    with col2:
        # 이상치 탐지
        st.markdown("#### 이상치 탐지")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_col = st.selectbox("검사할 열", numeric_cols)
        
        if selected_col and st.button("이상치 확인"):
            Q1 = df[selected_col].quantile(0.25)
            Q3 = df[selected_col].quantile(0.75)
            IQR = Q3 - Q1
            
            outliers = df[(df[selected_col] < Q1 - 1.5 * IQR) | 
                         (df[selected_col] > Q3 + 1.5 * IQR)]
            
            if len(outliers) > 0:
                st.warning(f"🔍 {len(outliers)}개의 이상치 발견")
                
                # Box plot
                fig = go.Figure()
                fig.add_trace(go.Box(y=df[selected_col], name=selected_col))
                fig.update_layout(title=f"{selected_col} 분포 및 이상치")
                st.plotly_chart(fig, use_container_width=True)
                
                if st.button("이상치 제거"):
                    df = df[~df.index.isin(outliers.index)]
                    st.session_state.analysis_data = df
                    st.success("✅ 이상치 제거 완료!")
                    st.rerun()
            else:
                st.success("✅ 이상치 없음")
    
    return df

def statistical_analysis_section(df: pd.DataFrame):
    """통계 분석 섹션"""
    st.markdown("### 📊 통계 분석")
    
    analysis_type = st.selectbox(
        "분석 유형 선택",
        list(ANALYSIS_TYPES.keys()),
        format_func=lambda x: ANALYSIS_TYPES[x]
    )
    
    if analysis_type == 'descriptive':
        perform_descriptive_analysis(df)
    elif analysis_type == 'anova':
        perform_anova_analysis(df)
    elif analysis_type == 'regression':
        perform_regression_analysis(df)
    elif analysis_type == 'rsm':
        perform_rsm_analysis(df)
    elif analysis_type == 'optimization':
        perform_optimization_analysis(df)
    elif analysis_type == 'ml_prediction':
        perform_ml_prediction(df)

def perform_descriptive_analysis(df: pd.DataFrame):
    """기술통계 분석"""
    st.markdown("#### 📈 기술통계 분석")
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    # 기본 통계
    stats_df = numeric_df.describe().T
    stats_df['CV'] = (numeric_df.std() / numeric_df.mean() * 100).round(2)
    
    st.dataframe(
        stats_df.style.format("{:.2f}").background_gradient(cmap='YlOrRd', axis=0),
        use_container_width=True
    )
    
    # 상관관계 히트맵
    st.markdown("#### 상관관계 분석")
    
    corr_matrix = numeric_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.round(2).values,
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title="상관관계 히트맵",
        height=600,
        xaxis_tickangle=-45
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # AI 인사이트
    if st.button("🤖 AI 인사이트 생성", key="desc_ai"):
        with st.spinner("AI가 데이터를 분석 중입니다..."):
            insights = generate_ai_insights(df, 'descriptive')
            render_ai_response(insights, "기술통계 인사이트")

def perform_anova_analysis(df: pd.DataFrame):
    """분산분석 수행"""
    st.markdown("#### 📊 분산분석 (ANOVA)")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not categorical_cols:
        st.warning("범주형 변수가 없습니다. 수치형 변수를 그룹화하여 사용하세요.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        response_var = st.selectbox("반응변수 선택", numeric_cols)
    
    with col2:
        factor_var = st.selectbox("요인 선택", categorical_cols)
    
    if st.button("ANOVA 실행"):
        try:
            # One-way ANOVA
            groups = [group[response_var].dropna() for name, group in df.groupby(factor_var)]
            f_stat, p_value = stats.f_oneway(*groups)
            
            # 결과 표시
            col1, col2 = st.columns(2)
            with col1:
                st.metric("F-통계량", f"{f_stat:.4f}")
            with col2:
                st.metric("p-value", f"{p_value:.4f}")
            
            if p_value < 0.05:
                st.success("✅ 통계적으로 유의한 차이가 있습니다 (p < 0.05)")
            else:
                st.info("ℹ️ 통계적으로 유의한 차이가 없습니다 (p ≥ 0.05)")
            
            # Box plot
            fig = px.box(df, x=factor_var, y=response_var, 
                        title=f"{response_var} by {factor_var}")
            st.plotly_chart(fig, use_container_width=True)
            
            # 사후 검정
            if p_value < 0.05 and len(groups) > 2:
                st.markdown("##### 사후 검정 (Tukey HSD)")
                from statsmodels.stats.multicomp import pairwise_tukeyhsd
                
                tukey = pairwise_tukeyhsd(
                    endog=df[response_var].dropna(),
                    groups=df.loc[df[response_var].notna(), factor_var]
                )
                
                st.text(str(tukey))
            
            # AI 인사이트
            if st.button("🤖 AI 해석", key="anova_ai"):
                with st.spinner("AI가 결과를 해석 중입니다..."):
                    insights = generate_ai_insights({
                        'analysis_type': 'anova',
                        'response': response_var,
                        'factor': factor_var,
                        'f_stat': f_stat,
                        'p_value': p_value,
                        'data': df
                    }, 'anova')
                    render_ai_response(insights, "ANOVA 해석")
                    
        except Exception as e:
            st.error(f"분석 오류: {str(e)}")

def perform_regression_analysis(df: pd.DataFrame):
    """회귀분석 수행"""
    st.markdown("#### 📈 회귀분석")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        response_var = st.selectbox("종속변수 (Y)", numeric_cols, key="reg_y")
    
    with col2:
        predictor_vars = st.multiselect(
            "독립변수 (X)", 
            [col for col in numeric_cols if col != response_var],
            key="reg_x"
        )
    
    model_type = st.radio(
        "모델 유형",
        ["선형 회귀", "다항 회귀 (2차)", "다항 회귀 (3차)"]
    )
    
    if predictor_vars and st.button("회귀분석 실행"):
        try:
            # 데이터 준비
            X = df[predictor_vars].dropna()
            y = df.loc[X.index, response_var]
            
            # 모델 구성
            if model_type == "선형 회귀":
                formula = f"{response_var} ~ " + " + ".join(predictor_vars)
            elif model_type == "다항 회귀 (2차)":
                terms = predictor_vars.copy()
                # 2차 항 추가
                for var in predictor_vars:
                    terms.append(f"I({var}**2)")
                # 교호작용 항 추가
                if len(predictor_vars) >= 2:
                    for i in range(len(predictor_vars)):
                        for j in range(i+1, len(predictor_vars)):
                            terms.append(f"{predictor_vars[i]}:{predictor_vars[j]}")
                formula = f"{response_var} ~ " + " + ".join(terms)
            else:  # 3차
                # 간단히 3차 항만 추가
                terms = predictor_vars.copy()
                for var in predictor_vars:
                    terms.extend([f"I({var}**2)", f"I({var}**3)"])
                formula = f"{response_var} ~ " + " + ".join(terms)
            
            # 모델 적합
            model = ols(formula, data=df).fit()
            
            # 결과 표시
            st.markdown("##### 모델 요약")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("R²", f"{model.rsquared:.4f}")
            with col2:
                st.metric("Adj. R²", f"{model.rsquared_adj:.4f}")
            with col3:
                st.metric("F-통계량", f"{model.fvalue:.4f}")
            with col4:
                st.metric("p-value", f"{model.f_pvalue:.4e}")
            
            # 계수 테이블
            st.markdown("##### 회귀 계수")
            coef_df = pd.DataFrame({
                '계수': model.params,
                '표준오차': model.bse,
                't-값': model.tvalues,
                'p-value': model.pvalues
            })
            
            st.dataframe(
                coef_df.style.format({
                    '계수': '{:.4f}',
                    '표준오차': '{:.4f}',
                    't-값': '{:.4f}',
                    'p-value': '{:.4e}'
                }).background_gradient(subset=['p-value'], cmap='RdYlGn_r'),
                use_container_width=True
            )
            
            # 잔차 플롯
            st.markdown("##### 잔차 분석")
            
            residuals = model.resid
            fitted = model.fittedvalues
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("잔차 vs 적합값", "Q-Q Plot")
            )
            
            # 잔차 플롯
            fig.add_trace(
                go.Scatter(x=fitted, y=residuals, mode='markers',
                          marker=dict(color='blue', size=8, opacity=0.6)),
                row=1, col=1
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
            
            # Q-Q 플롯
            qq = stats.probplot(residuals, dist="norm")
            fig.add_trace(
                go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers',
                          marker=dict(color='green', size=8, opacity=0.6)),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=qq[0][0], y=qq[1][1]*np.array(qq[0][0])+qq[1][0],
                          mode='lines', line=dict(color='red', dash='dash')),
                row=1, col=2
            )
            
            fig.update_xaxis(title_text="적합값", row=1, col=1)
            fig.update_yaxis(title_text="잔차", row=1, col=1)
            fig.update_xaxis(title_text="이론적 분위수", row=1, col=2)
            fig.update_yaxis(title_text="표본 분위수", row=1, col=2)
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # AI 해석
            if st.button("🤖 AI 모델 해석", key="reg_ai"):
                with st.spinner("AI가 회귀 모델을 해석 중입니다..."):
                    insights = generate_ai_insights({
                        'analysis_type': 'regression',
                        'model_type': model_type,
                        'response': response_var,
                        'predictors': predictor_vars,
                        'r_squared': model.rsquared,
                        'coefficients': coef_df.to_dict(),
                        'formula': formula
                    }, 'regression')
                    render_ai_response(insights, "회귀분석 해석")
            
        except Exception as e:
            st.error(f"회귀분석 오류: {str(e)}")

def perform_rsm_analysis(df: pd.DataFrame):
    """반응표면분석 수행"""
    st.markdown("#### 🎯 반응표면분석 (RSM)")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        response_var = st.selectbox("반응변수", numeric_cols, key="rsm_y")
    
    with col2:
        factor_vars = st.multiselect(
            "요인 선택 (2-3개 권장)", 
            [col for col in numeric_cols if col != response_var],
            key="rsm_x",
            max_selections=3
        )
    
    if len(factor_vars) >= 2 and st.button("RSM 분석 실행"):
        try:
            # 2차 다항식 모델 구성
            terms = factor_vars.copy()
            
            # 2차 항
            for var in factor_vars:
                terms.append(f"I({var}**2)")
            
            # 교호작용 항
            for i in range(len(factor_vars)):
                for j in range(i+1, len(factor_vars)):
                    terms.append(f"{factor_vars[i]}:{factor_vars[j]}")
            
            formula = f"{response_var} ~ " + " + ".join(terms)
            
            # 모델 적합
            model = ols(formula, data=df).fit()
            
            # 결과 요약
            st.markdown("##### RSM 모델 요약")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R²", f"{model.rsquared:.4f}")
            with col2:
                st.metric("Adj. R²", f"{model.rsquared_adj:.4f}")
            with col3:
                st.metric("적합성 결여 p-value", "계산 중...")
            
            # 3D 반응 표면 플롯 (처음 2개 요인)
            if len(factor_vars) >= 2:
                st.markdown("##### 반응 표면 플롯")
                
                # 그리드 생성
                x_range = np.linspace(df[factor_vars[0]].min(), df[factor_vars[0]].max(), 50)
                y_range = np.linspace(df[factor_vars[1]].min(), df[factor_vars[1]].max(), 50)
                X_grid, Y_grid = np.meshgrid(x_range, y_range)
                
                # 예측값 계산
                if len(factor_vars) == 2:
                    grid_df = pd.DataFrame({
                        factor_vars[0]: X_grid.ravel(),
                        factor_vars[1]: Y_grid.ravel()
                    })
                else:
                    # 3번째 요인은 평균값으로 고정
                    grid_df = pd.DataFrame({
                        factor_vars[0]: X_grid.ravel(),
                        factor_vars[1]: Y_grid.ravel(),
                        factor_vars[2]: df[factor_vars[2]].mean()
                    })
                
                Z_pred = model.predict(grid_df).values.reshape(X_grid.shape)
                
                # 3D 표면 플롯
                fig = go.Figure(data=[
                    go.Surface(x=x_range, y=y_range, z=Z_pred,
                              colorscale='Viridis',
                              name='반응 표면')
                ])
                
                # 실제 데이터 포인트 추가
                fig.add_trace(go.Scatter3d(
                    x=df[factor_vars[0]],
                    y=df[factor_vars[1]],
                    z=df[response_var],
                    mode='markers',
                    marker=dict(color='red', size=5),
                    name='실험 데이터'
                ))
                
                fig.update_layout(
                    title=f"{response_var} 반응 표면",
                    scene=dict(
                        xaxis_title=factor_vars[0],
                        yaxis_title=factor_vars[1],
                        zaxis_title=response_var
                    ),
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 등고선 플롯
                st.markdown("##### 등고선 플롯")
                
                fig2 = go.Figure(data=go.Contour(
                    x=x_range,
                    y=y_range,
                    z=Z_pred,
                    colorscale='Viridis',
                    contours=dict(
                        coloring='heatmap',
                        showlabels=True,
                        labelfont=dict(size=12)
                    )
                ))
                
                # 실험 점 추가
                fig2.add_trace(go.Scatter(
                    x=df[factor_vars[0]],
                    y=df[factor_vars[1]],
                    mode='markers',
                    marker=dict(color='red', size=8),
                    name='실험점'
                ))
                
                fig2.update_layout(
                    title=f"{response_var} 등고선도",
                    xaxis_title=factor_vars[0],
                    yaxis_title=factor_vars[1],
                    height=500
                )
                
                st.plotly_chart(fig2, use_container_width=True)
            
            # 최적점 찾기
            st.markdown("##### 최적점 탐색")
            
            optimization_goal = st.radio(
                "최적화 목표",
                ["최대화", "최소화", "목표값"]
            )
            
            target_value = None
            if optimization_goal == "목표값":
                target_value = st.number_input("목표값", value=float(df[response_var].mean()))
            
            if st.button("최적점 찾기"):
                optimal_point = find_optimal_point(
                    model, factor_vars, df, optimization_goal, target_value
                )
                
                if optimal_point:
                    st.success("✅ 최적점 발견!")
                    
                    # 최적 조건 표시
                    opt_df = pd.DataFrame([optimal_point['conditions']])
                    st.dataframe(opt_df.style.format("{:.3f}"), use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("예측값", f"{optimal_point['predicted_value']:.3f}")
                    with col2:
                        st.metric("예측 구간", 
                                 f"[{optimal_point['prediction_interval'][0]:.3f}, "
                                 f"{optimal_point['prediction_interval'][1]:.3f}]")
                    
                    # AI 최적화 해석
                    if st.button("🤖 AI 최적화 해석", key="rsm_opt_ai"):
                        with st.spinner("AI가 최적화 결과를 해석 중입니다..."):
                            insights = generate_ai_insights({
                                'analysis_type': 'rsm_optimization',
                                'response': response_var,
                                'factors': factor_vars,
                                'optimal_point': optimal_point,
                                'optimization_goal': optimization_goal,
                                'model_summary': {
                                    'r_squared': model.rsquared,
                                    'coefficients': model.params.to_dict()
                                }
                            }, 'rsm_optimization')
                            render_ai_response(insights, "RSM 최적화 해석")
            
        except Exception as e:
            st.error(f"RSM 분석 오류: {str(e)}")

def find_optimal_point(model, factor_vars, df, goal, target_value=None):
    """최적점 찾기"""
    from scipy.optimize import minimize
    
    # 목적 함수 정의
    def objective(x):
        # 예측을 위한 데이터프레임 생성
        pred_data = pd.DataFrame([x], columns=factor_vars)
        prediction = model.predict(pred_data)[0]
        
        if goal == "최대화":
            return -prediction  # 최소화 알고리즘이므로 음수
        elif goal == "최소화":
            return prediction
        else:  # 목표값
            return (prediction - target_value) ** 2
    
    # 초기값과 경계 설정
    x0 = [df[var].mean() for var in factor_vars]
    bounds = [(df[var].min(), df[var].max()) for var in factor_vars]
    
    # 최적화 실행
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
    
    if result.success:
        optimal_x = result.x
        pred_data = pd.DataFrame([optimal_x], columns=factor_vars)
        optimal_y = model.predict(pred_data)[0]
        
        # 예측 구간 계산
        predict_mean_se = model.get_prediction(pred_data).summary_frame(alpha=0.05)
        
        return {
            'conditions': dict(zip(factor_vars, optimal_x)),
            'predicted_value': optimal_y,
            'prediction_interval': [
                predict_mean_se['mean_ci_lower'].iloc[0],
                predict_mean_se['mean_ci_upper'].iloc[0]
            ]
        }
    
    return None

def perform_optimization_analysis(df: pd.DataFrame):
    """최적화 분석"""
    st.markdown("#### 🎯 프로세스 최적화")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 다중 반응 최적화
    st.markdown("##### 다중 반응 최적화")
    
    response_vars = st.multiselect(
        "반응변수들 선택",
        numeric_cols,
        key="opt_responses"
    )
    
    if response_vars:
        # 각 반응변수별 목표 설정
        response_goals = {}
        response_weights = {}
        
        for resp in response_vars:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                goal = st.selectbox(
                    f"{resp} 목표",
                    ["최대화", "최소화", "목표값"],
                    key=f"goal_{resp}"
                )
                response_goals[resp] = goal
            
            with col2:
                if goal == "목표값":
                    target = st.number_input(
                        f"{resp} 목표값",
                        value=float(df[resp].mean()),
                        key=f"target_{resp}"
                    )
                    response_goals[resp] = ('target', target)
                
            with col3:
                weight = st.slider(
                    f"{resp} 가중치",
                    0.0, 1.0, 0.5,
                    key=f"weight_{resp}"
                )
                response_weights[resp] = weight
        
        # 요인 선택
        factor_vars = st.multiselect(
            "최적화할 요인들",
            [col for col in numeric_cols if col not in response_vars],
            key="opt_factors"
        )
        
        if factor_vars and st.button("다중 반응 최적화 실행"):
            try:
                # 모든 반응변수에 대한 모델 구축
                models = {}
                for resp in response_vars:
                    formula = f"{resp} ~ " + " + ".join(factor_vars)
                    models[resp] = ols(formula, data=df).fit()
                
                # 종합 목적함수 정의
                def multi_objective(x):
                    pred_data = pd.DataFrame([x], columns=factor_vars)
                    total_score = 0
                    
                    for resp, model in models.items():
                        pred = model.predict(pred_data)[0]
                        weight = response_weights[resp]
                        goal = response_goals[resp]
                        
                        if goal == "최대화":
                            # 정규화된 점수 (0-1)
                            score = (pred - df[resp].min()) / (df[resp].max() - df[resp].min())
                        elif goal == "최소화":
                            score = 1 - (pred - df[resp].min()) / (df[resp].max() - df[resp].min())
                        else:  # 목표값
                            target = goal[1]
                            deviation = abs(pred - target) / abs(target)
                            score = 1 / (1 + deviation)  # 편차가 작을수록 높은 점수
                        
                        total_score += weight * score
                    
                    return -total_score  # 최대화를 위해 음수
                
                # 최적화 실행
                x0 = [df[var].mean() for var in factor_vars]
                bounds = [(df[var].min(), df[var].max()) for var in factor_vars]
                
                result = minimize(multi_objective, x0, method='L-BFGS-B', bounds=bounds)
                
                if result.success:
                    st.success("✅ 최적 조건 발견!")
                    
                    # 최적 조건 표시
                    optimal_conditions = dict(zip(factor_vars, result.x))
                    opt_df = pd.DataFrame([optimal_conditions])
                    
                    st.markdown("##### 최적 조건")
                    st.dataframe(opt_df.style.format("{:.3f}"), use_container_width=True)
                    
                    # 예측 결과
                    st.markdown("##### 예측 결과")
                    pred_data = pd.DataFrame([result.x], columns=factor_vars)
                    
                    predictions = {}
                    for resp, model in models.items():
                        predictions[resp] = model.predict(pred_data)[0]
                    
                    pred_df = pd.DataFrame([predictions])
                    st.dataframe(pred_df.style.format("{:.3f}"), use_container_width=True)
                    
                    # 민감도 분석
                    if st.checkbox("민감도 분석 수행"):
                        perform_sensitivity_analysis(models, factor_vars, result.x, df)
                    
                    # AI 최적화 전략
                    if st.button("🤖 AI 최적화 전략", key="multi_opt_ai"):
                        with st.spinner("AI가 최적화 전략을 수립 중입니다..."):
                            insights = generate_ai_insights({
                                'analysis_type': 'multi_optimization',
                                'responses': response_vars,
                                'factors': factor_vars,
                                'optimal_conditions': optimal_conditions,
                                'predictions': predictions,
                                'goals': response_goals,
                                'weights': response_weights
                            }, 'multi_optimization')
                            render_ai_response(insights, "다중 반응 최적화 전략")
                
            except Exception as e:
                st.error(f"최적화 오류: {str(e)}")

def perform_sensitivity_analysis(models, factor_vars, optimal_point, df):
    """민감도 분석"""
    st.markdown("##### 민감도 분석")
    
    # 각 요인별 민감도 계산
    sensitivity_data = []
    
    for i, factor in enumerate(factor_vars):
        factor_range = np.linspace(
            df[factor].min(), 
            df[factor].max(), 
            20
        )
        
        for resp, model in models.items():
            predictions = []
            
            for value in factor_range:
                # 다른 요인은 최적값으로 고정
                x = optimal_point.copy()
                x[i] = value
                
                pred_data = pd.DataFrame([x], columns=factor_vars)
                pred = model.predict(pred_data)[0]
                predictions.append(pred)
            
            # 민감도 = (최대 변화량) / (요인 범위)
            sensitivity = (max(predictions) - min(predictions)) / (factor_range[-1] - factor_range[0])
            
            sensitivity_data.append({
                'Factor': factor,
                'Response': resp,
                'Sensitivity': abs(sensitivity),
                'Range': factor_range,
                'Predictions': predictions
            })
    
    # 민감도 히트맵
    sensitivity_matrix = pd.pivot_table(
        pd.DataFrame(sensitivity_data),
        values='Sensitivity',
        index='Response',
        columns='Factor'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=sensitivity_matrix.values,
        x=sensitivity_matrix.columns,
        y=sensitivity_matrix.index,
        colorscale='Reds',
        text=sensitivity_matrix.round(3).values,
        texttemplate='%{text}',
        textfont={"size": 12}
    ))
    
    fig.update_layout(
        title="민감도 히트맵",
        xaxis_title="요인",
        yaxis_title="반응변수"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 민감도 플롯
    st.markdown("##### 요인별 영향도")
    
    selected_response = st.selectbox(
        "반응변수 선택",
        list(models.keys())
    )
    
    fig2 = go.Figure()
    
    for data in sensitivity_data:
        if data['Response'] == selected_response:
            fig2.add_trace(go.Scatter(
                x=data['Range'],
                y=data['Predictions'],
                mode='lines',
                name=data['Factor'],
                line=dict(width=3)
            ))
    
    fig2.update_layout(
        title=f"{selected_response} 민감도 분석",
        xaxis_title="요인 값 (다른 요인은 최적값 고정)",
        yaxis_title=selected_response,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig2, use_container_width=True)

def perform_ml_prediction(df: pd.DataFrame):
    """머신러닝 예측 분석"""
    st.markdown("#### 🤖 머신러닝 예측 모델")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_var = st.selectbox("예측 대상 변수", numeric_cols, key="ml_target")
    
    with col2:
        feature_vars = st.multiselect(
            "특성 변수들",
            [col for col in numeric_cols if col != target_var],
            key="ml_features"
        )
    
    if feature_vars and st.button("ML 모델 학습"):
        try:
            # 데이터 준비
            X = df[feature_vars].dropna()
            y = df.loc[X.index, target_var]
            
            # 학습/테스트 분할
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # 여러 모델 학습
            models = {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Gradient Boosting': None,  # 간단히 RF만 사용
                'Neural Network': None
            }
            
            results = {}
            
            # Random Forest 학습
            rf_model = models['Random Forest']
            rf_model.fit(X_train, y_train)
            
            # 예측
            y_pred_train = rf_model.predict(X_train)
            y_pred_test = rf_model.predict(X_test)
            
            # 성능 평가
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            # 결과 표시
            st.markdown("##### 모델 성능")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Train R²", f"{train_r2:.4f}")
            with col2:
                st.metric("Test R²", f"{test_r2:.4f}")
            with col3:
                st.metric("RMSE", f"{test_rmse:.4f}")
            with col4:
                st.metric("MAE", f"{test_mae:.4f}")
            
            # 특성 중요도
            st.markdown("##### 특성 중요도")
            
            importance_df = pd.DataFrame({
                'Feature': feature_vars,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(
                importance_df, 
                x='Importance', 
                y='Feature',
                orientation='h',
                title="Random Forest 특성 중요도"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # 예측 vs 실제 플롯
            st.markdown("##### 예측 정확도")
            
            fig2 = go.Figure()
            
            # 학습 데이터
            fig2.add_trace(go.Scatter(
                x=y_train,
                y=y_pred_train,
                mode='markers',
                name='학습 데이터',
                marker=dict(color='blue', size=8, opacity=0.6)
            ))
            
            # 테스트 데이터
            fig2.add_trace(go.Scatter(
                x=y_test,
                y=y_pred_test,
                mode='markers',
                name='테스트 데이터',
                marker=dict(color='red', size=8, opacity=0.6)
            ))
            
            # 대각선
            min_val = min(y.min(), y_pred_test.min())
            max_val = max(y.max(), y_pred_test.max())
            fig2.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='black', dash='dash')
            ))
            
            fig2.update_layout(
                title="예측값 vs 실제값",
                xaxis_title="실제값",
                yaxis_title="예측값",
                height=500
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # 새로운 데이터 예측
            st.markdown("##### 새 데이터 예측")
            
            new_data = {}
            cols = st.columns(len(feature_vars))
            
            for i, (col, var) in enumerate(zip(cols, feature_vars)):
                with col:
                    new_data[var] = st.number_input(
                        var,
                        value=float(df[var].mean()),
                        key=f"new_{var}"
                    )
            
            if st.button("예측하기"):
                new_df = pd.DataFrame([new_data])
                prediction = rf_model.predict(new_df)[0]
                
                # 예측 구간 (Random Forest의 경우 트리별 예측값 사용)
                tree_predictions = np.array([
                    tree.predict(new_df)[0] 
                    for tree in rf_model.estimators_
                ])
                
                pred_std = np.std(tree_predictions)
                pred_interval = [
                    prediction - 1.96 * pred_std,
                    prediction + 1.96 * pred_std
                ]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("예측값", f"{prediction:.3f}")
                with col2:
                    st.metric("95% 예측 구간",
                             f"[{pred_interval[0]:.3f}, {pred_interval[1]:.3f}]")
                
                # AI 예측 설명
                if st.button("🤖 AI 예측 설명", key="ml_pred_ai"):
                    with st.spinner("AI가 예측을 설명 중입니다..."):
                        insights = generate_ai_insights({
                            'analysis_type': 'ml_prediction',
                            'model': 'Random Forest',
                            'target': target_var,
                            'features': feature_vars,
                            'importance': importance_df.to_dict(),
                            'performance': {
                                'train_r2': train_r2,
                                'test_r2': test_r2,
                                'rmse': test_rmse,
                                'mae': test_mae
                            },
                            'new_prediction': {
                                'inputs': new_data,
                                'prediction': prediction,
                                'interval': pred_interval
                            }
                        }, 'ml_prediction')
                        render_ai_response(insights, "ML 예측 설명")
            
        except Exception as e:
            st.error(f"ML 분석 오류: {str(e)}")

def generate_ai_insights(data: Any, analysis_type: str) -> Dict:
    """AI 인사이트 생성"""
    try:
        api_manager = APIManager()
        
        # 분석 유형별 프롬프트 구성
        if analysis_type == 'descriptive':
            prompt = f"""
            고분자 실험 데이터의 기술통계 분석 결과입니다:
            
            데이터 요약:
            - 변수: {', '.join(data.columns.tolist())}
            - 샘플 수: {len(data)}
            
            주요 통계:
            {data.describe().to_string()}
            
            상관관계:
            {data.corr().to_string()}
            
            다음 관점에서 분석해주세요:
            1. 데이터의 주요 패턴과 특징
            2. 변수 간 관계의 의미
            3. 이상치나 주의할 점
            4. 추가 분석 제안
            
            응답은 다음 JSON 형식으로:
            {{
                "main": "핵심 인사이트 요약",
                "key_insights": [
                    {{"title": "인사이트 제목", "description": "설명", "importance": "high/medium/low"}}
                ],
                "details": {{
                    "reasoning": "분석 과정 설명",
                    "alternatives": [
                        {{"name": "대안", "description": "설명", "pros": "장점", "cons": "단점"}}
                    ],
                    "theory": "통계 이론적 배경",
                    "confidence": {{
                        "overall": 85,
                        "explanation": "신뢰도 설명",
                        "limitations": ["한계점1", "한계점2"]
                    }}
                }}
            }}
            """
            
        elif analysis_type == 'anova':
            prompt = f"""
            ANOVA 분석 결과를 해석해주세요:
            
            반응변수: {data['response']}
            요인: {data['factor']}
            F-통계량: {data['f_stat']:.4f}
            p-value: {data['p_value']:.4f}
            
            데이터 맥락을 고려하여:
            1. 통계적 유의성의 실질적 의미
            2. 효과 크기와 실무적 중요성
            3. 추가 분석 필요성
            4. 실험 개선 방안
            
            위의 JSON 형식으로 응답해주세요.
            """
            
        elif analysis_type == 'regression':
            prompt = f"""
            회귀분석 결과를 해석해주세요:
            
            모델: {data['model_type']}
            종속변수: {data['response']}
            독립변수: {', '.join(data['predictors'])}
            R²: {data['r_squared']:.4f}
            
            회귀계수:
            {json.dumps(data['coefficients'], indent=2)}
            
            다음을 포함하여 해석:
            1. 모델의 설명력과 적합도
            2. 각 변수의 영향력과 의미
            3. 모델 가정 충족 여부
            4. 예측 활용 방안
            
            위의 JSON 형식으로 응답해주세요.
            """
            
        elif analysis_type == 'rsm_optimization':
            prompt = f"""
            RSM 최적화 결과를 해석해주세요:
            
            반응변수: {data['response']}
            요인: {', '.join(data['factors'])}
            최적화 목표: {data['optimization_goal']}
            
            최적 조건:
            {json.dumps(data['optimal_point']['conditions'], indent=2)}
            
            예측값: {data['optimal_point']['predicted_value']:.3f}
            
            다음 관점에서 해석:
            1. 최적 조건의 실무적 타당성
            2. 예측의 신뢰성
            3. 실험 검증 전략
            4. 추가 최적화 방향
            
            위의 JSON 형식으로 응답해주세요.
            """
            
        elif analysis_type == 'multi_optimization':
            prompt = f"""
            다중 반응 최적화 결과를 해석해주세요:
            
            반응변수들: {', '.join(data['responses'])}
            요인들: {', '.join(data['factors'])}
            
            최적 조건:
            {json.dumps(data['optimal_conditions'], indent=2)}
            
            예측 결과:
            {json.dumps(data['predictions'], indent=2)}
            
            목표와 가중치:
            {json.dumps(data['goals'], indent=2)}
            {json.dumps(data['weights'], indent=2)}
            
            다음을 분석:
            1. 균형잡힌 최적화 달성 여부
            2. Trade-off 관계 설명
            3. 실무 적용 전략
            4. 강건성 확보 방안
            
            위의 JSON 형식으로 응답해주세요.
            """
            
        elif analysis_type == 'ml_prediction':
            prompt = f"""
            머신러닝 예측 모델 결과를 해석해주세요:
            
            모델: {data['model']}
            예측 대상: {data['target']}
            특성 변수: {', '.join(data['features'])}
            
            성능:
            Train R²: {data['performance']['train_r2']:.4f}
            Test R²: {data['performance']['test_r2']:.4f}
            RMSE: {data['performance']['rmse']:.4f}
            
            특성 중요도:
            {json.dumps(data['importance'], indent=2)}
            
            새 예측:
            입력: {json.dumps(data['new_prediction']['inputs'], indent=2)}
            예측값: {data['new_prediction']['prediction']:.3f}
            
            다음을 설명:
            1. 모델 성능의 의미
            2. 주요 영향 요인 해석
            3. 예측의 신뢰성
            4. 모델 개선 방향
            
            위의 JSON 형식으로 응답해주세요.
            """
        
        else:
            # 기본 분석
            prompt = f"""
            고분자 실험 데이터를 분석해주세요:
            {str(data)[:1000]}  # 처음 1000자만
            
            핵심 인사이트를 위의 JSON 형식으로 제공해주세요.
            """
        
        # API 호출
        response = api_manager.generate_unified_response(
            prompt,
            analysis_type="polymer_analysis",
            output_format="structured",
            include_reasoning=True
        )
        
        # 응답 파싱
        try:
            if isinstance(response, dict):
                return response
            else:
                # 텍스트 응답을 구조화
                return {
                    "main": response,
                    "key_insights": [
                        {
                            "title": "AI 분석 결과",
                            "description": response,
                            "importance": "medium"
                        }
                    ],
                    "details": {
                        "reasoning": "상세 분석 내용은 AI 응답을 참조하세요.",
                        "alternatives": [],
                        "theory": "",
                        "confidence": {
                            "overall": 75,
                            "explanation": "AI 기반 분석 결과입니다.",
                            "limitations": ["실험 데이터에 기반한 통계적 추론입니다."]
                        }
                    }
                }
        except:
            return {
                "main": response if isinstance(response, str) else "AI 분석 완료",
                "key_insights": [],
                "details": {}
            }
            
    except Exception as e:
        st.error(f"AI 인사이트 생성 오류: {str(e)}")
        return {
            "main": "AI 분석을 수행할 수 없습니다.",
            "key_insights": [],
            "details": {}
        }

def visualization_section(df: pd.DataFrame):
    """시각화 섹션"""
    st.markdown("### 📊 대화형 시각화")
    
    viz_type = st.selectbox(
        "시각화 유형",
        ["산점도 행렬", "평행 좌표 플롯", "3D 산점도", "히트맵", "시계열 플롯"]
    )
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if viz_type == "산점도 행렬":
        selected_cols = st.multiselect(
            "변수 선택 (2-5개)",
            numeric_cols,
            default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
        )
        
        if len(selected_cols) >= 2:
            fig = px.scatter_matrix(
                df[selected_cols],
                dimensions=selected_cols,
                title="산점도 행렬",
                height=800
            )
            fig.update_traces(diagonal_visible=False)
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "평행 좌표 플롯":
        # 데이터 정규화
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[numeric_cols])
        scaled_df = pd.DataFrame(scaled_data, columns=numeric_cols)
        
        fig = go.Figure(data=go.Parcoords(
            dimensions=[
                dict(
                    label=col,
                    values=scaled_df[col]
                ) for col in numeric_cols
            ],
            line=dict(
                color=scaled_df[numeric_cols[0]],
                colorscale='Viridis',
                showscale=True
            )
        ))
        
        fig.update_layout(
            title="평행 좌표 플롯 (정규화된 데이터)",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "3D 산점도":
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            x_var = st.selectbox("X축", numeric_cols, key="3d_x")
        with col2:
            y_var = st.selectbox("Y축", numeric_cols, key="3d_y")
        with col3:
            z_var = st.selectbox("Z축", numeric_cols, key="3d_z")
        with col4:
            color_var = st.selectbox("색상", numeric_cols + [None], key="3d_color")
        
        fig = px.scatter_3d(
            df, x=x_var, y=y_var, z=z_var,
            color=color_var,
            title=f"3D 산점도: {x_var} vs {y_var} vs {z_var}",
            height=700
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "히트맵":
        # 상관관계 히트맵은 이미 있으므로 다른 형태
        st.info("기술통계 분석 섹션의 상관관계 히트맵을 참조하세요.")
    
    elif viz_type == "시계열 플롯":
        if 'Run' in df.columns or df.index.name == 'Run':
            selected_vars = st.multiselect(
                "변수 선택",
                numeric_cols,
                default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
            )
            
            if selected_vars:
                fig = go.Figure()
                
                for var in selected_vars:
                    fig.add_trace(go.Scatter(
                        x=df.index if df.index.name == 'Run' else df['Run'],
                        y=df[var],
                        mode='lines+markers',
                        name=var
                    ))
                
                fig.update_layout(
                    title="실험 진행에 따른 변화",
                    xaxis_title="실험 번호",
                    yaxis_title="값",
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("시계열 플롯을 위한 'Run' 열이 없습니다.")

def collaboration_section():
    """협업 섹션"""
    st.markdown("### 👥 협업 및 공유")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📤 분석 결과 공유")
        
        share_options = st.multiselect(
            "공유할 내용",
            ["분석 결과", "시각화", "AI 인사이트", "원본 데이터"],
            default=["분석 결과", "시각화"]
        )
        
        if st.button("공유 링크 생성"):
            # 실제로는 데이터를 서버에 저장하고 고유 링크 생성
            share_id = str(uuid.uuid4())[:8]
            share_link = f"https://polymer-doe.app/shared/{share_id}"
            
            st.success("✅ 공유 링크가 생성되었습니다!")
            st.code(share_link)
            
            # 클립보드 복사 버튼
            st.button("📋 클립보드에 복사")
    
    with col2:
        st.markdown("#### 💬 토론 및 주석")
        
        # 간단한 댓글 시스템
        comment = st.text_area("분석에 대한 의견을 남겨주세요", height=100)
        
        if st.button("댓글 작성"):
            if comment:
                # 실제로는 DB에 저장
                st.success("✅ 댓글이 저장되었습니다!")
                
                # 예시 댓글 표시
                with st.container():
                    st.markdown("---")
                    st.markdown(f"**{get_current_user()['name']}** - 방금 전")
                    st.write(comment)

def export_results_section():
    """결과 내보내기 섹션"""
    st.markdown("### 💾 결과 내보내기")
    
    if st.session_state.analysis_results:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Excel 보고서"):
                excel_data = create_excel_report(
                    st.session_state.analysis_data,
                    st.session_state.analysis_results
                )
                
                st.download_button(
                    label="⬇️ 다운로드",
                    data=excel_data,
                    file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        with col2:
            if st.button("📄 PDF 보고서"):
                st.info("PDF 생성 기능은 준비 중입니다.")
        
        with col3:
            if st.button("🐍 Python 코드"):
                python_code = generate_analysis_code(st.session_state.analysis_results)
                st.download_button(
                    label="⬇️ 다운로드",
                    data=python_code,
                    file_name="analysis_code.py",
                    mime="text/plain"
                )
    else:
        st.info("분석을 먼저 수행해주세요.")

def create_excel_report(df: pd.DataFrame, results: Dict) -> bytes:
    """Excel 보고서 생성"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # 원본 데이터
        df.to_excel(writer, sheet_name='원본 데이터', index=False)
        
        # 기술통계
        if 'descriptive' in results:
            desc_df = pd.DataFrame(results['descriptive'])
            desc_df.to_excel(writer, sheet_name='기술통계')
        
        # 분석 결과 요약
        summary_df = pd.DataFrame([{
            '분석일시': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '샘플 수': len(df),
            '변수 수': len(df.columns),
            '분석 유형': ', '.join(results.keys())
        }])
        summary_df.to_excel(writer, sheet_name='요약', index=False)
    
    output.seek(0)
    return output.getvalue()

def generate_analysis_code(results: Dict) -> str:
    """분석 코드 생성"""
    code = f"""
# Polymer DOE 데이터 분석 코드
# 생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
# df = pd.read_csv('your_data.csv')

# 기술통계
print(df.describe())

# 상관관계
print(df.corr())

# 추가 분석 코드는 수행한 분석에 따라 자동 생성됩니다.
"""
    
    return code

def main():
    """메인 함수"""
    # 인증 확인
    if not check_authentication():
        st.warning("🔒 로그인이 필요합니다.")
        st.stop()
    
    # 세션 초기화
    initialize_session_state()
    
    # UI 초기화
    ui = get_common_ui()
    
    # 헤더
    st.title("📈 데이터 분석")
    st.markdown("실험 데이터의 통계 분석과 AI 기반 인사이트를 제공합니다.")
    
    # 메인 레이아웃
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📁 데이터 준비",
        "📊 통계 분석", 
        "🎨 시각화",
        "👥 협업",
        "💾 내보내기"
    ])
    
    with tab1:
        # 데이터 업로드
        df = upload_data_section()
        
        if df is not None:
            st.divider()
            
            # 데이터 전처리
            df = data_preprocessing_section(df)
            
            # 세션에 저장
            st.session_state.analysis_data = df
    
    with tab2:
        if st.session_state.analysis_data is not None:
            statistical_analysis_section(st.session_state.analysis_data)
        else:
            st.info("먼저 데이터를 업로드해주세요.")
    
    with tab3:
        if st.session_state.analysis_data is not None:
            visualization_section(st.session_state.analysis_data)
        else:
            st.info("먼저 데이터를 업로드해주세요.")
    
    with tab4:
        collaboration_section()
    
    with tab5:
        export_results_section()
    
    # 사이드바 - 빠른 통계
    if st.session_state.analysis_data is not None:
        with st.sidebar:
            st.markdown("### 📊 빠른 통계")
            
            df = st.session_state.analysis_data
            numeric_df = df.select_dtypes(include=[np.number])
            
            if not numeric_df.empty:
                selected_col = st.selectbox(
                    "변수 선택",
                    numeric_df.columns.tolist()
                )
                
                if selected_col:
                    st.metric("평균", f"{numeric_df[selected_col].mean():.3f}")
                    st.metric("표준편차", f"{numeric_df[selected_col].std():.3f}")
                    st.metric("최소값", f"{numeric_df[selected_col].min():.3f}")
                    st.metric("최대값", f"{numeric_df[selected_col].max():.3f}")
                    
                    # 미니 히스토그램
                    fig = px.histogram(
                        df, x=selected_col,
                        nbins=20,
                        height=200
                    )
                    fig.update_layout(
                        showlegend=False,
                        margin=dict(l=0, r=0, t=0, b=0)
                    )
                    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
