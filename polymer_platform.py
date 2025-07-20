"""
Polymer DOE Platform - Test Version
테스트용 간단 버전
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(
    page_title="🧬 Polymer DOE Platform",
    page_icon="🧬",
    layout="wide"
)

# CSS 스타일
st.markdown("""
<style>
    .metric-card {
        background: white;
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #E5E7EB;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1F2937;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #6B7280;
    }
</style>
""", unsafe_allow_html=True)

# 사이드바
with st.sidebar:
    st.title("🧬 Polymer DOE")
    menu = st.radio("메뉴", ["Overview", "Projects", "Analytics", "Settings"])

# 메인 컨텐츠
st.title("Dashboard")
st.markdown("고분자 실험 설계 플랫폼에 오신 것을 환영합니다")

if menu == "Overview":
    # 메트릭 카드
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Active Projects</div>
            <div class="metric-value">12</div>
            <div style="color: green;">↑ Up to 24%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Total Experiments</div>
            <div class="metric-value">48</div>
            <div style="color: red;">↓ Down to 15%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Success Rate</div>
            <div class="metric-value">87.5%</div>
            <div style="color: green;">↑ Up to 5%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Time Saved</div>
            <div class="metric-value">156h</div>
            <div style="color: green;">↑ Up to 12%</div>
        </div>
        """, unsafe_allow_html=True)
    
    # 차트
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("실험 성공률 추이")
        
        # 샘플 데이터
        dates = pd.date_range(end=datetime.now(), periods=30)
        values = np.random.normal(85, 5, 30).clip(70, 95)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            fill='tozeroy',
            fillcolor='rgba(124, 58, 237, 0.1)',
            line=dict(color='#7C3AED', width=3)
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("프로젝트 분포")
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=['PET', 'PLA', 'PP', 'PE', 'Others'],
            values=[30, 25, 20, 15, 10],
            hole=0.6
        )])
        
        fig_pie.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # 테이블
    st.subheader("최근 프로젝트")
    
    df = pd.DataFrame({
        'Project ID': ['POL-0112', 'POL-0118', 'POL-0110'],
        'Name': ['PET Film Optimization', 'PLA Biodegradation', 'PP Composite'],
        'Success Rate': ['87.5%', '92.3%', '78.4%'],
        'Status': ['Active', 'Active', 'Completed']
    })
    
    st.dataframe(df, use_container_width=True, hide_index=True)

elif menu == "Projects":
    st.info("프로젝트 페이지는 준비 중입니다.")

elif menu == "Analytics":
    st.info("분석 페이지는 준비 중입니다.")

elif menu == "Settings":
    st.subheader("API 키 설정")
    
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Google Gemini API Key", type="password")
        st.text_input("OpenAI API Key", type="password")
    with col2:
        st.text_input("DeepSeek API Key", type="password")
        st.text_input("Groq API Key", type="password")
    
    if st.button("저장"):
        st.success("API 키가 저장되었습니다!")
