"""
Polymer DOE Platform - Main Application
고분자 실험 설계 플랫폼 메인 앱
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Any
import json
import os
import sys

# utils 폴더를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 커스텀 UI 컴포넌트 임포트
from utils.common_ui import (
    setup_page_config, apply_custom_css, render_header, render_modern_sidebar,
    create_metric_card, create_analytics_chart, render_data_table,
    create_product_card, show_notification, create_stats_grid,
    create_action_button, create_search_bar, create_dropdown_filter,
    render_footer, create_progress_indicator, render_loading_skeleton,
    create_responsive_columns, format_currency, format_number,
    create_timeline, create_empty_state, show_info_message
)


class PolymerDOEPlatform:
    """메인 플랫폼 클래스"""
    
    def __init__(self):
        self.setup_initial_state()
        
    def setup_initial_state(self):
        """초기 상태 설정"""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.selected_menu = 'overview'
            st.session_state.user = {
                'name': 'Polymer Researcher',
                'level': 'intermediate',
                'organization': 'Research Lab'
            }
            st.session_state.api_keys = {}
            st.session_state.projects = []
            st.session_state.current_project = None
            st.session_state.notifications = []
    
    def render_overview_page(self):
        """Overview 페이지 렌더링"""
        render_header("Dashboard", "고분자 실험 설계 플랫폼에 오신 것을 환영합니다")
        
        # 메트릭 카드 섹션
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # 프로젝트 수 (Income 대신)
            create_metric_card(
                "Active Projects",
                "12",
                "Up to 24%",
                "positive"
            )
        
        with col2:
            # 실험 수 (Spending 대신)
            create_metric_card(
                "Total Experiments",
                "48",
                "Down to 15%",
                "negative"
            )
        
        with col3:
            # 성공률
            create_metric_card(
                "Success Rate",
                "87.5%",
                "Up to 5%",
                "positive"
            )
        
        with col4:
            # 저장된 시간
            create_metric_card(
                "Time Saved",
                "156h",
                "Up to 12%",
                "positive"
            )
        
        # 분석 차트 섹션
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="modern-card">
                <h3 style="margin-bottom: 1rem;">실험 성공률 추이</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # 샘플 데이터 생성
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            success_rates = np.random.normal(85, 5, 30).clip(70, 95)
            df_chart = pd.DataFrame({
                'Date': dates,
                'Success Rate': success_rates
            })
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_chart['Date'],
                y=df_chart['Success Rate'],
                fill='tozeroy',
                fillcolor='rgba(124, 58, 237, 0.1)',
                line=dict(color='#7C3AED', width=3),
                mode='lines',
                name='Success Rate'
            ))
            
            # 주석 추가
            latest_value = df_chart['Success Rate'].iloc[-1]
            fig.add_annotation(
                x=df_chart['Date'].iloc[-1],
                y=latest_value,
                text=f"<b>{latest_value:.1f}%</b><br><span style='font-size: 12px;'>↑ Up to 5%</span>",
                showarrow=True,
                arrowhead=0,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#F59E0B",
                ax=30,
                ay=-40,
                bgcolor="#FEF3C7",
                bordercolor="#F59E0B",
                borderwidth=1,
                borderpad=8,
                font=dict(color="#92400E", size=14)
            )
            
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                height=300,
                showlegend=False,
                xaxis=dict(
                    showgrid=False,
                    showline=False,
                    zeroline=False,
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='#F3F4F6',
                    showline=False,
                    zeroline=False,
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 프로젝트 분포 차트
            st.markdown("""
            <div class="modern-card">
                <h3 style="margin-bottom: 1rem;">프로젝트 분포</h3>
            </div>
            """, unsafe_allow_html=True)
            
            project_types = ['PET', 'PLA', 'PP', 'PE', 'Others']
            values = [30, 25, 20, 15, 10]
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=project_types,
                values=values,
                hole=0.6,
                marker_colors=['#7C3AED', '#A78BFA', '#C4B5FD', '#DDD6FE', '#EDE9FE']
            )])
            
            fig_pie.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                height=300,
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.1
                )
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # 최근 프로젝트 테이블
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("""
            <div class="modern-card">
                <h3 style="margin-bottom: 1rem;">최근 프로젝트</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            search_query = create_search_bar("프로젝트 검색...", "project_search")
            filter_value = create_dropdown_filter("정렬", ["All", "Active", "Completed"], "All", "project_filter")
        
        # 샘플 프로젝트 데이터
        projects_data = {
            'Project ID': ['POL-0112', 'POL-0118', 'POL-0110', 'POL-0104', 'POL-0099'],
            'Name': ['PET Film Optimization', 'PLA Biodegradation Study', 'PP Composite Development', 
                     'PE Recycling Process', 'Nanocellulose Integration'],
            'Experiments': [12, 8, 15, 6, 10],
            'Success Rate': ['87.5%', '92.3%', '78.4%', '95.0%', '88.2%'],
            'Status': ['Active', 'Active', 'Completed', 'Active', 'Completed']
        }
        
        df_projects = pd.DataFrame(projects_data)
        render_data_table(df_projects, show_status=True, height=300)
        
        # 인기 실험 설계 섹션
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="modern-card">
            <h3 style="margin-bottom: 1rem;">인기 실험 설계 템플릿</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            create_product_card(
                "https://via.placeholder.com/150",
                "Full Factorial Design",
                4.5,
                "12 uses",
                "Popular"
            )
        
        with col2:
            create_product_card(
                "https://via.placeholder.com/150",
                "Response Surface Method",
                4.8,
                "8 uses",
                "New"
            )
        
        with col3:
            create_product_card(
                "https://via.placeholder.com/150",
                "Mixture Design",
                4.3,
                "6 uses"
            )
    
    def render_projects_page(self):
        """프로젝트 페이지"""
        render_header("Projects", "프로젝트를 관리하고 새로운 실험을 시작하세요")
        
        # 액션 버튼
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            search_query = create_search_bar("프로젝트 검색...", "projects_search")
        
        with col2:
            if create_action_button("새 프로젝트", "➕", "primary", True, "new_project"):
                st.session_state.show_new_project_modal = True
        
        with col3:
            filter_status = create_dropdown_filter("상태", ["All", "Active", "Completed"], "All", "status_filter")
        
        with col4:
            filter_type = create_dropdown_filter("유형", ["All", "PET", "PLA", "PP", "PE"], "All", "type_filter")
        
        # 프로젝트 카드 그리드
        if not st.session_state.projects:
            create_empty_state(
                "아직 프로젝트가 없습니다",
                "첫 번째 프로젝트를 만들어 실험을 시작해보세요",
                "새 프로젝트 만들기"
            )
        else:
            # 프로젝트 카드 렌더링
            pass
    
    def render_analytics_page(self):
        """분석 페이지"""
        render_header("Analytics", "실험 데이터를 분석하고 인사이트를 얻으세요")
        
        # 통계 그리드
        stats = [
            {'icon': '📊', 'value': '156', 'label': '총 실험 수'},
            {'icon': '✅', 'value': '87.5%', 'label': '평균 성공률'},
            {'icon': '⏱️', 'value': '4.2일', 'label': '평균 실험 기간'},
            {'icon': '💡', 'value': '23', 'label': '발견된 인사이트'}
        ]
        
        create_stats_grid(stats)
        
        # 분석 차트들
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="modern-card">
                <h3 style="margin-bottom: 1rem;">월별 실험 추이</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # 월별 데이터
            months = pd.date_range(end=datetime.now(), periods=12, freq='M')
            experiments = np.random.randint(10, 30, 12)
            
            fig_bar = go.Figure(data=[
                go.Bar(
                    x=months,
                    y=experiments,
                    marker_color='#7C3AED',
                    marker_line_color='#5B21B6',
                    marker_line_width=1.5,
                    opacity=0.8
                )
            ])
            
            fig_bar.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                height=300,
                showlegend=False,
                xaxis=dict(tickformat='%b'),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class="modern-card">
                <h3 style="margin-bottom: 1rem;">실험 유형별 성공률</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # 레이더 차트
            categories = ['Full Factorial', 'RSM', 'Mixture', 'Taguchi', 'Custom']
            values = [88, 92, 85, 78, 83]
            
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                fillcolor='rgba(124, 58, 237, 0.2)',
                line=dict(color='#7C3AED', width=2)
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                margin=dict(l=40, r=40, t=40, b=40),
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
    
    def render_chat_page(self):
        """AI 채팅 페이지"""
        render_header("AI Assistant", "AI와 함께 실험을 설계하고 문제를 해결하세요")
        
        # 채팅 인터페이스
        chat_container = st.container()
        
        with chat_container:
            # 채팅 히스토리
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            # 채팅 메시지 표시
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
            
            # 사용자 입력
            if prompt := st.chat_input("AI에게 질문하세요..."):
                # 사용자 메시지 추가
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                
                # AI 응답 (임시)
                with st.chat_message("assistant"):
                    st.write("AI 응답을 처리 중입니다...")
                    # 실제 AI API 호출은 나중에 구현
    
    def render_settings_page(self):
        """설정 페이지"""
        render_header("Settings", "플랫폼 설정을 관리하세요")
        
        tabs = st.tabs(["🔑 API 키", "👤 프로필", "🔔 알림", "🎨 테마"])
        
        with tabs[0]:
            st.markdown("""
            <div class="modern-card">
                <h3 style="margin-bottom: 1rem;">API 키 관리</h3>
                <p style="color: #6B7280; margin-bottom: 2rem;">
                    Streamlit Secrets에서 관리되는 키는 자동으로 로드됩니다.
                    추가로 키를 입력하시려면 아래 필드를 사용하세요.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # API 키 입력 섹션
            col1, col2 = st.columns(2)
            
            with col1:
                st.text_input("Google Gemini API Key", type="password", key="gemini_key")
                st.text_input("X.AI Grok API Key", type="password", key="grok_key")
                st.text_input("SambaNova API Key", type="password", key="sambanova_key")
            
            with col2:
                st.text_input("DeepSeek API Key", type="password", key="deepseek_key")
                st.text_input("Groq API Key", type="password", key="groq_key")
                st.text_input("HuggingFace API Key", type="password", key="huggingface_key")
            
            if create_action_button("API 키 저장", "💾", "primary", False, "save_api_keys"):
                show_notification("API 키가 저장되었습니다", "success")
        
        with tabs[1]:
            # 프로필 설정
            st.markdown("""
            <div class="modern-card">
                <h3 style="margin-bottom: 1rem;">사용자 프로필</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("""
                <div style="text-align: center;">
                    <div style="width: 120px; height: 120px; margin: 0 auto; 
                                background: linear-gradient(135deg, #7C3AED 0%, #A78BFA 100%);
                                border-radius: 50%; display: flex; align-items: center; 
                                justify-content: center; color: white; font-size: 3rem;">
                        👤
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.text_input("이름", value=st.session_state.user['name'])
                st.selectbox("레벨", ["초급", "중급", "고급", "전문가"], 
                           index=1 if st.session_state.user['level'] == 'intermediate' else 0)
                st.text_input("소속", value=st.session_state.user['organization'])
    
    def run(self):
        """메인 앱 실행"""
        # 페이지 설정 및 CSS 적용
        setup_page_config()
        apply_custom_css()
        
        # 사이드바 렌더링
        render_modern_sidebar()
        
        # 선택된 메뉴에 따라 페이지 렌더링
        menu = st.session_state.get('selected_menu', 'overview')
        
        if menu == 'overview':
            self.render_overview_page()
        elif menu == 'projects':
            self.render_projects_page()
        elif menu == 'analytics':
            self.render_analytics_page()
        elif menu == 'chat':
            self.render_chat_page()
        elif menu == 'products':
            self.render_projects_page()  # 프로젝트 페이지로 대체
        elif menu == 'sales':
            self.render_analytics_page()  # 분석 페이지로 대체
        elif menu == 'review':
            self.render_analytics_page()  # 분석 페이지로 대체
        else:
            self.render_settings_page()
        
        # 푸터 렌더링
        render_footer()


# 메인 실행
if __name__ == "__main__":
    app = PolymerDOEPlatform()
    app.run()
