"""
📊 Dashboard Page - 개인 대시보드
===========================================================================
사용자별 맞춤 대시보드로 프로젝트 현황, 실험 진행상황, 
활동 타임라인, 성과 분석 등을 한눈에 보여주는 메인 화면
오프라인 우선 설계로 로컬 데이터베이스 기반 작동
===========================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import logging
from collections import defaultdict, Counter
from pathlib import Path
import sqlite3
import time

# 로컬 모듈
try:
    from utils.database_manager import DatabaseManager
    from utils.common_ui import CommonUI
    from utils.notification_manager import NotificationManager
    from utils.data_processor import DataProcessor
    from config.app_config import APP_CONFIG
    from config.theme_config import THEME_CONFIG
    from config.local_config import LOCAL_CONFIG
except ImportError as e:
    st.error(f"필요한 모듈을 불러올 수 없습니다: {e}")
    st.stop()

# ===========================================================================
# 🔧 설정 및 상수
# ===========================================================================

logger = logging.getLogger(__name__)

# 대시보드 설정
REFRESH_INTERVAL = 300  # 5분 (초)
CACHE_TTL = 300  # 5분

# 메트릭 카드 설정
METRIC_CARDS = {
    'total_projects': {
        'title': '전체 프로젝트',
        'icon': '📁',
        'color': '#7C3AED',
        'delta_prefix': '지난달 대비',
        'suffix': ''
    },
    'active_experiments': {
        'title': '진행중인 실험',
        'icon': '🧪',
        'color': '#F59E0B',
        'delta_prefix': '이번주',
        'suffix': ''
    },
    'success_rate': {
        'title': '실험 성공률',
        'icon': '📈',
        'color': '#10B981',
        'suffix': '%',
        'delta_prefix': '평균 대비'
    },
    'collaborations': {
        'title': '협업 프로젝트',
        'icon': '👥',
        'color': '#3B82F6',
        'delta_prefix': '새로운',
        'suffix': ''
    }
}

# 활동 타입
ACTIVITY_TYPES = {
    'project_created': {'icon': '🆕', 'color': '#7C3AED', 'label': '프로젝트 생성'},
    'experiment_completed': {'icon': '✅', 'color': '#10B981', 'label': '실험 완료'},
    'collaboration_joined': {'icon': '🤝', 'color': '#3B82F6', 'label': '협업 참여'},
    'file_uploaded': {'icon': '📎', 'color': '#6B7280', 'label': '파일 업로드'},
    'comment_added': {'icon': '💬', 'color': '#F59E0B', 'label': '댓글 작성'},
    'achievement_earned': {'icon': '🏆', 'color': '#FFD700', 'label': '업적 달성'}
}

# 레벨 시스템
LEVEL_THRESHOLDS = {
    'beginner': {'min': 0, 'max': 99, 'icon': '🌱', 'color': '#10B981'},
    'intermediate': {'min': 100, 'max': 499, 'icon': '🌿', 'color': '#3B82F6'},
    'advanced': {'min': 500, 'max': 1499, 'icon': '🌳', 'color': '#7C3AED'},
    'expert': {'min': 1500, 'max': 999999, 'icon': '🏆', 'color': '#F59E0B'}
}

# 업적 정의
ACHIEVEMENTS = {
    'first_project': {
        'name': '첫 발걸음',
        'description': '첫 번째 프로젝트를 생성했습니다',
        'icon': '🎯',
        'points': 10,
        'category': 'project'
    },
    'team_player': {
        'name': '팀 플레이어',
        'description': '5개 이상의 협업 프로젝트에 참여했습니다',
        'icon': '🤝',
        'points': 30,
        'category': 'collaboration'
    },
    'data_master': {
        'name': '데이터 마스터',
        'description': '100개 이상의 실험 데이터를 분석했습니다',
        'icon': '📊',
        'points': 50,
        'category': 'analysis'
    },
    'early_bird': {
        'name': '얼리버드',
        'description': '30일 연속 로그인했습니다',
        'icon': '🌅',
        'points': 20,
        'category': 'activity'
    },
    'innovator': {
        'name': '혁신가',
        'description': '새로운 실험 모듈을 개발했습니다',
        'icon': '💡',
        'points': 100,
        'category': 'contribution'
    }
}

# ===========================================================================
# 📊 대시보드 페이지 클래스
# ===========================================================================

class DashboardPage:
    """개인 대시보드 페이지"""
    
    def __init__(self):
        """초기화"""
        self.db_manager = DatabaseManager()
        self.ui = CommonUI()
        self.notification_manager = NotificationManager()
        self.data_processor = DataProcessor()
        
        # 사용자 정보
        self.user = st.session_state.get('user', {})
        self.user_id = self.user.get('user_id') or st.session_state.get('user_id')
        
        # 캐시 초기화
        self._initialize_cache()
        
        # 차트 테마 설정
        self._setup_chart_theme()
        
    def _initialize_cache(self):
        """캐시 초기화"""
        if 'dashboard_cache' not in st.session_state:
            st.session_state.dashboard_cache = {
                'metrics': {'data': None, 'timestamp': None},
                'projects': {'data': None, 'timestamp': None},
                'activities': {'data': None, 'timestamp': None},
                'charts': {'data': None, 'timestamp': None}
            }
            
    def _setup_chart_theme(self):
        """Plotly 차트 테마 설정"""
        import plotly.io as pio
        
        # 커스텀 테마 생성
        pio.templates["custom_theme"] = go.layout.Template(
            layout=go.Layout(
                font=dict(family="Pretendard, sans-serif"),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                colorway=['#7C3AED', '#F59E0B', '#10B981', '#3B82F6', '#EF4444'],
                hovermode='x unified'
            )
        )
        pio.templates.default = "custom_theme"
        
    def _is_cache_valid(self, cache_data: Dict) -> bool:
        """캐시 유효성 확인"""
        if not cache_data.get('data') or not cache_data.get('timestamp'):
            return False
            
        elapsed = (datetime.now() - cache_data['timestamp']).total_seconds()
        return elapsed < CACHE_TTL
        
    def _format_relative_time(self, timestamp: Union[str, datetime]) -> str:
        """상대 시간 포맷"""
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
            
        now = datetime.now()
        diff = now - timestamp
        
        if diff.days > 7:
            return timestamp.strftime('%Y-%m-%d')
        elif diff.days > 0:
            return f"{diff.days}일 전"
        elif diff.seconds > 3600:
            return f"{diff.seconds // 3600}시간 전"
        elif diff.seconds > 60:
            return f"{diff.seconds // 60}분 전"
        else:
            return "방금 전"
            
    # ===========================================================================
    # 🎨 렌더링 함수
    # ===========================================================================
    
    def render(self):
        """메인 렌더링 함수"""
        # 인증 확인
        if not st.session_state.get('authenticated', False) and not st.session_state.get('guest_mode', False):
            st.warning("로그인이 필요합니다.")
            if st.button("로그인 페이지로 이동"):
                st.session_state.current_page = 'auth'
                st.rerun()
            return
            
        # 페이지 제목
        st.markdown("""
        <style>
        .dashboard-container { padding: 0; }
        .metric-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: transform 0.2s;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .activity-item {
            padding: 0.75rem;
            border-left: 3px solid #E5E7EB;
            margin-bottom: 0.5rem;
            transition: all 0.2s;
        }
        .activity-item:hover {
            border-left-color: #7C3AED;
            background: #F9FAFB;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # 헤더 섹션
        self._render_header()
        
        # 메트릭 카드
        self._render_metrics()
        
        # 메인 콘텐츠
        col1, col2 = st.columns([2, 1], gap="large")
        
        with col1:
            # 프로젝트 섹션
            self._render_projects_section()
            
            # 차트 섹션
            self._render_charts_section()
            
        with col2:
            # 활동 타임라인
            self._render_activity_timeline()
            
            # 레벨 & 업적
            self._render_progress_section()
            
        # 추천 섹션
        self._render_recommendations()
        
    def _render_header(self):
        """헤더 섹션 렌더링"""
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            # 시간대별 인사
            hour = datetime.now().hour
            if hour < 12:
                greeting = "좋은 아침입니다"
                emoji = "🌅"
            elif hour < 18:
                greeting = "좋은 오후입니다"
                emoji = "☀️"
            else:
                greeting = "좋은 저녁입니다"
                emoji = "🌙"
                
            user_name = self.user.get('name', '사용자')
            if st.session_state.get('guest_mode'):
                user_name = '게스트'
                
            st.markdown(
                f"<h1 style='margin-bottom: 0;'>{emoji} {greeting}, {user_name}님!</h1>",
                unsafe_allow_html=True
            )
            
            # 마지막 로그인 정보
            if not st.session_state.get('guest_mode'):
                last_login = self.user.get('last_login')
                if last_login:
                    st.caption(f"마지막 로그인: {self._format_relative_time(last_login)}")
                    
        with col2:
            # 현재 날짜/시간
            now = datetime.now()
            st.markdown(
                f"""
                <div style='text-align: right; padding-top: 20px;'>
                    <h4 style='margin: 0;'>{now.strftime('%Y년 %m월 %d일')}</h4>
                    <p style='margin: 0; color: #6B7280;'>{now.strftime('%A %H:%M')}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with col3:
            # 빠른 작업 버튼
            st.markdown("<div style='padding-top: 15px;'>", unsafe_allow_html=True)
            
            if not st.session_state.get('guest_mode'):
                if st.button("🆕 새 프로젝트", use_container_width=True):
                    st.session_state.current_page = 'project_setup'
                    st.rerun()
                    
                # 알림 버튼
                notif_count = self._get_unread_notifications_count()
                notif_label = f"🔔 알림 ({notif_count})" if notif_count > 0 else "🔔 알림"
                if st.button(notif_label, use_container_width=True):
                    self._show_notifications_modal()
                    
            st.markdown("</div>", unsafe_allow_html=True)
            
    def _render_metrics(self):
        """메트릭 카드 렌더링"""
        st.markdown("### 📊 주요 지표")
        
        # 메트릭 데이터 가져오기
        metrics_data = self._get_metrics_data()
        
        # 4개 컬럼으로 메트릭 표시
        cols = st.columns(4)
        
        for idx, (key, config) in enumerate(METRIC_CARDS.items()):
            with cols[idx]:
                value = metrics_data.get(key, {}).get('value', 0)
                delta = metrics_data.get(key, {}).get('delta')
                
                # 메트릭 카드 렌더링
                self._render_metric_card(
                    title=config['title'],
                    value=value,
                    delta=delta,
                    delta_prefix=config['delta_prefix'],
                    icon=config['icon'],
                    color=config['color'],
                    suffix=config.get('suffix', '')
                )
                
    def _render_metric_card(self, title: str, value: Union[int, float], 
                           delta: Optional[float] = None, delta_prefix: str = "",
                           icon: str = "📊", color: str = "#7C3AED", 
                           suffix: str = ""):
        """개별 메트릭 카드 렌더링"""
        # 값 포맷팅
        if isinstance(value, float) and suffix != '%':
            value_str = f"{value:,.1f}"
        else:
            value_str = f"{value:,}" if suffix != '%' else f"{value:.1f}"
            
        # 카드 HTML
        st.markdown(
            f"""
            <div class='metric-card'>
                <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                    <span style='font-size: 24px; margin-right: 10px;'>{icon}</span>
                    <span style='color: #6B7280; font-size: 14px;'>{title}</span>
                </div>
                <div style='font-size: 32px; font-weight: bold; color: {color};'>
                    {value_str}{suffix}
                </div>
            """,
            unsafe_allow_html=True
        )
        
        # 변화량 표시
        if delta is not None:
            delta_color = '#10B981' if delta > 0 else '#EF4444'
            delta_icon = '↑' if delta > 0 else '↓'
            delta_str = f"{abs(delta):,.1f}" if suffix != '%' else f"{abs(delta):.1f}"
            
            st.markdown(
                f"""
                <div style='font-size: 14px; color: {delta_color}; margin-top: 5px;'>
                    {delta_icon} {delta_str}{suffix} {delta_prefix}
                </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown("</div>", unsafe_allow_html=True)
            
    def _render_projects_section(self):
        """프로젝트 섹션 렌더링"""
        st.markdown("### 📁 내 프로젝트")
        
        # 탭 생성
        tab1, tab2, tab3 = st.tabs(["🔥 활성 프로젝트", "📅 최근 프로젝트", "⭐ 즐겨찾기"])
        
        with tab1:
            self._render_active_projects()
            
        with tab2:
            self._render_recent_projects()
            
        with tab3:
            self._render_favorite_projects()
            
    def _render_active_projects(self):
        """활성 프로젝트 렌더링"""
        if st.session_state.get('guest_mode'):
            st.info("게스트 모드에서는 샘플 프로젝트를 보여드립니다.")
            # 샘플 데이터 표시
            self._render_sample_projects()
            return
            
        projects = self._get_active_projects()
        
        if not projects:
            st.info("진행 중인 프로젝트가 없습니다.")
            if st.button("새 프로젝트 시작하기"):
                st.session_state.current_page = 'project_setup'
                st.rerun()
            return
            
        # 프로젝트 카드 그리드
        cols = st.columns(2)
        for idx, project in enumerate(projects):
            with cols[idx % 2]:
                self._render_project_card(project)
                
    def _render_project_card(self, project: Dict):
        """프로젝트 카드 렌더링"""
        # 진행률 계산
        progress = project.get('progress', 0)
        status_color = {
            'active': '#10B981',
            'paused': '#F59E0B',
            'completed': '#3B82F6'
        }.get(project.get('status', 'active'), '#6B7280')
        
        st.markdown(
            f"""
            <div style='
                background: white;
                border-radius: 12px;
                padding: 1.5rem;
                margin-bottom: 1rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                border-left: 4px solid {status_color};
            '>
                <h4 style='margin: 0 0 0.5rem 0;'>{project.get('name', '프로젝트')}</h4>
                <p style='color: #6B7280; font-size: 14px; margin: 0;'>
                    {project.get('description', '')}
                </p>
                <div style='margin-top: 1rem;'>
                    <div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem;'>
                        <span style='font-size: 12px; color: #6B7280;'>진행률</span>
                        <span style='font-size: 12px; color: #6B7280;'>{progress}%</span>
                    </div>
                    <div style='background: #E5E7EB; border-radius: 4px; height: 8px;'>
                        <div style='
                            background: {status_color};
                            height: 100%;
                            border-radius: 4px;
                            width: {progress}%;
                            transition: width 0.3s;
                        '></div>
                    </div>
                </div>
                <div style='margin-top: 1rem; display: flex; gap: 0.5rem;'>
                    <span style='
                        background: #F3F4F6;
                        padding: 0.25rem 0.75rem;
                        border-radius: 4px;
                        font-size: 12px;
                        color: #6B7280;
                    '>
                        🧪 {project.get('experiment_count', 0)} 실험
                    </span>
                    <span style='
                        background: #F3F4F6;
                        padding: 0.25rem 0.75rem;
                        border-radius: 4px;
                        font-size: 12px;
                        color: #6B7280;
                    '>
                        📅 {self._format_relative_time(project.get('updated_at', datetime.now()))}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # 프로젝트 액션 버튼
        col1, col2 = st.columns(2)
        with col1:
            if st.button("열기", key=f"open_{project.get('id')}", use_container_width=True):
                st.session_state.current_project = project
                st.session_state.current_page = 'experiment_design'
                st.rerun()
        with col2:
            if st.button("상세", key=f"detail_{project.get('id')}", use_container_width=True):
                self._show_project_details(project)
                
    def _render_charts_section(self):
        """차트 섹션 렌더링"""
        st.markdown("### 📈 데이터 분석")
        
        # 차트 타입 선택
        chart_type = st.selectbox(
            "차트 유형",
            ["실험 트렌드", "성공률 분석", "시간별 활동", "프로젝트 분포"],
            label_visibility="collapsed"
        )
        
        if chart_type == "실험 트렌드":
            self._render_experiment_trend_chart()
        elif chart_type == "성공률 분석":
            self._render_success_rate_chart()
        elif chart_type == "시간별 활동":
            self._render_activity_heatmap()
        elif chart_type == "프로젝트 분포":
            self._render_project_distribution_chart()
            
    def _render_experiment_trend_chart(self):
        """실험 트렌드 차트"""
        # 데이터 준비
        data = self._get_experiment_trend_data()
        
        if data.empty:
            st.info("표시할 실험 데이터가 없습니다.")
            return
            
        # Plotly 차트 생성
        fig = go.Figure()
        
        # 성공한 실험
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=data['success'],
            mode='lines+markers',
            name='성공',
            line=dict(color='#10B981', width=3),
            marker=dict(size=8)
        ))
        
        # 실패한 실험
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=data['failed'],
            mode='lines+markers',
            name='실패',
            line=dict(color='#EF4444', width=3),
            marker=dict(size=8)
        ))
        
        # 레이아웃 설정
        fig.update_layout(
            title="일별 실험 결과 트렌드",
            xaxis_title="날짜",
            yaxis_title="실험 수",
            hovermode='x unified',
            showlegend=True,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def _render_activity_timeline(self):
        """활동 타임라인 렌더링"""
        st.markdown("### 📋 최근 활동")
        
        activities = self._get_recent_activities()
        
        if not activities:
            st.info("최근 활동이 없습니다.")
            return
            
        # 활동 목록 표시
        for activity in activities[:10]:  # 최근 10개
            activity_type = activity.get('type', 'unknown')
            activity_config = ACTIVITY_TYPES.get(activity_type, {
                'icon': '📌', 'color': '#6B7280', 'label': '활동'
            })
            
            st.markdown(
                f"""
                <div class='activity-item' style='border-left-color: {activity_config['color']};'>
                    <div style='display: flex; align-items: center; gap: 10px;'>
                        <span style='font-size: 20px;'>{activity_config['icon']}</span>
                        <div style='flex: 1;'>
                            <div style='font-weight: 500;'>{activity_config['label']}</div>
                            <div style='color: #6B7280; font-size: 14px;'>
                                {activity.get('description', '')}
                            </div>
                            <div style='color: #9CA3AF; font-size: 12px; margin-top: 4px;'>
                                {self._format_relative_time(activity.get('created_at', datetime.now()))}
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
    def _render_progress_section(self):
        """진행 상황 섹션 렌더링"""
        st.markdown("### 🎯 성장 현황")
        
        # 레벨 정보
        user_points = self.user.get('points', 0)
        user_level = self._get_user_level(user_points)
        next_level_points = self._get_next_level_threshold(user_points)
        progress_percentage = self._calculate_level_progress(user_points)
        
        # 레벨 카드
        level_info = LEVEL_THRESHOLDS.get(user_level, {})
        st.markdown(
            f"""
            <div style='
                background: linear-gradient(135deg, {level_info.get('color', '#7C3AED')}20, {level_info.get('color', '#7C3AED')}10);
                border-radius: 12px;
                padding: 1.5rem;
                text-align: center;
            '>
                <div style='font-size: 48px; margin-bottom: 10px;'>
                    {level_info.get('icon', '🌱')}
                </div>
                <h3 style='margin: 0; color: {level_info.get('color', '#7C3AED')};'>
                    {user_level.title()} Level
                </h3>
                <p style='margin: 10px 0; color: #6B7280;'>
                    {user_points} / {next_level_points} 포인트
                </p>
                <div style='background: #E5E7EB; border-radius: 4px; height: 8px; margin-top: 1rem;'>
                    <div style='
                        background: {level_info.get('color', '#7C3AED')};
                        height: 100%;
                        border-radius: 4px;
                        width: {progress_percentage}%;
                        transition: width 0.3s;
                    '></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # 최근 업적
        st.markdown("#### 🏆 최근 업적")
        achievements = self._get_user_achievements()
        
        if achievements:
            for achievement in achievements[:3]:  # 최근 3개
                ach_info = ACHIEVEMENTS.get(achievement.get('type', ''), {})
                st.markdown(
                    f"""
                    <div style='
                        display: flex;
                        align-items: center;
                        gap: 10px;
                        padding: 0.75rem;
                        background: #F9FAFB;
                        border-radius: 8px;
                        margin-bottom: 0.5rem;
                    '>
                        <span style='font-size: 24px;'>{ach_info.get('icon', '🏅')}</span>
                        <div>
                            <div style='font-weight: 500;'>{ach_info.get('name', '업적')}</div>
                            <div style='font-size: 12px; color: #6B7280;'>
                                {ach_info.get('description', '')}
                            </div>
                        </div>
                        <div style='margin-left: auto; color: #7C3AED; font-weight: bold;'>
                            +{ach_info.get('points', 0)}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("아직 획득한 업적이 없습니다. 계속 도전해보세요!")
            
    def _render_recommendations(self):
        """추천 섹션 렌더링"""
        st.markdown("### 💡 추천 활동")
        
        recommendations = self._get_recommendations()
        
        if not recommendations:
            return
            
        cols = st.columns(len(recommendations))
        
        for idx, rec in enumerate(recommendations):
            with cols[idx]:
                st.markdown(
                    f"""
                    <div style='
                        background: #F3F4F6;
                        border-radius: 12px;
                        padding: 1.5rem;
                        text-align: center;
                        cursor: pointer;
                        transition: all 0.2s;
                    '>
                        <div style='font-size: 32px; margin-bottom: 10px;'>
                            {rec.get('icon', '💡')}
                        </div>
                        <h4 style='margin: 0 0 0.5rem 0;'>{rec.get('title', '')}</h4>
                        <p style='font-size: 14px; color: #6B7280; margin: 0;'>
                            {rec.get('description', '')}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                if st.button(rec.get('action_label', '시작하기'), 
                           key=f"rec_{idx}", use_container_width=True):
                    if rec.get('action'):
                        rec['action']()
                        st.rerun()
                        
    # ===========================================================================
    # 📊 데이터 처리 함수
    # ===========================================================================
    
    def _get_metrics_data(self) -> Dict:
        """메트릭 데이터 조회"""
        # 캐시 확인
        cache_data = st.session_state.dashboard_cache.get('metrics', {})
        if self._is_cache_valid(cache_data):
            return cache_data['data']
            
        # 게스트 모드 처리
        if st.session_state.get('guest_mode'):
            metrics = {
                'total_projects': {'value': 5, 'delta': 2},
                'active_experiments': {'value': 3, 'delta': 1},
                'success_rate': {'value': 78.5, 'delta': 5.2},
                'collaborations': {'value': 2, 'delta': 1}
            }
        else:
            # 데이터베이스에서 조회
            metrics = self._calculate_metrics_from_db()
            
        # 캐시 업데이트
        st.session_state.dashboard_cache['metrics'] = {
            'data': metrics,
            'timestamp': datetime.now()
        }
        
        return metrics
        
    def _calculate_metrics_from_db(self) -> Dict:
        """데이터베이스에서 메트릭 계산"""
        metrics = {}
        
        try:
            # 전체 프로젝트 수
            total_projects = self.db_manager.get_user_projects_count(self.user_id)
            last_month_projects = self.db_manager.get_user_projects_count(
                self.user_id, 
                since=datetime.now() - timedelta(days=30)
            )
            
            metrics['total_projects'] = {
                'value': total_projects,
                'delta': last_month_projects
            }
            
            # 활성 실험 수
            active_experiments = self.db_manager.get_active_experiments_count(self.user_id)
            week_experiments = self.db_manager.get_experiments_count(
                self.user_id,
                since=datetime.now() - timedelta(days=7)
            )
            
            metrics['active_experiments'] = {
                'value': active_experiments,
                'delta': week_experiments
            }
            
            # 실험 성공률
            success_rate = self.db_manager.get_experiment_success_rate(self.user_id)
            avg_success_rate = self.db_manager.get_average_success_rate()
            
            metrics['success_rate'] = {
                'value': success_rate,
                'delta': success_rate - avg_success_rate if avg_success_rate else None
            }
            
            # 협업 프로젝트
            collab_count = self.db_manager.get_collaboration_count(self.user_id)
            new_collabs = self.db_manager.get_collaboration_count(
                self.user_id,
                since=datetime.now() - timedelta(days=7)
            )
            
            metrics['collaborations'] = {
                'value': collab_count,
                'delta': new_collabs
            }
            
        except Exception as e:
            logger.error(f"메트릭 계산 오류: {e}")
            # 기본값 반환
            metrics = {
                'total_projects': {'value': 0, 'delta': None},
                'active_experiments': {'value': 0, 'delta': None},
                'success_rate': {'value': 0, 'delta': None},
                'collaborations': {'value': 0, 'delta': None}
            }
            
        return metrics
        
    def _get_active_projects(self) -> List[Dict]:
        """활성 프로젝트 조회"""
        try:
            projects = self.db_manager.get_user_projects(
                self.user_id,
                status='active',
                limit=6
            )
            return projects
        except Exception as e:
            logger.error(f"프로젝트 조회 오류: {e}")
            return []
            
    def _get_recent_projects(self) -> List[Dict]:
        """최근 프로젝트 조회"""
        try:
            projects = self.db_manager.get_user_projects(
                self.user_id,
                order_by='updated_at DESC',
                limit=6
            )
            return projects
        except Exception as e:
            logger.error(f"최근 프로젝트 조회 오류: {e}")
            return []
            
    def _get_favorite_projects(self) -> List[Dict]:
        """즐겨찾기 프로젝트 조회"""
        try:
            projects = self.db_manager.get_user_projects(
                self.user_id,
                is_favorite=True,
                limit=6
            )
            return projects
        except Exception as e:
            logger.error(f"즐겨찾기 프로젝트 조회 오류: {e}")
            return []
            
    def _get_experiment_trend_data(self) -> pd.DataFrame:
        """실험 트렌드 데이터 조회"""
        try:
            # 최근 30일 데이터
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            data = self.db_manager.get_experiment_trend(
                self.user_id,
                start_date,
                end_date
            )
            
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"실험 트렌드 데이터 조회 오류: {e}")
            return pd.DataFrame()
            
    def _get_recent_activities(self) -> List[Dict]:
        """최근 활동 조회"""
        try:
            activities = self.db_manager.get_user_activities(
                self.user_id,
                limit=20
            )
            return activities
        except Exception as e:
            logger.error(f"활동 조회 오류: {e}")
            return []
            
    def _get_user_level(self, points: int) -> str:
        """사용자 레벨 계산"""
        for level, threshold in LEVEL_THRESHOLDS.items():
            if threshold['min'] <= points <= threshold['max']:
                return level
        return 'beginner'
        
    def _get_next_level_threshold(self, points: int) -> int:
        """다음 레벨 임계값"""
        current_level = self._get_user_level(points)
        level_order = list(LEVEL_THRESHOLDS.keys())
        
        current_idx = level_order.index(current_level)
        if current_idx < len(level_order) - 1:
            next_level = level_order[current_idx + 1]
            return LEVEL_THRESHOLDS[next_level]['min']
        else:
            return LEVEL_THRESHOLDS[current_level]['max']
            
    def _calculate_level_progress(self, points: int) -> float:
        """레벨 진행률 계산"""
        current_level = self._get_user_level(points)
        level_info = LEVEL_THRESHOLDS[current_level]
        
        level_range = level_info['max'] - level_info['min']
        level_progress = points - level_info['min']
        
        return min(100, (level_progress / level_range) * 100)
        
    def _get_user_achievements(self) -> List[Dict]:
        """사용자 업적 조회"""
        try:
            achievements = self.db_manager.get_user_achievements(
                self.user_id,
                limit=5
            )
            return achievements
        except Exception as e:
            logger.error(f"업적 조회 오류: {e}")
            return []
            
    def _get_recommendations(self) -> List[Dict]:
        """추천 활동 생성"""
        recommendations = []
        
        # 프로젝트가 없는 경우
        if not self._get_active_projects():
            recommendations.append({
                'icon': '🚀',
                'title': '첫 프로젝트 시작',
                'description': '고분자 연구의 첫 걸음을 내딛어보세요',
                'action_label': '프로젝트 생성',
                'action': lambda: setattr(st.session_state, 'current_page', 'project_setup')
            })
            
        # 최근 활동이 없는 경우
        recent_activities = self._get_recent_activities()
        if not recent_activities or len(recent_activities) < 5:
            recommendations.append({
                'icon': '📚',
                'title': '문헌 검색',
                'description': '최신 연구 동향을 확인해보세요',
                'action_label': '검색하기',
                'action': lambda: setattr(st.session_state, 'current_page', 'literature_search')
            })
            
        # 데이터 분석이 필요한 경우
        if self.db_manager.get_unanalyzed_experiments_count(self.user_id) > 0:
            recommendations.append({
                'icon': '📊',
                'title': '데이터 분석',
                'description': '완료된 실험 데이터를 분석해보세요',
                'action_label': '분석하기',
                'action': lambda: setattr(st.session_state, 'current_page', 'data_analysis')
            })
            
        return recommendations[:3]  # 최대 3개
        
    def _get_unread_notifications_count(self) -> int:
        """읽지 않은 알림 수 조회"""
        try:
            if st.session_state.get('guest_mode'):
                return 0
            return self.notification_manager.get_unread_count(self.user_id)
        except:
            return 0
            
    def _show_notifications_modal(self):
        """알림 모달 표시"""
        with st.expander("🔔 알림 센터", expanded=True):
            notifications = self.notification_manager.get_user_notifications(
                self.user_id,
                limit=20
            )
            
            if not notifications:
                st.info("새로운 알림이 없습니다.")
                return
                
            for notif in notifications:
                col1, col2 = st.columns([5, 1])
                
                with col1:
                    # 알림 타입별 아이콘
                    notif_icon = {
                        'project': '📁',
                        'experiment': '🧪',
                        'collaboration': '👥',
                        'system': '⚙️'
                    }.get(notif.get('type', 'system'), '🔔')
                    
                    # 읽음 여부에 따른 스타일
                    style = "font-weight: bold;" if not notif.get('is_read') else ""
                    
                    st.markdown(
                        f"""
                        <div style='padding: 0.5rem 0; {style}'>
                            <span>{notif_icon}</span> {notif.get('title', '')}
                            <div style='font-size: 14px; color: #6B7280; margin-top: 4px;'>
                                {notif.get('message', '')}
                            </div>
                            <div style='font-size: 12px; color: #9CA3AF; margin-top: 4px;'>
                                {self._format_relative_time(notif.get('created_at', datetime.now()))}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                with col2:
                    if not notif.get('is_read'):
                        if st.button("읽음", key=f"read_notif_{notif.get('id')}"):
                            self.notification_manager.mark_as_read(
                                notif.get('id'),
                                self.user_id
                            )
                            st.rerun()
                            
    def _show_project_details(self, project: Dict):
        """프로젝트 상세 정보 모달"""
        with st.expander(f"📁 {project.get('name', '프로젝트')} 상세 정보", expanded=True):
            # 기본 정보
            st.markdown("#### 기본 정보")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("생성일", project.get('created_at', '').split('T')[0])
                st.metric("실험 수", project.get('experiment_count', 0))
                
            with col2:
                st.metric("상태", project.get('status', 'active').title())
                st.metric("진행률", f"{project.get('progress', 0)}%")
                
            # 설명
            if project.get('description'):
                st.markdown("#### 설명")
                st.write(project['description'])
                
            # 최근 활동
            st.markdown("#### 최근 활동")
            activities = self.db_manager.get_project_activities(
                project.get('id'),
                limit=5
            )
            
            if activities:
                for activity in activities:
                    st.caption(
                        f"• {activity.get('description', '')} - "
                        f"{self._format_relative_time(activity.get('created_at'))}"
                    )
            else:
                st.info("최근 활동이 없습니다.")
                
    def _render_sample_projects(self):
        """게스트 모드용 샘플 프로젝트"""
        sample_projects = [
            {
                'id': 'sample_1',
                'name': '생분해성 고분자 합성',
                'description': 'PLA 기반 생분해성 고분자 개발',
                'status': 'active',
                'progress': 65,
                'experiment_count': 8,
                'updated_at': datetime.now() - timedelta(days=2)
            },
            {
                'id': 'sample_2',
                'name': '전도성 고분자 필름',
                'description': 'PEDOT:PSS 기반 투명 전극 개발',
                'status': 'active',
                'progress': 40,
                'experiment_count': 5,
                'updated_at': datetime.now() - timedelta(days=5)
            }
        ]
        
        cols = st.columns(2)
        for idx, project in enumerate(sample_projects):
            with cols[idx]:
                self._render_project_card(project)
                
    def _render_success_rate_chart(self):
        """성공률 분석 차트"""
        # 샘플 데이터 생성
        categories = ['열안정성', '기계적 강도', '전기적 특성', '광학적 특성', '내화학성']
        success_rates = [85, 72, 90, 78, 88]
        
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=success_rates,
                text=[f'{rate}%' for rate in success_rates],
                textposition='auto',
                marker_color=['#10B981' if rate >= 80 else '#F59E0B' for rate in success_rates]
            )
        ])
        
        fig.update_layout(
            title="특성별 실험 성공률",
            xaxis_title="특성 카테고리",
            yaxis_title="성공률 (%)",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def _render_activity_heatmap(self):
        """시간별 활동 히트맵"""
        # 샘플 데이터 생성
        import random
        
        days = ['월', '화', '수', '목', '금', '토', '일']
        hours = list(range(24))
        
        z = [[random.randint(0, 10) for _ in range(24)] for _ in range(7)]
        
        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=hours,
            y=days,
            colorscale='Blues',
            showscale=True
        ))
        
        fig.update_layout(
            title="요일/시간별 활동 패턴",
            xaxis_title="시간",
            yaxis_title="요일",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def _render_project_distribution_chart(self):
        """프로젝트 분포 차트"""
        # 샘플 데이터
        labels = ['생분해성', '전도성', '광학', '의료용', '기타']
        values = [30, 25, 20, 15, 10]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker_colors=['#7C3AED', '#F59E0B', '#10B981', '#3B82F6', '#EF4444']
        )])
        
        fig.update_layout(
            title="연구 분야별 프로젝트 분포",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# 🎯 메인 함수
# ===========================================================================

def render():
    """페이지 렌더링 함수"""
    page = DashboardPage()
    page.render()


if __name__ == "__main__":
    # 디버그 모드로 실행
    render()
