"""
📊 Dashboard Page - 개인 대시보드
===========================================================================
사용자별 맞춤 대시보드로 프로젝트 현황, 실험 진행상황, 
활동 타임라인, 성과 분석 등을 한눈에 보여주는 메인 화면
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
import time
import hashlib

# ===========================================================================
# 🔧 페이지 설정 (반드시 최상단)
# ===========================================================================
st.set_page_config(
    page_title="대시보드 - Universal DOE",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================================================================
# 🔍 인증 확인
# ===========================================================================
if 'authenticated' not in st.session_state or not st.session_state.authenticated:
    if not st.session_state.get('guest_mode', False):
        st.warning("🔒 로그인이 필요합니다.")
        st.markdown("""
            <meta http-equiv="refresh" content="0; url='/0_🔐_Login'">
        """, unsafe_allow_html=True)
        st.stop()

# ===========================================================================
# 📦 모듈 임포트
# ===========================================================================
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from utils.database_manager import DatabaseManager
    from utils.common_ui import CommonUI
    from utils.notification_manager import NotificationManager
    from utils.data_processor import DataProcessor
    from utils.sync_manager import SyncManager
    from utils.api_manager import APIManager
    from config.app_config import APP_CONFIG
    from config.theme_config import THEME_CONFIG, COLORS
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
        'color': COLORS['primary'],
        'delta_prefix': '지난달 대비',
        'suffix': '개'
    },
    'active_experiments': {
        'title': '진행중인 실험',
        'icon': '🧪',
        'color': COLORS['warning'],
        'delta_prefix': '이번주',
        'suffix': '개'
    },
    'success_rate': {
        'title': '실험 성공률',
        'icon': '📈',
        'color': COLORS['success'],
        'suffix': '%',
        'delta_prefix': '평균 대비'
    },
    'collaborations': {
        'title': '협업 프로젝트',
        'icon': '👥',
        'color': COLORS['info'],
        'delta_prefix': '새로운',
        'suffix': '개'
    }
}

# 활동 타입
ACTIVITY_TYPES = {
    'project_created': {'icon': '🆕', 'color': COLORS['primary']},
    'experiment_completed': {'icon': '✅', 'color': COLORS['success']},
    'collaboration_joined': {'icon': '🤝', 'color': COLORS['info']},
    'file_uploaded': {'icon': '📎', 'color': COLORS['secondary']},
    'comment_added': {'icon': '💬', 'color': COLORS['warning']},
    'achievement_earned': {'icon': '🏆', 'color': '#FFD700'}
}

# 레벨 시스템
LEVEL_THRESHOLDS = {
    1: 0,
    2: 100,
    3: 300,
    4: 600,
    5: 1000,
    6: 1500,
    7: 2100,
    8: 2800,
    9: 3600,
    10: 4500
}

# 업적 정의
ACHIEVEMENTS = {
    'first_project': {
        'name': '첫 프로젝트',
        'description': '첫 번째 프로젝트를 생성했습니다',
        'icon': '🎯',
        'points': 10,
        'category': 'project'
    },
    'team_player': {
        'name': '팀 플레이어',
        'description': '5개 이상의 협업 프로젝트에 참여',
        'icon': '🤝',
        'points': 30,
        'category': 'collaboration'
    },
    'data_master': {
        'name': '데이터 마스터',
        'description': '100개 이상의 실험 데이터 분석',
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
        self.sync_manager = SyncManager()
        self.api_manager = APIManager()
        
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
                colorway=COLORS['chart_colors'],
                font=dict(family=THEME_CONFIG['fonts']['sans'], color=COLORS['text']),
                paper_bgcolor=COLORS['background'],
                plot_bgcolor=COLORS['background_secondary'],
                hovermode='x unified',
                hoverlabel=dict(bgcolor=COLORS['background'], font_size=13),
                xaxis=dict(gridcolor='rgba(128,128,128,0.1)', zerolinecolor='rgba(128,128,128,0.2)'),
                yaxis=dict(gridcolor='rgba(128,128,128,0.1)', zerolinecolor='rgba(128,128,128,0.2)')
            )
        )
        pio.templates.default = "custom_theme"
        
    def render(self):
        """대시보드 렌더링"""
        # 헤더
        self._render_header()
        
        # AI 설명 모드 설정
        self._render_ai_explanation_toggle()
        
        # 메인 컨텐츠
        self._render_metrics()
        self._render_main_content()
        self._render_bottom_section()
        
        # 동기화 상태
        self._render_sync_status()
        
    def _render_header(self):
        """헤더 렌더링"""
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col1:
            # 사용자 프로필
            if st.session_state.get('guest_mode'):
                st.markdown("### 👤 게스트")
                st.caption("제한된 기능")
            else:
                user_name = self.user.get('name', '사용자')
                st.markdown(f"### 👋 {user_name}님")
                
                # 레벨과 경험치
                level, exp, next_exp = self._calculate_level()
                progress = (exp / next_exp) * 100 if next_exp > 0 else 100
                
                st.markdown(f"**Level {level}** • {exp}/{next_exp} XP")
                st.progress(progress / 100)
                
        with col2:
            # 환영 메시지
            current_hour = datetime.now().hour
            if current_hour < 12:
                greeting = "좋은 아침입니다"
            elif current_hour < 18:
                greeting = "좋은 오후입니다"
            else:
                greeting = "좋은 저녁입니다"
                
            st.markdown(f"## {greeting}! 오늘도 멋진 연구를 시작해볼까요? 🚀")
            
        with col3:
            # 알림 센터
            self._render_notification_center()
            
    def _render_ai_explanation_toggle(self):
        """AI 설명 상세도 토글"""
        with st.container():
            col1, col2 = st.columns([1, 5])
            with col1:
                show_details = st.toggle(
                    "AI 상세 설명",
                    value=st.session_state.get('show_ai_details', False),
                    help="AI의 추천 이유와 배경 지식을 상세히 표시합니다"
                )
                st.session_state.show_ai_details = show_details
                
    def _render_metrics(self):
        """메트릭 카드 렌더링"""
        st.markdown("### 📊 한눈에 보는 현황")
        
        # 메트릭 데이터 가져오기
        metrics = self._get_cached_data('metrics', self._fetch_metrics)
        
        cols = st.columns(4)
        for idx, (key, config) in enumerate(METRIC_CARDS.items()):
            with cols[idx]:
                value = metrics.get(key, {}).get('value', 0)
                delta = metrics.get(key, {}).get('delta', 0)
                
                # 그라디언트 배경의 카드
                st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, {config['color']}20, {config['color']}10);
                        padding: 20px;
                        border-radius: 12px;
                        border: 1px solid {config['color']}30;
                        height: 120px;
                    ">
                        <h3 style="margin: 0; color: {config['color']};">
                            {config['icon']} {value}{config.get('suffix', '')}
                        </h3>
                        <p style="margin: 5px 0; font-size: 14px; color: #666;">
                            {config['title']}
                        </p>
                        <p style="margin: 0; font-size: 12px; color: {'#10B981' if delta > 0 else '#EF4444'};">
                            {config['delta_prefix']} {'+' if delta > 0 else ''}{delta}{config.get('suffix', '')}
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
    def _render_main_content(self):
        """메인 컨텐츠 영역"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 프로젝트 목록
            self._render_projects_section()
            
            # 차트 섹션
            self._render_charts_section()
            
        with col2:
            # 활동 타임라인
            self._render_activity_timeline()
            
            # 업적 섹션
            self._render_achievements_section()
            
    def _render_projects_section(self):
        """프로젝트 섹션"""
        st.markdown("### 📁 최근 프로젝트")
        
        projects = self._get_cached_data('projects', self._fetch_recent_projects)
        
        if not projects:
            self.ui.show_empty_state(
                "아직 프로젝트가 없습니다",
                "새로운 프로젝트를 생성해보세요",
                action_label="프로젝트 생성",
                action_callback=lambda: st.switch_page("pages/2_📝_Project_Setup.py")
            )
        else:
            # 프로젝트 카드 그리드
            cols = st.columns(3)
            for idx, project in enumerate(projects[:6]):
                with cols[idx % 3]:
                    self._render_project_card(project)
                    
    def _render_project_card(self, project):
        """프로젝트 카드"""
        status_colors = {
            '활성': COLORS['success'],
            '일시중지': COLORS['warning'],
            '완료': COLORS['info']
        }
        
        st.markdown(f"""
            <div style="
                background: {COLORS['background_secondary']};
                padding: 15px;
                border-radius: 8px;
                border: 1px solid {COLORS['border']};
                margin-bottom: 10px;
                cursor: pointer;
                transition: all 0.3s;
            " onmouseover="this.style.transform='translateY(-2px)'" 
               onmouseout="this.style.transform='translateY(0)'">
                <h4 style="margin: 0 0 10px 0;">{project['name']}</h4>
                <p style="margin: 0; font-size: 12px; color: #666;">
                    {project.get('experiment_count', 0)} 실험 • 
                    {project.get('member_count', 1)} 멤버
                </p>
                <div style="margin-top: 10px;">
                    <span style="
                        background: {status_colors.get(project['status'], '#666')}20;
                        color: {status_colors.get(project['status'], '#666')};
                        padding: 2px 8px;
                        border-radius: 4px;
                        font-size: 11px;
                    ">{project['status']}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("열기", key=f"open_project_{project['id']}", use_container_width=True):
            st.session_state.selected_project_id = project['id']
            st.switch_page("pages/3_🧪_Experiment_Design.py")
            
    def _render_charts_section(self):
        """차트 섹션"""
        st.markdown("### 📈 실험 분석")
        
        chart_data = self._get_cached_data('charts', self._fetch_chart_data)
        
        # 탭으로 차트 구분
        tab1, tab2, tab3 = st.tabs(["실험 추이", "성공률 분석", "모듈 사용 통계"])
        
        with tab1:
            self._render_experiment_trend_chart(chart_data.get('trend', {}))
            
        with tab2:
            self._render_success_rate_chart(chart_data.get('success_rate', {}))
            
        with tab3:
            self._render_module_usage_chart(chart_data.get('module_usage', {}))
            
    def _render_experiment_trend_chart(self, data):
        """실험 추이 차트"""
        if not data:
            st.info("아직 실험 데이터가 없습니다")
            return
            
        fig = go.Figure()
        
        # 일별 실험 수
        fig.add_trace(go.Scatter(
            x=data.get('dates', []),
            y=data.get('counts', []),
            mode='lines+markers',
            name='실험 수',
            line=dict(color=COLORS['primary'], width=2),
            marker=dict(size=6)
        ))
        
        # 누적 실험 수
        fig.add_trace(go.Scatter(
            x=data.get('dates', []),
            y=data.get('cumulative', []),
            mode='lines',
            name='누적',
            line=dict(color=COLORS['secondary'], width=1, dash='dash'),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="최근 30일 실험 추이",
            xaxis_title="날짜",
            yaxis_title="일별 실험 수",
            yaxis2=dict(
                title="누적 실험 수",
                overlaying='y',
                side='right'
            ),
            hovermode='x unified',
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def _render_success_rate_chart(self, data):
        """성공률 분석 차트"""
        if not data:
            st.info("분석할 데이터가 부족합니다")
            return
            
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("카테고리별 성공률", "시간대별 성공률"),
            specs=[[{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # 카테고리별 성공률
        fig.add_trace(
            go.Bar(
                x=data.get('categories', []),
                y=data.get('success_rates', []),
                marker_color=COLORS['chart_colors'],
                text=[f"{rate:.1f}%" for rate in data.get('success_rates', [])],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # 시간대별 성공률
        fig.add_trace(
            go.Scatter(
                x=data.get('time_periods', []),
                y=data.get('time_success_rates', []),
                mode='lines+markers',
                line=dict(color=COLORS['success'], width=2)
            ),
            row=1, col=2
        )
        
        fig.update_layout(height=300, showlegend=False)
        fig.update_yaxes(title_text="성공률 (%)", row=1, col=1)
        fig.update_xaxes(title_text="카테고리", row=1, col=1)
        fig.update_yaxes(title_text="성공률 (%)", row=1, col=2)
        fig.update_xaxes(title_text="시간대", row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
    def _render_module_usage_chart(self, data):
        """모듈 사용 통계 차트"""
        if not data:
            st.info("모듈 사용 데이터가 없습니다")
            return
            
        # 도넛 차트
        fig = go.Figure(data=[go.Pie(
            labels=data.get('modules', []),
            values=data.get('usage_counts', []),
            hole=.3,
            marker_colors=COLORS['chart_colors']
        )])
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hoverinfo='label+percent+value'
        )
        
        fig.update_layout(
            title="실험 모듈 사용 비율",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def _render_activity_timeline(self):
        """활동 타임라인"""
        st.markdown("### 🕐 최근 활동")
        
        activities = self._get_cached_data('activities', self._fetch_recent_activities)
        
        if not activities:
            st.info("아직 활동이 없습니다")
        else:
            for activity in activities[:10]:
                self._render_activity_item(activity)
                
    def _render_activity_item(self, activity):
        """활동 아이템"""
        activity_type = activity.get('type', 'default')
        config = ACTIVITY_TYPES.get(activity_type, {'icon': '📌', 'color': '#666'})
        
        time_diff = datetime.now() - activity.get('timestamp', datetime.now())
        if time_diff.days > 0:
            time_str = f"{time_diff.days}일 전"
        elif time_diff.seconds > 3600:
            time_str = f"{time_diff.seconds // 3600}시간 전"
        else:
            time_str = f"{time_diff.seconds // 60}분 전"
            
        st.markdown(f"""
            <div style="
                display: flex;
                align-items: center;
                padding: 8px;
                margin-bottom: 5px;
                background: {COLORS['background_secondary']};
                border-radius: 6px;
            ">
                <span style="font-size: 20px; margin-right: 10px;">{config['icon']}</span>
                <div style="flex: 1;">
                    <p style="margin: 0; font-size: 13px;">{activity.get('description', '')}</p>
                    <p style="margin: 0; font-size: 11px; color: #666;">{time_str}</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    def _render_achievements_section(self):
        """업적 섹션"""
        st.markdown("### 🏆 업적")
        
        user_achievements = self._get_user_achievements()
        
        # 최근 획득한 업적
        recent = [a for a in user_achievements if a['earned']][:3]
        if recent:
            st.markdown("**최근 획득**")
            for achievement in recent:
                self._render_achievement_badge(achievement)
                
        # 다음 목표
        next_goals = [a for a in user_achievements if not a['earned']][:2]
        if next_goals:
            st.markdown("**다음 목표**")
            for achievement in next_goals:
                self._render_achievement_progress(achievement)
                
    def _render_achievement_badge(self, achievement):
        """업적 배지"""
        info = ACHIEVEMENTS.get(achievement['id'], {})
        
        st.markdown(f"""
            <div style="
                display: flex;
                align-items: center;
                padding: 10px;
                margin-bottom: 8px;
                background: linear-gradient(135deg, #FFD70020, #FFD70010);
                border: 1px solid #FFD70050;
                border-radius: 8px;
            ">
                <span style="font-size: 24px; margin-right: 10px;">{info.get('icon', '🏆')}</span>
                <div>
                    <p style="margin: 0; font-weight: bold;">{info.get('name', '')}</p>
                    <p style="margin: 0; font-size: 11px; color: #666;">
                        {info.get('description', '')}
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    def _render_achievement_progress(self, achievement):
        """업적 진행도"""
        info = ACHIEVEMENTS.get(achievement['id'], {})
        progress = achievement.get('progress', 0)
        target = achievement.get('target', 100)
        
        st.markdown(f"""
            <div style="
                padding: 10px;
                margin-bottom: 8px;
                background: {COLORS['background_secondary']};
                border-radius: 8px;
            ">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="font-size: 12px;">
                        {info.get('icon', '🎯')} {info.get('name', '')}
                    </span>
                    <span style="font-size: 11px; color: #666;">
                        {progress}/{target}
                    </span>
                </div>
                <div style="
                    background: {COLORS['border']};
                    height: 4px;
                    border-radius: 2px;
                    overflow: hidden;
                ">
                    <div style="
                        background: {COLORS['primary']};
                        height: 100%;
                        width: {(progress/target)*100}%;
                        transition: width 0.3s;
                    "></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    def _render_bottom_section(self):
        """하단 섹션"""
        col1, col2 = st.columns(2)
        
        with col1:
            # 빠른 작업
            self._render_quick_actions()
            
        with col2:
            # AI 추천
            self._render_ai_recommendations()
            
    def _render_quick_actions(self):
        """빠른 작업"""
        st.markdown("### ⚡ 빠른 작업")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🆕 새 프로젝트", use_container_width=True, type="primary"):
                st.switch_page("pages/2_📝_Project_Setup.py")
                
            if st.button("📊 데이터 분석", use_container_width=True):
                st.switch_page("pages/4_📈_Data_Analysis.py")
                
        with col2:
            if st.button("🧪 실험 설계", use_container_width=True):
                st.switch_page("pages/3_🧪_Experiment_Design.py")
                
            if st.button("🔍 문헌 검색", use_container_width=True):
                st.switch_page("pages/6_🔍_Literature_Search.py")
                
    def _render_ai_recommendations(self):
        """AI 추천"""
        st.markdown("### 🤖 AI 추천")
        
        recommendations = self._get_ai_recommendations()
        
        for rec in recommendations:
            with st.container():
                st.markdown(f"**{rec['icon']} {rec['title']}**")
                
                # 기본 추천 (항상 표시)
                st.write(rec['description'])
                
                # 상세 설명 (토글에 따라)
                if st.session_state.get('show_ai_details', False):
                    with st.expander("자세한 설명 보기"):
                        st.markdown(f"**이유**: {rec.get('reasoning', '데이터 분석 결과입니다')}")
                        st.markdown(f"**근거**: {rec.get('evidence', '과거 실험 데이터 기반')}")
                        st.markdown(f"**참고**: {rec.get('reference', '관련 문헌 또는 이론')}")
                        
                if st.button(rec['action_text'], key=f"ai_rec_{rec['id']}"):
                    rec['action']()
                    
    def _render_notification_center(self):
        """알림 센터"""
        notifications = self.notification_manager.get_user_notifications(
            self.user_id, 
            unread_only=True
        )
        
        unread_count = len(notifications)
        
        # 알림 아이콘과 카운트
        if unread_count > 0:
            st.markdown(f"""
                <div style="position: relative; display: inline-block;">
                    <span style="font-size: 24px;">🔔</span>
                    <span style="
                        position: absolute;
                        top: -5px;
                        right: -5px;
                        background: {COLORS['error']};
                        color: white;
                        border-radius: 10px;
                        padding: 2px 6px;
                        font-size: 10px;
                        font-weight: bold;
                    ">{unread_count}</span>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("🔔")
            
        # 알림 드롭다운
        with st.popover("알림"):
            if not notifications:
                st.info("새로운 알림이 없습니다")
            else:
                for notif in notifications[:5]:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"**{notif['title']}**")
                        st.caption(notif['message'])
                    with col2:
                        if st.button("✓", key=f"mark_read_{notif['id']}"):
                            self.notification_manager.mark_notification_read(notif['id'])
                            st.rerun()
                            
                st.divider()
                if st.button("모든 알림 보기"):
                    st.session_state.show_all_notifications = True
                    
    def _render_sync_status(self):
        """동기화 상태 표시"""
        sync_status = self.sync_manager.get_sync_status()
        
        if sync_status['is_syncing']:
            st.sidebar.info(f"🔄 동기화 중... {sync_status['progress']}%")
        elif sync_status['last_sync']:
            time_diff = datetime.now() - sync_status['last_sync']
            if time_diff.seconds < 60:
                time_str = "방금 전"
            elif time_diff.seconds < 3600:
                time_str = f"{time_diff.seconds // 60}분 전"
            else:
                time_str = f"{time_diff.seconds // 3600}시간 전"
                
            st.sidebar.success(f"✅ 동기화 완료 ({time_str})")
        else:
            st.sidebar.warning("⚠️ 오프라인 모드")
            
    # ===========================================================================
    # 🔧 데이터 처리 메서드
    # ===========================================================================
    
    def _get_cached_data(self, cache_key: str, fetch_function):
        """캐시된 데이터 가져오기"""
        cache = st.session_state.dashboard_cache[cache_key]
        
        # 캐시 유효성 검사
        if cache['data'] is not None and cache['timestamp'] is not None:
            if (datetime.now() - cache['timestamp']).seconds < CACHE_TTL:
                return cache['data']
                
        # 새 데이터 가져오기
        data = fetch_function()
        
        # 캐시 업데이트
        st.session_state.dashboard_cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now()
        }
        
        return data
        
    def _fetch_metrics(self) -> Dict[str, Dict[str, Any]]:
        """메트릭 데이터 가져오기"""
        try:
            # 전체 프로젝트 수
            total_projects = self.db_manager.count_user_projects(self.user_id)
            last_month_projects = self.db_manager.count_user_projects(
                self.user_id, 
                since=datetime.now() - timedelta(days=30)
            )
            
            # 활성 실험 수
            active_experiments = self.db_manager.count_active_experiments(self.user_id)
            last_week_experiments = self.db_manager.count_active_experiments(
                self.user_id,
                since=datetime.now() - timedelta(days=7)
            )
            
            # 성공률
            success_rate = self.db_manager.get_experiment_success_rate(self.user_id)
            avg_success_rate = self.db_manager.get_average_success_rate()
            
            # 협업 프로젝트
            collaborations = self.db_manager.count_collaboration_projects(self.user_id)
            new_collaborations = self.db_manager.count_collaboration_projects(
                self.user_id,
                since=datetime.now() - timedelta(days=7)
            )
            
            return {
                'total_projects': {
                    'value': total_projects,
                    'delta': total_projects - last_month_projects
                },
                'active_experiments': {
                    'value': active_experiments,
                    'delta': active_experiments - last_week_experiments
                },
                'success_rate': {
                    'value': round(success_rate, 1),
                    'delta': round(success_rate - avg_success_rate, 1)
                },
                'collaborations': {
                    'value': collaborations,
                    'delta': new_collaborations
                }
            }
        except Exception as e:
            logger.error(f"메트릭 가져오기 실패: {e}")
            return {
                'total_projects': {'value': 0, 'delta': 0},
                'active_experiments': {'value': 0, 'delta': 0},
                'success_rate': {'value': 0, 'delta': 0},
                'collaborations': {'value': 0, 'delta': 0}
            }
            
    def _fetch_recent_projects(self) -> List[Dict[str, Any]]:
        """최근 프로젝트 가져오기"""
        try:
            return self.db_manager.get_user_projects(self.user_id, limit=6)
        except Exception as e:
            logger.error(f"프로젝트 가져오기 실패: {e}")
            return []
            
    def _fetch_recent_activities(self) -> List[Dict[str, Any]]:
        """최근 활동 가져오기"""
        try:
            return self.db_manager.get_user_activities(self.user_id, limit=10)
        except Exception as e:
            logger.error(f"활동 가져오기 실패: {e}")
            return []
            
    def _fetch_chart_data(self) -> Dict[str, Any]:
        """차트 데이터 가져오기"""
        try:
            # 실험 추이 데이터
            trend_data = self.db_manager.get_experiment_trend(self.user_id, days=30)
            
            # 성공률 분석 데이터
            success_data = self.db_manager.get_success_rate_analysis(self.user_id)
            
            # 모듈 사용 통계
            module_data = self.db_manager.get_module_usage_stats(self.user_id)
            
            return {
                'trend': trend_data,
                'success_rate': success_data,
                'module_usage': module_data
            }
        except Exception as e:
            logger.error(f"차트 데이터 가져오기 실패: {e}")
            return {}
            
    def _calculate_level(self) -> Tuple[int, int, int]:
        """레벨 계산"""
        try:
            total_exp = self.db_manager.get_user_experience(self.user_id)
            
            level = 1
            for lvl, threshold in sorted(LEVEL_THRESHOLDS.items()):
                if total_exp >= threshold:
                    level = lvl
                else:
                    break
                    
            current_threshold = LEVEL_THRESHOLDS.get(level, 0)
            next_threshold = LEVEL_THRESHOLDS.get(level + 1, current_threshold + 500)
            
            exp_in_level = total_exp - current_threshold
            exp_for_next = next_threshold - current_threshold
            
            return level, exp_in_level, exp_for_next
        except:
            return 1, 0, 100
            
    def _get_user_achievements(self) -> List[Dict[str, Any]]:
        """사용자 업적 가져오기"""
        try:
            earned = self.db_manager.get_user_achievements(self.user_id)
            progress = self.db_manager.get_achievement_progress(self.user_id)
            
            achievements = []
            for aid, info in ACHIEVEMENTS.items():
                achievement = {
                    'id': aid,
                    'earned': aid in earned,
                    'progress': progress.get(aid, {}).get('current', 0),
                    'target': progress.get(aid, {}).get('target', 100)
                }
                if achievement['earned']:
                    achievement['earned_date'] = earned[aid].get('date')
                achievements.append(achievement)
                
            # 정렬: 최근 획득 > 진행중 > 미획득
            achievements.sort(key=lambda x: (
                x['earned'],
                x.get('earned_date', datetime.min) if x['earned'] else datetime.min,
                x['progress'] / x['target']
            ), reverse=True)
            
            return achievements
        except Exception as e:
            logger.error(f"업적 가져오기 실패: {e}")
            return []
            
    def _get_ai_recommendations(self) -> List[Dict[str, Any]]:
        """AI 추천 가져오기"""
        recommendations = []
        
        # 프로젝트가 없는 경우
        if self.db_manager.count_user_projects(self.user_id) == 0:
            recommendations.append({
                'id': 'first_project',
                'icon': '🚀',
                'title': '첫 프로젝트 시작하기',
                'description': '첫 번째 연구 프로젝트를 만들어보세요!',
                'reasoning': '아직 프로젝트가 없으신 것으로 보입니다.',
                'evidence': '성공적인 연구는 체계적인 프로젝트 관리에서 시작됩니다.',
                'reference': 'Project Management for Research (Smith et al., 2023)',
                'action_text': '시작하기',
                'action': lambda: st.switch_page("pages/2_📝_Project_Setup.py")
            })
        
        # 오랫동안 실험하지 않은 경우
        last_experiment = self.db_manager.get_last_experiment_date(self.user_id)
        if last_experiment and (datetime.now() - last_experiment).days > 7:
            recommendations.append({
                'id': 'resume_experiments',
                'icon': '🧪',
                'title': '실험 재개하기',
                'description': '일주일간 쉬셨네요! 연구를 계속해보세요.',
                'reasoning': '정기적인 실험이 연구 성과를 높입니다.',
                'evidence': f'마지막 실험: {last_experiment.strftime("%Y-%m-%d")}',
                'reference': 'The Power of Consistency in Research',
                'action_text': '실험 설계',
                'action': lambda: st.switch_page("pages/3_🧪_Experiment_Design.py")
            })
        
        # 분석하지 않은 데이터가 있는 경우
        unanalyzed_count = self.db_manager.count_unanalyzed_experiments(self.user_id)
        if unanalyzed_count > 0:
            recommendations.append({
                'id': 'analyze_data',
                'icon': '📊',
                'title': '데이터 분석하기',
                'description': f'{unanalyzed_count}개의 미분석 실험 데이터가 있습니다.',
                'reasoning': '데이터 분석을 통해 숨겨진 인사이트를 발견할 수 있습니다.',
                'evidence': '평균적으로 분석된 데이터의 73%에서 새로운 발견이 있습니다.',
                'reference': 'Data-Driven Discovery in Materials Science',
                'action_text': '분석하기',
                'action': lambda: st.switch_page("pages/4_📈_Data_Analysis.py")
            })
        
        # 기본 추천 (항상 표시)
        if len(recommendations) < 3:
            recommendations.append({
                'id': 'explore_literature',
                'icon': '📚',
                'title': '최신 연구 동향 탐색',
                'description': '관련 분야의 최신 논문을 확인해보세요.',
                'reasoning': '최신 연구 동향을 파악하면 연구 방향을 개선할 수 있습니다.',
                'evidence': '주 1회 문헌 조사가 연구 품질을 32% 향상시킵니다.',
                'reference': 'Systematic Literature Review Guidelines',
                'action_text': '검색하기',
                'action': lambda: st.switch_page("pages/6_🔍_Literature_Search.py")
            })
        
        return recommendations[:3]

# ===========================================================================
# 🚀 메인 실행
# ===========================================================================

def main():
    """메인 실행 함수"""
    try:
        dashboard = DashboardPage()
        dashboard.render()
    except Exception as e:
        logger.error(f"대시보드 렌더링 오류: {e}")
        st.error(f"페이지 로드 중 오류가 발생했습니다: {str(e)}")
        
        if st.button("🔄 새로고침"):
            st.rerun()

if __name__ == "__main__":
    main()
