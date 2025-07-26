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
    page_title="대시보드 - Polymer DOE",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================================================================
# 🔍 인증 확인
# ===========================================================================
if 'authenticated' not in st.session_state or not st.session_state.authenticated:
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
    'project_created': {'icon': '🆕', 'color': COLORS['primary'], 'label': '프로젝트 생성'},
    'experiment_completed': {'icon': '✅', 'color': COLORS['success'], 'label': '실험 완료'},
    'collaboration_joined': {'icon': '🤝', 'color': COLORS['info'], 'label': '협업 참여'},
    'file_uploaded': {'icon': '📎', 'color': COLORS['muted'], 'label': '파일 업로드'},
    'comment_added': {'icon': '💬', 'color': COLORS['warning'], 'label': '댓글 작성'},
    'achievement_earned': {'icon': '🏆', 'color': '#FFD700', 'label': '업적 달성'},
    'ai_analysis': {'icon': '🤖', 'color': COLORS['secondary'], 'label': 'AI 분석'}
}

# 레벨 시스템
LEVEL_SYSTEM = {
    'beginner': {'min': 0, 'max': 99, 'label': '초급 연구원', 'icon': '🌱'},
    'intermediate': {'min': 100, 'max': 499, 'label': '중급 연구원', 'icon': '🌿'},
    'advanced': {'min': 500, 'max': 1499, 'label': '고급 연구원', 'icon': '🌳'},
    'expert': {'min': 1500, 'max': None, 'label': '전문 연구원', 'icon': '🏆'}
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
# 📊 대시보드 클래스
# ===========================================================================

class DashboardPage:
    """개인 대시보드 페이지"""
    
    def __init__(self):
        """초기화"""
        self.db_manager = DatabaseManager()
        self.ui = CommonUI()
        self.notification_manager = NotificationManager()
        self.data_processor = DataProcessor()
        self.sync_manager = SyncManager(self.db_manager)
        
        # 사용자 정보
        self.user = st.session_state.get('user', {})
        self.user_id = self.user.get('user_id') or st.session_state.get('user_email')
        
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
            layout=dict(
                font=dict(family="Pretendard, sans-serif"),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                colorway=[COLORS['primary'], COLORS['secondary'], 
                         COLORS['success'], COLORS['warning'], COLORS['danger']]
            )
        )
        pio.templates.default = "custom_theme"
    
    def _check_cache_validity(self, cache_item: dict) -> bool:
        """캐시 유효성 확인"""
        if not cache_item['data'] or not cache_item['timestamp']:
            return False
        
        elapsed = (datetime.now() - cache_item['timestamp']).total_seconds()
        return elapsed < CACHE_TTL
    
    def _update_cache(self, key: str, data: Any):
        """캐시 업데이트"""
        st.session_state.dashboard_cache[key] = {
            'data': data,
            'timestamp': datetime.now()
        }
    
    # ===========================================================================
    # 📊 메트릭 데이터 처리
    # ===========================================================================
    
    def _get_metrics_data(self) -> Dict[str, Dict]:
        """메트릭 데이터 조회"""
        cache = st.session_state.dashboard_cache['metrics']
        if self._check_cache_validity(cache):
            return cache['data']
        
        metrics = {}
        
        # 전체 프로젝트
        total_projects = self.db_manager.count_user_projects(self.user_id)
        last_month_projects = self.db_manager.count_user_projects(
            self.user_id, 
            since=datetime.now() - timedelta(days=30)
        )
        
        metrics['total_projects'] = {
            'value': total_projects,
            'delta': total_projects - last_month_projects
        }
        
        # 진행중인 실험
        active_experiments = self.db_manager.count_active_experiments(self.user_id)
        last_week_experiments = self.db_manager.count_active_experiments(
            self.user_id,
            since=datetime.now() - timedelta(days=7)
        )
        
        metrics['active_experiments'] = {
            'value': active_experiments,
            'delta': active_experiments - last_week_experiments
        }
        
        # 실험 성공률
        success_rate = self.db_manager.calculate_success_rate(self.user_id)
        avg_success_rate = self.db_manager.get_average_success_rate()
        
        metrics['success_rate'] = {
            'value': round(success_rate * 100, 1),
            'delta': round((success_rate - avg_success_rate) * 100, 1)
        }
        
        # 협업 프로젝트
        collab_projects = self.db_manager.count_collaboration_projects(self.user_id)
        new_collabs = self.db_manager.count_collaboration_projects(
            self.user_id,
            since=datetime.now() - timedelta(days=7)
        )
        
        metrics['collaborations'] = {
            'value': collab_projects,
            'delta': new_collabs
        }
        
        self._update_cache('metrics', metrics)
        return metrics
    
    # ===========================================================================
    # 🎨 UI 렌더링
    # ===========================================================================
    
    def render(self):
        """메인 렌더링 함수"""
        # CSS 적용
        self.ui.apply_theme()
        
        # 헤더
        self._render_header()
        
        # 메트릭 카드
        self._render_metrics_section()
        
        # 메인 컨텐츠
        col1, col2 = st.columns([2, 1])
        
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
        
        # 알림 & 추천
        self._render_notifications_section()
        
        # 동기화 상태
        self._render_sync_status()
        
        # AI 설명 모드 설정 (전역)
        self._render_ai_explanation_mode()
    
    def _render_header(self):
        """헤더 렌더링"""
        col1, col2, col3 = st.columns([3, 1, 1])
        
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
            
            st.markdown(f"# {emoji} {greeting}, {self.user.get('name', '연구원')}님!")
            st.caption(f"오늘도 멋진 실험을 설계해보세요 🚀")
        
        with col2:
            # 현재 레벨 표시
            user_points = self.user.get('points', 0)
            user_level = self._get_user_level(user_points)
            
            st.markdown(f"""
                <div style='text-align: center; padding: 10px; 
                     background-color: {COLORS['light']}; border-radius: 10px;'>
                    <div style='font-size: 24px;'>{user_level['icon']}</div>
                    <div style='font-size: 14px; font-weight: bold;'>{user_level['label']}</div>
                    <div style='font-size: 12px; color: {COLORS['muted']};'>{user_points} 포인트</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # 빠른 액션
            if st.button("🆕 새 프로젝트", use_container_width=True, type="primary"):
                st.switch_page("pages/2_📝_Project_Setup.py")
            
            if st.button("🔔 알림", use_container_width=True):
                st.session_state.show_notifications = not st.session_state.get('show_notifications', False)
    
    def _render_metrics_section(self):
        """메트릭 카드 섹션"""
        st.markdown("### 📊 주요 지표")
        
        metrics = self._get_metrics_data()
        cols = st.columns(4)
        
        for idx, (key, config) in enumerate(METRIC_CARDS.items()):
            with cols[idx]:
                metric_data = metrics.get(key, {'value': 0, 'delta': 0})
                
                # 메트릭 카드 HTML
                st.markdown(f"""
                    <div style='
                        background: linear-gradient(135deg, {config['color']}20 0%, {config['color']}10 100%);
                        border-radius: 12px;
                        padding: 20px;
                        border: 1px solid {config['color']}30;
                        height: 140px;
                    '>
                        <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                            <span style='font-size: 24px; margin-right: 10px;'>{config['icon']}</span>
                            <span style='color: {COLORS['muted']}; font-size: 14px;'>{config['title']}</span>
                        </div>
                        <div style='font-size: 32px; font-weight: bold; color: {config['color']};'>
                            {metric_data['value']}{config['suffix']}
                        </div>
                        <div style='font-size: 14px; color: {"#10B981" if metric_data["delta"] > 0 else "#EF4444"}; margin-top: 5px;'>
                            {"↑" if metric_data["delta"] > 0 else "↓" if metric_data["delta"] < 0 else "─"} 
                            {abs(metric_data["delta"])}{config['suffix']} {config['delta_prefix']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    
    def _render_projects_section(self):
        """프로젝트 섹션"""
        st.markdown("### 📁 최근 프로젝트")
        
        # 프로젝트 목록 조회
        cache = st.session_state.dashboard_cache['projects']
        if self._check_cache_validity(cache):
            projects = cache['data']
        else:
            projects = self.db_manager.get_user_projects(self.user_id, limit=5)
            self._update_cache('projects', projects)
        
        if not projects:
            st.info("아직 프로젝트가 없습니다. 새 프로젝트를 시작해보세요!")
            if st.button("🚀 첫 프로젝트 만들기"):
                st.switch_page("pages/2_📝_Project_Setup.py")
        else:
            # 프로젝트 카드 그리드
            for project in projects:
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.markdown(f"""
                            <div style='
                                background-color: {COLORS['surface']};
                                border-radius: 8px;
                                padding: 15px;
                                border: 1px solid {COLORS['light']};
                            '>
                                <h4 style='margin: 0;'>{project['name']}</h4>
                                <p style='color: {COLORS['muted']}; margin: 5px 0;'>
                                    {project.get('description', '설명 없음')}
                                </p>
                                <div style='display: flex; gap: 10px; margin-top: 10px;'>
                                    <span style='font-size: 12px; background-color: {COLORS['primary']}20; 
                                          color: {COLORS['primary']}; padding: 4px 8px; border-radius: 4px;'>
                                        {project.get('field', '일반')}
                                    </span>
                                    <span style='font-size: 12px; color: {COLORS['muted']}'>
                                        실험: {project.get('experiment_count', 0)}개
                                    </span>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # 프로젝트 상태
                        status = project.get('status', 'active')
                        status_color = COLORS['success'] if status == 'active' else COLORS['muted']
                        status_text = '진행중' if status == 'active' else '완료'
                        
                        st.markdown(f"""
                            <div style='text-align: center; padding-top: 20px;'>
                                <span style='color: {status_color}; font-weight: bold;'>
                                    ● {status_text}
                                </span>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        # 액션 버튼
                        if st.button("열기", key=f"open_project_{project['id']}"):
                            st.session_state.current_project = project
                            st.switch_page("pages/3_🧪_Experiment_Design.py")
    
    def _render_charts_section(self):
        """차트 섹션"""
        st.markdown("### 📈 실험 분석")
        
        # 탭 생성
        tab1, tab2, tab3 = st.tabs(["실험 추이", "성공률 분석", "모듈별 사용"])
        
        with tab1:
            self._render_experiment_trend_chart()
        
        with tab2:
            self._render_success_rate_chart()
        
        with tab3:
            self._render_module_usage_chart()
    
    def _render_experiment_trend_chart(self):
        """실험 추이 차트"""
        # 데이터 준비
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        dates = pd.date_range(start_date, end_date, freq='D')
        
        # 실제 데이터 조회 (여기서는 시뮬레이션)
        experiments_per_day = []
        for date in dates:
            count = self.db_manager.count_experiments_on_date(self.user_id, date)
            experiments_per_day.append(count or np.random.randint(0, 5))
        
        # Plotly 차트
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=experiments_per_day,
            mode='lines+markers',
            name='실험 수',
            line=dict(color=COLORS['primary'], width=3),
            marker=dict(size=6),
            fill='tozeroy',
            fillcolor=f"rgba(124, 58, 237, 0.1)"
        ))
        
        fig.update_layout(
            title="최근 30일 실험 추이",
            xaxis_title="날짜",
            yaxis_title="실험 수",
            hovermode='x unified',
            height=350,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_success_rate_chart(self):
        """성공률 분석 차트"""
        # 데이터 준비
        success_data = self.db_manager.get_success_rate_by_month(self.user_id, months=6)
        
        if not success_data:
            # 시뮬레이션 데이터
            months = pd.date_range(end=datetime.now(), periods=6, freq='M')
            success_rates = [70 + np.random.randint(-10, 20) for _ in range(6)]
            avg_rate = 75
        else:
            months = [d['month'] for d in success_data]
            success_rates = [d['rate'] * 100 for d in success_data]
            avg_rate = sum(success_rates) / len(success_rates)
        
        # Plotly 차트
        fig = go.Figure()
        
        # 성공률 바
        fig.add_trace(go.Bar(
            x=months,
            y=success_rates,
            name='성공률',
            marker_color=[COLORS['success'] if r >= avg_rate else COLORS['warning'] 
                         for r in success_rates]
        ))
        
        # 평균선
        fig.add_hline(
            y=avg_rate,
            line_dash="dash",
            line_color=COLORS['muted'],
            annotation_text=f"평균: {avg_rate:.1f}%"
        )
        
        fig.update_layout(
            title="월별 실험 성공률",
            xaxis_title="월",
            yaxis_title="성공률 (%)",
            height=350,
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_module_usage_chart(self):
        """모듈별 사용 차트"""
        # 데이터 준비
        module_data = self.db_manager.get_module_usage_stats(self.user_id)
        
        if not module_data:
            # 시뮬레이션 데이터
            modules = ['화학합성', '재료특성', '바이오고분자', '복합재료', '기타']
            values = [30, 25, 20, 15, 10]
        else:
            modules = [d['module'] for d in module_data]
            values = [d['count'] for d in module_data]
        
        # Plotly 도넛 차트
        fig = go.Figure(data=[go.Pie(
            labels=modules,
            values=values,
            hole=.4,
            marker_colors=[COLORS['primary'], COLORS['secondary'], 
                          COLORS['success'], COLORS['warning'], COLORS['info']]
        )])
        
        fig.update_layout(
            title="실험 모듈 사용 비율",
            height=350,
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_activity_timeline(self):
        """활동 타임라인"""
        st.markdown("### 🕐 최근 활동")
        
        # 활동 데이터 조회
        activities = self.db_manager.get_user_activities(self.user_id, limit=10)
        
        if not activities:
            st.info("아직 활동 기록이 없습니다.")
        else:
            for activity in activities:
                activity_type = ACTIVITY_TYPES.get(activity['type'], {})
                
                # 시간 포맷
                time_diff = datetime.now() - activity['timestamp']
                if time_diff.days > 0:
                    time_str = f"{time_diff.days}일 전"
                elif time_diff.seconds > 3600:
                    time_str = f"{time_diff.seconds // 3600}시간 전"
                else:
                    time_str = f"{time_diff.seconds // 60}분 전"
                
                # 활동 카드
                st.markdown(f"""
                    <div style='
                        background-color: {COLORS['surface']};
                        border-left: 3px solid {activity_type.get('color', COLORS['muted'])};
                        padding: 10px 15px;
                        margin-bottom: 10px;
                        border-radius: 0 8px 8px 0;
                    '>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <div>
                                <span style='font-size: 18px; margin-right: 8px;'>
                                    {activity_type.get('icon', '📌')}
                                </span>
                                <span style='font-weight: bold;'>
                                    {activity_type.get('label', activity['type'])}
                                </span>
                            </div>
                            <span style='font-size: 12px; color: {COLORS['muted']};'>
                                {time_str}
                            </span>
                        </div>
                        <div style='margin-top: 5px; font-size: 14px; color: {COLORS['text_secondary']};'>
                            {activity.get('description', '')}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    
    def _render_progress_section(self):
        """레벨 & 업적 섹션"""
        st.markdown("### 🏆 성장 현황")
        
        # 레벨 진행률
        user_points = self.user.get('points', 0)
        current_level = self._get_user_level(user_points)
        next_level = self._get_next_level(user_points)
        
        if next_level:
            progress = (user_points - current_level['min']) / (next_level['min'] - current_level['min'])
            points_needed = next_level['min'] - user_points
            
            st.markdown(f"""
                <div style='margin-bottom: 20px;'>
                    <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                        <span>{current_level['label']}</span>
                        <span>{next_level['label']}</span>
                    </div>
                    <div style='background-color: {COLORS['light']}; border-radius: 10px; height: 20px;'>
                        <div style='
                            background: linear-gradient(90deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
                            width: {progress * 100}%;
                            height: 100%;
                            border-radius: 10px;
                            transition: width 0.5s ease;
                        '></div>
                    </div>
                    <div style='text-align: center; margin-top: 5px; font-size: 12px; color: {COLORS['muted']};'>
                        {points_needed} 포인트 더 필요
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        # 최근 획득 업적
        st.markdown("#### 최근 업적")
        
        recent_achievements = self.db_manager.get_user_achievements(self.user_id, limit=3)
        
        if not recent_achievements:
            st.info("아직 획득한 업적이 없습니다. 계속 활동하면 업적을 얻을 수 있어요!")
        else:
            for achievement in recent_achievements:
                ach_data = ACHIEVEMENTS.get(achievement['type'], {})
                
                st.markdown(f"""
                    <div style='
                        background-color: {COLORS['warning']}20;
                        border: 1px solid {COLORS['warning']}40;
                        border-radius: 8px;
                        padding: 10px;
                        margin-bottom: 10px;
                    '>
                        <div style='display: flex; align-items: center;'>
                            <span style='font-size: 24px; margin-right: 10px;'>
                                {ach_data.get('icon', '🏅')}
                            </span>
                            <div>
                                <div style='font-weight: bold;'>{ach_data.get('name', '업적')}</div>
                                <div style='font-size: 12px; color: {COLORS['muted']};'>
                                    {ach_data.get('description', '')}
                                </div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    
    def _render_notifications_section(self):
        """알림 & 추천 섹션"""
        if st.session_state.get('show_notifications', False):
            with st.container():
                st.markdown("### 🔔 알림")
                
                notifications = self.notification_manager.get_unread_notifications(self.user_id)
                
                if not notifications:
                    st.info("새로운 알림이 없습니다.")
                else:
                    for notif in notifications[:5]:
                        col1, col2 = st.columns([10, 1])
                        
                        with col1:
                            st.markdown(f"""
                                <div style='
                                    background-color: {COLORS['info']}10;
                                    border-left: 3px solid {COLORS['info']};
                                    padding: 10px;
                                    margin-bottom: 10px;
                                    border-radius: 0 8px 8px 0;
                                '>
                                    <div style='font-weight: bold;'>{notif['title']}</div>
                                    <div style='font-size: 14px; margin-top: 5px;'>{notif['message']}</div>
                                    <div style='font-size: 12px; color: {COLORS['muted']}; margin-top: 5px;'>
                                        {self._format_time(notif['created_at'])}
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            if st.button("✓", key=f"mark_read_{notif['id']}"):
                                self.notification_manager.mark_as_read(notif['id'])
                                st.rerun()
        
        # AI 추천
        st.markdown("### 💡 AI 추천")
        
        recommendations = self._get_ai_recommendations()
        
        cols = st.columns(len(recommendations))
        for idx, rec in enumerate(recommendations):
            with cols[idx]:
                st.markdown(f"""
                    <div style='
                        background: linear-gradient(135deg, {COLORS['secondary']}10 0%, {COLORS['primary']}10 100%);
                        border-radius: 12px;
                        padding: 15px;
                        height: 150px;
                        border: 1px solid {COLORS['light']};
                    '>
                        <div style='font-size: 24px; margin-bottom: 10px;'>{rec['icon']}</div>
                        <div style='font-weight: bold; margin-bottom: 5px;'>{rec['title']}</div>
                        <div style='font-size: 12px; color: {COLORS['text_secondary']};'>
                            {rec['description']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                if st.button(rec['action_text'], key=f"rec_{idx}", use_container_width=True):
                    rec['action']()
    
    def _render_sync_status(self):
        """동기화 상태 표시"""
        sync_status = self.sync_manager.get_sync_status()
        
        if sync_status['is_online']:
            status_color = COLORS['success']
            status_text = "온라인"
            status_icon = "🟢"
        else:
            status_color = COLORS['muted']
            status_text = "오프라인"
            status_icon = "⚫"
        
        # 우측 하단 고정 위치
        st.markdown(f"""
            <div style='
                position: fixed;
                bottom: 20px;
                right: 20px;
                background-color: {COLORS['surface']};
                border: 1px solid {status_color};
                border-radius: 20px;
                padding: 8px 16px;
                display: flex;
                align-items: center;
                gap: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                z-index: 999;
            '>
                <span>{status_icon}</span>
                <span style='font-size: 14px; color: {status_color};'>{status_text}</span>
            </div>
        """, unsafe_allow_html=True)
    
    def _render_ai_explanation_mode(self):
        """AI 설명 모드 설정 (전역)"""
        with st.sidebar.expander("🤖 AI 설명 설정", expanded=False):
            st.markdown("AI 응답의 상세도를 조절할 수 있습니다.")
            
            explanation_mode = st.radio(
                "설명 모드",
                ["자동 (레벨 기반)", "항상 간단히", "항상 상세히"],
                index=0,
                key="ai_explanation_mode"
            )
            
            st.info("""
                💡 **팁**: 언제든지 AI 응답 옆의 '🔍' 버튼을 클릭하여 
                상세 설명을 보거나 숨길 수 있습니다.
            """)
    
    # ===========================================================================
    # 🔧 헬퍼 함수
    # ===========================================================================
    
    def _get_user_level(self, points: int) -> Dict:
        """사용자 레벨 계산"""
        for level_key, level_data in LEVEL_SYSTEM.items():
            if level_data['max'] is None or points <= level_data['max']:
                return {
                    'key': level_key,
                    'label': level_data['label'],
                    'icon': level_data['icon'],
                    'min': level_data['min'],
                    'max': level_data['max']
                }
        return LEVEL_SYSTEM['expert']
    
    def _get_next_level(self, points: int) -> Optional[Dict]:
        """다음 레벨 정보"""
        levels = list(LEVEL_SYSTEM.items())
        for i, (level_key, level_data) in enumerate(levels):
            if level_data['max'] and points <= level_data['max']:
                if i + 1 < len(levels):
                    next_key, next_data = levels[i + 1]
                    return {
                        'key': next_key,
                        'label': next_data['label'],
                        'icon': next_data['icon'],
                        'min': next_data['min'],
                        'max': next_data['max']
                    }
        return None
    
    def _format_time(self, timestamp: datetime) -> str:
        """시간 포맷팅"""
        diff = datetime.now() - timestamp
        
        if diff.days > 0:
            return f"{diff.days}일 전"
        elif diff.seconds > 3600:
            return f"{diff.seconds // 3600}시간 전"
        elif diff.seconds > 60:
            return f"{diff.seconds // 60}분 전"
        else:
            return "방금 전"
    
    def _get_ai_recommendations(self) -> List[Dict]:
        """AI 추천 생성"""
        recommendations = []
        
        # 프로젝트가 없는 경우
        if self.db_manager.count_user_projects(self.user_id) == 0:
            recommendations.append({
                'icon': '🚀',
                'title': '첫 프로젝트 시작',
                'description': 'AI가 도와드릴게요!',
                'action_text': '시작하기',
                'action': lambda: st.switch_page("pages/2_📝_Project_Setup.py")
            })
        
        # 오랫동안 실험하지 않은 경우
        last_experiment = self.db_manager.get_last_experiment_date(self.user_id)
        if last_experiment and (datetime.now() - last_experiment).days > 7:
            recommendations.append({
                'icon': '🧪',
                'title': '실험 재개하기',
                'description': '일주일간 쉬셨네요!',
                'action_text': '실험 설계',
                'action': lambda: st.switch_page("pages/3_🧪_Experiment_Design.py")
            })
        
        # 분석하지 않은 데이터가 있는 경우
        unanalyzed_count = self.db_manager.count_unanalyzed_experiments(self.user_id)
        if unanalyzed_count > 0:
            recommendations.append({
                'icon': '📊',
                'title': '데이터 분석하기',
                'description': f'{unanalyzed_count}개의 미분석 데이터',
                'action_text': '분석하기',
                'action': lambda: st.switch_page("pages/4_📈_Data_Analysis.py")
            })
        
        # 기본 추천
        if len(recommendations) < 3:
            recommendations.append({
                'icon': '📚',
                'title': '문헌 검색',
                'description': '최신 연구 동향 확인',
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
