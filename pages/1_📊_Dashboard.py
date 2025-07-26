"""
📊 대시보드 - Universal DOE Platform
=============================================================================
데스크톱 앱용 개인 맞춤 대시보드
SQLite 로컬 DB 기반, 오프라인 우선 설계, 선택적 클라우드 동기화
=============================================================================
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import logging
from collections import defaultdict, Counter
import time
import sqlite3

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

# 로컬 모듈
try:
    from utils.database_manager import get_database_manager
    from utils.auth_manager import get_auth_manager, UserRole
    from utils.common_ui import get_common_ui
    from config.app_config import APP_INFO, AI_EXPLANATION_CONFIG
    from config.local_config import LOCAL_CONFIG
    from config.offline_config import OFFLINE_CONFIG
except ImportError as e:
    st.error(f"🚨 필수 모듈을 찾을 수 없습니다: {str(e)}")
    st.stop()

# 로깅 설정
logger = logging.getLogger(__name__)

# 페이지 설정
st.set_page_config(
    page_title="대시보드 - Universal DOE Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 🔒 인증 체크
# =============================================================================
if 'authenticated' not in st.session_state or not st.session_state.authenticated:
    st.warning("🔒 로그인이 필요합니다.")
    st.switch_page("pages/0_🔐_Login.py")
    st.stop()

# 게스트 모드 체크
is_guest = st.session_state.get('user_role') == UserRole.GUEST

# =============================================================================
# 🎨 커스텀 CSS
# =============================================================================
CUSTOM_CSS = """
<style>
    /* 메트릭 카드 스타일 */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid;
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-delta {
        font-size: 0.875rem;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        display: inline-block;
    }
    
    .delta-positive {
        background: #e6f7e6;
        color: #0d9e0d;
    }
    
    .delta-negative {
        background: #fee;
        color: #c33;
    }
    
    /* 활동 타임라인 */
    .activity-item {
        padding: 1rem;
        margin: 0.5rem 0;
        background: #f8f9fa;
        border-radius: 8px;
        border-left: 3px solid #667eea;
        transition: all 0.2s ease;
    }
    
    .activity-item:hover {
        background: #e9ecef;
        transform: translateX(2px);
    }
    
    /* 프로젝트 카드 */
    .project-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .project-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transform: translateY(-1px);
    }
    
    /* 진행률 바 */
    .progress-bar {
        height: 8px;
        background: #e9ecef;
        border-radius: 4px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    
    /* 알림 배지 */
    .notification-badge {
        background: #ff4757;
        color: white;
        font-size: 0.75rem;
        padding: 0.2rem 0.5rem;
        border-radius: 10px;
        position: absolute;
        top: -5px;
        right: -5px;
    }
    
    /* 동기화 상태 */
    .sync-status {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: #f8f9fa;
        border-radius: 20px;
        font-size: 0.875rem;
    }
    
    .sync-online {
        color: #10b981;
    }
    
    .sync-offline {
        color: #6b7280;
    }
    
    .sync-pending {
        color: #f59e0b;
    }
    
    /* 차트 컨테이너 */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }
    
    /* 레벨 프로그레스 */
    .level-progress {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
    }
    
    .level-icon {
        font-size: 2rem;
    }
    
    .level-info {
        flex: 1;
    }
    
    .level-bar {
        height: 10px;
        background: rgba(255,255,255,0.3);
        border-radius: 5px;
        overflow: hidden;
        margin-top: 0.5rem;
    }
    
    .level-fill {
        height: 100%;
        background: white;
        border-radius: 5px;
        transition: width 0.5s ease;
    }
</style>
"""

# =============================================================================
# 🔧 설정 및 상수
# =============================================================================

# 대시보드 설정
REFRESH_INTERVAL = 300  # 5분
CACHE_TTL = 300  # 5분

# 메트릭 카드 설정
METRIC_CARDS = {
    'total_projects': {
        'title': '전체 프로젝트',
        'icon': '📁',
        'color': '#667eea',
        'suffix': '개'
    },
    'active_experiments': {
        'title': '진행중인 실험',
        'icon': '🧪',
        'color': '#f59e0b',
        'suffix': '개'
    },
    'success_rate': {
        'title': '실험 성공률',
        'icon': '📈',
        'color': '#10b981',
        'suffix': '%'
    },
    'collaboration_count': {
        'title': '협업 프로젝트',
        'icon': '👥',
        'color': '#3b82f6',
        'suffix': '개'
    }
}

# 활동 타입
ACTIVITY_TYPES = {
    'project_created': {'icon': '🆕', 'text': '새 프로젝트 생성'},
    'experiment_started': {'icon': '🧪', 'text': '실험 시작'},
    'experiment_completed': {'icon': '✅', 'text': '실험 완료'},
    'data_analyzed': {'icon': '📊', 'text': '데이터 분석'},
    'report_generated': {'icon': '📄', 'text': '보고서 생성'},
    'collaboration_joined': {'icon': '🤝', 'text': '협업 참여'},
    'module_installed': {'icon': '📦', 'text': '모듈 설치'},
    'achievement_earned': {'icon': '🏆', 'text': '업적 달성'}
}

# 레벨 시스템
LEVEL_SYSTEM = {
    'beginner': {'min_points': 0, 'icon': '🌱', 'name': '초보 연구원'},
    'intermediate': {'min_points': 100, 'icon': '🌿', 'name': '중급 연구원'},
    'advanced': {'min_points': 500, 'icon': '🌳', 'name': '고급 연구원'},
    'expert': {'min_points': 1500, 'icon': '🏆', 'name': '전문 연구원'}
}

# =============================================================================
# 🔧 유틸리티 함수
# =============================================================================

def init_session_state():
    """세션 상태 초기화"""
    defaults = {
        'dashboard_cache': {},
        'last_refresh': datetime.now(),
        'sync_status': 'checking',
        'selected_project': None,
        'show_ai_details': st.session_state.get('show_ai_details', False)
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def check_cache(cache_key: str) -> Optional[Any]:
    """캐시 확인"""
    if cache_key in st.session_state.dashboard_cache:
        cache_data = st.session_state.dashboard_cache[cache_key]
        if datetime.now() - cache_data['timestamp'] < timedelta(seconds=CACHE_TTL):
            return cache_data['data']
    return None

def update_cache(cache_key: str, data: Any):
    """캐시 업데이트"""
    st.session_state.dashboard_cache[cache_key] = {
        'data': data,
        'timestamp': datetime.now()
    }

def check_sync_status() -> str:
    """동기화 상태 확인"""
    db_manager = get_database_manager()
    
    # 오프라인 모드 확인
    if not st.session_state.get('online_status', False):
        return 'offline'
    
    # 동기화 대기 중인 변경사항 확인
    pending_changes = db_manager.count('sync_queue', {'status': 'pending'})
    if pending_changes > 0:
        return 'pending'
    
    return 'synced'

def get_user_level(points: int) -> Dict[str, Any]:
    """사용자 레벨 계산"""
    current_level = None
    next_level = None
    
    for level_key, level_info in LEVEL_SYSTEM.items():
        if points >= level_info['min_points']:
            current_level = level_key
        else:
            if next_level is None:
                next_level = level_key
                break
    
    if current_level is None:
        current_level = 'beginner'
    
    return {
        'current': current_level,
        'next': next_level,
        'current_info': LEVEL_SYSTEM[current_level],
        'next_info': LEVEL_SYSTEM.get(next_level),
        'progress': calculate_level_progress(points, current_level, next_level)
    }

def calculate_level_progress(points: int, current_level: str, next_level: Optional[str]) -> float:
    """레벨 진행률 계산"""
    if next_level is None:
        return 100.0
    
    current_min = LEVEL_SYSTEM[current_level]['min_points']
    next_min = LEVEL_SYSTEM[next_level]['min_points']
    
    progress = (points - current_min) / (next_min - current_min) * 100
    return min(max(progress, 0), 100)

# =============================================================================
# 📊 데이터 로드 함수
# =============================================================================

def load_metrics_data() -> Dict[str, Any]:
    """메트릭 데이터 로드"""
    # 캐시 확인
    cached_data = check_cache('metrics')
    if cached_data:
        return cached_data
    
    db_manager = get_database_manager()
    user_id = st.session_state.user['id']
    
    metrics = {}
    
    # 전체 프로젝트 수
    metrics['total_projects'] = {
        'value': db_manager.count('projects', {'user_id': user_id}),
        'delta': calculate_monthly_change('projects', user_id)
    }
    
    # 진행중인 실험
    metrics['active_experiments'] = {
        'value': db_manager.count('experiments', {
            'user_id': user_id,
            'status': 'active'
        }),
        'delta': calculate_weekly_change('experiments', user_id)
    }
    
    # 성공률 계산
    completed = db_manager.count('experiments', {
        'user_id': user_id,
        'status': 'completed'
    })
    successful = db_manager.count('experiments', {
        'user_id': user_id,
        'status': 'completed',
        'result': 'success'
    })
    
    success_rate = (successful / completed * 100) if completed > 0 else 0
    metrics['success_rate'] = {
        'value': round(success_rate, 1),
        'delta': success_rate - 75.0  # 평균 대비
    }
    
    # 협업 프로젝트
    metrics['collaboration_count'] = {
        'value': db_manager.count('project_collaborators', {
            'user_id': user_id
        }),
        'delta': calculate_monthly_change('project_collaborators', user_id)
    }
    
    # 캐시 업데이트
    update_cache('metrics', metrics)
    
    return metrics

def calculate_monthly_change(table: str, user_id: str) -> int:
    """월간 변화량 계산"""
    db_manager = get_database_manager()
    
    # 이번 달
    this_month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0)
    this_month_count = db_manager.count(table, {
        'user_id': user_id,
        'created_at': f'>= "{this_month_start.isoformat()}"'
    })
    
    # 지난 달
    last_month_start = (this_month_start - timedelta(days=1)).replace(day=1)
    last_month_end = this_month_start - timedelta(seconds=1)
    last_month_count = db_manager.count(table, {
        'user_id': user_id,
        'created_at': f'BETWEEN "{last_month_start.isoformat()}" AND "{last_month_end.isoformat()}"'
    })
    
    return this_month_count - last_month_count

def calculate_weekly_change(table: str, user_id: str) -> int:
    """주간 변화량 계산"""
    db_manager = get_database_manager()
    
    # 이번 주
    this_week_start = datetime.now() - timedelta(days=datetime.now().weekday())
    this_week_start = this_week_start.replace(hour=0, minute=0, second=0)
    
    this_week_count = db_manager.count(table, {
        'user_id': user_id,
        'created_at': f'>= "{this_week_start.isoformat()}"'
    })
    
    return this_week_count

def load_recent_activities(limit: int = 10) -> List[Dict[str, Any]]:
    """최근 활동 로드"""
    cached_data = check_cache('activities')
    if cached_data:
        return cached_data
    
    db_manager = get_database_manager()
    user_id = st.session_state.user['id']
    
    # 활동 로그 조회
    activities = db_manager.query("""
        SELECT * FROM activity_logs 
        WHERE user_id = ? 
        ORDER BY created_at DESC 
        LIMIT ?
    """, (user_id, limit))
    
    # 활동 데이터 처리
    processed_activities = []
    for activity in activities:
        activity_type = activity['activity_type']
        if activity_type in ACTIVITY_TYPES:
            processed_activities.append({
                'id': activity['id'],
                'type': activity_type,
                'icon': ACTIVITY_TYPES[activity_type]['icon'],
                'text': ACTIVITY_TYPES[activity_type]['text'],
                'details': json.loads(activity.get('details', '{}')),
                'created_at': datetime.fromisoformat(activity['created_at'])
            })
    
    update_cache('activities', processed_activities)
    return processed_activities

def load_projects_data(status: Optional[str] = None) -> List[Dict[str, Any]]:
    """프로젝트 데이터 로드"""
    db_manager = get_database_manager()
    user_id = st.session_state.user['id']
    
    # 쿼리 조건
    conditions = {'user_id': user_id}
    if status:
        conditions['status'] = status
    
    # 프로젝트 조회
    projects = db_manager.query("""
        SELECT p.*, 
               COUNT(DISTINCT e.id) as experiment_count,
               COUNT(DISTINCT c.user_id) as collaborator_count
        FROM projects p
        LEFT JOIN experiments e ON p.id = e.project_id
        LEFT JOIN project_collaborators c ON p.id = c.project_id
        WHERE p.user_id = ?
        GROUP BY p.id
        ORDER BY p.updated_at DESC
    """, (user_id,))
    
    return projects

def load_chart_data() -> Dict[str, Any]:
    """차트 데이터 로드"""
    cached_data = check_cache('charts')
    if cached_data:
        return cached_data
    
    db_manager = get_database_manager()
    user_id = st.session_state.user['id']
    
    chart_data = {}
    
    # 프로젝트 상태 분포
    chart_data['project_status'] = db_manager.query("""
        SELECT status, COUNT(*) as count
        FROM projects
        WHERE user_id = ?
        GROUP BY status
    """, (user_id,))
    
    # 월별 실험 추이
    chart_data['monthly_experiments'] = db_manager.query("""
        SELECT strftime('%Y-%m', created_at) as month,
               COUNT(*) as count,
               COUNT(CASE WHEN result = 'success' THEN 1 END) as success_count
        FROM experiments
        WHERE user_id = ?
        AND created_at >= date('now', '-6 months')
        GROUP BY month
        ORDER BY month
    """, (user_id,))
    
    # 실험 모듈별 사용 통계
    chart_data['module_usage'] = db_manager.query("""
        SELECT module_type, COUNT(*) as count
        FROM experiments
        WHERE user_id = ?
        GROUP BY module_type
        ORDER BY count DESC
        LIMIT 5
    """, (user_id,))
    
    update_cache('charts', chart_data)
    return chart_data

# =============================================================================
# 🎨 UI 렌더링 함수
# =============================================================================

def render_header():
    """헤더 렌더링"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        user = st.session_state.user
        st.markdown(f"## 👋 안녕하세요, {user['name']}님!")
        st.caption(f"오늘도 멋진 실험을 설계해보세요 🚀")
    
    with col2:
        # 동기화 상태
        sync_status = check_sync_status()
        if sync_status == 'synced':
            st.markdown("""
                <div class="sync-status sync-online">
                    <span>🟢 동기화됨</span>
                </div>
            """, unsafe_allow_html=True)
        elif sync_status == 'pending':
            st.markdown("""
                <div class="sync-status sync-pending">
                    <span>🟡 동기화 대기중</span>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="sync-status sync-offline">
                    <span>🔴 오프라인</span>
                </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if st.button("🔄 새로고침"):
            st.session_state.dashboard_cache.clear()
            st.session_state.last_refresh = datetime.now()
            st.rerun()

def render_metrics_section():
    """메트릭 카드 섹션"""
    st.markdown("### 📊 주요 지표")
    
    metrics = load_metrics_data()
    cols = st.columns(4)
    
    for i, (key, config) in enumerate(METRIC_CARDS.items()):
        with cols[i]:
            metric_data = metrics.get(key, {'value': 0, 'delta': 0})
            
            # 카드 HTML
            delta_class = 'delta-positive' if metric_data['delta'] >= 0 else 'delta-negative'
            delta_symbol = '+' if metric_data['delta'] >= 0 else ''
            
            st.markdown(f"""
                <div class="metric-card" style="border-color: {config['color']}">
                    <div style="color: #6b7280; font-size: 0.875rem;">
                        {config['icon']} {config['title']}
                    </div>
                    <div class="metric-value" style="color: {config['color']}">
                        {metric_data['value']}{config['suffix']}
                    </div>
                    <div class="metric-delta {delta_class}">
                        {delta_symbol}{metric_data['delta']}
                    </div>
                </div>
            """, unsafe_allow_html=True)

def render_level_progress():
    """레벨 진행률 섹션"""
    user = st.session_state.user
    points = user.get('points', 0)
    level_info = get_user_level(points)
    
    current = level_info['current_info']
    next_info = level_info['next_info']
    progress = level_info['progress']
    
    st.markdown(f"""
        <div class="level-progress">
            <div class="level-icon">{current['icon']}</div>
            <div class="level-info">
                <div style="font-weight: 600; font-size: 1.1rem;">
                    {current['name']}
                </div>
                <div style="font-size: 0.875rem; opacity: 0.9;">
                    {points} 포인트
                    {f" / 다음 레벨까지 {next_info['min_points'] - points}점" if next_info else " (최고 레벨)"}
                </div>
                <div class="level-bar">
                    <div class="level-fill" style="width: {progress}%"></div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_activity_timeline():
    """활동 타임라인"""
    st.markdown("### 🕒 최근 활동")
    
    activities = load_recent_activities()
    
    if not activities:
        st.info("아직 활동 내역이 없습니다. 첫 프로젝트를 시작해보세요!")
        return
    
    for activity in activities[:5]:
        time_diff = datetime.now() - activity['created_at']
        
        if time_diff.days > 0:
            time_str = f"{time_diff.days}일 전"
        elif time_diff.seconds > 3600:
            time_str = f"{time_diff.seconds // 3600}시간 전"
        elif time_diff.seconds > 60:
            time_str = f"{time_diff.seconds // 60}분 전"
        else:
            time_str = "방금 전"
        
        st.markdown(f"""
            <div class="activity-item">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 1.2rem; margin-right: 0.5rem;">
                            {activity['icon']}
                        </span>
                        <span style="font-weight: 500;">
                            {activity['text']}
                        </span>
                    </div>
                    <div style="color: #6b7280; font-size: 0.875rem;">
                        {time_str}
                    </div>
                </div>
                {f"<div style='color: #6b7280; font-size: 0.875rem; margin-top: 0.5rem; margin-left: 2rem;'>{activity['details'].get('description', '')}</div>" if activity['details'].get('description') else ""}
            </div>
        """, unsafe_allow_html=True)
    
    if len(activities) > 5:
        if st.button("더 보기"):
            st.session_state.show_all_activities = True

def render_charts_section():
    """차트 섹션"""
    st.markdown("### 📈 데이터 분석")
    
    chart_data = load_chart_data()
    
    col1, col2 = st.columns(2)
    
    # 프로젝트 상태 분포
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### 프로젝트 상태")
        
        if chart_data['project_status']:
            df = pd.DataFrame(chart_data['project_status'])
            
            fig = go.Figure(data=[go.Pie(
                labels=df['status'].map({
                    'active': '진행중',
                    'completed': '완료',
                    'archived': '보관됨'
                }),
                values=df['count'],
                hole=0.3,
                marker_colors=['#10b981', '#3b82f6', '#6b7280']
            )])
            
            fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("아직 프로젝트가 없습니다.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 월별 실험 추이
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("#### 월별 실험 추이")
        
        if chart_data['monthly_experiments']:
            df = pd.DataFrame(chart_data['monthly_experiments'])
            
            fig = go.Figure()
            
            # 전체 실험
            fig.add_trace(go.Scatter(
                x=df['month'],
                y=df['count'],
                mode='lines+markers',
                name='전체',
                line=dict(color='#3b82f6', width=3),
                marker=dict(size=8)
            ))
            
            # 성공 실험
            fig.add_trace(go.Scatter(
                x=df['month'],
                y=df['success_count'],
                mode='lines+markers',
                name='성공',
                line=dict(color='#10b981', width=3),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis_title="월",
                yaxis_title="실험 수",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("아직 실험 데이터가 없습니다.")
        
        st.markdown('</div>', unsafe_allow_html=True)

def render_projects_section():
    """프로젝트 섹션"""
    st.markdown("### 📁 내 프로젝트")
    
    # 탭
    tab1, tab2, tab3 = st.tabs(["진행중", "완료", "전체"])
    
    with tab1:
        projects = load_projects_data(status='active')
        render_project_list(projects)
    
    with tab2:
        projects = load_projects_data(status='completed')
        render_project_list(projects)
    
    with tab3:
        projects = load_projects_data()
        render_project_list(projects)

def render_project_list(projects: List[Dict[str, Any]]):
    """프로젝트 목록 렌더링"""
    if not projects:
        st.info("프로젝트가 없습니다. 새 프로젝트를 시작해보세요!")
        if st.button("➕ 새 프로젝트 만들기"):
            st.switch_page("pages/2_📝_Project_Setup.py")
        return
    
    for project in projects[:5]:
        # 진행률 계산
        total_experiments = project.get('experiment_count', 0)
        completed_experiments = project.get('completed_count', 0)
        progress = (completed_experiments / total_experiments * 100) if total_experiments > 0 else 0
        
        # 상태 색상
        status_colors = {
            'active': '#10b981',
            'completed': '#3b82f6',
            'archived': '#6b7280'
        }
        
        st.markdown(f"""
            <div class="project-card">
                <div style="display: flex; justify-content: space-between; align-items: start;">
                    <div style="flex: 1;">
                        <h4 style="margin: 0; color: #1f2937;">
                            {project['name']}
                        </h4>
                        <p style="color: #6b7280; font-size: 0.875rem; margin: 0.5rem 0;">
                            {project.get('description', '설명 없음')}
                        </p>
                        <div style="display: flex; gap: 1rem; margin-top: 0.5rem;">
                            <span style="font-size: 0.875rem; color: #6b7280;">
                                🧪 실험 {total_experiments}개
                            </span>
                            <span style="font-size: 0.875rem; color: #6b7280;">
                                👥 협업자 {project.get('collaborator_count', 0)}명
                            </span>
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <span style="
                            display: inline-block;
                            padding: 0.25rem 0.75rem;
                            background: {status_colors.get(project['status'], '#6b7280')}20;
                            color: {status_colors.get(project['status'], '#6b7280')};
                            border-radius: 20px;
                            font-size: 0.75rem;
                            font-weight: 500;
                        ">
                            {{'active': '진행중', 'completed': '완료', 'archived': '보관됨'}.get(project['status'], project['status'])}
                        </span>
                    </div>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {progress}%"></div>
                </div>
                <div style="text-align: right; font-size: 0.75rem; color: #6b7280; margin-top: 0.25rem;">
                    진행률 {progress:.0f}%
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # 프로젝트 선택
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("열기", key=f"open_{project['id']}"):
                st.session_state.selected_project = project
                st.switch_page("pages/3_🧪_Experiment_Design.py")

def render_ai_recommendations():
    """AI 추천 섹션"""
    st.markdown("### 🤖 AI 추천")
    
    # AI 설명 모드 확인
    show_details = st.session_state.get('show_ai_details', False)
    
    # 추천 카드
    recommendations = [
        {
            'title': '다음 실험 제안',
            'icon': '🔬',
            'content': '최근 PLA 복합재료 실험 결과를 바탕으로, 강화제 함량을 5% 증가시킨 추가 실험을 권장합니다.',
            'reasoning': '이전 실험에서 강화제 3%일 때 인장강도가 15% 향상되었으며, 선형 관계를 고려할 때 5%에서 더 좋은 결과가 예상됩니다.'
        },
        {
            'title': '데이터 분석 인사이트',
            'icon': '📊',
            'content': '지난 달 실험 성공률이 85%로 평균보다 10% 높습니다. 현재 실험 설계 방법을 계속 유지하세요.',
            'reasoning': '통계적 분석 결과, 현재 사용 중인 Central Composite Design이 귀하의 실험 목적에 최적화되어 있습니다.'
        },
        {
            'title': '협업 기회',
            'icon': '🤝',
            'content': '비슷한 연구를 진행 중인 김박사님과의 협업을 추천합니다. 시너지 효과가 기대됩니다.',
            'reasoning': '두 분의 연구 키워드 매칭률이 78%이며, 상호 보완적인 전문 분야를 보유하고 있습니다.'
        }
    ]
    
    for rec in recommendations:
        with st.container():
            col1, col2 = st.columns([10, 1])
            
            with col1:
                st.markdown(f"""
                    <div style="
                        background: #f8f9fa;
                        padding: 1rem;
                        border-radius: 8px;
                        border-left: 3px solid #667eea;
                        margin-bottom: 0.5rem;
                    ">
                        <div style="font-weight: 600; margin-bottom: 0.5rem;">
                            {rec['icon']} {rec['title']}
                        </div>
                        <div style="color: #4b5563;">
                            {rec['content']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if st.button("🔍", key=f"detail_{rec['title']}", help="상세 설명"):
                    st.session_state[f"show_detail_{rec['title']}"] = not st.session_state.get(f"show_detail_{rec['title']}", False)
            
            # 상세 설명 (토글)
            if show_details or st.session_state.get(f"show_detail_{rec['title']}", False):
                st.markdown(f"""
                    <div style="
                        background: #e5e7eb;
                        padding: 1rem;
                        border-radius: 8px;
                        margin-bottom: 1rem;
                        font-size: 0.875rem;
                        color: #374151;
                    ">
                        <strong>🤔 AI 추론 과정:</strong><br>
                        {rec['reasoning']}
                    </div>
                """, unsafe_allow_html=True)

def render_quick_actions():
    """빠른 실행 버튼"""
    st.markdown("### ⚡ 빠른 실행")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🆕 새 프로젝트", use_container_width=True):
            st.switch_page("pages/2_📝_Project_Setup.py")
    
    with col2:
        if st.button("🧪 실험 설계", use_container_width=True):
            st.switch_page("pages/3_🧪_Experiment_Design.py")
    
    with col3:
        if st.button("📊 데이터 분석", use_container_width=True):
            st.switch_page("pages/4_📈_Data_Analysis.py")
    
    with col4:
        if st.button("🔍 문헌 검색", use_container_width=True):
            st.switch_page("pages/5_🔍_Literature_Search.py")

# =============================================================================
# 🎯 메인 함수
# =============================================================================

def main():
    """메인 함수"""
    # 초기화
    init_session_state()
    
    # 헤더
    render_header()
    
    # 게스트 모드 안내
    if is_guest:
        st.info("👋 게스트 모드로 둘러보는 중입니다. 일부 기능이 제한될 수 있습니다.")
    
    # 메인 레이아웃
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 메트릭 카드
        render_metrics_section()
        
        # 차트 섹션
        render_charts_section()
        
        # 프로젝트 섹션
        render_projects_section()
    
    with col2:
        # 레벨 진행률
        render_level_progress()
        
        # 활동 타임라인
        render_activity_timeline()
        
        # AI 추천
        render_ai_recommendations()
    
    # 빠른 실행
    st.divider()
    render_quick_actions()
    
    # 사이드바 - AI 설명 모드
    with st.sidebar:
        st.divider()
        st.markdown("### ⚙️ 대시보드 설정")
        
        # AI 설명 모드
        show_details = st.checkbox(
            "🤖 AI 상세 설명 표시",
            value=st.session_state.get('show_ai_details', False),
            help="AI의 추천 이유와 분석 과정을 자세히 볼 수 있습니다"
        )
        st.session_state.show_ai_details = show_details
        
        # 자동 새로고침
        auto_refresh = st.checkbox(
            "🔄 자동 새로고침",
            value=False,
            help="5분마다 대시보드를 자동으로 새로고침합니다"
        )
        
        if auto_refresh:
            time_passed = (datetime.now() - st.session_state.last_refresh).seconds
            if time_passed >= REFRESH_INTERVAL:
                st.rerun()
        
        # 동기화 설정
        st.divider()
        st.markdown("### 🔄 동기화 설정")
        
        sync_enabled = st.checkbox(
            "자동 동기화 활성화",
            value=True,
            help="온라인 상태에서 자동으로 데이터를 동기화합니다"
        )
        
        if st.button("지금 동기화", use_container_width=True):
            with st.spinner("동기화 중..."):
                time.sleep(2)  # 실제로는 동기화 로직 실행
                st.success("✅ 동기화 완료!")
        
        # 마지막 동기화 시간
        st.caption("마지막 동기화: 5분 전")

if __name__ == "__main__":
    main()
