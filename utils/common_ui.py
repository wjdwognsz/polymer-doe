"""
🎨 Universal DOE Platform - 공통 UI 컴포넌트
================================================================================
재사용 가능한 UI 컴포넌트 라이브러리
일관된 디자인 시스템, 테마 지원, 접근성, AI 투명성 원칙 구현
================================================================================
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple, Any, Union, Callable, Literal
import json
import base64
import time
from datetime import datetime, timedelta
from functools import wraps
import logging
from pathlib import Path
from PIL import Image
import io
import re

# 로컬 설정 임포트
try:
    from config.theme_config import COLORS, FONTS, LAYOUT, CUSTOM_CSS, apply_theme
    from config.app_config import (
        APP_INFO, UI_CONFIG, AI_EXPLANATION_CONFIG,
        FEATURE_FLAGS, get_config
    )
except ImportError:
    # 기본값 설정
    COLORS = {
        'primary': '#1E88E5',
        'secondary': '#43A047',
        'accent': '#E53935',
        'warning': '#FB8C00',
        'info': '#00ACC1',
        'success': '#43A047',
        'error': '#E53935',
        'background': '#FAFAFA',
        'surface': '#FFFFFF',
        'text_primary': '#212121',
        'text_secondary': '#757575'
    }
    UI_CONFIG = {'theme': {'default': 'light'}}
    AI_EXPLANATION_CONFIG = {
        'default_mode': 'auto',
        'detail_sections': {
            'reasoning': True,
            'alternatives': True,
            'background': True,
            'confidence': True,
            'limitations': True
        }
    }

# ===========================================================================
# 🔧 로깅 설정
# ===========================================================================

logger = logging.getLogger(__name__)

# ===========================================================================
# 📌 상수 정의
# ===========================================================================

# 아이콘 매핑
ICONS = {
    # 상태
    'success': '✅',
    'error': '❌',
    'warning': '⚠️',
    'info': 'ℹ️',
    'tip': '💡',
    'loading': '⏳',
    'complete': '✔️',
    
    # 사용자
    'user': '👤',
    'team': '👥',
    'admin': '👑',
    'guest': '👻',
    
    # 기능
    'project': '📁',
    'experiment': '🧪',
    'data': '📊',
    'analysis': '📈',
    'report': '📄',
    'settings': '⚙️',
    'notification': '🔔',
    'help': '❓',
    'search': '🔍',
    'filter': '🔽',
    
    # 액션
    'add': '➕',
    'edit': '✏️',
    'delete': '🗑️',
    'save': '💾',
    'download': '⬇️',
    'upload': '⬆️',
    'share': '🔗',
    'refresh': '🔄',
    'expand': '🔍',
    'collapse': '🔼'
}

# 버튼 스타일
BUTTON_STYLES = {
    'primary': {
        'bg_color': COLORS.get('primary', '#1E88E5'),
        'text_color': 'white',
        'hover_color': '#1565C0'
    },
    'secondary': {
        'bg_color': COLORS.get('light', '#F3F4F6'),
        'text_color': COLORS.get('text_primary', '#212121'),
        'hover_color': '#E5E7EB'
    },
    'success': {
        'bg_color': COLORS.get('success', '#43A047'),
        'text_color': 'white',
        'hover_color': '#388E3C'
    },
    'danger': {
        'bg_color': COLORS.get('error', '#E53935'),
        'text_color': 'white',
        'hover_color': '#C62828'
    }
}

# ===========================================================================
# 🎨 CSS 스타일
# ===========================================================================

COMPONENT_CSS = """
<style>
/* 애니메이션 */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}

@keyframes slideIn {
    from { transform: translateX(-100%); }
    to { transform: translateX(0); }
}

/* 컴포넌트 스타일 */
.custom-card {
    background: var(--surface-color, #FFFFFF);
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    transition: all 0.3s ease;
    animation: fadeIn 0.5s ease-out;
}

.custom-card:hover {
    box-shadow: 0 4px 16px rgba(0,0,0,0.12);
    transform: translateY(-2px);
}

.metric-card {
    text-align: center;
    padding: 1rem;
    border-radius: 8px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}

.ai-response-container {
    border-left: 4px solid var(--primary-color, #1E88E5);
    padding-left: 1rem;
    margin: 1rem 0;
}

.ai-detail-section {
    background: var(--background-color, #FAFAFA);
    border-radius: 8px;
    padding: 1rem;
    margin-top: 0.5rem;
}

.progress-step {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 40px;
    height: 40px;
    border-radius: 50%;
    font-weight: bold;
}

.progress-step.active {
    background: var(--primary-color, #1E88E5);
    color: white;
    animation: pulse 2s infinite;
}

.progress-step.completed {
    background: var(--success-color, #43A047);
    color: white;
}

.tooltip {
    position: relative;
    cursor: help;
}

.tooltip:hover::after {
    content: attr(data-tooltip);
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    background: #333;
    color: white;
    padding: 0.5rem;
    border-radius: 4px;
    font-size: 0.875rem;
    white-space: nowrap;
    z-index: 1000;
}

/* 반응형 디자인 */
@media (max-width: 768px) {
    .custom-card {
        padding: 1rem;
    }
    
    .hide-mobile {
        display: none;
    }
}

/* 접근성 */
.sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0,0,0,0);
    white-space: nowrap;
    border: 0;
}

/* 다크모드 지원 */
@media (prefers-color-scheme: dark) {
    .custom-card {
        background: #1F2937;
        color: #F9FAFB;
    }
}
</style>
"""

# ===========================================================================
# 🛠️ 유틸리티 함수
# ===========================================================================

def apply_custom_css():
    """커스텀 CSS 적용"""
    st.markdown(COMPONENT_CSS, unsafe_allow_html=True)
    if 'custom_css_applied' not in st.session_state:
        st.session_state.custom_css_applied = True

def get_user_level() -> str:
    """사용자 레벨 반환"""
    return st.session_state.get('user', {}).get('level', 'beginner')

def should_show_ai_details() -> bool:
    """AI 상세 설명 표시 여부 결정"""
    mode = st.session_state.get('show_ai_details', 'auto')
    
    if mode == 'always':
        return True
    elif mode == 'never':
        return False
    elif mode == 'auto':
        # 사용자 레벨에 따라 자동 결정
        user_level = get_user_level()
        return user_level in ['beginner', 'intermediate']
    
    return False

# ===========================================================================
# 🤖 AI 응답 컴포넌트 (핵심 기능)
# ===========================================================================

def render_ai_response(
    response: Dict[str, Any],
    response_type: str = "general",
    show_confidence: bool = True,
    allow_feedback: bool = True,
    key: Optional[str] = None
):
    """
    AI 응답 렌더링 (상세 설명 토글 포함)
    프로젝트 지침서의 AI 투명성 원칙 구현
    
    Args:
        response: AI 응답 딕셔너리
            - main: 핵심 답변 (필수)
            - reasoning: 추론 과정
            - alternatives: 대안
            - background: 이론적 배경
            - confidence: 신뢰도
            - limitations: 한계점
        response_type: 응답 유형
        show_confidence: 신뢰도 표시 여부
        allow_feedback: 피드백 허용 여부
        key: 고유 키
    """
    if not response:
        st.warning("AI 응답이 없습니다.")
        return
    
    # 고유 키 생성
    if not key:
        key = f"ai_response_{response_type}_{int(time.time())}"
    
    # AI 응답 컨테이너
    with st.container():
        st.markdown(f'<div class="ai-response-container">', unsafe_allow_html=True)
        
        # 헤더
        col1, col2, col3 = st.columns([6, 1, 1])
        
        with col1:
            st.markdown(f"### 🤖 {response_type}")
        
        with col2:
            # 신뢰도 표시
            if show_confidence and 'confidence' in response:
                confidence = response['confidence']
                if isinstance(confidence, (int, float)):
                    color = 'success' if confidence >= 80 else 'warning' if confidence >= 60 else 'error'
                    st.markdown(
                        f'<span style="color: {COLORS[color]}; font-weight: bold;">'
                        f'{confidence}%</span>',
                        unsafe_allow_html=True
                    )
        
        with col3:
            # 상세 설명 토글 버튼
            detail_key = f"{key}_show_details"
            if detail_key not in st.session_state:
                st.session_state[detail_key] = should_show_ai_details()
            
            if st.button(
                "🔍" if not st.session_state[detail_key] else "🔼",
                key=f"{key}_toggle",
                help="상세 설명 보기/숨기기"
            ):
                st.session_state[detail_key] = not st.session_state[detail_key]
                st.rerun()
        
        # 핵심 답변 (항상 표시)
        st.markdown("#### 💡 핵심 답변")
        st.write(response.get('main', '답변이 없습니다.'))
        
        # 상세 설명 (토글)
        if st.session_state.get(detail_key, False):
            st.markdown("---")
            
            # 탭으로 구성
            detail_tabs = []
            tab_contents = []
            
            if 'reasoning' in response:
                detail_tabs.append("🧠 추론 과정")
                tab_contents.append(response['reasoning'])
            
            if 'alternatives' in response:
                detail_tabs.append("🔄 대안")
                tab_contents.append(response['alternatives'])
            
            if 'background' in response:
                detail_tabs.append("📚 배경 지식")
                tab_contents.append(response['background'])
            
            if 'limitations' in response:
                detail_tabs.append("⚠️ 한계점")
                tab_contents.append(response['limitations'])
            
            if detail_tabs:
                tabs = st.tabs(detail_tabs)
                for i, (tab, content) in enumerate(zip(tabs, tab_contents)):
                    with tab:
                        st.markdown(f'<div class="ai-detail-section">', unsafe_allow_html=True)
                        if isinstance(content, list):
                            for item in content:
                                st.write(f"• {item}")
                        else:
                            st.write(content)
                        st.markdown('</div>', unsafe_allow_html=True)
        
        # 피드백 섹션
        if allow_feedback:
            st.markdown("---")
            col1, col2, col3 = st.columns([2, 1, 4])
            
            with col1:
                st.markdown("**이 답변이 도움이 되셨나요?**")
            
            with col2:
                feedback_col1, feedback_col2 = st.columns(2)
                with feedback_col1:
                    if st.button("👍", key=f"{key}_helpful"):
                        st.success("피드백 감사합니다!")
                
                with feedback_col2:
                    if st.button("👎", key=f"{key}_not_helpful"):
                        st.info("더 나은 답변을 위해 노력하겠습니다.")
        
        st.markdown('</div>', unsafe_allow_html=True)

# ===========================================================================
# 📊 레이아웃 컴포넌트
# ===========================================================================

def render_header(
    title: str,
    subtitle: Optional[str] = None,
    icon: Optional[str] = None,
    breadcrumb: Optional[List[str]] = None,
    actions: Optional[List[Dict[str, Any]]] = None
):
    """
    페이지 헤더 렌더링
    
    Args:
        title: 페이지 제목
        subtitle: 부제목
        icon: 아이콘
        breadcrumb: 경로 표시
        actions: 액션 버튼 리스트
    """
    # 브레드크럼
    if breadcrumb:
        st.markdown(
            " › ".join([f'<span style="color: {COLORS["text_secondary"]};">{item}</span>' 
                        for item in breadcrumb]),
            unsafe_allow_html=True
        )
    
    # 헤더
    col1, col2 = st.columns([4, 1])
    
    with col1:
        if icon:
            st.markdown(f"# {icon} {title}")
        else:
            st.markdown(f"# {title}")
        
        if subtitle:
            st.markdown(f'<p style="color: {COLORS["text_secondary"]}; '
                      f'margin-top: -1rem;">{subtitle}</p>',
                      unsafe_allow_html=True)
    
    with col2:
        if actions:
            for action in actions:
                if st.button(
                    action.get('label', ''),
                    key=action.get('key'),
                    type=action.get('type', 'secondary')
                ):
                    if 'callback' in action:
                        action['callback']()
    
    st.divider()

def render_footer():
    """푸터 렌더링"""
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Universal DOE Platform**")
        st.caption(f"Version {APP_INFO.get('version', '2.0.0')}")
    
    with col2:
        st.markdown("**지원**")
        st.caption("📧 support@universaldoe.com")
        st.caption("📚 [문서](https://docs.universaldoe.com)")
    
    with col3:
        st.markdown("**커뮤니티**")
        st.caption("💬 [Discord](https://discord.gg/universaldoe)")
        st.caption("🐦 [Twitter](https://twitter.com/universaldoe)")
    
    st.markdown(
        f'<div style="text-align: center; margin-top: 2rem; '
        f'color: {COLORS["text_secondary"]};">'
        f'<p>Made with ❤️ by DOE Team © 2024</p></div>',
        unsafe_allow_html=True
    )

# ===========================================================================
# 📈 메트릭 및 통계 컴포넌트
# ===========================================================================

def render_metric_card(
    label: str,
    value: Any,
    delta: Optional[Any] = None,
    delta_color: Literal["normal", "inverse", "off"] = "normal",
    help_text: Optional[str] = None,
    icon: Optional[str] = None
):
    """향상된 메트릭 카드"""
    col1, col2 = st.columns([1, 4])
    
    with col1:
        if icon:
            st.markdown(
                f'<div style="font-size: 2rem; text-align: center;">{icon}</div>',
                unsafe_allow_html=True
            )
    
    with col2:
        st.metric(
            label=label,
            value=value,
            delta=delta,
            delta_color=delta_color,
            help=help_text
        )

def render_metric_cards(
    metrics: List[Dict[str, Any]],
    columns: int = 3
):
    """여러 메트릭 카드를 그리드로 표시"""
    cols = st.columns(columns)
    
    for i, metric in enumerate(metrics):
        with cols[i % columns]:
            render_metric_card(**metric)

# ===========================================================================
# 💬 알림 및 메시지 컴포넌트
# ===========================================================================

def show_notification(
    message: str,
    type: Literal["success", "error", "warning", "info"] = "info",
    icon: bool = True,
    duration: Optional[int] = None
):
    """알림 메시지 표시"""
    icon_str = ICONS.get(type, "") if icon else ""
    
    if type == "success":
        st.success(f"{icon_str} {message}")
    elif type == "error":
        st.error(f"{icon_str} {message}")
    elif type == "warning":
        st.warning(f"{icon_str} {message}")
    else:
        st.info(f"{icon_str} {message}")
    
    # 자동 숨김 (duration이 설정된 경우)
    if duration:
        time.sleep(duration)
        st.empty()

def render_empty_state(
    message: str,
    icon: str = "📭",
    action_label: Optional[str] = None,
    action_callback: Optional[Callable] = None
):
    """빈 상태 UI"""
    st.markdown(
        f"""
        <div style="text-align: center; padding: 3rem; color: {COLORS['text_secondary']};">
            <div style="font-size: 4rem; margin-bottom: 1rem;">{icon}</div>
            <h3>{message}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if action_label and action_callback:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button(action_label, type="primary", use_container_width=True):
                action_callback()

# ===========================================================================
# 🔄 진행 상태 컴포넌트
# ===========================================================================

def render_progress_bar(
    current: int,
    total: int,
    label: str = "진행률",
    show_percentage: bool = True,
    color: Optional[str] = None
):
    """진행률 바"""
    progress = current / total if total > 0 else 0
    percentage = int(progress * 100)
    
    if show_percentage:
        text = f"{label}: {current}/{total} ({percentage}%)"
    else:
        text = f"{label}: {current}/{total}"
    
    st.progress(progress, text=text)

def render_step_progress(
    steps: List[str],
    current_step: int,
    completed_steps: Optional[List[int]] = None
):
    """단계별 진행 상황 표시"""
    if completed_steps is None:
        completed_steps = list(range(current_step))
    
    cols = st.columns(len(steps))
    
    for i, (col, step) in enumerate(zip(cols, steps)):
        with col:
            # 상태 결정
            if i in completed_steps:
                status = "completed"
                icon = "✔️"
            elif i == current_step:
                status = "active"
                icon = str(i + 1)
            else:
                status = "pending"
                icon = str(i + 1)
            
            # 스타일 적용
            if status == "completed":
                color = COLORS['success']
            elif status == "active":
                color = COLORS['primary']
            else:
                color = COLORS['light']
            
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <div class="progress-step {status}" 
                         style="background: {color}; color: white; 
                                margin: 0 auto;">
                        {icon}
                    </div>
                    <p style="margin-top: 0.5rem; font-size: 0.875rem;">
                        {step}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

# ===========================================================================
# 🗂️ 탭 및 네비게이션 컴포넌트
# ===========================================================================

def render_tabs_with_icons(
    tabs: List[Dict[str, Any]],
    key: str = "tabs"
) -> int:
    """
    아이콘이 있는 탭 렌더링
    
    Args:
        tabs: [{"label": "탭1", "icon": "📊"}, ...]
        key: 고유 키
    
    Returns:
        선택된 탭 인덱스
    """
    tab_labels = [f"{tab.get('icon', '')} {tab['label']}" for tab in tabs]
    selected_tab = st.tabs(tab_labels)
    
    return selected_tab

# ===========================================================================
# 📁 파일 관련 컴포넌트
# ===========================================================================

def render_file_uploader(
    label: str,
    file_types: List[str],
    accept_multiple: bool = False,
    help_text: Optional[str] = None,
    key: Optional[str] = None
) -> Union[Any, List[Any], None]:
    """향상된 파일 업로더"""
    uploaded_files = st.file_uploader(
        label,
        type=file_types,
        accept_multiple_files=accept_multiple,
        help=help_text,
        key=key
    )
    
    if uploaded_files:
        if accept_multiple:
            for file in uploaded_files:
                st.caption(f"📎 {file.name} ({file.size:,} bytes)")
        else:
            st.caption(f"📎 {uploaded_files.name} ({uploaded_files.size:,} bytes)")
    
    return uploaded_files

def render_download_button(
    data: Any,
    filename: str,
    label: str = "다운로드",
    mime: str = "application/octet-stream",
    help_text: Optional[str] = None,
    key: Optional[str] = None
):
    """다운로드 버튼"""
    st.download_button(
        label=f"{ICONS['download']} {label}",
        data=data,
        file_name=filename,
        mime=mime,
        help=help_text,
        key=key
    )

# ===========================================================================
# 🔍 필터 및 검색 컴포넌트
# ===========================================================================

def render_search_bar(
    placeholder: str = "검색...",
    key: str = "search",
    on_change: Optional[Callable] = None
) -> str:
    """검색 바"""
    search_value = st.text_input(
        label="검색",
        placeholder=placeholder,
        key=key,
        on_change=on_change,
        label_visibility="collapsed"
    )
    
    return search_value

def render_filter_panel(
    filters: Dict[str, Dict[str, Any]],
    key_prefix: str = "filter"
) -> Dict[str, Any]:
    """필터 패널"""
    filter_values = {}
    
    with st.expander("🔽 필터", expanded=True):
        for filter_key, config in filters.items():
            if config['type'] == 'select':
                filter_values[filter_key] = st.selectbox(
                    config['label'],
                    config['options'],
                    index=config.get('default', 0),
                    key=f"{key_prefix}_{filter_key}"
                )
            elif config['type'] == 'multiselect':
                filter_values[filter_key] = st.multiselect(
                    config['label'],
                    config['options'],
                    default=config.get('default', []),
                    key=f"{key_prefix}_{filter_key}"
                )
            elif config['type'] == 'slider':
                filter_values[filter_key] = st.slider(
                    config['label'],
                    min_value=config['min'],
                    max_value=config['max'],
                    value=config.get('default', config['min']),
                    key=f"{key_prefix}_{filter_key}"
                )
            elif config['type'] == 'date':
                filter_values[filter_key] = st.date_input(
                    config['label'],
                    value=config.get('default', datetime.now()),
                    key=f"{key_prefix}_{filter_key}"
                )
    
    return filter_values

# ===========================================================================
# 📊 데이터 테이블 컴포넌트
# ===========================================================================

def render_data_table(
    df: pd.DataFrame,
    editable: bool = False,
    height: int = 400,
    selection_mode: Optional[str] = None,
    key: Optional[str] = None
) -> Any:
    """데이터 테이블 렌더링"""
    return st.data_editor(
        df,
        use_container_width=True,
        height=height,
        disabled=not editable,
        key=key
    )

# ===========================================================================
# 🎨 차트 래퍼 컴포넌트
# ===========================================================================

def create_plotly_chart(
    chart_type: str,
    data: pd.DataFrame,
    x: str,
    y: str,
    **kwargs
) -> go.Figure:
    """Plotly 차트 생성 헬퍼"""
    # 테마 색상 적용
    color_sequence = [
        COLORS['primary'],
        COLORS['secondary'],
        COLORS['accent'],
        COLORS['success'],
        COLORS['warning']
    ]
    
    # 차트 타입별 생성
    if chart_type == 'line':
        fig = px.line(data, x=x, y=y, color_discrete_sequence=color_sequence, **kwargs)
    elif chart_type == 'bar':
        fig = px.bar(data, x=x, y=y, color_discrete_sequence=color_sequence, **kwargs)
    elif chart_type == 'scatter':
        fig = px.scatter(data, x=x, y=y, color_discrete_sequence=color_sequence, **kwargs)
    elif chart_type == 'box':
        fig = px.box(data, x=x, y=y, color_discrete_sequence=color_sequence, **kwargs)
    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")
    
    # 레이아웃 업데이트
    fig.update_layout(
        plot_bgcolor=COLORS.get('surface', 'white'),
        paper_bgcolor=COLORS.get('background', '#FAFAFA'),
        font=dict(color=COLORS.get('text_primary', '#212121'))
    )
    
    return fig

# ===========================================================================
# 🔐 권한 및 접근 제어 컴포넌트
# ===========================================================================

def check_permission(
    required_level: str = "user",
    show_message: bool = True
) -> bool:
    """권한 체크"""
    user_level = st.session_state.get('user', {}).get('level', 'guest')
    
    level_hierarchy = {
        'guest': 0,
        'user': 1,
        'premium': 2,
        'admin': 3
    }
    
    has_permission = level_hierarchy.get(user_level, 0) >= level_hierarchy.get(required_level, 1)
    
    if not has_permission and show_message:
        st.warning(
            f"🔒 이 기능을 사용하려면 {required_level} 레벨 이상이 필요합니다. "
            f"현재 레벨: {user_level}"
        )
    
    return has_permission

# ===========================================================================
# 🛠️ 유틸리티 컴포넌트
# ===========================================================================

def render_loading_spinner(
    message: str = "로딩 중...",
    spinner_type: str = "default"
):
    """로딩 스피너"""
    with st.spinner(message):
        # 실제 작업은 호출하는 곳에서 처리
        pass

def render_confirmation_dialog(
    message: str,
    confirm_label: str = "확인",
    cancel_label: str = "취소",
    key: str = "confirm"
) -> bool:
    """확인 대화상자"""
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.warning(message)
    
    with col2:
        confirm = st.button(confirm_label, key=f"{key}_yes", type="primary")
    
    with col3:
        cancel = st.button(cancel_label, key=f"{key}_no")
    
    return confirm and not cancel

def render_info_card(
    title: str,
    content: str,
    type: Literal["info", "success", "warning", "error"] = "info",
    icon: bool = True
):
    """정보 카드"""
    color = COLORS.get(type, COLORS['info'])
    icon_str = ICONS.get(type, "") if icon else ""
    
    st.markdown(
        f"""
        <div style="background-color: {color}20; 
                    border-left: 4px solid {color};
                    padding: 1rem; border-radius: 0.5rem;
                    margin: 1rem 0;">
            <h4>{icon_str} {title}</h4>
            <p>{content}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ===========================================================================
# 🎯 특수 용도 컴포넌트
# ===========================================================================

def render_experiment_card(
    experiment: Dict[str, Any],
    show_actions: bool = True,
    key: Optional[str] = None
):
    """실험 카드 (여러 페이지에서 재사용)"""
    with st.container():
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            st.subheader(experiment.get('name', 'Unnamed'))
            st.caption(f"생성일: {experiment.get('created_at', 'Unknown')}")
            st.write(experiment.get('description', ''))
        
        with col2:
            metrics = [
                {"label": "실험 수", "value": experiment.get('n_runs', 0)},
                {"label": "진행률", "value": f"{experiment.get('progress', 0)}%"}
            ]
            render_metric_cards(metrics, columns=2)
        
        with col3:
            if show_actions:
                if st.button("상세", key=f"view_{experiment.get('id', '')}_{key}"):
                    st.session_state.selected_experiment = experiment['id']
                    st.switch_page("pages/3_🧪_Experiment_Design.py")
        
        st.markdown('</div>', unsafe_allow_html=True)

def render_module_card(
    module: Dict[str, Any],
    show_install: bool = True,
    key: Optional[str] = None
):
    """모듈 카드 (마켓플레이스용)"""
    with st.container():
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        
        # 헤더
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.markdown(f"### {module.get('icon', '📦')} {module.get('name', 'Unknown')}")
            st.caption(f"v{module.get('version', '1.0.0')} by {module.get('author', 'Unknown')}")
        
        with col2:
            if module.get('verified', False):
                st.markdown("✅ **검증됨**")
        
        # 설명
        st.write(module.get('description', ''))
        
        # 메타 정보
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("다운로드", f"{module.get('downloads', 0):,}")
        
        with col2:
            rating = module.get('rating', 0)
            st.metric("평점", f"⭐ {rating:.1f}")
        
        with col3:
            st.metric("카테고리", module.get('category', 'General'))
        
        with col4:
            if show_install:
                if st.button("설치", key=f"install_{module.get('id', '')}_{key}"):
                    st.success("모듈이 설치되었습니다!")
        
        st.markdown('</div>', unsafe_allow_html=True)

# ===========================================================================
# 🎬 애니메이션 컴포넌트
# ===========================================================================

def render_animated_counter(
    target: int,
    duration: float = 1.0,
    prefix: str = "",
    suffix: str = "",
    key: str = "counter"
):
    """애니메이션 카운터"""
    placeholder = st.empty()
    
    steps = int(duration * 30)  # 30 FPS
    increment = target / steps
    
    current = 0
    for _ in range(steps):
        current = min(current + increment, target)
        placeholder.metric(
            label="",
            value=f"{prefix}{int(current):,}{suffix}",
            label_visibility="collapsed"
        )
        time.sleep(duration / steps)
    
    # 최종 값 표시
    placeholder.metric(
        label="",
        value=f"{prefix}{target:,}{suffix}",
        label_visibility="collapsed"
    )

# ===========================================================================
# 🌐 다국어 지원 준비
# ===========================================================================

def get_text(key: str, language: Optional[str] = None) -> str:
    """다국어 텍스트 반환 (향후 구현)"""
    # 현재는 한국어만 지원
    texts = {
        'welcome': '환영합니다',
        'login': '로그인',
        'logout': '로그아웃',
        'save': '저장',
        'cancel': '취소',
        'delete': '삭제',
        'confirm': '확인',
        'loading': '로딩 중...',
        'error': '오류가 발생했습니다',
        'success': '성공적으로 완료되었습니다'
    }
    
    return texts.get(key, key)

# ===========================================================================
# 🔧 초기화 및 설정
# ===========================================================================

def initialize_ui():
    """UI 시스템 초기화"""
    # CSS 적용
    apply_custom_css()
    
    # 세션 상태 초기화
    if 'ui_initialized' not in st.session_state:
        st.session_state.ui_initialized = True
        st.session_state.show_ai_details = 'auto'
        st.session_state.theme = 'light'
        st.session_state.animations_enabled = True

# ===========================================================================
# 📤 내보내기
# ===========================================================================

# 자동 초기화
initialize_ui()

# 싱글톤 인스턴스 (레거시 호환성)
class CommonUI:
    """레거시 호환성을 위한 클래스 래퍼"""
    
    @staticmethod
    def render_header(*args, **kwargs):
        return render_header(*args, **kwargs)
    
    @staticmethod
    def render_footer(*args, **kwargs):
        return render_footer(*args, **kwargs)
    
    @staticmethod
    def render_ai_response(*args, **kwargs):
        return render_ai_response(*args, **kwargs)
    
    @staticmethod
    def show_notification(*args, **kwargs):
        return show_notification(*args, **kwargs)
    
    @staticmethod
    def render_metric_card(*args, **kwargs):
        return render_metric_card(*args, **kwargs)
    
    @staticmethod
    def render_metric_cards(*args, **kwargs):
        return render_metric_cards(*args, **kwargs)

# 편의 함수
def get_common_ui() -> CommonUI:
    """CommonUI 인스턴스 반환 (레거시 호환)"""
    return CommonUI()

# 디버깅용
if __name__ == "__main__":
    st.set_page_config(page_title="Common UI Test", layout="wide")
    
    # 테스트 페이지
    render_header(
        "Common UI 컴포넌트 테스트",
        "모든 UI 컴포넌트 데모",
        "🎨",
        breadcrumb=["홈", "개발", "UI 테스트"]
    )
    
    # AI 응답 테스트
    st.subheader("AI 응답 컴포넌트")
    test_response = {
        'main': '이것은 AI의 핵심 답변입니다. 실험 설계에는 3개의 요인이 필요합니다.',
        'reasoning': '이러한 결론에 도달한 이유는 다음과 같습니다:\n1. 통계적 유의성\n2. 실험의 효율성\n3. 비용 대비 효과',
        'alternatives': ['2요인 설계도 가능하지만 정보가 제한적', '4요인은 너무 복잡할 수 있음'],
        'background': '실험계획법(DOE)은 1920년대 R.A. Fisher에 의해 개발되었습니다.',
        'confidence': 85,
        'limitations': '이 추천은 선형 관계를 가정합니다.'
    }
    
    render_ai_response(test_response, "실험 설계 추천")
    
    # 기타 컴포넌트 테스트
    st.subheader("메트릭 카드")
    metrics = [
        {"label": "프로젝트", "value": 42, "delta": 5, "icon": "📁"},
        {"label": "실험", "value": 128, "delta": -3, "icon": "🧪"},
        {"label": "성공률", "value": "94.5%", "icon": "📊"}
    ]
    render_metric_cards(metrics)
    
    render_footer()
