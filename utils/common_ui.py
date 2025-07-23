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
    'share': '📤',
    'copy': '📋',
    'refresh': '🔄',
    'sync': '🔄',
    
    # AI
    'ai': '🤖',
    'detail': '🔍',
    'simple': '📝',
    'reasoning': '🧠',
    'alternative': '🔀',
    'confidence': '📊',
    'limitation': '⚠️',
    
    # 상태
    'online': '🟢',
    'offline': '🔴',
    'syncing': '🔄',
    'local': '💾',
    'cloud': '☁️'
}

# 애니메이션 CSS
ANIMATIONS_CSS = """
<style>
/* 페이드 인 애니메이션 */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* 펄스 애니메이션 */
@keyframes pulse {
    0% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.05); opacity: 0.8; }
    100% { transform: scale(1); opacity: 1; }
}

/* 슬라이드 인 애니메이션 */
@keyframes slideIn {
    from { transform: translateX(-100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}

/* 회전 애니메이션 */
@keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

/* 애니메이션 클래스 */
.animate-fadeIn { animation: fadeIn 0.5s ease-out; }
.animate-pulse { animation: pulse 2s infinite; }
.animate-slideIn { animation: slideIn 0.3s ease-out; }
.animate-spin { animation: spin 1s linear infinite; }

/* 호버 효과 */
.hover-scale { transition: transform 0.2s; cursor: pointer; }
.hover-scale:hover { transform: scale(1.05); }

/* 그림자 효과 */
.shadow-sm { box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05); }
.shadow-md { box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
.shadow-lg { box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); }

/* 카드 스타일 */
.custom-card {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    border: 1px solid #e5e7eb;
    transition: all 0.3s ease;
}

.custom-card:hover {
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    transform: translateY(-2px);
}

/* AI 응답 스타일 */
.ai-response-container {
    background: linear-gradient(135deg, #f3e7ff 0%, #e7f3ff 100%);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
}

.ai-detail-section {
    background: rgba(255, 255, 255, 0.8);
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    border-left: 3px solid #7c3aed;
}

/* 토글 버튼 스타일 */
.toggle-button {
    background: #f3f4f6;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    cursor: pointer;
    transition: all 0.2s;
}

.toggle-button:hover {
    background: #e5e7eb;
    transform: translateY(-1px);
}

/* 오프라인 배지 */
.offline-badge {
    position: fixed;
    top: 10px;
    right: 10px;
    background: #ef4444;
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-size: 0.875rem;
    z-index: 1000;
    animation: pulse 2s infinite;
}
</style>
"""

# ===========================================================================
# 🎨 페이지 설정 함수
# ===========================================================================

def setup_page_config(
    page_title: Optional[str] = None,
    page_icon: str = "🧬",
    layout: Literal["centered", "wide"] = "wide",
    initial_sidebar_state: Literal["auto", "expanded", "collapsed"] = "expanded"
):
    """
    Streamlit 페이지 설정
    
    Args:
        page_title: 페이지 제목
        page_icon: 페이지 아이콘
        layout: 레이아웃 (centered/wide)
        initial_sidebar_state: 사이드바 초기 상태
    """
    st.set_page_config(
        page_title=page_title or APP_INFO['name'],
        page_icon=page_icon,
        layout=layout,
        initial_sidebar_state=initial_sidebar_state,
        menu_items={
            'Get Help': APP_INFO.get('github', '#'),
            'Report a bug': f"{APP_INFO.get('github', '#')}/issues",
            'About': APP_INFO.get('description', '')
        }
    )
    
    # 커스텀 CSS 적용
    st.markdown(ANIMATIONS_CSS, unsafe_allow_html=True)
    if 'CUSTOM_CSS' in globals():
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # 테마 적용
    if 'apply_theme' in globals():
        apply_theme()

# ===========================================================================
# 🎯 헤더 컴포넌트
# ===========================================================================

def render_header(
    title: str,
    subtitle: Optional[str] = None,
    breadcrumb: Optional[List[Tuple[str, str]]] = None,
    show_user_info: bool = True,
    show_notifications: bool = True,
    actions: Optional[List[Dict[str, Any]]] = None
):
    """
    페이지 헤더 렌더링
    
    Args:
        title: 페이지 제목
        subtitle: 부제목
        breadcrumb: 브레드크럼 [(label, page), ...]
        show_user_info: 사용자 정보 표시 여부
        show_notifications: 알림 아이콘 표시 여부
        actions: 추가 액션 버튼 리스트
    """
    # 브레드크럼
    if breadcrumb:
        breadcrumb_html = " › ".join([
            f'<a href="#" onclick="return false;" style="color: {COLORS["primary"]};">{label}</a>'
            for label, _ in breadcrumb[:-1]
        ])
        if breadcrumb:
            breadcrumb_html += f' › <span style="color: {COLORS["text_secondary"]};">{breadcrumb[-1][0]}</span>'
        st.markdown(breadcrumb_html, unsafe_allow_html=True)
    
    # 헤더 컨테이너
    col1, col2, col3 = st.columns([6, 2, 2])
    
    with col1:
        if subtitle:
            st.markdown(f"# {title}\n{subtitle}")
        else:
            st.markdown(f"# {title}")
    
    with col2:
        if show_user_info and st.session_state.get('authenticated', False):
            user = st.session_state.get('user', {})
            render_user_badge(user)
    
    with col3:
        action_cols = st.columns(len(actions) + (1 if show_notifications else 0))
        
        # 알림 버튼
        if show_notifications:
            with action_cols[0]:
                notification_count = st.session_state.get('unread_notifications', 0)
                if st.button(
                    f"{ICONS['notification']} {notification_count if notification_count > 0 else ''}",
                    key="header_notifications",
                    help="알림"
                ):
                    st.session_state.show_notifications = True
        
        # 추가 액션 버튼
        if actions:
            for i, action in enumerate(actions):
                with action_cols[i + (1 if show_notifications else 0)]:
                    if st.button(
                        action.get('label', ''),
                        key=f"header_action_{i}",
                        help=action.get('help', '')
                    ):
                        if 'callback' in action:
                            action['callback']()
    
    st.divider()

# ===========================================================================
# 📊 메트릭 카드
# ===========================================================================

def render_metric_card(
    label: str,
    value: Union[str, int, float],
    delta: Optional[Union[str, int, float]] = None,
    delta_color: Literal["normal", "inverse", "off"] = "normal",
    help: Optional[str] = None,
    icon: Optional[str] = None,
    background: Optional[str] = None,
    animate: bool = True
):
    """
    메트릭 카드 표시
    
    Args:
        label: 메트릭 레이블
        value: 메트릭 값
        delta: 변화량
        delta_color: 델타 색상 모드
        help: 도움말 텍스트
        icon: 아이콘
        background: 배경색
        animate: 애니메이션 여부
    """
    animation_class = "animate-fadeIn" if animate else ""
    bg_style = f"background: {background};" if background else ""
    
    # 델타 색상 결정
    delta_colors = {
        'normal': COLORS['success'] if str(delta).startswith('+') else COLORS['error'],
        'inverse': COLORS['error'] if str(delta).startswith('+') else COLORS['success'],
        'off': COLORS['text_secondary']
    }
    
    html = f"""
    <div class="custom-card {animation_class}" style="{bg_style}">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="flex: 1;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    {f'<span style="font-size: 1.5rem;">{icon}</span>' if icon else ''}
                    <p style="color: {COLORS['text_secondary']}; margin: 0; font-size: 0.875rem;">
                        {label}
                    </p>
                </div>
                <h2 style="margin: 0.5rem 0; color: {COLORS['text_primary']};">
                    {value}
                </h2>
                {f'<p style="color: {delta_colors.get(delta_color, COLORS["text_secondary"])}; margin: 0; font-size: 0.875rem;">{delta}</p>' if delta else ''}
            </div>
            {f'<span title="{help}" style="cursor: help; color: {COLORS["text_secondary"]};">{ICONS["help"]}</span>' if help else ''}
        </div>
    </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)

# ===========================================================================
# 🤖 AI 응답 컴포넌트 (AI 투명성 원칙 구현)
# ===========================================================================

def render_ai_response(
    response: Dict[str, Any],
    response_type: str = "general",
    show_details_default: Optional[bool] = None,
    key: Optional[str] = None
):
    """
    AI 응답 표시 (상세 설명 토글 기능 포함)
    
    Args:
        response: AI 응답 딕셔너리
            - main: 핵심 답변 (필수)
            - reasoning: 추론 과정
            - alternatives: 대안
            - background: 이론적 배경
            - confidence: 신뢰도
            - limitations: 한계점
        response_type: 응답 유형
        show_details_default: 기본 상세 표시 여부
        key: 고유 키
    """
    # 세션 상태 초기화
    detail_key = f"ai_details_{key}" if key else "ai_details_global"
    if detail_key not in st.session_state:
        if show_details_default is not None:
            st.session_state[detail_key] = show_details_default
        else:
            # 사용자 레벨에 따른 기본값
            user_level = st.session_state.get('user', {}).get('level', 'beginner')
            default_mode = AI_EXPLANATION_CONFIG['auto_mode_rules'].get(
                user_level, 
                AI_EXPLANATION_CONFIG['default_mode']
            )
            st.session_state[detail_key] = default_mode in ['detailed', 'balanced']
    
    # AI 응답 컨테이너
    with st.container():
        st.markdown('<div class="ai-response-container">', unsafe_allow_html=True)
        
        # 헤더와 토글 버튼
        col1, col2 = st.columns([10, 2])
        
        with col1:
            st.markdown(f"### {ICONS['ai']} {response_type} AI 응답")
        
        with col2:
            # 상세 설명 토글 버튼
            button_label = f"{ICONS['simple']} 간단히" if st.session_state[detail_key] else f"{ICONS['detail']} 자세히"
            if st.button(
                button_label,
                key=f"{detail_key}_toggle",
                help=f"단축키: {AI_EXPLANATION_CONFIG.get('keyboard_shortcut', 'Ctrl+D')}"
            ):
                st.session_state[detail_key] = not st.session_state[detail_key]
                st.rerun()
        
        # 핵심 답변 (항상 표시)
        st.markdown("#### 💡 핵심 답변")
        st.write(response.get('main', ''))
        
        # 상세 설명 (토글 가능)
        if st.session_state[detail_key]:
            st.markdown("---")
            
            # 탭으로 구성된 상세 정보
            detail_tabs = []
            detail_contents = []
            
            if response.get('reasoning') and AI_EXPLANATION_CONFIG['detail_sections']['reasoning']:
                detail_tabs.append(f"{ICONS['reasoning']} 추론 과정")
                detail_contents.append(response['reasoning'])
            
            if response.get('alternatives') and AI_EXPLANATION_CONFIG['detail_sections']['alternatives']:
                detail_tabs.append(f"{ICONS['alternative']} 대안 검토")
                detail_contents.append(response['alternatives'])
            
            if response.get('background') and AI_EXPLANATION_CONFIG['detail_sections']['background']:
                detail_tabs.append(f"{ICONS['info']} 배경 지식")
                detail_contents.append(response['background'])
            
            if response.get('confidence') and AI_EXPLANATION_CONFIG['detail_sections']['confidence']:
                detail_tabs.append(f"{ICONS['confidence']} 신뢰도")
                detail_contents.append(response['confidence'])
            
            if response.get('limitations') and AI_EXPLANATION_CONFIG['detail_sections']['limitations']:
                detail_tabs.append(f"{ICONS['limitation']} 한계점")
                detail_contents.append(response['limitations'])
            
            if detail_tabs:
                tabs = st.tabs(detail_tabs)
                for i, (tab, content) in enumerate(zip(tabs, detail_contents)):
                    with tab:
                        st.markdown(f'<div class="ai-detail-section">{content}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# ===========================================================================
# 📢 알림 메시지
# ===========================================================================

def show_notification(
    message: str,
    type: Literal["success", "error", "warning", "info"] = "info",
    icon: Optional[str] = None,
    duration: Optional[int] = 5,
    position: Literal["top-right", "top-center", "bottom-right"] = "top-right"
):
    """
    토스트 알림 표시
    
    Args:
        message: 알림 메시지
        type: 알림 유형
        icon: 커스텀 아이콘
        duration: 표시 시간 (초)
        position: 표시 위치
    """
    icon = icon or ICONS.get(type, ICONS['info'])
    colors = {
        'success': COLORS['success'],
        'error': COLORS['error'],
        'warning': COLORS['warning'],
        'info': COLORS['info']
    }
    
    # Streamlit 기본 알림 사용
    if type == "success":
        st.success(f"{icon} {message}")
    elif type == "error":
        st.error(f"{icon} {message}")
    elif type == "warning":
        st.warning(f"{icon} {message}")
    else:
        st.info(f"{icon} {message}")
    
    # 자동 숨김 (선택적)
    if duration:
        time.sleep(duration)
        st.empty()

# 헬퍼 함수들
def show_success(message: str, **kwargs):
    """성공 메시지 표시"""
    show_notification(message, "success", **kwargs)

def show_error(message: str, **kwargs):
    """에러 메시지 표시"""
    show_notification(message, "error", **kwargs)

def show_warning(message: str, **kwargs):
    """경고 메시지 표시"""
    show_notification(message, "warning", **kwargs)

def show_info(message: str, **kwargs):
    """정보 메시지 표시"""
    show_notification(message, "info", **kwargs)

# ===========================================================================
# 📊 데이터 테이블
# ===========================================================================

def render_data_table(
    data: pd.DataFrame,
    key: Optional[str] = None,
    editable: bool = False,
    use_checkbox: bool = False,
    hide_index: bool = True,
    column_config: Optional[Dict[str, Any]] = None,
    disabled: Optional[List[str]] = None,
    on_change: Optional[Callable] = None,
    **kwargs
):
    """
    향상된 데이터 테이블
    
    Args:
        data: 표시할 데이터프레임
        key: 고유 키
        editable: 편집 가능 여부
        use_checkbox: 체크박스 사용 여부
        hide_index: 인덱스 숨김 여부
        column_config: 컬럼 설정
        disabled: 비활성화할 컬럼 리스트
        on_change: 변경 시 콜백
        **kwargs: 추가 파라미터
    """
    # 빈 데이터 처리
    if data.empty:
        render_empty_state(
            icon=ICONS['data'],
            title="데이터가 없습니다",
            description="표시할 데이터가 없습니다."
        )
        return None
    
    # 데이터 에디터 렌더링
    edited_data = st.data_editor(
        data,
        key=key,
        use_container_width=True,
        hide_index=hide_index,
        num_rows="dynamic" if editable else "fixed",
        disabled=False if editable else True,
        column_config=column_config,
        on_change=on_change,
        **kwargs
    )
    
    return edited_data

# ===========================================================================
# 🚀 진행률 표시
# ===========================================================================

def render_progress(
    value: float,
    max_value: float = 100.0,
    label: Optional[str] = None,
    format_str: str = "{value}/{max_value} ({percentage}%)",
    show_eta: bool = False,
    color: Optional[str] = None
):
    """
    진행률 바 표시
    
    Args:
        value: 현재 값
        max_value: 최대 값
        label: 레이블
        format_str: 표시 형식
        show_eta: 예상 시간 표시
        color: 진행률 바 색상
    """
    percentage = min(value / max_value, 1.0) if max_value > 0 else 0
    
    # 진행률 바
    progress_bar = st.progress(percentage)
    
    # 텍스트 표시
    if label:
        display_text = format_str.format(
            value=value,
            max_value=max_value,
            percentage=int(percentage * 100)
        )
        st.caption(f"{label}: {display_text}")
    
    return progress_bar

def render_circular_progress(
    value: float,
    max_value: float = 100.0,
    size: int = 120,
    thickness: int = 10,
    color: Optional[str] = None,
    label: Optional[str] = None
):
    """
    원형 진행률 표시
    
    Args:
        value: 현재 값
        max_value: 최대 값
        size: 크기 (픽셀)
        thickness: 두께
        color: 색상
        label: 레이블
    """
    percentage = min(value / max_value * 100, 100) if max_value > 0 else 0
    color = color or COLORS['primary']
    
    # SVG 원형 진행률
    radius = (size - thickness) / 2
    circumference = 2 * 3.14159 * radius
    stroke_dashoffset = circumference - (percentage / 100) * circumference
    
    svg = f"""
    <svg width="{size}" height="{size}" style="transform: rotate(-90deg);">
        <circle
            cx="{size/2}"
            cy="{size/2}"
            r="{radius}"
            stroke="{COLORS['background']}"
            stroke-width="{thickness}"
            fill="none"
        />
        <circle
            cx="{size/2}"
            cy="{size/2}"
            r="{radius}"
            stroke="{color}"
            stroke-width="{thickness}"
            fill="none"
            stroke-dasharray="{circumference}"
            stroke-dashoffset="{stroke_dashoffset}"
            style="transition: stroke-dashoffset 0.5s ease;"
        />
    </svg>
    <div style="
        position: relative;
        top: -{size}px;
        text-align: center;
        line-height: {size}px;
        font-size: 1.5rem;
        font-weight: bold;
    ">
        {int(percentage)}%
    </div>
    """
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            f'<div style="text-align: center;">{svg}{f"<p>{label}</p>" if label else ""}</div>',
            unsafe_allow_html=True
        )

# ===========================================================================
# 🗂️ 탭 컴포넌트
# ===========================================================================

def render_tabs(
    tabs: Dict[str, Callable],
    default_tab: Optional[str] = None,
    key: Optional[str] = None
):
    """
    탭 인터페이스 렌더링
    
    Args:
        tabs: {탭이름: 렌더링함수} 딕셔너리
        default_tab: 기본 선택 탭
        key: 고유 키
    """
    tab_names = list(tabs.keys())
    tab_objects = st.tabs(tab_names)
    
    for tab_obj, (tab_name, render_func) in zip(tab_objects, tabs.items()):
        with tab_obj:
            render_func()

# ===========================================================================
# 📁 파일 업로드
# ===========================================================================

def create_file_uploader(
    label: str = "파일 선택",
    accept: Optional[List[str]] = None,
    multiple: bool = False,
    max_size_mb: Optional[int] = None,
    show_preview: bool = True,
    key: Optional[str] = None
) -> Optional[Union[Any, List[Any]]]:
    """
    향상된 파일 업로더
    
    Args:
        label: 업로더 레이블
        accept: 허용 파일 확장자 리스트
        multiple: 다중 파일 허용
        max_size_mb: 최대 파일 크기 (MB)
        show_preview: 미리보기 표시
        key: 고유 키
        
    Returns:
        업로드된 파일 또는 파일 리스트
    """
    # 파일 타입 설명
    if accept:
        type_desc = {
            'csv': 'CSV 스프레드시트',
            'xlsx': 'Excel 파일',
            'json': 'JSON 데이터',
            'pdf': 'PDF 문서',
            'png': 'PNG 이미지',
            'jpg': 'JPEG 이미지',
            'jpeg': 'JPEG 이미지',
            'txt': '텍스트 파일',
            'py': 'Python 코드'
        }
        
        accepted_desc = ", ".join([
            type_desc.get(ext.replace('.', ''), ext.upper())
            for ext in accept
        ])
        help_text = f"지원 형식: {accepted_desc}"
        if max_size_mb:
            help_text += f" (최대 {max_size_mb}MB)"
    else:
        help_text = None
    
    # 파일 업로더
    uploaded = st.file_uploader(
        label,
        type=accept,
        accept_multiple_files=multiple,
        help=help_text,
        key=key
    )
    
    if uploaded:
        files = uploaded if multiple else [uploaded]
        
        for file in files:
            # 파일 크기 확인
            if max_size_mb:
                file_size_mb = file.size / (1024 * 1024)
                if file_size_mb > max_size_mb:
                    show_error(f"{file.name}: 파일 크기가 {max_size_mb}MB를 초과합니다. ({file_size_mb:.1f}MB)")
                    continue
            
            # 파일 정보 표시
            with st.expander(f"📎 {file.name}", expanded=show_preview):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.caption(f"타입: {file.type}")
                with col2:
                    st.caption(f"크기: {file.size / 1024:.1f} KB")
                with col3:
                    if st.button(f"{ICONS['delete']}", key=f"del_{file.name}", help="삭제"):
                        # 삭제 로직은 부모 컴포넌트에서 처리
                        pass
                
                # 미리보기
                if show_preview:
                    render_file_preview(file)
    
    return uploaded

def render_file_preview(file):
    """파일 미리보기 렌더링"""
    file_ext = Path(file.name).suffix.lower()
    
    try:
        if file_ext in ['.png', '.jpg', '.jpeg', '.gif']:
            st.image(file)
        elif file_ext == '.csv':
            df = pd.read_csv(file)
            st.dataframe(df.head(10))
            st.caption(f"총 {len(df)} 행")
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file)
            st.dataframe(df.head(10))
            st.caption(f"총 {len(df)} 행")
        elif file_ext == '.json':
            data = json.load(file)
            st.json(data)
        elif file_ext in ['.txt', '.md']:
            content = file.read().decode('utf-8')
            st.text_area("내용", content, height=200, disabled=True)
        else:
            st.info("미리보기를 지원하지 않는 파일 형식입니다.")
    except Exception as e:
        st.error(f"파일 미리보기 중 오류 발생: {str(e)}")

# ===========================================================================
# 🔤 입력 검증
# ===========================================================================

def create_validated_input(
    label: str,
    value: Any = "",
    type: Literal["text", "number", "email", "password", "url"] = "text",
    required: bool = False,
    validation_pattern: Optional[str] = None,
    validation_func: Optional[Callable] = None,
    error_message: str = "올바른 값을 입력하세요",
    help: Optional[str] = None,
    key: Optional[str] = None,
    **kwargs
) -> Optional[Any]:
    """
    검증 기능이 있는 입력 필드
    
    Args:
        label: 입력 필드 레이블
        value: 기본값
        type: 입력 타입
        required: 필수 여부
        validation_pattern: 정규식 패턴
        validation_func: 커스텀 검증 함수
        error_message: 검증 실패 메시지
        help: 도움말
        key: 고유 키
        **kwargs: 추가 파라미터
        
    Returns:
        검증된 입력값 또는 None
    """
    # 필수 표시
    display_label = f"{label} *" if required else label
    
    # 입력 컴포넌트
    if type == "text":
        input_value = st.text_input(display_label, value, help=help, key=key, **kwargs)
    elif type == "number":
        input_value = st.number_input(display_label, value=value, help=help, key=key, **kwargs)
    elif type == "email":
        input_value = st.text_input(display_label, value, help=help, key=key, **kwargs)
        # 이메일 패턴
        validation_pattern = validation_pattern or r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    elif type == "password":
        input_value = st.text_input(display_label, value, type="password", help=help, key=key, **kwargs)
    elif type == "url":
        input_value = st.text_input(display_label, value, help=help, key=key, **kwargs)
        # URL 패턴
        validation_pattern = validation_pattern or r'^https?://[^\s]+$'
    else:
        input_value = st.text_input(display_label, value, help=help, key=key, **kwargs)
    
    # 검증
    is_valid = True
    
    # 필수 필드 검증
    if required and not input_value:
        st.error(f"{label}은(는) 필수 입력 항목입니다.")
        is_valid = False
    
    # 패턴 검증
    elif validation_pattern and input_value:
        if not re.match(validation_pattern, str(input_value)):
            st.error(error_message)
            is_valid = False
    
    # 커스텀 검증
    elif validation_func and input_value:
        validation_result = validation_func(input_value)
        if validation_result is not True:
            st.error(validation_result if isinstance(validation_result, str) else error_message)
            is_valid = False
    
    return input_value if is_valid else None

# ===========================================================================
# 📅 날짜/시간 선택
# ===========================================================================

def create_date_range_picker(
    label: str = "기간 선택",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    max_days: Optional[int] = None,
    key: Optional[str] = None
) -> Optional[Tuple[datetime, datetime]]:
    """
    날짜 범위 선택기
    
    Args:
        label: 레이블
        start_date: 시작일 기본값
        end_date: 종료일 기본값
        max_days: 최대 선택 가능 일수
        key: 고유 키
        
    Returns:
        (시작일, 종료일) 튜플 또는 None
    """
    col1, col2 = st.columns(2)
    
    with col1:
        selected_start = st.date_input(
            "시작일",
            value=start_date or datetime.now().date(),
            key=f"{key}_start" if key else None
        )
    
    with col2:
        selected_end = st.date_input(
            "종료일",
            value=end_date or datetime.now().date(),
            key=f"{key}_end" if key else None
        )
    
    # 검증
    if selected_start > selected_end:
        st.error("시작일이 종료일보다 늦을 수 없습니다.")
        return None
    
    if max_days:
        delta = (selected_end - selected_start).days
        if delta > max_days:
            st.error(f"최대 {max_days}일까지만 선택 가능합니다.")
            return None
    
    return (selected_start, selected_end)

# ===========================================================================
# 🏷️ 빈 상태
# ===========================================================================

def render_empty_state(
    icon: str = "📭",
    title: str = "데이터가 없습니다",
    description: Optional[str] = None,
    action_label: Optional[str] = None,
    action_callback: Optional[Callable] = None
):
    """
    빈 상태 UI 렌더링
    
    Args:
        icon: 아이콘
        title: 제목
        description: 설명
        action_label: 액션 버튼 레이블
        action_callback: 액션 콜백
    """
    st.markdown(
        f"""
        <div style="
            text-align: center;
            padding: 3rem;
            color: {COLORS['text_secondary']};
        ">
            <div style="font-size: 4rem; margin-bottom: 1rem;">{icon}</div>
            <h3 style="color: {COLORS['text_primary']}; margin-bottom: 0.5rem;">{title}</h3>
            {f'<p style="margin-bottom: 1.5rem;">{description}</p>' if description else ''}
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
# 🔄 로딩 상태
# ===========================================================================

def show_loading(
    message: str = "처리 중...",
    spinner: bool = True
):
    """
    로딩 상태 표시
    
    Args:
        message: 로딩 메시지
        spinner: 스피너 표시 여부
    """
    if spinner:
        with st.spinner(message):
            placeholder = st.empty()
    else:
        st.info(f"{ICONS['loading']} {message}")
        placeholder = st.empty()
    
    return placeholder

# ===========================================================================
# 👤 사용자 컴포넌트
# ===========================================================================

def render_user_badge(user: Dict[str, Any]):
    """사용자 배지 표시"""
    if not user:
        return
    
    level_colors = {
        'beginner': COLORS['info'],
        'intermediate': COLORS['success'],
        'advanced': COLORS['warning'],
        'expert': COLORS['accent']
    }
    
    st.markdown(
        f"""
        <div style="
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem;
            background: {COLORS['surface']};
            border-radius: 20px;
            border: 1px solid {COLORS['background']};
        ">
            <span style="font-size: 1.5rem;">{ICONS['user']}</span>
            <div>
                <div style="font-weight: 500;">{user.get('name', 'User')}</div>
                <div style="
                    font-size: 0.75rem;
                    color: {level_colors.get(user.get('level', 'beginner'), COLORS['text_secondary'])};
                ">
                    {user.get('level', 'beginner').title()}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ===========================================================================
# 📱 사이드바
# ===========================================================================

def render_sidebar_menu(
    menu_items: List[Dict[str, Any]],
    current_page: Optional[str] = None
) -> Optional[str]:
    """
    사이드바 메뉴 렌더링
    
    Args:
        menu_items: 메뉴 아이템 리스트
            - name: 메뉴명
            - icon: 아이콘
            - page: 페이지 ID
            - badge: 배지 (선택)
        current_page: 현재 페이지
        
    Returns:
        선택된 페이지 ID
    """
    selected_page = None
    
    for item in menu_items:
        # 배지 처리
        label = f"{item.get('icon', '')} {item['name']}"
        if 'badge' in item and item['badge']:
            label += f" ({item['badge']})"
        
        # 현재 페이지 강조
        if current_page == item.get('page'):
            st.markdown(
                f"""
                <div style="
                    background: {COLORS['primary']}20;
                    padding: 0.5rem 1rem;
                    border-radius: 8px;
                    margin: 0.25rem 0;
                    border-left: 3px solid {COLORS['primary']};
                ">
                    {label}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            if st.button(
                label,
                key=f"menu_{item.get('page', item['name'])}",
                use_container_width=True
            ):
                selected_page = item.get('page')
    
    return selected_page

# ===========================================================================
# 🎯 오프라인 모드 표시
# ===========================================================================

def render_offline_indicator(is_online: bool = True):
    """
    온/오프라인 상태 표시
    
    Args:
        is_online: 온라인 상태
    """
    if not is_online:
        st.markdown(
            f"""
            <div class="offline-badge">
                {ICONS['offline']} 오프라인 모드
            </div>
            """,
            unsafe_allow_html=True
        )

# ===========================================================================
# 🔧 유틸리티 함수
# ===========================================================================

def format_datetime(
    dt: datetime,
    format: Literal["full", "date", "time", "relative"] = "full"
) -> str:
    """
    날짜/시간 포맷팅
    
    Args:
        dt: datetime 객체
        format: 포맷 유형
        
    Returns:
        포맷된 문자열
    """
    if format == "full":
        return dt.strftime("%Y년 %m월 %d일 %H:%M")
    elif format == "date":
        return dt.strftime("%Y-%m-%d")
    elif format == "time":
        return dt.strftime("%H:%M:%S")
    elif format == "relative":
        delta = datetime.now() - dt
        if delta.days > 365:
            return f"{delta.days // 365}년 전"
        elif delta.days > 30:
            return f"{delta.days // 30}개월 전"
        elif delta.days > 0:
            return f"{delta.days}일 전"
        elif delta.seconds > 3600:
            return f"{delta.seconds // 3600}시간 전"
        elif delta.seconds > 60:
            return f"{delta.seconds // 60}분 전"
        else:
            return "방금 전"

def format_number(
    number: Union[int, float],
    decimals: int = 0,
    use_comma: bool = True
) -> str:
    """
    숫자 포맷팅
    
    Args:
        number: 숫자
        decimals: 소수점 자리수
        use_comma: 천단위 구분 기호 사용
        
    Returns:
        포맷된 문자열
    """
    if use_comma:
        return f"{number:,.{decimals}f}"
    else:
        return f"{number:.{decimals}f}"

def truncate_text(
    text: str,
    max_length: int = 100,
    suffix: str = "..."
) -> str:
    """
    텍스트 자르기
    
    Args:
        text: 원본 텍스트
        max_length: 최대 길이
        suffix: 말줄임표
        
    Returns:
        잘린 텍스트
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

# ===========================================================================
# 🎨 색상 팔레트 함수
# ===========================================================================

def get_color_palette(name: str = "default") -> List[str]:
    """
    색상 팔레트 가져오기
    
    Args:
        name: 팔레트 이름
        
    Returns:
        색상 리스트
    """
    palettes = {
        'default': [
            COLORS['primary'], COLORS['secondary'], COLORS['accent'],
            COLORS['warning'], COLORS['info'], COLORS['success']
        ],
        'gradient': [
            '#1E88E5', '#1976D2', '#1565C0', '#0D47A1', '#01579B'
        ],
        'category': px.colors.qualitative.Set3,
        'sequential': px.colors.sequential.Blues,
        'diverging': px.colors.diverging.RdBu
    }
    
    return palettes.get(name, palettes['default'])

# ===========================================================================
# 📤 Export
# ===========================================================================

__all__ = [
    # 페이지 설정
    'setup_page_config',
    
    # 헤더/레이아웃
    'render_header',
    'render_sidebar_menu',
    'render_offline_indicator',
    
    # 메트릭/데이터
    'render_metric_card',
    'render_data_table',
    'render_progress',
    'render_circular_progress',
    
    # AI 컴포넌트
    'render_ai_response',
    
    # 알림
    'show_notification',
    'show_success',
    'show_error',
    'show_warning',
    'show_info',
    
    # 입력
    'create_validated_input',
    'create_file_uploader',
    'create_date_range_picker',
    
    # 상태
    'render_empty_state',
    'show_loading',
    
    # 유틸리티
    'format_datetime',
    'format_number',
    'truncate_text',
    'get_color_palette',
    
    # 상수
    'ICONS',
    'COLORS'
]
