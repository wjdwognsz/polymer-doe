#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎨 Common UI Components Library
================================================================================
재사용 가능한 UI 컴포넌트 모음
일관된 디자인 시스템과 향상된 사용자 경험 제공
================================================================================
"""

# ==================== 표준 라이브러리 ====================
import streamlit as st
from typing import Optional, Dict, List, Any, Tuple, Union, Callable
import json
import base64
from datetime import datetime, timedelta
import re
from functools import wraps
import logging
import pandas as pd
from io import StringIO, BytesIO
from pathlib import Path

# ==================== UI 확장 라이브러리 ====================
try:
    from streamlit_extras.colored_header import colored_header
    from streamlit_extras.metric_cards import style_metric_cards
    from streamlit_extras.badges import badge
    from streamlit_option_menu import option_menu
    EXTRAS_AVAILABLE = True
except ImportError:
    EXTRAS_AVAILABLE = False

# ==================== 시각화 ====================
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# ==================== 로컬 모듈 ====================
try:
    from config.theme_config import THEME_CONFIG
    from config.app_config import APP_CONFIG, UI_CONFIG, LEVEL_CONFIG
except ImportError:
    THEME_CONFIG = {}
    APP_CONFIG = {}
    UI_CONFIG = {}
    LEVEL_CONFIG = {}

# ==================== 로깅 설정 ====================
logger = logging.getLogger(__name__)

# ==================== UI 상수 ====================
DEFAULT_PAGE_ICON = "🧬"
DEFAULT_PAGE_TITLE = "Polymer DOE Platform"
DEFAULT_LAYOUT = "wide"

# 색상 팔레트
COLORS = {
    'primary': '#7C3AED',      # 보라색
    'secondary': '#F59E0B',    # 주황색
    'success': '#10B981',      # 초록색
    'danger': '#EF4444',       # 빨간색
    'warning': '#F59E0B',      # 노란색
    'info': '#3B82F6',         # 파란색
    'dark': '#1F2937',         # 어두운 회색
    'light': '#F3F4F6',        # 밝은 회색
    'muted': '#6B7280'         # 중간 회색
}

# 아이콘 매핑
ICONS = {
    'success': '✅',
    'error': '❌',
    'warning': '⚠️',
    'info': 'ℹ️',
    'loading': '⏳',
    'user': '👤',
    'team': '👥',
    'project': '📁',
    'experiment': '🧪',
    'data': '📊',
    'settings': '⚙️',
    'notification': '🔔',
    'help': '❓'
}

# 애니메이션 CSS
ANIMATIONS = """
<style>
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

.animate-fadeIn {
    animation: fadeIn 0.5s ease-out;
}

.animate-pulse {
    animation: pulse 2s infinite;
}

.animate-slideIn {
    animation: slideIn 0.3s ease-out;
}

/* 커스텀 스크롤바 */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #f1f1f1;
}

::-webkit-scrollbar-thumb {
    background: #888;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #555;
}

/* 버튼 호버 효과 */
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    transition: all 0.3s ease;
}

/* 카드 호버 효과 */
.hover-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
    transition: all 0.3s ease;
}
</style>
"""

# ==================== 페이지 설정 ====================
def setup_page_config(
    title: str = DEFAULT_PAGE_TITLE,
    icon: str = DEFAULT_PAGE_ICON,
    layout: str = DEFAULT_LAYOUT,
    initial_sidebar_state: str = "expanded"
):
    """페이지 기본 설정"""
    try:
        st.set_page_config(
            page_title=title,
            page_icon=icon,
            layout=layout,
            initial_sidebar_state=initial_sidebar_state,
            menu_items={
                'Get Help': 'https://github.com/your-repo/polymer-doe/wiki',
                'Report a bug': 'https://github.com/your-repo/polymer-doe/issues',
                'About': f"""
                # {title}
                
                AI 기반 고분자 실험 설계 플랫폼
                
                Version: 2.0.0
                """
            }
        )
    except Exception as e:
        logger.warning(f"페이지 설정 실패 (이미 설정됨): {e}")

# ==================== CSS 스타일 적용 ====================
def apply_custom_css(theme: str = "light"):
    """커스텀 CSS 적용"""
    # 애니메이션 CSS
    st.markdown(ANIMATIONS, unsafe_allow_html=True)
    
    # 테마별 CSS
    if theme == "dark":
        theme_css = """
        <style>
        .stApp {
            background-color: #1a1a1a;
            color: #ffffff;
        }
        
        .stSidebar {
            background-color: #2d2d2d;
        }
        
        .metric-card {
            background-color: #2d2d2d !important;
            border-color: #404040 !important;
        }
        </style>
        """
    else:
        theme_css = """
        <style>
        .stApp {
            background-color: #ffffff;
        }
        
        .stSidebar {
            background-color: #f8f9fa;
        }
        
        .metric-card {
            background-color: #ffffff !important;
        }
        </style>
        """
    
    st.markdown(theme_css, unsafe_allow_html=True)

# ==================== 헤더 컴포넌트 ====================
def render_header(
    title: str,
    subtitle: Optional[str] = None,
    user_name: Optional[str] = None,
    show_notifications: bool = True,
    custom_buttons: Optional[List[Dict]] = None
):
    """페이지 헤더 렌더링"""
    col1, col2, col3 = st.columns([6, 2, 2])
    
    with col1:
        st.markdown(f"# {title}")
        if subtitle:
            st.markdown(f"<p style='color: {COLORS['muted']};'>{subtitle}</p>", 
                       unsafe_allow_html=True)
            
    with col2:
        if user_name:
            st.markdown(
                f"""
                <div style='text-align: right; padding: 1rem 0;'>
                    <span style='color: {COLORS['muted']};'>안녕하세요,</span>
                    <br>
                    <span style='font-weight: 600;'>{user_name}님</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            
    with col3:
        if show_notifications or custom_buttons:
            button_cols = st.columns(len(custom_buttons) + 1 if custom_buttons else 1)
            
            # 알림 버튼
            if show_notifications:
                with button_cols[0]:
                    notification_count = st.session_state.get('unread_count', 0)
                    if st.button(
                        f"🔔 {notification_count}" if notification_count > 0 else "🔔",
                        key="header_notifications",
                        help="알림 확인"
                    ):
                        st.session_state.show_notifications = True
                        
            # 커스텀 버튼
            if custom_buttons:
                for i, btn in enumerate(custom_buttons):
                    with button_cols[i+1 if show_notifications else i]:
                        if st.button(btn['label'], key=f"header_btn_{i}"):
                            btn['callback']()
                            
    st.markdown("---")

# ==================== 푸터 컴포넌트 ====================
def render_footer():
    """페이지 푸터 렌더링"""
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            """
            ### 🧬 Polymer DOE Platform
            AI 기반 고분자 실험 설계 플랫폼
            
            [📚 사용자 가이드](/) | [🎓 튜토리얼](/)
            """
        )
        
    with col2:
        st.markdown(
            """
            ### 🔗 빠른 링크
            - [프로젝트 관리](/)
            - [실험 설계](/)
            - [데이터 분석](/)
            - [협업 공간](/)
            """
        )
        
    with col3:
        st.markdown(
            """
            ### 📞 지원
            - 이메일: support@polymer-doe.com
            - 전화: 02-1234-5678
            - [💬 실시간 채팅](/)
            """
        )
        
    st.markdown(
        """
        <div style='text-align: center; margin-top: 2rem; padding: 1rem; 
                    color: #6B7280; font-size: 0.875rem;'>
            © 2024 Polymer DOE Platform. All rights reserved. | 
            <a href='/' style='color: #6B7280;'>개인정보처리방침</a> | 
            <a href='/' style='color: #6B7280;'>이용약관</a>
        </div>
        """,
        unsafe_allow_html=True
    )

# ==================== 알림 컴포넌트 ====================
def show_notification(
    message: str,
    type: str = "info",
    duration: int = 3,
    position: str = "top-right"
):
    """토스트 알림 표시"""
    icon = ICONS.get(type, ICONS['info'])
    color = COLORS.get(type, COLORS['info'])
    
    # Streamlit의 기본 알림 사용
    if type == "success":
        st.success(f"{icon} {message}")
    elif type == "error":
        st.error(f"{icon} {message}")
    elif type == "warning":
        st.warning(f"{icon} {message}")
    else:
        st.info(f"{icon} {message}")
        
    # 자동 숨김을 위한 JavaScript (선택사항)
    if duration > 0:
        st.markdown(
            f"""
            <script>
            setTimeout(function() {{
                var alerts = document.querySelectorAll('[role="alert"]');
                if (alerts.length > 0) {{
                    alerts[alerts.length - 1].style.display = 'none';
                }}
            }}, {duration * 1000});
            </script>
            """,
            unsafe_allow_html=True
        )

# 편의 함수들
def show_success(message: str, **kwargs):
    show_notification(message, type="success", **kwargs)

def show_error(message: str, **kwargs):
    show_notification(message, type="error", **kwargs)

def show_warning(message: str, **kwargs):
    show_notification(message, type="warning", **kwargs)

def show_info(message: str, **kwargs):
    show_notification(message, type="info", **kwargs)

# ==================== 메트릭 카드 ====================
def show_metric_card(
    title: str,
    value: Union[str, int, float],
    delta: Optional[Union[str, int, float]] = None,
    delta_color: str = "normal",
    icon: Optional[str] = None,
    help_text: Optional[str] = None,
    background_color: Optional[str] = None,
    animate: bool = True
):
    """메트릭 카드 표시"""
    animation_class = "animate-fadeIn" if animate else ""
    bg_style = f"background-color: {background_color};" if background_color else ""
    
    # 델타 색상 설정
    if delta_color == "normal":
        delta_color_value = COLORS['success'] if delta and str(delta).startswith('+') else COLORS['danger']
    else:
        delta_color_value = COLORS.get(delta_color, COLORS['dark'])
    
    html = f"""
    <div class="metric-card hover-card {animation_class}" style="{bg_style}
         border: 1px solid #E5E7EB; 
         border-radius: 12px; 
         padding: 1.5rem;
         margin-bottom: 1rem;
         box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
         transition: all 0.3s ease;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <p style="color: {COLORS['muted']}; margin: 0; font-size: 0.875rem;">
                    {icon + ' ' if icon else ''}{title}
                </p>
                <h2 style="margin: 0.5rem 0; color: {COLORS['dark']};">
                    {value}
                </h2>
                {f'<p style="color: {delta_color_value}; margin: 0; font-size: 0.875rem;">{"▲" if str(delta).startswith("+") else "▼"} {delta}</p>' if delta else ''}
            </div>
            {f'<span title="{help_text}" style="cursor: help; color: {COLORS["muted"]};">❓</span>' if help_text else ''}
        </div>
    </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)

# ==================== 프로그레스 컴포넌트 ====================
def render_progress_ring(
    value: float,
    max_value: float = 100,
    title: str = "",
    size: int = 120,
    color: str = None
):
    """원형 프로그레스 표시"""
    percentage = min(value / max_value * 100, 100) if max_value > 0 else 0
    color = color or COLORS['primary']
    
    svg = f"""
    <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
        <circle cx="{size/2}" cy="{size/2}" r="{size/2-10}" 
                fill="none" stroke="#E5E7EB" stroke-width="8"/>
        <circle cx="{size/2}" cy="{size/2}" r="{size/2-10}" 
                fill="none" stroke="{color}" stroke-width="8"
                stroke-dasharray="{percentage * 3.14 * (size-20) / 100} {3.14 * (size-20)}"
                stroke-dashoffset="{3.14 * (size-20) / 4}"
                transform="rotate(-90 {size/2} {size/2})"/>
        <text x="{size/2}" y="{size/2}" 
              text-anchor="middle" dominant-baseline="middle"
              font-size="24" font-weight="bold" fill="{COLORS['dark']}">
            {int(percentage)}%
        </text>
    </svg>
    """
    
    st.markdown(
        f"""
        <div style="text-align: center;" class="animate-fadeIn">
            {svg}
            {f'<p style="margin-top: 0.5rem; font-weight: 500;">{title}</p>' if title else ''}
        </div>
        """,
        unsafe_allow_html=True
    )

# ==================== 활동 카드 ====================
def show_activity_card(
    user_name: str,
    action: str,
    timestamp: datetime,
    avatar_url: Optional[str] = None,
    details: Optional[str] = None,
    icon: Optional[str] = None
):
    """활동 카드 표시"""
    avatar = avatar_url or get_default_avatar()
    time_str = format_datetime(timestamp)
    
    html = f"""
    <div class="activity-card animate-fadeIn" style="
         display: flex;
         gap: 1rem;
         padding: 1rem;
         border-bottom: 1px solid #E5E7EB;
         transition: all 0.3s ease;">
        <img src="{avatar}" style="
             width: 40px; 
             height: 40px; 
             border-radius: 50%;">
        <div style="flex: 1;">
            <p style="margin: 0;">
                <strong>{user_name}</strong> {action}
                {f' {icon}' if icon else ''}
            </p>
            {f'<p style="margin: 0.25rem 0; color: {COLORS["muted"]}; font-size: 0.875rem;">{details}</p>' if details else ''}
            <p style="margin: 0; color: {COLORS["muted"]}; font-size: 0.75rem;">
                {time_str}
            </p>
        </div>
    </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)

# ==================== 업적 뱃지 ====================
def render_achievement_badge(
    title: str,
    description: str,
    icon: str,
    earned: bool = True,
    progress: Optional[int] = None,
    max_progress: Optional[int] = None
):
    """업적 뱃지 렌더링"""
    opacity = "1" if earned else "0.3"
    
    html = f"""
    <div class="achievement-badge hover-card" style="
         text-align: center;
         padding: 1.5rem;
         border: 2px solid {'#FFD700' if earned else '#E5E7EB'};
         border-radius: 12px;
         background: {'linear-gradient(135deg, #FFF9E6 0%, #FFFDF7 100%)' if earned else '#F9FAFB'};
         opacity: {opacity};
         transition: all 0.3s ease;">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">
            {icon}
        </div>
        <h4 style="margin: 0.5rem 0; color: {COLORS['dark']};">
            {title}
        </h4>
        <p style="margin: 0; color: {COLORS['muted']}; font-size: 0.875rem;">
            {description}
        </p>
        {f'<div style="margin-top: 0.5rem;"><small>{progress}/{max_progress}</small></div>' if progress is not None else ''}
    </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)

# ==================== 입력 검증 컴포넌트 ====================
def create_validated_input(
    label: str,
    input_type: str = "text",
    key: str = None,
    value: Any = None,
    validation_func: Optional[Callable] = None,
    validation_message: str = "입력값이 올바르지 않습니다.",
    required: bool = False,
    help_text: Optional[str] = None,
    **kwargs
):
    """검증 기능이 있는 입력 컴포넌트"""
    # 필수 표시
    if required:
        label = f"{label} *"
        
    # 입력 컴포넌트 생성
    if input_type == "text":
        input_value = st.text_input(label, value=value, key=key, help=help_text, **kwargs)
    elif input_type == "number":
        input_value = st.number_input(label, value=value, key=key, help=help_text, **kwargs)
    elif input_type == "email":
        input_value = st.text_input(label, value=value, key=key, help=help_text, **kwargs)
        # 이메일 검증
        if input_value and not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', input_value):
            st.error("올바른 이메일 주소를 입력해주세요.")
            return None
    elif input_type == "password":
        input_value = st.text_input(label, type="password", value=value, key=key, help=help_text, **kwargs)
    elif input_type == "textarea":
        input_value = st.text_area(label, value=value, key=key, help=help_text, **kwargs)
    else:
        input_value = st.text_input(label, value=value, key=key, help=help_text, **kwargs)
        
    # 필수 필드 검증
    if required and not input_value:
        st.error(f"{label.replace(' *', '')}은(는) 필수 입력 항목입니다.")
        return None
        
    # 커스텀 검증
    if validation_func and input_value:
        if not validation_func(input_value):
            st.error(validation_message)
            return None
            
    return input_value

# ==================== 로딩 상태 ====================
def render_loading(
    message: str = "로딩 중...",
    spinner_type: str = "dots"
):
    """로딩 상태 표시"""
    spinner_html = """
    <div style="text-align: center; padding: 2rem;">
        <div class="spinner"></div>
        <p style="margin-top: 1rem; color: #6B7280;">{message}</p>
    </div>
    
    <style>
    .spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #7C3AED;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 0 auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
    """.format(message=message)
    
    st.markdown(spinner_html, unsafe_allow_html=True)

# ==================== 사용자 정보 컴포넌트 ====================
def render_user_profile_card(
    user: Dict,
    show_actions: bool = True
):
    """사용자 프로필 카드"""
    col1, col2 = st.columns([1, 3])
    
    with col1:
        avatar_url = user.get('profile_image', get_default_avatar())
        st.markdown(
            f"""
            <img src="{avatar_url}" style="
                 width: 100px;
                 height: 100px;
                 border-radius: 50%;
                 border: 3px solid {COLORS['primary']};
                 margin: 0 auto;
                 display: block;">
            """,
            unsafe_allow_html=True
        )
        
    with col2:
        st.markdown(f"### {user.get('name', 'User')}")
        st.caption(f"{user.get('organization', '')} • {user.get('email', '')}")
        
        # 레벨과 포인트
        col_a, col_b = st.columns(2)
        with col_a:
            level_info = LEVEL_CONFIG.get('levels', {}).get(user.get('level', 'beginner'), {})
            show_metric_card(
                "레벨", 
                level_info.get('name', user.get('level', 'beginner').title()),
                icon=level_info.get('icon', '🎯')
            )
        with col_b:
            show_metric_card("포인트", f"{user.get('points', 0):,}", icon="⭐")
            
    if show_actions:
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("프로필 편집", key="edit_profile", use_container_width=True):
                st.session_state.current_page = 'profile'
                
        with col2:
            if st.button("내 프로젝트", key="my_projects", use_container_width=True):
                st.session_state.current_page = 'dashboard'
                
        with col3:
            if st.button("로그아웃", key="logout", use_container_width=True):
                st.session_state.authenticated = False
                st.rerun()

# ==================== 파일 업로드 컴포넌트 ====================
def create_file_uploader(
    label: str,
    accepted_types: List[str],
    max_size_mb: int = 10,
    multiple: bool = False,
    help_text: Optional[str] = None,
    show_preview: bool = True,
    key: str = None
):
    """향상된 파일 업로더"""
    # 파일 타입 설명
    type_descriptions = {
        'csv': 'CSV 파일',
        'xlsx': 'Excel 파일',
        'pdf': 'PDF 문서',
        'png': 'PNG 이미지',
        'jpg': 'JPEG 이미지',
        'txt': '텍스트 파일'
    }
    
    accepted_desc = ", ".join([
        type_descriptions.get(t, t.upper()) for t in accepted_types
    ])
    
    # 업로더
    uploaded_files = st.file_uploader(
        label,
        type=accepted_types,
        accept_multiple_files=multiple,
        help=help_text or f"지원 형식: {accepted_desc} (최대 {max_size_mb}MB)",
        key=key
    )
    
    if uploaded_files:
        files = uploaded_files if multiple else [uploaded_files]
        
        for file in files:
            # 파일 크기 확인
            file_size_mb = file.size / (1024 * 1024)
            if file_size_mb > max_size_mb:
                st.error(f"{file.name}: 파일 크기가 {max_size_mb}MB를 초과합니다.")
                continue
                
            # 파일 정보 표시
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.text(f"📄 {file.name}")
                
            with col2:
                st.caption(f"{file_size_mb:.1f} MB")
                
            with col3:
                if st.button("❌", key=f"remove_{file.name}", help="파일 제거"):
                    # 파일 제거 로직
                    pass
                    
            # 미리보기
            if show_preview and file.type.startswith('image/'):
                try:
                    img = Image.open(file)
                    st.image(img, width=200)
                except Exception as e:
                    st.error(f"이미지 미리보기 실패: {e}")
                    
    return uploaded_files

# ==================== 빈 상태 컴포넌트 ====================
def show_empty_state(
    icon: str = "📭",
    title: str = "데이터가 없습니다",
    description: str = "",
    action_label: Optional[str] = None,
    action_callback: Optional[Callable] = None
):
    """빈 상태 표시"""
    st.markdown(
        f"""
        <div style="text-align: center; padding: 3rem; color: {COLORS['muted']};" 
             class="animate-fadeIn">
            <div style="font-size: 4rem; margin-bottom: 1rem;">{icon}</div>
            <h3 style="color: {COLORS['dark']}; margin-bottom: 0.5rem;">{title}</h3>
            <p style="margin-bottom: 1.5rem;">{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if action_label and action_callback:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button(action_label, type="primary", use_container_width=True):
                action_callback()

# ==================== 협업 관련 컴포넌트 ====================
def render_collaborator_list(
    collaborators: List[Dict],
    show_actions: bool = True
):
    """협업자 목록"""
    for collab in collaborators:
        col1, col2, col3, col4 = st.columns([1, 3, 2, 1])
        
        with col1:
            # 온라인 상태 표시
            status_color = COLORS['success'] if collab.get('online') else COLORS['muted']
            st.markdown(
                f"""
                <div style="position: relative;">
                    <img src="{collab.get('avatar', get_default_avatar())}" 
                         style="width: 40px; height: 40px; border-radius: 50%;">
                    <div style="position: absolute; bottom: 0; right: 0; 
                                width: 12px; height: 12px; 
                                background-color: {status_color}; 
                                border-radius: 50%; 
                                border: 2px solid white;">
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with col2:
            st.markdown(f"**{collab.get('name', 'Unknown')}**")
            st.caption(f"{collab.get('role', 'Viewer')} • {collab.get('email', '')}")
            
        with col3:
            if collab.get('last_active'):
                st.caption(f"최근 활동: {format_datetime(collab['last_active'])}")
                
        with col4:
            if show_actions:
                if st.button("⚙️", key=f"collab_settings_{collab.get('id')}", help="설정"):
                    # 협업자 설정 로직
                    pass

def render_comment_thread(
    comments: List[Dict],
    allow_reply: bool = True
):
    """댓글 스레드"""
    for comment in comments:
        # 댓글 카드
        st.markdown(
            f"""
            <div class="comment-card" style="
                 margin-bottom: 1rem;
                 padding: 1rem;
                 background-color: {COLORS['light']};
                 border-radius: 8px;
                 border-left: 3px solid {COLORS['primary']};">
                <div style="display: flex; gap: 1rem;">
                    <img src="{comment.get('author_avatar', get_default_avatar())}" 
                         style="width: 32px; height: 32px; border-radius: 50%;">
                    <div style="flex: 1;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <strong>{comment.get('author_name', 'Unknown')}</strong>
                            <small style="color: {COLORS['muted']};">
                                {format_datetime(comment.get('created_at', datetime.now()))}
                            </small>
                        </div>
                        <p style="margin: 0.5rem 0;">{comment.get('content', '')}</p>
                        <div style="display: flex; gap: 1rem; margin-top: 0.5rem;">
                            <button style="background: none; border: none; color: {COLORS['primary']}; 
                                         cursor: pointer; font-size: 0.875rem;">
                                👍 {comment.get('likes', 0)}
                            </button>
                            {"<button style='background: none; border: none; color: #6B7280; cursor: pointer; font-size: 0.875rem;'>💬 답글</button>" if allow_reply else ""}
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # 답글들
        if comment.get('replies'):
            for reply in comment['replies']:
                st.markdown(
                    f"""
                    <div style="margin-left: 3rem; margin-bottom: 0.5rem;">
                        <div class="comment-card" style="
                             padding: 0.75rem;
                             background-color: white;
                             border-radius: 8px;">
                            <!-- 답글 내용 -->
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# ==================== 다운로드 버튼 ====================
def create_download_button(
    data: Any,
    filename: str,
    label: str = "다운로드",
    mime_type: str = "text/csv",
    key: str = None
):
    """다운로드 버튼 생성"""
    if isinstance(data, pd.DataFrame):
        data = data.to_csv(index=False)
    elif isinstance(data, dict):
        data = json.dumps(data, ensure_ascii=False, indent=2)
    elif isinstance(data, list):
        data = json.dumps(data, ensure_ascii=False, indent=2)
        
    st.download_button(
        label=f"📥 {label}",
        data=data,
        file_name=filename,
        mime=mime_type,
        key=key,
        use_container_width=True
    )

# ==================== 유틸리티 함수 ====================
def get_default_avatar():
    """기본 아바타 이미지 URL"""
    return "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ccircle cx='50' cy='50' r='40' fill='%237C3AED'/%3E%3Ctext x='50' y='50' text-anchor='middle' dy='.3em' fill='white' font-size='40'%3E👤%3C/text%3E%3C/svg%3E"

def format_datetime(dt: datetime, format: str = "relative") -> str:
    """날짜/시간 포맷팅"""
    if not dt:
        return ""
        
    if format == "relative":
        now = datetime.now()
        # 타임존 처리
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=now.tzinfo)
        if now.tzinfo is None:
            now = now.replace(tzinfo=dt.tzinfo)
            
        delta = now - dt
        
        if delta.days > 7:
            return dt.strftime("%Y-%m-%d")
        elif delta.days > 0:
            return f"{delta.days}일 전"
        elif delta.total_seconds() > 3600:
            hours = int(delta.total_seconds() / 3600)
            return f"{hours}시간 전"
        elif delta.total_seconds() > 60:
            minutes = int(delta.total_seconds() / 60)
            return f"{minutes}분 전"
        else:
            return "방금 전"
    else:
        return dt.strftime(format)

def render_modern_sidebar():
    """모던한 사이드바 렌더링"""
    with st.sidebar:
        # 로고 및 타이틀
        st.markdown(
            """
            <div style='text-align: center; padding: 1rem 0;'>
                <h1 style='color: #7C3AED; margin: 0;'>🧬 Polymer DOE</h1>
                <p style='color: #6B7280; font-size: 0.9em; margin: 0;'>v2.0.0</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.divider()
        
        # 메뉴 옵션
        if EXTRAS_AVAILABLE:
            selected = option_menu(
                menu_title=None,
                options=["대시보드", "프로젝트", "실험", "분석", "문헌", "시각화", "협업"],
                icons=["house", "folder", "flask", "graph-up", "search", "bar-chart", "people"],
                menu_icon="cast",
                default_index=0,
                styles={
                    "container": {"padding": "0!important", "background-color": "#fafafa"},
                    "icon": {"color": "#7C3AED", "font-size": "20px"},
                    "nav-link": {
                        "font-size": "16px",
                        "text-align": "left",
                        "margin": "0px",
                        "--hover-color": "#eee"
                    },
                    "nav-link-selected": {"background-color": "#7C3AED"},
                }
            )
            st.session_state.selected_menu = selected.lower()
        else:
            # 폴백: 기본 Streamlit 버튼 사용
            menu_items = {
                'dashboard': {'title': '대시보드', 'icon': '📊'},
                'project': {'title': '프로젝트', 'icon': '📁'},
                'experiment': {'title': '실험', 'icon': '🧪'},
                'analysis': {'title': '분석', 'icon': '📈'},
                'literature': {'title': '문헌', 'icon': '🔍'},
                'visualization': {'title': '시각화', 'icon': '📊'},
                'collaboration': {'title': '협업', 'icon': '👥'}
            }
            
            for key, item in menu_items.items():
                if st.button(
                    f"{item['icon']} {item['title']}", 
                    use_container_width=True,
                    key=f"menu_{key}"
                ):
                    st.session_state.selected_menu = key

# ==================== 권한 체크 컴포넌트 ====================
def render_permission_required(
    required_level: str = "intermediate",
    custom_message: Optional[str] = None
):
    """권한 없음 안내 - 교육적 성장 중심으로 모든 기능은 접근 가능"""
    # 교육적 성장 중심 플랫폼에서는 레벨 제한 없음
    # 대신 레벨에 따른 교육적 지원 제공
    current_level = st.session_state.get('user', {}).get('level', 'beginner')
    
    # 모든 사용자가 접근 가능하므로 항상 True 반환
    return True

# ==================== 활동 피드 ====================
def render_activity_feed(
    activities: List[Dict],
    max_items: int = 10
):
    """활동 피드 렌더링"""
    st.markdown("### 📋 최근 활동")
    
    if not activities:
        show_empty_state(
            icon="🏃",
            title="아직 활동이 없습니다",
            description="프로젝트를 시작하면 여기에 활동이 표시됩니다."
        )
        return
        
    for activity in activities[:max_items]:
        show_activity_card(
            user_name=activity.get('user_name', 'Unknown'),
            action=activity.get('action', ''),
            timestamp=activity.get('timestamp', datetime.now()),
            avatar_url=activity.get('avatar_url'),
            details=activity.get('details'),
            icon=activity.get('icon')
        )
        
    if len(activities) > max_items:
        if st.button("더 보기", key="load_more_activities"):
            st.session_state.show_all_activities = True

# ==================== 파일 미리보기 ====================
def render_file_preview(file):
    """파일 미리보기 렌더링"""
    try:
        if file.type == 'text/csv':
            df = pd.read_csv(file)
            st.dataframe(df.head(10))
            st.caption(f"총 {len(df)}개 행")
            
        elif file.type in ['application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']:
            df = pd.read_excel(file)
            st.dataframe(df.head(10))
            st.caption(f"총 {len(df)}개 행")
            
        elif file.type == 'text/plain':
            content = file.read().decode('utf-8')
            st.text_area("내용", content[:1000], height=200)
            if len(content) > 1000:
                st.caption(f"... 외 {len(content) - 1000}자 더")
                
        elif file.type.startswith('image/'):
            img = Image.open(file)
            st.image(img, width=300)
            st.caption(f"크기: {img.size[0]} x {img.size[1]}")
            
    except Exception as e:
        st.error(f"파일 미리보기 실패: {e}")

# ==================== 차트 테마 적용 ====================
def apply_chart_theme(fig):
    """Plotly 차트에 테마 적용"""
    theme = st.session_state.get('theme', 'light')
    
    if theme == 'dark':
        fig.update_layout(
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#1a1a1a',
            font_color='white'
        )
    else:
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_color='black'
        )
        
    return fig

# ==================== 내보내기 ====================
__all__ = [
    # 페이지 설정
    'setup_page_config',
    'apply_custom_css',
    
    # 헤더/푸터
    'render_header',
    'render_footer',
    
    # 알림
    'show_notification',
    'show_success',
    'show_error',
    'show_warning',
    'show_info',
    
    # 메트릭 및 프로그레스
    'show_metric_card',
    'render_progress_ring',
    
    # 활동 및 업적
    'show_activity_card',
    'render_achievement_badge',
    
    # 입력 및 파일
    'create_validated_input',
    'create_file_uploader',
    
    # 사용자 정보
    'render_user_profile_card',
    
    # 협업
    'render_collaborator_list',
    'render_comment_thread',
    
    # 유틸리티
    'create_download_button',
    'show_empty_state',
    'render_loading',
    'render_modern_sidebar',
    'render_permission_required',
    'render_activity_feed',
    'format_datetime',
    'get_default_avatar',
    
    # 상수
    'COLORS',
    'ICONS'
]
