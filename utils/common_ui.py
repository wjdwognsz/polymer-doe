"""
🎨 Common UI Components - 공통 UI 컴포넌트 라이브러리
===========================================================================
데스크톱 애플리케이션을 위한 재사용 가능한 UI 컴포넌트 모음
일관된 디자인 시스템, 테마 지원, 오프라인 최적화
===========================================================================
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import json
import base64
from datetime import datetime, timedelta
import time
from functools import wraps
import logging
from pathlib import Path
from PIL import Image
import io

# 로컬 설정
try:
    from config.theme_config import COLORS, FONTS, LAYOUT, CUSTOM_CSS
    from config.app_config import APP_INFO, UI_CONFIG
    from config.local_config import LOCAL_CONFIG
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

# ===========================================================================
# 🔧 설정 및 초기화
# ===========================================================================

logger = logging.getLogger(__name__)

# 아이콘 매핑
ICONS = {
    'success': '✅',
    'error': '❌',
    'warning': '⚠️',
    'info': 'ℹ️',
    'tip': '💡',
    'loading': '⏳',
    'user': '👤',
    'team': '👥',
    'project': '📁',
    'experiment': '🧪',
    'data': '📊',
    'analysis': '📈',
    'settings': '⚙️',
    'notification': '🔔',
    'help': '❓',
    'search': '🔍',
    'filter': '🔽',
    'calendar': '📅',
    'clock': '⏰',
    'download': '⬇️',
    'upload': '⬆️',
    'share': '📤',
    'lock': '🔒',
    'unlock': '🔓',
    'star': '⭐',
    'heart': '❤️',
    'fire': '🔥',
    'rocket': '🚀',
    'trophy': '🏆',
    'medal': '🥇',
    'flag': '🚩',
    'pin': '📌',
    'tag': '🏷️',
    'folder': '📂',
    'file': '📄',
    'save': '💾',
    'delete': '🗑️',
    'edit': '✏️',
    'copy': '📋',
    'paste': '📋',
    'cut': '✂️',
    'link': '🔗',
    'external': '🔗',
    'home': '🏠',
    'back': '⬅️',
    'forward': '➡️',
    'refresh': '🔄',
    'sync': '🔄',
    'offline': '📴',
    'online': '📶',
    'api': '🔌',
    'database': '🗄️',
    'cloud': '☁️',
    'local': '💾'
}

# 애니메이션 CSS
ANIMATIONS_CSS = """
<style>
/* 페이드 인 애니메이션 */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* 슬라이드 인 애니메이션 */
@keyframes slideIn {
    from { transform: translateX(-100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}

/* 펄스 애니메이션 */
@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}

/* 회전 애니메이션 */
@keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

/* 바운스 애니메이션 */
@keyframes bounce {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-10px); }
}

/* 애니메이션 클래스 */
.animate-fadeIn { animation: fadeIn 0.5s ease-out; }
.animate-slideIn { animation: slideIn 0.3s ease-out; }
.animate-pulse { animation: pulse 2s infinite; }
.animate-spin { animation: spin 1s linear infinite; }
.animate-bounce { animation: bounce 1s ease-in-out infinite; }

/* 호버 효과 */
.hover-scale:hover { transform: scale(1.05); transition: transform 0.2s; }
.hover-shadow:hover { box-shadow: 0 8px 16px rgba(0,0,0,0.1); transition: box-shadow 0.2s; }
.hover-bright:hover { filter: brightness(1.1); transition: filter 0.2s; }

/* 커스텀 버튼 스타일 */
.custom-button {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.3s;
}

.custom-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}

/* 카드 스타일 */
.custom-card {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    transition: all 0.3s;
}

.custom-card:hover {
    box-shadow: 0 4px 16px rgba(0,0,0,0.12);
}

/* 메트릭 카드 */
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
    transform: rotate(45deg);
}

/* 프로그레스 바 */
.custom-progress {
    background: #e0e0e0;
    border-radius: 10px;
    height: 10px;
    overflow: hidden;
    position: relative;
}

.custom-progress-bar {
    background: linear-gradient(90deg, #4CAF50 0%, #8BC34A 100%);
    height: 100%;
    transition: width 0.5s ease-out;
    position: relative;
    overflow: hidden;
}

.custom-progress-bar::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(
        90deg,
        transparent,
        rgba(255,255,255,0.3),
        transparent
    );
    animation: shimmer 2s infinite;
}

@keyframes shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

/* 스크롤바 커스터마이징 */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: #888;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #555;
}

/* 툴팁 */
.custom-tooltip {
    position: relative;
    display: inline-block;
}

.custom-tooltip .tooltiptext {
    visibility: hidden;
    width: 200px;
    background-color: #555;
    color: #fff;
    text-align: center;
    border-radius: 6px;
    padding: 5px;
    position: absolute;
    z-index: 1;
    bottom: 125%;
    left: 50%;
    margin-left: -100px;
    opacity: 0;
    transition: opacity 0.3s;
}

.custom-tooltip:hover .tooltiptext {
    visibility: visible;
    opacity: 1;
}

/* 오프라인 배지 */
.offline-badge {
    position: fixed;
    top: 10px;
    right: 10px;
    background: #ff5722;
    color: white;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.875rem;
    z-index: 1000;
    animation: pulse 2s infinite;
}

/* 반응형 그리드 */
.responsive-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
    padding: 1rem;
}

/* 다크모드 지원 */
@media (prefers-color-scheme: dark) {
    .custom-card {
        background: #2d2d2d;
        color: #ffffff;
    }
    
    .custom-progress {
        background: #404040;
    }
}
</style>
"""

# ===========================================================================
# 🎨 페이지 설정 및 테마
# ===========================================================================

def setup_page(
    title: str = "Universal DOE Platform",
    icon: str = "🧬",
    layout: str = "wide",
    initial_sidebar_state: str = "expanded",
    menu_items: Optional[Dict] = None
):
    """
    페이지 초기 설정
    
    Args:
        title: 페이지 제목
        icon: 페이지 아이콘
        layout: 레이아웃 ("wide" or "centered")
        initial_sidebar_state: 사이드바 상태
        menu_items: 메뉴 아이템 설정
    """
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout=layout,
        initial_sidebar_state=initial_sidebar_state,
        menu_items=menu_items or {
            'Get Help': APP_INFO.get('github', 'https://github.com'),
            'Report a bug': f"{APP_INFO.get('github', 'https://github.com')}/issues",
            'About': APP_INFO.get('description', 'Universal DOE Platform')
        }
    )
    
    # CSS 주입
    st.markdown(ANIMATIONS_CSS, unsafe_allow_html=True)
    if 'CUSTOM_CSS' in globals():
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # 오프라인 모드 표시
    if LOCAL_CONFIG.get('offline_mode', {}).get('default', False):
        if not st.session_state.get('is_online', True):
            st.markdown(
                '<div class="offline-badge">📴 오프라인 모드</div>',
                unsafe_allow_html=True
            )

def apply_custom_theme():
    """커스텀 테마 적용"""
    st.markdown(f"""
    <style>
    :root {{
        --primary-color: {COLORS['primary']};
        --secondary-color: {COLORS['secondary']};
        --background-color: {COLORS['background']};
        --surface-color: {COLORS['surface']};
        --text-primary: {COLORS['text_primary']};
        --text-secondary: {COLORS['text_secondary']};
    }}
    
    .stApp {{
        background-color: var(--background-color);
        color: var(--text-primary);
    }}
    
    .stButton > button {{
        background-color: var(--primary-color);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s;
    }}
    
    .stButton > button:hover {{
        background-color: var(--secondary-color);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}
    
    .stTextInput > div > div > input {{
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        padding: 0.5rem;
        transition: border-color 0.3s;
    }}
    
    .stTextInput > div > div > input:focus {{
        border-color: var(--primary-color);
        box-shadow: 0 0 0 2px rgba(30, 136, 229, 0.2);
    }}
    
    .stSelectbox > div > div > div {{
        border-radius: 8px;
        border: 2px solid #e0e0e0;
    }}
    
    .stExpander {{
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        overflow: hidden;
    }}
    
    .stTabs [data-baseweb="tab-list"] {{
        gap: 1rem;
        background-color: transparent;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        padding: 0.5rem 1rem;
        border-radius: 8px;
        background-color: var(--surface-color);
        transition: all 0.3s;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background-color: var(--primary-color);
        color: white;
    }}
    
    .stMetric {{
        background-color: var(--surface-color);
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }}
    </style>
    """, unsafe_allow_html=True)

# ===========================================================================
# 🎯 헤더 및 네비게이션
# ===========================================================================

def render_header(
    title: str,
    subtitle: Optional[str] = None,
    show_user_info: bool = True,
    show_notifications: bool = True,
    custom_buttons: Optional[List[Dict]] = None
):
    """
    페이지 헤더 렌더링
    
    Args:
        title: 페이지 제목
        subtitle: 부제목
        show_user_info: 사용자 정보 표시 여부
        show_notifications: 알림 아이콘 표시 여부
        custom_buttons: 추가 버튼 리스트
    """
    col1, col2, col3 = st.columns([6, 2, 2])
    
    with col1:
        st.markdown(f'<h1 class="animate-fadeIn">{title}</h1>', unsafe_allow_html=True)
        if subtitle:
            st.markdown(f'<p style="color: {COLORS["text_secondary"]};">{subtitle}</p>', 
                       unsafe_allow_html=True)
    
    with col2:
        if show_user_info and st.session_state.get('authenticated', False):
            user = st.session_state.get('user', {})
            if user:
                st.markdown(f"""
                <div class="animate-slideIn" style="text-align: right;">
                    <small>{ICONS['user']} {user.get('name', 'User')}</small><br>
                    <small style="color: {COLORS['text_secondary']};">
                        {user.get('role', 'user').title()}
                    </small>
                </div>
                """, unsafe_allow_html=True)
    
    with col3:
        button_cols = st.columns(3 if show_notifications else 2)
        
        # 알림 버튼
        if show_notifications:
            with button_cols[0]:
                notification_count = st.session_state.get('unread_notifications', 0)
                notification_label = f"{ICONS['notification']} {notification_count}" if notification_count > 0 else ICONS['notification']
                if st.button(notification_label, key="header_notifications", help="알림"):
                    st.session_state.show_notifications = not st.session_state.get('show_notifications', False)
        
        # 커스텀 버튼
        if custom_buttons:
            for i, btn in enumerate(custom_buttons[:2]):
                with button_cols[i+1 if show_notifications else i]:
                    if st.button(btn['label'], key=f"header_btn_{i}", help=btn.get('help')):
                        if 'callback' in btn:
                            btn['callback']()
    
    st.markdown("---")

def render_sidebar_menu(
    menu_items: Dict[str, Dict[str, Any]],
    default_index: int = 0
) -> str:
    """
    사이드바 메뉴 렌더링
    
    Args:
        menu_items: 메뉴 아이템 딕셔너리
        default_index: 기본 선택 인덱스
        
    Returns:
        선택된 메뉴 키
    """
    with st.sidebar:
        st.markdown(f"### {ICONS['home']} 메뉴")
        
        # 메뉴 옵션 생성
        options = []
        keys = []
        for key, item in menu_items.items():
            icon = item.get('icon', '')
            label = item.get('label', key)
            options.append(f"{icon} {label}")
            keys.append(key)
        
        # 현재 선택된 인덱스 찾기
        current_page = st.session_state.get('current_page')
        if current_page in keys:
            default_index = keys.index(current_page)
        
        # 메뉴 렌더링
        selected_option = st.radio(
            "페이지 선택",
            options,
            index=default_index,
            label_visibility="collapsed"
        )
        
        # 선택된 키 찾기
        selected_index = options.index(selected_option)
        selected_key = keys[selected_index]
        
        # 페이지 전환
        if selected_key != st.session_state.get('current_page'):
            st.session_state.current_page = selected_key
            st.rerun()
        
        return selected_key

def render_breadcrumb(items: List[Tuple[str, Optional[str]]]):
    """
    브레드크럼 네비게이션
    
    Args:
        items: [(label, page_key), ...] 형태의 리스트
    """
    breadcrumb_html = []
    for i, (label, page_key) in enumerate(items):
        if i > 0:
            breadcrumb_html.append(' > ')
        
        if page_key and i < len(items) - 1:  # 마지막 아이템이 아니면 링크
            breadcrumb_html.append(f'<a href="#" onclick="return false;" style="color: {COLORS["primary"]};">{label}</a>')
        else:
            breadcrumb_html.append(f'<span style="color: {COLORS["text_secondary"]};">{label}</span>')
    
    st.markdown(
        f'<div class="animate-fadeIn" style="padding: 0.5rem 0;">{"".join(breadcrumb_html)}</div>',
        unsafe_allow_html=True
    )

# ===========================================================================
# 💬 메시지 및 알림
# ===========================================================================

def show_message(
    message: str,
    type: str = "info",
    icon: Optional[str] = None,
    duration: Optional[int] = None
):
    """
    메시지 표시
    
    Args:
        message: 메시지 내용
        type: 메시지 타입 (success, error, warning, info)
        icon: 커스텀 아이콘
        duration: 표시 시간 (초)
    """
    # 아이콘 선택
    if not icon:
        icon = ICONS.get(type, ICONS['info'])
    
    # 색상 선택
    color_map = {
        'success': COLORS['success'],
        'error': COLORS['error'],
        'warning': COLORS['warning'],
        'info': COLORS['info']
    }
    color = color_map.get(type, COLORS['info'])
    
    # 배경색 (연한 색)
    bg_color = color + '20'  # 20% 투명도
    
    # 메시지 표시
    message_html = f"""
    <div class="animate-fadeIn" style="
        background-color: {bg_color};
        border-left: 4px solid {color};
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    ">
        <span style="font-size: 1.2rem; margin-right: 0.5rem;">{icon}</span>
        <span>{message}</span>
    </div>
    """
    
    if duration:
        placeholder = st.empty()
        placeholder.markdown(message_html, unsafe_allow_html=True)
        time.sleep(duration)
        placeholder.empty()
    else:
        st.markdown(message_html, unsafe_allow_html=True)

def show_success(message: str, **kwargs):
    """성공 메시지"""
    show_message(message, type="success", **kwargs)

def show_error(message: str, **kwargs):
    """에러 메시지"""
    show_message(message, type="error", **kwargs)

def show_warning(message: str, **kwargs):
    """경고 메시지"""
    show_message(message, type="warning", **kwargs)

def show_info(message: str, **kwargs):
    """정보 메시지"""
    show_message(message, type="info", **kwargs)

def show_notification(
    title: str,
    message: str,
    type: str = "info",
    position: str = "top-right",
    duration: int = 5
):
    """
    팝업 알림 표시 (토스트)
    
    Args:
        title: 알림 제목
        message: 알림 내용
        type: 알림 타입
        position: 위치 (top-right, top-left, bottom-right, bottom-left)
        duration: 표시 시간
    """
    # 위치 스타일
    position_styles = {
        'top-right': 'top: 20px; right: 20px;',
        'top-left': 'top: 20px; left: 20px;',
        'bottom-right': 'bottom: 20px; right: 20px;',
        'bottom-left': 'bottom: 20px; left: 20px;'
    }
    
    # 색상
    color = {
        'success': COLORS['success'],
        'error': COLORS['error'],
        'warning': COLORS['warning'],
        'info': COLORS['info']
    }.get(type, COLORS['info'])
    
    notification_html = f"""
    <div class="notification animate-slideIn" style="
        position: fixed;
        {position_styles.get(position, position_styles['top-right'])}
        background: white;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        padding: 1rem;
        min-width: 300px;
        z-index: 9999;
        border-left: 4px solid {color};
    ">
        <h4 style="margin: 0 0 0.5rem 0; color: {color};">
            {ICONS.get(type, '')} {title}
        </h4>
        <p style="margin: 0; color: {COLORS['text_secondary']};">
            {message}
        </p>
    </div>
    
    <script>
        setTimeout(function() {{
            document.querySelector('.notification').style.display = 'none';
        }}, {duration * 1000});
    </script>
    """
    
    st.markdown(notification_html, unsafe_allow_html=True)

# ===========================================================================
# 📊 메트릭 및 통계
# ===========================================================================

def render_metric_card(
    label: str,
    value: Union[str, int, float],
    delta: Optional[Union[str, int, float]] = None,
    delta_color: str = "normal",
    icon: Optional[str] = None,
    help: Optional[str] = None,
    background_gradient: bool = True
):
    """
    메트릭 카드 렌더링
    
    Args:
        label: 메트릭 레이블
        value: 메트릭 값
        delta: 변화량
        delta_color: 변화량 색상 (normal, inverse, off)
        icon: 아이콘
        help: 도움말
        background_gradient: 그라데이션 배경 사용
    """
    # 배경 스타일
    if background_gradient:
        bg_style = f"""
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%);
        color: white;
        """
    else:
        bg_style = f"""
        background: {COLORS['surface']};
        color: {COLORS['text_primary']};
        border: 1px solid #e0e0e0;
        """
    
    # 델타 표시
    delta_html = ""
    if delta is not None:
        delta_icon = "↑" if float(str(delta).replace('%', '').replace(',', '')) > 0 else "↓"
        delta_color_value = {
            'normal': COLORS['success'] if delta_icon == "↑" else COLORS['error'],
            'inverse': COLORS['error'] if delta_icon == "↑" else COLORS['success'],
            'off': COLORS['text_secondary']
        }.get(delta_color, COLORS['text_secondary'])
        
        delta_html = f"""
        <div style="color: {delta_color_value}; font-size: 0.9rem; margin-top: 0.5rem;">
            {delta_icon} {delta}
        </div>
        """
    
    # 카드 렌더링
    card_html = f"""
    <div class="custom-card hover-shadow animate-fadeIn" style="{bg_style}">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <div style="font-size: 0.9rem; opacity: 0.8;">
                    {icon + ' ' if icon else ''}{label}
                </div>
                <div style="font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">
                    {value}
                </div>
                {delta_html}
            </div>
            {f'<div style="font-size: 3rem; opacity: 0.2;">{icon}</div>' if icon else ''}
        </div>
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)
    
    if help:
        st.caption(help)

def render_progress_bar(
    value: float,
    max_value: float = 100,
    label: Optional[str] = None,
    format_string: str = "{:.1f}%",
    color: Optional[str] = None,
    show_value: bool = True,
    height: int = 10
):
    """
    프로그레스 바 렌더링
    
    Args:
        value: 현재 값
        max_value: 최대 값
        label: 레이블
        format_string: 값 포맷 문자열
        color: 프로그레스 바 색상
        show_value: 값 표시 여부
        height: 높이 (픽셀)
    """
    percentage = min(max(value / max_value * 100, 0), 100)
    color = color or COLORS['primary']
    
    progress_html = f"""
    <div class="animate-fadeIn">
        {f'<div style="margin-bottom: 0.5rem; display: flex; justify-content: space-between;"><span>{label}</span><span>{format_string.format(percentage)}</span></div>' if label or show_value else ''}
        <div class="custom-progress" style="height: {height}px;">
            <div class="custom-progress-bar" style="width: {percentage}%; background: {color};"></div>
        </div>
    </div>
    """
    
    st.markdown(progress_html, unsafe_allow_html=True)

def render_circular_progress(
    value: float,
    max_value: float = 100,
    title: str = "",
    size: int = 120,
    color: Optional[str] = None,
    track_color: str = "#E5E7EB",
    thickness: int = 8
):
    """
    원형 프로그레스 표시
    
    Args:
        value: 현재 값
        max_value: 최대 값
        title: 제목
        size: 크기
        color: 색상
        track_color: 트랙 색상
        thickness: 두께
    """
    percentage = min(max(value / max_value * 100, 0), 100)
    color = color or COLORS['primary']
    
    # SVG 원형 프로그레스
    radius = (size - thickness) / 2
    circumference = 2 * 3.14159 * radius
    stroke_dashoffset = circumference * (1 - percentage / 100)
    
    svg_html = f"""
    <div class="animate-fadeIn" style="text-align: center;">
        <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
            <!-- 배경 트랙 -->
            <circle
                cx="{size/2}"
                cy="{size/2}"
                r="{radius}"
                fill="none"
                stroke="{track_color}"
                stroke-width="{thickness}"
            />
            <!-- 프로그레스 -->
            <circle
                cx="{size/2}"
                cy="{size/2}"
                r="{radius}"
                fill="none"
                stroke="{color}"
                stroke-width="{thickness}"
                stroke-dasharray="{circumference}"
                stroke-dashoffset="{stroke_dashoffset}"
                stroke-linecap="round"
                transform="rotate(-90 {size/2} {size/2})"
                style="transition: stroke-dashoffset 0.5s ease-out;"
            />
            <!-- 중앙 텍스트 -->
            <text
                x="{size/2}"
                y="{size/2}"
                text-anchor="middle"
                dominant-baseline="middle"
                font-size="24"
                font-weight="bold"
                fill="{COLORS['text_primary']}"
            >
                {int(percentage)}%
            </text>
        </svg>
        {f'<p style="margin-top: 0.5rem; font-weight: 500;">{title}</p>' if title else ''}
    </div>
    """
    
    st.markdown(svg_html, unsafe_allow_html=True)

# ===========================================================================
# 📈 차트 및 시각화
# ===========================================================================

def create_plotly_chart(
    data: pd.DataFrame,
    chart_type: str,
    x: Optional[str] = None,
    y: Optional[str] = None,
    color: Optional[str] = None,
    title: Optional[str] = None,
    **kwargs
) -> go.Figure:
    """
    Plotly 차트 생성 헬퍼
    
    Args:
        data: 데이터프레임
        chart_type: 차트 타입 (line, bar, scatter, pie, etc.)
        x: X축 컬럼
        y: Y축 컬럼
        color: 색상 구분 컬럼
        title: 차트 제목
        **kwargs: 추가 파라미터
        
    Returns:
        Plotly Figure 객체
    """
    # 차트 타입별 생성
    chart_functions = {
        'line': px.line,
        'bar': px.bar,
        'scatter': px.scatter,
        'pie': px.pie,
        'histogram': px.histogram,
        'box': px.box,
        'violin': px.violin,
        'heatmap': px.imshow,
        'scatter_3d': px.scatter_3d,
        'surface': lambda **kw: go.Figure(data=[go.Surface(**kw)])
    }
    
    if chart_type not in chart_functions:
        raise ValueError(f"지원하지 않는 차트 타입: {chart_type}")
    
    # 차트 생성
    chart_func = chart_functions[chart_type]
    
    # 파라미터 준비
    params = {'data_frame': data}
    if x: params['x'] = x
    if y: params['y'] = y
    if color: params['color'] = color
    if title: params['title'] = title
    params.update(kwargs)
    
    # 특수 차트 처리
    if chart_type == 'pie' and not x and not y:
        params['values'] = params.get('values', data.columns[0])
        params['names'] = params.get('names', data.index)
    
    fig = chart_func(**params)
    
    # 테마 적용
    fig.update_layout(
        template="plotly_white",
        font_family=FONTS.get('body', 'Arial'),
        title_font_size=20,
        title_font_color=COLORS['text_primary'],
        plot_bgcolor=COLORS['surface'],
        paper_bgcolor=COLORS['background'],
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode='x unified' if chart_type in ['line', 'bar'] else 'closest'
    )
    
    # 색상 팔레트 적용
    fig.update_traces(
        marker_color=COLORS['primary'] if chart_type in ['bar', 'scatter'] else None
    )
    
    return fig

def render_chart_container(
    fig: go.Figure,
    use_container_width: bool = True,
    height: Optional[int] = None,
    key: Optional[str] = None
):
    """
    차트 컨테이너 렌더링
    
    Args:
        fig: Plotly Figure
        use_container_width: 컨테이너 너비 사용
        height: 높이
        key: 고유 키
    """
    with st.container():
        st.plotly_chart(
            fig,
            use_container_width=use_container_width,
            height=height,
            key=key,
            config={
                'displaylogo': False,
                'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'chart',
                    'height': 600,
                    'width': 800,
                    'scale': 2
                }
            }
        )

# ===========================================================================
# 📝 폼 및 입력
# ===========================================================================

def render_form_input(
    label: str,
    input_type: str = "text",
    value: Any = None,
    required: bool = False,
    help: Optional[str] = None,
    validation_func: Optional[Callable] = None,
    validation_message: str = "유효하지 않은 입력입니다.",
    key: Optional[str] = None,
    **kwargs
) -> Any:
    """
    폼 입력 필드 렌더링
    
    Args:
        label: 필드 레이블
        input_type: 입력 타입
        value: 기본값
        required: 필수 여부
        help: 도움말
        validation_func: 검증 함수
        validation_message: 검증 실패 메시지
        key: 고유 키
        **kwargs: 추가 파라미터
        
    Returns:
        입력값
    """
    # 필수 표시
    if required:
        label = f"{label} *"
    
    # 입력 타입별 처리
    input_value = None
    
    if input_type == "text":
        input_value = st.text_input(label, value=value, help=help, key=key, **kwargs)
    elif input_type == "number":
        input_value = st.number_input(label, value=value, help=help, key=key, **kwargs)
    elif input_type == "textarea":
        input_value = st.text_area(label, value=value, help=help, key=key, **kwargs)
    elif input_type == "select":
        options = kwargs.pop('options', [])
        input_value = st.selectbox(label, options=options, index=options.index(value) if value in options else 0, help=help, key=key, **kwargs)
    elif input_type == "multiselect":
        options = kwargs.pop('options', [])
        input_value = st.multiselect(label, options=options, default=value or [], help=help, key=key, **kwargs)
    elif input_type == "checkbox":
        input_value = st.checkbox(label, value=bool(value), help=help, key=key, **kwargs)
    elif input_type == "radio":
        options = kwargs.pop('options', [])
        input_value = st.radio(label, options=options, index=options.index(value) if value in options else 0, help=help, key=key, **kwargs)
    elif input_type == "slider":
        input_value = st.slider(label, value=value, help=help, key=key, **kwargs)
    elif input_type == "date":
        input_value = st.date_input(label, value=value, help=help, key=key, **kwargs)
    elif input_type == "time":
        input_value = st.time_input(label, value=value, help=help, key=key, **kwargs)
    elif input_type == "file":
        input_value = st.file_uploader(label, help=help, key=key, **kwargs)
    elif input_type == "color":
        input_value = st.color_picker(label, value=value or "#000000", help=help, key=key, **kwargs)
    
    # 검증
    if validation_func and input_value:
        if not validation_func(input_value):
            show_error(validation_message)
            return None
    
    # 필수 필드 검증
    if required and not input_value:
        show_error(f"{label.replace(' *', '')}은(는) 필수 입력 항목입니다.")
        return None
    
    return input_value

def render_form_section(
    title: str,
    fields: List[Dict[str, Any]],
    columns: int = 1,
    submit_label: str = "제출",
    reset_label: str = "초기화",
    key_prefix: str = ""
) -> Optional[Dict[str, Any]]:
    """
    폼 섹션 렌더링
    
    Args:
        title: 섹션 제목
        fields: 필드 정의 리스트
        columns: 컬럼 수
        submit_label: 제출 버튼 레이블
        reset_label: 초기화 버튼 레이블
        key_prefix: 키 접두사
        
    Returns:
        제출된 데이터 또는 None
    """
    st.subheader(title)
    
    with st.form(f"{key_prefix}_form"):
        # 필드 렌더링
        field_values = {}
        
        if columns > 1:
            cols = st.columns(columns)
            for i, field in enumerate(fields):
                with cols[i % columns]:
                    value = render_form_input(
                        key=f"{key_prefix}_{field.get('name', i)}",
                        **field
                    )
                    if field.get('name'):
                        field_values[field['name']] = value
        else:
            for i, field in enumerate(fields):
                value = render_form_input(
                    key=f"{key_prefix}_{field.get('name', i)}",
                    **field
                )
                if field.get('name'):
                    field_values[field['name']] = value
        
        # 버튼
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            submitted = st.form_submit_button(submit_label, type="primary")
        with col2:
            reset = st.form_submit_button(reset_label)
        
        if submitted:
            # 필수 필드 검증
            for field in fields:
                if field.get('required') and not field_values.get(field.get('name')):
                    show_error("모든 필수 필드를 입력해주세요.")
                    return None
            
            return field_values
        
        if reset:
            st.rerun()
    
    return None

# ===========================================================================
# 🔄 로딩 및 진행 상태
# ===========================================================================

def show_loading(
    message: str = "로딩 중...",
    spinner_type: str = "default"
):
    """
    로딩 표시
    
    Args:
        message: 로딩 메시지
        spinner_type: 스피너 타입
    """
    if spinner_type == "dots":
        spinner_html = """
        <div class="animate-pulse" style="text-align: center; padding: 2rem;">
            <span style="font-size: 2rem;">⏳</span>
            <p>{message}</p>
        </div>
        """
    else:
        with st.spinner(message):
            time.sleep(0.1)  # 최소 표시 시간

@st.cache_data(show_spinner=False)
def with_loading(func):
    """로딩 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with st.spinner("처리 중..."):
            return func(*args, **kwargs)
    return wrapper

# ===========================================================================
# 🗂️ 데이터 테이블
# ===========================================================================

def render_data_table(
    data: pd.DataFrame,
    page_size: int = 10,
    show_index: bool = False,
    enable_search: bool = True,
    enable_sort: bool = True,
    enable_download: bool = True,
    selection_mode: Optional[str] = None,
    key: str = "data_table"
) -> Optional[pd.DataFrame]:
    """
    데이터 테이블 렌더링
    
    Args:
        data: 데이터프레임
        page_size: 페이지 크기
        show_index: 인덱스 표시
        enable_search: 검색 활성화
        enable_sort: 정렬 활성화
        enable_download: 다운로드 활성화
        selection_mode: 선택 모드 (None, 'single', 'multi')
        key: 고유 키
        
    Returns:
        선택된 데이터 (selection_mode가 설정된 경우)
    """
    # 검색
    if enable_search:
        search_term = st.text_input(
            "🔍 검색",
            key=f"{key}_search",
            placeholder="검색어를 입력하세요..."
        )
        
        if search_term:
            # 모든 문자열 컬럼에서 검색
            mask = data.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
            filtered_data = data[mask]
        else:
            filtered_data = data
    else:
        filtered_data = data
    
    # 결과 수 표시
    st.caption(f"총 {len(filtered_data)}개 항목")
    
    # 페이지네이션
    total_pages = (len(filtered_data) - 1) // page_size + 1
    page = st.number_input(
        "페이지",
        min_value=1,
        max_value=max(1, total_pages),
        value=1,
        key=f"{key}_page"
    )
    
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_data = filtered_data.iloc[start_idx:end_idx]
    
    # 테이블 표시
    if selection_mode == 'single':
        selected = st.radio(
            "선택",
            options=page_data.index,
            format_func=lambda x: f"행 {x}",
            key=f"{key}_selection"
        )
        selected_data = page_data.loc[[selected]]
    elif selection_mode == 'multi':
        selected = st.multiselect(
            "선택",
            options=page_data.index,
            format_func=lambda x: f"행 {x}",
            key=f"{key}_selection"
        )
        selected_data = page_data.loc[selected] if selected else pd.DataFrame()
    else:
        selected_data = None
    
    # 데이터 에디터
    edited_data = st.data_editor(
        page_data,
        use_container_width=True,
        hide_index=not show_index,
        disabled=not enable_sort,
        key=f"{key}_editor"
    )
    
    # 다운로드
    if enable_download and not filtered_data.empty:
        col1, col2 = st.columns([1, 3])
        with col1:
            csv = filtered_data.to_csv(index=False)
            st.download_button(
                "📥 CSV 다운로드",
                data=csv,
                file_name=f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key=f"{key}_download"
            )
    
    return selected_data

# ===========================================================================
# 🎨 기타 UI 컴포넌트
# ===========================================================================

def render_empty_state(
    title: str = "데이터가 없습니다",
    message: str = "",
    icon: str = "📭",
    action_label: Optional[str] = None,
    action_callback: Optional[Callable] = None
):
    """
    빈 상태 표시
    
    Args:
        title: 제목
        message: 메시지
        icon: 아이콘
        action_label: 액션 버튼 레이블
        action_callback: 액션 콜백
    """
    empty_html = f"""
    <div class="animate-fadeIn" style="
        text-align: center;
        padding: 3rem;
        background: {COLORS['surface']};
        border-radius: 12px;
        border: 2px dashed #e0e0e0;
    ">
        <div style="font-size: 4rem; margin-bottom: 1rem;">{icon}</div>
        <h3 style="color: {COLORS['text_primary']}; margin-bottom: 0.5rem;">{title}</h3>
        <p style="color: {COLORS['text_secondary']};">{message}</p>
    </div>
    """
    
    st.markdown(empty_html, unsafe_allow_html=True)
    
    if action_label and action_callback:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button(action_label, key="empty_state_action", type="primary"):
                action_callback()

def render_card(
    title: str,
    content: Any,
    footer: Optional[str] = None,
    icon: Optional[str] = None,
    color: Optional[str] = None,
    hover_effect: bool = True
):
    """
    카드 컴포넌트
    
    Args:
        title: 카드 제목
        content: 카드 내용
        footer: 푸터 텍스트
        icon: 아이콘
        color: 테두리 색상
        hover_effect: 호버 효과
    """
    hover_class = "hover-shadow" if hover_effect else ""
    border_style = f"border-left: 4px solid {color};" if color else ""
    
    card_html = f"""
    <div class="custom-card {hover_class} animate-fadeIn" style="{border_style}">
        <h4 style="margin-bottom: 1rem;">
            {icon + ' ' if icon else ''}{title}
        </h4>
        <div style="margin-bottom: 1rem;">
            {content if isinstance(content, str) else ''}
        </div>
        {f'<small style="color: {COLORS["text_secondary"]};">{footer}</small>' if footer else ''}
    </div>
    """
    
    if isinstance(content, str):
        st.markdown(card_html, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="custom-card {hover_class} animate-fadeIn" style="{border_style}">
            <h4 style="margin-bottom: 1rem;">
                {icon + ' ' if icon else ''}{title}
            </h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.write(content)
        
        if footer:
            st.caption(footer)

def render_tabs(
    tabs: List[str],
    default_index: int = 0,
    key: str = "tabs"
) -> int:
    """
    탭 컴포넌트
    
    Args:
        tabs: 탭 이름 리스트
        default_index: 기본 선택 인덱스
        key: 고유 키
        
    Returns:
        선택된 탭 인덱스
    """
    selected_tab = st.tabs(tabs)
    return tabs.index(selected_tab) if selected_tab in tabs else default_index

def render_modal(
    title: str,
    content: Any,
    show: bool = False,
    key: str = "modal"
):
    """
    모달 다이얼로그
    
    Args:
        title: 모달 제목
        content: 모달 내용
        show: 표시 여부
        key: 고유 키
    """
    if show:
        modal_html = f"""
        <div class="modal-backdrop" style="
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.5);
            z-index: 9998;
            display: flex;
            align-items: center;
            justify-content: center;
        ">
            <div class="modal-content animate-fadeIn" style="
                background: white;
                border-radius: 12px;
                padding: 2rem;
                max-width: 500px;
                width: 90%;
                max-height: 80vh;
                overflow-y: auto;
            ">
                <h3>{title}</h3>
                <div style="margin: 1rem 0;">
                    {content if isinstance(content, str) else ''}
                </div>
            </div>
        </div>
        """
        
        if isinstance(content, str):
            st.markdown(modal_html, unsafe_allow_html=True)
        else:
            with st.container():
                st.subheader(title)
                st.write(content)

def render_badge(
    text: str,
    color: str = "primary",
    size: str = "normal"
):
    """
    배지/태그 렌더링
    
    Args:
        text: 배지 텍스트
        color: 색상 (primary, secondary, success, warning, error, info)
        size: 크기 (small, normal, large)
    """
    # 색상 매핑
    bg_color = COLORS.get(color, COLORS['primary'])
    text_color = "white" if color in ['primary', 'secondary', 'error'] else COLORS['text_primary']
    
    # 크기 매핑
    size_styles = {
        'small': 'font-size: 0.75rem; padding: 0.25rem 0.5rem;',
        'normal': 'font-size: 0.875rem; padding: 0.375rem 0.75rem;',
        'large': 'font-size: 1rem; padding: 0.5rem 1rem;'
    }
    
    badge_html = f"""
    <span class="animate-fadeIn" style="
        display: inline-block;
        background: {bg_color};
        color: {text_color};
        border-radius: 20px;
        {size_styles.get(size, size_styles['normal'])}
        font-weight: 500;
    ">{text}</span>
    """
    
    st.markdown(badge_html, unsafe_allow_html=True)

def render_divider(
    style: str = "solid",
    color: Optional[str] = None,
    margin: str = "1rem"
):
    """
    구분선 렌더링
    
    Args:
        style: 선 스타일 (solid, dashed, dotted)
        color: 색상
        margin: 여백
    """
    color = color or COLORS['text_secondary'] + '40'  # 40% 투명도
    
    st.markdown(f"""
    <hr style="
        border: none;
        border-top: 1px {style} {color};
        margin: {margin} 0;
    ">
    """, unsafe_allow_html=True)

# ===========================================================================
# 🛠️ 유틸리티 함수
# ===========================================================================

def get_default_avatar():
    """기본 아바타 이미지 URL"""
    avatar_svg = """
    <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <circle cx="50" cy="50" r="40" fill="#7C3AED"/>
        <text x="50" y="50" text-anchor="middle" dy=".3em" fill="white" font-size="40">👤</text>
    </svg>
    """
    return f"data:image/svg+xml,{avatar_svg}"

def format_datetime(
    dt: datetime,
    format: str = "relative"
) -> str:
    """
    날짜/시간 포맷팅
    
    Args:
        dt: datetime 객체
        format: 포맷 타입 (relative, date, datetime, time)
        
    Returns:
        포맷된 문자열
    """
    if format == "relative":
        now = datetime.now()
        delta = now - dt
        
        if delta.days > 365:
            return f"{delta.days // 365}년 전"
        elif delta.days > 30:
            return f"{delta.days // 30}개월 전"
        elif delta.days > 7:
            return f"{delta.days // 7}주 전"
        elif delta.days > 0:
            return f"{delta.days}일 전"
        elif delta.seconds > 3600:
            return f"{delta.seconds // 3600}시간 전"
        elif delta.seconds > 60:
            return f"{delta.seconds // 60}분 전"
        else:
            return "방금 전"
    elif format == "date":
        return dt.strftime("%Y-%m-%d")
    elif format == "datetime":
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    elif format == "time":
        return dt.strftime("%H:%M:%S")
    else:
        return dt.strftime(format)

def format_number(
    value: Union[int, float],
    format: str = "comma"
) -> str:
    """
    숫자 포맷팅
    
    Args:
        value: 숫자 값
        format: 포맷 타입 (comma, percent, currency, scientific)
        
    Returns:
        포맷된 문자열
    """
    if format == "comma":
        return f"{value:,}"
    elif format == "percent":
        return f"{value:.1%}"
    elif format == "currency":
        return f"₩{value:,.0f}"
    elif format == "scientific":
        return f"{value:.2e}"
    else:
        return str(value)

def create_download_link(
    data: Any,
    filename: str,
    link_text: str = "다운로드"
) -> str:
    """
    다운로드 링크 생성
    
    Args:
        data: 다운로드할 데이터
        filename: 파일명
        link_text: 링크 텍스트
        
    Returns:
        HTML 링크
    """
    if isinstance(data, pd.DataFrame):
        data = data.to_csv(index=False)
    elif isinstance(data, dict):
        data = json.dumps(data, ensure_ascii=False, indent=2)
    
    b64 = base64.b64encode(data.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def validate_email(email: str) -> bool:
    """이메일 유효성 검증"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password: str) -> Tuple[bool, str]:
    """
    비밀번호 유효성 검증
    
    Returns:
        (유효여부, 메시지)
    """
    if len(password) < 8:
        return False, "비밀번호는 8자 이상이어야 합니다."
    if not any(c.isupper() for c in password):
        return False, "대문자가 포함되어야 합니다."
    if not any(c.islower() for c in password):
        return False, "소문자가 포함되어야 합니다."
    if not any(c.isdigit() for c in password):
        return False, "숫자가 포함되어야 합니다."
    if not any(c in "!@#$%^&*()_+-=" for c in password):
        return False, "특수문자가 포함되어야 합니다."
    
    return True, "유효한 비밀번호입니다."

# ===========================================================================
# 🎯 초기화
# ===========================================================================

def init_ui():
    """UI 시스템 초기화"""
    # 테마 적용
    apply_custom_theme()
    
    # 세션 상태 초기화
    if 'ui_initialized' not in st.session_state:
        st.session_state.ui_initialized = True
        st.session_state.show_notifications = False
        st.session_state.theme = 'light'
    
    logger.info("UI system initialized")

# 자동 초기화
init_ui()
