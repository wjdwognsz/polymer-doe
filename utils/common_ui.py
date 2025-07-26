"""
utils/common_ui.py
재사용 가능한 UI 컴포넌트 라이브러리

Universal DOE Platform의 시각적 일관성과 코드 재사용성을 담당합니다.
모든 페이지에서 사용되는 공통 UI 컴포넌트들을 제공합니다.
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
    from config.theme_config import COLORS, FONTS, LAYOUT, COMPONENTS
except ImportError:
    # 기본값 설정
    COLORS = {
        'primary': '#7C3AED',
        'secondary': '#F59E0B', 
        'success': '#10B981',
        'danger': '#EF4444',
        'warning': '#F59E0B',
        'info': '#3B82F6',
        'dark': '#1F2937',
        'light': '#F3F4F6',
        'muted': '#6B7280'
    }

try:
    from config.app_config import APP_NAME, APP_DESCRIPTION, UI_CONFIG, AI_EXPLANATION_CONFIG
except ImportError:
    APP_NAME = "Universal DOE Platform"
    APP_DESCRIPTION = "AI 기반 고분자 실험 설계 플랫폼"
    UI_CONFIG = {'theme': {'default': 'light'}}
    AI_EXPLANATION_CONFIG = {
        'default_mode': 'auto',
        'auto_mode_rules': {
            'beginner': 'detailed',
            'intermediate': 'balanced',
            'expert': 'concise'
        }
    }

# error_handler는 아직 구현되지 않았으므로 조건부 처리
try:
    from utils.error_handler import handle_ui_error
except ImportError:
    def handle_ui_error(func):
        """에러 핸들러 데코레이터 (임시)"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                st.error(f"오류가 발생했습니다: {str(e)}")
                return None
        return wrapper

# 로깅 설정
logger = logging.getLogger(__name__)

# ============================================================================
# 상수 정의
# ============================================================================

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
    'refresh': '🔄',
    'share': '🔗',
    'copy': '📋',
    
    # AI
    'ai': '🤖',
    'brain': '🧠',
    'magic': '✨',
    'thinking': '🤔'
}

# 애니메이션 CSS
ANIMATIONS_CSS = """
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

/* AI 응답 컨테이너 스타일 */
.ai-response-container {
    background: linear-gradient(to right, #f3f4f6, #ffffff);
    border-left: 4px solid #7C3AED;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}

.ai-detail-section {
    background: #f9fafb;
    border-radius: 0.375rem;
    padding: 0.75rem;
    margin-top: 0.5rem;
}

/* 메트릭 카드 스타일 */
.metric-card {
    background: white;
    border-radius: 0.5rem;
    padding: 1.5rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
}

.metric-card:hover {
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transform: translateY(-2px);
}

/* 데이터 테이블 스타일 */
.dataframe {
    font-size: 0.875rem !important;
}

.dataframe th {
    background-color: #f3f4f6 !important;
    font-weight: 600 !important;
}

.dataframe tr:hover {
    background-color: #f9fafb !important;
}

/* 프로그레스 바 스타일 */
.progress-container {
    background: #e5e7eb;
    border-radius: 9999px;
    height: 0.5rem;
    overflow: hidden;
}

.progress-bar {
    background: linear-gradient(to right, #7C3AED, #a78bfa);
    height: 100%;
    transition: width 0.3s ease;
}

/* 빈 상태 스타일 */
.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: #6b7280;
}

.empty-state-icon {
    font-size: 3rem;
    opacity: 0.5;
    margin-bottom: 1rem;
}

/* 오프라인 인디케이터 */
.offline-indicator {
    position: fixed;
    top: 1rem;
    right: 1rem;
    background: #fef3c7;
    color: #92400e;
    padding: 0.5rem 1rem;
    border-radius: 0.375rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    z-index: 1000;
}
</style>
"""

# ============================================================================
# 페이지 설정
# ============================================================================

def setup_page_config(
    page_title: Optional[str] = None,
    page_icon: Optional[str] = None,
    layout: Literal["centered", "wide"] = "wide",
    initial_sidebar_state: Literal["auto", "expanded", "collapsed"] = "expanded"
):
    """
    Streamlit 페이지 기본 설정
    
    Args:
        page_title: 페이지 제목
        page_icon: 페이지 아이콘
        layout: 레이아웃
        initial_sidebar_state: 사이드바 초기 상태
    """
    st.set_page_config(
        page_title=page_title or APP_NAME,
        page_icon=page_icon or "🧬",
        layout=layout,
        initial_sidebar_state=initial_sidebar_state,
        menu_items={
            'Get Help': 'https://github.com/your-repo/polymer-doe/wiki',
            'Report a bug': 'https://github.com/your-repo/polymer-doe/issues',
            'About': APP_DESCRIPTION
        }
    )
    
    # CSS 적용
    st.markdown(ANIMATIONS_CSS, unsafe_allow_html=True)
    
    # 세션 상태 초기화
    _initialize_session_state()


def _initialize_session_state():
    """세션 상태 초기화"""
    defaults = {
        'show_ai_details': 'auto',  # AI 상세 설명 표시 모드
        'ui_animations': True,      # 애니메이션 활성화
        'compact_mode': False,      # 컴팩트 모드
        'help_tooltips': True,      # 도움말 툴팁 표시
        'theme': 'light',          # 테마
        'last_interaction': datetime.now()  # 마지막 상호작용 시간
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================================
# 헤더/레이아웃 컴포넌트
# ============================================================================

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
        actions: 우측 액션 버튼들
    """
    # 브레드크럼
    if breadcrumb:
        breadcrumb_html = " › ".join(breadcrumb)
        st.markdown(
            f'<div style="color: {COLORS.get("muted", "#6B7280")}; '
            f'font-size: 0.875rem; margin-bottom: 0.5rem;">'
            f'{breadcrumb_html}</div>',
            unsafe_allow_html=True
        )
    
    # 헤더 컨테이너
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # 제목
        if icon:
            st.markdown(f"# {icon} {title}")
        else:
            st.markdown(f"# {title}")
        
        # 부제목
        if subtitle:
            st.markdown(f"_{subtitle}_")
    
    with col2:
        # 액션 버튼
        if actions:
            for action in actions:
                if st.button(
                    action.get('label', ''),
                    key=action.get('key'),
                    type=action.get('type', 'secondary'),
                    disabled=action.get('disabled', False),
                    use_container_width=True
                ):
                    if 'callback' in action:
                        action['callback']()
    
    st.divider()


def render_sidebar_menu(menu_items: List[Dict[str, Any]], active_key: Optional[str] = None):
    """
    사이드바 메뉴 렌더링
    
    Args:
        menu_items: 메뉴 아이템 리스트
        active_key: 활성 메뉴 키
    """
    with st.sidebar:
        st.markdown("### 🧬 Universal DOE")
        st.divider()
        
        for item in menu_items:
            is_active = item.get('key') == active_key
            
            # 메뉴 아이템 스타일
            if is_active:
                st.markdown(
                    f"**{item.get('icon', '')} {item.get('label', '')}**",
                    help=item.get('help', '')
                )
            else:
                if st.button(
                    f"{item.get('icon', '')} {item.get('label', '')}",
                    key=f"menu_{item.get('key')}",
                    use_container_width=True,
                    help=item.get('help', '')
                ):
                    if 'callback' in item:
                        item['callback']()


def render_offline_indicator():
    """오프라인 인디케이터 표시"""
    if st.session_state.get('offline_mode', False):
        st.markdown(
            '<div class="offline-indicator animate-pulse">'
            '🔌 오프라인 모드'
            '</div>',
            unsafe_allow_html=True
        )


# ============================================================================
# AI 응답 컴포넌트 (프로젝트 핵심 기능)
# ============================================================================

def render_ai_response(
    response: Dict[str, Any],
    response_type: str = "general",
    show_confidence: bool = True,
    allow_feedback: bool = True,
    key: Optional[str] = None
):
    """
    AI 응답 표시 (상세 설명 토글 기능 포함)
    
    이 함수는 프로젝트의 핵심 요구사항인 AI 투명성 원칙을 구현합니다.
    사용자 레벨과 무관하게 누구나 AI의 추론 과정을 볼 수 있습니다.
    
    Args:
        response: AI 응답 딕셔너리
            - main: 핵심 답변 (필수)
            - details: 상세 설명 딕셔너리 (선택)
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
    # 응답 유효성 검사
    if not response or 'main' not in response:
        st.error("AI 응답이 올바르지 않습니다.")
        return
    
    # 상세 설명 토글 상태 관리
    detail_key = f"ai_details_{key}" if key else "ai_details_global"
    
    # 기본 표시 모드 결정
    if detail_key not in st.session_state:
        user_level = st.session_state.get('user', {}).get('level', 'beginner')
        auto_mode = AI_EXPLANATION_CONFIG.get('auto_mode_rules', {}).get(
            user_level,
            AI_EXPLANATION_CONFIG.get('default_mode', 'balanced')
        )
        st.session_state[detail_key] = auto_mode == 'detailed'
    
    # AI 응답 컨테이너
    with st.container():
        st.markdown('<div class="ai-response-container">', unsafe_allow_html=True)
        
        # 헤더
        col1, col2 = st.columns([5, 1])
        
        with col1:
            st.markdown(f"### {ICONS['ai']} AI 응답")
        
        with col2:
            # 상세 설명 토글 버튼
            if st.button(
                "🔍 상세" if not st.session_state[detail_key] else "📌 간단",
                key=f"toggle_{detail_key}",
                help="AI의 추론 과정과 상세 설명을 확인하세요"
            ):
                st.session_state[detail_key] = not st.session_state[detail_key]
        
        # 메인 응답
        st.markdown(f"**{response['main']}**")
        
        # 신뢰도 표시
        if show_confidence and 'details' in response and 'confidence' in response['details']:
            confidence = response['details']['confidence']
            if isinstance(confidence, (int, float)):
                confidence_pct = int(confidence * 100) if confidence <= 1 else int(confidence)
                confidence_color = (
                    'green' if confidence_pct >= 80 
                    else 'orange' if confidence_pct >= 60 
                    else 'red'
                )
                st.markdown(
                    f"<div style='margin-top: 0.5rem;'>"
                    f"신뢰도: <span style='color: {confidence_color}; font-weight: bold;'>"
                    f"{confidence_pct}%</span></div>",
                    unsafe_allow_html=True
                )
        
        # 상세 설명 (토글 상태에 따라)
        if st.session_state[detail_key] and 'details' in response:
            st.markdown("---")
            details = response['details']
            
            # 추론 과정
            if 'reasoning' in details:
                with st.expander("🧠 추론 과정", expanded=True):
                    st.markdown(details['reasoning'])
            
            # 대안
            if 'alternatives' in details:
                with st.expander("💡 다른 옵션", expanded=False):
                    st.markdown(details['alternatives'])
            
            # 이론적 배경
            if 'background' in details:
                with st.expander("📚 이론적 배경", expanded=False):
                    st.markdown(details['background'])
            
            # 한계점
            if 'limitations' in details:
                with st.expander("⚠️ 주의사항 및 한계", expanded=False):
                    st.markdown(details['limitations'])
        
        # 피드백 섹션
        if allow_feedback:
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 3])
            
            with col1:
                if st.button("👍", key=f"like_{key}", help="도움이 되었어요"):
                    _record_feedback(key, 'like')
                    st.success("피드백 감사합니다!")
            
            with col2:
                if st.button("👎", key=f"dislike_{key}", help="개선이 필요해요"):
                    _record_feedback(key, 'dislike')
                    st.info("피드백을 반영하여 개선하겠습니다.")
        
        st.markdown('</div>', unsafe_allow_html=True)


def _record_feedback(response_key: str, feedback_type: str):
    """AI 응답 피드백 기록"""
    feedback_data = {
        'response_key': response_key,
        'feedback_type': feedback_type,
        'timestamp': datetime.now().isoformat(),
        'user': st.session_state.get('user', {}).get('id', 'anonymous')
    }
    
    # 피드백 저장 (추후 구현)
    logger.info(f"AI feedback recorded: {feedback_data}")


# ============================================================================
# 메트릭/데이터 표시 컴포넌트
# ============================================================================

def render_metric_card(
    label: str,
    value: Union[str, int, float],
    delta: Optional[Union[str, int, float]] = None,
    delta_color: Literal["normal", "inverse", "off"] = "normal",
    icon: Optional[str] = None,
    help: Optional[str] = None,
    background_color: Optional[str] = None
):
    """
    메트릭 카드 렌더링
    
    Args:
        label: 레이블
        value: 값
        delta: 변화량
        delta_color: 델타 색상 모드
        icon: 아이콘
        help: 도움말
        background_color: 배경색
    """
    # 배경 스타일
    bg_style = f"background-color: {background_color};" if background_color else ""
    
    # 델타 색상 매핑
    delta_colors = {
        "normal": COLORS.get('success', '#10B981') if delta and str(delta).startswith('+') else COLORS.get('danger', '#EF4444'),
        "inverse": COLORS.get('danger', '#EF4444') if delta and str(delta).startswith('+') else COLORS.get('success', '#10B981'),
        "off": COLORS.get('muted', '#6B7280')
    }
    
    # HTML 렌더링
    html = f"""
    <div class="metric-card" style="{bg_style}">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="flex: 1;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    {f'<span style="font-size: 1.5rem;">{icon}</span>' if icon else ''}
                    <p style="color: {COLORS.get('muted', '#6B7280')}; margin: 0; font-size: 0.875rem;">
                        {label}
                    </p>
                </div>
                <h2 style="margin: 0.5rem 0; color: {COLORS.get('dark', '#1F2937')};">
                    {value}
                </h2>
                {f'<p style="color: {delta_colors.get(delta_color, COLORS.get("muted", "#6B7280"))}; margin: 0; font-size: 0.875rem;">{delta}</p>' if delta else ''}
            </div>
            {f'<span title="{help}" style="cursor: help; color: {COLORS.get("muted", "#6B7280")};">{ICONS["help"]}</span>' if help else ''}
        </div>
    </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)


def render_data_table(
    data: pd.DataFrame,
    title: Optional[str] = None,
    show_index: bool = False,
    enable_search: bool = True,
    enable_download: bool = True,
    height: Optional[int] = None,
    key: Optional[str] = None
):
    """
    데이터 테이블 렌더링
    
    Args:
        data: 데이터프레임
        title: 제목
        show_index: 인덱스 표시 여부
        enable_search: 검색 기능
        enable_download: 다운로드 기능
        height: 높이
        key: 고유 키
    """
    # 제목
    if title:
        st.markdown(f"### {title}")
    
    # 검색 기능
    if enable_search and len(data) > 10:
        search_key = f"search_{key}" if key else "search_table"
        search_term = st.text_input(
            "🔍 검색",
            key=search_key,
            placeholder="검색어를 입력하세요..."
        )
        
        if search_term:
            # 모든 컬럼에서 검색
            mask = data.astype(str).apply(
                lambda x: x.str.contains(search_term, case=False, na=False)
            ).any(axis=1)
            filtered_data = data[mask]
        else:
            filtered_data = data
    else:
        filtered_data = data
    
    # 테이블 표시
    st.dataframe(
        filtered_data,
        hide_index=not show_index,
        height=height,
        use_container_width=True
    )
    
    # 다운로드 버튼
    if enable_download:
        csv = filtered_data.to_csv(index=show_index, encoding='utf-8-sig')
        st.download_button(
            label=f"📥 CSV 다운로드 ({len(filtered_data)}행)",
            data=csv,
            file_name=f"{title or 'data'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv',
            key=f"download_{key}" if key else "download_table"
        )


def render_progress(
    current: Union[int, float],
    total: Union[int, float],
    label: Optional[str] = None,
    show_percentage: bool = True,
    color: Optional[str] = None
):
    """
    진행률 바 렌더링
    
    Args:
        current: 현재 값
        total: 전체 값
        label: 레이블
        show_percentage: 백분율 표시
        color: 색상
    """
    # 백분율 계산
    percentage = min(100, max(0, (current / total * 100) if total > 0 else 0))
    
    # 레이블
    if label:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(label)
        with col2:
            if show_percentage:
                st.markdown(f"**{percentage:.0f}%**", unsafe_allow_html=True)
    
    # 진행률 바
    progress_color = color or COLORS.get('primary', '#7C3AED')
    st.markdown(
        f"""
        <div class="progress-container">
            <div class="progress-bar" style="width: {percentage}%; background: {progress_color};"></div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_circular_progress(
    value: float,
    max_value: float = 100,
    title: str = "",
    size: int = 120,
    color: Optional[str] = None
):
    """
    원형 진행률 표시
    
    Args:
        value: 현재 값
        max_value: 최대 값
        title: 제목
        size: 크기
        color: 색상
    """
    percentage = min(100, max(0, (value / max_value * 100)))
    color = color or COLORS.get('primary', '#7C3AED')
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percentage,
        title={'text': title},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 100], 'color': COLORS.get('light', '#F3F4F6')}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=size,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# 알림/메시지 컴포넌트
# ============================================================================

def show_notification(
    message: str,
    type: Literal["success", "error", "warning", "info"] = "info",
    icon: Optional[str] = None,
    duration: Optional[int] = None
):
    """
    알림 메시지 표시
    
    Args:
        message: 메시지
        type: 알림 유형
        icon: 커스텀 아이콘
        duration: 표시 시간 (초)
    """
    # 아이콘 설정
    if not icon:
        icon = ICONS.get(type, ICONS['info'])
    
    # Streamlit 네이티브 알림 사용
    if type == "success":
        st.success(f"{icon} {message}")
    elif type == "error":
        st.error(f"{icon} {message}")
    elif type == "warning":
        st.warning(f"{icon} {message}")
    else:
        st.info(f"{icon} {message}")
    
    # 자동 숨김 (duration이 설정된 경우)
    if duration:
        time.sleep(duration)
        st.empty()


def show_success(message: str, icon: str = None):
    """성공 메시지"""
    show_notification(message, "success", icon)


def show_error(message: str, icon: str = None):
    """에러 메시지"""
    show_notification(message, "error", icon)


def show_warning(message: str, icon: str = None):
    """경고 메시지"""
    show_notification(message, "warning", icon)


def show_info(message: str, icon: str = None):
    """정보 메시지"""
    show_notification(message, "info", icon)


# ============================================================================
# 입력 컴포넌트
# ============================================================================

def create_validated_input(
    label: str,
    input_type: Literal["text", "number", "email", "password"] = "text",
    placeholder: Optional[str] = None,
    help: Optional[str] = None,
    required: bool = False,
    validation_func: Optional[Callable] = None,
    error_message: str = "입력값이 올바르지 않습니다.",
    key: Optional[str] = None,
    **kwargs
) -> Optional[Any]:
    """
    검증 기능이 있는 입력 필드
    
    Args:
        label: 레이블
        input_type: 입력 유형
        placeholder: 플레이스홀더
        help: 도움말
        required: 필수 여부
        validation_func: 검증 함수
        error_message: 에러 메시지
        key: 고유 키
        **kwargs: 추가 인자
        
    Returns:
        검증된 입력값 또는 None
    """
    # 필수 표시
    if required:
        label = f"{label} *"
    
    # 입력 필드 생성
    if input_type == "text":
        value = st.text_input(
            label,
            placeholder=placeholder,
            help=help,
            key=key,
            **kwargs
        )
    elif input_type == "number":
        value = st.number_input(
            label,
            help=help,
            key=key,
            **kwargs
        )
    elif input_type == "email":
        value = st.text_input(
            label,
            placeholder=placeholder or "user@example.com",
            help=help,
            key=key,
            **kwargs
        )
    elif input_type == "password":
        value = st.text_input(
            label,
            type="password",
            placeholder=placeholder,
            help=help,
            key=key,
            **kwargs
        )
    else:
        value = None
    
    # 검증
    if value:
        # 기본 검증
        if input_type == "email" and "@" not in value:
            st.error("올바른 이메일 주소를 입력하세요.")
            return None
        
        # 커스텀 검증
        if validation_func and not validation_func(value):
            st.error(error_message)
            return None
    
    # 필수 입력 체크
    if required and not value:
        st.error(f"{label.replace(' *', '')}은(는) 필수 입력 항목입니다.")
        return None
    
    return value


def create_file_uploader(
    label: str,
    file_types: List[str],
    accept_multiple: bool = False,
    help: Optional[str] = None,
    max_size_mb: int = 200,
    key: Optional[str] = None
) -> Optional[Union[Any, List[Any]]]:
    """
    파일 업로더 컴포넌트
    
    Args:
        label: 레이블
        file_types: 허용 파일 타입
        accept_multiple: 다중 파일 허용
        help: 도움말
        max_size_mb: 최대 파일 크기 (MB)
        key: 고유 키
        
    Returns:
        업로드된 파일
    """
    # 도움말 텍스트 생성
    if not help:
        help = f"허용 파일: {', '.join(file_types)} (최대 {max_size_mb}MB)"
    
    # 파일 업로더
    uploaded_files = st.file_uploader(
        label,
        type=file_types,
        accept_multiple_files=accept_multiple,
        help=help,
        key=key
    )
    
    # 파일 크기 검증
    if uploaded_files:
        if accept_multiple:
            valid_files = []
            for file in uploaded_files:
                if file.size > max_size_mb * 1024 * 1024:
                    st.error(f"{file.name}: 파일 크기가 {max_size_mb}MB를 초과합니다.")
                else:
                    valid_files.append(file)
            return valid_files if valid_files else None
        else:
            if uploaded_files.size > max_size_mb * 1024 * 1024:
                st.error(f"파일 크기가 {max_size_mb}MB를 초과합니다.")
                return None
            return uploaded_files
    
    return None


def create_date_range_picker(
    label: str = "날짜 범위",
    default_days: int = 7,
    help: Optional[str] = None,
    key: Optional[str] = None
) -> Tuple[datetime, datetime]:
    """
    날짜 범위 선택기
    
    Args:
        label: 레이블
        default_days: 기본 일수
        help: 도움말
        key: 고유 키
        
    Returns:
        (시작일, 종료일) 튜플
    """
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            f"{label} - 시작",
            value=datetime.now() - timedelta(days=default_days),
            help=help,
            key=f"{key}_start" if key else "date_start"
        )
    
    with col2:
        end_date = st.date_input(
            f"{label} - 종료",
            value=datetime.now(),
            help=help,
            key=f"{key}_end" if key else "date_end"
        )
    
    # 날짜 유효성 검사
    if start_date > end_date:
        st.error("시작일이 종료일보다 늦을 수 없습니다.")
        return None, None
    
    return start_date, end_date


# ============================================================================
# 상태 표시 컴포넌트
# ============================================================================

def render_empty_state(
    message: str = "데이터가 없습니다",
    icon: str = "📭",
    suggestion: Optional[str] = None
):
    """
    빈 상태 표시
    
    Args:
        message: 메시지
        icon: 아이콘
        suggestion: 제안 사항
    """
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="empty-state-icon">{icon}</div>
            <h3>{message}</h3>
            {f'<p style="color: #6b7280;">{suggestion}</p>' if suggestion else ''}
        </div>
        """,
        unsafe_allow_html=True
    )


@handle_ui_error
def show_loading(message: str = "로딩 중...", spinner: bool = True):
    """
    로딩 상태 표시
    
    Args:
        message: 로딩 메시지
        spinner: 스피너 표시 여부
    """
    if spinner:
        with st.spinner(message):
            yield
    else:
        placeholder = st.empty()
        placeholder.info(f"{ICONS['loading']} {message}")
        yield
        placeholder.empty()


# ============================================================================
# 유틸리티 함수
# ============================================================================

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
        return dt.strftime("%Y년 %m월 %d일")
    elif format == "time":
        return dt.strftime("%H:%M:%S")
    elif format == "relative":
        # 상대 시간 계산
        now = datetime.now()
        diff = now - dt
        
        if diff.days > 7:
            return dt.strftime("%Y-%m-%d")
        elif diff.days > 0:
            return f"{diff.days}일 전"
        elif diff.seconds > 3600:
            return f"{diff.seconds // 3600}시간 전"
        elif diff.seconds > 60:
            return f"{diff.seconds // 60}분 전"
        else:
            return "방금 전"
    
    return str(dt)


def format_number(
    number: Union[int, float],
    decimals: int = 2,
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
            COLORS.get('primary', '#7C3AED'),
            COLORS.get('secondary', '#F59E0B'),
            COLORS.get('success', '#10B981'),
            COLORS.get('danger', '#EF4444'),
            COLORS.get('warning', '#F59E0B'),
            COLORS.get('info', '#3B82F6')
        ],
        'gradient': [
            '#1E88E5', '#1976D2', '#1565C0', '#0D47A1', '#01579B'
        ],
        'polymer': [
            '#E91E63', '#9C27B0', '#673AB7', '#3F51B5', '#2196F3'
        ],
        'category': px.colors.qualitative.Set3,
        'sequential': px.colors.sequential.Blues,
        'diverging': px.colors.diverging.RdBu
    }
    
    return palettes.get(name, palettes['default'])


# ============================================================================
# Export
# ============================================================================

__all__ = [
    # 페이지 설정
    'setup_page_config',
    
    # 헤더/레이아웃
    'render_header',
    'render_sidebar_menu',
    'render_offline_indicator',
    
    # AI 컴포넌트
    'render_ai_response',
    
    # 메트릭/데이터
    'render_metric_card',
    'render_data_table',
    'render_progress',
    'render_circular_progress',
    
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
