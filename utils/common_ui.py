"""
🎨 Universal DOE Platform - 공통 UI 컴포넌트
================================================================================
재사용 가능한 UI 컴포넌트 라이브러리
일관된 디자인 시스템, 테마 지원, 접근성, AI 투명성 원칙 구현
고분자 과학 특화 기능 및 벤치마크 비교 기능 포함
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
from functools import wraps, lru_cache
import logging
from pathlib import Path
from PIL import Image
import io
import re
import hashlib

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
        'text_secondary': '#757575',
        'light': '#F3F4F6',
        'dark': '#1a1a1a'
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
    'collapse': '🔼',
    
    # 고분자 관련
    'polymer': '🧬',
    'solvent': '💧',
    'fiber': '🕸️',
    'coating': '🎨',
    'nanoparticle': '⚪'
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
/* CSS 변수 정의 */
:root {
    --primary-color: #1E88E5;
    --secondary-color: #43A047;
    --accent-color: #E53935;
    --warning-color: #FB8C00;
    --info-color: #00ACC1;
    --success-color: #43A047;
    --error-color: #E53935;
    --background-color: #FAFAFA;
    --surface-color: #FFFFFF;
    --text-primary: #212121;
    --text-secondary: #757575;
}

/* 다크 모드 */
[data-theme="dark"] {
    --background-color: #1a1a1a;
    --surface-color: #2d2d2d;
    --text-primary: #ffffff;
    --text-secondary: #b0b0b0;
}

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
    background: var(--surface-color);
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
    border-left: 4px solid var(--primary-color);
    padding-left: 1rem;
    margin: 1rem 0;
}

.ai-detail-section {
    background: var(--background-color);
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
    background: var(--primary-color);
    color: white;
    animation: pulse 2s infinite;
}

.progress-step.completed {
    background: var(--success-color);
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

/* 오프라인 상태 표시 */
.offline-indicator {
    position: fixed;
    top: 60px;
    right: 20px;
    background: var(--warning-color);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-size: 0.875rem;
    z-index: 999;
    animation: pulse 3s infinite;
}

/* 벤치마크 비교 스타일 */
.benchmark-card {
    background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%);
    border: 1px solid #667eea40;
    border-radius: 12px;
    padding: 1.5rem;
}

.percentile-indicator {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
}

/* 반응형 디자인 */
@media (max-width: 768px) {
    .custom-card {
        padding: 1rem;
    }
    
    .hide-mobile {
        display: none;
    }
    
    .offline-indicator {
        top: auto;
        bottom: 20px;
        right: 20px;
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

/* 키보드 네비게이션 */
.accessible-button:focus {
    outline: 3px solid var(--primary-color);
    outline-offset: 2px;
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

@lru_cache(maxsize=128)
def get_cached_icon(icon_name: str) -> str:
    """아이콘 캐싱"""
    return ICONS.get(icon_name, '')

# ===========================================================================
# 🌐 오프라인/온라인 상태 컴포넌트
# ===========================================================================

def render_connection_status():
    """네트워크 연결 상태 표시"""
    is_offline = st.session_state.get('offline_mode', False)
    
    if is_offline:
        st.markdown(
            '<div class="offline-indicator">🔌 오프라인 모드</div>',
            unsafe_allow_html=True
        )

def render_theme_toggle():
    """테마 전환 토글"""
    col1, col2 = st.columns([4, 1])
    
    with col2:
        current_theme = st.session_state.get('theme', 'light')
        
        if st.button(
            "🌙" if current_theme == 'light' else "☀️",
            key="theme_toggle",
            help="다크모드 전환"
        ):
            new_theme = 'dark' if current_theme == 'light' else 'light'
            st.session_state.theme = new_theme
            
            # CSS 업데이트
            theme_css = f'<div data-theme="{new_theme}"></div>'
            st.markdown(theme_css, unsafe_allow_html=True)
            
            st.rerun()

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
        st.markdown('<div class="ai-response-container">', unsafe_allow_html=True)
        
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
                        st.markdown('<div class="ai-detail-section">', unsafe_allow_html=True)
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
    # 상단 바
    col1, col2, col3 = st.columns([6, 1, 1])
    
    with col2:
        render_theme_toggle()
    
    with col3:
        render_connection_status()
    
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
# 🧪 고분자 과학 전용 컴포넌트
# ===========================================================================

def render_solvent_system_card(
    system: Dict[str, Any],
    show_details: bool = True,
    key: Optional[str] = None
):
    """용매 시스템 카드"""
    with st.container():
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        
        # 헤더
        col1, col2 = st.columns([3, 1])
        with col1:
            components = system.get('components', [])
            if len(components) == 1:
                title = f"단일 용매: {components[0]['name']}"
            else:
                title = f"{len(components)}성분 시스템"
            st.subheader(title)
        
        with col2:
            phase_count = system.get('phase_count', 1)
            phase_color = 'success' if phase_count == 1 else 'warning'
            st.markdown(
                f'<div style="text-align: center; padding: 0.5rem; '
                f'background: {COLORS[phase_color]}20; '
                f'border-radius: 8px;">'
                f'<strong>{phase_count}상</strong></div>',
                unsafe_allow_html=True
            )
        
        # 성분 정보
        if show_details:
            st.markdown("**성분 구성:**")
            for i, comp in enumerate(components):
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"• {comp['name']}")
                with col2:
                    st.write(f"{comp.get('ratio', 0)}%")
                with col3:
                    if 'hansen_distance' in comp:
                        color = 'success' if comp['hansen_distance'] < 5 else 'warning'
                        st.markdown(f"Ra = {comp['hansen_distance']:.1f}", 
                                  help="Hansen 거리")
        
        # 특성
        if 'properties' in system:
            st.markdown("**주요 특성:**")
            props = system['properties']
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'viscosity' in props:
                    st.metric("점도", f"{props['viscosity']} cP")
            
            with col2:
                if 'boiling_point' in props:
                    st.metric("끓는점", f"{props['boiling_point']}°C")
            
            with col3:
                if 'polarity' in props:
                    st.metric("극성", props['polarity'])
        
        st.markdown('</div>', unsafe_allow_html=True)

def render_polymer_processing_params(
    process_type: str,
    params: Dict[str, Any],
    editable: bool = False,
    key: str = "polymer_params"
):
    """고분자 가공 파라미터 표시/편집"""
    
    process_configs = {
        'electrospinning': {
            'title': '전기방사 파라미터',
            'icon': '⚡',
            'params': [
                {'name': 'voltage', 'label': '전압 (kV)', 'min': 5, 'max': 30},
                {'name': 'flow_rate', 'label': '유속 (mL/h)', 'min': 0.1, 'max': 10},
                {'name': 'distance', 'label': '거리 (cm)', 'min': 5, 'max': 30},
                {'name': 'temperature', 'label': '온도 (°C)', 'min': 20, 'max': 80}
            ]
        },
        'coating': {
            'title': '코팅 파라미터',
            'icon': '🎨',
            'params': [
                {'name': 'speed', 'label': '속도 (rpm)', 'min': 100, 'max': 5000},
                {'name': 'time', 'label': '시간 (s)', 'min': 10, 'max': 300},
                {'name': 'temperature', 'label': '건조 온도 (°C)', 'min': 20, 'max': 150}
            ]
        },
        'extrusion': {
            'title': '압출 파라미터',
            'icon': '🏭',
            'params': [
                {'name': 'temperature', 'label': '용융 온도 (°C)', 'min': 150, 'max': 300},
                {'name': 'screw_speed', 'label': '스크류 속도 (rpm)', 'min': 10, 'max': 200},
                {'name': 'pressure', 'label': '압력 (bar)', 'min': 10, 'max': 100}
            ]
        }
    }
    
    config = process_configs.get(process_type, {})
    if not config:
        return params
    
    st.markdown(f"### {config['icon']} {config['title']}")
    
    updated_params = {}
    cols = st.columns(2)
    
    for i, param_config in enumerate(config['params']):
        param_name = param_config['name']
        with cols[i % 2]:
            if editable:
                value = st.slider(
                    param_config['label'],
                    min_value=param_config['min'],
                    max_value=param_config['max'],
                    value=params.get(param_name, param_config['min']),
                    key=f"{key}_{param_name}"
                )
                updated_params[param_name] = value
            else:
                st.metric(
                    param_config['label'],
                    f"{params.get(param_name, 'N/A')}"
                )
    
    return updated_params if editable else params

# ===========================================================================
# 📊 벤치마크 및 비교 분석 컴포넌트
# ===========================================================================

@st.cache_data(ttl=3600)
def process_benchmark_data(data: List[Dict], metric: str) -> Dict:
    """벤치마크 데이터 전처리 캐싱"""
    values = [item['value'] for item in data if metric in str(item)]
    return {
        'values': values,
        'mean': sum(values) / len(values) if values else 0,
        'std': pd.Series(values).std() if len(values) > 1 else 0,
        'min': min(values) if values else 0,
        'max': max(values) if values else 0
    }

def render_benchmark_comparison(
    my_data: Dict[str, float],
    benchmark_data: List[Dict[str, Any]],
    metric_name: str,
    key: Optional[str] = None
):
    """벤치마크 비교 시각화"""
    
    # 데이터 전처리 (캐싱)
    stats = process_benchmark_data(benchmark_data, metric_name)
    values = stats['values']
    my_value = my_data['value']
    
    # 백분위 계산
    percentile = (len([v for v in values if v <= my_value]) / len(values)) * 100 if values else 0
    
    # 색상 결정
    if percentile >= 90:
        color = 'success'
        status = "최상위"
    elif percentile >= 75:
        color = 'primary'
        status = "상위"
    elif percentile >= 50:
        color = 'warning'
        status = "중위"
    else:
        color = 'error'
        status = "하위"
    
    # 카드 렌더링
    with st.container():
        st.markdown('<div class="benchmark-card">', unsafe_allow_html=True)
        
        # 헤더
        st.markdown(f"### 📊 {metric_name} 벤치마크")
        
        # 주요 지표
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.metric(
                "내 결과",
                f"{my_value:.2f}",
                help=my_data.get('details', '')
            )
        
        with col2:
            st.markdown(
                f'<div class="percentile-indicator" style="color: {COLORS[color]};">'
                f'{percentile:.0f}%</div>'
                f'<div style="text-align: center;">{status}</div>',
                unsafe_allow_html=True
            )
        
        with col3:
            st.metric(
                "평균값",
                f"{stats['mean']:.2f}",
                delta=f"{my_value - stats['mean']:.2f}"
            )
        
        # 상세 비교
        with st.expander("상세 비교 보기"):
            # 분포 차트
            fig = go.Figure()
            
            # 히스토그램
            fig.add_trace(go.Histogram(
                x=values,
                name='문헌 데이터',
                opacity=0.7,
                marker_color=COLORS['light']
            ))
            
            # 내 데이터 표시
            fig.add_vline(
                x=my_value,
                line_dash="dash",
                line_color=COLORS[color],
                annotation_text="내 결과",
                annotation_position="top"
            )
            
            # 평균선
            fig.add_vline(
                x=stats['mean'],
                line_dash="dot",
                line_color=COLORS['text_secondary'],
                annotation_text="평균",
                annotation_position="bottom"
            )
            
            fig.update_layout(
                title=f"{metric_name} 분포",
                xaxis_title=metric_name,
                yaxis_title="빈도",
                showlegend=True,
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 통계 요약
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("최소값", f"{stats['min']:.2f}")
            with col2:
                st.metric("최대값", f"{stats['max']:.2f}")
            with col3:
                st.metric("표준편차", f"{stats['std']:.2f}")
            with col4:
                st.metric("데이터 수", len(values))
            
            # 상위 5개 비교
            st.markdown("**상위 5개 결과:**")
            top_5 = sorted(benchmark_data, key=lambda x: x['value'], reverse=True)[:5]
            
            df_top5 = pd.DataFrame([
                {
                    '순위': i,
                    '출처': item.get('source', 'Unknown'),
                    '값': f"{item['value']:.2f}",
                    'DOI': item.get('doi', 'N/A')
                }
                for i, item in enumerate(top_5, 1)
            ])
            
            st.dataframe(df_top5, use_container_width=True, hide_index=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def render_protocol_comparison(
    my_protocol: Dict[str, Any],
    reference_protocol: Dict[str, Any],
    key: Optional[str] = None
):
    """프로토콜 비교 UI"""
    
    st.markdown("### 🔬 프로토콜 비교 분석")
    
    # 차이점 분석
    differences = []
    similarities = []
    
    # 재료 비교
    my_materials = set(my_protocol.get('materials', []))
    ref_materials = set(reference_protocol.get('materials', []))
    
    common_materials = my_materials & ref_materials
    unique_my = my_materials - ref_materials
    unique_ref = ref_materials - my_materials
    
    # 조건 비교
    conditions_diff = []
    for param in ['temperature', 'time', 'ph', 'concentration']:
        my_val = my_protocol.get('conditions', {}).get(param)
        ref_val = reference_protocol.get('conditions', {}).get(param)
        
        if my_val is not None and ref_val is not None:
            diff = my_val - ref_val
            if abs(diff) > 0.1:
                conditions_diff.append({
                    'parameter': param,
                    'my_value': my_val,
                    'ref_value': ref_val,
                    'difference': diff,
                    'percent_diff': (diff / ref_val * 100) if ref_val != 0 else 0
                })
    
    # 결과 표시
    tab1, tab2, tab3, tab4 = st.tabs(["📊 요약", "🔍 차이점", "✅ 공통점", "💡 제안"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**내 프로토콜**")
            metrics = [
                {"label": "재료 수", "value": len(my_materials), "icon": "🧪"},
                {"label": "단계 수", "value": len(my_protocol.get('steps', [])), "icon": "📝"}
            ]
            render_metric_cards(metrics, columns=2)
        
        with col2:
            st.markdown("**참조 프로토콜**")
            metrics = [
                {"label": "재료 수", "value": len(ref_materials), "icon": "🧪"},
                {"label": "단계 수", "value": len(reference_protocol.get('steps', [])), "icon": "📝"}
            ]
            render_metric_cards(metrics, columns=2)
            st.caption(f"출처: {reference_protocol.get('source', 'Unknown')}")
    
    with tab2:
        if unique_my or unique_ref:
            st.markdown("#### 재료 차이")
            col1, col2 = st.columns(2)
            
            with col1:
                if unique_my:
                    st.markdown("**내 프로토콜에만 있음:**")
                    for item in unique_my:
                        st.write(f"• {item}")
            
            with col2:
                if unique_ref:
                    st.markdown("**참조에만 있음:**")
                    for item in unique_ref:
                        st.write(f"• {item}")
        
        if conditions_diff:
            st.markdown("#### 조건 차이")
            df_conditions = pd.DataFrame(conditions_diff)
            
            # 차이를 시각적으로 표시
            for _, row in df_conditions.iterrows():
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                
                with col1:
                    st.write(f"**{row['parameter'].title()}**")
                
                with col2:
                    st.metric("내 값", f"{row['my_value']:.1f}")
                
                with col3:
                    st.metric("참조 값", f"{row['ref_value']:.1f}")
                
                with col4:
                    color = 'error' if abs(row['percent_diff']) > 20 else 'warning'
                    st.metric("차이", f"{row['percent_diff']:.1f}%", 
                            delta_color='normal' if row['difference'] > 0 else 'inverse')
    
    with tab3:
        if common_materials:
            st.markdown("#### 공통 재료")
            for i, item in enumerate(common_materials):
                if i % 3 == 0:
                    cols = st.columns(3)
                cols[i % 3].write(f"• {item}")
        
        # 유사한 조건
        st.markdown("#### 유사한 조건")
        similar_conditions = []
        for param in ['temperature', 'time', 'ph', 'concentration']:
            my_val = my_protocol.get('conditions', {}).get(param)
            ref_val = reference_protocol.get('conditions', {}).get(param)
            
            if my_val is not None and ref_val is not None and abs(my_val - ref_val) <= 0.1:
                similar_conditions.append(f"{param}: {my_val:.1f}")
        
        if similar_conditions:
            st.write(", ".join(similar_conditions))
    
    with tab4:
        st.markdown("#### 💡 개선 제안")
        
        suggestions = []
        
        # 재료 제안
        if unique_ref:
            suggestions.append({
                'type': 'material',
                'priority': 'high' if len(unique_ref) > 2 else 'medium',
                'suggestion': f"참조 프로토콜의 다음 재료 사용을 고려하세요: {', '.join(list(unique_ref)[:3])}"
            })
        
        # 조건 제안
        for cond in conditions_diff:
            if abs(cond['percent_diff']) > 20:
                suggestions.append({
                    'type': 'condition',
                    'priority': 'high',
                    'suggestion': f"{cond['parameter']}을(를) {cond['ref_value']:.1f}로 조정하면 참조 프로토콜과 유사해집니다"
                })
        
        # 제안 표시
        for suggestion in suggestions:
            icon = "🔴" if suggestion['priority'] == 'high' else "🟡"
            st.write(f"{icon} {suggestion['suggestion']}")

# ===========================================================================
# 📋 프로토콜 추출 결과 UI
# ===========================================================================

def render_protocol_extraction_result(
    extraction_result: Dict[str, Any],
    key: Optional[str] = None
):
    """프로토콜 추출 결과 표시"""
    
    if extraction_result.get('status') == 'error':
        error_code = extraction_result.get('error_code', 'Unknown')
        error_message = extraction_result.get('message', 'Unknown error')
        
        # 에러 타입별 아이콘 및 색상
        error_icons = {
            '4200': '📄',  # 파일 형식
            '4201': '🔤',  # 인코딩
            '4202': '🔍',  # 추출 실패
            '4203': '📏',  # 텍스트 길이
            '4205': '👁️',  # OCR
        }
        
        icon = error_icons.get(error_code[:4], '❌')
        st.error(f"{icon} 추출 실패: {error_message}")
        
        # 복구 제안
        if 'recovery' in extraction_result:
            st.info(f"💡 해결 방법: {extraction_result['recovery']}")
        
        return
    
    st.success("✅ 프로토콜 추출 완료!")
    
    # 메타데이터
    with st.expander("📋 문서 정보", expanded=False):
        metadata = extraction_result.get('metadata', {})
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**제목:** {metadata.get('title', 'N/A')}")
            st.write(f"**저자:** {', '.join(metadata.get('authors', []))}")
            st.write(f"**추출일:** {metadata.get('extraction_date', 'N/A')}")
        
        with col2:
            st.write(f"**출처:** {metadata.get('source', 'N/A')}")
            if metadata.get('doi'):
                st.write(f"**DOI:** [{metadata['doi']}](https://doi.org/{metadata['doi']})")
            st.write(f"**페이지:** {metadata.get('pages', 'N/A')}")
    
    # 추출된 프로토콜
    protocol = extraction_result.get('protocol', {})
    
    # 재료
    st.markdown("### 🧪 재료")
    materials = protocol.get('materials', [])
    if materials:
        # 재료를 카테고리별로 그룹화
        categorized = {}
        for mat in materials:
            category = mat.get('category', 'Uncategorized')
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(mat)
        
        # 카테고리별 표시
        for category, items in categorized.items():
            st.markdown(f"**{category}**")
            df = pd.DataFrame(items)
            # 필요한 컬럼만 선택
            display_cols = ['name', 'amount', 'unit', 'purity', 'supplier']
            display_cols = [col for col in display_cols if col in df.columns]
            if display_cols:
                st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
    else:
        st.info("재료 정보를 찾을 수 없습니다.")
    
    # 실험 조건
    st.markdown("### ⚙️ 실험 조건")
    conditions = protocol.get('conditions', {})
    if conditions:
        # 조건을 그룹별로 정리
        condition_groups = {
            '온도': ['temperature', 'heating_rate', 'cooling_rate'],
            '시간': ['time', 'duration', 'reaction_time'],
            '속도/유량': ['stirring_speed', 'flow_rate', 'rpm'],
            '농도/pH': ['concentration', 'ph', 'molarity'],
            '압력/전압': ['pressure', 'voltage', 'current']
        }
        
        cols = st.columns(3)
        col_idx = 0
        
        for group_name, params in condition_groups.items():
            group_conditions = {k: v for k, v in conditions.items() if k in params}
            if group_conditions:
                with cols[col_idx % 3]:
                    st.markdown(f"**{group_name}**")
                    for key, value in group_conditions.items():
                        if isinstance(value, dict):
                            st.write(f"{key}: {value.get('value', 'N/A')} {value.get('unit', '')}")
                        else:
                            st.write(f"{key}: {value}")
                col_idx += 1
    
    # 절차
    st.markdown("### 📝 실험 절차")
    steps = protocol.get('procedure', [])
    if steps:
        # 단계별 표시 with 시간 정보
        for i, step in enumerate(steps, 1):
            col1, col2 = st.columns([5, 1])
            
            with col1:
                if isinstance(step, dict):
                    st.write(f"**Step {i}:** {step.get('description', '')}")
                    if 'details' in step:
                        st.caption(step['details'])
                else:
                    st.write(f"**Step {i}:** {step}")
            
            with col2:
                if isinstance(step, dict) and 'duration' in step:
                    st.caption(f"⏱️ {step['duration']}")
    else:
        st.info("절차 정보를 찾을 수 없습니다.")
    
    # 신뢰도 및 품질 지표
    confidence = extraction_result.get('confidence', {})
    quality_metrics = extraction_result.get('quality_metrics', {})
    
    if confidence or quality_metrics:
        st.markdown("### 📊 추출 품질 평가")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            score = confidence.get('overall', 0)
            color = 'success' if score > 80 else 'warning' if score > 60 else 'error'
            st.markdown(
                f'<div style="text-align: center;">'
                f'<h2 style="color: {COLORS[color]};">{score}%</h2>'
                f'<p>전체 신뢰도</p></div>',
                unsafe_allow_html=True
            )
        
        with col2:
            completeness = quality_metrics.get('completeness', 0)
            st.metric("완전성", f"{completeness}%", 
                     help="필수 정보가 모두 추출되었는지")
        
        with col3:
            accuracy = quality_metrics.get('accuracy', 0)
            st.metric("정확도", f"{accuracy}%",
                     help="추출된 정보의 정확성")
        
        with col4:
            consistency = quality_metrics.get('consistency', 0)
            st.metric("일관성", f"{consistency}%",
                     help="정보 간 일관성")
    
    # 추출 액션
    st.markdown("### 🚀 다음 단계")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📝 실험 설계 생성", key=f"{key}_create_design"):
            st.session_state.extracted_protocol = protocol
            st.switch_page("pages/3_🧪_Experiment_Design.py")
    
    with col2:
        if st.button("🔍 유사 프로토콜 검색", key=f"{key}_search_similar"):
            st.session_state.search_protocol = protocol
            st.switch_page("pages/6_🔍_Literature_Search.py")
    
    with col3:
        if st.button("💾 프로토콜 저장", key=f"{key}_save"):
            st.success("프로토콜이 저장되었습니다!")

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
    # 파일 형식별 도움말
    format_help = {
        'pdf': 'PDF 문서에서 프로토콜을 자동 추출합니다',
        'docx': 'Word 문서를 지원합니다',
        'txt': '텍스트 파일 (UTF-8 인코딩 권장)',
        'csv': 'CSV 데이터 파일',
        'xlsx': 'Excel 스프레드시트'
    }
    
    # 도움말 생성
    if not help_text:
        helps = [format_help.get(ft, ft) for ft in file_types]
        help_text = f"지원 형식: {', '.join(helps)}"
    
    uploaded_files = st.file_uploader(
        label,
        type=file_types,
        accept_multiple_files=accept_multiple,
        help=help_text,
        key=key
    )
    
    if uploaded_files:
        if accept_multiple:
            total_size = sum(f.size for f in uploaded_files)
            st.caption(f"📎 {len(uploaded_files)}개 파일 ({total_size:,} bytes)")
            
            # 파일 리스트
            with st.expander("업로드된 파일 목록"):
                for file in uploaded_files:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"• {file.name}")
                    with col2:
                        st.caption(f"{file.size:,} bytes")
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

def render_accessible_button(
    label: str,
    key: str,
    aria_label: Optional[str] = None,
    **kwargs
):
    """접근성 개선된 버튼"""
    # aria-label 자동 생성
    if not aria_label:
        # 아이콘 제거하고 텍스트만 추출
        text_only = label
        for icon in ICONS.values():
            text_only = text_only.replace(icon, '').strip()
        aria_label = text_only
    
    # Streamlit 버튼에 접근성 속성 추가
    button = st.button(label, key=key, **kwargs)
    
    # 추가 접근성 정보 (화면 리더용)
    if button:
        st.markdown(f'<span class="sr-only">{aria_label} 버튼이 클릭되었습니다</span>', 
                   unsafe_allow_html=True)
    
    return button

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
            
            # 태그
            if 'tags' in experiment:
                tags_html = ' '.join([
                    f'<span style="background: {COLORS["light"]}; '
                    f'padding: 0.25rem 0.5rem; border-radius: 15px; '
                    f'font-size: 0.875rem; margin-right: 0.5rem;">{tag}</span>'
                    for tag in experiment['tags']
                ])
                st.markdown(tags_html, unsafe_allow_html=True)
        
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
                
                if experiment.get('status') == 'completed':
                    if st.button("분석", key=f"analyze_{experiment.get('id', '')}_{key}"):
                        st.session_state.selected_experiment = experiment['id']
                        st.switch_page("pages/4_📈_Data_Analysis.py")
        
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
                installed = module.get('id') in st.session_state.get('installed_modules', [])
                
                if installed:
                    st.success("설치됨")
                else:
                    if st.button("설치", key=f"install_{module.get('id', '')}_{key}"):
                        # 설치 로직
                        if 'installed_modules' not in st.session_state:
                            st.session_state.installed_modules = []
                        st.session_state.installed_modules.append(module['id'])
                        st.success("모듈이 설치되었습니다!")
                        st.rerun()
        
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
    increment = target / steps if steps > 0 else target
    
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
        st.session_state.offline_mode = False

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
    
    # 신규 메서드 추가
    @staticmethod
    def render_solvent_system_card(*args, **kwargs):
        return render_solvent_system_card(*args, **kwargs)
    
    @staticmethod
    def render_benchmark_comparison(*args, **kwargs):
        return render_benchmark_comparison(*args, **kwargs)
    
    @staticmethod
    def render_protocol_comparison(*args, **kwargs):
        return render_protocol_comparison(*args, **kwargs)
    
    @staticmethod
    def render_protocol_extraction_result(*args, **kwargs):
        return render_protocol_extraction_result(*args, **kwargs)

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
    
    # 벤치마크 테스트
    st.subheader("벤치마크 비교")
    my_result = {'value': 85.5, 'details': '최적화된 조건에서 측정'}
    benchmark_data = [
        {'value': 78.2, 'source': 'Kim et al. (2023)', 'doi': '10.1234/example'},
        {'value': 82.1, 'source': 'Lee et al. (2023)', 'doi': '10.1234/example2'},
        {'value': 91.3, 'source': 'Park et al. (2024)', 'doi': '10.1234/example3'},
        {'value': 76.5, 'source': 'Choi et al. (2022)', 'doi': '10.1234/example4'},
        {'value': 88.7, 'source': 'Jung et al. (2024)', 'doi': '10.1234/example5'}
    ]
    
    render_benchmark_comparison(my_result, benchmark_data, "수율 (%)")
    
    # 고분자 컴포넌트 테스트
    st.subheader("고분자 시스템")
    solvent_system = {
        'components': [
            {'name': 'DMF', 'ratio': 70, 'hansen_distance': 3.2},
            {'name': 'THF', 'ratio': 30, 'hansen_distance': 4.8}
        ],
        'phase_count': 1,
        'properties': {
            'viscosity': 2.5,
            'boiling_point': 153,
            'polarity': 'High'
        }
    }
    
    render_solvent_system_card(solvent_system)
    
    render_footer()
