"""
Common UI Components for Polymer DOE Platform
모던하고 반응형 UI 컴포넌트 모음
"""

import streamlit as st
from typing import Optional, Dict, Any, List, Tuple, Union
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
from datetime import datetime, timedelta
import base64
from io import BytesIO

# 🎨 모던 색상 팔레트 정의
COLORS = {
    'primary': '#7C3AED',      # 보라색 (메인)
    'primary_light': '#A78BFA', # 연한 보라색
    'primary_dark': '#5B21B6',  # 진한 보라색
    'secondary': '#F59E0B',     # 주황색
    'success': '#10B981',       # 초록색
    'warning': '#F59E0B',       # 주황색
    'danger': '#EF4444',        # 빨간색
    'info': '#3B82F6',          # 파란색
    'dark': '#1F2937',          # 진한 회색
    'gray': '#6B7280',          # 중간 회색
    'light': '#F3F4F6',         # 연한 회색
    'white': '#FFFFFF',         # 흰색
    'background': '#F9FAFB',    # 배경색
    'card_bg': '#FFFFFF',       # 카드 배경
    'border': '#E5E7EB',        # 테두리
}

# 🎯 사용자 레벨 정의
USER_LEVELS = {
    'beginner': {'label': '초급', 'emoji': '🌱', 'color': COLORS['success']},
    'intermediate': {'label': '중급', 'emoji': '🌿', 'color': COLORS['info']},
    'advanced': {'label': '고급', 'emoji': '🌳', 'color': COLORS['warning']},
    'expert': {'label': '전문가', 'emoji': '🌲', 'color': COLORS['primary']}
}


def setup_page_config():
    """페이지 기본 설정 - 모던 UI"""
    st.set_page_config(
        page_title="🧬 Polymer DOE Platform",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/your-repo/polymer-doe',
            'Report a bug': 'https://github.com/your-repo/polymer-doe/issues',
            'About': '# Polymer DOE Platform\nAI 기반 고분자 실험 설계 플랫폼'
        }
    )


def apply_custom_css():
    """모던하고 반응형 CSS 스타일 적용"""
    st.markdown("""
    <style>
    /* 구글 폰트 임포트 */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* 전체 폰트 설정 */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* 메인 컨테이너 스타일 */
    .main {
        padding: 0;
        background-color: #F9FAFB;
    }
    
    /* 사이드바 스타일 - 모던하게 */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E5E7EB;
        box-shadow: 2px 0 5px rgba(0,0,0,0.05);
    }
    
    section[data-testid="stSidebar"] .css-1d391kg {
        padding: 2rem 1rem;
    }
    
    /* 사이드바 메뉴 아이템 */
    .sidebar-menu-item {
        display: flex;
        align-items: center;
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        border-radius: 0.5rem;
        color: #4B5563;
        text-decoration: none;
        transition: all 0.2s ease;
        cursor: pointer;
    }
    
    .sidebar-menu-item:hover {
        background-color: #F3F4F6;
        color: #7C3AED;
        transform: translateX(2px);
    }
    
    .sidebar-menu-item.active {
        background-color: #EDE9FE;
        color: #7C3AED;
        font-weight: 600;
    }
    
    /* 카드 스타일 */
    .modern-card {
        background: white;
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #E5E7EB;
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .modern-card:hover {
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    /* 메트릭 카드 */
    .metric-card {
        background: white;
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #E5E7EB;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #7C3AED 0%, #A78BFA 100%);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1F2937;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #6B7280;
        font-weight: 500;
    }
    
    .metric-delta {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    .metric-delta.positive {
        background-color: #D1FAE5;
        color: #065F46;
    }
    
    .metric-delta.negative {
        background-color: #FEE2E2;
        color: #991B1B;
    }
    
    /* 버튼 스타일 */
    .stButton > button {
        background: linear-gradient(135deg, #7C3AED 0%, #A78BFA 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 0.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(124, 58, 237, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(124, 58, 237, 0.3);
    }
    
    /* 보조 버튼 */
    .secondary-button > button {
        background: #F3F4F6;
        color: #4B5563;
        border: 1px solid #E5E7EB;
    }
    
    .secondary-button > button:hover {
        background: #E5E7EB;
        border-color: #D1D5DB;
    }
    
    /* 입력 필드 스타일 */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stTextArea > div > div > textarea {
        border: 1px solid #E5E7EB;
        border-radius: 0.5rem;
        padding: 0.75rem;
        font-size: 0.875rem;
        transition: all 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #7C3AED;
        box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.1);
    }
    
    /* 테이블 스타일 */
    .dataframe {
        border: none !important;
        font-size: 0.875rem;
    }
    
    .dataframe thead tr th {
        background-color: #F9FAFB !important;
        color: #4B5563 !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        font-size: 0.75rem !important;
        letter-spacing: 0.05em;
        border-bottom: 2px solid #E5E7EB !important;
    }
    
    .dataframe tbody tr {
        border-bottom: 1px solid #F3F4F6 !important;
    }
    
    .dataframe tbody tr:hover {
        background-color: #F9FAFB !important;
    }
    
    /* 상태 배지 */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.375rem 0.875rem;
        border-radius: 1rem;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.025em;
    }
    
    .status-badge.available {
        background-color: #D1FAE5;
        color: #065F46;
    }
    
    .status-badge.sold-out {
        background-color: #FED7D7;
        color: #9B2C2C;
    }
    
    /* 반응형 그리드 */
    @media (max-width: 768px) {
        .row-widget.stHorizontalBlock {
            flex-direction: column !important;
        }
        
        .row-widget.stHorizontalBlock > div {
            width: 100% !important;
            margin-bottom: 1rem;
        }
        
        .modern-card {
            padding: 1rem;
        }
        
        .metric-value {
            font-size: 1.5rem;
        }
        
        section[data-testid="stSidebar"] {
            transform: translateX(-100%);
        }
        
        section[data-testid="stSidebar"][aria-expanded="true"] {
            transform: translateX(0);
        }
    }
    
    /* 차트 컨테이너 */
    .chart-container {
        background: white;
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #E5E7EB;
    }
    
    /* 프로그레스 바 */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #7C3AED 0%, #A78BFA 100%);
        border-radius: 1rem;
    }
    
    /* 정보 박스 */
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid;
        margin: 1rem 0;
        position: relative;
        padding-left: 3rem;
    }
    
    .info-box::before {
        content: '';
        position: absolute;
        left: 1rem;
        top: 50%;
        transform: translateY(-50%);
        width: 1.5rem;
        height: 1.5rem;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .info-box.info {
        background-color: #DBEAFE;
        border-color: #3B82F6;
        color: #1E40AF;
    }
    
    .info-box.success {
        background-color: #D1FAE5;
        border-color: #10B981;
        color: #065F46;
    }
    
    .info-box.warning {
        background-color: #FEF3C7;
        border-color: #F59E0B;
        color: #92400E;
    }
    
    .info-box.error {
        background-color: #FEE2E2;
        border-color: #EF4444;
        color: #991B1B;
    }
    
    /* 스크롤바 커스터마이징 */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #F3F4F6;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #D1D5DB;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #9CA3AF;
    }
    
    /* 애니메이션 */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* 탭 스타일 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: #F3F4F6;
        padding: 0.25rem;
        border-radius: 0.75rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        background-color: transparent;
        border: none;
        color: #6B7280;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white;
        color: #7C3AED;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)


def render_header(title: str, subtitle: Optional[str] = None, breadcrumb: Optional[List[str]] = None):
    """모던 헤더 렌더링"""
    # 브레드크럼
    if breadcrumb:
        breadcrumb_html = " / ".join([f'<span style="color: #6B7280;">{item}</span>' for item in breadcrumb[:-1]])
        if len(breadcrumb) > 1:
            breadcrumb_html += f' / <span style="color: #7C3AED; font-weight: 600;">{breadcrumb[-1]}</span>'
        st.markdown(f'<p style="font-size: 0.875rem; margin-bottom: 1rem;">{breadcrumb_html}</p>', 
                   unsafe_allow_html=True)
    
    # 타이틀
    st.markdown(f"""
    <div class="fade-in">
        <h1 style="font-size: 2rem; font-weight: 700; color: #1F2937; margin-bottom: 0.5rem;">
            {title}
        </h1>
        {f'<p style="color: #6B7280; font-size: 1rem; margin-bottom: 2rem;">{subtitle}</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


def render_modern_sidebar():
    """모던 사이드바 렌더링"""
    with st.sidebar:
        # 로고 영역
        st.markdown("""
        <div style="padding: 1rem 0; border-bottom: 1px solid #E5E7EB; margin-bottom: 1rem;">
            <h2 style="color: #7C3AED; font-weight: 700; font-size: 1.5rem; display: flex; align-items: center;">
                🧬 <span style="margin-left: 0.5rem;">Polymer DOE</span>
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        # 메뉴 아이템
        menu_items = [
            {"icon": "🏠", "label": "Overview", "key": "overview"},
            {"icon": "💬", "label": "Chat", "key": "chat"},
            {"icon": "📊", "label": "Analytics", "key": "analytics"},
            {"icon": "🛍️", "label": "Sales", "key": "sales"},
            {"icon": "📝", "label": "Review", "key": "review"},
            {"icon": "📦", "label": "Products", "key": "products"},
        ]
        
        selected_menu = st.session_state.get('selected_menu', 'overview')
        
        for item in menu_items:
            is_active = selected_menu == item['key']
            if st.button(
                f"{item['icon']} {item['label']}", 
                key=f"menu_{item['key']}",
                use_container_width=True,
                type="secondary" if not is_active else "primary"
            ):
                st.session_state.selected_menu = item['key']
                st.rerun()


def show_info_message(message: str, message_type: str = 'info'):
    """정보 메시지 표시"""
    icons = {
        'info': 'ℹ️',
        'success': '✅',
        'warning': '⚠️',
        'error': '❌'
    }
    
    colors = {
        'info': COLORS['info'],
        'success': COLORS['success'],
        'warning': COLORS['warning'],
        'error': COLORS['danger']
    }
    
    st.markdown(f"""
    <div class="info-box" style="border-left: 4px solid {colors.get(message_type, colors['info'])};">
        <strong>{icons.get(message_type, icons['info'])} {message}</strong>
    </div>
    """, unsafe_allow_html=True)


def create_metric_card(title: str, value: str, delta: Optional[str] = None, 
                      delta_type: str = "positive", chart_data: Optional[pd.DataFrame] = None):
    """모던 메트릭 카드 생성"""
    delta_class = "positive" if delta_type == "positive" else "negative"
    delta_icon = "↑" if delta_type == "positive" else "↓"
    
    card_html = f"""
    <div class="metric-card fade-in">
        <div class="metric-label">{title}</div>
        <div class="metric-value">{value}</div>
        {f'<div class="metric-delta {delta_class}">{delta_icon} {delta}</div>' if delta else ''}
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)
    
    # 차트가 있으면 추가
    if chart_data is not None:
        fig = px.line(chart_data, height=60)
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def create_analytics_chart(data: pd.DataFrame, chart_type: str = "area"):
    """분석 차트 생성"""
    fig = go.Figure()
    
    if chart_type == "area":
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data.values,
            fill='tozeroy',
            fillcolor='rgba(124, 58, 237, 0.1)',
            line=dict(color='#7C3AED', width=3),
            mode='lines',
            name='Value'
        ))
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=300,
        showlegend=False,
        xaxis=dict(
            showgrid=False,
            showline=False,
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#F3F4F6',
            showline=False,
            zeroline=False,
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified'
    )
    
    return fig


def render_data_table(df: pd.DataFrame, show_status: bool = False, 
                     selectable: bool = False, height: int = 400):
    """모던 데이터 테이블 렌더링"""
    # 상태 컬럼이 있으면 스타일링
    if show_status and 'Status' in df.columns:
        def style_status(val):
            if val == 'Available':
                return 'background-color: #D1FAE5; color: #065F46; padding: 4px 12px; border-radius: 12px;'
            else:
                return 'background-color: #FED7D7; color: #9B2C2C; padding: 4px 12px; border-radius: 12px;'
        
        styled_df = df.style.applymap(style_status, subset=['Status'])
        st.dataframe(
            styled_df,
            use_container_width=True,
            height=height,
            hide_index=True,
            selection_mode="multi-row" if selectable else None
        )
    else:
        st.dataframe(
            df,
            use_container_width=True,
            height=height,
            hide_index=True,
            selection_mode="multi-row" if selectable else None
        )


def create_product_card(image_url: str, title: str, rating: float, 
                       price: str, badge: Optional[str] = None):
    """제품 카드 생성"""
    stars = "⭐" * int(rating)
    
    card_html = f"""
    <div class="modern-card fade-in" style="text-align: center;">
        <div style="position: relative;">
            <img src="{image_url}" style="width: 100%; height: 150px; object-fit: cover; border-radius: 0.5rem;">
            {f'<span class="status-badge available" style="position: absolute; top: 10px; right: 10px;">{badge}</span>' if badge else ''}
        </div>
        <h3 style="margin: 1rem 0 0.5rem 0; font-size: 1.125rem; font-weight: 600; color: #1F2937;">
            {title}
        </h3>
        <div style="color: #6B7280; font-size: 0.875rem; margin-bottom: 0.5rem;">
            {stars} {rating}
        </div>
        <div style="font-size: 1.5rem; font-weight: 700; color: #7C3AED;">
            {price}
        </div>
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)


def show_notification(message: str, type: str = "info", duration: int = 3):
    """토스트 알림 표시"""
    notification_container = st.empty()
    
    icons = {
        'info': 'ℹ️',
        'success': '✅',
        'warning': '⚠️',
        'error': '❌'
    }
    
    colors = {
        'info': '#3B82F6',
        'success': '#10B981',
        'warning': '#F59E0B',
        'error': '#EF4444'
    }
    
    with notification_container.container():
        st.markdown(f"""
        <div style="
            position: fixed;
            top: 20px;
            right: 20px;
            background: white;
            padding: 1rem 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-left: 4px solid {colors[type]};
            z-index: 1000;
            animation: slideIn 0.3s ease-out;
        ">
            <div style="display: flex; align-items: center;">
                <span style="font-size: 1.25rem; margin-right: 0.75rem;">{icons[type]}</span>
                <span style="color: #1F2937; font-weight: 500;">{message}</span>
            </div>
        </div>
        
        <style>
        @keyframes slideIn {{
            from {{ transform: translateX(100%); opacity: 0; }}
            to {{ transform: translateX(0); opacity: 1; }}
        }}
        </style>
        """, unsafe_allow_html=True)
    
    time.sleep(duration)
    notification_container.empty()


def create_empty_state(title: str, description: str, action_label: Optional[str] = None):
    """빈 상태 UI"""
    st.markdown(f"""
    <div style="
        text-align: center;
        padding: 4rem 2rem;
        background: #F9FAFB;
        border-radius: 1rem;
        border: 2px dashed #E5E7EB;
    ">
        <div style="font-size: 3rem; margin-bottom: 1rem;">📭</div>
        <h3 style="color: #1F2937; margin-bottom: 0.5rem;">{title}</h3>
        <p style="color: #6B7280; margin-bottom: 2rem;">{description}</p>
        {f'<button class="stButton">{action_label}</button>' if action_label else ''}
    </div>
    """, unsafe_allow_html=True)


def create_progress_indicator(current: int, total: int, label: str = ""):
    """프로그레스 인디케이터"""
    percentage = (current / total) * 100 if total > 0 else 0
    
    st.markdown(f"""
    <div style="margin: 1rem 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="color: #6B7280; font-size: 0.875rem;">{label}</span>
            <span style="color: #7C3AED; font-weight: 600; font-size: 0.875rem;">
                {current}/{total} ({percentage:.0f}%)
            </span>
        </div>
        <div style="width: 100%; height: 8px; background: #E5E7EB; border-radius: 4px; overflow: hidden;">
            <div style="width: {percentage}%; height: 100%; background: linear-gradient(90deg, #7C3AED 0%, #A78BFA 100%); 
                        border-radius: 4px; transition: width 0.3s ease;">
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def create_stats_grid(stats: List[Dict[str, Any]]):
    """통계 그리드 생성"""
    cols = st.columns(len(stats))
    
    for idx, (col, stat) in enumerate(zip(cols, stats)):
        with col:
            st.markdown(f"""
            <div class="modern-card" style="text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{stat['icon']}</div>
                <div style="font-size: 2rem; font-weight: 700; color: #1F2937;">
                    {stat['value']}
                </div>
                <div style="color: #6B7280; font-size: 0.875rem;">
                    {stat['label']}
                </div>
            </div>
            """, unsafe_allow_html=True)


def create_action_button(label: str, icon: str = "", variant: str = "primary", 
                        full_width: bool = True, key: Optional[str] = None):
    """액션 버튼 생성"""
    button_classes = {
        'primary': 'background: linear-gradient(135deg, #7C3AED 0%, #A78BFA 100%); color: white;',
        'secondary': 'background: #F3F4F6; color: #4B5563; border: 1px solid #E5E7EB;',
        'success': 'background: #10B981; color: white;',
        'danger': 'background: #EF4444; color: white;'
    }
    
    width_style = "width: 100%;" if full_width else ""
    
    if st.button(f"{icon} {label}".strip(), key=key, use_container_width=full_width):
        return True
    
    return False


def create_search_bar(placeholder: str = "검색...", key: str = "search"):
    """모던 검색바"""
    search_value = st.text_input(
        label="",
        placeholder=placeholder,
        key=key,
        label_visibility="collapsed"
    )
    
    # 검색 아이콘 추가를 위한 CSS
    st.markdown("""
    <style>
    input[type="text"] {
        padding-left: 2.5rem !important;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='%236B7280' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='11' cy='11' r='8'%3E%3C/circle%3E%3Cpath d='m21 21-4.35-4.35'%3E%3C/path%3E%3C/svg%3E");
        background-repeat: no-repeat;
        background-position: 0.75rem center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    return search_value


def create_dropdown_filter(label: str, options: List[str], default: str = "All", 
                          key: str = "filter"):
    """드롭다운 필터"""
    selected = st.selectbox(
        label,
        options=options,
        index=options.index(default) if default in options else 0,
        key=key
    )
    return selected


def render_loading_skeleton(rows: int = 3):
    """로딩 스켈레톤 UI"""
    for i in range(rows):
        st.markdown(f"""
        <div class="modern-card" style="margin-bottom: 1rem;">
            <div style="background: linear-gradient(90deg, #F3F4F6 25%, #E5E7EB 50%, #F3F4F6 75%);
                        background-size: 200% 100%;
                        animation: loading 1.5s infinite;
                        height: 20px;
                        border-radius: 4px;
                        margin-bottom: 0.75rem;">
            </div>
            <div style="background: linear-gradient(90deg, #F3F4F6 25%, #E5E7EB 50%, #F3F4F6 75%);
                        background-size: 200% 100%;
                        animation: loading 1.5s infinite;
                        height: 16px;
                        width: 60%;
                        border-radius: 4px;">
            </div>
        </div>
        
        <style>
        @keyframes loading {{
            0% {{ background-position: 200% 0; }}
            100% {{ background-position: -200% 0; }}
        }}
        </style>
        """, unsafe_allow_html=True)


def create_timeline(events: List[Dict[str, Any]]):
    """타임라인 생성"""
    for idx, event in enumerate(events):
        is_last = idx == len(events) - 1
        
        st.markdown(f"""
        <div style="display: flex; margin-bottom: 2rem;">
            <div style="flex-shrink: 0; width: 40px; position: relative;">
                <div style="width: 12px; height: 12px; background: #7C3AED; border-radius: 50%;
                           position: absolute; left: 14px; top: 6px;">
                </div>
                {f'<div style="width: 2px; background: #E5E7EB; position: absolute; left: 19px; top: 24px; bottom: -2rem;"></div>' if not is_last else ''}
            </div>
            <div style="flex-grow: 1; padding-left: 1rem;">
                <div style="font-weight: 600; color: #1F2937; margin-bottom: 0.25rem;">
                    {event['title']}
                </div>
                <div style="color: #6B7280; font-size: 0.875rem; margin-bottom: 0.5rem;">
                    {event['time']}
                </div>
                <div style="color: #4B5563; font-size: 0.875rem;">
                    {event['description']}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def create_user_avatar(name: str, image_url: Optional[str] = None, size: str = "md"):
    """사용자 아바타 생성"""
    sizes = {
        'sm': '32px',
        'md': '40px',
        'lg': '56px'
    }
    
    size_px = sizes.get(size, sizes['md'])
    
    if image_url:
        avatar_html = f'<img src="{image_url}" style="width: {size_px}; height: {size_px}; border-radius: 50%; object-fit: cover;">'
    else:
        # 이름의 첫 글자로 아바타 생성
        initials = ''.join([word[0].upper() for word in name.split()[:2]])
        avatar_html = f"""
        <div style="width: {size_px}; height: {size_px}; border-radius: 50%; 
                    background: linear-gradient(135deg, #7C3AED 0%, #A78BFA 100%);
                    display: flex; align-items: center; justify-content: center;
                    color: white; font-weight: 600; font-size: {'0.875rem' if size == 'sm' else '1rem'};">
            {initials}
        </div>
        """
    
    return avatar_html


def render_footer():
    """모던 푸터 렌더링"""
    st.markdown("""
    <div style="margin-top: 4rem; padding: 2rem 0; border-top: 1px solid #E5E7EB;">
        <div style="text-align: center; color: #6B7280;">
            <p style="margin-bottom: 0.5rem;">🧬 Polymer DOE Platform © 2024</p>
            <p style="font-size: 0.875rem;">Made with ❤️ for Polymer Researchers</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# 레이아웃 헬퍼 함수들
def create_responsive_columns(ratios: List[int], gap: str = "medium"):
    """반응형 컬럼 생성"""
    gaps = {
        'small': '0.5rem',
        'medium': '1rem',
        'large': '2rem'
    }
    
    gap_size = gaps.get(gap, gaps['medium'])
    
    # CSS로 반응형 그리드 구현
    st.markdown(f"""
    <style>
    .responsive-grid {{
        display: grid;
        grid-template-columns: repeat({len(ratios)}, {' '.join([f'{r}fr' for r in ratios])});
        gap: {gap_size};
    }}
    
    @media (max-width: 768px) {{
        .responsive-grid {{
            grid-template-columns: 1fr;
        }}
    }}
    </style>
    """, unsafe_allow_html=True)
    
    return st.columns(ratios, gap=gap)


def create_modal(title: str, content: str, key: str = "modal"):
    """모달 다이얼로그"""
    if st.session_state.get(f'show_{key}', False):
        st.markdown(f"""
        <div style="position: fixed; top: 0; left: 0; right: 0; bottom: 0;
                    background: rgba(0,0,0,0.5); z-index: 1000;
                    display: flex; align-items: center; justify-content: center;">
            <div style="background: white; border-radius: 1rem; padding: 2rem;
                        max-width: 500px; width: 90%; max-height: 80vh; overflow-y: auto;
                        box-shadow: 0 20px 25px -5px rgba(0,0,0,0.1);">
                <h2 style="margin-bottom: 1rem;">{title}</h2>
                <div style="margin-bottom: 2rem;">{content}</div>
                <button onclick="window.location.reload();" 
                        style="padding: 0.5rem 1rem; background: #7C3AED; color: white;
                               border: none; border-radius: 0.5rem; cursor: pointer;">
                    닫기
                </button>
            </div>
        </div>
        """, unsafe_allow_html=True)


# 유틸리티 함수들
def format_number(number: float, decimals: int = 0, prefix: str = "", suffix: str = "") -> str:
    """숫자 포맷팅"""
    if decimals > 0:
        formatted = f"{number:,.{decimals}f}"
    else:
        formatted = f"{number:,.0f}"
    return f"{prefix}{formatted}{suffix}"


def format_currency(amount: float, currency: str = "$") -> str:
    """통화 포맷팅"""
    return f"{currency}{amount:,.2f}"


def get_color_by_value(value: float, thresholds: List[Tuple[float, str]]) -> str:
    """값에 따른 색상 반환"""
    for threshold, color in sorted(thresholds, reverse=True):
        if value >= threshold:
            return color
    return COLORS['gray']


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """텍스트 자르기"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix
