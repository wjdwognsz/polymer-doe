"""
🎨 Universal DOE Platform - UI 테마 설정
================================================================================
데스크톱 애플리케이션에 최적화된 시각적 테마 시스템
눈의 피로를 최소화하면서도 전문적인 느낌을 주는 색상 체계
================================================================================
"""

from typing import Dict, Any, Optional
import streamlit as st

# ============================================================================
# 🎨 색상 시스템 (Color System)
# ============================================================================

class Colors:
    """플랫폼 전체 색상 팔레트 - 눈의 피로를 최소화하는 부드러운 색상"""
    
    # Primary Colors - 보라색 계열
    PRIMARY = "#a880ed"          # 메인 보라색 (부드러운 톤)
    PRIMARY_DARK = "#8b5cf6"     # 진한 보라색
    PRIMARY_LIGHT = "#c4b5fd"    # 연한 보라색
    PRIMARY_LIGHTER = "#e9d5ff"  # 매우 연한 보라색 (배경용)
    
    # Secondary Colors - 보완색
    SECONDARY = "#06b6d4"        # 청록색 (시원한 느낌)
    SECONDARY_DARK = "#0891b2"   
    SECONDARY_LIGHT = "#67e8f9"
    
    # Accent Colors - 포인트 색상
    ACCENT = "#f59e0b"           # 따뜻한 주황색
    ACCENT_GREEN = "#10b981"     # 성공/긍정 녹색
    ACCENT_BLUE = "#3b82f6"      # 정보 파란색
    
    # Status Colors - 상태 표시 (채도를 낮춰 눈의 피로 감소)
    SUCCESS = "#059669"          # 진한 녹색 (너무 밝지 않게)
    SUCCESS_LIGHT = "#d1fae5"    # 연한 녹색 배경
    WARNING = "#d97706"          # 진한 주황색
    WARNING_LIGHT = "#fef3c7"    # 연한 주황색 배경
    ERROR = "#dc2626"            # 진한 빨간색
    ERROR_LIGHT = "#fee2e2"      # 연한 빨간색 배경
    INFO = "#2563eb"             # 진한 파란색
    INFO_LIGHT = "#dbeafe"       # 연한 파란색 배경
    
    # Neutral Colors - 회색 계열 (따뜻한 톤으로 눈의 피로 감소)
    GRAY_50 = "#fafaf9"
    GRAY_100 = "#f5f5f4"
    GRAY_200 = "#e7e5e4"
    GRAY_300 = "#d6d3d1"
    GRAY_400 = "#a8a29e"
    GRAY_500 = "#78716c"
    GRAY_600 = "#57534e"
    GRAY_700 = "#44403c"
    GRAY_800 = "#292524"
    GRAY_900 = "#1c1917"
    
    # Background Colors - 데스크톱 앱에 적합한 배경
    BG_PRIMARY = "#ffffff"       # 메인 배경 (순백색)
    BG_SECONDARY = "#fafaf9"     # 보조 배경 (약간 따뜻한 회색)
    BG_TERTIARY = "#f5f5f4"      # 카드 배경
    BG_HOVER = "#e7e5e4"         # 호버 상태
    
    # Text Colors - 높은 대비로 가독성 확보
    TEXT_PRIMARY = "#1c1917"     # 주 텍스트 (거의 검정)
    TEXT_SECONDARY = "#57534e"   # 보조 텍스트
    TEXT_TERTIARY = "#78716c"    # 설명 텍스트
    TEXT_DISABLED = "#a8a29e"    # 비활성 텍스트
    TEXT_ON_PRIMARY = "#ffffff"  # Primary 배경 위 텍스트
    
    # Border Colors - 명확한 경계선
    BORDER_DEFAULT = "#e7e5e4"   # 기본 경계선
    BORDER_HOVER = "#d6d3d1"     # 호버 시 경계선
    BORDER_FOCUS = "#a880ed"     # 포커스 시 경계선 (Primary)
    
    # Shadow Colors - 깊이감 표현
    SHADOW_SM = "rgba(0, 0, 0, 0.05)"
    SHADOW_MD = "rgba(0, 0, 0, 0.1)"
    SHADOW_LG = "rgba(0, 0, 0, 0.15)"
    SHADOW_XL = "rgba(0, 0, 0, 0.25)"


# ============================================================================
# 🔤 타이포그래피 (Typography)
# ============================================================================

class Typography:
    """폰트 및 텍스트 스타일 설정"""
    
    # Font Families
    FONT_FAMILY_PRIMARY = '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
    FONT_FAMILY_HEADING = '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
    FONT_FAMILY_MONO = '"JetBrains Mono", "Consolas", "Monaco", monospace'
    
    # Font Sizes - 데스크톱에 적합한 크기
    SIZE_XS = "0.75rem"    # 12px
    SIZE_SM = "0.875rem"   # 14px
    SIZE_BASE = "1rem"     # 16px
    SIZE_LG = "1.125rem"   # 18px
    SIZE_XL = "1.25rem"    # 20px
    SIZE_2XL = "1.5rem"    # 24px
    SIZE_3XL = "1.875rem"  # 30px
    SIZE_4XL = "2.25rem"   # 36px
    
    # Font Weights
    WEIGHT_NORMAL = "400"
    WEIGHT_MEDIUM = "500"
    WEIGHT_SEMIBOLD = "600"
    WEIGHT_BOLD = "700"
    
    # Line Heights
    LINE_HEIGHT_TIGHT = "1.25"
    LINE_HEIGHT_NORMAL = "1.5"
    LINE_HEIGHT_RELAXED = "1.75"
    
    # Letter Spacing
    LETTER_SPACING_TIGHT = "-0.025em"
    LETTER_SPACING_NORMAL = "0"
    LETTER_SPACING_WIDE = "0.025em"


# ============================================================================
# 🎯 컴포넌트 스타일 (Component Styles)
# ============================================================================

class ComponentStyles:
    """UI 컴포넌트별 스타일 정의"""
    
    # Button Styles
    BUTTON_STYLES = {
        "primary": {
            "bg": Colors.PRIMARY,
            "bg_hover": Colors.PRIMARY_DARK,
            "color": Colors.TEXT_ON_PRIMARY,
            "border": "none",
            "padding": "0.625rem 1.25rem",
            "border_radius": "0.5rem",
            "font_weight": Typography.WEIGHT_MEDIUM,
            "box_shadow": Colors.SHADOW_SM,
            "box_shadow_hover": Colors.SHADOW_MD,
            "transition": "all 0.2s ease"
        },
        "secondary": {
            "bg": Colors.BG_TERTIARY,
            "bg_hover": Colors.BG_HOVER,
            "color": Colors.TEXT_PRIMARY,
            "border": f"1px solid {Colors.BORDER_DEFAULT}",
            "padding": "0.625rem 1.25rem",
            "border_radius": "0.5rem",
            "font_weight": Typography.WEIGHT_MEDIUM,
            "box_shadow": "none",
            "box_shadow_hover": Colors.SHADOW_SM,
            "transition": "all 0.2s ease"
        },
        "ghost": {
            "bg": "transparent",
            "bg_hover": Colors.BG_HOVER,
            "color": Colors.TEXT_PRIMARY,
            "border": "none",
            "padding": "0.625rem 1.25rem",
            "border_radius": "0.5rem",
            "font_weight": Typography.WEIGHT_MEDIUM,
            "transition": "all 0.2s ease"
        }
    }
    
    # Input Styles
    INPUT_STYLES = {
        "default": {
            "bg": Colors.BG_PRIMARY,
            "border": f"1px solid {Colors.BORDER_DEFAULT}",
            "border_radius": "0.5rem",
            "padding": "0.625rem 0.875rem",
            "font_size": Typography.SIZE_BASE,
            "focus_border": Colors.PRIMARY,
            "focus_shadow": f"0 0 0 3px {Colors.PRIMARY}20",
            "placeholder_color": Colors.TEXT_TERTIARY
        }
    }
    
    # Card Styles
    CARD_STYLES = {
        "default": {
            "bg": Colors.BG_PRIMARY,
            "border": f"1px solid {Colors.BORDER_DEFAULT}",
            "border_radius": "0.75rem",
            "padding": "1.5rem",
            "box_shadow": Colors.SHADOW_SM
        },
        "elevated": {
            "bg": Colors.BG_PRIMARY,
            "border": "none",
            "border_radius": "0.75rem",
            "padding": "1.5rem",
            "box_shadow": Colors.SHADOW_MD
        }
    }
    
    # Badge Styles
    BADGE_STYLES = {
        "default": {
            "bg": Colors.BG_TERTIARY,
            "color": Colors.TEXT_SECONDARY,
            "padding": "0.25rem 0.75rem",
            "border_radius": "9999px",
            "font_size": Typography.SIZE_SM,
            "font_weight": Typography.WEIGHT_MEDIUM
        },
        "primary": {
            "bg": Colors.PRIMARY_LIGHTER,
            "color": Colors.PRIMARY_DARK,
            "padding": "0.25rem 0.75rem",
            "border_radius": "9999px",
            "font_size": Typography.SIZE_SM,
            "font_weight": Typography.WEIGHT_MEDIUM
        },
        "success": {
            "bg": Colors.SUCCESS_LIGHT,
            "color": Colors.SUCCESS,
            "padding": "0.25rem 0.75rem",
            "border_radius": "9999px",
            "font_size": Typography.SIZE_SM,
            "font_weight": Typography.WEIGHT_MEDIUM
        }
    }


# ============================================================================
# 📐 레이아웃 (Layout)
# ============================================================================

class Layout:
    """레이아웃 및 간격 설정"""
    
    # Container Widths
    CONTAINER_SM = "640px"
    CONTAINER_MD = "768px"
    CONTAINER_LG = "1024px"
    CONTAINER_XL = "1280px"
    CONTAINER_2XL = "1536px"
    
    # Spacing Scale
    SPACING = {
        "0": "0",
        "1": "0.25rem",   # 4px
        "2": "0.5rem",    # 8px
        "3": "0.75rem",   # 12px
        "4": "1rem",      # 16px
        "5": "1.25rem",   # 20px
        "6": "1.5rem",    # 24px
        "8": "2rem",      # 32px
        "10": "2.5rem",   # 40px
        "12": "3rem",     # 48px
        "16": "4rem",     # 64px
    }
    
    # Grid
    GRID_COLS = 12
    GRID_GAP = "1.5rem"
    
    # Breakpoints
    BREAKPOINTS = {
        "sm": "640px",
        "md": "768px",
        "lg": "1024px",
        "xl": "1280px",
        "2xl": "1536px"
    }


# ============================================================================
# 🎭 아이콘 (Icons)
# ============================================================================

class Icons:
    """자주 사용하는 아이콘 모음"""
    
    # Status Icons
    SUCCESS = "✅"
    ERROR = "❌"
    WARNING = "⚠️"
    INFO = "ℹ️"
    
    # Action Icons
    ADD = "➕"
    DELETE = "🗑️"
    EDIT = "✏️"
    SAVE = "💾"
    SEARCH = "🔍"
    FILTER = "🔽"
    REFRESH = "🔄"
    DOWNLOAD = "⬇️"
    UPLOAD = "⬆️"
    
    # Navigation Icons
    HOME = "🏠"
    BACK = "◀️"
    FORWARD = "▶️"
    MENU = "☰"
    CLOSE = "✖️"
    
    # Feature Icons
    EXPERIMENT = "🧪"
    DATA = "📊"
    CHART = "📈"
    REPORT = "📄"
    SETTINGS = "⚙️"
    USER = "👤"
    TEAM = "👥"
    NOTIFICATION = "🔔"
    HELP = "❓"
    AI = "🤖"


# ============================================================================
# 🎬 애니메이션 (Animations)
# ============================================================================

class Animations:
    """애니메이션 설정"""
    
    # Transitions
    TRANSITION_FAST = "all 0.15s ease"
    TRANSITION_BASE = "all 0.2s ease"
    TRANSITION_SLOW = "all 0.3s ease"
    
    # Hover Effects
    HOVER_SCALE = "transform: scale(1.02);"
    HOVER_SHADOW = f"box-shadow: {Colors.SHADOW_MD};"
    HOVER_BRIGHTNESS = "filter: brightness(1.05);"
    
    # Loading Animation
    LOADING_SPINNER = """
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        .spinner {
            animation: spin 1s linear infinite;
        }
    """


# ============================================================================
# 🎯 Streamlit CSS 생성 함수
# ============================================================================

def generate_streamlit_css() -> str:
    """Streamlit 앱용 커스텀 CSS 생성"""
    
    return f"""
    <style>
        /* 폰트 임포트 */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');
        
        /* 전역 스타일 */
        .stApp {{
            font-family: {Typography.FONT_FAMILY_PRIMARY};
            background-color: {Colors.BG_SECONDARY};
            color: {Colors.TEXT_PRIMARY};
        }}
        
        /* 헤더 스타일 */
        h1, h2, h3, h4, h5, h6 {{
            font-family: {Typography.FONT_FAMILY_HEADING};
            color: {Colors.TEXT_PRIMARY};
            font-weight: {Typography.WEIGHT_SEMIBOLD};
            line-height: {Typography.LINE_HEIGHT_TIGHT};
        }}
        
        /* 사이드바 스타일 */
        section[data-testid="stSidebar"] {{
            background-color: {Colors.BG_PRIMARY};
            border-right: 1px solid {Colors.BORDER_DEFAULT};
            box-shadow: {Colors.SHADOW_SM};
        }}
        
        /* 버튼 스타일 */
        .stButton > button {{
            background-color: {Colors.PRIMARY};
            color: {Colors.TEXT_ON_PRIMARY};
            border: none;
            padding: {ComponentStyles.BUTTON_STYLES['primary']['padding']};
            border-radius: {ComponentStyles.BUTTON_STYLES['primary']['border_radius']};
            font-weight: {ComponentStyles.BUTTON_STYLES['primary']['font_weight']};
            box-shadow: {ComponentStyles.BUTTON_STYLES['primary']['box_shadow']};
            transition: {ComponentStyles.BUTTON_STYLES['primary']['transition']};
            font-family: {Typography.FONT_FAMILY_PRIMARY};
        }}
        
        .stButton > button:hover {{
            background-color: {Colors.PRIMARY_DARK};
            box-shadow: {ComponentStyles.BUTTON_STYLES['primary']['box_shadow_hover']};
            transform: translateY(-1px);
        }}
        
        /* 입력 필드 스타일 */
        .stTextInput > div > div > input,
        .stSelectbox > div > div > select,
        .stTextArea > div > div > textarea {{
            background-color: {ComponentStyles.INPUT_STYLES['default']['bg']};
            border: {ComponentStyles.INPUT_STYLES['default']['border']};
            border-radius: {ComponentStyles.INPUT_STYLES['default']['border_radius']};
            padding: {ComponentStyles.INPUT_STYLES['default']['padding']};
            font-size: {ComponentStyles.INPUT_STYLES['default']['font_size']};
            font-family: {Typography.FONT_FAMILY_PRIMARY};
            transition: all 0.2s ease;
        }}
        
        .stTextInput > div > div > input:focus,
        .stSelectbox > div > div > select:focus,
        .stTextArea > div > div > textarea:focus {{
            border-color: {ComponentStyles.INPUT_STYLES['default']['focus_border']};
            box-shadow: {ComponentStyles.INPUT_STYLES['default']['focus_shadow']};
            outline: none;
        }}
        
        /* 메트릭 카드 스타일 */
        [data-testid="metric-container"] {{
            background-color: {Colors.BG_PRIMARY};
            padding: {Layout.SPACING['6']};
            border-radius: {ComponentStyles.CARD_STYLES['default']['border_radius']};
            border: {ComponentStyles.CARD_STYLES['default']['border']};
            box-shadow: {ComponentStyles.CARD_STYLES['default']['box_shadow']};
        }}
        
        /* 익스팬더 스타일 */
        .streamlit-expanderHeader {{
            background-color: {Colors.BG_TERTIARY};
            border: 1px solid {Colors.BORDER_DEFAULT};
            border-radius: 0.5rem;
            font-weight: {Typography.WEIGHT_MEDIUM};
            transition: all 0.2s ease;
        }}
        
        .streamlit-expanderHeader:hover {{
            background-color: {Colors.BG_HOVER};
            border-color: {Colors.BORDER_HOVER};
        }}
        
        /* 탭 스타일 */
        .stTabs [data-baseweb="tab-list"] {{
            gap: {Layout.SPACING['2']};
            border-bottom: 2px solid {Colors.BORDER_DEFAULT};
        }}
        
        .stTabs [data-baseweb="tab"] {{
            height: 3rem;
            padding: 0 {Layout.SPACING['6']};
            background-color: transparent;
            border: none;
            color: {Colors.TEXT_SECONDARY};
            font-weight: {Typography.WEIGHT_MEDIUM};
            transition: all 0.2s ease;
        }}
        
        .stTabs [aria-selected="true"] {{
            color: {Colors.PRIMARY};
            border-bottom: 3px solid {Colors.PRIMARY};
        }}
        
        /* 알림 스타일 */
        .stAlert {{
            border-radius: 0.5rem;
            border: 1px solid;
            font-weight: {Typography.WEIGHT_MEDIUM};
        }}
        
        /* 성공 알림 */
        .stSuccess, [data-testid="stSuccess"] {{
            background-color: {Colors.SUCCESS_LIGHT};
            color: {Colors.SUCCESS};
            border-color: {Colors.SUCCESS};
        }}
        
        /* 경고 알림 */
        .stWarning, [data-testid="stWarning"] {{
            background-color: {Colors.WARNING_LIGHT};
            color: {Colors.WARNING};
            border-color: {Colors.WARNING};
        }}
        
        /* 에러 알림 */
        .stError, [data-testid="stError"] {{
            background-color: {Colors.ERROR_LIGHT};
            color: {Colors.ERROR};
            border-color: {Colors.ERROR};
        }}
        
        /* 정보 알림 */
        .stInfo, [data-testid="stInfo"] {{
            background-color: {Colors.INFO_LIGHT};
            color: {Colors.INFO};
            border-color: {Colors.INFO};
        }}
        
        /* 코드 블록 스타일 */
        .stCodeBlock {{
            background-color: {Colors.GRAY_900};
            border-radius: 0.5rem;
            font-family: {Typography.FONT_FAMILY_MONO};
        }}
        
        /* 데이터프레임 스타일 */
        .dataframe {{
            border: 1px solid {Colors.BORDER_DEFAULT};
            border-radius: 0.5rem;
            overflow: hidden;
        }}
        
        /* 스크롤바 스타일 */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: {Colors.BG_SECONDARY};
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: {Colors.GRAY_400};
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: {Colors.GRAY_500};
        }}
        
        /* 커스텀 컨테이너 */
        .custom-container {{
            background-color: {Colors.BG_PRIMARY};
            border: 1px solid {Colors.BORDER_DEFAULT};
            border-radius: 0.75rem;
            padding: {Layout.SPACING['6']};
            margin-bottom: {Layout.SPACING['4']};
            box-shadow: {Colors.SHADOW_SM};
        }}
        
        /* 호버 효과가 있는 카드 */
        .hover-card {{
            transition: all 0.2s ease;
            cursor: pointer;
        }}
        
        .hover-card:hover {{
            transform: translateY(-2px);
            box-shadow: {Colors.SHADOW_MD};
            border-color: {Colors.PRIMARY};
        }}
        
        /* 그라데이션 텍스트 */
        .gradient-text {{
            background: linear-gradient(135deg, {Colors.PRIMARY} 0%, {Colors.SECONDARY} 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: {Typography.WEIGHT_BOLD};
        }}
        
        /* 로딩 애니메이션 */
        {Animations.LOADING_SPINNER}
        
        /* 페이드 인 애니메이션 */
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .fade-in {{
            animation: fadeIn 0.5s ease-out;
        }}
    </style>
    """


# ============================================================================
# 🎯 테마 적용 함수
# ============================================================================

def apply_theme():
    """Streamlit 앱에 테마 적용"""
    st.markdown(generate_streamlit_css(), unsafe_allow_html=True)


def get_color(color_name: str) -> str:
    """색상 이름으로 색상 값 가져오기"""
    return getattr(Colors, color_name.upper(), Colors.TEXT_PRIMARY)


def create_gradient_text(text: str, start_color: str = None, end_color: str = None) -> str:
    """그라데이션 텍스트 HTML 생성"""
    start = start_color or Colors.PRIMARY
    end = end_color or Colors.SECONDARY
    
    return f"""
    <span style="
        background: linear-gradient(135deg, {start} 0%, {end} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: {Typography.WEIGHT_BOLD};
    ">{text}</span>
    """


def create_custom_container(content: str, type: str = "default") -> str:
    """커스텀 컨테이너 HTML 생성"""
    styles = {
        "default": f"""
            background-color: {Colors.BG_PRIMARY};
            border: 1px solid {Colors.BORDER_DEFAULT};
            border-radius: 0.75rem;
            padding: {Layout.SPACING['6']};
            margin-bottom: {Layout.SPACING['4']};
            box-shadow: {Colors.SHADOW_SM};
        """,
        "elevated": f"""
            background-color: {Colors.BG_PRIMARY};
            border: none;
            border-radius: 0.75rem;
            padding: {Layout.SPACING['6']};
            margin-bottom: {Layout.SPACING['4']};
            box-shadow: {Colors.SHADOW_MD};
        """,
        "primary": f"""
            background-color: {Colors.PRIMARY_LIGHTER};
            border: 1px solid {Colors.PRIMARY_LIGHT};
            border-radius: 0.75rem;
            padding: {Layout.SPACING['6']};
            margin-bottom: {Layout.SPACING['4']};
            color: {Colors.PRIMARY_DARK};
        """
    }
    
    style = styles.get(type, styles["default"])
    
    return f"""
    <div style="{style}">
        {content}
    </div>
    """


# ============================================================================
# 📤 Export
# ============================================================================

__all__ = [
    'Colors',
    'Typography',
    'ComponentStyles',
    'Layout',
    'Icons',
    'Animations',
    'apply_theme',
    'get_color',
    'create_gradient_text',
    'create_custom_container',
    'generate_streamlit_css'
]
