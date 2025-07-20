"""
🎨 Universal DOE Platform - UI Theme Configuration
전체 플랫폼의 시각적 일관성을 위한 중앙 테마 설정
"""

from typing import Dict, List, Tuple
import streamlit as st

# ============================================================================
# 🎨 색상 시스템 (Color System)
# ============================================================================

class Colors:
    """플랫폼 전체 색상 팔레트"""
    
    # Primary Colors
    PRIMARY = "#1E88E5"          # 밝은 파란색 (신뢰, 과학)
    PRIMARY_DARK = "#1565C0"     # 진한 파란색
    PRIMARY_LIGHT = "#42A5F5"    # 연한 파란색
    
    # Secondary Colors  
    SECONDARY = "#00ACC1"        # 청록색 (혁신, 기술)
    SECONDARY_DARK = "#00838F"
    SECONDARY_LIGHT = "#26C6DA"
    
    # Accent Colors
    ACCENT = "#7C4DFF"          # 보라색 (창의성)
    ACCENT_ORANGE = "#FF6F00"   # 주황색 (활력)
    
    # Status Colors
    SUCCESS = "#4CAF50"         # 녹색
    WARNING = "#FFA726"         # 주황색
    ERROR = "#EF5350"           # 빨간색
    INFO = "#29B6F6"            # 하늘색
    
    # Neutral Colors
    BLACK = "#000000"
    WHITE = "#FFFFFF"
    GRAY_900 = "#212121"
    GRAY_800 = "#424242"
    GRAY_700 = "#616161"
    GRAY_600 = "#757575"
    GRAY_500 = "#9E9E9E"
    GRAY_400 = "#BDBDBD"
    GRAY_300 = "#E0E0E0"
    GRAY_200 = "#EEEEEE"
    GRAY_100 = "#F5F5F5"
    GRAY_50 = "#FAFAFA"
    
    # Background Colors
    BG_PRIMARY = "#FFFFFF"
    BG_SECONDARY = "#F8F9FA"
    BG_TERTIARY = "#F3F4F6"
    
    # Text Colors
    TEXT_PRIMARY = "#212121"
    TEXT_SECONDARY = "#616161"
    TEXT_DISABLED = "#9E9E9E"
    TEXT_ON_PRIMARY = "#FFFFFF"
    
    # 연구 분야별 테마 색상
    FIELD_COLORS = {
        "polymer": "#1E88E5",        # 파란색 - 고분자
        "inorganic": "#43A047",      # 녹색 - 무기재료
        "nano": "#E53935",           # 빨간색 - 나노재료
        "organic": "#FB8C00",        # 주황색 - 유기합성
        "composite": "#8E24AA",      # 보라색 - 복합재료
        "bio": "#00ACC1",            # 청록색 - 바이오재료
        "energy": "#FFD600",         # 노란색 - 에너지재료
        "environmental": "#00897B"   # 에메랄드 - 환경재료
    }
    
    # 차트 색상 팔레트
    CHART_COLORS = [
        "#1E88E5", "#43A047", "#E53935", "#FB8C00",
        "#8E24AA", "#00ACC1", "#FFD600", "#00897B",
        "#5E35B1", "#3949AB", "#1E88E5", "#039BE5"
    ]
    
    # 그라디언트
    GRADIENT_PRIMARY = "linear-gradient(135deg, #1E88E5 0%, #42A5F5 100%)"
    GRADIENT_SUCCESS = "linear-gradient(135deg, #43A047 0%, #66BB6A 100%)"
    GRADIENT_ACCENT = "linear-gradient(135deg, #7C4DFF 0%, #B388FF 100%)"


# ============================================================================
# 📝 타이포그래피 (Typography)
# ============================================================================

class Typography:
    """폰트 및 텍스트 스타일 설정"""
    
    # Font Families
    FONT_FAMILY_PRIMARY = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
    FONT_FAMILY_HEADING = "'Poppins', 'Inter', sans-serif"
    FONT_FAMILY_MONO = "'JetBrains Mono', 'Consolas', 'Monaco', monospace"
    
    # Font Sizes
    SIZE_XXXL = "2.5rem"    # 40px - 대형 헤더
    SIZE_XXL = "2rem"       # 32px - 페이지 제목
    SIZE_XL = "1.5rem"      # 24px - 섹션 제목
    SIZE_LG = "1.25rem"     # 20px - 부제목
    SIZE_MD = "1rem"        # 16px - 본문
    SIZE_SM = "0.875rem"    # 14px - 보조 텍스트
    SIZE_XS = "0.75rem"     # 12px - 캡션
    
    # Font Weights
    WEIGHT_LIGHT = 300
    WEIGHT_REGULAR = 400
    WEIGHT_MEDIUM = 500
    WEIGHT_SEMIBOLD = 600
    WEIGHT_BOLD = 700
    
    # Line Heights
    LINE_HEIGHT_TIGHT = 1.2
    LINE_HEIGHT_NORMAL = 1.5
    LINE_HEIGHT_RELAXED = 1.8
    
    # Letter Spacing
    LETTER_SPACING_TIGHT = "-0.02em"
    LETTER_SPACING_NORMAL = "0"
    LETTER_SPACING_WIDE = "0.02em"


# ============================================================================
# 🎯 컴포넌트 스타일 (Component Styles)
# ============================================================================

class ComponentStyles:
    """UI 컴포넌트별 스타일 정의"""
    
    # Buttons
    BUTTON_STYLES = {
        "primary": {
            "bg": Colors.PRIMARY,
            "color": Colors.WHITE,
            "hover_bg": Colors.PRIMARY_DARK,
            "padding": "0.75rem 1.5rem",
            "border_radius": "0.5rem",
            "font_weight": Typography.WEIGHT_MEDIUM,
            "transition": "all 0.2s ease"
        },
        "secondary": {
            "bg": Colors.WHITE,
            "color": Colors.PRIMARY,
            "hover_bg": Colors.GRAY_50,
            "border": f"2px solid {Colors.PRIMARY}",
            "padding": "0.75rem 1.5rem",
            "border_radius": "0.5rem"
        },
        "ghost": {
            "bg": "transparent",
            "color": Colors.PRIMARY,
            "hover_bg": Colors.GRAY_50,
            "padding": "0.5rem 1rem",
            "border_radius": "0.5rem"
        },
        "danger": {
            "bg": Colors.ERROR,
            "color": Colors.WHITE,
            "hover_bg": "#D32F2F",
            "padding": "0.75rem 1.5rem",
            "border_radius": "0.5rem"
        }
    }
    
    # Cards
    CARD_STYLES = {
        "default": {
            "bg": Colors.WHITE,
            "border": f"1px solid {Colors.GRAY_200}",
            "border_radius": "0.75rem",
            "padding": "1.5rem",
            "shadow": "0 1px 3px rgba(0,0,0,0.1)"
        },
        "elevated": {
            "bg": Colors.WHITE,
            "border": "none",
            "border_radius": "1rem",
            "padding": "2rem",
            "shadow": "0 4px 6px rgba(0,0,0,0.1)"
        },
        "interactive": {
            "bg": Colors.WHITE,
            "border": f"1px solid {Colors.GRAY_200}",
            "border_radius": "0.75rem",
            "padding": "1.5rem",
            "shadow": "0 1px 3px rgba(0,0,0,0.1)",
            "hover_shadow": "0 4px 12px rgba(0,0,0,0.15)",
            "transition": "all 0.3s ease"
        }
    }
    
    # Input Fields
    INPUT_STYLES = {
        "default": {
            "bg": Colors.WHITE,
            "border": f"1px solid {Colors.GRAY_300}",
            "border_radius": "0.5rem",
            "padding": "0.75rem 1rem",
            "font_size": Typography.SIZE_MD,
            "focus_border": Colors.PRIMARY,
            "focus_shadow": f"0 0 0 3px {Colors.PRIMARY}20"
        },
        "error": {
            "border": f"1px solid {Colors.ERROR}",
            "focus_border": Colors.ERROR,
            "focus_shadow": f"0 0 0 3px {Colors.ERROR}20"
        }
    }
    
    # Badges
    BADGE_STYLES = {
        "default": {
            "bg": Colors.GRAY_200,
            "color": Colors.GRAY_700,
            "padding": "0.25rem 0.75rem",
            "border_radius": "1rem",
            "font_size": Typography.SIZE_SM
        },
        "primary": {
            "bg": f"{Colors.PRIMARY}20",
            "color": Colors.PRIMARY_DARK
        },
        "success": {
            "bg": f"{Colors.SUCCESS}20",
            "color": "#2E7D32"
        },
        "warning": {
            "bg": f"{Colors.WARNING}20",
            "color": "#F57C00"
        }
    }


# ============================================================================
# 📐 레이아웃 시스템 (Layout System)
# ============================================================================

class Layout:
    """레이아웃 및 간격 설정"""
    
    # Container Widths
    MAX_WIDTH = "1200px"
    CONTENT_WIDTH = "900px"
    NARROW_WIDTH = "600px"
    
    # Spacing Scale
    SPACING = {
        "xxs": "0.25rem",   # 4px
        "xs": "0.5rem",     # 8px
        "sm": "0.75rem",    # 12px
        "md": "1rem",       # 16px
        "lg": "1.5rem",     # 24px
        "xl": "2rem",       # 32px
        "xxl": "3rem",      # 48px
        "xxxl": "4rem"      # 64px
    }
    
    # Grid System
    GRID_COLUMNS = 12
    GRID_GAP = "1rem"
    
    # Breakpoints
    BREAKPOINTS = {
        "mobile": "640px",
        "tablet": "768px",
        "desktop": "1024px",
        "wide": "1280px"
    }
    
    # Border Radius
    RADIUS = {
        "sm": "0.25rem",
        "md": "0.5rem",
        "lg": "0.75rem",
        "xl": "1rem",
        "full": "9999px"
    }


# ============================================================================
# 🎭 아이콘 및 이모지 (Icons & Emojis)
# ============================================================================

class Icons:
    """아이콘 및 이모지 매핑"""
    
    # 기능별 아이콘
    FEATURES = {
        "home": "🏠",
        "experiment": "🧪",
        "analysis": "📊",
        "collaboration": "👥",
        "settings": "⚙️",
        "help": "❓",
        "search": "🔍",
        "save": "💾",
        "export": "📤",
        "import": "📥",
        "delete": "🗑️",
        "edit": "✏️",
        "add": "➕",
        "remove": "➖",
        "refresh": "🔄",
        "filter": "🔽",
        "sort": "↕️",
        "calendar": "📅",
        "clock": "⏰",
        "notification": "🔔",
        "lock": "🔒",
        "unlock": "🔓",
        "user": "👤",
        "team": "👥",
        "document": "📄",
        "folder": "📁",
        "database": "🗄️",
        "cloud": "☁️",
        "download": "⬇️",
        "upload": "⬆️",
        "star": "⭐",
        "heart": "❤️",
        "flag": "🚩",
        "warning": "⚠️",
        "info": "ℹ️",
        "success": "✅",
        "error": "❌"
    }
    
    # 연구 분야별 아이콘
    RESEARCH_FIELDS = {
        "polymer": "🧬",
        "inorganic": "💎",
        "nano": "⚛️",
        "organic": "🧪",
        "composite": "🔗",
        "bio": "🧫",
        "energy": "🔋",
        "environmental": "🌱",
        "general": "🔬"
    }
    
    # 상태 아이콘
    STATUS = {
        "pending": "⏳",
        "running": "🔄",
        "completed": "✅",
        "failed": "❌",
        "paused": "⏸️",
        "cancelled": "🚫"
    }
    
    # AI 모델 아이콘
    AI_MODELS = {
        "gemini": "✨",
        "grok": "🚀",
        "groq": "⚡",
        "deepseek": "🔍",
        "sambanova": "🌊",
        "huggingface": "🤗"
    }


# ============================================================================
# 🎬 애니메이션 (Animations)
# ============================================================================

class Animations:
    """애니메이션 및 전환 효과"""
    
    # Transitions
    TRANSITION_DEFAULT = "all 0.3s ease"
    TRANSITION_FAST = "all 0.15s ease"
    TRANSITION_SLOW = "all 0.5s ease"
    
    # Loading Animations
    LOADING_SPINNER = """
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .spinner {
            animation: spin 1s linear infinite;
        }
    """
    
    LOADING_PULSE = """
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .pulse {
            animation: pulse 2s ease-in-out infinite;
        }
    """
    
    # Hover Effects
    HOVER_SCALE = "transform: scale(1.05);"
    HOVER_SHADOW = "box-shadow: 0 8px 16px rgba(0,0,0,0.15);"
    HOVER_BRIGHTNESS = "filter: brightness(1.1);"


# ============================================================================
# 🎯 특수 스타일 (Special Styles)
# ============================================================================

class SpecialStyles:
    """코드 블록, 수식, 테이블 등 특수 요소 스타일"""
    
    # Code Block Styles
    CODE_BLOCK = {
        "bg": "#282C34",
        "color": "#ABB2BF",
        "padding": "1rem",
        "border_radius": "0.5rem",
        "font_family": Typography.FONT_FAMILY_MONO,
        "font_size": Typography.SIZE_SM,
        "line_height": Typography.LINE_HEIGHT_RELAXED,
        "overflow": "auto"
    }
    
    # Syntax Highlighting
    SYNTAX_COLORS = {
        "keyword": "#C678DD",
        "string": "#98C379",
        "number": "#D19A66",
        "comment": "#5C6370",
        "function": "#61AFEF",
        "variable": "#E06C75"
    }
    
    # Math Formula Styles
    MATH_FORMULA = {
        "font_size": Typography.SIZE_MD,
        "color": Colors.TEXT_PRIMARY,
        "padding": "0.5rem 0",
        "text_align": "center"
    }
    
    # Table Styles
    TABLE = {
        "header_bg": Colors.GRAY_100,
        "header_color": Colors.TEXT_PRIMARY,
        "border": f"1px solid {Colors.GRAY_200}",
        "row_hover": Colors.GRAY_50,
        "cell_padding": "0.75rem",
        "font_size": Typography.SIZE_SM
    }
    
    # Scrollbar Styles
    SCROLLBAR = """
        /* Webkit browsers */
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
    """


# ============================================================================
# 🎨 Streamlit CSS 생성기 (Streamlit CSS Generator)
# ============================================================================

def generate_streamlit_theme() -> str:
    """Streamlit 커스텀 CSS 생성"""
    
    return f"""
    <style>
        /* 폰트 임포트 */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        
        /* 전역 스타일 */
        html, body, [class*="css"] {{
            font-family: {Typography.FONT_FAMILY_PRIMARY};
            color: {Colors.TEXT_PRIMARY};
        }}
        
        /* 헤더 스타일 */
        h1, h2, h3, h4, h5, h6 {{
            font-family: {Typography.FONT_FAMILY_HEADING};
            color: {Colors.TEXT_PRIMARY};
            font-weight: {Typography.WEIGHT_SEMIBOLD};
        }}
        
        /* Streamlit 메인 컨테이너 */
        .stApp {{
            background-color: {Colors.BG_SECONDARY};
        }}
        
        /* 사이드바 스타일 */
        section[data-testid="stSidebar"] {{
            background-color: {Colors.WHITE};
            border-right: 1px solid {Colors.GRAY_200};
        }}
        
        /* 버튼 스타일 */
        .stButton > button {{
            background-color: {Colors.PRIMARY};
            color: {Colors.WHITE};
            border: none;
            padding: {ComponentStyles.BUTTON_STYLES['primary']['padding']};
            border-radius: {ComponentStyles.BUTTON_STYLES['primary']['border_radius']};
            font-weight: {ComponentStyles.BUTTON_STYLES['primary']['font_weight']};
            transition: {ComponentStyles.BUTTON_STYLES['primary']['transition']};
        }}
        
        .stButton > button:hover {{
            background-color: {Colors.PRIMARY_DARK};
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        
        /* 입력 필드 스타일 */
        .stTextInput > div > div > input,
        .stSelectbox > div > div > div,
        .stTextArea > div > div > textarea {{
            background-color: {ComponentStyles.INPUT_STYLES['default']['bg']};
            border: {ComponentStyles.INPUT_STYLES['default']['border']};
            border-radius: {ComponentStyles.INPUT_STYLES['default']['border_radius']};
            padding: {ComponentStyles.INPUT_STYLES['default']['padding']};
            font-size: {ComponentStyles.INPUT_STYLES['default']['font_size']};
            transition: all 0.2s ease;
        }}
        
        .stTextInput > div > div > input:focus,
        .stSelectbox > div > div > div:focus,
        .stTextArea > div > div > textarea:focus {{
            border-color: {ComponentStyles.INPUT_STYLES['default']['focus_border']};
            box-shadow: {ComponentStyles.INPUT_STYLES['default']['focus_shadow']};
            outline: none;
        }}
        
        /* 메트릭 카드 스타일 */
        [data-testid="metric-container"] {{
            background-color: {Colors.WHITE};
            border: 1px solid {Colors.GRAY_200};
            padding: 1.5rem;
            border-radius: 0.75rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        
        /* Expander 스타일 */
        .streamlit-expanderHeader {{
            background-color: {Colors.GRAY_50};
            border: 1px solid {Colors.GRAY_200};
            border-radius: 0.5rem;
            font-weight: {Typography.WEIGHT_MEDIUM};
        }}
        
        /* 탭 스타일 */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 2rem;
            border-bottom: 2px solid {Colors.GRAY_200};
        }}
        
        .stTabs [data-baseweb="tab"] {{
            height: 3rem;
            padding: 0 1rem;
            background-color: transparent;
            border: none;
            color: {Colors.TEXT_SECONDARY};
            font-weight: {Typography.WEIGHT_MEDIUM};
        }}
        
        .stTabs [aria-selected="true"] {{
            color: {Colors.PRIMARY};
            border-bottom: 3px solid {Colors.PRIMARY};
        }}
        
        /* 성공/오류 메시지 */
        .stSuccess {{
            background-color: {Colors.SUCCESS}20;
            color: {Colors.SUCCESS};
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid {Colors.SUCCESS};
        }}
        
        .stError {{
            background-color: {Colors.ERROR}20;
            color: {Colors.ERROR};
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid {Colors.ERROR};
        }}
        
        /* 스크롤바 스타일 */
        {SpecialStyles.SCROLLBAR}
        
        /* 애니메이션 */
        {Animations.LOADING_SPINNER}
        {Animations.LOADING_PULSE}
        
        /* 코드 블록 스타일 */
        .stCodeBlock {{
            background-color: {SpecialStyles.CODE_BLOCK['bg']};
            border-radius: {SpecialStyles.CODE_BLOCK['border_radius']};
            padding: {SpecialStyles.CODE_BLOCK['padding']};
        }}
        
        /* 커스텀 클래스 */
        .primary-button {{
            background-color: {Colors.PRIMARY} !important;
            color: {Colors.WHITE} !important;
        }}
        
        .secondary-button {{
            background-color: {Colors.WHITE} !important;
            color: {Colors.PRIMARY} !important;
            border: 2px solid {Colors.PRIMARY} !important;
        }}
        
        .danger-button {{
            background-color: {Colors.ERROR} !important;
            color: {Colors.WHITE} !important;
        }}
        
        .card {{
            background-color: {ComponentStyles.CARD_STYLES['default']['bg']};
            border: {ComponentStyles.CARD_STYLES['default']['border']};
            border-radius: {ComponentStyles.CARD_STYLES['default']['border_radius']};
            padding: {ComponentStyles.CARD_STYLES['default']['padding']};
            box-shadow: {ComponentStyles.CARD_STYLES['default']['shadow']};
            margin-bottom: 1rem;
        }}
        
        .card-elevated {{
            background-color: {ComponentStyles.CARD_STYLES['elevated']['bg']};
            border: {ComponentStyles.CARD_STYLES['elevated']['border']};
            border-radius: {ComponentStyles.CARD_STYLES['elevated']['border_radius']};
            padding: {ComponentStyles.CARD_STYLES['elevated']['padding']};
            box-shadow: {ComponentStyles.CARD_STYLES['elevated']['shadow']};
        }}
        
        /* 연구 분야별 색상 클래스 */
        .field-polymer {{ color: {Colors.FIELD_COLORS['polymer']}; }}
        .field-inorganic {{ color: {Colors.FIELD_COLORS['inorganic']}; }}
        .field-nano {{ color: {Colors.FIELD_COLORS['nano']}; }}
        .field-organic {{ color: {Colors.FIELD_COLORS['organic']}; }}
        .field-composite {{ color: {Colors.FIELD_COLORS['composite']}; }}
        .field-bio {{ color: {Colors.FIELD_COLORS['bio']}; }}
        .field-energy {{ color: {Colors.FIELD_COLORS['energy']}; }}
        .field-environmental {{ color: {Colors.FIELD_COLORS['environmental']}; }}
    </style>
    """


# ============================================================================
# 🎯 테마 적용 함수 (Theme Application)
# ============================================================================

def apply_theme():
    """Streamlit 앱에 테마 적용"""
    st.markdown(generate_streamlit_theme(), unsafe_allow_html=True)


def get_field_color(field: str) -> str:
    """연구 분야별 색상 반환"""
    return Colors.FIELD_COLORS.get(field.lower(), Colors.PRIMARY)


def get_status_color(status: str) -> str:
    """상태별 색상 반환"""
    status_colors = {
        "success": Colors.SUCCESS,
        "warning": Colors.WARNING,
        "error": Colors.ERROR,
        "info": Colors.INFO,
        "pending": Colors.WARNING,
        "completed": Colors.SUCCESS,
        "failed": Colors.ERROR
    }
    return status_colors.get(status.lower(), Colors.GRAY_500)


def create_gradient_text(text: str, gradient: str = Colors.GRADIENT_PRIMARY) -> str:
    """그라디언트 텍스트 생성"""
    return f"""
    <span style="
        background: {gradient};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: {Typography.WEIGHT_BOLD};
    ">{text}</span>
    """


def create_badge(text: str, type: str = "default") -> str:
    """배지 HTML 생성"""
    style = ComponentStyles.BADGE_STYLES.get(type, ComponentStyles.BADGE_STYLES["default"])
    
    return f"""
    <span style="
        background-color: {style.get('bg', Colors.GRAY_200)};
        color: {style.get('color', Colors.GRAY_700)};
        padding: {style.get('padding', '0.25rem 0.75rem')};
        border-radius: {style.get('border_radius', '1rem')};
        font-size: {style.get('font_size', Typography.SIZE_SM)};
        font-weight: {Typography.WEIGHT_MEDIUM};
        display: inline-block;
    ">{text}</span>
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
    'SpecialStyles',
    'apply_theme',
    'get_field_color',
    'get_status_color',
    'create_gradient_text',
    'create_badge'
]
