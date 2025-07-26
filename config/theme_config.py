"""
config/theme_config.py
=======================
Universal DOE Platform - UI 테마 설정
Material Design 3 기반 적응형 테마 시스템
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import streamlit as st
import plotly.io as pio
import plotly.graph_objects as go


# ===========================================================================
# 🎨 Material Design 3 색상 시스템
# ===========================================================================

@dataclass
class ColorPalette:
    """Material Design 3 색상 팔레트"""
    # Primary Colors
    primary: str = "#1976D2"            # Material Blue 700
    primary_light: str = "#42A5F5"      # Material Blue 400
    primary_dark: str = "#0D47A1"       # Material Blue 900
    on_primary: str = "#FFFFFF"
    
    # Secondary Colors
    secondary: str = "#00796B"          # Material Teal 700
    secondary_light: str = "#26A69A"    # Material Teal 400
    secondary_dark: str = "#004D40"     # Material Teal 900
    on_secondary: str = "#FFFFFF"
    
    # Tertiary Colors
    tertiary: str = "#F57C00"           # Material Orange 700
    tertiary_light: str = "#FFB74D"     # Material Orange 300
    tertiary_dark: str = "#E65100"      # Material Orange 900
    on_tertiary: str = "#000000"
    
    # Error Colors
    error: str = "#D32F2F"              # Material Red 700
    error_light: str = "#EF5350"        # Material Red 400
    error_dark: str = "#B71C1C"         # Material Red 900
    on_error: str = "#FFFFFF"
    
    # Warning Colors
    warning: str = "#F57C00"            # Material Orange 700
    warning_light: str = "#FFB74D"      # Material Orange 300
    warning_dark: str = "#E65100"       # Material Orange 900
    on_warning: str = "#000000"
    
    # Success Colors
    success: str = "#388E3C"            # Material Green 700
    success_light: str = "#66BB6A"      # Material Green 400
    success_dark: str = "#1B5E20"       # Material Green 900
    on_success: str = "#FFFFFF"
    
    # Info Colors
    info: str = "#0288D1"               # Material Light Blue 700
    info_light: str = "#4FC3F7"         # Material Light Blue 300
    info_dark: str = "#01579B"          # Material Light Blue 900
    on_info: str = "#FFFFFF"
    
    # Surface Colors
    background: str = "#FAFAFA"         # Material Grey 50
    on_background: str = "#212121"      # Material Grey 900
    surface: str = "#FFFFFF"
    surface_variant: str = "#F5F5F5"    # Material Grey 100
    on_surface: str = "#212121"
    on_surface_variant: str = "#616161" # Material Grey 700
    
    # Outline Colors
    outline: str = "#BDBDBD"            # Material Grey 400
    outline_variant: str = "#E0E0E0"    # Material Grey 300
    
    # Chart Colors (순서대로 사용)
    chart_colors: List[str] = None
    
    def __post_init__(self):
        if self.chart_colors is None:
            self.chart_colors = [
                self.primary,
                self.secondary,
                self.tertiary,
                self.error,
                self.warning,
                self.success,
                "#9C27B0",  # Purple
                "#FF5722",  # Deep Orange
                "#795548",  # Brown
                "#607D8B"   # Blue Grey
            ]


# ===========================================================================
# 🎯 레벨별 테마 정의
# ===========================================================================

class UserLevel(Enum):
    """사용자 레벨"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ThemeMode(Enum):
    """테마 모드"""
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"


# 레벨별 테마 설정
LEVEL_THEMES = {
    UserLevel.BEGINNER: {
        "name": "🌱 초급자 테마",
        "description": "친근하고 따뜻한 색상, 큰 UI 요소, 많은 가이드",
        "colors": {
            "primary": "#4CAF50",        # 친근한 초록색
            "secondary": "#2196F3",      # 밝은 파란색
            "accent": "#FF9800",         # 따뜻한 주황색
            "background": "#FAFAFA",
            "surface": "#FFFFFF",
            "text_primary": "#212121",
            "text_secondary": "#757575",
        },
        "ui": {
            "border_radius": 16,         # 둥근 모서리
            "button_size": "large",      # 큰 버튼
            "spacing": "relaxed",        # 넓은 간격
            "font_size_base": 16,        # 큰 글자
            "animation_speed": "normal", # 부드러운 애니메이션
            "show_tooltips": True,       # 툴팁 항상 표시
            "show_guides": True,         # 가이드 표시
            "complexity": "simple"       # 단순한 UI
        }
    },
    
    UserLevel.INTERMEDIATE: {
        "name": "🌿 중급자 테마",
        "description": "균형잡힌 전문적 색상, 표준 UI 요소",
        "colors": {
            "primary": "#1976D2",        # 전문적인 파란색
            "secondary": "#00796B",      # 차분한 청록색
            "accent": "#F57C00",         # 강조 주황색
            "background": "#FAFAFA",
            "surface": "#FFFFFF",
            "text_primary": "#212121",
            "text_secondary": "#616161",
        },
        "ui": {
            "border_radius": 12,
            "button_size": "medium",
            "spacing": "comfortable",
            "font_size_base": 15,
            "animation_speed": "fast",
            "show_tooltips": "hover",    # 호버시만 툴팁
            "show_guides": False,
            "complexity": "standard"
        }
    },
    
    UserLevel.ADVANCED: {
        "name": "🎯 고급자 테마",
        "description": "세련된 색상, 컴팩트한 UI",
        "colors": {
            "primary": "#5E35B1",        # 고급스러운 보라색
            "secondary": "#00897B",      # 깊은 청록색
            "accent": "#FF6F00",         # 진한 주황색
            "background": "#F5F5F5",
            "surface": "#FFFFFF",
            "text_primary": "#212121",
            "text_secondary": "#616161",
        },
        "ui": {
            "border_radius": 8,
            "button_size": "small",
            "spacing": "compact",
            "font_size_base": 14,
            "animation_speed": "instant",
            "show_tooltips": False,
            "show_guides": False,
            "complexity": "advanced"
        }
    },
    
    UserLevel.EXPERT: {
        "name": "🚀 전문가 테마",
        "description": "미니멀한 디자인, 최대 효율성",
        "colors": {
            "primary": "#424242",        # 중성적인 회색
            "secondary": "#37474F",      # 블루그레이
            "accent": "#D32F2F",         # 강렬한 빨간색
            "background": "#EEEEEE",
            "surface": "#FAFAFA",
            "text_primary": "#212121",
            "text_secondary": "#616161",
        },
        "ui": {
            "border_radius": 4,
            "button_size": "small",
            "spacing": "dense",
            "font_size_base": 13,
            "animation_speed": "none",
            "show_tooltips": False,
            "show_guides": False,
            "complexity": "expert"
        }
    }
}


# ===========================================================================
# 🌙 다크모드 색상 오버라이드
# ===========================================================================

DARK_MODE_OVERRIDES = {
    UserLevel.BEGINNER: {
        "background": "#121212",
        "surface": "#1E1E1E",
        "surface_variant": "#2C2C2C",
        "text_primary": "#E0E0E0",
        "text_secondary": "#BDBDBD",
        "primary": "#81C784",        # 부드러운 초록
        "secondary": "#64B5F6",      # 부드러운 파란
    },
    
    UserLevel.INTERMEDIATE: {
        "background": "#0A0A0A",
        "surface": "#1A1A1A",
        "surface_variant": "#252525",
        "text_primary": "#E0E0E0",
        "text_secondary": "#BDBDBD",
        "primary": "#42A5F5",
        "secondary": "#26A69A",
    },
    
    UserLevel.ADVANCED: {
        "background": "#000000",
        "surface": "#121212",
        "surface_variant": "#1E1E1E",
        "text_primary": "#FFFFFF",
        "text_secondary": "#BDBDBD",
        "primary": "#9575CD",
        "secondary": "#4DB6AC",
    },
    
    UserLevel.EXPERT: {
        "background": "#000000",
        "surface": "#0A0A0A",
        "surface_variant": "#141414",
        "text_primary": "#FFFFFF",
        "text_secondary": "#9E9E9E",
        "primary": "#757575",
        "secondary": "#546E7A",
    }
}


# ===========================================================================
# 📐 레이아웃 및 간격 시스템
# ===========================================================================

@dataclass
class Spacing:
    """간격 시스템 (8px 기반)"""
    none: int = 0
    xs: int = 4     # 0.25rem
    sm: int = 8     # 0.5rem
    md: int = 16    # 1rem
    lg: int = 24    # 1.5rem
    xl: int = 32    # 2rem
    xxl: int = 48   # 3rem
    xxxl: int = 64  # 4rem


@dataclass
class Typography:
    """타이포그래피 시스템"""
    # Font Families
    font_family_base: str = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
    font_family_heading: str = "'Pretendard', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
    font_family_mono: str = "'JetBrains Mono', 'Fira Code', Consolas, Monaco, 'Courier New', monospace"
    
    # Font Sizes (rem)
    size_xs: float = 0.75    # 12px
    size_sm: float = 0.875   # 14px
    size_base: float = 1     # 16px
    size_lg: float = 1.125   # 18px
    size_xl: float = 1.25    # 20px
    size_2xl: float = 1.5    # 24px
    size_3xl: float = 1.875  # 30px
    size_4xl: float = 2.25   # 36px
    size_5xl: float = 3      # 48px
    
    # Font Weights
    weight_light: int = 300
    weight_normal: int = 400
    weight_medium: int = 500
    weight_semibold: int = 600
    weight_bold: int = 700
    
    # Line Heights
    line_height_tight: float = 1.25
    line_height_normal: float = 1.5
    line_height_relaxed: float = 1.75
    line_height_loose: float = 2


@dataclass
class Elevation:
    """그림자 시스템 (Material Design)"""
    none: str = "none"
    level1: str = "0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24)"
    level2: str = "0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23)"
    level3: str = "0 10px 20px rgba(0,0,0,0.19), 0 6px 6px rgba(0,0,0,0.23)"
    level4: str = "0 14px 28px rgba(0,0,0,0.25), 0 10px 10px rgba(0,0,0,0.22)"
    level5: str = "0 19px 38px rgba(0,0,0,0.30), 0 15px 12px rgba(0,0,0,0.22)"


@dataclass
class BorderRadius:
    """모서리 둥글기"""
    none: int = 0
    xs: int = 2
    sm: int = 4
    md: int = 8
    lg: int = 12
    xl: int = 16
    xxl: int = 24
    full: int = 9999


@dataclass
class Breakpoints:
    """반응형 브레이크포인트"""
    mobile: int = 600      # < 600px
    tablet: int = 960      # 600px - 960px
    desktop: int = 1280    # 960px - 1280px
    wide: int = 1920       # > 1280px


# ===========================================================================
# 🎨 테마 클래스
# ===========================================================================

class Theme:
    """테마 관리 클래스"""
    
    def __init__(self, mode: ThemeMode = ThemeMode.LIGHT):
        self.mode = mode
        self.colors = self._get_colors(mode)
        self.typography = Typography()
        self.spacing = Spacing()
        self.elevation = Elevation()
        self.border_radius = BorderRadius()
        self.breakpoints = Breakpoints()
        
    def _get_colors(self, mode: ThemeMode) -> ColorPalette:
        """테마 모드에 따른 색상 팔레트 반환"""
        if mode == ThemeMode.DARK:
            return self._get_dark_colors()
        else:
            return ColorPalette()  # Light mode is default
    
    def _get_dark_colors(self) -> ColorPalette:
        """다크 모드 색상 팔레트"""
        return ColorPalette(
            # Primary Colors
            primary="#90CAF9",          # Material Blue 200
            primary_light="#BBDEFB",    # Material Blue 100
            primary_dark="#42A5F5",     # Material Blue 400
            on_primary="#000000",
            
            # Surface Colors
            background="#121212",
            on_background="#E0E0E0",
            surface="#1E1E1E",
            surface_variant="#2C2C2C",
            on_surface="#E0E0E0",
            on_surface_variant="#BDBDBD",
            
            # Outline Colors
            outline="#616161",
            outline_variant="#424242",
        )
    
    def get_css(self) -> str:
        """완전한 CSS 스타일시트 생성"""
        c = self.colors
        t = self.typography
        s = self.spacing
        e = self.elevation
        r = self.border_radius
        
        return f"""
        <style>
        /* ========== CSS 변수 정의 ========== */
        :root {{
            /* Colors */
            --color-primary: {c.primary};
            --color-primary-light: {c.primary_light};
            --color-primary-dark: {c.primary_dark};
            --color-secondary: {c.secondary};
            --color-error: {c.error};
            --color-warning: {c.warning};
            --color-success: {c.success};
            --color-info: {c.info};
            --color-background: {c.background};
            --color-surface: {c.surface};
            --color-on-surface: {c.on_surface};
            --color-outline: {c.outline};
            
            /* Typography */
            --font-family-base: {t.font_family_base};
            --font-family-heading: {t.font_family_heading};
            --font-family-mono: {t.font_family_mono};
            
            /* Spacing */
            --spacing-xs: {s.xs}px;
            --spacing-sm: {s.sm}px;
            --spacing-md: {s.md}px;
            --spacing-lg: {s.lg}px;
            --spacing-xl: {s.xl}px;
            
            /* Border Radius */
            --radius-sm: {r.sm}px;
            --radius-md: {r.md}px;
            --radius-lg: {r.lg}px;
            
            /* Shadows */
            --shadow-sm: {e.level1};
            --shadow-md: {e.level2};
            --shadow-lg: {e.level3};
        }}
        
        /* ========== 기본 스타일 ========== */
        .stApp {{
            background-color: var(--color-background);
            color: var(--color-on-surface);
            font-family: var(--font-family-base);
        }}
        
        /* ========== 버튼 스타일 ========== */
        .stButton > button {{
            background-color: var(--color-primary);
            color: white;
            border: none;
            border-radius: var(--radius-md);
            padding: var(--spacing-sm) var(--spacing-md);
            font-weight: 500;
            transition: all 0.2s ease;
            box-shadow: var(--shadow-sm);
        }}
        
        .stButton > button:hover {{
            background-color: var(--color-primary-dark);
            box-shadow: var(--shadow-md);
            transform: translateY(-2px);
        }}
        
        /* ========== 입력 필드 스타일 ========== */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div > select {{
            background-color: var(--color-surface);
            border: 1px solid var(--color-outline);
            border-radius: var(--radius-sm);
            color: var(--color-on-surface);
            padding: var(--spacing-sm);
            transition: border-color 0.2s ease;
        }}
        
        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus,
        .stSelectbox > div > div > select:focus {{
            border-color: var(--color-primary);
            outline: none;
            box-shadow: 0 0 0 3px {c.primary}20;
        }}
        
        /* ========== 카드 스타일 ========== */
        .element-container {{
            background-color: var(--color-surface);
            border-radius: var(--radius-md);
            padding: var(--spacing-md);
            margin-bottom: var(--spacing-md);
            box-shadow: var(--shadow-sm);
        }}
        
        /* ========== 사이드바 스타일 ========== */
        .css-1d391kg {{
            background-color: var(--color-surface-variant);
        }}
        
        /* ========== 메트릭 카드 스타일 ========== */
        div[data-testid="metric-container"] {{
            background-color: var(--color-surface);
            border: 1px solid var(--color-outline-variant);
            padding: var(--spacing-md);
            border-radius: var(--radius-md);
            box-shadow: var(--shadow-sm);
        }}
        
        /* ========== 탭 스타일 ========== */
        .stTabs [data-baseweb="tab-list"] {{
            gap: var(--spacing-sm);
        }}
        
        .stTabs [data-baseweb="tab"] {{
            height: 3rem;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: var(--radius-md);
            color: var(--color-on-surface-variant);
            padding: 0 var(--spacing-md);
        }}
        
        .stTabs [aria-selected="true"] {{
            background-color: var(--color-primary);
            color: white;
        }}
        
        /* ========== 애니메이션 ========== */
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .element-container {{
            animation: fadeIn 0.3s ease-out;
        }}
        
        /* ========== 커스텀 클래스 ========== */
        .primary-button {{
            background-color: var(--color-primary) !important;
            color: white !important;
        }}
        
        .secondary-button {{
            background-color: var(--color-secondary) !important;
            color: white !important;
        }}
        
        .success-message {{
            background-color: {c.success}10;
            border-left: 4px solid var(--color-success);
            padding: var(--spacing-md);
            border-radius: var(--radius-sm);
            margin: var(--spacing-md) 0;
        }}
        
        .warning-message {{
            background-color: {c.warning}10;
            border-left: 4px solid var(--color-warning);
            padding: var(--spacing-md);
            border-radius: var(--radius-sm);
            margin: var(--spacing-md) 0;
        }}
        
        .error-message {{
            background-color: {c.error}10;
            border-left: 4px solid var(--color-error);
            padding: var(--spacing-md);
            border-radius: var(--radius-sm);
            margin: var(--spacing-md) 0;
        }}
        
        .info-message {{
            background-color: {c.info}10;
            border-left: 4px solid var(--color-info);
            padding: var(--spacing-md);
            border-radius: var(--radius-sm);
            margin: var(--spacing-md) 0;
        }}
        
        /* ========== 툴팁 스타일 ========== */
        .tooltip {{
            position: relative;
            display: inline-block;
        }}
        
        .tooltip .tooltiptext {{
            visibility: hidden;
            width: 200px;
            background-color: rgba(0, 0, 0, 0.9);
            color: #fff;
            text-align: center;
            border-radius: var(--radius-sm);
            padding: var(--spacing-sm);
            position: absolute;
            z-index: 1000;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 0.875rem;
        }}
        
        .tooltip:hover .tooltiptext {{
            visibility: visible;
            opacity: 1;
        }}
        
        /* ========== 프로그레스 바 스타일 ========== */
        .stProgress > div > div > div > div {{
            background-color: var(--color-primary);
        }}
        
        /* ========== 반응형 디자인 ========== */
        @media (max-width: {self.breakpoints.mobile}px) {{
            .element-container {{
                padding: var(--spacing-sm);
            }}
            
            .stButton > button {{
                width: 100%;
            }}
        }}
        
        @media (max-width: {self.breakpoints.tablet}px) {{
            .row-widget {{
                flex-direction: column;
            }}
        }}
        </style>
        """
    
    def configure_plotly(self):
        """Plotly 차트 테마 설정"""
        plotly_template = go.layout.Template()
        
        # 레이아웃 설정
        plotly_template.layout = go.Layout(
            paper_bgcolor=self.colors.background,
            plot_bgcolor=self.colors.surface,
            font=dict(
                family=self.typography.font_family_base,
                size=14,
                color=self.colors.on_surface
            ),
            title=dict(
                font=dict(
                    family=self.typography.font_family_heading,
                    size=20,
                    color=self.colors.on_surface
                )
            ),
            colorway=self.colors.chart_colors,
            hovermode='closest',
            hoverlabel=dict(
                bgcolor=self.colors.surface,
                bordercolor=self.colors.outline,
                font=dict(color=self.colors.on_surface)
            ),
            xaxis=dict(
                gridcolor=self.colors.outline_variant,
                linecolor=self.colors.outline,
                tickfont=dict(color=self.colors.on_surface_variant),
                title=dict(font=dict(color=self.colors.on_surface))
            ),
            yaxis=dict(
                gridcolor=self.colors.outline_variant,
                linecolor=self.colors.outline,
                tickfont=dict(color=self.colors.on_surface_variant),
                title=dict(font=dict(color=self.colors.on_surface))
            ),
            legend=dict(
                bgcolor=self.colors.surface,
                bordercolor=self.colors.outline,
                borderwidth=1,
                font=dict(color=self.colors.on_surface)
            ),
            margin=dict(l=60, r=30, t=60, b=60)
        )
        
        # 테마 등록
        pio.templates["universal_doe"] = plotly_template
        pio.templates.default = "universal_doe"
    
    @staticmethod
    def apply() -> 'Theme':
        """현재 Streamlit 앱에 테마 적용"""
        # 세션 상태에서 테마 모드 확인
        mode_str = st.session_state.get('theme', 'light')
        mode = ThemeMode(mode_str)
        
        # 테마 인스턴스 생성
        theme = Theme(mode)
        
        # CSS 적용
        st.markdown(theme.get_css(), unsafe_allow_html=True)
        
        # Plotly 테마 설정
        theme.configure_plotly()
        
        return theme


# ===========================================================================
# 🎭 레벨별 적응형 UI 컴포넌트
# ===========================================================================

def get_user_level() -> UserLevel:
    """현재 사용자 레벨 가져오기"""
    level_str = st.session_state.get('user_level', 'beginner')
    return UserLevel(level_str)


def get_level_theme() -> Dict[str, Any]:
    """현재 레벨의 테마 설정 가져오기"""
    level = get_user_level()
    base_theme = LEVEL_THEMES[level].copy()
    
    # 다크모드 적용
    if st.session_state.get('dark_mode', False):
        colors = base_theme['colors'].copy()
        colors.update(DARK_MODE_OVERRIDES[level])
        base_theme['colors'] = colors
    
    return base_theme


def apply_level_based_css():
    """레벨 기반 CSS 적용"""
    theme = get_level_theme()
    level = get_user_level()
    
    css = f"""
    <style>
    /* 레벨별 커스텀 스타일 */
    .stApp {{
        --level-primary: {theme['colors']['primary']};
        --level-secondary: {theme['colors']['secondary']};
        --level-accent: {theme['colors']['accent']};
        --level-radius: {theme['ui']['border_radius']}px;
        --level-font-size: {theme['ui']['font_size_base']}px;
    }}
    
    /* 레벨별 버튼 크기 */
    .stButton > button {{
        font-size: var(--level-font-size);
        border-radius: var(--level-radius);
        {'padding: 12px 24px;' if theme['ui']['button_size'] == 'large' else ''}
        {'padding: 8px 16px;' if theme['ui']['button_size'] == 'medium' else ''}
        {'padding: 6px 12px;' if theme['ui']['button_size'] == 'small' else ''}
    }}
    
    /* 레벨별 애니메이션 */
    {'* { transition: none !important; }' if theme['ui']['animation_speed'] == 'none' else ''}
    {'* { transition-duration: 0.1s !important; }' if theme['ui']['animation_speed'] == 'instant' else ''}
    {'* { transition-duration: 0.2s !important; }' if theme['ui']['animation_speed'] == 'fast' else ''}
    {'* { transition-duration: 0.3s !important; }' if theme['ui']['animation_speed'] == 'normal' else ''}
    
    /* 초급자용 하이라이트 */
    {'''
    .beginner-highlight {
        animation: pulse 2s infinite;
        box-shadow: 0 0 0 2px var(--level-accent);
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 152, 0, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(255, 152, 0, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 152, 0, 0); }
    }
    ''' if level == UserLevel.BEGINNER else ''}
    
    /* 툴팁 표시 제어 */
    {'.tooltip { display: none !important; }' if not theme['ui']['show_tooltips'] else ''}
    
    /* 가이드 화살표 (초급자용) */
    {'''
    .guide-arrow {
        position: absolute;
        width: 50px;
        height: 50px;
        background: var(--level-accent);
        clip-path: polygon(0% 0%, 100% 50%, 0% 100%, 25% 50%);
        animation: bounce 1s infinite;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateX(0); }
        50% { transform: translateX(10px); }
    }
    ''' if level == UserLevel.BEGINNER else ''}
    </style>
    """
    
    st.markdown(css, unsafe_allow_html=True)


# ===========================================================================
# 🎨 교육적 UI 요소
# ===========================================================================

def show_educational_tooltip(
    text: str,
    help_text: str,
    key: Optional[str] = None
):
    """레벨별 교육적 툴팁 표시"""
    level = get_user_level()
    theme = get_level_theme()
    
    if not theme['ui']['show_tooltips']:
        st.write(text)
        return
    
    if theme['ui']['show_tooltips'] == True:
        # 항상 표시
        st.markdown(f"""
        <div class="tooltip">
            {text}
            <span class="tooltiptext">{help_text}</span>
        </div>
        """, unsafe_allow_html=True)
    elif theme['ui']['show_tooltips'] == "hover":
        # 호버시만 표시
        st.markdown(f"""
        <div class="tooltip">
            {text} ℹ️
            <span class="tooltiptext">{help_text}</span>
        </div>
        """, unsafe_allow_html=True)


def highlight_for_beginners(element_key: str):
    """초급자용 요소 하이라이트"""
    level = get_user_level()
    
    if level == UserLevel.BEGINNER:
        st.markdown(f"""
        <script>
        document.addEventListener('DOMContentLoaded', function() {{
            const element = document.querySelector('[data-testid="{element_key}"]');
            if (element) {{
                element.classList.add('beginner-highlight');
                setTimeout(() => element.classList.remove('beginner-highlight'), 5000);
            }}
        }});
        </script>
        """, unsafe_allow_html=True)


def show_guide_arrow(target_key: str, direction: str = "right"):
    """초급자용 가이드 화살표 표시"""
    level = get_user_level()
    theme = get_level_theme()
    
    if level == UserLevel.BEGINNER and theme['ui']['show_guides']:
        positions = {
            "right": "left: -60px; top: 50%; transform: translateY(-50%);",
            "left": "right: -60px; top: 50%; transform: translateY(-50%) rotate(180deg);",
            "up": "bottom: -60px; left: 50%; transform: translateX(-50%) rotate(90deg);",
            "down": "top: -60px; left: 50%; transform: translateX(-50%) rotate(-90deg);"
        }
        
        st.markdown(f"""
        <div style="position: relative;">
            <div class="guide-arrow" style="{positions.get(direction, positions['right'])}"></div>
        </div>
        """, unsafe_allow_html=True)


# ===========================================================================
# 🎯 편의 함수
# ===========================================================================

def get_theme() -> Theme:
    """현재 테마 인스턴스 반환"""
    if 'theme_instance' not in st.session_state:
        st.session_state.theme_instance = Theme.apply()
    return st.session_state.theme_instance


def apply_theme():
    """테마 적용 (간단한 호출용)"""
    Theme.apply()
    apply_level_based_css()


def get_colors() -> ColorPalette:
    """현재 색상 팔레트 반환"""
    return get_theme().colors


def get_chart_colors() -> List[str]:
    """차트용 색상 리스트 반환"""
    return get_colors().chart_colors


def create_custom_component_style(
    component_type: str,
    **kwargs
) -> str:
    """커스텀 컴포넌트 스타일 생성"""
    theme = get_theme()
    colors = theme.colors
    spacing = theme.spacing
    
    styles = {
        'info_box': f"""
            background-color: {colors.info}10;
            border-left: 4px solid {colors.info};
            padding: {spacing.md}px;
            border-radius: {theme.border_radius.md}px;
            margin: {spacing.md}px 0;
        """,
        'success_box': f"""
            background-color: {colors.success}10;
            border-left: 4px solid {colors.success};
            padding: {spacing.md}px;
            border-radius: {theme.border_radius.md}px;
            margin: {spacing.md}px 0;
        """,
        'warning_box': f"""
            background-color: {colors.warning}10;
            border-left: 4px solid {colors.warning};
            padding: {spacing.md}px;
            border-radius: {theme.border_radius.md}px;
            margin: {spacing.md}px 0;
        """,
        'error_box': f"""
            background-color: {colors.error}10;
            border-left: 4px solid {colors.error};
            padding: {spacing.md}px;
            border-radius: {theme.border_radius.md}px;
            margin: {spacing.md}px 0;
        """,
        'gradient_header': f"""
            background: linear-gradient(135deg, {colors.primary} 0%, {colors.primary_dark} 100%);
            color: {colors.on_primary};
            padding: {spacing.xl}px;
            border-radius: {theme.border_radius.lg}px;
            text-align: center;
            box-shadow: {theme.elevation.level2};
        """
    }
    
    base_style = styles.get(component_type, '')
    
    # 추가 스타일 병합
    for key, value in kwargs.items():
        base_style += f"{key}: {value};"
    
    return base_style


# ===========================================================================
# 🎨 테마 프리셋
# ===========================================================================

THEME_PRESETS = {
    'ocean': ColorPalette(
        primary="#006BA4",
        primary_light="#5B9BD5",
        primary_dark="#003D5B",
        secondary="#00A86B",
        chart_colors=["#006BA4", "#00A86B", "#FF6B6B", "#4ECDC4", "#FFE66D"]
    ),
    'forest': ColorPalette(
        primary="#228B22",
        primary_light="#90EE90",
        primary_dark="#006400",
        secondary="#8B4513",
        chart_colors=["#228B22", "#8B4513", "#FF8C00", "#DAA520", "#2E8B57"]
    ),
    'sunset': ColorPalette(
        primary="#FF6B6B",
        primary_light="#FF9999",
        primary_dark="#CC0000",
        secondary="#FFE66D",
        chart_colors=["#FF6B6B", "#FFE66D", "#FF8E53", "#FE6B8B", "#FF8E53"]
    ),
    'monochrome': ColorPalette(
        primary="#424242",
        primary_light="#757575",
        primary_dark="#212121",
        secondary="#616161",
        chart_colors=["#000000", "#424242", "#616161", "#757575", "#9E9E9E"]
    )
}


# ===========================================================================
# 🔧 초기화 함수
# ===========================================================================

def initialize_theme():
    """테마 시스템 초기화"""
    # 기본 테마 설정
    if 'theme' not in st.session_state:
        st.session_state.theme = 'light'
    
    # 사용자 레벨 설정
    if 'user_level' not in st.session_state:
        st.session_state.user_level = 'beginner'
    
    # 다크모드 설정
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False
    
    # 테마 적용
    apply_theme()


# 모듈 로드시 자동 실행
if __name__ != "__main__":
    initialize_theme()
