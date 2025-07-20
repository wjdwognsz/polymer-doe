# config/theme_config.py

import os
from typing import Dict, List, Optional, Any, Tuple
import streamlit as st
from pathlib import Path

# ===========================
# 1. 기본 색상 팔레트
# ===========================

# 브랜드 색상
BRAND_COLORS = {
    'primary': '#7C3AED',      # 보라색 - 메인 브랜드 색상
    'secondary': '#F59E0B',    # 주황색 - 강조색
    'tertiary': '#10B981',     # 초록색 - 성공/긍정
    
    # 의미 색상
    'success': '#10B981',      # 초록색
    'warning': '#F59E0B',      # 주황색
    'error': '#EF4444',        # 빨간색
    'info': '#3B82F6',         # 파란색
    
    # 중성 색상
    'gray': {
        50: '#F9FAFB',
        100: '#F3F4F6',
        200: '#E5E7EB',
        300: '#D1D5DB',
        400: '#9CA3AF',
        500: '#6B7280',
        600: '#4B5563',
        700: '#374151',
        800: '#1F2937',
        900: '#111827'
    }
}

# ===========================
# 2. 레벨별 색상 테마
# ===========================

LEVEL_THEMES = {
    'beginner': {
        'name': '🌱 초급 테마',
        'description': '밝고 친근한 색상으로 학습 동기 부여',
        
        # 색상 설정
        'colors': {
            'primary': '#10B981',        # 부드러운 초록색
            'primary_hover': '#059669',
            'primary_light': '#D1FAE5',
            'secondary': '#3B82F6',      # 친근한 파란색
            'background': '#FFFFFF',
            'surface': '#F9FAFB',
            'text': '#1F2937',
            'text_secondary': '#6B7280',
            'border': '#E5E7EB',
            'shadow': 'rgba(0, 0, 0, 0.05)',
            
            # 교육적 하이라이트
            'highlight': '#FEF3C7',       # 노란색 하이라이트
            'guide': '#DBEAFE',          # 파란색 가이드
            'tip': '#D1FAE5',            # 초록색 팁
            'warning_bg': '#FEE2E2'      # 경고 배경
        },
        
        # UI 특성
        'ui': {
            'border_radius': '12px',      # 둥근 모서리
            'spacing': 'relaxed',         # 넓은 간격
            'font_size_base': '16px',    # 큰 글자
            'line_height': '1.6',        # 넓은 줄간격
            'contrast': 'high'           # 높은 대비
        }
    },
    
    'intermediate': {
        'name': '🌿 중급 테마',
        'description': '균형잡힌 전문적 색상',
        
        'colors': {
            'primary': '#3B82F6',        # 전문적 파란색
            'primary_hover': '#2563EB',
            'primary_light': '#DBEAFE',
            'secondary': '#8B5CF6',      # 보라색
            'background': '#FFFFFF',
            'surface': '#F9FAFB',
            'text': '#1F2937',
            'text_secondary': '#6B7280',
            'border': '#E5E7EB',
            'shadow': 'rgba(0, 0, 0, 0.08)',
            
            # 교육적 하이라이트 (축소)
            'highlight': '#FEF3C7',
            'guide': '#E0E7FF',
            'tip': '#D1FAE5',
            'warning_bg': '#FEE2E2'
        },
        
        'ui': {
            'border_radius': '8px',
            'spacing': 'normal',
            'font_size_base': '15px',
            'line_height': '1.5',
            'contrast': 'normal'
        }
    },
    
    'advanced': {
        'name': '🌳 고급 테마',
        'description': '세련되고 효율적인 색상',
        
        'colors': {
            'primary': '#7C3AED',        # 브랜드 보라색
            'primary_hover': '#6D28D9',
            'primary_light': '#EDE9FE',
            'secondary': '#EC4899',      # 핑크색
            'background': '#FFFFFF',
            'surface': '#FAFAFA',
            'text': '#111827',
            'text_secondary': '#6B7280',
            'border': '#E5E7EB',
            'shadow': 'rgba(0, 0, 0, 0.1)',
            
            # 최소 하이라이트
            'highlight': 'transparent',
            'guide': 'transparent',
            'tip': '#F3F4F6',
            'warning_bg': '#FEF2F2'
        },
        
        'ui': {
            'border_radius': '6px',
            'spacing': 'compact',
            'font_size_base': '14px',
            'line_height': '1.5',
            'contrast': 'normal'
        }
    },
    
    'expert': {
        'name': '🏆 전문가 테마',
        'description': '미니멀하고 집중적인 디자인',
        
        'colors': {
            'primary': '#1F2937',        # 다크 그레이
            'primary_hover': '#111827',
            'primary_light': '#F3F4F6',
            'secondary': '#7C3AED',      # 포인트 색상
            'background': '#FFFFFF',
            'surface': '#FAFAFA',
            'text': '#111827',
            'text_secondary': '#6B7280',
            'border': '#E5E7EB',
            'shadow': 'rgba(0, 0, 0, 0.05)',
            
            # 하이라이트 없음
            'highlight': 'transparent',
            'guide': 'transparent',
            'tip': 'transparent',
            'warning_bg': 'transparent'
        },
        
        'ui': {
            'border_radius': '4px',
            'spacing': 'dense',
            'font_size_base': '14px',
            'line_height': '1.4',
            'contrast': 'low'
        }
    }
}

# ===========================
# 3. 다크 모드 테마
# ===========================

DARK_THEMES = {
    'beginner': {
        'colors': {
            'primary': '#34D399',        # 밝은 초록
            'primary_hover': '#10B981',
            'primary_light': '#064E3B',
            'secondary': '#60A5FA',      # 밝은 파란색
            'background': '#111827',
            'surface': '#1F2937',
            'text': '#F9FAFB',
            'text_secondary': '#D1D5DB',
            'border': '#374151',
            'shadow': 'rgba(0, 0, 0, 0.3)',
            
            'highlight': '#422006',      # 어두운 노란색
            'guide': '#1E3A8A',         # 어두운 파란색
            'tip': '#064E3B',           # 어두운 초록색
            'warning_bg': '#7F1D1D'     # 어두운 빨간색
        }
    },
    
    'intermediate': {
        'colors': {
            'primary': '#60A5FA',
            'primary_hover': '#3B82F6',
            'primary_light': '#1E3A8A',
            'secondary': '#A78BFA',
            'background': '#0F172A',
            'surface': '#1E293B',
            'text': '#F1F5F9',
            'text_secondary': '#CBD5E1',
            'border': '#334155',
            'shadow': 'rgba(0, 0, 0, 0.4)'
        }
    },
    
    'advanced': {
        'colors': {
            'primary': '#9333EA',
            'primary_hover': '#7C3AED',
            'primary_light': '#4C1D95',
            'secondary': '#EC4899',
            'background': '#0F172A',
            'surface': '#1E293B',
            'text': '#F1F5F9',
            'text_secondary': '#CBD5E1',
            'border': '#334155',
            'shadow': 'rgba(0, 0, 0, 0.5)'
        }
    },
    
    'expert': {
        'colors': {
            'primary': '#F3F4F6',
            'primary_hover': '#E5E7EB',
            'primary_light': '#374151',
            'secondary': '#9333EA',
            'background': '#000000',
            'surface': '#111827',
            'text': '#F9FAFB',
            'text_secondary': '#9CA3AF',
            'border': '#1F2937',
            'shadow': 'rgba(0, 0, 0, 0.8)'
        }
    }
}

# ===========================
# 4. 애니메이션 설정
# ===========================

ANIMATIONS = {
    'beginner': {
        'enabled': True,
        'duration': {
            'fast': '200ms',
            'normal': '300ms',
            'slow': '500ms'
        },
        'easing': 'ease-out',
        'effects': {
            'hover': 'scale(1.05) translateY(-2px)',
            'click': 'scale(0.98)',
            'page_transition': 'fadeIn',
            'loading': 'pulse',
            'success': 'bounceIn',
            'error': 'shake'
        },
        'guide_animations': {
            'arrow': 'floating',
            'highlight': 'glow',
            'tooltip': 'fadeInUp'
        }
    },
    
    'intermediate': {
        'enabled': True,
        'duration': {
            'fast': '150ms',
            'normal': '200ms',
            'slow': '300ms'
        },
        'easing': 'ease-in-out',
        'effects': {
            'hover': 'translateY(-1px)',
            'click': 'scale(0.99)',
            'page_transition': 'slideIn',
            'loading': 'spin'
        },
        'guide_animations': {
            'arrow': 'none',
            'highlight': 'subtle',
            'tooltip': 'fade'
        }
    },
    
    'advanced': {
        'enabled': True,
        'duration': {
            'fast': '100ms',
            'normal': '150ms',
            'slow': '200ms'
        },
        'easing': 'ease',
        'effects': {
            'hover': 'opacity(0.8)',
            'click': 'none',
            'page_transition': 'none',
            'loading': 'minimal'
        }
    },
    
    'expert': {
        'enabled': False,  # 애니메이션 비활성화
        'duration': {
            'fast': '0ms',
            'normal': '0ms',
            'slow': '0ms'
        }
    }
}

# ===========================
# 5. 타이포그래피 설정
# ===========================

TYPOGRAPHY = {
    'fonts': {
        'primary': "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        'secondary': "'Roboto', sans-serif",
        'mono': "'Fira Code', 'Consolas', monospace"
    },
    
    'beginner': {
        'base_size': 16,
        'scale_ratio': 1.25,  # 큰 크기 차이
        'weights': {
            'normal': 400,
            'medium': 500,
            'bold': 700
        },
        'sizes': {
            'xs': '0.75rem',   # 12px
            'sm': '0.875rem',  # 14px
            'base': '1rem',    # 16px
            'lg': '1.125rem',  # 18px
            'xl': '1.25rem',   # 20px
            'h6': '1.25rem',   # 20px
            'h5': '1.5rem',    # 24px
            'h4': '1.875rem',  # 30px
            'h3': '2.25rem',   # 36px
            'h2': '2.75rem',   # 44px
            'h1': '3.5rem'     # 56px
        }
    },
    
    'intermediate': {
        'base_size': 15,
        'scale_ratio': 1.2,
        'weights': {
            'normal': 400,
            'medium': 500,
            'bold': 600
        }
    },
    
    'advanced': {
        'base_size': 14,
        'scale_ratio': 1.15,
        'weights': {
            'normal': 400,
            'medium': 500,
            'bold': 600
        }
    },
    
    'expert': {
        'base_size': 14,
        'scale_ratio': 1.125,
        'weights': {
            'normal': 400,
            'medium': 500,
            'bold': 600
        }
    }
}

# ===========================
# 6. 레이아웃 설정
# ===========================

LAYOUT = {
    'beginner': {
        'container_width': '1200px',
        'sidebar_width': '320px',
        'spacing': {
            'xs': '0.25rem',   # 4px
            'sm': '0.5rem',    # 8px
            'md': '1rem',      # 16px
            'lg': '1.5rem',    # 24px
            'xl': '2rem',      # 32px
            'xxl': '3rem'      # 48px
        },
        'grid_gap': '1.5rem',
        'card_padding': '1.5rem',
        'section_margin': '3rem'
    },
    
    'intermediate': {
        'container_width': '1280px',
        'sidebar_width': '280px',
        'spacing': {
            'xs': '0.25rem',
            'sm': '0.5rem',
            'md': '0.75rem',
            'lg': '1rem',
            'xl': '1.5rem',
            'xxl': '2rem'
        },
        'grid_gap': '1rem',
        'card_padding': '1rem',
        'section_margin': '2rem'
    },
    
    'advanced': {
        'container_width': '1400px',
        'sidebar_width': '260px',
        'spacing': {
            'xs': '0.125rem',
            'sm': '0.25rem',
            'md': '0.5rem',
            'lg': '0.75rem',
            'xl': '1rem',
            'xxl': '1.5rem'
        },
        'grid_gap': '0.75rem',
        'card_padding': '0.75rem',
        'section_margin': '1.5rem'
    },
    
    'expert': {
        'container_width': '100%',
        'sidebar_width': '240px',
        'spacing': {
            'xs': '0.125rem',
            'sm': '0.25rem',
            'md': '0.375rem',
            'lg': '0.5rem',
            'xl': '0.75rem',
            'xxl': '1rem'
        },
        'grid_gap': '0.5rem',
        'card_padding': '0.5rem',
        'section_margin': '1rem'
    }
}

# ===========================
# 7. 컴포넌트 스타일
# ===========================

COMPONENTS = {
    'beginner': {
        'button': {
            'height': '48px',
            'padding': '12px 24px',
            'font_size': '16px',
            'border_radius': '12px',
            'shadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
            'hover_shadow': '0 6px 12px rgba(0, 0, 0, 0.15)'
        },
        'input': {
            'height': '48px',
            'padding': '12px 16px',
            'font_size': '16px',
            'border_radius': '8px',
            'border_width': '2px',
            'focus_ring_width': '3px'
        },
        'card': {
            'border_radius': '16px',
            'shadow': '0 2px 8px rgba(0, 0, 0, 0.08)',
            'hover_shadow': '0 4px 16px rgba(0, 0, 0, 0.12)',
            'border_width': '1px'
        },
        'modal': {
            'border_radius': '16px',
            'shadow': '0 20px 25px rgba(0, 0, 0, 0.15)',
            'overlay_opacity': '0.5'
        }
    },
    
    'intermediate': {
        'button': {
            'height': '40px',
            'padding': '10px 20px',
            'font_size': '15px',
            'border_radius': '8px',
            'shadow': '0 2px 4px rgba(0, 0, 0, 0.08)',
            'hover_shadow': '0 4px 8px rgba(0, 0, 0, 0.12)'
        }
    },
    
    'advanced': {
        'button': {
            'height': '36px',
            'padding': '8px 16px',
            'font_size': '14px',
            'border_radius': '6px',
            'shadow': '0 1px 3px rgba(0, 0, 0, 0.08)',
            'hover_shadow': '0 2px 6px rgba(0, 0, 0, 0.12)'
        }
    },
    
    'expert': {
        'button': {
            'height': '32px',
            'padding': '6px 12px',
            'font_size': '14px',
            'border_radius': '4px',
            'shadow': 'none',
            'hover_shadow': 'none'
        }
    }
}

# ===========================
# 8. 교육적 UI 요소
# ===========================

EDUCATIONAL_UI = {
    'beginner': {
        'tooltips': {
            'enabled': True,
            'delay': 0,
            'position': 'top',
            'max_width': '300px',
            'show_icon': True,
            'style': {
                'background': '#1F2937',
                'color': '#FFFFFF',
                'border_radius': '8px',
                'padding': '8px 12px',
                'font_size': '14px'
            }
        },
        'highlights': {
            'enabled': True,
            'style': 'glow',
            'color': '#FEF3C7',
            'animation': 'pulse'
        },
        'guides': {
            'arrows': True,
            'overlays': True,
            'step_numbers': True,
            'progress_bar': True
        },
        'feedback': {
            'success_animation': 'confetti',
            'error_shake': True,
            'sound_effects': False
        }
    },
    
    'intermediate': {
        'tooltips': {
            'enabled': True,
            'delay': 500,
            'position': 'auto',
            'max_width': '250px',
            'show_icon': False
        },
        'highlights': {
            'enabled': False
        },
        'guides': {
            'arrows': False,
            'overlays': False,
            'step_numbers': True,
            'progress_bar': False
        }
    },
    
    'advanced': {
        'tooltips': {
            'enabled': True,
            'delay': 1000,
            'on_hover_only': True
        },
        'highlights': {
            'enabled': False
        },
        'guides': {
            'all_disabled': True
        }
    },
    
    'expert': {
        'all_disabled': True
    }
}

# ===========================
# 9. 접근성 설정
# ===========================

ACCESSIBILITY = {
    'high_contrast': {
        'enabled': False,  # 사용자 설정
        'colors': {
            'primary': '#0000FF',
            'secondary': '#FF00FF',
            'background': '#FFFFFF',
            'text': '#000000',
            'border': '#000000'
        }
    },
    
    'color_blind_modes': {
        'protanopia': {  # 적색맹
            'primary': '#0173B2',
            'secondary': '#DE8F05',
            'success': '#029E73',
            'error': '#CC78BC'
        },
        'deuteranopia': {  # 녹색맹
            'primary': '#0173B2',
            'secondary': '#DE8F05',
            'success': '#029E73',
            'error': '#CC78BC'
        },
        'tritanopia': {  # 청색맹
            'primary': '#E51C23',
            'secondary': '#F57C00',
            'success': '#00BFA5',
            'error': '#9C27B0'
        }
    },
    
    'font_size_multiplier': {
        'small': 0.85,
        'normal': 1.0,
        'large': 1.15,
        'extra_large': 1.3
    },
    
    'focus_indicators': {
        'style': 'ring',
        'color': '#2563EB',
        'width': '3px',
        'offset': '2px'
    }
}

# ===========================
# 10. 반응형 브레이크포인트
# ===========================

BREAKPOINTS = {
    'xs': '0px',      # 모바일
    'sm': '640px',    # 태블릿
    'md': '768px',    # 작은 데스크톱
    'lg': '1024px',   # 데스크톱
    'xl': '1280px',   # 큰 데스크톱
    '2xl': '1536px'   # 초대형 화면
}

RESPONSIVE = {
    'beginner': {
        'mobile_first': True,
        'touch_targets': {
            'min_size': '44px',  # Apple 가이드라인
            'spacing': '8px'
        },
        'font_scale': {
            'mobile': 0.9,
            'tablet': 0.95,
            'desktop': 1.0
        }
    },
    
    'expert': {
        'mobile_first': False,
        'touch_targets': {
            'min_size': '32px',
            'spacing': '4px'
        },
        'font_scale': {
            'mobile': 1.0,
            'tablet': 1.0,
            'desktop': 1.0
        }
    }
}

# ===========================
# 11. CSS 생성 함수
# ===========================

def generate_theme_css(user_level: str = 'beginner', dark_mode: bool = False) -> str:
    """레벨별 테마 CSS 생성"""
    
    # 테마 선택
    theme = DARK_THEMES[user_level] if dark_mode else LEVEL_THEMES[user_level]
    colors = theme['colors']
    ui = LEVEL_THEMES[user_level]['ui']
    animations = ANIMATIONS[user_level]
    typography = TYPOGRAPHY[user_level]
    layout = LAYOUT[user_level]
    components = COMPONENTS[user_level]
    educational = EDUCATIONAL_UI[user_level]
    
    css = f"""
    <style>
    /* ===== CSS 변수 정의 ===== */
    :root {{
        /* 색상 */
        --primary: {colors['primary']};
        --primary-hover: {colors['primary_hover']};
        --primary-light: {colors['primary_light']};
        --secondary: {colors['secondary']};
        --background: {colors['background']};
        --surface: {colors['surface']};
        --text: {colors['text']};
        --text-secondary: {colors['text_secondary']};
        --border: {colors['border']};
        --shadow: {colors['shadow']};
        
        /* 교육적 색상 */
        --highlight: {colors.get('highlight', 'transparent')};
        --guide: {colors.get('guide', 'transparent')};
        --tip: {colors.get('tip', 'transparent')};
        --warning-bg: {colors.get('warning_bg', 'transparent')};
        
        /* UI 설정 */
        --border-radius: {ui['border_radius']};
        --font-size-base: {ui['font_size_base']};
        --line-height: {ui['line_height']};
        
        /* 애니메이션 */
        --anim-fast: {animations['duration']['fast'] if animations['enabled'] else '0ms'};
        --anim-normal: {animations['duration']['normal'] if animations['enabled'] else '0ms'};
        --anim-slow: {animations['duration']['slow'] if animations['enabled'] else '0ms'};
        --anim-easing: {animations.get('easing', 'ease')};
        
        /* 타이포그래피 */
        --font-primary: {TYPOGRAPHY['fonts']['primary']};
        --font-mono: {TYPOGRAPHY['fonts']['mono']};
        
        /* 레이아웃 */
        --container-width: {layout['container_width']};
        --sidebar-width: {layout['sidebar_width']};
        --spacing-xs: {layout['spacing']['xs']};
        --spacing-sm: {layout['spacing']['sm']};
        --spacing-md: {layout['spacing']['md']};
        --spacing-lg: {layout['spacing']['lg']};
        --spacing-xl: {layout['spacing']['xl']};
    }}
    
    /* ===== 기본 스타일 ===== */
    * {{
        transition: all var(--anim-fast) var(--anim-easing);
    }}
    
    body {{
        font-family: var(--font-primary);
        font-size: var(--font-size-base);
        line-height: var(--line-height);
        color: var(--text);
        background-color: var(--background);
    }}
    
    /* ===== 메인 컨테이너 ===== */
    .main {{
        max-width: var(--container-width);
        margin: 0 auto;
        padding: var(--spacing-lg);
    }}
    
    .sidebar {{
        width: var(--sidebar-width);
        background-color: var(--surface);
        border-right: 1px solid var(--border);
    }}
    
    /* ===== 버튼 스타일 ===== */
    .stButton > button {{
        height: {components['button']['height']};
        padding: {components['button']['padding']};
        font-size: {components['button']['font_size']};
        font-weight: 500;
        border-radius: {components['button']['border_radius']};
        border: none;
        background-color: var(--primary);
        color: white;
        box-shadow: {components['button']['shadow']};
        cursor: pointer;
        transition: all var(--anim-fast) var(--anim-easing);
    }}
    
    .stButton > button:hover {{
        background-color: var(--primary-hover);
        box-shadow: {components['button']['hover_shadow']};
        {f"transform: {animations['effects']['hover']};" if animations['enabled'] else ""}
    }}
    
    .stButton > button:active {{
        {f"transform: {animations['effects']['click']};" if animations['enabled'] else ""}
    }}
    
    /* ===== 입력 필드 ===== */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {{
        height: {components['input']['height']};
        padding: {components['input']['padding']};
        font-size: {components['input']['font_size']};
        border-radius: {components['input']['border_radius']};
        border: {components['input']['border_width']} solid var(--border);
        background-color: var(--surface);
        color: var(--text);
        transition: all var(--anim-fast) var(--anim-easing);
    }}
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stSelectbox > div > div > select:focus {{
        outline: none;
        border-color: var(--primary);
        box-shadow: 0 0 0 {components['input']['focus_ring_width']} rgba(124, 58, 237, 0.1);
    }}
    
    /* ===== 카드 스타일 ===== */
    .card {{
        background-color: var(--surface);
        border-radius: {components['card']['border_radius']};
        border: {components['card']['border_width']} solid var(--border);
        box-shadow: {components['card']['shadow']};
        padding: var(--spacing-lg);
        margin-bottom: var(--spacing-md);
        transition: all var(--anim-normal) var(--anim-easing);
    }}
    
    .card:hover {{
        box-shadow: {components['card']['hover_shadow']};
        {f"transform: translateY(-2px);" if user_level in ['beginner', 'intermediate'] else ""}
    }}
    
    /* ===== 교육적 요소 (초보자) ===== */
    {f'''
    .highlight {{
        background-color: var(--highlight) !important;
        padding: 2px 4px;
        border-radius: 4px;
        {f"animation: {educational['highlights']['animation']} 2s infinite;" if user_level == 'beginner' else ""}
    }}
    
    .guide-arrow {{
        position: absolute;
        color: var(--primary);
        font-size: 24px;
        animation: floating 2s ease-in-out infinite;
    }}
    
    @keyframes floating {{
        0%, 100% {{ transform: translateY(0); }}
        50% {{ transform: translateY(-10px); }}
    }}
    
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.6; }}
    }}
    
    .tooltip {{
        background: {educational['tooltips']['style']['background']};
        color: {educational['tooltips']['style']['color']};
        padding: {educational['tooltips']['style']['padding']};
        border-radius: {educational['tooltips']['style']['border_radius']};
        font-size: {educational['tooltips']['style']['font_size']};
        max-width: {educational['tooltips']['max_width']};
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }}
    ''' if user_level == 'beginner' else ''}
    
    /* ===== 헤딩 스타일 ===== */
    h1 {{ font-size: {typography['sizes']['h1']}; font-weight: {typography['weights']['bold']}; }}
    h2 {{ font-size: {typography['sizes']['h2']}; font-weight: {typography['weights']['bold']}; }}
    h3 {{ font-size: {typography['sizes']['h3']}; font-weight: {typography['weights']['medium']}; }}
    h4 {{ font-size: {typography['sizes']['h4']}; font-weight: {typography['weights']['medium']}; }}
    h5 {{ font-size: {typography['sizes']['h5']}; font-weight: {typography['weights']['medium']}; }}
    h6 {{ font-size: {typography['sizes']['h6']}; font-weight: {typography['weights']['medium']}; }}
    
    /* ===== 메트릭 카드 ===== */
    div[data-testid="metric-container"] {{
        background-color: var(--surface);
        border: 1px solid var(--border);
        padding: var(--spacing-lg);
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
        transition: all var(--anim-normal) var(--anim-easing);
    }}
    
    div[data-testid="metric-container"]:hover {{
        {f"transform: translateY(-2px);" if user_level == 'beginner' else ""}
        box-shadow: {components['card']['hover_shadow']};
    }}
    
    /* ===== 사이드바 스타일 ===== */
    .css-1d391kg {{
        background-color: var(--surface);
        padding: var(--spacing-lg);
    }}
    
    /* ===== 스크롤바 ===== */
    ::-webkit-scrollbar {{
        width: {f"12px" if user_level == 'beginner' else "8px"};
        height: {f"12px" if user_level == 'beginner' else "8px"};
    }}
    
    ::-webkit-scrollbar-track {{
        background: var(--surface);
        border-radius: var(--border-radius);
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: var(--border);
        border-radius: var(--border-radius);
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: var(--text-secondary);
    }}
    
    /* ===== 반응형 디자인 ===== */
    @media (max-width: 768px) {{
        .main {{
            padding: var(--spacing-md);
        }}
        
        .stButton > button {{
            font-size: {f"{int(components['button']['font_size'][:-2]) * 0.9}px"};
            height: {f"{int(components['button']['height'][:-2]) * 0.9}px"};
        }}
        
        h1 {{ font-size: {f"{float(typography['sizes']['h1'][:-3]) * 0.8}rem"}; }}
        h2 {{ font-size: {f"{float(typography['sizes']['h2'][:-3]) * 0.8}rem"}; }}
    }}
    
    /* ===== 접근성 포커스 ===== */
    *:focus-visible {{
        outline: {ACCESSIBILITY['focus_indicators']['width']} solid {ACCESSIBILITY['focus_indicators']['color']};
        outline-offset: {ACCESSIBILITY['focus_indicators']['offset']};
    }}
    
    /* ===== 페이지 전환 애니메이션 ===== */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    @keyframes slideIn {{
        from {{ transform: translateX(-100%); }}
        to {{ transform: translateX(0); }}
    }}
    
    .page-enter {{
        animation: {animations['effects'].get('page_transition', 'fadeIn')} var(--anim-normal) var(--anim-easing);
    }}
    
    /* ===== 다크모드 조정 ===== */
    {f'''
    @media (prefers-color-scheme: dark) {{
        img {{ opacity: 0.9; }}
        .highlight {{ opacity: 0.8; }}
    }}
    ''' if dark_mode else ''}
    
    /* ===== 프린트 스타일 ===== */
    @media print {{
        .sidebar, .stButton, .stTextInput {{
            display: none !important;
        }}
        
        body {{
            font-size: 12pt;
            color: black;
            background: white;
        }}
    }}
    </style>
    """
    
    return css

# ===========================
# 12. 테마 적용 함수
# ===========================

def apply_theme(user_level: str = None, dark_mode: bool = None):
    """사용자 레벨에 맞는 테마 적용"""
    
    # 사용자 레벨 확인
    if user_level is None:
        user_level = st.session_state.get('user', {}).get('level', 'beginner')
    
    # 다크모드 설정 확인
    if dark_mode is None:
        dark_mode = st.session_state.get('dark_mode', False)
    
    # CSS 생성 및 적용
    theme_css = generate_theme_css(user_level, dark_mode)
    st.markdown(theme_css, unsafe_allow_html=True)
    
    # Streamlit 네이티브 테마 설정
    theme_colors = DARK_THEMES[user_level]['colors'] if dark_mode else LEVEL_THEMES[user_level]['colors']
    
    # 세션에 테마 정보 저장
    st.session_state['current_theme'] = {
        'level': user_level,
        'dark_mode': dark_mode,
        'colors': theme_colors
    }

def get_color(color_name: str) -> str:
    """현재 테마의 색상 가져오기"""
    current_theme = st.session_state.get('current_theme', {})
    colors = current_theme.get('colors', LEVEL_THEMES['beginner']['colors'])
    return colors.get(color_name, '#000000')

def get_spacing(size: str = 'md') -> str:
    """현재 레벨의 간격 가져오기"""
    user_level = st.session_state.get('user', {}).get('level', 'beginner')
    return LAYOUT[user_level]['spacing'].get(size, '1rem')

def should_animate(animation_type: str = 'hover') -> bool:
    """애니메이션 활성화 여부 확인"""
    user_level = st.session_state.get('user', {}).get('level', 'beginner')
    return ANIMATIONS[user_level]['enabled']

# ===========================
# 13. 교육적 UI 헬퍼 함수
# ===========================

def show_educational_tooltip(text: str, key: str = None):
    """레벨별 툴팁 표시"""
    user_level = st.session_state.get('user', {}).get('level', 'beginner')
    
    if user_level == 'expert':
        return  # 전문가는 툴팁 없음
    
    tooltip_config = EDUCATIONAL_UI[user_level]['tooltips']
    
    if tooltip_config['enabled']:
        st.markdown(
            f"""
            <div class="tooltip-container">
                {f'<span class="tooltip-icon">ℹ️</span>' if tooltip_config.get('show_icon') else ''}
                <div class="tooltip" style="display: none;">
                    {text}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

def highlight_element(element_id: str, duration: int = 3000):
    """초보자용 요소 하이라이트"""
    user_level = st.session_state.get('user', {}).get('level', 'beginner')
    
    if user_level == 'beginner':
        st.markdown(
            f"""
            <script>
            setTimeout(() => {{
                const element = document.getElementById('{element_id}');
                if (element) {{
                    element.classList.add('highlight');
                    setTimeout(() => element.classList.remove('highlight'), {duration});
                }}
            }}, 100);
            </script>
            """,
            unsafe_allow_html=True
        )

# ===========================
# 14. 설정 검증
# ===========================

def validate_theme_config() -> Tuple[bool, List[str]]:
    """테마 설정 검증"""
    warnings = []
    
    # 필수 레벨 확인
    required_levels = ['beginner', 'intermediate', 'advanced', 'expert']
    for level in required_levels:
        if level not in LEVEL_THEMES:
            warnings.append(f"레벨 '{level}'의 테마가 정의되지 않았습니다.")
        if level not in DARK_THEMES:
            warnings.append(f"레벨 '{level}'의 다크 테마가 정의되지 않았습니다.")
    
    # 색상 값 검증
    for level, theme in LEVEL_THEMES.items():
        for color_name, color_value in theme['colors'].items():
            if not color_value.startswith('#') and color_value != 'transparent':
                warnings.append(f"{level} 테마의 {color_name} 색상값이 올바르지 않습니다: {color_value}")
    
    return len(warnings) == 0, warnings

# 설정 검증 실행
if __name__ != "__main__":
    success, warnings = validate_theme_config()
    if warnings:
        for warning in warnings:
            print(f"경고: {warning}")
