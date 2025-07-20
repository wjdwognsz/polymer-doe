# utils/common_ui.py
"""
Universal DOE Platform - 공통 UI 컴포넌트
모든 페이지에서 재사용되는 UI 요소들을 제공합니다.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any, Callable
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import time

# 테마 설정 임포트 (theme_config.py가 생성된 후 활성화)
try:
    from config.theme_config import THEME_CONFIG
except ImportError:
    # 기본 테마 설정
    THEME_CONFIG = {
        'primary_color': '#FF6B6B',
        'secondary_color': '#4ECDC4',
        'background_color': '#FFFFFF',
        'text_color': '#2D3436'
    }

# =============================================================================
# 기본 레이아웃 컴포넌트
# =============================================================================

def render_header(title: str = "🧬 Universal DOE Platform", 
                  subtitle: Optional[str] = None,
                  show_user_info: bool = True):
    """애플리케이션 헤더"""
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        st.markdown(f"<h1 style='text-align: center; color: {THEME_CONFIG['primary_color']};'>{title}</h1>", 
                   unsafe_allow_html=True)
        if subtitle:
            st.markdown(f"<p style='text-align: center; color: gray;'>{subtitle}</p>", 
                       unsafe_allow_html=True)
    
    with col3:
        if show_user_info and st.session_state.get('user'):
            user = st.session_state.user
            st.markdown(f"""
                <div style='text-align: right; padding: 10px;'>
                    <small>👤 {user.get('name', 'User')}</small><br>
                    <small>📧 {user.get('email', '')}</small>
                </div>
            """, unsafe_allow_html=True)

def render_navigation():
    """메인 네비게이션 바"""
    pages = {
        "🏠 홈": "home",
        "📊 대시보드": "dashboard",
        "🔬 실험 설계": "experiment_design",
        "📈 데이터 분석": "data_analysis",
        "👥 협업": "collaboration",
        "📚 문헌 검색": "literature_search",
        "🧩 모듈 마켓": "module_marketplace"
    }
    
    cols = st.columns(len(pages))
    for idx, (label, page_key) in enumerate(pages.items()):
        with cols[idx]:
            if st.button(label, key=f"nav_{page_key}", use_container_width=True):
                st.session_state.current_page = page_key
                st.rerun()

def render_sidebar_menu():
    """사이드바 메뉴"""
    with st.sidebar:
        st.markdown("### 🧬 Universal DOE")
        
        # 현재 프로젝트 정보
        if st.session_state.get('current_project'):
            project = st.session_state.current_project
            st.info(f"📁 {project.get('name', '프로젝트 미선택')}")
        
        st.markdown("---")
        
        # 빠른 메뉴
        st.markdown("### ⚡ 빠른 메뉴")
        if st.button("➕ 새 프로젝트", use_container_width=True):
            st.session_state.show_new_project = True
        if st.button("📂 프로젝트 열기", use_container_width=True):
            st.session_state.show_project_list = True
        if st.button("💾 저장", use_container_width=True):
            save_current_work()
        
        st.markdown("---")
        
        # 도움말
        with st.expander("❓ 도움말"):
            st.markdown("""
            - **새 프로젝트**: 새로운 실험 설계 시작
            - **모듈 선택**: 연구 분야별 실험 모듈
            - **AI 지원**: 6개 AI 엔진 활용
            - **협업**: 팀원과 실시간 공유
            """)

def render_footer():
    """애플리케이션 푸터"""
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col2:
        st.markdown(
            "<p style='text-align: center; color: gray; font-size: 12px;'>"
            "© 2024 Universal DOE Platform | 모든 연구자를 위한 실험 설계 플랫폼"
            "</p>", 
            unsafe_allow_html=True
        )

# =============================================================================
# 카드/컨테이너 컴포넌트
# =============================================================================

def info_card(title: str, content: str, icon: str = "ℹ️", 
              color: Optional[str] = None):
    """정보 카드"""
    color = color or THEME_CONFIG['primary_color']
    st.markdown(f"""
        <div style='
            background-color: {color}20;
            border-left: 4px solid {color};
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        '>
            <h4 style='margin: 0; color: {color};'>{icon} {title}</h4>
            <p style='margin: 5px 0 0 0;'>{content}</p>
        </div>
    """, unsafe_allow_html=True)

def metric_card(label: str, value: Any, delta: Optional[str] = None,
                delta_color: str = "normal"):
    """메트릭 카드"""
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)

def expandable_section(title: str, content_func: Callable, 
                      expanded: bool = False, icon: str = "📋"):
    """확장 가능한 섹션"""
    with st.expander(f"{icon} {title}", expanded=expanded):
        content_func()

def tab_container(tabs: Dict[str, Callable], key_prefix: str = "tab"):
    """탭 컨테이너"""
    tab_names = list(tabs.keys())
    tab_contents = list(tabs.values())
    
    selected_tabs = st.tabs(tab_names)
    for idx, (tab, content_func) in enumerate(zip(selected_tabs, tab_contents)):
        with tab:
            content_func()

# =============================================================================
# 입력 컴포넌트
# =============================================================================

def range_input(label: str, min_val: float, max_val: float, 
                default: Optional[Tuple[float, float]] = None,
                step: Optional[float] = None, key: Optional[str] = None) -> Tuple[float, float]:
    """범위 입력 (최소-최대)"""
    col1, col2 = st.columns(2)
    
    default_min = default[0] if default else min_val
    default_max = default[1] if default else max_val
    
    with col1:
        min_value = st.number_input(
            f"{label} (최소)", 
            min_value=min_val, 
            max_value=max_val,
            value=default_min,
            step=step,
            key=f"{key}_min" if key else None
        )
    
    with col2:
        max_value = st.number_input(
            f"{label} (최대)", 
            min_value=min_val, 
            max_value=max_val,
            value=default_max,
            step=step,
            key=f"{key}_max" if key else None
        )
    
    if min_value > max_value:
        st.error("최소값이 최대값보다 클 수 없습니다.")
        return default_min, default_max
    
    return min_value, max_value

def multi_select_with_all(label: str, options: List[str], 
                         default: Optional[List[str]] = None,
                         key: Optional[str] = None) -> List[str]:
    """전체 선택 옵션이 있는 다중 선택"""
    all_option = "🔹 전체 선택"
    options_with_all = [all_option] + options
    
    selected = st.multiselect(
        label, 
        options=options_with_all,
        default=default,
        key=key
    )
    
    if all_option in selected:
        return options
    return [opt for opt in selected if opt != all_option]

def tag_input(label: str, placeholder: str = "태그 입력 후 Enter",
              key: Optional[str] = None) -> List[str]:
    """태그 입력 컴포넌트"""
    if f"{key}_tags" not in st.session_state:
        st.session_state[f"{key}_tags"] = []
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        new_tag = st.text_input(label, placeholder=placeholder, 
                               key=f"{key}_input", label_visibility="collapsed")
    
    with col2:
        if st.button("추가", key=f"{key}_add") and new_tag:
            if new_tag not in st.session_state[f"{key}_tags"]:
                st.session_state[f"{key}_tags"].append(new_tag)
                st.rerun()
    
    # 태그 표시
    if st.session_state[f"{key}_tags"]:
        tags_html = ""
        for idx, tag in enumerate(st.session_state[f"{key}_tags"]):
            tags_html += f"""
                <span style='
                    background-color: {THEME_CONFIG['secondary_color']}30;
                    padding: 5px 10px;
                    border-radius: 15px;
                    margin: 2px;
                    display: inline-block;
                '>
                    {tag}
                    <button onclick='removeTag({idx})' style='
                        background: none;
                        border: none;
                        color: red;
                        cursor: pointer;
                    '>×</button>
                </span>
            """
        st.markdown(f"<div>{tags_html}</div>", unsafe_allow_html=True)
    
    return st.session_state[f"{key}_tags"]

# =============================================================================
# 데이터 표시 컴포넌트
# =============================================================================

def styled_dataframe(df: pd.DataFrame, height: int = 400, 
                    highlight_cols: Optional[List[str]] = None):
    """스타일이 적용된 데이터프레임"""
    if highlight_cols:
        styled_df = df.style.apply(
            lambda x: ['background-color: #ffd70020' if col in highlight_cols else '' 
                      for col in x.index], 
            axis=1
        )
        st.dataframe(styled_df, height=height, use_container_width=True)
    else:
        st.dataframe(df, height=height, use_container_width=True)

def progress_bar(label: str, value: float, max_value: float = 100,
                show_percentage: bool = True):
    """진행률 표시줄"""
    progress = value / max_value
    percentage = int(progress * 100)
    
    if show_percentage:
        st.write(f"{label}: {percentage}%")
    else:
        st.write(label)
    
    st.progress(progress)

def status_badge(status: str, type: str = "info"):
    """상태 배지"""
    colors = {
        "success": "#28a745",
        "warning": "#ffc107",
        "error": "#dc3545",
        "info": "#17a2b8",
        "default": "#6c757d"
    }
    
    color = colors.get(type, colors["default"])
    
    st.markdown(f"""
        <span style='
            background-color: {color};
            color: white;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        '>{status}</span>
    """, unsafe_allow_html=True)

# =============================================================================
# 알림/피드백 컴포넌트
# =============================================================================

def show_message(message: str, type: str = "info", duration: int = 3):
    """임시 메시지 표시"""
    placeholder = st.empty()
    
    if type == "success":
        placeholder.success(message)
    elif type == "error":
        placeholder.error(message)
    elif type == "warning":
        placeholder.warning(message)
    else:
        placeholder.info(message)
    
    time.sleep(duration)
    placeholder.empty()

def confirm_dialog(message: str, key: str) -> bool:
    """확인 다이얼로그"""
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.write(message)
    with col2:
        confirm = st.button("✅ 확인", key=f"{key}_confirm")
    with col3:
        cancel = st.button("❌ 취소", key=f"{key}_cancel")
    
    return confirm and not cancel

def help_tooltip(text: str, help_text: str):
    """도움말 툴팁"""
    st.markdown(f"""
        <span>{text} 
            <span style='
                cursor: help;
                color: {THEME_CONFIG['primary_color']};
                font-size: 12px;
            ' title='{help_text}'>ⓘ</span>
        </span>
    """, unsafe_allow_html=True)

# =============================================================================
# 모듈 시스템 지원 컴포넌트
# =============================================================================

def module_card(module: Dict[str, Any], on_select: Optional[Callable] = None):
    """모듈 카드"""
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"### {module.get('icon', '🔬')} {module['name']}")
            st.write(module.get('description', ''))
            
            # 태그 표시
            tags = module.get('tags', [])
            if tags:
                tags_html = " ".join([f"<span style='background-color: #e0e0e0; padding: 2px 8px; border-radius: 10px; font-size: 12px; margin-right: 5px;'>{tag}</span>" for tag in tags])
                st.markdown(tags_html, unsafe_allow_html=True)
            
            # 메타 정보
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.caption(f"👤 {module.get('author', 'Unknown')}")
            with col_b:
                st.caption(f"⭐ {module.get('rating', 0)}/5")
            with col_c:
                st.caption(f"📥 {module.get('downloads', 0)}")
        
        with col2:
            if on_select:
                if st.button("선택", key=f"select_{module['id']}", use_container_width=True):
                    on_select(module)
            
            if module.get('installed', False):
                status_badge("설치됨", "success")
            else:
                status_badge("미설치", "default")

def module_selector(available_modules: List[Dict[str, Any]], 
                   selected_module: Optional[str] = None) -> Optional[str]:
    """모듈 선택기"""
    # 카테고리별 그룹화
    categories = {}
    for module in available_modules:
        cat = module.get('category', '기타')
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(module)
    
    # 카테고리 탭
    if categories:
        tabs = st.tabs(list(categories.keys()))
        
        for tab, (category, modules) in zip(tabs, categories.items()):
            with tab:
                cols = st.columns(2)
                for idx, module in enumerate(modules):
                    with cols[idx % 2]:
                        selected = st.button(
                            f"{module.get('icon', '🔬')} {module['name']}\n{module.get('description', '')[:50]}...",
                            key=f"mod_sel_{module['id']}",
                            use_container_width=True,
                            type="primary" if selected_module == module['id'] else "secondary"
                        )
                        if selected:
                            return module['id']
    
    return selected_module

def experiment_factor_input(factor: Dict[str, Any], key_prefix: str) -> Any:
    """실험 요인 입력 폼"""
    factor_type = factor.get('type', 'continuous')
    
    if factor_type == 'continuous':
        # 연속형 변수
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if factor.get('range'):
                value = range_input(
                    factor['name'],
                    factor['range'][0],
                    factor['range'][1],
                    key=f"{key_prefix}_{factor['id']}"
                )
            else:
                value = st.number_input(
                    factor['name'],
                    key=f"{key_prefix}_{factor['id']}",
                    help=factor.get('description')
                )
        
        with col2:
            st.caption(f"단위: {factor.get('unit', 'N/A')}")
        
        with col3:
            levels = st.number_input(
                "수준 수",
                min_value=2,
                max_value=10,
                value=3,
                key=f"{key_prefix}_{factor['id']}_levels"
            )
        
        return {'value': value, 'levels': levels}
    
    elif factor_type == 'categorical':
        # 범주형 변수
        options = factor.get('options', [])
        selected = st.multiselect(
            factor['name'],
            options=options,
            default=options[:1] if options else [],
            key=f"{key_prefix}_{factor['id']}",
            help=factor.get('description')
        )
        return {'value': selected}
    
    elif factor_type == 'ordinal':
        # 순서형 변수
        options = factor.get('options', [])
        selected = st.select_slider(
            factor['name'],
            options=options,
            key=f"{key_prefix}_{factor['id']}",
            help=factor.get('description')
        )
        return {'value': selected}

def response_variable_input(response: Dict[str, Any], key_prefix: str) -> Dict[str, Any]:
    """반응변수 입력 폼"""
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.write(f"**{response['name']}**")
        if response.get('description'):
            st.caption(response['description'])
    
    with col2:
        goal = st.selectbox(
            "목표",
            ["최대화", "최소화", "목표값", "범위"],
            key=f"{key_prefix}_{response['id']}_goal"
        )
    
    with col3:
        st.caption(f"단위: {response.get('unit', 'N/A')}")
    
    # 목표값 입력
    target_value = None
    if goal == "목표값":
        target_value = st.number_input(
            "목표값",
            key=f"{key_prefix}_{response['id']}_target"
        )
    elif goal == "범위":
        target_value = range_input(
            "목표 범위",
            0.0, 100.0,
            key=f"{key_prefix}_{response['id']}_range"
        )
    
    return {
        'response_id': response['id'],
        'name': response['name'],
        'goal': goal,
        'target': target_value,
        'unit': response.get('unit')
    }

# =============================================================================
# 시각화 컴포넌트
# =============================================================================

def plot_3d_surface(x: pd.Series, y: pd.Series, z: pd.Series, 
                   title: str = "3D Surface Plot",
                   x_label: str = "X", y_label: str = "Y", z_label: str = "Z"):
    """3D 표면 플롯"""
    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=8,
            color=z,
            colorscale='Viridis',
            showscale=True
        )
    )])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=z_label
        ),
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_pareto_chart(data: pd.DataFrame, x_col: str, y_col: str,
                     title: str = "Pareto Chart"):
    """파레토 차트"""
    # 데이터 정렬
    sorted_data = data.sort_values(by=y_col, ascending=False)
    
    # 누적 백분율 계산
    sorted_data['cumulative_percent'] = sorted_data[y_col].cumsum() / sorted_data[y_col].sum() * 100
    
    # 차트 생성
    fig = go.Figure()
    
    # 막대 그래프
    fig.add_trace(go.Bar(
        x=sorted_data[x_col],
        y=sorted_data[y_col],
        name='값',
        yaxis='y'
    ))
    
    # 누적 선 그래프
    fig.add_trace(go.Scatter(
        x=sorted_data[x_col],
        y=sorted_data['cumulative_percent'],
        name='누적 %',
        yaxis='y2',
        line=dict(color='red', width=2),
        mode='lines+markers'
    ))
    
    # 레이아웃 설정
    fig.update_layout(
        title=title,
        xaxis=dict(title=x_col),
        yaxis=dict(title=y_col, side='left'),
        yaxis2=dict(title='누적 백분율 (%)', side='right', overlaying='y', range=[0, 100]),
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# 유틸리티 함수
# =============================================================================

def save_current_work():
    """현재 작업 저장"""
    if st.session_state.get('current_project'):
        # Google Sheets 또는 로컬 저장소에 저장
        show_message("✅ 저장되었습니다", "success")
    else:
        show_message("⚠️ 저장할 프로젝트가 없습니다", "warning")

def format_datetime(dt: datetime) -> str:
    """날짜/시간 포맷팅"""
    return dt.strftime("%Y-%m-%d %H:%M")

def truncate_text(text: str, max_length: int = 100) -> str:
    """텍스트 자르기"""
    if len(text) > max_length:
        return text[:max_length-3] + "..."
    return text

def create_download_button(data: Any, filename: str, label: str = "다운로드"):
    """다운로드 버튼 생성"""
    if isinstance(data, pd.DataFrame):
        csv = data.to_csv(index=False)
        st.download_button(
            label=f"📥 {label}",
            data=csv,
            file_name=filename,
            mime="text/csv"
        )
    elif isinstance(data, dict):
        json_str = json.dumps(data, ensure_ascii=False, indent=2)
        st.download_button(
            label=f"📥 {label}",
            data=json_str,
            file_name=filename,
            mime="application/json"
        )
    else:
        st.download_button(
            label=f"📥 {label}",
            data=str(data),
            file_name=filename
        )

# =============================================================================
# 애니메이션 컴포넌트
# =============================================================================

def loading_animation(text: str = "처리 중..."):
    """로딩 애니메이션"""
    with st.spinner(text):
        time.sleep(0.5)  # 실제로는 작업 수행

def success_animation():
    """성공 애니메이션"""
    placeholder = st.empty()
    placeholder.markdown(
        "<h1 style='text-align: center; color: #28a745;'>✅</h1>",
        unsafe_allow_html=True
    )
    time.sleep(1)
    placeholder.empty()

# =============================================================================
# 반응형 레이아웃
# =============================================================================

def responsive_columns(num_items: int, max_cols: int = 4) -> List[Any]:
    """반응형 컬럼 생성"""
    # 아이템 수에 따라 적절한 컬럼 수 결정
    if num_items <= max_cols:
        return st.columns(num_items)
    else:
        # 여러 행으로 분할
        cols_per_row = max_cols
        rows = []
        for i in range(0, num_items, cols_per_row):
            row_items = min(cols_per_row, num_items - i)
            rows.append(st.columns(row_items))
        return [col for row in rows for col in row]

# =============================================================================
# 고급 입력 컴포넌트
# =============================================================================

def matrix_input(rows: List[str], cols: List[str], 
                default_value: float = 0.0,
                key: str = "matrix") -> pd.DataFrame:
    """행렬 입력 컴포넌트"""
    matrix_data = {}
    
    # 헤더 행
    header_cols = [""] + cols
    header_row = st.columns(len(header_cols))
    for i, col_name in enumerate(header_cols):
        with header_row[i]:
            if i > 0:
                st.write(f"**{col_name}**")
    
    # 데이터 행
    for row_idx, row_name in enumerate(rows):
        row_cols = st.columns(len(header_cols))
        matrix_data[row_name] = {}
        
        with row_cols[0]:
            st.write(f"**{row_name}**")
        
        for col_idx, col_name in enumerate(cols):
            with row_cols[col_idx + 1]:
                value = st.number_input(
                    label="",
                    value=default_value,
                    key=f"{key}_{row_idx}_{col_idx}",
                    label_visibility="collapsed"
                )
                matrix_data[row_name][col_name] = value
    
    return pd.DataFrame(matrix_data).T

def slider_with_input(label: str, min_value: float, max_value: float,
                     default_value: Optional[float] = None,
                     step: Optional[float] = None,
                     key: Optional[str] = None) -> float:
    """슬라이더와 입력 필드가 결합된 컴포넌트"""
    col1, col2 = st.columns([3, 1])
    
    default = default_value if default_value is not None else min_value
    
    with col1:
        slider_value = st.slider(
            label,
            min_value=min_value,
            max_value=max_value,
            value=default,
            step=step,
            key=f"{key}_slider" if key else None,
            label_visibility="collapsed"
        )
    
    with col2:
        input_value = st.number_input(
            label,
            min_value=min_value,
            max_value=max_value,
            value=slider_value,
            step=step,
            key=f"{key}_input" if key else None,
            label_visibility="collapsed"
        )
    
    return input_value

# =============================================================================
# 차트 템플릿
# =============================================================================

def create_scatter_matrix(df: pd.DataFrame, color_col: Optional[str] = None):
    """산점도 행렬"""
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("산점도 행렬을 생성하려면 최소 2개의 수치형 변수가 필요합니다.")
        return
    
    fig = px.scatter_matrix(
        df,
        dimensions=numeric_cols,
        color=color_col,
        title="변수 간 관계 분석",
        height=800
    )
    
    fig.update_traces(diagonal_visible=False)
    st.plotly_chart(fig, use_container_width=True)

def create_radar_chart(categories: List[str], values: List[float],
                      title: str = "Radar Chart"):
    """레이더 차트"""
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=title
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.1]
            )
        ),
        showlegend=False,
        title=title,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# 테이블 컴포넌트
# =============================================================================

def editable_dataframe(df: pd.DataFrame, key: str = "editable_df") -> pd.DataFrame:
    """편집 가능한 데이터프레임"""
    return st.data_editor(
        df,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        key=key
    )

def summary_table(data: Dict[str, Any], title: str = "요약"):
    """요약 테이블"""
    st.markdown(f"### {title}")
    
    summary_df = pd.DataFrame(list(data.items()), columns=['항목', '값'])
    st.table(summary_df)
