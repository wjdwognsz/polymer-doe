"""
📝 Project Setup Page - 프로젝트 설정
===========================================================================
연구 프로젝트를 생성, 관리, 편집하는 핵심 페이지
오프라인 우선 설계로 완전한 로컬 작동 지원
===========================================================================
"""

import streamlit as st
import pandas as pd
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import base64
import io

# ===========================================================================
# 🔧 페이지 설정 (반드시 최상단)
# ===========================================================================
st.set_page_config(
    page_title="프로젝트 설정 - Universal DOE",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================================================================
# 🔍 인증 확인
# ===========================================================================
if 'authenticated' not in st.session_state or not st.session_state.authenticated:
    if not st.session_state.get('guest_mode', False):
        st.warning("🔐 로그인이 필요합니다")
        st.switch_page("pages/0_🔐_Login.py")
        st.stop()

# ===========================================================================
# 📦 모듈 임포트
# ===========================================================================
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from utils.database_manager import DatabaseManager
    from utils.common_ui import CommonUI
    from utils.api_manager import APIManager
    from utils.notification_manager import NotificationManager
    from modules.module_registry import ModuleRegistry
    from config.app_config import APP_CONFIG
    from config.theme_config import THEME_CONFIG, COLORS
except ImportError as e:
    st.error(f"필수 모듈 임포트 오류: {e}")
    st.stop()

# ===========================================================================
# 🔧 설정 및 상수
# ===========================================================================

logger = logging.getLogger(__name__)

# 프로젝트 관련 상수
PROJECT_TYPES = {
    'polymer': {
        'name': '고분자 연구',
        'icon': '🧬',
        'description': '고분자 합성, 가공, 특성 연구',
        'subcategories': ['합성', '블렌드', '복합재료', '나노재료', '바이오고분자']
    },
    'chemistry': {
        'name': '화학 실험',
        'icon': '🧪',
        'description': '유기/무기 합성, 촉매, 반응 최적화',
        'subcategories': ['유기합성', '무기합성', '촉매', '전기화학', '분석화학']
    },
    'materials': {
        'name': '재료 과학',
        'icon': '⚛️',
        'description': '금속, 세라믹, 반도체 등 재료 연구',
        'subcategories': ['금속', '세라믹', '반도체', '복합재료', '나노재료']
    },
    'biology': {
        'name': '생명 과학',
        'icon': '🧫',
        'description': '세포 배양, 단백질, 효소 연구',
        'subcategories': ['세포배양', '단백질', '효소', '미생물', '유전자']
    },
    'general': {
        'name': '일반 연구',
        'icon': '🔬',
        'description': '기타 과학/공학 연구',
        'subcategories': ['물리', '공학', '환경', '에너지', '기타']
    }
}

PROJECT_STATUS = {
    'active': {'name': '활성', 'icon': '🟢', 'color': COLORS['success']},
    'paused': {'name': '일시중지', 'icon': '🟡', 'color': COLORS['warning']},
    'completed': {'name': '완료', 'icon': '🔵', 'color': COLORS['info']},
    'archived': {'name': '보관', 'icon': '⚫', 'color': COLORS['secondary']}
}

PROJECT_VISIBILITY = {
    'private': {'name': '비공개', 'icon': '🔒', 'description': '나만 볼 수 있음'},
    'team': {'name': '팀 공개', 'icon': '👥', 'description': '초대된 멤버만'},
    'public': {'name': '전체 공개', 'icon': '🌍', 'description': '모든 사용자'}
}

PERMISSION_LEVELS = {
    'owner': {
        'name': '소유자',
        'can_edit': True,
        'can_delete': True,
        'can_invite': True,
        'can_remove_members': True,
        'can_change_visibility': True,
        'can_export': True
    },
    'editor': {
        'name': '편집자',
        'can_edit': True,
        'can_delete': False,
        'can_invite': True,
        'can_remove_members': False,
        'can_change_visibility': False,
        'can_export': True
    },
    'viewer': {
        'name': '뷰어',
        'can_edit': False,
        'can_delete': False,
        'can_invite': False,
        'can_remove_members': False,
        'can_change_visibility': False,
        'can_export': True
    }
}

# ===========================================================================
# 📊 프로젝트 설정 페이지 클래스
# ===========================================================================

class ProjectSetupPage:
    """프로젝트 설정 페이지"""
    
    def __init__(self):
        """초기화"""
        self.db_manager = DatabaseManager()
        self.ui = CommonUI()
        self.api_manager = APIManager()
        self.notification_manager = NotificationManager()
        self.module_registry = ModuleRegistry()
        
        # 사용자 정보
        self.user = st.session_state.get('user', {})
        self.user_id = self.user.get('user_id') or st.session_state.get('user_id')
        self.is_guest = st.session_state.get('guest_mode', False)
        
        # 세션 상태 초기화
        self._initialize_session_state()
        
    def _initialize_session_state(self):
        """세션 상태 초기화"""
        defaults = {
            'project_step': 0,
            'new_project': {},
            'selected_modules': [],
            'project_view': 'grid',
            'show_ai_details': False,
            'ai_recommendations': None,
            'editing_project': None,
            'project_filter': {'status': '전체', 'search': ''},
            'selected_project_id': None,
            'show_template_save': False
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
                
    def render(self):
        """메인 렌더링"""
        # 헤더
        self._render_header()
        
        # 게스트 모드 체크
        if self.is_guest:
            st.info("👀 게스트 모드로 둘러보는 중입니다. 프로젝트 생성은 로그인 후 가능합니다.")
            
        # 탭 선택
        tab1, tab2, tab3, tab4 = st.tabs([
            "📁 내 프로젝트",
            "➕ 새 프로젝트",
            "📚 템플릿",
            "⚙️ 설정"
        ])
        
        with tab1:
            self._render_projects_tab()
            
        with tab2:
            if self.is_guest:
                st.warning("프로젝트 생성은 로그인이 필요합니다.")
                if st.button("로그인하기", type="primary"):
                    st.switch_page("pages/0_🔐_Login.py")
            else:
                self._render_new_project_tab()
                
        with tab3:
            self._render_templates_tab()
            
        with tab4:
            self._render_settings_tab()
            
    def _render_header(self):
        """헤더 렌더링"""
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col1:
            st.markdown("## 📝 프로젝트 설정")
            
        with col2:
            # AI 설명 모드 토글
            show_details = st.toggle(
                "AI 상세 설명",
                value=st.session_state.show_ai_details,
                help="AI의 추천 이유와 배경 지식을 상세히 표시합니다"
            )
            st.session_state.show_ai_details = show_details
            
        with col3:
            # 빠른 작업
            if not self.is_guest:
                if st.button("📤 내보내기"):
                    self._show_export_dialog()
                    
    # ===========================================================================
    # 📁 내 프로젝트 탭
    # ===========================================================================
    
    def _render_projects_tab(self):
        """프로젝트 목록 탭"""
        # 필터 및 검색
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        
        with col1:
            search = st.text_input(
                "🔍 검색",
                value=st.session_state.project_filter['search'],
                placeholder="프로젝트명, 태그..."
            )
            st.session_state.project_filter['search'] = search
            
        with col2:
            status = st.selectbox(
                "상태",
                options=['전체'] + [info['name'] for info in PROJECT_STATUS.values()],
                index=0
            )
            st.session_state.project_filter['status'] = status
            
        with col3:
            sort_by = st.selectbox(
                "정렬",
                options=['최근 수정', '이름순', '생성일', '진행률'],
                index=0
            )
            
        with col4:
            # 보기 모드 전환
            view_mode = st.radio(
                "보기",
                options=['grid', 'list'],
                format_func=lambda x: '⊞' if x == 'grid' else '☰',
                horizontal=True,
                label_visibility='collapsed'
            )
            st.session_state.project_view = view_mode
            
        # 프로젝트 가져오기
        projects = self._get_filtered_projects(search, status, sort_by)
        
        if not projects:
            self.ui.show_empty_state(
                "프로젝트가 없습니다",
                "새 프로젝트를 생성해보세요",
                action_label="프로젝트 생성",
                action_callback=lambda: st.session_state.update({'project_step': 0})
            )
        else:
            # 통계 표시
            self._render_project_stats(projects)
            
            # 프로젝트 목록
            if view_mode == 'grid':
                self._render_projects_grid(projects)
            else:
                self._render_projects_list(projects)
                
    def _render_project_stats(self, projects: List[Dict]):
        """프로젝트 통계"""
        col1, col2, col3, col4 = st.columns(4)
        
        # 통계 계산
        total = len(projects)
        active = len([p for p in projects if p['status'] == 'active'])
        completed = len([p for p in projects if p['status'] == 'completed'])
        collab = len([p for p in projects if len(p.get('collaborators', [])) > 1])
        
        with col1:
            st.metric("전체 프로젝트", total)
            
        with col2:
            st.metric("진행중", active, f"{active/total*100:.0f}%" if total > 0 else "0%")
            
        with col3:
            st.metric("완료", completed)
            
        with col4:
            st.metric("협업 프로젝트", collab)
            
    def _render_projects_grid(self, projects: List[Dict]):
        """그리드 뷰"""
        cols = st.columns(3)
        
        for idx, project in enumerate(projects):
            with cols[idx % 3]:
                self._render_project_card(project)
                
    def _render_project_card(self, project: Dict):
        """프로젝트 카드"""
        status_info = PROJECT_STATUS.get(project['status'], PROJECT_STATUS['active'])
        visibility_info = PROJECT_VISIBILITY.get(project['visibility'], PROJECT_VISIBILITY['private'])
        
        # 카드 컨테이너
        with st.container():
            st.markdown(f"""
                <div style="
                    background: {COLORS['background_secondary']};
                    padding: 20px;
                    border-radius: 12px;
                    border: 1px solid {COLORS['border']};
                    margin-bottom: 20px;
                    transition: all 0.3s;
                ">
                    <div style="display: flex; justify-content: space-between; align-items: start;">
                        <h4 style="margin: 0;">{project['name']}</h4>
                        <span style="color: {status_info['color']};">{status_info['icon']}</span>
                    </div>
                    <p style="color: #666; font-size: 14px; margin: 10px 0;">
                        {project.get('description', '설명 없음')[:100]}...
                    </p>
                    <div style="display: flex; gap: 10px; flex-wrap: wrap; margin: 10px 0;">
                        <span style="background: {COLORS['primary']}20; color: {COLORS['primary']}; 
                                     padding: 2px 8px; border-radius: 4px; font-size: 12px;">
                            {PROJECT_TYPES.get(project['type'], {}).get('icon', '🔬')} 
                            {PROJECT_TYPES.get(project['type'], {}).get('name', '일반')}
                        </span>
                        <span style="background: {COLORS['secondary']}20; color: {COLORS['secondary']}; 
                                     padding: 2px 8px; border-radius: 4px; font-size: 12px;">
                            {visibility_info['icon']} {visibility_info['name']}
                        </span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; 
                                color: #666; font-size: 12px;">
                        <span>🧪 {project.get('experiment_count', 0)} 실험</span>
                        <span>👥 {len(project.get('collaborators', []))} 멤버</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # 액션 버튼
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("열기", key=f"open_{project['id']}", use_container_width=True):
                    st.session_state.selected_project_id = project['id']
                    st.switch_page("pages/3_🧪_Experiment_Design.py")
                    
            with col2:
                if st.button("편집", key=f"edit_{project['id']}", use_container_width=True,
                           disabled=self.is_guest or not self._can_edit_project(project)):
                    st.session_state.editing_project = project
                    
            with col3:
                if st.button("⋮", key=f"more_{project['id']}", use_container_width=True):
                    self._show_project_menu(project)
                    
    def _render_projects_list(self, projects: List[Dict]):
        """리스트 뷰"""
        # 테이블 헤더
        df_data = []
        
        for project in projects:
            status_info = PROJECT_STATUS.get(project['status'], PROJECT_STATUS['active'])
            visibility_info = PROJECT_VISIBILITY.get(project['visibility'], PROJECT_VISIBILITY['private'])
            
            df_data.append({
                '상태': f"{status_info['icon']} {status_info['name']}",
                '프로젝트명': project['name'],
                '유형': PROJECT_TYPES.get(project['type'], {}).get('name', '일반'),
                '공개범위': f"{visibility_info['icon']} {visibility_info['name']}",
                '실험': f"{project.get('experiment_count', 0)}개",
                '멤버': f"{len(project.get('collaborators', []))}명",
                '수정일': project.get('updated_at', '').split('T')[0] if project.get('updated_at') else '-'
            })
            
        df = pd.DataFrame(df_data)
        
        # 대화형 테이블
        selected = st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            selection_mode="single-row",
            on_select="rerun"
        )
        
        # 선택된 프로젝트 처리
        if selected and selected.selection.rows:
            selected_idx = selected.selection.rows[0]
            selected_project = projects[selected_idx]
            
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                if st.button("🔍 열기", use_container_width=True):
                    st.session_state.selected_project_id = selected_project['id']
                    st.switch_page("pages/3_🧪_Experiment_Design.py")
                    
            with col2:
                if st.button("✏️ 편집", use_container_width=True,
                           disabled=self.is_guest or not self._can_edit_project(selected_project)):
                    st.session_state.editing_project = selected_project
                    
    # ===========================================================================
    # ➕ 새 프로젝트 탭
    # ===========================================================================
    
    def _render_new_project_tab(self):
        """새 프로젝트 생성 탭"""
        # 진행 상태 표시
        steps = ["기본 정보", "실험 모듈", "AI 설정", "협업 설정"]
        current_step = st.session_state.project_step
        
        # 진행률 표시
        progress = (current_step + 1) / len(steps)
        st.progress(progress)
        
        # 단계 표시
        cols = st.columns(len(steps))
        for idx, (col, step) in enumerate(zip(cols, steps)):
            with col:
                if idx < current_step:
                    st.success(f"✅ {step}")
                elif idx == current_step:
                    st.info(f"👉 {step}")
                else:
                    st.text(f"⭕ {step}")
                    
        st.divider()
        
        # 현재 단계 렌더링
        if current_step == 0:
            self._render_basic_info_step()
        elif current_step == 1:
            self._render_module_selection_step()
        elif current_step == 2:
            self._render_ai_settings_step()
        elif current_step == 3:
            self._render_collaboration_step()
            
    def _render_basic_info_step(self):
        """1단계: 기본 정보"""
        st.markdown("### 1️⃣ 기본 정보 입력")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 프로젝트명
            name = st.text_input(
                "프로젝트명 *",
                value=st.session_state.new_project.get('name', ''),
                placeholder="예: PET 필름 투명도 개선",
                help="프로젝트를 구별할 수 있는 명확한 이름"
            )
            
            # 프로젝트 유형
            project_type = st.selectbox(
                "프로젝트 유형 *",
                options=list(PROJECT_TYPES.keys()),
                format_func=lambda x: f"{PROJECT_TYPES[x]['icon']} {PROJECT_TYPES[x]['name']}",
                index=list(PROJECT_TYPES.keys()).index(
                    st.session_state.new_project.get('type', 'polymer')
                )
            )
            
            # 세부 분야
            if project_type:
                subcategory = st.selectbox(
                    "세부 분야",
                    options=PROJECT_TYPES[project_type]['subcategories'],
                    index=0
                )
            else:
                subcategory = None
                
        with col2:
            # 공개 범위
            visibility = st.radio(
                "공개 범위",
                options=list(PROJECT_VISIBILITY.keys()),
                format_func=lambda x: f"{PROJECT_VISIBILITY[x]['icon']} {PROJECT_VISIBILITY[x]['name']}",
                index=list(PROJECT_VISIBILITY.keys()).index(
                    st.session_state.new_project.get('visibility', 'private')
                ),
                help="나중에 변경 가능합니다"
            )
            
            # 우선순위
            priority = st.select_slider(
                "우선순위",
                options=['낮음', '보통', '높음', '긴급'],
                value=st.session_state.new_project.get('priority', '보통')
            )
            
            # 예상 기간
            duration = st.number_input(
                "예상 기간 (주)",
                min_value=1,
                max_value=52,
                value=st.session_state.new_project.get('duration', 4),
                help="대략적인 프로젝트 기간"
            )
            
        # 설명
        description = st.text_area(
            "프로젝트 설명",
            value=st.session_state.new_project.get('description', ''),
            placeholder="프로젝트의 배경, 목표, 기대효과 등을 자세히 설명해주세요",
            height=150
        )
        
        # 목표 (태그 형식)
        st.markdown("#### 프로젝트 목표")
        objectives = st.multiselect(
            "달성하고자 하는 목표를 선택하세요",
            options=[
                "성능 향상", "비용 절감", "공정 개선", "신제품 개발",
                "품질 향상", "환경 친화", "안전성 향상", "생산성 증대"
            ],
            default=st.session_state.new_project.get('objectives', [])
        )
        
        # AI 추천 받기
        if st.button("🤖 AI 추천 받기", help="프로젝트 설정에 대한 AI 추천"):
            self._get_ai_project_recommendations(name, project_type, description)
            
        # AI 추천 표시
        if st.session_state.ai_recommendations:
            self._render_ai_recommendations()
            
        # 다음 단계 버튼
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col3:
            if st.button("다음 →", type="primary", use_container_width=True,
                       disabled=not name):
                # 데이터 저장
                st.session_state.new_project.update({
                    'name': name,
                    'type': project_type,
                    'subcategory': subcategory,
                    'visibility': visibility,
                    'priority': priority,
                    'duration': duration,
                    'description': description,
                    'objectives': objectives
                })
                st.session_state.project_step = 1
                st.rerun()
                
    def _render_module_selection_step(self):
        """2단계: 실험 모듈 선택"""
        st.markdown("### 2️⃣ 실험 모듈 선택")
        
        # 모듈 카테고리
        categories = self.module_registry.get_categories()
        selected_category = st.selectbox(
            "모듈 카테고리",
            options=['전체'] + categories,
            index=0
        )
        
        # 모듈 목록 가져오기
        if selected_category == '전체':
            modules = self.module_registry.list_modules()
        else:
            modules = self.module_registry.list_modules(category=selected_category)
            
        # 모듈 선택
        st.markdown("#### 사용할 모듈 선택")
        
        # 모듈 카드 그리드
        cols = st.columns(3)
        for idx, module in enumerate(modules):
            with cols[idx % 3]:
                self._render_module_card(module)
                
        # 선택된 모듈 표시
        if st.session_state.selected_modules:
            st.markdown("#### 선택된 모듈")
            for module_id in st.session_state.selected_modules:
                module = self.module_registry.get_module(module_id)
                if module:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        module_info = module.get_module_info()
                        st.write(f"• {module_info['name']} - {module_info['description']}")
                    with col2:
                        if st.button("제거", key=f"remove_{module_id}"):
                            st.session_state.selected_modules.remove(module_id)
                            st.rerun()
                            
        # 네비게이션 버튼
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col1:
            if st.button("← 이전", use_container_width=True):
                st.session_state.project_step = 0
                st.rerun()
                
        with col3:
            if st.button("다음 →", type="primary", use_container_width=True,
                       disabled=not st.session_state.selected_modules):
                st.session_state.project_step = 2
                st.rerun()
                
    def _render_module_card(self, module: Dict):
        """모듈 카드"""
        module_id = module['id']
        is_selected = module_id in st.session_state.selected_modules
        
        # 카드 스타일
        border_color = COLORS['primary'] if is_selected else COLORS['border']
        bg_color = f"{COLORS['primary']}10" if is_selected else COLORS['background_secondary']
        
        st.markdown(f"""
            <div style="
                background: {bg_color};
                border: 2px solid {border_color};
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 10px;
                cursor: pointer;
            ">
                <h5 style="margin: 0 0 10px 0;">{module['name']}</h5>
                <p style="font-size: 12px; color: #666; margin: 0;">
                    {module['description'][:80]}...
                </p>
                <div style="margin-top: 10px;">
                    <span style="font-size: 11px; color: #999;">
                        v{module['version']} • {module['author']}
                    </span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button(
            "✅ 선택됨" if is_selected else "선택",
            key=f"select_{module_id}",
            use_container_width=True,
            type="primary" if is_selected else "secondary"
        ):
            if is_selected:
                st.session_state.selected_modules.remove(module_id)
            else:
                st.session_state.selected_modules.append(module_id)
            st.rerun()
            
    def _render_ai_settings_step(self):
        """3단계: AI 설정"""
        st.markdown("### 3️⃣ AI 지원 설정")
        
        # AI 사용 여부
        use_ai = st.checkbox(
            "AI 지원 활성화",
            value=st.session_state.new_project.get('use_ai', True),
            help="실험 설계와 데이터 분석에 AI를 활용합니다"
        )
        
        if use_ai:
            col1, col2 = st.columns(2)
            
            with col1:
                # AI 엔진 선택
                available_engines = self.api_manager.get_available_engines()
                selected_engines = st.multiselect(
                    "사용할 AI 엔진",
                    options=list(available_engines.keys()),
                    default=st.session_state.new_project.get('ai_engines', ['gemini']),
                    format_func=lambda x: available_engines[x]['name']
                )
                
                # AI 모드
                ai_mode = st.radio(
                    "AI 지원 수준",
                    options=['자동', '추천만', '수동'],
                    index=['자동', '추천만', '수동'].index(
                        st.session_state.new_project.get('ai_mode', '자동')
                    ),
                    help="AI가 개입하는 수준을 설정합니다"
                )
                
            with col2:
                # 설명 상세도
                explanation_detail = st.select_slider(
                    "AI 설명 상세도",
                    options=['간단', '보통', '상세', '전문가'],
                    value=st.session_state.new_project.get('explanation_detail', '보통'),
                    help="AI 응답의 상세 수준"
                )
                
                # 자동 최적화
                auto_optimization = st.checkbox(
                    "자동 최적화 허용",
                    value=st.session_state.new_project.get('auto_optimization', False),
                    help="AI가 실험 조건을 자동으로 최적화합니다"
                )
                
            # AI 활용 영역
            st.markdown("#### AI 활용 영역")
            ai_features = st.multiselect(
                "AI를 활용할 기능을 선택하세요",
                options=[
                    "실험 설계 추천",
                    "요인 수준 최적화",
                    "데이터 분석 자동화",
                    "이상치 탐지",
                    "결과 해석 지원",
                    "보고서 자동 생성"
                ],
                default=st.session_state.new_project.get('ai_features', [
                    "실험 설계 추천",
                    "데이터 분석 자동화",
                    "결과 해석 지원"
                ])
            )
        else:
            selected_engines = []
            ai_mode = None
            explanation_detail = None
            auto_optimization = False
            ai_features = []
            
        # 네비게이션 버튼
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col1:
            if st.button("← 이전", use_container_width=True):
                st.session_state.project_step = 1
                st.rerun()
                
        with col3:
            if st.button("다음 →", type="primary", use_container_width=True):
                # 데이터 저장
                st.session_state.new_project.update({
                    'use_ai': use_ai,
                    'ai_engines': selected_engines,
                    'ai_mode': ai_mode,
                    'explanation_detail': explanation_detail,
                    'auto_optimization': auto_optimization,
                    'ai_features': ai_features
                })
                st.session_state.project_step = 3
                st.rerun()
                
    def _render_collaboration_step(self):
        """4단계: 협업 설정"""
        st.markdown("### 4️⃣ 협업 설정")
        
        # 협업 모드
        collab_mode = st.radio(
            "협업 모드",
            options=['personal', 'team', 'open'],
            format_func=lambda x: {
                'personal': '🔒 개인 프로젝트',
                'team': '👥 팀 프로젝트',
                'open': '🌍 오픈 프로젝트'
            }[x],
            index=['personal', 'team', 'open'].index(
                st.session_state.new_project.get('collab_mode', 'personal')
            )
        )
        
        if collab_mode in ['team', 'open']:
            # 팀원 초대
            st.markdown("#### 팀원 초대")
            
            # 이메일 입력
            invites = st.text_area(
                "이메일 주소 입력",
                placeholder="한 줄에 하나씩 입력\nexample1@email.com\nexample2@email.com",
                height=100,
                value=st.session_state.new_project.get('invitations', {}).get('emails_text', '')
            )
            
            # 권한 설정
            default_role = st.selectbox(
                "기본 권한",
                options=['viewer', 'editor'],
                format_func=lambda x: PERMISSION_LEVELS[x]['name'],
                index=1
            )
            
            # 추가 설정
            col1, col2 = st.columns(2)
            
            with col1:
                require_approval = st.checkbox(
                    "참여 승인 필요",
                    value=st.session_state.new_project.get('require_approval', True),
                    help="새 멤버 참여시 승인이 필요합니다"
                )
                
            with col2:
                allow_guest_view = st.checkbox(
                    "게스트 읽기 허용",
                    value=st.session_state.new_project.get('allow_guest_view', False),
                    help="로그인하지 않은 사용자도 볼 수 있습니다"
                )
        else:
            invites = ""
            default_role = "viewer"
            require_approval = False
            allow_guest_view = False
            
        # 프로젝트 생성 요약
        st.markdown("### 📋 프로젝트 생성 요약")
        
        # 설정 요약
        col1, col2 = st.columns(2)
        
        with col1:
            project = st.session_state.new_project
            
            st.markdown("#### 기본 정보")
            st.write(f"- **프로젝트명**: {project.get('name', '')}")
            st.write(f"- **유형**: {PROJECT_TYPES.get(project.get('type', ''), {}).get('name', '')}")
            st.write(f"- **공개 범위**: {PROJECT_VISIBILITY.get(project.get('visibility', ''), {}).get('name', '')}")
            st.write(f"- **예상 기간**: {project.get('duration', 0)}주")
            
            st.markdown("#### 실험 모듈")
            if st.session_state.selected_modules:
                for module_id in st.session_state.selected_modules:
                    module = self.module_registry.get_module(module_id)
                    if module:
                        st.write(f"- {module.get_module_info()['name']}")
                        
        with col2:
            st.markdown("#### AI 설정")
            if project.get('use_ai'):
                st.write(f"- **AI 엔진**: {', '.join(project.get('ai_engines', []))}")
                st.write(f"- **지원 수준**: {project.get('ai_mode', '')}")
                st.write(f"- **설명 상세도**: {project.get('explanation_detail', '')}")
            else:
                st.write("- AI 지원 비활성화")
                
            st.markdown("#### 협업 설정")
            st.write(f"- **모드**: {{'personal': '개인', 'team': '팀', 'open': '오픈'}[collab_mode]}")
            if invites:
                invite_list = [e.strip() for e in invites.split('\n') if e.strip()]
                st.write(f"- **초대**: {len(invite_list)}명")
                
        # 템플릿 저장 옵션
        save_as_template = st.checkbox(
            "이 설정을 템플릿으로 저장",
            value=st.session_state.show_template_save
        )
        
        if save_as_template:
            template_name = st.text_input(
                "템플릿 이름",
                placeholder="예: 고분자 합성 기본 템플릿"
            )
        else:
            template_name = None
            
        # 최종 버튼
        col1, col2, col3 = st.columns([1, 2, 2])
        
        with col1:
            if st.button("← 이전", use_container_width=True):
                st.session_state.project_step = 2
                st.rerun()
                
        with col2:
            if st.button("📥 임시 저장", use_container_width=True):
                self._save_draft_project()
                
        with col3:
            if st.button("✅ 프로젝트 생성", type="primary", use_container_width=True,
                       disabled=not st.session_state.new_project.get('name')):
                # 최종 데이터 저장
                st.session_state.new_project.update({
                    'collab_mode': collab_mode,
                    'invitations': {
                        'emails_text': invites,
                        'emails': [e.strip() for e in invites.split('\n') if e.strip()],
                        'default_role': default_role
                    },
                    'require_approval': require_approval,
                    'allow_guest_view': allow_guest_view
                })
                
                # 프로젝트 생성
                success, project_id = self._create_project(template_name)
                
                if success:
                    st.success("✅ 프로젝트가 성공적으로 생성되었습니다!")
                    st.balloons()
                    
                    # 초대 발송
                    if invites and collab_mode in ['team', 'open']:
                        self._send_invitations(project_id)
                        
                    # 리셋
                    st.session_state.new_project = {}
                    st.session_state.selected_modules = []
                    st.session_state.project_step = 0
                    
                    # 프로젝트로 이동
                    time.sleep(1)
                    st.session_state.selected_project_id = project_id
                    st.switch_page("pages/3_🧪_Experiment_Design.py")
                    
    # ===========================================================================
    # 📚 템플릿 탭
    # ===========================================================================
    
    def _render_templates_tab(self):
        """템플릿 탭"""
        # 템플릿 카테고리
        template_categories = ['전체', '내 템플릿', '공유 템플릿', '공식 템플릿']
        selected_category = st.selectbox(
            "템플릿 카테고리",
            options=template_categories,
            index=0
        )
        
        # 템플릿 목록 가져오기
        templates = self._get_templates(selected_category)
        
        if not templates:
            st.info("사용 가능한 템플릿이 없습니다.")
        else:
            # 템플릿 그리드
            cols = st.columns(3)
            for idx, template in enumerate(templates):
                with cols[idx % 3]:
                    self._render_template_card(template)
                    
    def _render_template_card(self, template: Dict):
        """템플릿 카드"""
        st.markdown(f"""
            <div style="
                background: {COLORS['background_secondary']};
                padding: 15px;
                border-radius: 8px;
                border: 1px solid {COLORS['border']};
                margin-bottom: 15px;
            ">
                <h5 style="margin: 0 0 10px 0;">{template['name']}</h5>
                <p style="font-size: 12px; color: #666; margin: 0 0 10px 0;">
                    {template['description']}
                </p>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 11px; color: #999;">
                        ⭐ {template.get('rating', 0):.1f} • 
                        📥 {template.get('usage_count', 0)}회 사용
                    </span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("미리보기", key=f"preview_tmpl_{template['id']}", use_container_width=True):
                self._show_template_preview(template)
                
        with col2:
            if st.button("사용하기", key=f"use_tmpl_{template['id']}", use_container_width=True,
                       type="primary", disabled=self.is_guest):
                self._use_template(template)
                
    # ===========================================================================
    # ⚙️ 설정 탭
    # ===========================================================================
    
    def _render_settings_tab(self):
        """설정 탭"""
        st.markdown("### ⚙️ 프로젝트 기본 설정")
        
        # 기본 설정
        st.markdown("#### 새 프로젝트 기본값")
        
        col1, col2 = st.columns(2)
        
        with col1:
            default_visibility = st.selectbox(
                "기본 공개 범위",
                options=list(PROJECT_VISIBILITY.keys()),
                format_func=lambda x: f"{PROJECT_VISIBILITY[x]['icon']} {PROJECT_VISIBILITY[x]['name']}",
                index=0
            )
            
            default_duration = st.number_input(
                "기본 프로젝트 기간 (주)",
                min_value=1,
                max_value=52,
                value=4
            )
            
        with col2:
            auto_save = st.checkbox(
                "자동 저장 활성화",
                value=True,
                help="5분마다 자동으로 저장합니다"
            )
            
            show_tips = st.checkbox(
                "도움말 표시",
                value=True,
                help="프로젝트 생성시 도움말을 표시합니다"
            )
            
        # 알림 설정
        st.markdown("#### 알림 설정")
        
        col1, col2 = st.columns(2)
        
        with col1:
            notify_invites = st.checkbox("초대 알림", value=True)
            notify_updates = st.checkbox("프로젝트 업데이트 알림", value=True)
            
        with col2:
            notify_comments = st.checkbox("댓글 알림", value=True)
            notify_milestones = st.checkbox("마일스톤 알림", value=True)
            
        # 저장 버튼
        if st.button("설정 저장", type="primary"):
            # 설정 저장 로직
            st.success("설정이 저장되었습니다!")
            
    # ===========================================================================
    # 🔧 헬퍼 메서드
    # ===========================================================================
    
    def _get_filtered_projects(self, search: str, status: str, sort_by: str) -> List[Dict]:
        """필터링된 프로젝트 목록"""
        try:
            # 모든 프로젝트 가져오기
            if self.is_guest:
                # 게스트는 공개 프로젝트만
                projects = self.db_manager.get_public_projects()
            else:
                # 사용자 프로젝트 + 협업 프로젝트
                projects = self.db_manager.get_user_projects(self.user_id)
                
            # 검색 필터
            if search:
                search_lower = search.lower()
                projects = [
                    p for p in projects
                    if search_lower in p['name'].lower() or
                    search_lower in p.get('description', '').lower() or
                    any(search_lower in tag for tag in p.get('tags', []))
                ]
                
            # 상태 필터
            if status != '전체':
                status_key = None
                for key, info in PROJECT_STATUS.items():
                    if info['name'] == status:
                        status_key = key
                        break
                        
                if status_key:
                    projects = [p for p in projects if p['status'] == status_key]
                    
            # 정렬
            if sort_by == '최근 수정':
                projects.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
            elif sort_by == '이름순':
                projects.sort(key=lambda x: x['name'])
            elif sort_by == '생성일':
                projects.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            elif sort_by == '진행률':
                projects.sort(
                    key=lambda x: x.get('progress', 0) / x.get('total_experiments', 1) if x.get('total_experiments', 0) > 0 else 0,
                    reverse=True
                )
                
            return projects
            
        except Exception as e:
            logger.error(f"프로젝트 필터링 오류: {e}")
            return []
            
    def _can_edit_project(self, project: Dict) -> bool:
        """프로젝트 편집 권한 확인"""
        if self.is_guest:
            return False
            
        # 소유자 확인
        if project.get('owner_id') == self.user_id:
            return True
            
        # 협업자 권한 확인
        for collab in project.get('collaborators', []):
            if collab['user_id'] == self.user_id:
                role = collab.get('role', 'viewer')
                return PERMISSION_LEVELS[role]['can_edit']
                
        return False
        
    def _get_ai_project_recommendations(self, name: str, project_type: str, description: str):
        """AI 프로젝트 추천"""
        if not self.api_manager.has_api_key():
            st.warning("AI 추천을 위한 API 키가 설정되지 않았습니다.")
            return
            
        with st.spinner("AI가 프로젝트를 분석하고 있습니다..."):
            prompt = f"""
            프로젝트 정보:
            - 이름: {name}
            - 유형: {PROJECT_TYPES.get(project_type, {}).get('name', project_type)}
            - 설명: {description}
            
            다음을 추천해주세요:
            1. 적합한 실험 모듈 3개
            2. 프로젝트 목표 제안
            3. 예상되는 도전 과제
            4. 유사 프로젝트 사례
            
            JSON 형식으로 응답하세요.
            """
            
            response = self.api_manager.get_completion(prompt, response_format="json")
            
            if response['success']:
                st.session_state.ai_recommendations = response['data']
            else:
                st.error(f"AI 추천 실패: {response['error']}")
                
    def _render_ai_recommendations(self):
        """AI 추천 렌더링"""
        recommendations = st.session_state.ai_recommendations
        
        with st.expander("🤖 AI 추천 사항", expanded=True):
            # 기본 추천 (항상 표시)
            st.markdown("**추천 실험 모듈:**")
            for module in recommendations.get('recommended_modules', []):
                st.write(f"• {module['name']} - {module['reason']}")
                
            st.markdown("**추천 프로젝트 목표:**")
            for goal in recommendations.get('suggested_goals', []):
                st.write(f"• {goal}")
                
            # 상세 설명 (토글에 따라)
            if st.session_state.show_ai_details:
                st.divider()
                st.markdown("**🔍 상세 분석:**")
                
                st.markdown("**예상되는 도전 과제:**")
                for challenge in recommendations.get('challenges', []):
                    st.warning(f"⚠️ {challenge['issue']}")
                    st.caption(f"💡 대응 방안: {challenge['solution']}")
                    
                st.markdown("**유사 프로젝트 사례:**")
                for case in recommendations.get('similar_projects', []):
                    with st.container():
                        st.write(f"**{case['name']}**")
                        st.caption(f"결과: {case['outcome']}")
                        st.caption(f"핵심 성공 요인: {case['key_factor']}")
                        
    def _create_project(self, template_name: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """프로젝트 생성"""
        try:
            project_data = st.session_state.new_project.copy()
            
            # 프로젝트 ID 생성
            project_id = str(uuid.uuid4())
            
            # 프로젝트 데이터 구성
            project = {
                'id': project_id,
                'owner_id': self.user_id,
                'name': project_data['name'],
                'type': project_data['type'],
                'subcategory': project_data.get('subcategory'),
                'description': project_data.get('description', ''),
                'visibility': project_data.get('visibility', 'private'),
                'status': 'active',
                'priority': project_data.get('priority', '보통'),
                'duration_weeks': project_data.get('duration', 4),
                'objectives': project_data.get('objectives', []),
                'selected_modules': st.session_state.selected_modules,
                'ai_config': {
                    'enabled': project_data.get('use_ai', True),
                    'engines': project_data.get('ai_engines', []),
                    'mode': project_data.get('ai_mode', '자동'),
                    'explanation_detail': project_data.get('explanation_detail', '보통'),
                    'auto_optimization': project_data.get('auto_optimization', False),
                    'features': project_data.get('ai_features', [])
                },
                'collaboration': {
                    'mode': project_data.get('collab_mode', 'personal'),
                    'require_approval': project_data.get('require_approval', True),
                    'allow_guest_view': project_data.get('allow_guest_view', False)
                },
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            # 협업자 추가 (소유자)
            project['collaborators'] = [{
                'user_id': self.user_id,
                'email': self.user.get('email', ''),
                'name': self.user.get('name', ''),
                'role': 'owner',
                'joined_at': datetime.now().isoformat()
            }]
            
            # 데이터베이스에 저장
            success = self.db_manager.create_project(project)
            
            if success:
                # 템플릿으로 저장
                if template_name:
                    self._save_as_template(project, template_name)
                    
                return True, project_id
            else:
                st.error("프로젝트 생성 중 오류가 발생했습니다.")
                return False, None
                
        except Exception as e:
            logger.error(f"프로젝트 생성 오류: {e}")
            st.error(f"프로젝트 생성 실패: {str(e)}")
            return False, None
            
    def _send_invitations(self, project_id: str):
        """초대 발송"""
        try:
            invitations = st.session_state.new_project.get('invitations', {})
            emails = invitations.get('emails', [])
            default_role = invitations.get('default_role', 'viewer')
            
            for email in emails:
                # 초대 데이터
                invitation = {
                    'project_id': project_id,
                    'invited_by': self.user_id,
                    'email': email,
                    'role': default_role,
                    'status': 'pending',
                    'created_at': datetime.now().isoformat()
                }
                
                # 데이터베이스에 저장
                self.db_manager.create_invitation(invitation)
                
                # 알림 발송
                self.notification_manager.send_notification(
                    user_email=email,
                    title="프로젝트 초대",
                    message=f"{self.user.get('name', '사용자')}님이 '{st.session_state.new_project['name']}' 프로젝트에 초대했습니다.",
                    type='collaboration'
                )
                
            st.success(f"✉️ {len(emails)}명에게 초대를 발송했습니다.")
            
        except Exception as e:
            logger.error(f"초대 발송 오류: {e}")
            st.error("초대 발송 중 오류가 발생했습니다.")
            
    def _save_draft_project(self):
        """임시 저장"""
        try:
            draft_data = {
                'user_id': self.user_id,
                'project_data': st.session_state.new_project,
                'selected_modules': st.session_state.selected_modules,
                'current_step': st.session_state.project_step,
                'saved_at': datetime.now().isoformat()
            }
            
            # 로컬 스토리지에 저장
            draft_id = f"draft_{self.user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.db_manager.save_draft(draft_id, draft_data)
            
            st.success("💾 임시 저장되었습니다!")
            
        except Exception as e:
            logger.error(f"임시 저장 오류: {e}")
            st.error("임시 저장 중 오류가 발생했습니다.")
            
    def _show_project_menu(self, project: Dict):
        """프로젝트 메뉴"""
        with st.popover("프로젝트 옵션"):
            if st.button("📋 복제", use_container_width=True):
                self._clone_project(project)
                
            if st.button("📤 내보내기", use_container_width=True):
                self._export_project(project)
                
            if st.button("🗑️ 삭제", use_container_width=True,
                       disabled=not self._can_delete_project(project)):
                if st.checkbox("정말 삭제하시겠습니까?"):
                    self._delete_project(project)
                    
    def _can_delete_project(self, project: Dict) -> bool:
        """프로젝트 삭제 권한 확인"""
        if self.is_guest:
            return False
            
        # 소유자만 삭제 가능
        return project.get('owner_id') == self.user_id
        
    def _get_templates(self, category: str) -> List[Dict]:
        """템플릿 목록 가져오기"""
        try:
            if category == '전체':
                return self.db_manager.get_all_templates()
            elif category == '내 템플릿':
                return self.db_manager.get_user_templates(self.user_id)
            elif category == '공유 템플릿':
                return self.db_manager.get_public_templates()
            elif category == '공식 템플릿':
                return self.db_manager.get_official_templates()
            else:
                return []
        except Exception as e:
            logger.error(f"템플릿 가져오기 오류: {e}")
            return []
            
    def _show_template_preview(self, template: Dict):
        """템플릿 미리보기"""
        with st.expander(f"📋 {template['name']} 미리보기", expanded=True):
            st.write(f"**설명**: {template['description']}")
            st.write(f"**프로젝트 유형**: {template.get('project_type', 'N/A')}")
            st.write(f"**포함된 모듈**: {len(template.get('modules', []))}개")
            
            if template.get('ai_config'):
                st.write("**AI 설정**:")
                st.json(template['ai_config'])
                
    def _use_template(self, template: Dict):
        """템플릿 사용"""
        # 템플릿 데이터로 새 프로젝트 초기화
        st.session_state.new_project = template.get('project_data', {}).copy()
        st.session_state.selected_modules = template.get('modules', []).copy()
        st.session_state.project_step = 0
        
        st.success(f"✅ '{template['name']}' 템플릿을 불러왔습니다.")
        st.rerun()
        
    def _save_as_template(self, project: Dict, template_name: str):
        """템플릿으로 저장"""
        try:
            template = {
                'id': str(uuid.uuid4()),
                'name': template_name,
                'description': f"{project['name']} 기반 템플릿",
                'project_type': project['type'],
                'project_data': {
                    'type': project['type'],
                    'subcategory': project.get('subcategory'),
                    'visibility': project.get('visibility'),
                    'objectives': project.get('objectives', []),
                    'ai_config': project.get('ai_config', {}),
                    'collaboration': project.get('collaboration', {})
                },
                'modules': project.get('selected_modules', []),
                'created_by': self.user_id,
                'is_public': False,
                'created_at': datetime.now().isoformat()
            }
            
            self.db_manager.save_template(template)
            st.success(f"✅ 템플릿 '{template_name}'이 저장되었습니다.")
            
        except Exception as e:
            logger.error(f"템플릿 저장 오류: {e}")
            st.error("템플릿 저장 중 오류가 발생했습니다.")
            
    def _show_export_dialog(self):
        """내보내기 다이얼로그"""
        with st.expander("📤 프로젝트 내보내기", expanded=True):
            export_format = st.radio(
                "내보내기 형식",
                options=['JSON', 'Excel', 'PDF'],
                horizontal=True
            )
            
            include_data = st.checkbox("실험 데이터 포함", value=True)
            include_analysis = st.checkbox("분석 결과 포함", value=True)
            
            if st.button("내보내기", type="primary"):
                self._export_projects(export_format, include_data, include_analysis)
                
    def _export_projects(self, format: str, include_data: bool, include_analysis: bool):
        """프로젝트 내보내기"""
        try:
            # 선택된 프로젝트들 가져오기
            projects = self._get_filtered_projects(
                st.session_state.project_filter['search'],
                st.session_state.project_filter['status'],
                '최근 수정'
            )
            
            if format == 'JSON':
                # JSON 내보내기
                export_data = {
                    'export_date': datetime.now().isoformat(),
                    'user': self.user.get('email', 'unknown'),
                    'projects': projects
                }
                
                if include_data:
                    # 실험 데이터 추가
                    for project in export_data['projects']:
                        project['experiments'] = self.db_manager.get_project_experiments(project['id'])
                        
                if include_analysis:
                    # 분석 결과 추가
                    for project in export_data['projects']:
                        project['analysis'] = self.db_manager.get_project_analysis(project['id'])
                        
                # 다운로드 버튼
                json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
                st.download_button(
                    "💾 JSON 다운로드",
                    json_str,
                    f"projects_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )
                
            elif format == 'Excel':
                # Excel 내보내기
                output = io.BytesIO()
                
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # 프로젝트 목록
                    df_projects = pd.DataFrame(projects)
                    df_projects.to_excel(writer, sheet_name='Projects', index=False)
                    
                    if include_data:
                        # 각 프로젝트의 실험 데이터
                        for project in projects[:10]:  # 최대 10개 프로젝트
                            experiments = self.db_manager.get_project_experiments(project['id'])
                            if experiments:
                                df_exp = pd.DataFrame(experiments)
                                sheet_name = f"Exp_{project['name'][:20]}"
                                df_exp.to_excel(writer, sheet_name=sheet_name, index=False)
                                
                output.seek(0)
                
                st.download_button(
                    "📊 Excel 다운로드",
                    output,
                    f"projects_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
            elif format == 'PDF':
                st.info("PDF 내보내기는 준비 중입니다.")
                
        except Exception as e:
            logger.error(f"내보내기 오류: {e}")
            st.error(f"내보내기 중 오류가 발생했습니다: {str(e)}")

# ===========================================================================
# 🚀 메인 실행
# ===========================================================================

def main():
    """메인 실행 함수"""
    try:
        page = ProjectSetupPage()
        page.render()
    except Exception as e:
        logger.error(f"프로젝트 설정 페이지 오류: {e}")
        st.error(f"페이지 로드 중 오류가 발생했습니다: {str(e)}")
        
        if st.button("🔄 새로고침"):
            st.rerun()

if __name__ == "__main__":
    main()
