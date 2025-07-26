"""
2_📝_Project_Setup.py - 프로젝트 설정 및 관리
Universal DOE Platform의 프로젝트 생성, 관리, 모듈 선택 페이지
"""
import streamlit as st
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

# 필수 모듈 임포트
try:
    from utils.database_manager import get_database_manager
    from utils.auth_manager import get_auth_manager
    from utils.common_ui import get_common_ui
    from utils.api_manager import get_api_manager
    from modules.module_registry import get_module_registry
    from utils.notification_manager import get_notification_manager
    from config.app_config import EXPERIMENT_DEFAULTS
except ImportError as e:
    st.error(f"필수 모듈 임포트 오류: {e}")
    st.stop()

# 페이지 설정
st.set_page_config(
    page_title="프로젝트 설정 - Universal DOE",
    page_icon="📝",
    layout="wide"
)

# 인증 확인
auth_manager = get_auth_manager()
if not auth_manager.check_authentication():
    st.warning("로그인이 필요합니다")
    st.switch_page("pages/0_🔐_Login.py")
    st.stop()

# 연구 분야 계층 구조
RESEARCH_FIELDS = {
    "화학": {
        "유기화학": ["합성", "반응 메커니즘", "촉매", "천연물"],
        "무기화학": ["배위화학", "고체화학", "나노소재", "촉매"],
        "분석화학": ["크로마토그래피", "분광학", "질량분석", "전기화학"],
        "물리화학": ["열역학", "반응속도론", "표면화학", "계산화학"]
    },
    "재료과학": {
        "고분자": ["합성", "물성", "가공", "복합재료"],
        "세라믹": ["구조세라믹", "기능세라믹", "바이오세라믹", "나노세라믹"],
        "금속": ["합금설계", "열처리", "부식", "표면처리"],
        "전자재료": ["반도체", "디스플레이", "배터리", "태양전지"]
    },
    "생명공학": {
        "분자생물학": ["유전자조작", "단백질공학", "세포배양", "오믹스"],
        "의약품": ["신약개발", "제형", "약물전달", "바이오시밀러"],
        "식품공학": ["발효", "가공", "기능성식품", "품질관리"],
        "환경생물": ["생물정화", "바이오에너지", "미생물", "생태계"]
    },
    "기타": {
        "융합연구": ["바이오소재", "나노바이오", "에너지", "환경"],
        "공정개발": ["반응기설계", "분리정제", "스케일업", "최적화"],
        "품질관리": ["분석법개발", "안정성", "표준화", "인증"],
        "커스텀": ["사용자정의"]
    }
}

class ProjectSetupPage:
    """프로젝트 설정 페이지 클래스"""
    
    def __init__(self):
        self.ui = get_common_ui()
        self.db_manager = get_database_manager()
        self.module_registry = get_module_registry()
        self.api_manager = get_api_manager()
        self.notification_manager = get_notification_manager()
        
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
            'editing_project': None
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def render(self):
        """메인 렌더링"""
        self.ui.render_header("📝 프로젝트 설정", "연구 프로젝트를 생성하고 관리합니다")
        
        # 탭 구성
        tabs = st.tabs([
            "📋 프로젝트 목록",
            "➕ 새 프로젝트",
            "🔧 프로젝트 편집",
            "📚 템플릿 관리"
        ])
        
        with tabs[0]:
            self._render_project_list()
        
        with tabs[1]:
            self._render_new_project_wizard()
        
        with tabs[2]:
            self._render_project_editor()
        
        with tabs[3]:
            self._render_template_manager()
    
    def _render_project_list(self):
        """프로젝트 목록 렌더링"""
        st.subheader("내 프로젝트")
        
        # 필터 및 뷰 옵션
        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
        
        with col1:
            search_query = st.text_input("🔍 검색", placeholder="프로젝트명, 태그...")
        
        with col2:
            status_filter = st.multiselect(
                "상태 필터",
                ["활성", "완료", "보관", "공유됨"],
                default=["활성"]
            )
        
        with col3:
            sort_by = st.selectbox(
                "정렬",
                ["최근 수정", "이름", "생성일", "진행률"]
            )
        
        with col4:
            view_mode = st.radio(
                "보기",
                ["그리드", "리스트"],
                horizontal=True,
                key="project_view_toggle"
            )
            st.session_state.project_view = view_mode.lower()
        
        # 프로젝트 데이터 조회
        projects = self._get_user_projects(search_query, status_filter, sort_by)
        
        if not projects:
            self.ui.render_empty_state(
                "아직 프로젝트가 없습니다",
                "🚀"
            )
            if st.button("첫 프로젝트 만들기", type="primary"):
                st.session_state.project_step = 0
                st.rerun()
        else:
            if st.session_state.project_view == "grid":
                self._render_projects_grid(projects)
            else:
                self._render_projects_list(projects)
    
    def _render_projects_grid(self, projects: List[Dict]):
        """프로젝트 그리드 뷰"""
        cols = st.columns(3)
        
        for idx, project in enumerate(projects):
            with cols[idx % 3]:
                with st.container():
                    # 프로젝트 카드
                    st.markdown(f"""
                    <div class="custom-card">
                        <h4>{project['name']}</h4>
                        <p><small>{project['field']} > {project['subfield']}</small></p>
                        <div style="margin: 1rem 0;">
                            <p>{project.get('description', '설명 없음')}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 진행률 표시
                    progress = project.get('progress', 0) / 100
                    st.progress(progress, text=f"진행률: {project.get('progress', 0)}%")
                    
                    # 메타 정보
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption(f"🧪 실험: {project.get('experiment_count', 0)}")
                    with col2:
                        st.caption(f"📅 {project['updated_at'][:10]}")
                    
                    # 액션 버튼
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("열기", key=f"open_{project['id']}"):
                            st.session_state.current_project = project
                            st.switch_page("pages/3_🧪_Experiment_Design.py")
                    with col2:
                        if st.button("편집", key=f"edit_{project['id']}"):
                            st.session_state.editing_project = project
                            st.rerun()
                    with col3:
                        if st.button("공유", key=f"share_{project['id']}"):
                            self._show_share_dialog(project)
    
    def _render_projects_list(self, projects: List[Dict]):
        """프로젝트 리스트 뷰"""
        # 테이블 데이터 준비
        df = pd.DataFrame(projects)
        df['작업'] = '선택'
        
        # 테이블 표시
        edited_df = st.data_editor(
            df[['name', 'field', 'subfield', 'progress', 'updated_at', '작업']],
            column_config={
                "name": st.column_config.TextColumn("프로젝트명", width="large"),
                "field": st.column_config.TextColumn("분야", width="medium"),
                "subfield": st.column_config.TextColumn("세부분야", width="medium"),
                "progress": st.column_config.ProgressColumn("진행률", width="small"),
                "updated_at": st.column_config.DateColumn("수정일", width="small"),
                "작업": st.column_config.SelectboxColumn(
                    "작업",
                    options=["선택", "열기", "편집", "공유", "삭제"],
                    width="small"
                )
            },
            hide_index=True,
            use_container_width=True
        )
        
        # 선택된 작업 처리
        for idx, row in edited_df.iterrows():
            if row['작업'] != '선택':
                self._handle_project_action(
                    projects[idx], 
                    row['작업']
                )
    
    def _render_new_project_wizard(self):
        """새 프로젝트 생성 마법사"""
        st.subheader("새 프로젝트 만들기")
        
        # 진행 표시
        steps = ["기본 정보", "연구 분야", "실험 모듈", "협업 설정", "확인"]
        progress = st.session_state.project_step / (len(steps) - 1)
        st.progress(progress)
        st.write(f"단계 {st.session_state.project_step + 1}/{len(steps)}: {steps[st.session_state.project_step]}")
        
        # 단계별 렌더링
        if st.session_state.project_step == 0:
            self._render_basic_info_step()
        elif st.session_state.project_step == 1:
            self._render_field_selection_step()
        elif st.session_state.project_step == 2:
            self._render_module_selection_step()
        elif st.session_state.project_step == 3:
            self._render_collaboration_step()
        elif st.session_state.project_step == 4:
            self._render_confirmation_step()
        
        # 네비게이션 버튼
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.session_state.project_step > 0:
                if st.button("⬅️ 이전", use_container_width=True):
                    st.session_state.project_step -= 1
                    st.rerun()
        
        with col2:
            if st.button("❌ 취소", use_container_width=True):
                if st.confirm("프로젝트 생성을 취소하시겠습니까?"):
                    st.session_state.project_step = 0
                    st.session_state.new_project = {}
                    st.rerun()
        
        with col3:
            if st.session_state.project_step < len(steps) - 1:
                if st.button("다음 ➡️", use_container_width=True, type="primary"):
                    if self._validate_current_step():
                        st.session_state.project_step += 1
                        st.rerun()
            else:
                if st.button("✅ 생성", use_container_width=True, type="primary"):
                    self._create_project()
    
    def _render_basic_info_step(self):
        """기본 정보 입력 단계"""
        st.markdown("### 1️⃣ 기본 정보")
        
        # 프로젝트명
        project_name = st.text_input(
            "프로젝트명 *",
            value=st.session_state.new_project.get('name', ''),
            placeholder="예: 신규 촉매 개발",
            help="명확하고 구체적인 이름을 사용하세요"
        )
        st.session_state.new_project['name'] = project_name
        
        # 설명
        description = st.text_area(
            "프로젝트 설명",
            value=st.session_state.new_project.get('description', ''),
            height=100,
            placeholder="프로젝트의 목적과 주요 내용을 간단히 설명하세요"
        )
        st.session_state.new_project['description'] = description
        
        # 프로젝트 유형
        col1, col2 = st.columns(2)
        
        with col1:
            project_type = st.selectbox(
                "프로젝트 유형",
                ["연구개발", "품질관리", "공정개선", "분석법개발", "기타"],
                index=["연구개발", "품질관리", "공정개선", "분석법개발", "기타"].index(
                    st.session_state.new_project.get('type', '연구개발')
                )
            )
            st.session_state.new_project['type'] = project_type
        
        with col2:
            priority = st.select_slider(
                "우선순위",
                options=["낮음", "보통", "높음", "긴급"],
                value=st.session_state.new_project.get('priority', '보통')
            )
            st.session_state.new_project['priority'] = priority
        
        # 일정
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input(
                "시작일",
                value=st.session_state.new_project.get('start_date', datetime.now().date())
            )
            st.session_state.new_project['start_date'] = start_date.isoformat()
        
        with col2:
            end_date = st.date_input(
                "목표 종료일",
                value=st.session_state.new_project.get('end_date', None)
            )
            if end_date:
                st.session_state.new_project['end_date'] = end_date.isoformat()
    
    def _render_field_selection_step(self):
        """연구 분야 선택 단계"""
        st.markdown("### 2️⃣ 연구 분야 선택")
        
        # 대분야 선택
        main_field = st.selectbox(
            "대분야 *",
            list(RESEARCH_FIELDS.keys()),
            index=list(RESEARCH_FIELDS.keys()).index(
                st.session_state.new_project.get('field', '화학')
            )
        )
        st.session_state.new_project['field'] = main_field
        
        # 중분야 선택
        if main_field:
            sub_fields = list(RESEARCH_FIELDS[main_field].keys())
            sub_field = st.selectbox(
                "중분야 *",
                sub_fields,
                index=sub_fields.index(
                    st.session_state.new_project.get('subfield', sub_fields[0])
                ) if st.session_state.new_project.get('subfield') in sub_fields else 0
            )
            st.session_state.new_project['subfield'] = sub_field
            
            # 세부분야 선택
            if sub_field:
                detail_fields = RESEARCH_FIELDS[main_field][sub_field]
                detail_field = st.multiselect(
                    "세부분야 (복수 선택 가능)",
                    detail_fields,
                    default=st.session_state.new_project.get('detail_fields', [])
                )
                st.session_state.new_project['detail_fields'] = detail_field
        
        # 키워드
        keywords = st.text_input(
            "키워드 (쉼표로 구분)",
            value=', '.join(st.session_state.new_project.get('keywords', [])),
            placeholder="예: 촉매, 반응속도, 선택성"
        )
        st.session_state.new_project['keywords'] = [k.strip() for k in keywords.split(',') if k.strip()]
        
        # AI 분야 추천
        if st.button("🤖 AI 추천 받기"):
            self._get_field_recommendations()
    
    def _render_module_selection_step(self):
        """실험 모듈 선택 단계"""
        st.markdown("### 3️⃣ 실험 모듈 선택")
        
        # 모듈 추천
        field = st.session_state.new_project.get('field', '')
        subfield = st.session_state.new_project.get('subfield', '')
        
        if field and subfield:
            # AI 추천 모듈
            st.markdown("#### 🤖 AI 추천 모듈")
            
            if st.button("AI 모듈 추천 받기"):
                self._get_module_recommendations()
            
            # AI 추천 결과 표시
            if st.session_state.ai_recommendations:
                self._render_ai_recommendations()
        
        # 모듈 카탈로그
        st.markdown("#### 📚 모듈 카탈로그")
        
        # 카테고리 필터
        categories = self.module_registry.get_categories()
        selected_category = st.selectbox(
            "카테고리",
            ["전체"] + categories,
            help="모듈 카테고리를 선택하세요"
        )
        
        # 모듈 목록
        if selected_category == "전체":
            modules = self.module_registry.list_modules()
        else:
            modules = self.module_registry.list_modules(category=selected_category)
        
        # 모듈 선택 UI
        selected_modules = []
        for module in modules:
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                st.write(f"**{module['name']}**")
                st.caption(module['description'])
            
            with col2:
                tags = module.get('tags', [])
                if tags:
                    st.write(' '.join([f"`{tag}`" for tag in tags[:3]]))
            
            with col3:
                if st.checkbox("선택", key=f"module_{module['id']}"):
                    selected_modules.append(module['id'])
        
        st.session_state.selected_modules = selected_modules
        
        # 선택된 모듈 요약
        if selected_modules:
            st.success(f"✅ {len(selected_modules)}개 모듈 선택됨")
    
    def _render_collaboration_step(self):
        """협업 설정 단계"""
        st.markdown("### 4️⃣ 협업 설정")
        
        # 공개 범위
        visibility = st.radio(
            "프로젝트 공개 범위",
            ["비공개", "팀 공개", "전체 공개"],
            index=["비공개", "팀 공개", "전체 공개"].index(
                st.session_state.new_project.get('visibility', '비공개')
            ),
            help="프로젝트의 공개 범위를 설정합니다"
        )
        st.session_state.new_project['visibility'] = visibility
        
        # 협업자 초대
        st.markdown("#### 협업자 초대")
        
        # 이메일 입력
        invited_emails = st.text_area(
            "이메일 주소 (한 줄에 하나씩)",
            value='\n'.join(st.session_state.new_project.get('collaborators', [])),
            height=100,
            placeholder="user1@example.com\nuser2@example.com"
        )
        
        # 권한 설정
        if invited_emails:
            emails = [e.strip() for e in invited_emails.split('\n') if e.strip()]
            st.session_state.new_project['collaborators'] = emails
            
            default_permission = st.selectbox(
                "기본 권한",
                ["보기", "편집", "관리"],
                help="초대된 사용자의 기본 권한"
            )
            st.session_state.new_project['default_permission'] = default_permission
            
            # 초대 메시지
            invite_message = st.text_area(
                "초대 메시지 (선택사항)",
                value=st.session_state.new_project.get('invite_message', ''),
                placeholder="프로젝트에 대한 간단한 소개나 협업 요청 사항을 작성하세요"
            )
            st.session_state.new_project['invite_message'] = invite_message
    
    def _render_confirmation_step(self):
        """확인 단계"""
        st.markdown("### 5️⃣ 프로젝트 생성 확인")
        
        project = st.session_state.new_project
        
        # 프로젝트 요약
        st.markdown("#### 📋 프로젝트 요약")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**기본 정보**")
            st.write(f"- 프로젝트명: {project.get('name', '')}")
            st.write(f"- 유형: {project.get('type', '')}")
            st.write(f"- 우선순위: {project.get('priority', '')}")
            st.write(f"- 기간: {project.get('start_date', '')} ~ {project.get('end_date', '미정')}")
        
        with col2:
            st.write("**연구 분야**")
            st.write(f"- 대분야: {project.get('field', '')}")
            st.write(f"- 중분야: {project.get('subfield', '')}")
            if project.get('detail_fields'):
                st.write(f"- 세부분야: {', '.join(project.get('detail_fields', []))}")
            if project.get('keywords'):
                st.write(f"- 키워드: {', '.join(project.get('keywords', []))}")
        
        # 선택된 모듈
        if st.session_state.selected_modules:
            st.write("**선택된 실험 모듈**")
            for module_id in st.session_state.selected_modules:
                module = self.module_registry.get_module(module_id)
                if module:
                    st.write(f"- {module.get_module_info()['name']}")
        
        # 협업 설정
        if project.get('collaborators'):
            st.write("**협업자**")
            st.write(f"- {len(project['collaborators'])}명 초대 예정")
            st.write(f"- 기본 권한: {project.get('default_permission', '보기')}")
        
        # 템플릿 저장 옵션
        st.divider()
        save_as_template = st.checkbox(
            "이 설정을 템플릿으로 저장",
            help="나중에 비슷한 프로젝트를 만들 때 사용할 수 있습니다"
        )
        
        if save_as_template:
            template_name = st.text_input(
                "템플릿 이름",
                placeholder="예: 촉매 개발 프로젝트"
            )
            st.session_state.new_project['save_as_template'] = True
            st.session_state.new_project['template_name'] = template_name
    
    def _render_project_editor(self):
        """프로젝트 편집기"""
        if not st.session_state.editing_project:
            st.info("편집할 프로젝트를 선택하세요")
            return
        
        project = st.session_state.editing_project
        st.subheader(f"프로젝트 편집: {project['name']}")
        
        # 편집 폼
        with st.form("project_edit_form"):
            # 기본 정보
            st.markdown("#### 기본 정보")
            
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("프로젝트명", value=project['name'])
                project_type = st.selectbox(
                    "유형",
                    ["연구개발", "품질관리", "공정개선", "분석법개발", "기타"],
                    index=["연구개발", "품질관리", "공정개선", "분석법개발", "기타"].index(project.get('type', '연구개발'))
                )
            
            with col2:
                status = st.selectbox(
                    "상태",
                    ["활성", "일시중지", "완료", "보관"],
                    index=["활성", "일시중지", "완료", "보관"].index(project.get('status', '활성'))
                )
                priority = st.select_slider(
                    "우선순위",
                    options=["낮음", "보통", "높음", "긴급"],
                    value=project.get('priority', '보통')
                )
            
            description = st.text_area(
                "설명",
                value=project.get('description', ''),
                height=100
            )
            
            # 실험 모듈
            st.markdown("#### 실험 모듈")
            current_modules = project.get('modules', [])
            
            # 현재 모듈 표시
            if current_modules:
                st.write("현재 모듈:")
                for module_id in current_modules:
                    module = self.module_registry.get_module(module_id)
                    if module:
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.write(f"- {module.get_module_info()['name']}")
                        with col2:
                            if st.button("제거", key=f"remove_{module_id}"):
                                current_modules.remove(module_id)
            
            # 모듈 추가
            if st.checkbox("모듈 추가/변경"):
                modules = self.module_registry.list_modules()
                module_options = {m['name']: m['id'] for m in modules}
                
                selected_new = st.multiselect(
                    "추가할 모듈",
                    list(module_options.keys())
                )
                
                for module_name in selected_new:
                    module_id = module_options[module_name]
                    if module_id not in current_modules:
                        current_modules.append(module_id)
            
            # 저장 버튼
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.form_submit_button("💾 저장", type="primary", use_container_width=True):
                    # 업데이트 데이터 준비
                    updated_data = {
                        'name': name,
                        'type': project_type,
                        'status': status,
                        'priority': priority,
                        'description': description,
                        'modules': current_modules,
                        'updated_at': datetime.now().isoformat()
                    }
                    
                    # 데이터베이스 업데이트
                    if self.db_manager.update_project(project['id'], updated_data):
                        st.success("✅ 프로젝트가 업데이트되었습니다")
                        st.session_state.editing_project = None
                        st.rerun()
                    else:
                        st.error("프로젝트 업데이트 실패")
            
            with col3:
                if st.form_submit_button("취소", use_container_width=True):
                    st.session_state.editing_project = None
                    st.rerun()
    
    def _render_template_manager(self):
        """템플릿 관리"""
        st.subheader("프로젝트 템플릿")
        
        # 템플릿 목록
        templates = self.db_manager.get_project_templates(st.session_state.user['id'])
        
        if not templates:
            st.info("저장된 템플릿이 없습니다. 새 프로젝트를 만들 때 템플릿으로 저장할 수 있습니다.")
        else:
            for template in templates:
                with st.expander(template['name']):
                    # 템플릿 정보
                    st.write(f"**설명**: {template.get('description', '없음')}")
                    st.write(f"**분야**: {template['field']} > {template['subfield']}")
                    st.write(f"**생성일**: {template['created_at'][:10]}")
                    
                    # 액션 버튼
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("사용", key=f"use_template_{template['id']}"):
                            self._load_template(template)
                    
                    with col2:
                        if st.button("수정", key=f"edit_template_{template['id']}"):
                            st.session_state.editing_template = template
                    
                    with col3:
                        if st.button("삭제", key=f"delete_template_{template['id']}"):
                            if st.confirm("템플릿을 삭제하시겠습니까?"):
                                self.db_manager.delete_template(template['id'])
                                st.rerun()
    
    def _render_ai_recommendations(self):
        """AI 추천 결과 렌더링"""
        recommendations = st.session_state.ai_recommendations
        
        # AI 설명 상세도 제어
        col1, col2 = st.columns([4, 1])
        with col2:
            show_details = st.checkbox(
                "🔍 상세 설명",
                value=st.session_state.show_ai_details,
                key="ai_details_toggle"
            )
            st.session_state.show_ai_details = show_details
        
        # 추천 모듈 표시
        for idx, rec in enumerate(recommendations):
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"**{rec['module_name']}**")
                    st.write(rec['reason'])
                
                with col2:
                    st.write(f"적합도: {rec['score']}%")
                
                with col3:
                    if st.button("선택", key=f"select_ai_{idx}"):
                        if rec['module_id'] not in st.session_state.selected_modules:
                            st.session_state.selected_modules.append(rec['module_id'])
                
                # 상세 설명 (토글)
                if show_details:
                    with st.expander("상세 설명"):
                        st.write("**추론 과정**")
                        st.write(rec.get('reasoning', ''))
                        
                        st.write("**대안**")
                        for alt in rec.get('alternatives', []):
                            st.write(f"- {alt}")
                        
                        st.write("**주의사항**")
                        st.write(rec.get('limitations', ''))
    
    def _get_module_recommendations(self):
        """AI 모듈 추천"""
        project = st.session_state.new_project
        
        with st.spinner("AI가 최적의 모듈을 추천하고 있습니다..."):
            prompt = f"""
            다음 프로젝트에 적합한 실험 모듈을 추천해주세요:
            
            분야: {project.get('field')} > {project.get('subfield')}
            세부분야: {', '.join(project.get('detail_fields', []))}
            키워드: {', '.join(project.get('keywords', []))}
            프로젝트 유형: {project.get('type')}
            설명: {project.get('description', '')}
            
            사용 가능한 모듈:
            {self._get_available_modules_list()}
            
            응답 형식:
            1. 추천 모듈 3-5개
            2. 각 모듈별 추천 이유
            3. 적합도 점수 (0-100)
            4. 추론 과정 (상세)
            5. 대안 모듈
            6. 주의사항
            """
            
            response = self.api_manager.call_ai(
                prompt,
                response_format="structured",
                detail_level='detailed' if st.session_state.show_ai_details else 'auto'
            )
            
            if response:
                st.session_state.ai_recommendations = response['recommendations']
                st.success("✅ AI 추천이 완료되었습니다")
            else:
                st.error("AI 추천을 받을 수 없습니다")
    
    def _validate_current_step(self) -> bool:
        """현재 단계 검증"""
        step = st.session_state.project_step
        project = st.session_state.new_project
        
        if step == 0:  # 기본 정보
            if not project.get('name'):
                st.error("프로젝트명은 필수입니다")
                return False
            if len(project['name']) < 3:
                st.error("프로젝트명은 3자 이상이어야 합니다")
                return False
        
        elif step == 1:  # 연구 분야
            if not project.get('field') or not project.get('subfield'):
                st.error("연구 분야를 선택해주세요")
                return False
        
        elif step == 2:  # 실험 모듈
            if not st.session_state.selected_modules:
                st.warning("실험 모듈을 하나 이상 선택해주세요")
                return False
        
        return True
    
    def _create_project(self):
        """프로젝트 생성"""
        project = st.session_state.new_project
        
        # 프로젝트 데이터 준비
        project_data = {
            'user_id': st.session_state.user['id'],
            'name': project['name'],
            'description': project.get('description', ''),
            'type': project.get('type', '연구개발'),
            'field': project['field'],
            'subfield': project['subfield'],
            'detail_fields': project.get('detail_fields', []),
            'keywords': project.get('keywords', []),
            'modules': st.session_state.selected_modules,
            'priority': project.get('priority', '보통'),
            'visibility': project.get('visibility', '비공개'),
            'start_date': project.get('start_date'),
            'end_date': project.get('end_date'),
            'status': '활성',
            'progress': 0,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        # 데이터베이스에 저장
        with st.spinner("프로젝트를 생성하고 있습니다..."):
            project_id = self.db_manager.create_project(project_data)
            
            if project_id:
                # 협업자 초대
                if project.get('collaborators'):
                    for email in project['collaborators']:
                        self._invite_collaborator(
                            project_id, 
                            email, 
                            project.get('default_permission', '보기'),
                            project.get('invite_message', '')
                        )
                
                # 템플릿 저장
                if project.get('save_as_template'):
                    self._save_as_template(project_data, project.get('template_name'))
                
                st.success("✅ 프로젝트가 성공적으로 생성되었습니다!")
                st.balloons()
                
                # 초기화
                st.session_state.project_step = 0
                st.session_state.new_project = {}
                st.session_state.selected_modules = []
                
                # 프로젝트 페이지로 이동
                if st.button("프로젝트로 이동", type="primary"):
                    st.session_state.current_project = {'id': project_id}
                    st.switch_page("pages/3_🧪_Experiment_Design.py")
            else:
                st.error("프로젝트 생성에 실패했습니다")
    
    def _get_user_projects(self, search_query: str, 
                          status_filter: List[str], 
                          sort_by: str) -> List[Dict]:
        """사용자 프로젝트 조회"""
        # 데이터베이스에서 프로젝트 조회
        projects = self.db_manager.get_user_projects(
            user_id=st.session_state.user['id'],
            search=search_query,
            status=status_filter,
            sort_by=sort_by
        )
        
        return projects
    
    def _handle_project_action(self, project: Dict, action: str):
        """프로젝트 액션 처리"""
        if action == "열기":
            st.session_state.current_project = project
            st.switch_page("pages/3_🧪_Experiment_Design.py")
        elif action == "편집":
            st.session_state.editing_project = project
            st.rerun()
        elif action == "공유":
            self._show_share_dialog(project)
        elif action == "삭제":
            if st.confirm(f"'{project['name']}' 프로젝트를 삭제하시겠습니까?"):
                self.db_manager.delete_project(project['id'])
                st.rerun()
    
    def _show_share_dialog(self, project: Dict):
        """공유 대화상자"""
        with st.dialog("프로젝트 공유"):
            st.write(f"**{project['name']}** 프로젝트 공유")
            
            # 공유 링크 생성
            share_link = f"https://universaldoe.com/project/{project['id']}"
            st.code(share_link)
            
            # 이메일로 초대
            emails = st.text_area(
                "이메일 주소 (한 줄에 하나씩)",
                placeholder="user@example.com"
            )
            
            permission = st.selectbox(
                "권한",
                ["보기", "편집", "관리"]
            )
            
            if st.button("초대 보내기", type="primary"):
                # 초대 처리
                st.success("초대가 발송되었습니다")
    
    def _invite_collaborator(self, project_id: str, email: str, 
                           permission: str, message: str):
        """협업자 초대"""
        # 데이터베이스에 초대 기록
        self.db_manager.add_collaborator(
            project_id=project_id,
            email=email,
            permission=permission
        )
        
        # 알림 발송
        self.notification_manager.send_notification(
            to_email=email,
            type='project_invitation',
            data={
                'project_id': project_id,
                'inviter': st.session_state.user['name'],
                'message': message
            }
        )
    
    def _load_template(self, template: Dict):
        """템플릿 로드"""
        # 새 프로젝트 설정에 템플릿 적용
        st.session_state.new_project = {
            'name': '',  # 이름은 비워둠
            'type': template.get('type'),
            'field': template.get('field'),
            'subfield': template.get('subfield'),
            'detail_fields': template.get('detail_fields', []),
            'keywords': template.get('keywords', []),
            'description': template.get('description', '')
        }
        
        st.session_state.selected_modules = template.get('modules', [])
        st.session_state.project_step = 0
        
        st.success(f"템플릿 '{template['name']}'이 적용되었습니다")
        st.rerun()
    
    def _save_as_template(self, project_data: Dict, template_name: str):
        """프로젝트를 템플릿으로 저장"""
        template_data = {
            'user_id': st.session_state.user['id'],
            'name': template_name or f"{project_data['name']} 템플릿",
            'type': project_data['type'],
            'field': project_data['field'],
            'subfield': project_data['subfield'],
            'detail_fields': project_data.get('detail_fields', []),
            'keywords': project_data.get('keywords', []),
            'modules': project_data.get('modules', []),
            'description': project_data.get('description', ''),
            'created_at': datetime.now().isoformat()
        }
        
        self.db_manager.save_project_template(template_data)
    
    def _get_available_modules_list(self) -> str:
        """사용 가능한 모듈 목록 문자열"""
        modules = self.module_registry.list_modules()
        module_list = []
        
        for module in modules:
            module_list.append(f"- {module['name']}: {module['description']}")
        
        return '\n'.join(module_list)
    
    def _get_field_recommendations(self):
        """AI 연구 분야 추천"""
        project = st.session_state.new_project
        
        with st.spinner("AI가 연구 분야를 분석하고 있습니다..."):
            prompt = f"""
            다음 프로젝트 정보를 바탕으로 가장 적합한 연구 분야를 추천해주세요:
            
            프로젝트명: {project.get('name', '')}
            설명: {project.get('description', '')}
            유형: {project.get('type', '')}
            
            사용 가능한 분야:
            {json.dumps(RESEARCH_FIELDS, ensure_ascii=False, indent=2)}
            
            추천 형식:
            1. 가장 적합한 대분야/중분야/세부분야
            2. 추천 이유
            3. 관련 키워드 5-10개
            """
            
            response = self.api_manager.call_ai(prompt)
            
            if response:
                st.info(response)

# 페이지 렌더링
def render():
    """페이지 렌더링 함수"""
    page = ProjectSetupPage()
    page.render()

# 메인 실행
if __name__ == "__main__":
    render()
