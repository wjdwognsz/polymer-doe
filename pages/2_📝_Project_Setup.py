"""
📝 프로젝트 설정 페이지
연구 프로젝트를 생성, 관리, 편집하는 핵심 페이지
"""
import streamlit as st
import pandas as pd
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import sys
from pathlib import Path

# 경로 설정
sys.path.append(str(Path(__file__).parent.parent))

# 페이지 설정 (Streamlit Pages 필수)
st.set_page_config(
    page_title="프로젝트 설정 - Universal DOE",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 인증 체크
if 'authenticated' not in st.session_state or not st.session_state.authenticated:
    if not st.session_state.get('guest_mode', False):
        st.warning("🔐 로그인이 필요합니다")
        st.switch_page("pages/0_🔐_Login.py")
        st.stop()

# 모듈 임포트
try:
    from utils.database_manager import get_database_manager
    from utils.common_ui import get_common_ui
    from utils.api_manager import get_api_manager
    from utils.notification_manager import get_notification_manager
    from modules.module_registry import get_module_registry
    from config.app_config import PROJECT_TYPES, EXPERIMENT_DEFAULTS
except ImportError as e:
    st.error(f"필수 모듈 임포트 오류: {e}")
    st.stop()

# 전역 인스턴스
db_manager = get_database_manager()
ui = get_common_ui()
api_manager = get_api_manager()
notifier = get_notification_manager()
module_registry = get_module_registry()

# 프로젝트 관련 상수
PROJECT_STATUS = ["활성", "일시중지", "완료", "보관"]
PROJECT_VISIBILITY = ["비공개", "팀 공개", "전체 공개"]
COLLABORATOR_ROLES = ["소유자", "편집자", "뷰어"]

def initialize_session_state():
    """세션 상태 초기화"""
    defaults = {
        'project_step': 0,
        'new_project': {},
        'selected_modules': [],
        'project_view': 'grid',
        'show_ai_details': False,  # AI 설명 상세도
        'ai_recommendations': None,
        'editing_project': None,
        'project_filter': {'status': '전체', 'search': ''},
        'selected_project_id': None,
        'show_template_save': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def render_ai_response(response: Dict[str, Any], response_type: str = "general"):
    """
    AI 응답 렌더링 (상세 설명 토글 포함)
    프로젝트 지침서의 AI 투명성 원칙 구현
    """
    # 핵심 답변 (항상 표시)
    st.markdown(f"### 🤖 {response_type} AI 추천")
    st.write(response.get('main', '추천 내용이 없습니다.'))
    
    # 상세 설명 토글
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🔍 상세 설명", key=f"toggle_{response_type}"):
            st.session_state.show_ai_details = not st.session_state.show_ai_details
    
    # 상세 설명 (조건부 표시)
    if st.session_state.show_ai_details:
        with st.expander("📚 AI 추론 과정", expanded=True):
            tabs = st.tabs(["추론 과정", "대안", "배경", "신뢰도"])
            
            with tabs[0]:
                st.markdown("#### 추론 과정")
                st.write(response.get('reasoning', '추론 과정 정보가 없습니다.'))
            
            with tabs[1]:
                st.markdown("#### 검토한 대안들")
                alternatives = response.get('alternatives', [])
                if alternatives:
                    for alt in alternatives:
                        st.write(f"- **{alt.get('name')}**: {alt.get('description')}")
                        st.caption(f"  장점: {alt.get('pros', 'N/A')}")
                        st.caption(f"  단점: {alt.get('cons', 'N/A')}")
                else:
                    st.info("대안 정보가 없습니다.")
            
            with tabs[2]:
                st.markdown("#### 이론적 배경")
                st.write(response.get('theory', '이론적 배경 정보가 없습니다.'))
            
            with tabs[3]:
                st.markdown("#### 신뢰도 평가")
                confidence = response.get('confidence', 85)
                st.progress(confidence / 100)
                st.write(f"신뢰도: {confidence}%")
                st.write(response.get('limitations', '한계점 정보가 없습니다.'))

def render_project_list():
    """프로젝트 목록 렌더링"""
    st.subheader("📋 내 프로젝트")
    
    # 필터 및 뷰 옵션
    col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
    
    with col1:
        search_query = st.text_input(
            "🔍 검색",
            placeholder="프로젝트명, 태그, 설명...",
            value=st.session_state.project_filter['search']
        )
    
    with col2:
        status_filter = st.selectbox(
            "상태",
            ["전체"] + PROJECT_STATUS,
            index=0
        )
    
    with col3:
        view_mode = st.radio(
            "보기",
            ["그리드", "리스트"],
            horizontal=True,
            index=0 if st.session_state.project_view == 'grid' else 1
        )
        st.session_state.project_view = view_mode.lower()
    
    with col4:
        sort_by = st.selectbox(
            "정렬",
            ["최신순", "이름순", "수정일순"],
            index=0
        )
    
    # 프로젝트 로드
    user_id = st.session_state.user.get('id') if not st.session_state.get('guest_mode') else None
    projects = db_manager.get_user_projects(user_id) if user_id else []
    
    # 필터링
    if search_query:
        projects = [p for p in projects if 
                   search_query.lower() in p['name'].lower() or
                   search_query.lower() in p.get('description', '').lower() or
                   any(search_query.lower() in tag.lower() for tag in p.get('tags', []))]
    
    if status_filter != "전체":
        projects = [p for p in projects if p.get('status') == status_filter]
    
    # 정렬
    if sort_by == "최신순":
        projects.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    elif sort_by == "이름순":
        projects.sort(key=lambda x: x.get('name', ''))
    elif sort_by == "수정일순":
        projects.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
    
    # 프로젝트 표시
    if not projects:
        ui.render_empty_state("프로젝트가 없습니다", "📭")
        if st.button("🚀 첫 프로젝트 만들기", type="primary"):
            st.session_state.project_step = 0
            st.rerun()
    else:
        if view_mode == "그리드":
            render_project_grid(projects)
        else:
            render_project_list_view(projects)

def render_project_grid(projects: List[Dict]):
    """프로젝트 그리드 뷰"""
    cols = st.columns(3)
    for idx, project in enumerate(projects):
        with cols[idx % 3]:
            render_project_card(project)

def render_project_card(project: Dict):
    """프로젝트 카드 렌더링"""
    with st.container():
        # 카드 스타일
        st.markdown("""
        <style>
        .project-card {
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            transition: all 0.3s ease;
        }
        .project-card:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }
        </style>
        """, unsafe_allow_html=True)
        
        with st.container():
            # 프로젝트 헤더
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"### {project['name']}")
            with col2:
                status_emoji = {
                    "활성": "🟢",
                    "일시중지": "🟡",
                    "완료": "🔵",
                    "보관": "⚫"
                }
                st.write(status_emoji.get(project.get('status', '활성'), '⚪'))
            
            # 프로젝트 정보
            st.caption(f"생성일: {project.get('created_at', 'N/A')[:10]}")
            
            if project.get('description'):
                st.write(project['description'][:100] + "..." if len(project['description']) > 100 else project['description'])
            
            # 태그
            if project.get('tags'):
                tag_html = " ".join([f"<span style='background-color: #e3f2fd; padding: 2px 8px; border-radius: 12px; margin-right: 4px; font-size: 0.8em;'>{tag}</span>" for tag in project['tags'][:3]])
                st.markdown(tag_html, unsafe_allow_html=True)
            
            # 협업자 수
            collaborators = project.get('collaborators', [])
            if len(collaborators) > 1:
                st.caption(f"👥 {len(collaborators)}명 협업 중")
            
            # 액션 버튼
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("열기", key=f"open_{project['id']}", use_container_width=True):
                    st.session_state.selected_project_id = project['id']
                    st.switch_page("pages/3_🧪_Experiment_Design.py")
            
            with col2:
                if st.button("편집", key=f"edit_{project['id']}", use_container_width=True):
                    st.session_state.editing_project = project
                    st.rerun()
            
            with col3:
                if st.button("⋮", key=f"more_{project['id']}", use_container_width=True):
                    show_project_menu(project)

def render_project_list_view(projects: List[Dict]):
    """프로젝트 리스트 뷰"""
    df_data = []
    for project in projects:
        df_data.append({
            "프로젝트명": project['name'],
            "상태": project.get('status', '활성'),
            "유형": project.get('type', 'N/A'),
            "생성일": project.get('created_at', '')[:10],
            "수정일": project.get('updated_at', '')[:10],
            "협업자": len(project.get('collaborators', [])),
            "ID": project['id']
        })
    
    df = pd.DataFrame(df_data)
    
    # 데이터프레임 표시
    selected_rows = st.dataframe(
        df.drop(columns=['ID']),
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun"
    )
    
    # 선택된 행 처리
    if selected_rows and selected_rows.selection.rows:
        selected_idx = selected_rows.selection.rows[0]
        selected_project = projects[selected_idx]
        
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            if st.button("🔍 열기", type="primary"):
                st.session_state.selected_project_id = selected_project['id']
                st.switch_page("pages/3_🧪_Experiment_Design.py")
        with col2:
            if st.button("✏️ 편집"):
                st.session_state.editing_project = selected_project
                st.rerun()

def render_new_project_wizard():
    """새 프로젝트 생성 마법사"""
    st.subheader("🚀 새 프로젝트 만들기")
    
    # 진행 단계 표시
    steps = ["기본 정보", "실험 모듈", "AI 설정", "협업 설정", "검토 및 생성"]
    progress = (st.session_state.project_step + 1) / len(steps)
    st.progress(progress)
    st.write(f"단계 {st.session_state.project_step + 1}/{len(steps)}: {steps[st.session_state.project_step]}")
    
    # 각 단계별 렌더링
    if st.session_state.project_step == 0:
        render_basic_info_step()
    elif st.session_state.project_step == 1:
        render_module_selection_step()
    elif st.session_state.project_step == 2:
        render_ai_settings_step()
    elif st.session_state.project_step == 3:
        render_collaboration_step()
    elif st.session_state.project_step == 4:
        render_review_and_create_step()
    
    # 네비게이션 버튼
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        if st.session_state.project_step > 0:
            if st.button("⬅️ 이전", use_container_width=True):
                st.session_state.project_step -= 1
                st.rerun()
    
    with col3:
        if st.session_state.project_step < len(steps) - 1:
            if st.button("다음 ➡️", use_container_width=True, type="primary"):
                if validate_current_step():
                    st.session_state.project_step += 1
                    st.rerun()

def render_basic_info_step():
    """기본 정보 입력 단계"""
    st.markdown("### 📋 기본 정보")
    
    col1, col2 = st.columns(2)
    
    with col1:
        project_name = st.text_input(
            "프로젝트명 *",
            value=st.session_state.new_project.get('name', ''),
            placeholder="예: 고강도 PET 필름 개발"
        )
        st.session_state.new_project['name'] = project_name
        
        project_type = st.selectbox(
            "프로젝트 유형 *",
            list(PROJECT_TYPES.keys()),
            format_func=lambda x: PROJECT_TYPES[x]['name'],
            index=0
        )
        st.session_state.new_project['type'] = project_type
        
        # 유형별 세부 카테고리
        if project_type in PROJECT_TYPES:
            subcategory = st.selectbox(
                "세부 분야",
                PROJECT_TYPES[project_type]['subcategories']
            )
            st.session_state.new_project['subcategory'] = subcategory
    
    with col2:
        visibility = st.radio(
            "공개 범위",
            PROJECT_VISIBILITY,
            index=0,
            help="프로젝트의 공개 범위를 설정합니다. 나중에 변경 가능합니다."
        )
        st.session_state.new_project['visibility'] = visibility
        
        priority = st.select_slider(
            "우선순위",
            options=["낮음", "보통", "높음", "긴급"],
            value=st.session_state.new_project.get('priority', '보통')
        )
        st.session_state.new_project['priority'] = priority
        
        # 예상 기간
        duration = st.number_input(
            "예상 기간 (주)",
            min_value=1,
            max_value=52,
            value=st.session_state.new_project.get('duration', 4)
        )
        st.session_state.new_project['duration'] = duration
    
    # 프로젝트 설명
    description = st.text_area(
        "프로젝트 설명",
        value=st.session_state.new_project.get('description', ''),
        placeholder="프로젝트의 목적, 배경, 기대 효과 등을 자세히 설명해주세요.",
        height=150
    )
    st.session_state.new_project['description'] = description
    
    # 태그
    tags_input = st.text_input(
        "태그 (쉼표로 구분)",
        value=', '.join(st.session_state.new_project.get('tags', [])),
        placeholder="예: PET, 필름, 고강도, 투명성"
    )
    if tags_input:
        st.session_state.new_project['tags'] = [tag.strip() for tag in tags_input.split(',') if tag.strip()]
    
    # AI 추천 활용
    if st.button("🤖 AI 프로젝트 설정 추천"):
        with st.spinner("AI가 추천을 생성하는 중..."):
            recommendations = get_ai_project_recommendations()
            if recommendations:
                st.session_state.ai_recommendations = recommendations
                render_ai_response(recommendations, "프로젝트 설정")

def render_module_selection_step():
    """실험 모듈 선택 단계"""
    st.markdown("### 🧪 실험 모듈 선택")
    st.info("프로젝트에서 사용할 실험 설계 모듈을 선택하세요.")
    
    # 모듈 카테고리
    categories = module_registry.get_categories()
    selected_category = st.selectbox(
        "모듈 카테고리",
        ["전체"] + categories,
        index=0
    )
    
    # 모듈 목록
    if selected_category == "전체":
        modules = module_registry.list_modules()
    else:
        modules = module_registry.list_modules(category=selected_category)
    
    # 선택된 모듈 표시
    if st.session_state.selected_modules:
        st.write("### 선택된 모듈")
        for module_id in st.session_state.selected_modules:
            module = module_registry.get_module(module_id)
            if module:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"✅ {module.get_module_info()['name']}")
                with col2:
                    if st.button("제거", key=f"remove_{module_id}"):
                        st.session_state.selected_modules.remove(module_id)
                        st.rerun()
    
    # 모듈 그리드
    st.write("### 사용 가능한 모듈")
    
    if not modules:
        st.warning("사용 가능한 모듈이 없습니다.")
    else:
        cols = st.columns(2)
        for idx, module_info in enumerate(modules):
            with cols[idx % 2]:
                render_module_card(module_info)
    
    # AI 모듈 추천
    if st.button("🤖 AI 모듈 추천"):
        with st.spinner("AI가 적합한 모듈을 추천하는 중..."):
            recommendations = get_ai_module_recommendations()
            if recommendations:
                render_ai_response(recommendations, "모듈 추천")

def render_module_card(module_info: Dict):
    """모듈 카드 렌더링"""
    with st.expander(f"{module_info['name']} - {module_info.get('category', 'general')}"):
        st.write(module_info.get('description', '설명이 없습니다.'))
        
        # 태그
        if module_info.get('tags'):
            st.write("태그:", ', '.join(module_info['tags']))
        
        # 선택 버튼
        module_id = module_info['id']
        if module_id not in st.session_state.selected_modules:
            if st.button(f"선택", key=f"select_{module_id}", use_container_width=True):
                st.session_state.selected_modules.append(module_id)
                st.rerun()
        else:
            st.success("✅ 선택됨")

def render_ai_settings_step():
    """AI 설정 단계"""
    st.markdown("### 🤖 AI 설정")
    st.info("프로젝트에서 사용할 AI 엔진과 설정을 구성합니다.")
    
    # AI 엔진 활성화
    st.write("#### AI 엔진 선택")
    
    available_engines = api_manager.get_available_engines()
    selected_engines = st.multiselect(
        "사용할 AI 엔진",
        available_engines,
        default=st.session_state.new_project.get('ai_engines', ['google_gemini']),
        format_func=lambda x: api_manager.get_engine_info(x)['name']
    )
    st.session_state.new_project['ai_engines'] = selected_engines
    
    # AI 설정
    st.write("#### AI 동작 설정")
    
    col1, col2 = st.columns(2)
    
    with col1:
        ai_mode = st.radio(
            "AI 지원 수준",
            ["자동", "제안", "수동"],
            index=0,
            help="자동: AI가 적극적으로 제안\n제안: 요청 시에만 AI 활용\n수동: AI 기능 최소화"
        )
        st.session_state.new_project['ai_mode'] = ai_mode
        
        auto_optimization = st.checkbox(
            "자동 최적화 활성화",
            value=st.session_state.new_project.get('auto_optimization', True),
            help="AI가 실험 설계를 자동으로 최적화합니다"
        )
        st.session_state.new_project['auto_optimization'] = auto_optimization
    
    with col2:
        explanation_detail = st.select_slider(
            "AI 설명 상세도",
            options=["간단", "보통", "상세", "전문가"],
            value=st.session_state.new_project.get('explanation_detail', '보통'),
            help="AI 응답의 기본 상세도를 설정합니다"
        )
        st.session_state.new_project['explanation_detail'] = explanation_detail
        
        use_citations = st.checkbox(
            "참고문헌 포함",
            value=st.session_state.new_project.get('use_citations', False),
            help="AI 응답에 과학 문헌 인용을 포함합니다"
        )
        st.session_state.new_project['use_citations'] = use_citations
    
    # 고급 설정
    with st.expander("고급 AI 설정"):
        temperature = st.slider(
            "창의성 (Temperature)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.new_project.get('ai_temperature', 0.7),
            step=0.1,
            help="낮을수록 일관된 응답, 높을수록 창의적인 응답"
        )
        st.session_state.new_project['ai_temperature'] = temperature
        
        max_iterations = st.number_input(
            "최대 최적화 반복 횟수",
            min_value=1,
            max_value=100,
            value=st.session_state.new_project.get('max_iterations', 10)
        )
        st.session_state.new_project['max_iterations'] = max_iterations

def render_collaboration_step():
    """협업 설정 단계"""
    st.markdown("### 👥 협업 설정")
    
    # 협업 모드
    collab_mode = st.radio(
        "협업 모드",
        ["개인 프로젝트", "팀 프로젝트", "오픈 협업"],
        index=0,
        help="개인: 본인만 접근\n팀: 초대된 멤버만 접근\n오픈: 누구나 참여 가능"
    )
    st.session_state.new_project['collab_mode'] = collab_mode
    
    if collab_mode != "개인 프로젝트":
        # 팀원 초대
        st.write("#### 팀원 초대")
        
        invite_method = st.radio(
            "초대 방법",
            ["이메일로 초대", "링크 공유", "사용자 검색"],
            horizontal=True
        )
        
        if invite_method == "이메일로 초대":
            emails = st.text_area(
                "이메일 주소 (한 줄에 하나씩)",
                placeholder="user1@example.com\nuser2@example.com",
                height=100
            )
            
            if emails:
                email_list = [e.strip() for e in emails.split('\n') if e.strip()]
                
                # 권한 설정
                default_role = st.selectbox(
                    "기본 권한",
                    ["뷰어", "편집자"],
                    index=1
                )
                
                if st.button("초대장 발송"):
                    st.session_state.new_project['invitations'] = {
                        'emails': email_list,
                        'role': default_role
                    }
                    st.success(f"{len(email_list)}명에게 초대장을 발송했습니다.")
        
        elif invite_method == "링크 공유":
            st.info("프로젝트 생성 후 공유 링크가 생성됩니다.")
            
        elif invite_method == "사용자 검색":
            search_user = st.text_input("사용자 검색", placeholder="이름 또는 이메일")
            if search_user:
                # 더미 검색 결과 (실제로는 DB에서 검색)
                st.write("검색 결과:")
                if st.button("김연구원 추가"):
                    st.success("김연구원을 팀에 추가했습니다.")
    
    # 권한 정책
    st.write("#### 권한 정책")
    
    col1, col2 = st.columns(2)
    
    with col1:
        allow_guest_view = st.checkbox(
            "게스트 읽기 허용",
            value=False,
            help="로그인하지 않은 사용자도 프로젝트를 볼 수 있습니다"
        )
        st.session_state.new_project['allow_guest_view'] = allow_guest_view
        
        require_approval = st.checkbox(
            "참여 승인 필요",
            value=True,
            help="새 멤버 참여 시 관리자 승인이 필요합니다"
        )
        st.session_state.new_project['require_approval'] = require_approval
    
    with col2:
        enable_comments = st.checkbox(
            "댓글 기능 활성화",
            value=True
        )
        st.session_state.new_project['enable_comments'] = enable_comments
        
        enable_version_control = st.checkbox(
            "버전 관리 활성화",
            value=True,
            help="모든 변경사항의 히스토리를 저장합니다"
        )
        st.session_state.new_project['enable_version_control'] = enable_version_control

def render_review_and_create_step():
    """검토 및 생성 단계"""
    st.markdown("### 📋 프로젝트 검토")
    st.info("프로젝트 설정을 검토하고 생성하세요.")
    
    # 설정 요약
    project = st.session_state.new_project
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### 기본 정보")
        st.write(f"- **프로젝트명**: {project.get('name', 'N/A')}")
        st.write(f"- **유형**: {PROJECT_TYPES.get(project.get('type', ''), {}).get('name', 'N/A')}")
        st.write(f"- **세부 분야**: {project.get('subcategory', 'N/A')}")
        st.write(f"- **공개 범위**: {project.get('visibility', '비공개')}")
        st.write(f"- **우선순위**: {project.get('priority', '보통')}")
        st.write(f"- **예상 기간**: {project.get('duration', 4)}주")
        
        st.write("#### 실험 모듈")
        if st.session_state.selected_modules:
            for module_id in st.session_state.selected_modules:
                module = module_registry.get_module(module_id)
                if module:
                    st.write(f"- {module.get_module_info()['name']}")
        else:
            st.write("- 선택된 모듈 없음")
    
    with col2:
        st.write("#### AI 설정")
        st.write(f"- **AI 엔진**: {', '.join(project.get('ai_engines', ['없음']))}")
        st.write(f"- **지원 수준**: {project.get('ai_mode', '자동')}")
        st.write(f"- **설명 상세도**: {project.get('explanation_detail', '보통')}")
        st.write(f"- **자동 최적화**: {'활성' if project.get('auto_optimization') else '비활성'}")
        
        st.write("#### 협업 설정")
        st.write(f"- **협업 모드**: {project.get('collab_mode', '개인 프로젝트')}")
        st.write(f"- **게스트 읽기**: {'허용' if project.get('allow_guest_view') else '차단'}")
        st.write(f"- **참여 승인**: {'필요' if project.get('require_approval') else '불필요'}")
        
        if project.get('invitations'):
            st.write(f"- **초대 대기**: {len(project['invitations']['emails'])}명")
    
    # 템플릿 저장 옵션
    st.divider()
    save_as_template = st.checkbox(
        "이 설정을 템플릿으로 저장",
        value=st.session_state.new_project.get('save_as_template', False),
        help="나중에 비슷한 프로젝트를 만들 때 사용할 수 있습니다"
    )
    
    if save_as_template:
        template_name = st.text_input(
            "템플릿 이름",
            placeholder="예: 고분자 필름 개발 템플릿"
        )
        st.session_state.new_project['template_name'] = template_name
    
    st.session_state.new_project['save_as_template'] = save_as_template
    
    # 생성 버튼
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 프로젝트 생성", type="primary", use_container_width=True):
            create_project()

def render_project_editor():
    """프로젝트 편집기"""
    if not st.session_state.editing_project:
        st.info("편집할 프로젝트를 선택하세요")
        return
    
    project = st.session_state.editing_project
    st.subheader(f"✏️ 프로젝트 편집: {project['name']}")
    
    # 편집 폼
    with st.form("project_edit_form"):
        # 기본 정보
        st.markdown("#### 기본 정보")
        
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("프로젝트명", value=project['name'])
            project_type = st.selectbox(
                "유형",
                list(PROJECT_TYPES.keys()),
                format_func=lambda x: PROJECT_TYPES[x]['name'],
                index=list(PROJECT_TYPES.keys()).index(project.get('type', 'general'))
            )
        
        with col2:
            status = st.selectbox(
                "상태",
                PROJECT_STATUS,
                index=PROJECT_STATUS.index(project.get('status', '활성'))
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
        
        # 협업자 관리
        st.markdown("#### 협업자 관리")
        
        collaborators = project.get('collaborators', [])
        if collaborators:
            for idx, collab in enumerate(collaborators):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"{collab.get('name', collab.get('email', 'Unknown'))}")
                with col2:
                    new_role = st.selectbox(
                        "권한",
                        COLLABORATOR_ROLES,
                        index=COLLABORATOR_ROLES.index(collab.get('role', '뷰어')),
                        key=f"role_{idx}"
                    )
                with col3:
                    if st.button("제거", key=f"remove_collab_{idx}"):
                        collaborators.pop(idx)
        
        # 저장 버튼
        submitted = st.form_submit_button("💾 변경사항 저장", type="primary", use_container_width=True)
        
        if submitted:
            # 업데이트 로직
            updated_project = {
                'id': project['id'],
                'name': name,
                'type': project_type,
                'status': status,
                'priority': priority,
                'description': description,
                'updated_at': datetime.now().isoformat()
            }
            
            if db_manager.update_project(project['id'], updated_project):
                st.success("프로젝트가 업데이트되었습니다!")
                st.session_state.editing_project = None
                st.rerun()
            else:
                st.error("프로젝트 업데이트 중 오류가 발생했습니다.")
    
    # 취소 버튼
    if st.button("취소"):
        st.session_state.editing_project = None
        st.rerun()

def render_template_manager():
    """템플릿 관리"""
    st.subheader("📚 프로젝트 템플릿")
    
    # 템플릿 필터
    col1, col2 = st.columns([3, 1])
    with col1:
        template_search = st.text_input("템플릿 검색", placeholder="템플릿 이름 또는 태그")
    with col2:
        template_sort = st.selectbox("정렬", ["인기순", "최신순", "이름순"])
    
    # 템플릿 로드
    templates = db_manager.get_templates(st.session_state.user.get('id'))
    
    if not templates:
        ui.render_empty_state("템플릿이 없습니다", "📚")
    else:
        # 템플릿 그리드
        cols = st.columns(2)
        for idx, template in enumerate(templates):
            with cols[idx % 2]:
                render_template_card(template)

def render_template_card(template: Dict):
    """템플릿 카드 렌더링"""
    with st.expander(f"📋 {template['name']}"):
        st.write(template.get('description', '설명이 없습니다.'))
        
        # 템플릿 정보
        st.caption(f"생성자: {template.get('creator_name', 'Unknown')}")
        st.caption(f"사용 횟수: {template.get('usage_count', 0)}회")
        
        if template.get('tags'):
            st.write("태그:", ', '.join(template['tags']))
        
        # 액션 버튼
        col1, col2 = st.columns(2)
        with col1:
            if st.button("사용하기", key=f"use_template_{template['id']}", use_container_width=True):
                load_template(template)
        with col2:
            if template.get('creator_id') == st.session_state.user.get('id'):
                if st.button("삭제", key=f"delete_template_{template['id']}", use_container_width=True):
                    if db_manager.delete_template(template['id']):
                        st.success("템플릿이 삭제되었습니다.")
                        st.rerun()

def validate_current_step() -> bool:
    """현재 단계 검증"""
    if st.session_state.project_step == 0:
        # 기본 정보 검증
        if not st.session_state.new_project.get('name'):
            st.error("프로젝트명을 입력해주세요.")
            return False
        if not st.session_state.new_project.get('type'):
            st.error("프로젝트 유형을 선택해주세요.")
            return False
    
    elif st.session_state.project_step == 1:
        # 모듈 선택 검증
        if not st.session_state.selected_modules:
            if not st.confirm("실험 모듈을 선택하지 않았습니다. 계속하시겠습니까?"):
                return False
    
    return True

def create_project():
    """프로젝트 생성"""
    project_data = st.session_state.new_project.copy()
    
    # 프로젝트 ID 생성
    project_data['id'] = str(uuid.uuid4())
    project_data['user_id'] = st.session_state.user.get('id')
    project_data['modules'] = st.session_state.selected_modules
    project_data['status'] = '활성'
    project_data['created_at'] = datetime.now().isoformat()
    project_data['updated_at'] = datetime.now().isoformat()
    
    # 협업자 초기화
    project_data['collaborators'] = [{
        'user_id': st.session_state.user.get('id'),
        'email': st.session_state.user.get('email'),
        'name': st.session_state.user.get('name'),
        'role': '소유자'
    }]
    
    # DB에 저장
    if db_manager.create_project(project_data):
        st.success("✅ 프로젝트가 성공적으로 생성되었습니다!")
        
        # 템플릿 저장
        if project_data.get('save_as_template') and project_data.get('template_name'):
            save_as_template(project_data)
        
        # 초대장 발송
        if project_data.get('invitations'):
            send_invitations(project_data)
        
        # 상태 초기화
        st.session_state.new_project = {}
        st.session_state.selected_modules = []
        st.session_state.project_step = 0
        st.session_state.selected_project_id = project_data['id']
        
        # 실험 설계 페이지로 이동
        st.balloons()
        time.sleep(1)
        st.switch_page("pages/3_🧪_Experiment_Design.py")
    else:
        st.error("프로젝트 생성 중 오류가 발생했습니다.")

def save_as_template(project_data: Dict):
    """프로젝트를 템플릿으로 저장"""
    template_data = {
        'id': str(uuid.uuid4()),
        'name': project_data['template_name'],
        'description': f"{project_data['name']} 프로젝트 템플릿",
        'creator_id': st.session_state.user.get('id'),
        'creator_name': st.session_state.user.get('name'),
        'project_data': {
            'type': project_data['type'],
            'subcategory': project_data.get('subcategory'),
            'modules': project_data['modules'],
            'ai_engines': project_data.get('ai_engines', []),
            'ai_mode': project_data.get('ai_mode'),
            'tags': project_data.get('tags', [])
        },
        'usage_count': 0,
        'created_at': datetime.now().isoformat()
    }
    
    db_manager.create_template(template_data)
    st.info("📋 템플릿이 저장되었습니다.")

def send_invitations(project_data: Dict):
    """초대장 발송"""
    invitations = project_data.get('invitations', {})
    emails = invitations.get('emails', [])
    role = invitations.get('role', '뷰어')
    
    for email in emails:
        # 알림 발송
        notifier.send_project_invitation(
            project_id=project_data['id'],
            project_name=project_data['name'],
            inviter_name=st.session_state.user.get('name'),
            invitee_email=email,
            role=role
        )
    
    st.info(f"📧 {len(emails)}명에게 초대장을 발송했습니다.")

def load_template(template: Dict):
    """템플릿 로드"""
    template_data = template.get('project_data', {})
    
    # 새 프로젝트 데이터로 설정
    st.session_state.new_project = {
        'name': '',  # 비워둠
        'type': template_data.get('type'),
        'subcategory': template_data.get('subcategory'),
        'ai_engines': template_data.get('ai_engines', []),
        'ai_mode': template_data.get('ai_mode'),
        'tags': template_data.get('tags', [])
    }
    st.session_state.selected_modules = template_data.get('modules', [])
    st.session_state.project_step = 0
    
    # 사용 횟수 증가
    db_manager.increment_template_usage(template['id'])
    
    st.success(f"템플릿 '{template['name']}'을 불러왔습니다.")
    st.rerun()

def get_ai_project_recommendations() -> Dict[str, Any]:
    """AI 프로젝트 추천 생성"""
    project_info = st.session_state.new_project
    
    prompt = f"""
    다음 프로젝트 정보를 바탕으로 상세한 추천을 제공해주세요:
    - 프로젝트명: {project_info.get('name', '미정')}
    - 유형: {project_info.get('type', '일반')}
    - 설명: {project_info.get('description', '없음')}
    
    다음 형식으로 응답해주세요:
    {{
        "main": "핵심 추천 내용 (2-3문장)",
        "reasoning": "이런 추천을 하는 이유와 단계별 분석",
        "alternatives": [
            {{"name": "대안1", "description": "설명", "pros": "장점", "cons": "단점"}},
            {{"name": "대안2", "description": "설명", "pros": "장점", "cons": "단점"}}
        ],
        "theory": "관련 이론적 배경과 과학적 원리",
        "confidence": 85,
        "limitations": "이 추천의 한계점과 주의사항"
    }}
    """
    
    try:
        response = api_manager.generate_structured_response(prompt)
        return json.loads(response) if isinstance(response, str) else response
    except Exception as e:
        st.error(f"AI 추천 생성 중 오류: {e}")
        return None

def get_ai_module_recommendations() -> Dict[str, Any]:
    """AI 모듈 추천 생성"""
    project_info = st.session_state.new_project
    available_modules = module_registry.list_modules()
    
    prompt = f"""
    프로젝트 정보:
    - 유형: {project_info.get('type')}
    - 세부분야: {project_info.get('subcategory')}
    - 설명: {project_info.get('description', '없음')}
    
    사용 가능한 모듈:
    {json.dumps([m['name'] for m in available_modules[:10]], ensure_ascii=False)}
    
    적합한 실험 모듈을 추천하고 다음 형식으로 응답해주세요:
    {{
        "main": "추천 모듈과 이유 요약",
        "reasoning": "각 모듈 선택의 상세한 이유",
        "alternatives": [
            {{"name": "모듈명", "description": "왜 적합한지", "pros": "장점", "cons": "단점"}}
        ],
        "theory": "실험 설계 이론과 모듈 선택의 과학적 근거",
        "confidence": 90,
        "limitations": "주의사항"
    }}
    """
    
    try:
        response = api_manager.generate_structured_response(prompt)
        return json.loads(response) if isinstance(response, str) else response
    except Exception as e:
        st.error(f"AI 모듈 추천 중 오류: {e}")
        return None

def show_project_menu(project: Dict):
    """프로젝트 추가 메뉴"""
    with st.popover("프로젝트 옵션"):
        if st.button("📤 내보내기", use_container_width=True):
            export_project(project)
        
        if st.button("📋 복제", use_container_width=True):
            duplicate_project(project)
        
        if st.button("📊 통계 보기", use_container_width=True):
            show_project_stats(project)
        
        if st.button("🗑️ 삭제", use_container_width=True):
            if st.confirm("정말 삭제하시겠습니까?"):
                if db_manager.delete_project(project['id']):
                    st.success("프로젝트가 삭제되었습니다.")
                    st.rerun()

def export_project(project: Dict):
    """프로젝트 내보내기"""
    export_data = {
        'project': project,
        'modules': [module_registry.get_module(m).get_module_info() for m in project.get('modules', [])],
        'exported_at': datetime.now().isoformat(),
        'version': '2.0'
    }
    
    json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
    
    st.download_button(
        label="📥 JSON 다운로드",
        data=json_str,
        file_name=f"project_{project['name']}_{datetime.now().strftime('%Y%m%d')}.json",
        mime="application/json"
    )

def duplicate_project(project: Dict):
    """프로젝트 복제"""
    new_project = project.copy()
    new_project['id'] = str(uuid.uuid4())
    new_project['name'] = f"{project['name']} (복사본)"
    new_project['created_at'] = datetime.now().isoformat()
    new_project['updated_at'] = datetime.now().isoformat()
    
    if db_manager.create_project(new_project):
        st.success("프로젝트가 복제되었습니다.")
        st.rerun()

def show_project_stats(project: Dict):
    """프로젝트 통계"""
    st.write(f"### {project['name']} 통계")
    
    # 더미 통계 (실제로는 DB에서 계산)
    stats = {
        "실험 수": 15,
        "완료율": 73,
        "평균 성공률": 85,
        "협업자 활동": "높음"
    }
    
    cols = st.columns(len(stats))
    for idx, (key, value) in enumerate(stats.items()):
        with cols[idx]:
            st.metric(key, value)

# 메인 실행
def main():
    """메인 함수"""
    initialize_session_state()
    
    # 헤더
    ui.render_header("📝 프로젝트 설정", "연구 프로젝트를 생성하고 관리합니다")
    
    # AI 설명 모드 전역 설정
    with st.sidebar:
        st.divider()
        st.markdown("### 🤖 AI 설정")
        ai_detail_mode = st.radio(
            "AI 설명 모드",
            ["자동", "항상 간단히", "항상 상세히"],
            index=0,
            help="AI 응답의 상세도를 설정합니다"
        )
        
        if ai_detail_mode == "항상 상세히":
            st.session_state.show_ai_details = True
        elif ai_detail_mode == "항상 간단히":
            st.session_state.show_ai_details = False
        # "자동"은 사용자 레벨에 따라 결정
    
    # 게스트 모드 체크
    if st.session_state.get('guest_mode'):
        st.info("👀 게스트 모드로 둘러보는 중입니다. 일부 기능이 제한됩니다.")
    
    # 탭 구성
    tabs = st.tabs([
        "📋 프로젝트 목록",
        "➕ 새 프로젝트",
        "✏️ 프로젝트 편집",
        "📚 템플릿 관리"
    ])
    
    with tabs[0]:
        render_project_list()
    
    with tabs[1]:
        render_new_project_wizard()
    
    with tabs[2]:
        render_project_editor()
    
    with tabs[3]:
        render_template_manager()

# 필요한 추가 임포트
import time

if __name__ == "__main__":
    main()
