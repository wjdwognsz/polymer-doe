"""
pages/project_setup.py - 프로젝트 설정 페이지
Universal DOE Platform의 프로젝트 생성 및 관리 페이지
"""

import streamlit as st
import pandas as pd
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

# 내부 모듈 임포트
from utils.common_ui import get_common_ui
from utils.database_manager import get_database_manager
from utils.auth_manager import get_auth_manager
from utils.api_manager import get_api_manager
from utils.notification_manager import get_notification_manager
from modules.module_registry import get_module_registry
from config.app_config import EXPERIMENT_DEFAULTS, SECURITY_CONFIG

# 로깅 설정
logger = logging.getLogger(__name__)

# 권한 레벨 정의
PERMISSION_LEVELS = {
    "owner": {
        "can_edit": True,
        "can_delete": True,
        "can_invite": True,
        "can_remove_members": True,
        "can_change_visibility": True,
        "can_export": True
    },
    "editor": {
        "can_edit": True,
        "can_delete": False,
        "can_invite": True,
        "can_remove_members": False,
        "can_change_visibility": False,
        "can_export": True
    },
    "viewer": {
        "can_edit": False,
        "can_delete": False,
        "can_invite": False,
        "can_remove_members": False,
        "can_change_visibility": False,
        "can_export": True
    }
}

class ProjectSetupManager:
    """프로젝트 설정 관리 클래스"""
    
    def __init__(self):
        self.ui = get_common_ui()
        self.db = get_database_manager()
        self.auth = get_auth_manager()
        self.api = get_api_manager()
        self.notifier = get_notification_manager()
        self.module_registry = get_module_registry()
        self.current_user = self.auth.get_current_user()
        
    def render_page(self):
        """프로젝트 설정 페이지 메인"""
        # 인증 확인
        if not self.current_user:
            st.warning("로그인이 필요합니다.")
            st.stop()
            
        # 페이지 헤더
        self.ui.render_header(
            "프로젝트 관리",
            "실험 프로젝트를 생성하고 관리합니다",
            "📝"
        )
        
        # 프로젝트 탭
        tabs = st.tabs([
            "내 프로젝트",
            "공유된 프로젝트",
            "템플릿",
            "새 프로젝트"
        ])
        
        with tabs[0]:
            self._render_my_projects()
            
        with tabs[1]:
            self._render_shared_projects()
            
        with tabs[2]:
            self._render_templates()
            
        with tabs[3]:
            self._render_new_project()
    
    def _render_my_projects(self):
        """내 프로젝트 목록"""
        st.subheader("내 프로젝트")
        
        # 필터링 옵션
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        
        with col1:
            search_query = st.text_input(
                "검색",
                placeholder="프로젝트명, 태그...",
                label_visibility="collapsed"
            )
            
        with col2:
            status_filter = st.selectbox(
                "상태",
                ["전체", "진행중", "완료", "보관"],
                label_visibility="collapsed"
            )
            
        with col3:
            sort_by = st.selectbox(
                "정렬",
                ["최근 수정", "이름순", "생성일순"],
                label_visibility="collapsed"
            )
            
        with col4:
            if st.button("🔄 새로고침", use_container_width=True):
                st.rerun()
        
        # 프로젝트 목록 로드
        projects = self._load_user_projects(
            search_query,
            status_filter,
            sort_by
        )
        
        if not projects:
            self.ui.render_empty_state(
                "아직 프로젝트가 없습니다",
                "🗂️"
            )
            if st.button("첫 프로젝트 만들기", type="primary"):
                st.session_state.current_tab = 3
                st.rerun()
        else:
            # 프로젝트 카드 렌더링
            for i in range(0, len(projects), 3):
                cols = st.columns(3)
                for j, col in enumerate(cols):
                    if i + j < len(projects):
                        with col:
                            self._render_project_card(projects[i + j])
    
    def _render_project_card(self, project: Dict):
        """프로젝트 카드 렌더링"""
        with st.container():
            st.markdown(
                f"""
                <div class="custom-card" style="height: 250px;">
                    <h4>{project['name']}</h4>
                    <p style="color: #666; font-size: 0.9em;">
                        {project['field']} | {project['status']}
                    </p>
                    <p style="margin: 10px 0;">
                        {project.get('description', 'No description')[:100]}...
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # 액션 버튼
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("열기", key=f"open_{project['id']}", use_container_width=True):
                    st.session_state.current_project = project['id']
                    st.session_state.current_page = 'experiment_design'
                    st.rerun()
                    
            with col2:
                if st.button("편집", key=f"edit_{project['id']}", use_container_width=True):
                    self._show_edit_dialog(project)
                    
            with col3:
                if st.button("공유", key=f"share_{project['id']}", use_container_width=True):
                    self._show_share_dialog(project)
    
    def _render_new_project(self):
        """새 프로젝트 생성"""
        st.subheader("새 프로젝트 만들기")
        
        # 생성 방법 선택
        creation_method = st.radio(
            "프로젝트 생성 방법",
            ["🚀 빠른 시작", "📋 템플릿 사용", "🎯 AI 추천", "⚙️ 고급 설정"],
            horizontal=True
        )
        
        if creation_method == "🚀 빠른 시작":
            self._render_quick_start()
        elif creation_method == "📋 템플릿 사용":
            self._render_template_selection()
        elif creation_method == "🎯 AI 추천":
            self._render_ai_guided_creation()
        else:
            self._render_advanced_creation()
    
    def _render_quick_start(self):
        """빠른 시작 - 간단한 프로젝트 생성"""
        with st.form("quick_start_form"):
            # 기본 정보
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input(
                    "프로젝트명 *",
                    placeholder="예: 신약 후보물질 스크리닝"
                )
                
                field = st.selectbox(
                    "연구 분야 *",
                    ["화학", "재료과학", "생명공학", "제약", "환경", "기타"]
                )
                
            with col2:
                module_category = st.selectbox(
                    "실험 유형",
                    self.module_registry.get_categories()
                )
                
                visibility = st.radio(
                    "공개 범위",
                    ["🔒 비공개", "👥 팀 공개", "🌍 전체 공개"],
                    index=0
                )
            
            description = st.text_area(
                "프로젝트 설명",
                placeholder="프로젝트의 목표와 배경을 간단히 설명해주세요",
                height=100
            )
            
            # AI 설명 상세도 설정 (필수 구현)
            st.markdown("### 🤖 AI 지원 설정")
            ai_detail_level = st.select_slider(
                "AI 설명 상세도",
                options=["간단히", "보통", "상세히", "매우 상세히"],
                value="보통",
                help="AI가 제공하는 설명의 상세도를 설정합니다. 언제든지 변경 가능합니다."
            )
            
            # 제출
            submitted = st.form_submit_button(
                "프로젝트 생성",
                type="primary",
                use_container_width=True
            )
            
            if submitted:
                if not name or not field:
                    st.error("필수 항목을 입력해주세요")
                else:
                    project_data = {
                        "name": name,
                        "field": field,
                        "description": description,
                        "module_id": self._get_default_module(module_category),
                        "visibility": visibility.split()[0],
                        "ai_detail_level": ai_detail_level,
                        "created_by": "quick_start"
                    }
                    
                    project_id = self.create_project(project_data)
                    if project_id:
                        st.success("프로젝트가 생성되었습니다!")
                        st.balloons()
                        
                        # 바로 실험 설계로 이동
                        if st.button("실험 설계 시작하기", type="primary"):
                            st.session_state.current_project = project_id
                            st.session_state.current_page = 'experiment_design'
                            st.rerun()
    
    def _render_ai_guided_creation(self):
        """AI 가이드 프로젝트 생성"""
        st.info("AI가 프로젝트 설정을 도와드립니다. 몇 가지 질문에 답해주세요.")
        
        # AI 대화형 인터페이스
        if 'ai_creation_state' not in st.session_state:
            st.session_state.ai_creation_state = {
                'step': 0,
                'responses': {},
                'recommendations': None
            }
        
        state = st.session_state.ai_creation_state
        
        # 단계별 질문
        questions = [
            "어떤 문제를 해결하고 싶으신가요?",
            "현재 가지고 있는 리소스는 무엇인가요? (장비, 재료, 시간 등)",
            "목표로 하는 결과는 무엇인가요?",
            "이전에 유사한 실험을 해보신 적이 있나요?"
        ]
        
        if state['step'] < len(questions):
            st.markdown(f"### 질문 {state['step'] + 1}/{len(questions)}")
            st.write(questions[state['step']])
            
            response = st.text_area(
                "답변",
                key=f"ai_response_{state['step']}",
                height=100
            )
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if state['step'] > 0:
                    if st.button("이전", use_container_width=True):
                        state['step'] -= 1
                        st.rerun()
                        
            with col2:
                if st.button("다음", type="primary", use_container_width=True):
                    if response:
                        state['responses'][state['step']] = response
                        state['step'] += 1
                        st.rerun()
                    else:
                        st.warning("답변을 입력해주세요")
        
        else:
            # AI 분석 및 추천
            if not state['recommendations']:
                with st.spinner("AI가 최적의 프로젝트 설정을 분석하고 있습니다..."):
                    recommendations = self._get_ai_project_recommendations(
                        state['responses']
                    )
                    state['recommendations'] = recommendations
            
            # 추천 결과 표시
            self._render_ai_recommendations(state['recommendations'])
            
            # 프로젝트 생성 버튼
            if st.button("추천 설정으로 프로젝트 생성", type="primary", use_container_width=True):
                project_id = self.create_project(state['recommendations']['project_data'])
                if project_id:
                    st.success("AI 추천 프로젝트가 생성되었습니다!")
                    st.session_state.ai_creation_state = None
                    st.session_state.current_project = project_id
                    st.rerun()
    
    def _get_ai_project_recommendations(self, responses: Dict) -> Dict:
        """AI 기반 프로젝트 추천"""
        # AI 프롬프트 구성
        prompt = f"""
        사용자가 새로운 실험 프로젝트를 만들려고 합니다.
        
        사용자 응답:
        1. 해결하려는 문제: {responses.get(0, '')}
        2. 보유 리소스: {responses.get(1, '')}
        3. 목표 결과: {responses.get(2, '')}
        4. 이전 경험: {responses.get(3, '')}
        
        다음을 추천해주세요:
        1. 프로젝트명
        2. 적합한 실험 모듈
        3. 주요 실험 요인
        4. 예상 실험 횟수
        5. 유사 프로젝트 사례
        6. 주의사항
        
        JSON 형식으로 응답해주세요.
        """
        
        # AI 호출 (상세 설명 포함)
        response = self.api.generate_structured_response(
            prompt,
            detail_level=st.session_state.get('ai_detail_level', 'normal'),
            include_reasoning=True
        )
        
        # 기본 추천 (오프라인 폴백)
        if not response:
            return self._get_default_recommendations(responses)
        
        return response
    
    def _render_ai_recommendations(self, recommendations: Dict):
        """AI 추천 결과 표시"""
        st.markdown("### 🤖 AI 추천 결과")
        
        # AI 설명 상세도 토글
        show_details = st.checkbox(
            "🔍 상세 설명 보기",
            value=st.session_state.get('show_ai_details', True),
            key="project_ai_details"
        )
        
        # 기본 추천 사항
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**추천 프로젝트명**")
            st.info(recommendations.get('project_name', 'AI 추천 프로젝트'))
            
            st.markdown("**추천 실험 모듈**")
            st.info(recommendations.get('module', '범용 실험 설계'))
            
        with col2:
            st.markdown("**예상 실험 횟수**")
            st.info(f"{recommendations.get('estimated_runs', 20)}회")
            
            st.markdown("**예상 소요 시간**")
            st.info(recommendations.get('estimated_duration', '2-4주'))
        
        # 상세 설명 (토글)
        if show_details:
            tabs = st.tabs([
                "추론 과정",
                "실험 요인",
                "유사 프로젝트",
                "주의사항"
            ])
            
            with tabs[0]:
                st.markdown("**AI 추론 과정**")
                reasoning = recommendations.get('reasoning', {})
                for step, explanation in reasoning.items():
                    st.write(f"• {step}: {explanation}")
            
            with tabs[1]:
                st.markdown("**추천 실험 요인**")
                factors = recommendations.get('factors', [])
                for factor in factors:
                    with st.expander(factor['name']):
                        st.write(f"**범위**: {factor['min']} - {factor['max']} {factor['unit']}")
                        st.write(f"**중요도**: {factor['importance']}")
                        st.write(f"**근거**: {factor['rationale']}")
            
            with tabs[2]:
                st.markdown("**유사 프로젝트 사례**")
                similar = recommendations.get('similar_projects', [])
                for proj in similar:
                    with st.expander(f"{proj['name']} (유사도: {proj['similarity']}%)"):
                        st.write(f"**분야**: {proj['field']}")
                        st.write(f"**결과**: {proj['outcome']}")
                        st.write(f"**배울 점**: {proj['lessons']}")
            
            with tabs[3]:
                st.markdown("**⚠️ 주의사항**")
                for warning in recommendations.get('warnings', []):
                    st.warning(warning)
                
                st.markdown("**💡 성공 팁**")
                for tip in recommendations.get('tips', []):
                    st.info(tip)
    
    def create_project(self, project_data: Dict) -> Optional[str]:
        """새 프로젝트 생성"""
        try:
            # 프로젝트 ID 생성
            project_id = f"proj_{uuid.uuid4().hex[:8]}"
            
            # 프로젝트 데이터 구성
            project = {
                "id": project_id,
                "user_id": self.current_user['id'],
                "name": project_data['name'],
                "description": project_data.get('description', ''),
                "field": project_data['field'],
                "module_id": project_data.get('module_id'),
                "status": "active",
                "visibility": project_data.get('visibility', '🔒'),
                "collaborators": json.dumps([{
                    "user_id": self.current_user['id'],
                    "role": "owner",
                    "joined_at": datetime.now().isoformat()
                }]),
                "settings": json.dumps({
                    "ai_detail_level": project_data.get('ai_detail_level', '보통'),
                    "notifications": True,
                    "auto_save": True
                }),
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
            
            # 데이터베이스에 저장
            self.db.create_project(project)
            
            # 알림 발송
            self.notifier.send(
                "프로젝트 생성",
                f"'{project['name']}' 프로젝트가 생성되었습니다.",
                "success"
            )
            
            # 활동 로그
            logger.info(f"Project created: {project_id} by user {self.current_user['id']}")
            
            return project_id
            
        except Exception as e:
            logger.error(f"Failed to create project: {e}")
            st.error(f"프로젝트 생성 실패: {str(e)}")
            return None
    
    def _load_user_projects(self, search_query: str = "", 
                           status_filter: str = "전체",
                           sort_by: str = "최근 수정") -> List[Dict]:
        """사용자 프로젝트 로드"""
        try:
            # 데이터베이스에서 프로젝트 조회
            projects = self.db.get_user_projects(
                user_id=self.current_user['id'],
                include_shared=False
            )
            
            # 필터링
            if search_query:
                projects = [
                    p for p in projects
                    if search_query.lower() in p['name'].lower() or
                       search_query.lower() in p.get('description', '').lower()
                ]
            
            if status_filter != "전체":
                status_map = {
                    "진행중": "active",
                    "완료": "completed",
                    "보관": "archived"
                }
                projects = [
                    p for p in projects
                    if p['status'] == status_map.get(status_filter, status_filter)
                ]
            
            # 정렬
            if sort_by == "최근 수정":
                projects.sort(key=lambda x: x['updated_at'], reverse=True)
            elif sort_by == "이름순":
                projects.sort(key=lambda x: x['name'])
            elif sort_by == "생성일순":
                projects.sort(key=lambda x: x['created_at'], reverse=True)
            
            return projects
            
        except Exception as e:
            logger.error(f"Failed to load projects: {e}")
            return []
    
    def _render_shared_projects(self):
        """공유된 프로젝트 목록"""
        st.subheader("공유된 프로젝트")
        
        # 공유된 프로젝트 로드
        shared_projects = self.db.get_shared_projects(self.current_user['id'])
        
        if not shared_projects:
            self.ui.render_empty_state(
                "공유된 프로젝트가 없습니다",
                "👥"
            )
        else:
            # 역할별 그룹화
            by_role = {"editor": [], "viewer": []}
            
            for project in shared_projects:
                # 협업자 정보 파싱
                collaborators = json.loads(project.get('collaborators', '[]'))
                user_role = None
                
                for collab in collaborators:
                    if collab['user_id'] == self.current_user['id']:
                        user_role = collab['role']
                        break
                
                if user_role in by_role:
                    by_role[user_role].append(project)
            
            # 편집 가능한 프로젝트
            if by_role['editor']:
                st.markdown("### ✏️ 편집 가능")
                for project in by_role['editor']:
                    self._render_shared_project_card(project, 'editor')
            
            # 보기만 가능한 프로젝트
            if by_role['viewer']:
                st.markdown("### 👁️ 보기 전용")
                for project in by_role['viewer']:
                    self._render_shared_project_card(project, 'viewer')
    
    def _render_shared_project_card(self, project: Dict, role: str):
        """공유 프로젝트 카드"""
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"**{project['name']}**")
                st.caption(f"소유자: {self._get_owner_name(project)}")
                
            with col2:
                st.write(f"역할: {role}")
                
            with col3:
                if st.button("열기", key=f"open_shared_{project['id']}"):
                    st.session_state.current_project = project['id']
                    st.session_state.current_page = 'experiment_design'
                    st.rerun()
    
    def _render_templates(self):
        """템플릿 목록"""
        st.subheader("프로젝트 템플릿")
        
        # 템플릿 카테고리
        categories = ["전체", "화학", "재료과학", "생명공학", "인기", "내 템플릿"]
        selected_category = st.selectbox(
            "카테고리",
            categories,
            label_visibility="collapsed"
        )
        
        # 템플릿 로드
        templates = self._load_templates(selected_category)
        
        if not templates:
            self.ui.render_empty_state(
                "템플릿이 없습니다",
                "📋"
            )
        else:
            # 템플릿 그리드
            for i in range(0, len(templates), 3):
                cols = st.columns(3)
                for j, col in enumerate(cols):
                    if i + j < len(templates):
                        with col:
                            self._render_template_card(templates[i + j])
    
    def _render_template_card(self, template: Dict):
        """템플릿 카드"""
        with st.container():
            st.markdown(
                f"""
                <div class="custom-card">
                    <h5>{template['name']}</h5>
                    <p style="font-size: 0.9em; color: #666;">
                        {template['category']} | ⭐ {template.get('rating', 0)}/5
                    </p>
                    <p>{template['description'][:100]}...</p>
                    <p style="font-size: 0.8em; color: #888;">
                        사용 {template.get('usage_count', 0)}회
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            if st.button(
                "이 템플릿 사용",
                key=f"use_template_{template['id']}",
                use_container_width=True
            ):
                self._create_from_template(template)
    
    def _create_from_template(self, template: Dict):
        """템플릿으로부터 프로젝트 생성"""
        with st.form("template_project_form"):
            st.markdown(f"### '{template['name']}' 템플릿으로 프로젝트 만들기")
            
            # 프로젝트명
            name = st.text_input(
                "프로젝트명 *",
                value=f"{template['name']} - 복사본"
            )
            
            # 설명 (템플릿 설명 포함)
            description = st.text_area(
                "프로젝트 설명",
                value=template.get('description', ''),
                height=100
            )
            
            # 커스터마이징 옵션
            st.markdown("### 커스터마이징")
            
            # 템플릿 설정 로드
            template_data = json.loads(template.get('data', '{}'))
            
            # 실험 요인 편집
            if 'factors' in template_data:
                st.write("**실험 요인**")
                edited_factors = []
                
                for factor in template_data['factors']:
                    with st.expander(factor['name']):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            min_val = st.number_input(
                                "최소값",
                                value=factor.get('min', 0),
                                key=f"min_{factor['name']}"
                            )
                            
                        with col2:
                            max_val = st.number_input(
                                "최대값",
                                value=factor.get('max', 100),
                                key=f"max_{factor['name']}"
                            )
                        
                        edited_factors.append({
                            **factor,
                            'min': min_val,
                            'max': max_val
                        })
            
            # 제출
            if st.form_submit_button("프로젝트 생성", type="primary"):
                project_data = {
                    "name": name,
                    "description": description,
                    "field": template.get('category', '기타'),
                    "module_id": template.get('module_id'),
                    "template_id": template['id'],
                    **template_data
                }
                
                project_id = self.create_project(project_data)
                if project_id:
                    st.success("템플릿으로부터 프로젝트가 생성되었습니다!")
                    
                    # 템플릿 사용 횟수 증가
                    self.db.increment_template_usage(template['id'])
    
    def _show_share_dialog(self, project: Dict):
        """프로젝트 공유 대화상자"""
        with st.expander("🔗 프로젝트 공유", expanded=True):
            st.markdown(f"### '{project['name']}' 공유 설정")
            
            # 현재 협업자 목록
            st.markdown("**현재 팀원**")
            collaborators = json.loads(project.get('collaborators', '[]'))
            
            for collab in collaborators:
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    user_info = self.db.get_user(collab['user_id'])
                    st.write(f"{user_info['name']} ({user_info['email']})")
                    
                with col2:
                    st.write(collab['role'])
                    
                with col3:
                    if collab['role'] != 'owner' and self._can_manage_collaborators(project):
                        if st.button("제거", key=f"remove_{collab['user_id']}"):
                            self._remove_collaborator(project['id'], collab['user_id'])
            
            # 새 협업자 초대
            if self._can_manage_collaborators(project):
                st.markdown("**팀원 초대**")
                
                with st.form("invite_form"):
                    emails = st.text_area(
                        "이메일 주소",
                        placeholder="한 줄에 하나씩 입력\nexample@email.com",
                        height=100
                    )
                    
                    role = st.selectbox(
                        "권한",
                        ["viewer", "editor"],
                        format_func=lambda x: {"viewer": "보기 전용", "editor": "편집 가능"}[x]
                    )
                    
                    if st.form_submit_button("초대 보내기"):
                        if emails:
                            email_list = [e.strip() for e in emails.split('\n') if e.strip()]
                            self._invite_collaborators(project['id'], email_list, role)
    
    def _invite_collaborators(self, project_id: str, emails: List[str], role: str):
        """협업자 초대"""
        invited = []
        failed = []
        
        for email in emails:
            try:
                # 사용자 조회
                user = self.db.get_user_by_email(email)
                
                if user:
                    # 기존 사용자 추가
                    success = self.db.add_collaborator(
                        project_id,
                        user['id'],
                        role
                    )
                    
                    if success:
                        invited.append(email)
                        
                        # 알림 발송
                        self.notifier.send_to_user(
                            user['id'],
                            "프로젝트 초대",
                            f"{self.current_user['name']}님이 프로젝트에 초대했습니다.",
                            "info"
                        )
                    else:
                        failed.append(f"{email} (이미 팀원)")
                else:
                    # 신규 사용자 - 초대 이메일 발송
                    # (이메일 시스템 구현 필요)
                    failed.append(f"{email} (미가입)")
                    
            except Exception as e:
                failed.append(f"{email} ({str(e)})")
        
        # 결과 표시
        if invited:
            st.success(f"{len(invited)}명을 초대했습니다: {', '.join(invited)}")
        
        if failed:
            st.warning(f"초대 실패: {', '.join(failed)}")
    
    def _get_default_module(self, category: str) -> str:
        """기본 모듈 ID 반환"""
        modules = self.module_registry.list_modules(category)
        if modules:
            return modules[0]['id']
        return "core.general_experiment"
    
    def _load_templates(self, category: str) -> List[Dict]:
        """템플릿 로드"""
        try:
            if category == "내 템플릿":
                return self.db.get_user_templates(self.current_user['id'])
            elif category == "인기":
                return self.db.get_popular_templates(limit=12)
            elif category != "전체":
                return self.db.get_templates_by_category(category)
            else:
                return self.db.get_all_templates(limit=12)
                
        except Exception as e:
            logger.error(f"Failed to load templates: {e}")
            return []
    
    def _get_owner_name(self, project: Dict) -> str:
        """프로젝트 소유자 이름 반환"""
        try:
            owner_info = self.db.get_user(project['user_id'])
            return owner_info.get('name', 'Unknown')
        except:
            return 'Unknown'
    
    def _can_manage_collaborators(self, project: Dict) -> bool:
        """협업자 관리 권한 확인"""
        collaborators = json.loads(project.get('collaborators', '[]'))
        
        for collab in collaborators:
            if collab['user_id'] == self.current_user['id']:
                return collab['role'] in ['owner', 'editor']
        
        return False
    
    def _show_edit_dialog(self, project: Dict):
        """프로젝트 편집 대화상자"""
        # 편집 권한 확인
        if not self._has_edit_permission(project):
            st.error("프로젝트를 편집할 권한이 없습니다.")
            return
        
        with st.expander("✏️ 프로젝트 편집", expanded=True):
            with st.form("edit_project_form"):
                # 기본 정보
                name = st.text_input("프로젝트명", value=project['name'])
                description = st.text_area(
                    "설명",
                    value=project.get('description', ''),
                    height=100
                )
                
                # 상태 변경
                status = st.selectbox(
                    "상태",
                    ["active", "completed", "archived"],
                    index=["active", "completed", "archived"].index(project['status']),
                    format_func=lambda x: {
                        "active": "진행중",
                        "completed": "완료",
                        "archived": "보관"
                    }[x]
                )
                
                # AI 설정
                settings = json.loads(project.get('settings', '{}'))
                ai_detail_level = st.select_slider(
                    "AI 설명 상세도",
                    options=["간단히", "보통", "상세히", "매우 상세히"],
                    value=settings.get('ai_detail_level', '보통')
                )
                
                # 저장
                if st.form_submit_button("변경사항 저장"):
                    updates = {
                        "name": name,
                        "description": description,
                        "status": status,
                        "settings": json.dumps({
                            **settings,
                            "ai_detail_level": ai_detail_level
                        }),
                        "updated_at": datetime.now()
                    }
                    
                    if self.db.update_project(project['id'], updates):
                        st.success("프로젝트가 업데이트되었습니다.")
                        st.rerun()
                    else:
                        st.error("업데이트 실패")
    
    def _has_edit_permission(self, project: Dict) -> bool:
        """편집 권한 확인"""
        if project['user_id'] == self.current_user['id']:
            return True
        
        collaborators = json.loads(project.get('collaborators', '[]'))
        for collab in collaborators:
            if collab['user_id'] == self.current_user['id']:
                return PERMISSION_LEVELS[collab['role']]['can_edit']
        
        return False
    
    def _get_default_recommendations(self, responses: Dict) -> Dict:
        """오프라인 기본 추천 (AI 사용 불가 시)"""
        # 응답 분석을 통한 기본 추천
        problem = responses.get(0, '').lower()
        
        # 키워드 기반 간단한 추천
        if any(word in problem for word in ['합성', '화학', '반응']):
            module = '화학합성'
            field = '화학'
        elif any(word in problem for word in ['재료', '물성', '강도']):
            module = '재료특성'
            field = '재료과학'
        elif any(word in problem for word in ['분석', '측정', '검출']):
            module = '분석실험'
            field = '분석화학'
        else:
            module = '범용실험'
            field = '기타'
        
        return {
            'project_name': f"{field} 최적화 프로젝트",
            'module': module,
            'field': field,
            'estimated_runs': 20,
            'estimated_duration': '2-4주',
            'factors': [
                {
                    'name': '온도',
                    'min': 20,
                    'max': 100,
                    'unit': '°C',
                    'importance': '높음',
                    'rationale': '대부분의 화학/재료 실험에서 중요'
                },
                {
                    'name': '시간',
                    'min': 30,
                    'max': 180,
                    'unit': '분',
                    'importance': '중간',
                    'rationale': '반응 완료도에 영향'
                }
            ],
            'similar_projects': [],
            'warnings': [
                '초기 실험은 넓은 범위로 시작하세요',
                '안전 규정을 반드시 준수하세요'
            ],
            'tips': [
                '중심점 반복실험으로 재현성 확인',
                '요인 간 상호작용 고려'
            ],
            'project_data': {
                'name': f"{field} 최적화 프로젝트",
                'field': field,
                'module_id': f"core.{module.lower()}",
                'description': f"{problem} 해결을 위한 실험 설계"
            }
        }

def render():
    """페이지 렌더링 함수"""
    manager = ProjectSetupManager()
    manager.render_page()

if __name__ == "__main__":
    render()
