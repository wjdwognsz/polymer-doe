"""
👥 collaboration.py - 실시간 협업 플랫폼
팀원들과 실시간으로 프로젝트를 공유하고 협업할 수 있는 통합 환경
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import uuid
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import time
from collections import defaultdict
import networkx as nx
from pyvis.network import Network
import tempfile
import os

# 내부 모듈 임포트
from utils.auth_manager import check_authentication, get_current_user
from utils.sheets_manager import GoogleSheetsManager
from utils.api_manager import APIManager
from utils.common_ui import render_header, show_success, show_error, show_info, show_warning
from utils.notification_manager import NotificationManager
from utils.data_processor import DataProcessor

class CollaborationHub:
    """협업 허브 관리 클래스"""
    
    def __init__(self):
        """초기화"""
        self.sheets = GoogleSheetsManager()
        self.api = APIManager()
        self.notifier = NotificationManager()
        self.current_user = get_current_user()
        self.project_id = st.session_state.get('current_project', {}).get('id')
        
        # 세션 상태 초기화
        self._initialize_session_state()
        
    def _initialize_session_state(self):
        """세션 상태 초기화"""
        if 'collaboration_tab' not in st.session_state:
            st.session_state.collaboration_tab = 0
        if 'selected_channel' not in st.session_state:
            st.session_state.selected_channel = 'general'
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = []
        if 'active_task' not in st.session_state:
            st.session_state.active_task = None
        if 'wiki_page' not in st.session_state:
            st.session_state.wiki_page = None
        if 'show_ai_details' not in st.session_state:
            st.session_state.show_ai_details = False
            
    def render_page(self):
        """협업 페이지 메인 렌더링"""
        render_header("👥 팀 협업 센터", 
                     "실시간으로 팀원들과 소통하고 협업하세요")
        
        if not self.project_id:
            show_warning("프로젝트를 먼저 선택하거나 생성해주세요.")
            return
            
        # 상단 메트릭
        self._render_team_metrics()
        
        # 메인 탭
        tabs = st.tabs([
            "📊 개요", 
            "💬 커뮤니케이션", 
            "📋 작업 관리",
            "📚 지식 공유", 
            "📁 파일 협업", 
            "🏆 성과 추적"
        ])
        
        with tabs[0]:
            self._render_overview_tab()
        with tabs[1]:
            self._render_communication_tab()
        with tabs[2]:
            self._render_task_management_tab()
        with tabs[3]:
            self._render_knowledge_sharing_tab()
        with tabs[4]:
            self._render_file_collaboration_tab()
        with tabs[5]:
            self._render_performance_tab()
            
    def _render_team_metrics(self):
        """팀 메트릭 표시"""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        team_stats = self._get_team_statistics()
        
        with col1:
            online_members = self._get_online_members()
            st.metric(
                "온라인 멤버",
                f"{len(online_members)}/{team_stats['total_members']}",
                f"{len(online_members) - team_stats.get('online_yesterday', 0)}명"
            )
            
        with col2:
            st.metric(
                "오늘의 활동",
                team_stats['activities_today'],
                f"{team_stats.get('activity_change', 0)}%"
            )
            
        with col3:
            st.metric(
                "진행중 작업",
                team_stats['active_tasks'],
                f"{team_stats.get('tasks_completed_today', 0)} 완료"
            )
            
        with col4:
            st.metric(
                "대기중 리뷰",
                team_stats['pending_reviews'],
                delta_color="inverse"
            )
            
        with col5:
            st.metric(
                "팀 건강도",
                f"{team_stats['team_health']:.1f}/10",
                f"{team_stats.get('health_change', 0):+.1f}"
            )
            
    def _render_overview_tab(self):
        """개요 탭"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📈 팀 활동 타임라인")
            self._render_activity_timeline()
            
            st.subheader("🎯 이번 주 목표")
            self._render_weekly_goals()
            
        with col2:
            st.subheader("👥 팀 멤버")
            self._render_team_members()
            
            st.subheader("🔔 최근 알림")
            self._render_recent_notifications()
            
    def _render_communication_tab(self):
        """커뮤니케이션 탭"""
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("채널")
            self._render_channel_list()
            
            if st.button("➕ 새 채널", use_container_width=True):
                self._show_create_channel_dialog()
                
        with col2:
            st.subheader(f"💬 {st.session_state.selected_channel}")
            self._render_chat_interface()
            
    def _render_task_management_tab(self):
        """작업 관리 탭"""
        task_view = st.radio(
            "보기 방식",
            ["📋 목록", "📊 간트 차트", "🎯 칸반 보드"],
            horizontal=True
        )
        
        if task_view == "📋 목록":
            self._render_task_list()
        elif task_view == "📊 간트 차트":
            self._render_gantt_chart()
        else:
            self._render_kanban_board()
            
        # 작업 생성 버튼
        if st.button("➕ 새 작업 생성", type="primary"):
            self._show_create_task_dialog()
            
    def _render_knowledge_sharing_tab(self):
        """지식 공유 탭"""
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("📚 카테고리")
            categories = self._get_wiki_categories()
            selected_category = st.selectbox(
                "카테고리 선택",
                categories,
                label_visibility="collapsed"
            )
            
            st.divider()
            
            if st.button("➕ 새 문서", use_container_width=True):
                self._show_create_wiki_dialog()
                
        with col2:
            if st.session_state.wiki_page:
                self._render_wiki_page(st.session_state.wiki_page)
            else:
                self._render_wiki_list(selected_category)
                
    def _render_file_collaboration_tab(self):
        """파일 협업 탭"""
        st.subheader("📁 공유 파일")
        
        # 파일 업로드
        uploaded_file = st.file_uploader(
            "파일 업로드",
            accept_multiple_files=False,
            help="팀과 공유할 파일을 업로드하세요"
        )
        
        if uploaded_file:
            self._handle_file_upload(uploaded_file)
            
        # 파일 목록
        self._render_shared_files()
        
    def _render_performance_tab(self):
        """성과 추적 탭"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 팀 성과 대시보드")
            self._render_performance_dashboard()
            
            st.subheader("🏆 이번 달 MVP")
            self._render_mvp_section()
            
        with col2:
            st.subheader("🎖️ 리더보드")
            self._render_leaderboard()
            
            st.subheader("🎯 업적")
            self._render_achievements()
            
    # === 팀 관리 메서드 ===
    
    def _get_team_members(self) -> List[Dict]:
        """팀 멤버 정보 조회"""
        try:
            members_data = self.sheets.read_data(
                'TeamMembers',
                filters={'project_id': self.project_id, 'status': 'active'}
            )
            
            # 사용자 정보 조인
            members = []
            for member in members_data:
                user_info = self.sheets.get_user(member['user_id'])
                if user_info:
                    members.append({
                        **member,
                        'name': user_info['name'],
                        'email': user_info['email'],
                        'avatar': user_info.get('avatar', '👤')
                    })
                    
            return members
            
        except Exception as e:
            st.error(f"팀 멤버 조회 실패: {str(e)}")
            return []
            
    def _get_online_members(self) -> List[Dict]:
        """온라인 멤버 조회"""
        members = self._get_team_members()
        online = []
        
        # 최근 5분 이내 활동한 멤버를 온라인으로 간주
        cutoff_time = datetime.now() - timedelta(minutes=5)
        
        for member in members:
            last_active = member.get('last_active')
            if last_active:
                if isinstance(last_active, str):
                    last_active = datetime.fromisoformat(last_active)
                if last_active > cutoff_time:
                    online.append(member)
                    
        return online
        
    def _render_team_members(self):
        """팀 멤버 목록 렌더링"""
        members = self._get_team_members()
        online_ids = [m['user_id'] for m in self._get_online_members()]
        
        for member in members:
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                # 온라인 상태 표시
                status = "🟢" if member['user_id'] in online_ids else "⚫"
                st.write(f"{status} {member['avatar']}")
                
            with col2:
                st.write(f"**{member['name']}**")
                st.caption(f"{member['role']} • {member['email']}")
                
            with col3:
                if st.button("💬", key=f"chat_{member['user_id']}"):
                    self._start_direct_message(member['user_id'])
                    
    # === 커뮤니케이션 메서드 ===
    
    def _render_channel_list(self):
        """채널 목록 렌더링"""
        channels = self._get_channels()
        
        for channel in channels:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if st.button(
                    f"#{channel['name']}",
                    key=f"channel_{channel['id']}",
                    use_container_width=True,
                    type="primary" if channel['id'] == st.session_state.selected_channel else "secondary"
                ):
                    st.session_state.selected_channel = channel['id']
                    st.rerun()
                    
            with col2:
                unread = channel.get('unread_count', 0)
                if unread > 0:
                    st.write(f"🔴 {unread}")
                    
    def _render_chat_interface(self):
        """채팅 인터페이스 렌더링"""
        # 메시지 표시 영역
        message_container = st.container()
        
        with message_container:
            messages = self._get_channel_messages(st.session_state.selected_channel)
            
            for msg in messages:
                self._render_message(msg)
                
        # 메시지 입력
        col1, col2 = st.columns([5, 1])
        
        with col1:
            message_input = st.text_input(
                "메시지 입력",
                key="message_input",
                placeholder="메시지를 입력하세요...",
                label_visibility="collapsed"
            )
            
        with col2:
            if st.button("전송", type="primary", use_container_width=True):
                if message_input:
                    self._send_message(message_input)
                    st.session_state.message_input = ""
                    st.rerun()
                    
    def _render_message(self, message: Dict):
        """개별 메시지 렌더링"""
        col1, col2 = st.columns([1, 9])
        
        with col1:
            st.write(message.get('avatar', '👤'))
            
        with col2:
            # 메시지 헤더
            col_name, col_time = st.columns([3, 1])
            with col_name:
                st.markdown(f"**{message['author_name']}**")
            with col_time:
                st.caption(self._format_time(message['created_at']))
                
            # 메시지 본문
            st.write(message['content'])
            
            # 반응 및 액션
            col_react, col_thread, col_more = st.columns([3, 1, 1])
            
            with col_react:
                # 이모지 반응
                reactions = message.get('reactions', {})
                reaction_text = ""
                for emoji, users in reactions.items():
                    reaction_text += f"{emoji} {len(users)} "
                if reaction_text:
                    st.caption(reaction_text)
                    
            with col_thread:
                replies = message.get('reply_count', 0)
                if replies > 0:
                    if st.button(f"💬 {replies}", key=f"thread_{message['id']}"):
                        self._show_thread(message['id'])
                        
            with col_more:
                if st.button("⋯", key=f"more_{message['id']}"):
                    self._show_message_menu(message['id'])
                    
    # === 작업 관리 메서드 ===
    
    def _render_task_list(self):
        """작업 목록 렌더링"""
        tasks = self._get_project_tasks()
        
        # 필터
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status_filter = st.selectbox(
                "상태",
                ["전체", "할 일", "진행중", "검토중", "완료"],
                key="task_status_filter"
            )
            
        with col2:
            assignee_filter = st.selectbox(
                "담당자",
                ["전체"] + [m['name'] for m in self._get_team_members()],
                key="task_assignee_filter"
            )
            
        with col3:
            priority_filter = st.selectbox(
                "우선순위",
                ["전체", "높음", "중간", "낮음"],
                key="task_priority_filter"
            )
            
        with col4:
            sort_by = st.selectbox(
                "정렬",
                ["마감일", "우선순위", "생성일", "업데이트"],
                key="task_sort_by"
            )
            
        # 작업 목록
        filtered_tasks = self._filter_tasks(tasks, status_filter, assignee_filter, priority_filter)
        sorted_tasks = self._sort_tasks(filtered_tasks, sort_by)
        
        for task in sorted_tasks:
            self._render_task_card(task)
            
    def _render_task_card(self, task: Dict):
        """작업 카드 렌더링"""
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
            
            with col1:
                # 제목과 설명
                st.markdown(f"### {task['title']}")
                st.caption(task['description'][:100] + "..." if len(task['description']) > 100 else task['description'])
                
            with col2:
                # 담당자와 마감일
                st.write(f"👤 {task.get('assignee_name', '미할당')}")
                if task.get('due_date'):
                    due_date = datetime.fromisoformat(task['due_date'])
                    days_left = (due_date - datetime.now()).days
                    color = "red" if days_left < 0 else "orange" if days_left < 3 else "green"
                    st.markdown(f"📅 <span style='color:{color}'>{due_date.strftime('%Y-%m-%d')} ({days_left}일)</span>", 
                               unsafe_allow_html=True)
                    
            with col3:
                # 상태와 우선순위
                status_emoji = {
                    'todo': '📋',
                    'in_progress': '🔄',
                    'review': '👀',
                    'completed': '✅'
                }.get(task['status'], '❓')
                
                priority_emoji = {
                    'high': '🔴',
                    'medium': '🟡',
                    'low': '🟢'
                }.get(task['priority'], '⚪')
                
                st.write(f"{status_emoji} {priority_emoji}")
                
            with col4:
                # 액션 버튼
                if st.button("상세", key=f"task_detail_{task['id']}"):
                    self._show_task_detail(task['id'])
                    
            st.divider()
            
    def _render_gantt_chart(self):
        """간트 차트 렌더링"""
        tasks = self._get_project_tasks()
        
        if not tasks:
            show_info("표시할 작업이 없습니다.")
            return
            
        # 간트 차트 데이터 준비
        gantt_data = []
        for task in tasks:
            if task.get('start_date') and task.get('due_date'):
                gantt_data.append({
                    'Task': task['title'],
                    'Start': task['start_date'],
                    'Finish': task['due_date'],
                    'Resource': task.get('assignee_name', '미할당'),
                    'Complete': task.get('progress', 0)
                })
                
        if gantt_data:
            df = pd.DataFrame(gantt_data)
            
            # Plotly 간트 차트
            fig = px.timeline(
                df, 
                x_start="Start", 
                x_end="Finish", 
                y="Task",
                color="Resource",
                hover_data=["Complete"],
                title="프로젝트 간트 차트"
            )
            
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(height=600)
            
            st.plotly_chart(fig, use_container_width=True)
            
    def _render_kanban_board(self):
        """칸반 보드 렌더링"""
        tasks = self._get_project_tasks()
        
        # 상태별 컬럼
        columns = {
            'todo': {'title': '📋 할 일', 'tasks': []},
            'in_progress': {'title': '🔄 진행중', 'tasks': []},
            'review': {'title': '👀 검토중', 'tasks': []},
            'completed': {'title': '✅ 완료', 'tasks': []}
        }
        
        # 작업 분류
        for task in tasks:
            status = task.get('status', 'todo')
            if status in columns:
                columns[status]['tasks'].append(task)
                
        # 칸반 보드 렌더링
        cols = st.columns(4)
        
        for idx, (status, column) in enumerate(columns.items()):
            with cols[idx]:
                st.subheader(column['title'])
                
                # 작업 카드
                for task in column['tasks']:
                    with st.container():
                        st.markdown(f"**{task['title']}**")
                        st.caption(f"👤 {task.get('assignee_name', '미할당')}")
                        
                        # 우선순위 표시
                        priority_colors = {
                            'high': '🔴',
                            'medium': '🟡', 
                            'low': '🟢'
                        }
                        st.write(priority_colors.get(task.get('priority', 'medium'), '⚪'))
                        
                        # 드래그 앤 드롭 시뮬레이션 (실제 구현은 JavaScript 필요)
                        if st.button("이동", key=f"move_{task['id']}"):
                            self._show_move_task_dialog(task['id'])
                            
                        st.divider()
                        
    # === 지식 공유 메서드 ===
    
    def _render_wiki_list(self, category: str):
        """위키 목록 렌더링"""
        wiki_pages = self._get_wiki_pages(category)
        
        # 검색
        search_term = st.text_input("🔍 문서 검색", key="wiki_search")
        
        if search_term:
            wiki_pages = [p for p in wiki_pages if search_term.lower() in p['title'].lower() 
                         or search_term.lower() in p.get('content', '').lower()]
            
        # 문서 목록
        for page in wiki_pages:
            col1, col2, col3 = st.columns([4, 1, 1])
            
            with col1:
                if st.button(
                    f"📄 {page['title']}",
                    key=f"wiki_{page['id']}",
                    use_container_width=True
                ):
                    st.session_state.wiki_page = page['id']
                    st.rerun()
                    
            with col2:
                st.caption(f"👁️ {page.get('views', 0)}")
                
            with col3:
                st.caption(f"👍 {len(page.get('likes', []))}")
                
    def _render_wiki_page(self, page_id: str):
        """위키 페이지 렌더링"""
        page = self._get_wiki_page(page_id)
        
        if not page:
            show_error("페이지를 찾을 수 없습니다.")
            return
            
        # 페이지 헤더
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.title(page['title'])
            st.caption(f"작성자: {page['author_name']} • "
                      f"최종 수정: {self._format_time(page['updated_at'])} • "
                      f"버전: {page['version']}")
                      
        with col2:
            if st.button("← 목록으로"):
                st.session_state.wiki_page = None
                st.rerun()
                
        st.divider()
        
        # 페이지 내용
        st.markdown(page['content'])
        
        # 첨부 파일
        if page.get('attachments'):
            st.subheader("📎 첨부 파일")
            for attachment in page['attachments']:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"📄 {attachment['name']}")
                with col2:
                    if st.button("다운로드", key=f"download_{attachment['id']}"):
                        self._download_attachment(attachment['id'])
                        
        # 관련 페이지
        if page.get('related_pages'):
            st.subheader("🔗 관련 문서")
            related = self._get_related_pages(page['related_pages'])
            for rel_page in related:
                if st.button(f"📄 {rel_page['title']}", key=f"related_{rel_page['id']}"):
                    st.session_state.wiki_page = rel_page['id']
                    st.rerun()
                    
        st.divider()
        
        # 액션 버튼
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            liked = self.current_user['id'] in page.get('likes', [])
            if st.button(f"{'💙' if liked else '🤍'} 좋아요 ({len(page.get('likes', []))})",
                        key="like_wiki"):
                self._toggle_wiki_like(page_id)
                st.rerun()
                
        with col2:
            if st.button("✏️ 편집", key="edit_wiki"):
                self._show_edit_wiki_dialog(page_id)
                
        with col3:
            if st.button("🔄 버전 기록", key="version_wiki"):
                self._show_version_history(page_id)
                
        with col4:
            if st.button("🗑️ 삭제", key="delete_wiki"):
                if st.confirm("정말 삭제하시겠습니까?"):
                    self._delete_wiki_page(page_id)
                    
    # === 파일 협업 메서드 ===
    
    def _render_shared_files(self):
        """공유 파일 목록 렌더링"""
        files = self._get_shared_files()
        
        # 필터
        col1, col2, col3 = st.columns(3)
        
        with col1:
            file_type = st.selectbox(
                "파일 유형",
                ["전체", "문서", "데이터", "이미지", "코드"],
                key="file_type_filter"
            )
            
        with col2:
            sort_by = st.selectbox(
                "정렬",
                ["최신순", "이름순", "크기순", "다운로드순"],
                key="file_sort_by"
            )
            
        with col3:
            view_mode = st.radio(
                "보기",
                ["목록", "그리드"],
                horizontal=True,
                key="file_view_mode"
            )
            
        # 파일 목록/그리드
        filtered_files = self._filter_files(files, file_type)
        sorted_files = self._sort_files(filtered_files, sort_by)
        
        if view_mode == "목록":
            self._render_file_list(sorted_files)
        else:
            self._render_file_grid(sorted_files)
            
    def _render_file_list(self, files: List[Dict]):
        """파일 목록 뷰 렌더링"""
        for file in files:
            col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
            
            with col1:
                # 파일 아이콘과 이름
                icon = self._get_file_icon(file['type'])
                locked = "🔒" if file.get('locked_by') else ""
                st.write(f"{icon} {locked} **{file['name']}**")
                
            with col2:
                # 크기
                st.caption(self._format_file_size(file['size']))
                
            with col3:
                # 수정일
                st.caption(self._format_time(file['modified_at']))
                
            with col4:
                # 버전
                st.caption(f"v{file['version']}")
                
            with col5:
                # 액션
                if st.button("⋯", key=f"file_menu_{file['id']}"):
                    self._show_file_menu(file['id'])
                    
            st.divider()
            
    # === 성과 추적 메서드 ===
    
    def _render_performance_dashboard(self):
        """성과 대시보드 렌더링"""
        # 기간 선택
        period = st.select_slider(
            "기간",
            options=["일일", "주간", "월간", "분기", "연간"],
            value="월간",
            key="performance_period"
        )
        
        # 성과 메트릭
        metrics = self._calculate_team_metrics(period)
        
        # 차트 표시
        fig = go.Figure()
        
        # 활동 추이
        fig.add_trace(go.Scatter(
            x=metrics['dates'],
            y=metrics['activities'],
            mode='lines+markers',
            name='활동',
            line=dict(color='blue', width=2)
        ))
        
        # 완료 작업
        fig.add_trace(go.Scatter(
            x=metrics['dates'],
            y=metrics['completed_tasks'],
            mode='lines+markers',
            name='완료 작업',
            line=dict(color='green', width=2)
        ))
        
        fig.update_layout(
            title="팀 성과 추이",
            xaxis_title="날짜",
            yaxis_title="수",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 개인별 기여도
        st.subheader("👥 개인별 기여도")
        contributions = self._calculate_contributions(period)
        
        fig2 = px.bar(
            contributions,
            x='member',
            y='score',
            color='type',
            title="멤버별 기여도 점수"
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
    def _render_leaderboard(self):
        """리더보드 렌더링"""
        leaderboard = self._get_leaderboard()
        
        for idx, member in enumerate(leaderboard[:10]):
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                # 순위
                if idx == 0:
                    st.write("🥇")
                elif idx == 1:
                    st.write("🥈")
                elif idx == 2:
                    st.write("🥉")
                else:
                    st.write(f"#{idx + 1}")
                    
            with col2:
                # 멤버 정보
                st.write(f"**{member['name']}**")
                st.caption(f"{member['role']} • {member['specialty']}")
                
            with col3:
                # 점수
                st.metric(
                    "점수",
                    member['total_score'],
                    f"+{member['score_change']}"
                )
                
    # === 헬퍼 메서드 ===
    
    def _get_team_statistics(self) -> Dict:
        """팀 통계 조회"""
        try:
            # 팀 멤버 수
            members = self._get_team_members()
            online_members = self._get_online_members()
            
            # 오늘의 활동
            today_activities = self.sheets.read_data(
                'Activities',
                filters={
                    'project_id': self.project_id,
                    'date': datetime.now().strftime('%Y-%m-%d')
                }
            )
            
            # 작업 통계
            tasks = self._get_project_tasks()
            active_tasks = len([t for t in tasks if t['status'] in ['todo', 'in_progress']])
            completed_today = len([t for t in tasks 
                                 if t.get('completed_at', '').startswith(datetime.now().strftime('%Y-%m-%d'))])
            
            # 리뷰 대기
            pending_reviews = len([t for t in tasks if t['status'] == 'review'])
            
            # 팀 건강도 계산
            health_score = self._calculate_team_health()
            
            return {
                'total_members': len(members),
                'online_members': len(online_members),
                'online_yesterday': 3,  # 예시 데이터
                'activities_today': len(today_activities),
                'activity_change': 15,  # 예시 데이터
                'active_tasks': active_tasks,
                'tasks_completed_today': completed_today,
                'pending_reviews': pending_reviews,
                'team_health': health_score,
                'health_change': 0.5  # 예시 데이터
            }
            
        except Exception as e:
            logger.error(f"팀 통계 조회 실패: {e}")
            return {
                'total_members': 0,
                'online_members': 0,
                'online_yesterday': 0,
                'activities_today': 0,
                'activity_change': 0,
                'active_tasks': 0,
                'tasks_completed_today': 0,
                'pending_reviews': 0,
                'team_health': 5.0,
                'health_change': 0
            }
            
    def _calculate_team_health(self) -> float:
        """팀 건강도 계산"""
        # 여러 요소를 종합하여 건강도 점수 계산
        factors = {
            'communication': 8.5,  # 커뮤니케이션 빈도
            'task_completion': 7.8,  # 작업 완료율
            'collaboration': 8.2,  # 협업 지수
            'knowledge_sharing': 7.5,  # 지식 공유
            'response_time': 8.0  # 응답 시간
        }
        
        return sum(factors.values()) / len(factors)
        
    def _get_channels(self) -> List[Dict]:
        """채널 목록 조회"""
        return [
            {'id': 'general', 'name': 'general', 'unread_count': 0},
            {'id': 'experiment', 'name': '실험', 'unread_count': 2},
            {'id': 'analysis', 'name': '분석', 'unread_count': 0},
            {'id': 'paper', 'name': '논문', 'unread_count': 1}
        ]
        
    def _get_channel_messages(self, channel_id: str) -> List[Dict]:
        """채널 메시지 조회"""
        try:
            messages = self.sheets.read_data(
                'Messages',
                filters={
                    'project_id': self.project_id,
                    'channel_id': channel_id
                }
            )
            
            # 사용자 정보 추가
            for msg in messages:
                user_info = self.sheets.get_user(msg['author_id'])
                if user_info:
                    msg['author_name'] = user_info['name']
                    msg['avatar'] = user_info.get('avatar', '👤')
                    
            return sorted(messages, key=lambda x: x['created_at'])
            
        except Exception as e:
            logger.error(f"메시지 조회 실패: {e}")
            return []
            
    def _send_message(self, content: str):
        """메시지 전송"""
        try:
            message_data = {
                'id': str(uuid.uuid4()),
                'project_id': self.project_id,
                'channel_id': st.session_state.selected_channel,
                'author_id': self.current_user['id'],
                'content': content,
                'type': 'text',
                'created_at': datetime.now().isoformat(),
                'reactions': {},
                'thread_id': None
            }
            
            self.sheets.create_data('Messages', message_data)
            
            # 알림 발송
            self.notifier.send_notification(
                f"#{st.session_state.selected_channel}",
                f"{self.current_user['name']}: {content[:50]}...",
                'message'
            )
            
            show_success("메시지가 전송되었습니다.")
            
        except Exception as e:
            show_error(f"메시지 전송 실패: {str(e)}")
            
    def _get_project_tasks(self) -> List[Dict]:
        """프로젝트 작업 목록 조회"""
        try:
            tasks = self.sheets.read_data(
                'Tasks',
                filters={'project_id': self.project_id}
            )
            
            # 담당자 정보 추가
            for task in tasks:
                if task.get('assignee_id'):
                    user_info = self.sheets.get_user(task['assignee_id'])
                    if user_info:
                        task['assignee_name'] = user_info['name']
                        
            return tasks
            
        except Exception as e:
            logger.error(f"작업 조회 실패: {e}")
            return []
            
    def _filter_tasks(self, tasks: List[Dict], status: str, assignee: str, priority: str) -> List[Dict]:
        """작업 필터링"""
        filtered = tasks
        
        if status != "전체":
            status_map = {
                "할 일": "todo",
                "진행중": "in_progress",
                "검토중": "review",
                "완료": "completed"
            }
            filtered = [t for t in filtered if t['status'] == status_map.get(status)]
            
        if assignee != "전체":
            filtered = [t for t in filtered if t.get('assignee_name') == assignee]
            
        if priority != "전체":
            priority_map = {
                "높음": "high",
                "중간": "medium",
                "낮음": "low"
            }
            filtered = [t for t in filtered if t.get('priority') == priority_map.get(priority)]
            
        return filtered
        
    def _sort_tasks(self, tasks: List[Dict], sort_by: str) -> List[Dict]:
        """작업 정렬"""
        if sort_by == "마감일":
            return sorted(tasks, key=lambda x: x.get('due_date', '9999-12-31'))
        elif sort_by == "우선순위":
            priority_order = {'high': 0, 'medium': 1, 'low': 2}
            return sorted(tasks, key=lambda x: priority_order.get(x.get('priority', 'medium'), 1))
        elif sort_by == "생성일":
            return sorted(tasks, key=lambda x: x.get('created_at', ''), reverse=True)
        else:  # 업데이트
            return sorted(tasks, key=lambda x: x.get('updated_at', ''), reverse=True)
            
    def _format_time(self, timestamp: str) -> str:
        """시간 포맷팅"""
        try:
            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp)
            else:
                dt = timestamp
                
            now = datetime.now()
            diff = now - dt
            
            if diff.days > 0:
                return dt.strftime('%Y-%m-%d')
            elif diff.seconds > 3600:
                return f"{diff.seconds // 3600}시간 전"
            elif diff.seconds > 60:
                return f"{diff.seconds // 60}분 전"
            else:
                return "방금 전"
                
        except:
            return timestamp
            
    def _format_file_size(self, size: int) -> str:
        """파일 크기 포맷팅"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
        
    def _get_file_icon(self, file_type: str) -> str:
        """파일 아이콘 반환"""
        icons = {
            'document': '📄',
            'spreadsheet': '📊',
            'presentation': '📑',
            'image': '🖼️',
            'video': '🎥',
            'audio': '🎵',
            'code': '💻',
            'archive': '📦',
            'pdf': '📕'
        }
        return icons.get(file_type, '📎')
        
    # === AI 관련 메서드 ===
    
    def _render_ai_response(self, response: Dict, response_type: str = "general"):
        """AI 응답 렌더링 (상세도 제어 포함)"""
        # 핵심 답변 (항상 표시)
        st.markdown(f"### 🤖 {response_type} AI 추천")
        st.write(response['main'])
        
        # 상세 설명 토글
        show_details = st.session_state.get('show_ai_details', False)
        
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("🔍 상세", key=f"toggle_{response_type}"):
                st.session_state.show_ai_details = not show_details
                st.rerun()
                
        # 상세 설명 (조건부 표시)
        if show_details:
            tabs = st.tabs(["추론 과정", "대안", "배경", "신뢰도"])
            
            with tabs[0]:
                st.write(response.get('reasoning', '추론 과정이 없습니다.'))
                
            with tabs[1]:
                st.write(response.get('alternatives', '대안이 없습니다.'))
                
            with tabs[2]:
                st.write(response.get('background', '배경 정보가 없습니다.'))
                
            with tabs[3]:
                confidence = response.get('confidence', 0.8)
                st.progress(confidence)
                st.write(f"신뢰도: {confidence * 100:.1f}%")
                st.write(response.get('limitations', '제한사항이 없습니다.'))


def render():
    """페이지 렌더링 함수"""
    # 인증 확인
    if not check_authentication():
        st.error("로그인이 필요합니다.")
        st.stop()
        
    # 협업 허브 초기화 및 렌더링
    hub = CollaborationHub()
    hub.render_page()


if __name__ == "__main__":
    render()
