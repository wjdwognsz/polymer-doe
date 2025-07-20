#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧬 Polymer DOE Platform - Main Application
================================================================================
Version: 2.0.0
Description: AI-powered polymer experiment design platform with multi-user support
Author: Polymer DOE Research Team
License: MIT
================================================================================
"""

# ==================== 표준 라이브러리 ====================
import streamlit as st
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import os
from pathlib import Path
import traceback
import time

# ==================== 로깅 설정 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log', mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== 앱 메타데이터 ====================
APP_VERSION = "2.0.0"
APP_NAME = "Polymer DOE Platform"
MIN_PASSWORD_LENGTH = 8
SESSION_TIMEOUT_MINUTES = 30

# ==================== 페이지 및 유틸리티 임포트 ====================
try:
    # 페이지 모듈
    from pages.auth_page import AuthPage
    from pages.dashboard_page import DashboardPage
    from pages.project_setup import ProjectSetupPage
    from pages.experiment_design import ExperimentDesignPage
    from pages.data_analysis import DataAnalysisPage
    from pages.literature_search import LiteratureSearchPage
    from pages.visualization import VisualizationPage
    from pages.collaboration import CollaborationPage
    
    # 유틸리티 모듈
    from utils.auth_manager import GoogleSheetsAuthManager
    from utils.sheets_manager import GoogleSheetsManager
    from utils.api_manager import APIManager
    from utils.common_ui import (
        setup_page_config, apply_custom_css, render_header, 
        render_footer, show_notification
    )
    from utils.notification_manager import NotificationManager
    from utils.data_processor import DataProcessor
    
    # 설정 모듈
    from config.app_config import APP_CONFIG, LEVEL_CONFIG
    from config.theme_config import THEME_CONFIG
    
    MODULES_LOADED = True
except ImportError as e:
    logger.error(f"모듈 임포트 실패: {e}")
    MODULES_LOADED = False

# ==================== 기본 세션 상태 스키마 ====================
DEFAULT_SESSION_STATE = {
    # 인증 정보
    'authenticated': False,
    'user': {
        'user_id': None,
        'email': None,
        'name': None,
        'organization': None,
        'level': 'beginner',
        'points': 0,
        'profile_image': None,
        'created_at': None,
        'last_login': None,
        'settings': {}
    },
    
    # 앱 상태
    'current_page': 'auth',
    'previous_page': None,
    'page_params': {},
    
    # 프로젝트 관련
    'current_project': None,
    'recent_projects': [],
    'shared_projects': [],
    
    # UI 설정
    'theme': 'light',
    'language': 'ko',
    'sidebar_state': 'expanded',
    
    # 알림
    'notifications': [],
    'unread_count': 0,
    
    # 캐시
    'cache': {},
    'last_sync': None,
    
    # 세션 관리
    'session_token': None,
    'last_activity': datetime.now(),
    'expires_at': datetime.now() + timedelta(minutes=SESSION_TIMEOUT_MINUTES)
}

# ==================== 메인 애플리케이션 클래스 ====================
class PolymerDOEPlatform:
    """
    메인 애플리케이션 클래스
    - 전체 앱의 라이프사이클 관리
    - 인증 상태 확인 및 라우팅
    - 세션 관리 및 권한 제어
    """
    
    def __init__(self):
        """초기화 메서드"""
        self.initialize_session_state()
        if MODULES_LOADED:
            self.setup_managers()
            self.load_configuration()
        
    def initialize_session_state(self):
        """세션 상태 초기화"""
        for key, value in DEFAULT_SESSION_STATE.items():
            if key not in st.session_state:
                st.session_state[key] = value
                
        # 페이지 설정 초기화
        if 'page_initialized' not in st.session_state:
            st.session_state.page_initialized = False
            
    def setup_managers(self):
        """매니저 인스턴스 설정"""
        try:
            self.auth_manager = GoogleSheetsAuthManager()
            self.sheets_manager = GoogleSheetsManager()
            self.api_manager = APIManager()
            self.notification_manager = NotificationManager()
            self.data_processor = DataProcessor()
        except Exception as e:
            logger.error(f"매니저 초기화 실패: {e}")
            st.error("시스템 초기화 중 오류가 발생했습니다.")
            
    def load_configuration(self):
        """설정 파일 로드"""
        try:
            # 앱 설정 로드
            st.session_state.app_config = APP_CONFIG
            st.session_state.level_config = LEVEL_CONFIG
            st.session_state.theme_config = THEME_CONFIG
            
            # 사용자 설정 적용
            if st.session_state.authenticated and st.session_state.user.get('settings'):
                user_settings = st.session_state.user['settings']
                st.session_state.theme = user_settings.get('theme', 'light')
                st.session_state.language = user_settings.get('language', 'ko')
                
        except Exception as e:
            logger.error(f"설정 로드 실패: {e}")
            
    def check_authentication(self) -> bool:
        """인증 상태 확인"""
        try:
            # 세션 토큰 확인
            if not st.session_state.authenticated:
                return False
                
            # 세션 만료 확인
            if datetime.now() > st.session_state.expires_at:
                self.logout()
                return False
                
            return True
            
        except Exception:
            return False
            
    def check_session_timeout(self):
        """세션 타임아웃 확인"""
        try:
            # 마지막 활동 시간 확인
            last_activity = st.session_state.last_activity
            time_since_activity = datetime.now() - last_activity
            
            # 경고 표시 (25분 경과)
            if time_since_activity.total_seconds() > (SESSION_TIMEOUT_MINUTES - 5) * 60:
                st.warning("세션이 곧 만료됩니다. 활동을 계속하시려면 페이지를 새로고침하세요.")
                
            # 세션 만료 (30분 경과)
            if time_since_activity.total_seconds() > SESSION_TIMEOUT_MINUTES * 60:
                self.logout()
                st.rerun()
                
        except Exception as e:
            logger.error(f"세션 타임아웃 체크 실패: {e}")
            
    def update_last_activity(self):
        """마지막 활동 시간 업데이트"""
        st.session_state.last_activity = datetime.now()
        
    def check_permission(self, resource: str, action: str) -> bool:
        """권한 확인 - 교육적 성장 중심으로 모든 레벨에서 모든 기능 사용 가능"""
        # 모든 사용자는 모든 기능에 접근 가능
        # 레벨은 단지 교육적 지원의 정도만 결정
        return st.session_state.authenticated
        
    def logout(self):
        """로그아웃 처리"""
        try:
            # 세션 초기화
            for key in list(st.session_state.keys()):
                del st.session_state[key]
                
            # 기본 세션 상태 재설정
            self.initialize_session_state()
            
            st.success("로그아웃되었습니다.")
            
        except Exception as e:
            logger.error(f"로그아웃 실패: {e}")
            
    def render_sidebar(self):
        """사이드바 렌더링"""
        with st.sidebar:
            # 로고 및 타이틀
            st.markdown(
                f"""
                <div style='text-align: center; padding: 1rem 0;'>
                    <h1 style='color: #FF6B6B; margin: 0;'>🧬 {APP_NAME}</h1>
                    <p style='color: #666; font-size: 0.9em; margin: 0;'>v{APP_VERSION}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.divider()
            
            # 인증된 사용자 정보
            if st.session_state.authenticated:
                self.render_user_profile_section()
                st.divider()
                self.render_navigation_menu()
            else:
                self.render_login_prompt()
                
            # 하단 정보
            st.divider()
            self.render_sidebar_footer()
            
    def render_user_profile_section(self):
        """사용자 프로필 섹션"""
        user = st.session_state.user
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # 프로필 이미지 또는 아바타
            st.markdown(
                f"""
                <div style='
                    width: 50px; 
                    height: 50px; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-weight: bold;
                    font-size: 1.2em;
                '>
                    {user['name'][0].upper() if user['name'] else 'U'}
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with col2:
            st.markdown(f"**{user['name']}**")
            st.caption(f"{user['organization']}")
            
            # 레벨 및 포인트 표시
            level_info = LEVEL_CONFIG['levels'][user['level']]
            st.markdown(
                f"""
                <div style='margin-top: 0.5rem;'>
                    <span style='
                        background: {level_info['color']}20;
                        color: {level_info['color']};
                        padding: 2px 8px;
                        border-radius: 12px;
                        font-size: 0.8em;
                    '>
                        {level_info['icon']} {level_info['name']}
                    </span>
                    <span style='margin-left: 0.5rem; color: #FFD93D;'>
                        ⭐ {user['points']}
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )
            
    def render_navigation_menu(self):
        """네비게이션 메뉴"""
        menu_items = {
            'dashboard': {'title': '대시보드', 'icon': '📊'},
            'project_setup': {'title': '프로젝트 설정', 'icon': '📝'},
            'experiment_design': {'title': '실험 설계', 'icon': '🧪'},
            'data_analysis': {'title': '데이터 분석', 'icon': '📈'},
            'literature_search': {'title': '문헌 검색', 'icon': '🔍'},
            'visualization': {'title': '시각화', 'icon': '📊'},
            'collaboration': {'title': '협업', 'icon': '👥'}
        }
        
        # 알림 카운트 가져오기
        unread_count = st.session_state.unread_count
        
        for key, item in menu_items.items():
            # 협업 메뉴에 알림 뱃지 추가
            if key == 'collaboration' and unread_count > 0:
                label = f"{item['icon']} {item['title']} 🔴 {unread_count}"
            else:
                label = f"{item['icon']} {item['title']}"
                
            if st.button(
                label, 
                use_container_width=True,
                type="primary" if st.session_state.current_page == key else "secondary",
                key=f"nav_{key}"
            ):
                st.session_state.current_page = key
                self.update_last_activity()
                st.rerun()
                
    def render_login_prompt(self):
        """로그인 프롬프트"""
        st.info("🔐 로그인하여 모든 기능을 이용하세요")
        
        if st.button("로그인 / 회원가입", use_container_width=True, type="primary"):
            st.session_state.current_page = 'auth'
            st.rerun()
            
    def render_sidebar_footer(self):
        """사이드바 하단"""
        st.caption(
            """
            [📚 도움말](/) | [🐛 버그 신고](/) | [💡 제안하기](/)
            
            © 2024 Polymer DOE Platform
            """
        )
        
    def render_page_router(self):
        """페이지 라우팅"""
        try:
            current_page = st.session_state.current_page
            
            # 인증 상태 확인
            if not st.session_state.authenticated and current_page != 'auth':
                current_page = 'auth'
                st.session_state.current_page = 'auth'
                
            # 페이지 렌더링
            if current_page == 'auth':
                auth_page = AuthPage(self.auth_manager)
                auth_page.render()
                
            elif current_page == 'dashboard':
                dashboard_page = DashboardPage(self.sheets_manager)
                dashboard_page.render()
                
            elif current_page == 'project_setup':
                project_page = ProjectSetupPage(self.sheets_manager, self.api_manager)
                project_page.render()
                
            elif current_page == 'experiment_design':
                experiment_page = ExperimentDesignPage(self.sheets_manager, self.api_manager)
                experiment_page.render()
                
            elif current_page == 'data_analysis':
                analysis_page = DataAnalysisPage(self.sheets_manager, self.data_processor)
                analysis_page.render()
                
            elif current_page == 'literature_search':
                literature_page = LiteratureSearchPage(self.api_manager)
                literature_page.render()
                
            elif current_page == 'visualization':
                viz_page = VisualizationPage(self.sheets_manager, self.data_processor)
                viz_page.render()
                
            elif current_page == 'collaboration':
                collab_page = CollaborationPage(self.sheets_manager, self.notification_manager)
                collab_page.render()
                
            else:
                st.error(f"페이지를 찾을 수 없습니다: {current_page}")
                
        except Exception as e:
            logger.error(f"페이지 라우팅 실패: {e}")
            st.error("페이지 로드 중 오류가 발생했습니다.")
            st.exception(e)
            
    def check_notifications(self):
        """새 알림 확인"""
        try:
            if st.session_state.authenticated:
                # 5초마다 알림 확인
                notifications = self.notification_manager.get_unread_notifications(
                    st.session_state.user['user_id']
                )
                
                st.session_state.notifications = notifications
                st.session_state.unread_count = len(notifications)
                
                # 새 알림이 있으면 토스트 메시지
                if notifications:
                    latest = notifications[0]
                    show_notification(
                        f"새 알림: {latest['title']}", 
                        type="info"
                    )
                    
        except Exception as e:
            logger.error(f"알림 확인 실패: {e}")
            
    def render_notification_panel(self):
        """알림 패널 (우측)"""
        if st.session_state.authenticated and st.session_state.notifications:
            with st.expander(f"🔔 알림 ({st.session_state.unread_count})", expanded=False):
                for notif in st.session_state.notifications[:5]:
                    with st.container():
                        st.markdown(f"**{notif['title']}**")
                        st.caption(f"{notif['message']}")
                        st.caption(f"_{notif['created_at']}_")
                        
                if st.button("모든 알림 보기"):
                    st.session_state.current_page = 'collaboration'
                    st.rerun()
                    
    def run(self):
        """앱 실행 메인 함수"""
        try:
            # 페이지 기본 설정
            if not st.session_state.page_initialized:
                setup_page_config(
                    title=f"{APP_NAME} v{APP_VERSION}",
                    icon="🧬",
                    layout="wide"
                )
                st.session_state.page_initialized = True
                
            # 테마 및 스타일 적용
            apply_custom_css(st.session_state.get('theme', 'light'))
            
            # 인증 상태 확인
            if self.check_authentication():
                self.check_session_timeout()
                self.update_last_activity()
                self.check_notifications()
                
            # 메인 레이아웃
            col1, col2, col3 = st.columns([1, 5, 1])
            
            # 사이드바 (좌측)
            self.render_sidebar()
            
            # 메인 컨텐츠 (중앙)
            with col2:
                # 헤더
                if st.session_state.authenticated:
                    render_header(
                        title=st.session_state.get('current_page', 'Dashboard').replace('_', ' ').title(),
                        user_name=st.session_state.user['name']
                    )
                    
                # 페이지 컨텐츠
                self.render_page_router()
                
                # 푸터
                render_footer()
                
            # 알림 패널 (우측)
            with col3:
                if st.session_state.authenticated:
                    self.render_notification_panel()
                    
        except Exception as e:
            logger.error(f"앱 실행 중 오류: {e}\n{traceback.format_exc()}")
            st.error(
                """
                ### 😵 앱 실행 중 오류가 발생했습니다
                
                관리자에게 다음 정보와 함께 문의해주세요:
                - 시간: {timestamp}
                - 오류 코드: {error_code}
                
                [🔄 새로고침](/)
                """.format(
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    error_code=str(e)[:50]
                )
            )

# ==================== 메인 실행 ====================
def main():
    """앱 진입점"""
    try:
        # 모듈 로드 확인
        if not MODULES_LOADED:
            st.error(
                """
                ### 🚨 필수 모듈을 불러올 수 없습니다
                
                다음 사항을 확인해주세요:
                1. 모든 의존성 패키지가 설치되었는지 확인
                2. pages/ 및 utils/ 폴더가 존재하는지 확인
                3. config/ 폴더에 설정 파일이 있는지 확인
                
                ```bash
                pip install -r requirements.txt
                ```
                """
            )
            return
            
        # 앱 인스턴스 생성 및 실행
        app = PolymerDOEPlatform()
        app.run()
        
    except Exception as e:
        logger.critical(f"치명적 오류: {e}\n{traceback.format_exc()}")
        st.error(
            """
            ### 💥 치명적 오류가 발생했습니다
            
            앱을 시작할 수 없습니다. 관리자에게 문의하세요.
            
            [📧 지원팀 문의](mailto:support@polymer-doe.com)
            """
        )

if __name__ == "__main__":
    main()
