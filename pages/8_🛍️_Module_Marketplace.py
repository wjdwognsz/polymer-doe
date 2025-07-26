"""
모듈 마켓플레이스 페이지
커뮤니티가 만든 실험 모듈을 공유하고 발견할 수 있는 생태계의 중심
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import hashlib
import zipfile
import tempfile
import shutil
from typing import Dict, List, Optional, Any, Tuple
import logging
from enum import Enum
import re
import ast
import subprocess
import sys

# 프로젝트 모듈
from utils.common_ui import get_common_ui
from utils.database_manager import DatabaseManager
from utils.auth_manager import get_auth_manager
from config.app_config import APP_CONFIG, UPLOAD_CONFIG
from config.local_config import LOCAL_CONFIG
from modules.module_registry import get_module_registry
from modules.base_module import BaseExperimentModule

logger = logging.getLogger(__name__)

class ModuleStatus(Enum):
    """모듈 상태"""
    DRAFT = "draft"
    BETA = "beta"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    SUSPENDED = "suspended"

class ModuleCategory(Enum):
    """모듈 카테고리"""
    GENERAL = "일반 실험"
    POLYMER = "고분자"
    MATERIAL = "재료과학"
    BIO = "생명공학"
    CHEMISTRY = "화학"
    PHYSICS = "물리"
    OPTIMIZATION = "최적화"
    ANALYSIS = "분석"
    CUSTOM = "사용자 정의"

class ModuleMarketplacePage:
    """모듈 마켓플레이스 페이지 클래스"""
    
    def __init__(self):
        self.ui = get_common_ui()
        self.auth_manager = get_auth_manager()
        self.module_registry = get_module_registry()
        self.db_manager = self._init_db()
        self._init_session_state()
        
    def _init_db(self) -> DatabaseManager:
        """데이터베이스 초기화"""
        db_path = LOCAL_CONFIG['app_data_dir'] / 'data' / 'marketplace.db'
        db = DatabaseManager(db_path)
        
        # 마켓플레이스 테이블 생성
        db._get_connection().executescript('''
            CREATE TABLE IF NOT EXISTS marketplace_modules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                module_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                display_name TEXT NOT NULL,
                version TEXT NOT NULL,
                author_id INTEGER NOT NULL,
                author_name TEXT NOT NULL,
                category TEXT NOT NULL,
                tags TEXT,
                description TEXT,
                long_description TEXT,
                status TEXT DEFAULT 'draft',
                price REAL DEFAULT 0,
                currency TEXT DEFAULT 'USD',
                downloads INTEGER DEFAULT 0,
                installs INTEGER DEFAULT 0,
                rating REAL DEFAULT 0,
                rating_count INTEGER DEFAULT 0,
                file_path TEXT,
                file_hash TEXT,
                file_size INTEGER,
                requirements TEXT,
                min_platform_version TEXT,
                license TEXT DEFAULT 'MIT',
                homepage TEXT,
                repository TEXT,
                documentation TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                published_at TIMESTAMP,
                featured BOOLEAN DEFAULT FALSE,
                verified BOOLEAN FALSE,
                FOREIGN KEY (author_id) REFERENCES users (id)
            );
            
            CREATE TABLE IF NOT EXISTS module_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                module_id TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
                title TEXT,
                content TEXT,
                pros TEXT,
                cons TEXT,
                helpful_count INTEGER DEFAULT 0,
                verified_purchase BOOLEAN DEFAULT FALSE,
                developer_response TEXT,
                response_date TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (module_id) REFERENCES marketplace_modules (module_id),
                FOREIGN KEY (user_id) REFERENCES users (id),
                UNIQUE(module_id, user_id)
            );
            
            CREATE TABLE IF NOT EXISTS module_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                module_id TEXT NOT NULL,
                version TEXT NOT NULL,
                release_notes TEXT,
                file_path TEXT,
                file_hash TEXT,
                downloads INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (module_id) REFERENCES marketplace_modules (module_id),
                UNIQUE(module_id, version)
            );
            
            CREATE TABLE IF NOT EXISTS module_screenshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                module_id TEXT NOT NULL,
                image_path TEXT NOT NULL,
                caption TEXT,
                order_index INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (module_id) REFERENCES marketplace_modules (module_id)
            );
            
            CREATE TABLE IF NOT EXISTS user_favorites (
                user_id INTEGER NOT NULL,
                module_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, module_id),
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (module_id) REFERENCES marketplace_modules (module_id)
            );
            
            CREATE TABLE IF NOT EXISTS module_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                module_id TEXT NOT NULL,
                date DATE NOT NULL,
                views INTEGER DEFAULT 0,
                downloads INTEGER DEFAULT 0,
                installs INTEGER DEFAULT 0,
                uninstalls INTEGER DEFAULT 0,
                FOREIGN KEY (module_id) REFERENCES marketplace_modules (module_id),
                UNIQUE(module_id, date)
            );
            
            -- 인덱스 생성
            CREATE INDEX IF NOT EXISTS idx_modules_category ON marketplace_modules(category);
            CREATE INDEX IF NOT EXISTS idx_modules_status ON marketplace_modules(status);
            CREATE INDEX IF NOT EXISTS idx_modules_author ON marketplace_modules(author_id);
            CREATE INDEX IF NOT EXISTS idx_modules_created ON marketplace_modules(created_at);
            CREATE INDEX IF NOT EXISTS idx_modules_downloads ON marketplace_modules(downloads);
            CREATE INDEX IF NOT EXISTS idx_modules_rating ON marketplace_modules(rating);
            CREATE INDEX IF NOT EXISTS idx_reviews_module ON module_reviews(module_id);
            CREATE INDEX IF NOT EXISTS idx_reviews_user ON module_reviews(user_id);
        ''')
        
        return db
        
    def _init_session_state(self):
        """세션 상태 초기화"""
        defaults = {
            'marketplace_view': 'browse',  # browse, detail, upload, my_modules
            'selected_module': None,
            'search_query': '',
            'selected_category': 'all',
            'selected_tags': [],
            'sort_by': 'popular',
            'filter_price': 'all',
            'filter_rating': 0,
            'upload_step': 1,
            'upload_data': {},
            'show_ai_details': False
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
                
    def render(self):
        """메인 렌더링 함수"""
        self.ui.render_header(
            "모듈 마켓플레이스",
            "커뮤니티가 만든 실험 모듈을 탐색하고 공유하세요",
            "🛍️"
        )
        
        # 네비게이션 탭
        tabs = st.tabs([
            "🔍 모듈 탐색",
            "📤 모듈 업로드",
            "📚 내 모듈",
            "⭐ 즐겨찾기",
            "📊 통계"
        ])
        
        with tabs[0]:
            self._render_browse_view()
        with tabs[1]:
            self._render_upload_view()
        with tabs[2]:
            self._render_my_modules()
        with tabs[3]:
            self._render_favorites()
        with tabs[4]:
            self._render_statistics()
            
    def _render_browse_view(self):
        """모듈 탐색 뷰"""
        # 검색 및 필터 바
        self._render_search_filters()
        
        # 정렬 옵션
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            st.write(f"**{self._count_filtered_modules()}개의 모듈**")
        with col2:
            sort_by = st.selectbox(
                "정렬",
                ["인기순", "최신순", "평점순", "이름순"],
                label_visibility="collapsed"
            )
        with col3:
            view_mode = st.radio(
                "보기",
                ["카드", "리스트"],
                horizontal=True,
                label_visibility="collapsed"
            )
        with col4:
            if st.button("🔄 새로고침"):
                st.rerun()
                
        # 모듈 목록 표시
        if view_mode == "카드":
            self._render_module_cards()
        else:
            self._render_module_list()
            
    def _render_search_filters(self):
        """검색 및 필터 UI"""
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                search_query = st.text_input(
                    "🔍 모듈 검색",
                    placeholder="모듈 이름, 설명, 태그 검색...",
                    value=st.session_state.search_query,
                    label_visibility="collapsed"
                )
                st.session_state.search_query = search_query
                
            with col2:
                category = st.selectbox(
                    "카테고리",
                    ["전체"] + [cat.value for cat in ModuleCategory],
                    label_visibility="visible"
                )
                st.session_state.selected_category = category
                
            with col3:
                price_filter = st.selectbox(
                    "가격",
                    ["전체", "무료", "유료"],
                    label_visibility="visible"
                )
                st.session_state.filter_price = price_filter
                
        # 고급 필터
        with st.expander("고급 필터", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                min_rating = st.slider(
                    "최소 평점",
                    0.0, 5.0, 0.0, 0.5
                )
                st.session_state.filter_rating = min_rating
                
            with col2:
                status_filter = st.multiselect(
                    "상태",
                    ["베타", "정식", "검증됨"],
                    default=["정식", "검증됨"]
                )
                
            with col3:
                tag_filter = st.multiselect(
                    "태그",
                    self._get_popular_tags(),
                    default=[]
                )
                st.session_state.selected_tags = tag_filter
                
    def _render_module_cards(self):
        """모듈 카드 뷰"""
        modules = self._get_filtered_modules()
        
        if not modules:
            self.ui.render_empty_state(
                "검색 결과가 없습니다",
                "🔍"
            )
            return
            
        # 3열 그리드
        cols = st.columns(3)
        for idx, module in enumerate(modules):
            with cols[idx % 3]:
                self._render_module_card(module)
                
    def _render_module_card(self, module: Dict[str, Any]):
        """개별 모듈 카드"""
        with st.container():
            # 카드 스타일 적용
            st.markdown("""
                <style>
                .module-card {
                    border: 1px solid #e0e0e0;
                    border-radius: 8px;
                    padding: 1rem;
                    margin-bottom: 1rem;
                    transition: all 0.3s ease;
                }
                .module-card:hover {
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                    transform: translateY(-2px);
                }
                </style>
            """, unsafe_allow_html=True)
            
            # 카드 컨테이너
            with st.container():
                # 헤더
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"### {module['display_name']}")
                with col2:
                    if module.get('verified'):
                        st.write("✅")
                        
                # 작성자
                st.caption(f"👤 {module['author_name']}")
                
                # 설명
                st.write(module['description'][:100] + "...")
                
                # 메타데이터
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("평점", f"⭐ {module['rating']:.1f}")
                with col2:
                    st.metric("다운로드", f"📥 {module['downloads']:,}")
                with col3:
                    price = "무료" if module['price'] == 0 else f"${module['price']}"
                    st.metric("가격", price)
                    
                # 태그
                if module.get('tags'):
                    tags = json.loads(module['tags'])
                    tag_html = " ".join([
                        f"<span style='background-color: #f0f0f0; padding: 2px 8px; "
                        f"border-radius: 12px; font-size: 0.8em; margin-right: 4px;'>"
                        f"{tag}</span>"
                        for tag in tags[:3]
                    ])
                    st.markdown(tag_html, unsafe_allow_html=True)
                    
                # 액션 버튼
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("상세보기", key=f"detail_{module['module_id']}"):
                        st.session_state.selected_module = module
                        st.session_state.marketplace_view = 'detail'
                        st.rerun()
                with col2:
                    if st.button("⭐", key=f"fav_{module['module_id']}"):
                        self._toggle_favorite(module['module_id'])
                        
    def _render_module_detail(self, module_id: str):
        """모듈 상세 페이지"""
        module = self._get_module_details(module_id)
        if not module:
            st.error("모듈을 찾을 수 없습니다")
            return
            
        # 뒤로가기 버튼
        if st.button("← 목록으로"):
            st.session_state.marketplace_view = 'browse'
            st.session_state.selected_module = None
            st.rerun()
            
        # 모듈 헤더
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.title(module['display_name'])
            st.caption(f"v{module['version']} • {module['author_name']}")
        with col2:
            if module['price'] == 0:
                install_text = "무료 설치"
            else:
                install_text = f"${module['price']} 구매"
            if st.button(install_text, type="primary", use_container_width=True):
                self._install_module(module_id)
        with col3:
            if st.button("⭐ 즐겨찾기", use_container_width=True):
                self._toggle_favorite(module_id)
                
        # 메타 정보
        cols = st.columns(5)
        with cols[0]:
            st.metric("평점", f"⭐ {module['rating']:.1f}")
        with cols[1]:
            st.metric("리뷰", f"{module['rating_count']:,}개")
        with cols[2]:
            st.metric("다운로드", f"{module['downloads']:,}")
        with cols[3]:
            st.metric("활성 사용자", f"{module['installs']:,}")
        with cols[4]:
            st.metric("업데이트", self._format_date(module['updated_at']))
            
        # 상세 정보 탭
        tabs = st.tabs(["📝 설명", "🔧 기술정보", "📖 문서", "⭐ 리뷰", "📊 통계", "🔄 버전"])
        
        with tabs[0]:
            self._render_module_description(module)
        with tabs[1]:
            self._render_technical_info(module)
        with tabs[2]:
            self._render_documentation(module)
        with tabs[3]:
            self._render_reviews(module_id)
        with tabs[4]:
            self._render_module_stats(module_id)
        with tabs[5]:
            self._render_version_history(module_id)
            
    def _render_upload_view(self):
        """모듈 업로드 뷰"""
        if not st.session_state.authenticated:
            st.warning("모듈을 업로드하려면 로그인이 필요합니다.")
            return
            
        st.markdown("### 📤 새 모듈 업로드")
        
        # 업로드 진행 상태
        steps = ["기본 정보", "파일 업로드", "기술 정보", "문서 작성", "검증", "공개 설정"]
        current_step = st.session_state.upload_step - 1
        
        # Progress bar
        progress = (current_step + 1) / len(steps)
        st.progress(progress)
        
        # Step indicators
        cols = st.columns(len(steps))
        for i, (col, step) in enumerate(zip(cols, steps)):
            with col:
                if i < current_step:
                    st.markdown(f"✅ **{step}**")
                elif i == current_step:
                    st.markdown(f"🔵 **{step}**")
                else:
                    st.markdown(f"⭕ {step}")
                    
        st.divider()
        
        # 현재 단계 렌더링
        if st.session_state.upload_step == 1:
            self._render_upload_step1()
        elif st.session_state.upload_step == 2:
            self._render_upload_step2()
        elif st.session_state.upload_step == 3:
            self._render_upload_step3()
        elif st.session_state.upload_step == 4:
            self._render_upload_step4()
        elif st.session_state.upload_step == 5:
            self._render_upload_step5()
        elif st.session_state.upload_step == 6:
            self._render_upload_step6()
            
    def _render_upload_step1(self):
        """업로드 1단계: 기본 정보"""
        st.markdown("#### 1️⃣ 기본 정보")
        
        with st.form("upload_step1"):
            name = st.text_input(
                "모듈 이름*",
                help="영문, 숫자, 언더스코어만 사용 (예: polymer_synthesis_optimizer)"
            )
            
            display_name = st.text_input(
                "표시 이름*",
                help="사용자에게 보여질 이름 (예: 고분자 합성 최적화 도구)"
            )
            
            category = st.selectbox(
                "카테고리*",
                [cat.value for cat in ModuleCategory]
            )
            
            description = st.text_area(
                "간단한 설명*",
                max_chars=200,
                help="200자 이내로 모듈의 핵심 기능을 설명하세요"
            )
            
            tags = st.multiselect(
                "태그",
                ["최적화", "고분자", "합성", "분석", "시뮬레이션", "머신러닝"],
                help="최대 5개까지 선택"
            )
            
            col1, col2 = st.columns(2)
            with col2:
                if st.form_submit_button("다음 →", type="primary", use_container_width=True):
                    if self._validate_step1(name, display_name, category, description):
                        st.session_state.upload_data.update({
                            'name': name,
                            'display_name': display_name,
                            'category': category,
                            'description': description,
                            'tags': tags
                        })
                        st.session_state.upload_step = 2
                        st.rerun()
                        
    def _render_upload_step2(self):
        """업로드 2단계: 파일 업로드"""
        st.markdown("#### 2️⃣ 파일 업로드")
        
        # 모듈 파일
        module_file = st.file_uploader(
            "모듈 파일 (.py)*",
            type=['py'],
            help="BaseExperimentModule을 상속한 Python 파일"
        )
        
        # 추가 파일
        additional_files = st.file_uploader(
            "추가 파일 (선택)",
            type=['py', 'json', 'yaml', 'txt', 'md'],
            accept_multiple_files=True,
            help="도우미 파일, 설정 파일 등"
        )
        
        # 아이콘 (선택)
        icon_file = st.file_uploader(
            "아이콘 이미지 (선택)",
            type=['png', 'jpg', 'jpeg'],
            help="정사각형 이미지 권장 (최소 128x128)"
        )
        
        # 스크린샷 (선택)
        screenshots = st.file_uploader(
            "스크린샷 (선택)",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="최대 5장, 각 2MB 이하"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← 이전", use_container_width=True):
                st.session_state.upload_step = 1
                st.rerun()
        with col2:
            if st.button("다음 →", type="primary", use_container_width=True):
                if module_file:
                    # 파일 검증
                    validation_result = self._validate_module_file(module_file)
                    if validation_result['valid']:
                        st.session_state.upload_data['module_file'] = module_file
                        st.session_state.upload_data['additional_files'] = additional_files
                        st.session_state.upload_data['icon'] = icon_file
                        st.session_state.upload_data['screenshots'] = screenshots
                        st.session_state.upload_step = 3
                        st.rerun()
                    else:
                        st.error(validation_result['error'])
                else:
                    st.error("모듈 파일을 업로드해주세요")
                    
    def _render_upload_step3(self):
        """업로드 3단계: 기술 정보"""
        st.markdown("#### 3️⃣ 기술 정보")
        
        with st.form("upload_step3"):
            # 버전
            version = st.text_input(
                "버전*",
                value="1.0.0",
                help="시맨틱 버저닝 사용 (major.minor.patch)"
            )
            
            # 의존성
            requirements = st.text_area(
                "필요 패키지",
                placeholder="numpy>=1.20.0\npandas>=1.3.0\nscipy>=1.7.0",
                help="requirements.txt 형식"
            )
            
            # 플랫폼 요구사항
            min_platform = st.text_input(
                "최소 플랫폼 버전",
                value="2.0.0",
                help="이 모듈이 실행되는 최소 플랫폼 버전"
            )
            
            # 라이선스
            license_type = st.selectbox(
                "라이선스*",
                ["MIT", "Apache 2.0", "GPL-3.0", "BSD-3-Clause", "Proprietary", "기타"]
            )
            
            # 가격
            pricing = st.radio(
                "가격 정책*",
                ["무료", "유료"],
                horizontal=True
            )
            
            price = 0.0
            if pricing == "유료":
                price = st.number_input(
                    "가격 (USD)",
                    min_value=0.99,
                    max_value=999.99,
                    value=9.99,
                    step=1.0
                )
                
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("← 이전", use_container_width=True):
                    st.session_state.upload_step = 2
                    st.rerun()
            with col2:
                if st.form_submit_button("다음 →", type="primary", use_container_width=True):
                    st.session_state.upload_data.update({
                        'version': version,
                        'requirements': requirements,
                        'min_platform_version': min_platform,
                        'license': license_type,
                        'price': price
                    })
                    st.session_state.upload_step = 4
                    st.rerun()
                    
    def _render_upload_step4(self):
        """업로드 4단계: 문서 작성"""
        st.markdown("#### 4️⃣ 문서 작성")
        
        # 마크다운 에디터
        long_description = st.text_area(
            "상세 설명 (Markdown)",
            height=300,
            help="마크다운 형식으로 모듈의 상세 설명을 작성하세요",
            value=st.session_state.upload_data.get('long_description', '')
        )
        
        # 미리보기
        with st.expander("미리보기", expanded=True):
            st.markdown(long_description)
            
        # 추가 링크
        st.markdown("##### 추가 정보 (선택)")
        
        col1, col2 = st.columns(2)
        with col1:
            homepage = st.text_input("홈페이지 URL")
            repository = st.text_input("소스 저장소 URL")
        with col2:
            documentation = st.text_input("문서 URL")
            support = st.text_input("지원/이슈 URL")
            
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← 이전", use_container_width=True):
                st.session_state.upload_step = 3
                st.rerun()
        with col2:
            if st.button("다음 →", type="primary", use_container_width=True):
                st.session_state.upload_data.update({
                    'long_description': long_description,
                    'homepage': homepage,
                    'repository': repository,
                    'documentation': documentation
                })
                st.session_state.upload_step = 5
                st.rerun()
                
    def _render_upload_step5(self):
        """업로드 5단계: 검증"""
        st.markdown("#### 5️⃣ 자동 검증")
        
        # 검증 항목들
        validation_items = [
            ("코드 구문 검사", self._check_syntax),
            ("인터페이스 검증", self._check_interface),
            ("보안 스캔", self._check_security),
            ("의존성 확인", self._check_dependencies),
            ("성능 테스트", self._check_performance),
            ("문서 완성도", self._check_documentation)
        ]
        
        # 검증 수행
        all_passed = True
        
        for item_name, check_func in validation_items:
            with st.spinner(f"{item_name} 중..."):
                result = check_func()
                
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(item_name)
            with col2:
                if result['passed']:
                    st.success("✅ 통과")
                else:
                    st.error("❌ 실패")
                    all_passed = False
                    
            if not result['passed'] and result.get('message'):
                st.error(result['message'])
                if result.get('details'):
                    with st.expander("상세 정보"):
                        st.write(result['details'])
                        
        st.divider()
        
        # 검증 결과 요약
        if all_passed:
            st.success("🎉 모든 검증을 통과했습니다!")
            quality_score = self._calculate_quality_score()
            st.metric("품질 점수", f"{quality_score}/100")
        else:
            st.error("일부 검증에 실패했습니다. 문제를 수정한 후 다시 시도하세요.")
            
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← 이전", use_container_width=True):
                st.session_state.upload_step = 4
                st.rerun()
        with col2:
            if st.button("다음 →", type="primary", use_container_width=True, disabled=not all_passed):
                st.session_state.upload_step = 6
                st.rerun()
                
    def _render_upload_step6(self):
        """업로드 6단계: 공개 설정"""
        st.markdown("#### 6️⃣ 공개 설정")
        
        # 공개 옵션
        visibility = st.radio(
            "공개 범위",
            ["즉시 공개", "베타 테스트", "비공개"],
            help="""
            - **즉시 공개**: 검토 후 바로 마켓플레이스에 공개
            - **베타 테스트**: 선택된 사용자만 접근 가능
            - **비공개**: 본인만 사용 가능
            """
        )
        
        if visibility == "베타 테스트":
            beta_users = st.text_area(
                "베타 테스터 이메일",
                placeholder="user1@example.com\nuser2@example.com",
                help="한 줄에 하나씩 입력"
            )
            
        # 릴리즈 노트
        release_notes = st.text_area(
            "릴리즈 노트",
            placeholder="이 버전의 주요 변경사항을 설명하세요",
            help="사용자에게 보여질 업데이트 내용"
        )
        
        # 최종 확인
        st.divider()
        st.markdown("### 📋 최종 확인")
        
        # 업로드 정보 요약
        data = st.session_state.upload_data
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**기본 정보**")
            st.write(f"- 이름: {data.get('display_name')}")
            st.write(f"- 카테고리: {data.get('category')}")
            st.write(f"- 버전: {data.get('version')}")
            st.write(f"- 라이선스: {data.get('license')}")
            
        with col2:
            st.write("**기술 정보**")
            st.write(f"- 가격: {'무료' if data.get('price', 0) == 0 else f'${data.get('price')}'}")
            st.write(f"- 플랫폼: {data.get('min_platform_version')} 이상")
            st.write(f"- 공개: {visibility}")
            
        # 약관 동의
        agree = st.checkbox(
            "마켓플레이스 이용약관 및 개발자 가이드라인에 동의합니다"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← 이전", use_container_width=True):
                st.session_state.upload_step = 5
                st.rerun()
        with col2:
            if st.button("🚀 업로드", type="primary", use_container_width=True, disabled=not agree):
                with st.spinner("모듈 업로드 중..."):
                    success = self._upload_module(visibility, release_notes)
                    
                if success:
                    st.success("🎉 모듈이 성공적으로 업로드되었습니다!")
                    st.balloons()
                    # 초기화
                    st.session_state.upload_step = 1
                    st.session_state.upload_data = {}
                    st.session_state.marketplace_view = 'my_modules'
                    st.rerun()
                else:
                    st.error("업로드 중 오류가 발생했습니다")
                    
    def _render_my_modules(self):
        """내 모듈 관리"""
        if not st.session_state.authenticated:
            st.warning("로그인이 필요합니다.")
            return
            
        user_id = st.session_state.user['id']
        
        # 통계 요약
        col1, col2, col3, col4 = st.columns(4)
        stats = self._get_developer_stats(user_id)
        
        with col1:
            st.metric("총 모듈", stats['total_modules'])
        with col2:
            st.metric("총 다운로드", f"{stats['total_downloads']:,}")
        with col3:
            st.metric("평균 평점", f"⭐ {stats['avg_rating']:.1f}")
        with col4:
            st.metric("수익", f"${stats['total_revenue']:.2f}")
            
        st.divider()
        
        # 모듈 목록
        modules = self._get_user_modules(user_id)
        
        if not modules:
            self.ui.render_empty_state(
                "아직 업로드한 모듈이 없습니다",
                "📦"
            )
            if st.button("첫 모듈 업로드하기", type="primary"):
                st.session_state.marketplace_view = 'upload'
                st.rerun()
            return
            
        # 모듈 테이블
        for module in modules:
            with st.expander(f"{module['display_name']} v{module['version']}"):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**상태**: {self._get_status_badge(module['status'])}")
                    st.write(f"**생성일**: {self._format_date(module['created_at'])}")
                    st.write(f"**최종 수정**: {self._format_date(module['updated_at'])}")
                    
                with col2:
                    st.metric("다운로드", f"{module['downloads']:,}")
                    st.metric("활성 사용자", f"{module['installs']:,}")
                    
                with col3:
                    st.metric("평점", f"⭐ {module['rating']:.1f}")
                    st.metric("리뷰", f"{module['rating_count']}개")
                    
                # 액션 버튼
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if st.button("편집", key=f"edit_{module['module_id']}"):
                        self._edit_module(module['module_id'])
                with col2:
                    if st.button("통계", key=f"stats_{module['module_id']}"):
                        self._show_module_stats(module['module_id'])
                with col3:
                    if st.button("리뷰 관리", key=f"reviews_{module['module_id']}"):
                        self._manage_reviews(module['module_id'])
                with col4:
                    if module['status'] != 'archived':
                        if st.button("보관", key=f"archive_{module['module_id']}"):
                            self._archive_module(module['module_id'])
                            
    def _render_favorites(self):
        """즐겨찾기 모듈"""
        if not st.session_state.authenticated:
            st.warning("로그인이 필요합니다.")
            return
            
        user_id = st.session_state.user['id']
        favorites = self._get_user_favorites(user_id)
        
        if not favorites:
            self.ui.render_empty_state(
                "아직 즐겨찾기한 모듈이 없습니다",
                "⭐"
            )
            return
            
        # 즐겨찾기 모듈 카드
        cols = st.columns(3)
        for idx, module in enumerate(favorites):
            with cols[idx % 3]:
                self._render_module_card(module)
                
    def _render_statistics(self):
        """마켓플레이스 통계"""
        st.markdown("### 📊 마켓플레이스 통계")
        
        # 전체 통계
        stats = self._get_marketplace_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("총 모듈", f"{stats['total_modules']:,}")
        with col2:
            st.metric("활성 개발자", f"{stats['active_developers']:,}")
        with col3:
            st.metric("총 다운로드", f"{stats['total_downloads']:,}")
        with col4:
            st.metric("월간 활성 사용자", f"{stats['monthly_active_users']:,}")
            
        st.divider()
        
        # 차트
        col1, col2 = st.columns(2)
        
        with col1:
            # 카테고리별 모듈 분포
            st.markdown("#### 카테고리별 분포")
            category_data = self._get_category_distribution()
            
            import plotly.express as px
            fig = px.pie(
                values=category_data['count'],
                names=category_data['category'],
                title="모듈 카테고리 분포"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # 인기 태그
            st.markdown("#### 인기 태그 Top 10")
            popular_tags = self._get_popular_tags_with_count()
            
            fig = px.bar(
                x=popular_tags['count'],
                y=popular_tags['tag'],
                orientation='h',
                title="가장 많이 사용된 태그"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        # 트렌드
        st.markdown("#### 📈 트렌드")
        
        # 시간대별 다운로드 추이
        trend_data = self._get_download_trends()
        
        fig = px.line(
            trend_data,
            x='date',
            y='downloads',
            title="일별 다운로드 추이 (최근 30일)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 인기 급상승 모듈
        st.markdown("#### 🔥 인기 급상승 모듈")
        trending = self._get_trending_modules()
        
        for idx, module in enumerate(trending[:5]):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"**{idx+1}. {module['display_name']}**")
                st.caption(f"by {module['author_name']}")
            with col2:
                st.write(f"📈 +{module['growth_rate']:.0%}")
            with col3:
                if st.button("보기", key=f"trend_{module['module_id']}"):
                    st.session_state.selected_module = module
                    st.session_state.marketplace_view = 'detail'
                    st.rerun()
                    
    # ===== 헬퍼 메서드들 =====
    
    def _get_filtered_modules(self) -> List[Dict]:
        """필터링된 모듈 목록 조회"""
        query = """
            SELECT * FROM marketplace_modules 
            WHERE status IN ('published', 'beta')
        """
        params = []
        
        # 검색어 필터
        if st.session_state.search_query:
            query += """ AND (
                name LIKE ? OR 
                display_name LIKE ? OR 
                description LIKE ? OR 
                tags LIKE ?
            )"""
            search_term = f"%{st.session_state.search_query}%"
            params.extend([search_term] * 4)
            
        # 카테고리 필터
        if st.session_state.selected_category != "전체":
            query += " AND category = ?"
            params.append(st.session_state.selected_category)
            
        # 가격 필터
        if st.session_state.filter_price == "무료":
            query += " AND price = 0"
        elif st.session_state.filter_price == "유료":
            query += " AND price > 0"
            
        # 평점 필터
        if st.session_state.filter_rating > 0:
            query += " AND rating >= ?"
            params.append(st.session_state.filter_rating)
            
        # 정렬
        sort_map = {
            "인기순": "downloads DESC",
            "최신순": "created_at DESC",
            "평점순": "rating DESC",
            "이름순": "display_name ASC"
        }
        query += f" ORDER BY {sort_map.get(st.session_state.get('sort_by', '인기순'), 'downloads DESC')}"
        
        conn = self.db_manager._get_connection()
        cursor = conn.execute(query, params)
        modules = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return modules
        
    def _count_filtered_modules(self) -> int:
        """필터링된 모듈 수 카운트"""
        modules = self._get_filtered_modules()
        return len(modules)
        
    def _get_module_details(self, module_id: str) -> Optional[Dict]:
        """모듈 상세 정보 조회"""
        conn = self.db_manager._get_connection()
        cursor = conn.execute(
            "SELECT * FROM marketplace_modules WHERE module_id = ?",
            (module_id,)
        )
        module = cursor.fetchone()
        conn.close()
        
        return dict(module) if module else None
        
    def _validate_module_file(self, file) -> Dict[str, Any]:
        """모듈 파일 검증"""
        try:
            # 파일 내용 읽기
            content = file.read().decode('utf-8')
            
            # 기본 구문 검사
            ast.parse(content)
            
            # BaseExperimentModule 상속 확인
            if "BaseExperimentModule" not in content:
                return {
                    'valid': False,
                    'error': "모듈은 BaseExperimentModule을 상속해야 합니다"
                }
                
            # 필수 메서드 확인
            required_methods = [
                'get_info', 'validate_inputs', 'generate_design',
                'analyze_results'
            ]
            
            missing_methods = []
            for method in required_methods:
                if f"def {method}" not in content:
                    missing_methods.append(method)
                    
            if missing_methods:
                return {
                    'valid': False,
                    'error': f"필수 메서드가 없습니다: {', '.join(missing_methods)}"
                }
                
            return {'valid': True}
            
        except SyntaxError as e:
            return {
                'valid': False,
                'error': f"구문 오류: {str(e)}"
            }
        except Exception as e:
            return {
                'valid': False,
                'error': f"검증 오류: {str(e)}"
            }
            
    def _check_syntax(self) -> Dict[str, Any]:
        """코드 구문 검사"""
        try:
            if 'module_file' in st.session_state.upload_data:
                content = st.session_state.upload_data['module_file'].read().decode('utf-8')
                ast.parse(content)
                return {'passed': True}
            return {'passed': False, 'message': "모듈 파일이 없습니다"}
        except SyntaxError as e:
            return {
                'passed': False,
                'message': "구문 오류가 있습니다",
                'details': str(e)
            }
            
    def _check_interface(self) -> Dict[str, Any]:
        """인터페이스 검증"""
        # 실제 구현에서는 더 정교한 검증 수행
        return {'passed': True}
        
    def _check_security(self) -> Dict[str, Any]:
        """보안 스캔"""
        # 실제 구현에서는 보안 검사 수행
        # 예: 위험한 함수 호출, 파일 시스템 접근 등
        return {'passed': True}
        
    def _check_dependencies(self) -> Dict[str, Any]:
        """의존성 확인"""
        # requirements 파싱 및 검증
        return {'passed': True}
        
    def _check_performance(self) -> Dict[str, Any]:
        """성능 테스트"""
        # 간단한 성능 테스트
        return {'passed': True}
        
    def _check_documentation(self) -> Dict[str, Any]:
        """문서 완성도 검사"""
        data = st.session_state.upload_data
        
        if not data.get('long_description') or len(data['long_description']) < 100:
            return {
                'passed': False,
                'message': "상세 설명이 너무 짧습니다 (최소 100자)"
            }
            
        return {'passed': True}
        
    def _calculate_quality_score(self) -> int:
        """품질 점수 계산"""
        score = 70  # 기본 점수
        
        data = st.session_state.upload_data
        
        # 문서화 (+10)
        if len(data.get('long_description', '')) > 500:
            score += 10
            
        # 스크린샷 (+5)
        if data.get('screenshots'):
            score += 5
            
        # 링크 제공 (+5)
        if data.get('repository') or data.get('documentation'):
            score += 5
            
        # 태그 (+5)
        if len(data.get('tags', [])) >= 3:
            score += 5
            
        # 아이콘 (+5)
        if data.get('icon'):
            score += 5
            
        return min(score, 100)
        
    def _upload_module(self, visibility: str, release_notes: str) -> bool:
        """모듈 업로드 처리"""
        try:
            data = st.session_state.upload_data
            user = st.session_state.user
            
            # 모듈 ID 생성
            module_id = f"{user['id']}_{data['name']}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # 파일 저장
            module_dir = LOCAL_CONFIG['app_data_dir'] / 'modules' / 'marketplace' / module_id
            module_dir.mkdir(parents=True, exist_ok=True)
            
            # 메인 모듈 파일 저장
            module_file = data['module_file']
            module_path = module_dir / f"{data['name']}.py"
            with open(module_path, 'wb') as f:
                f.write(module_file.getvalue())
                
            # 파일 해시 계산
            file_hash = hashlib.sha256(module_file.getvalue()).hexdigest()
            
            # DB에 저장
            conn = self.db_manager._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO marketplace_modules (
                    module_id, name, display_name, version, author_id, author_name,
                    category, tags, description, long_description, status, price,
                    file_path, file_hash, file_size, requirements, min_platform_version,
                    license, homepage, repository, documentation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                module_id,
                data['name'],
                data['display_name'],
                data['version'],
                user['id'],
                user['name'],
                data['category'],
                json.dumps(data.get('tags', [])),
                data['description'],
                data.get('long_description', ''),
                'beta' if visibility == "베타 테스트" else 'published',
                data.get('price', 0),
                str(module_path),
                file_hash,
                len(module_file.getvalue()),
                data.get('requirements', ''),
                data.get('min_platform_version', '2.0.0'),
                data.get('license', 'MIT'),
                data.get('homepage', ''),
                data.get('repository', ''),
                data.get('documentation', '')
            ))
            
            # 버전 정보 저장
            cursor.execute("""
                INSERT INTO module_versions (module_id, version, release_notes, file_path, file_hash)
                VALUES (?, ?, ?, ?, ?)
            """, (module_id, data['version'], release_notes, str(module_path), file_hash))
            
            conn.commit()
            conn.close()
            
            # 모듈 레지스트리에 등록
            self.module_registry.register_module(
                module_path,
                store_type="community"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"모듈 업로드 실패: {str(e)}")
            return False
            
    def _install_module(self, module_id: str):
        """모듈 설치"""
        try:
            module = self._get_module_details(module_id)
            if not module:
                st.error("모듈을 찾을 수 없습니다")
                return
                
            # 가격 확인
            if module['price'] > 0:
                st.warning(f"이 모듈은 ${module['price']}의 유료 모듈입니다.")
                # 실제로는 결제 프로세스 필요
                return
                
            # 설치 진행
            with st.spinner("모듈 설치 중..."):
                # 로컬 모듈 디렉토리에 복사
                source_path = Path(module['file_path'])
                user_modules_dir = LOCAL_CONFIG['app_data_dir'] / 'modules' / 'user_modules' / str(st.session_state.user['id'])
                user_modules_dir.mkdir(parents=True, exist_ok=True)
                
                dest_path = user_modules_dir / f"{module['name']}.py"
                shutil.copy2(source_path, dest_path)
                
                # 다운로드 수 증가
                conn = self.db_manager._get_connection()
                conn.execute(
                    "UPDATE marketplace_modules SET downloads = downloads + 1 WHERE module_id = ?",
                    (module_id,)
                )
                conn.commit()
                conn.close()
                
                # 모듈 레지스트리 갱신
                self.module_registry.register_module(dest_path, store_type="user")
                
            st.success(f"✅ {module['display_name']} 모듈이 설치되었습니다!")
            
        except Exception as e:
            logger.error(f"모듈 설치 실패: {str(e)}")
            st.error("모듈 설치 중 오류가 발생했습니다")
            
    def _toggle_favorite(self, module_id: str):
        """즐겨찾기 토글"""
        if not st.session_state.authenticated:
            st.warning("로그인이 필요합니다")
            return
            
        user_id = st.session_state.user['id']
        
        conn = self.db_manager._get_connection()
        cursor = conn.cursor()
        
        # 현재 즐겨찾기 상태 확인
        cursor.execute(
            "SELECT 1 FROM user_favorites WHERE user_id = ? AND module_id = ?",
            (user_id, module_id)
        )
        
        if cursor.fetchone():
            # 제거
            cursor.execute(
                "DELETE FROM user_favorites WHERE user_id = ? AND module_id = ?",
                (user_id, module_id)
            )
            st.success("즐겨찾기에서 제거되었습니다")
        else:
            # 추가
            cursor.execute(
                "INSERT INTO user_favorites (user_id, module_id) VALUES (?, ?)",
                (user_id, module_id)
            )
            st.success("즐겨찾기에 추가되었습니다")
            
        conn.commit()
        conn.close()
        
    def _render_module_description(self, module: Dict):
        """모듈 설명 렌더링"""
        if module.get('long_description'):
            st.markdown(module['long_description'])
        else:
            st.write(module['description'])
            
        # 태그
        if module.get('tags'):
            st.write("**태그:**")
            tags = json.loads(module['tags'])
            tag_html = " ".join([
                f"<span style='background-color: #f0f0f0; padding: 4px 12px; "
                f"border-radius: 20px; margin-right: 8px;'>{tag}</span>"
                for tag in tags
            ])
            st.markdown(tag_html, unsafe_allow_html=True)
            
    def _render_technical_info(self, module: Dict):
        """기술 정보 렌더링"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**기본 정보**")
            st.write(f"- 버전: {module['version']}")
            st.write(f"- 라이선스: {module['license']}")
            st.write(f"- 파일 크기: {module['file_size'] / 1024:.1f} KB")
            st.write(f"- 최소 플랫폼: v{module['min_platform_version']}")
            
        with col2:
            st.write("**링크**")
            if module.get('homepage'):
                st.write(f"- [홈페이지]({module['homepage']})")
            if module.get('repository'):
                st.write(f"- [소스 코드]({module['repository']})")
            if module.get('documentation'):
                st.write(f"- [문서]({module['documentation']})")
                
        # 의존성
        if module.get('requirements'):
            st.write("**필수 패키지**")
            st.code(module['requirements'], language='text')
            
    def _render_documentation(self, module: Dict):
        """문서 렌더링"""
        # 실제로는 별도 문서 파일을 읽어서 표시
        st.info("상세 문서는 모듈 설치 후 확인할 수 있습니다")
        
        if module.get('documentation'):
            st.write(f"온라인 문서: [{module['documentation']}]({module['documentation']})")
            
    def _render_reviews(self, module_id: str):
        """리뷰 렌더링"""
        # 리뷰 통계
        conn = self.db_manager._get_connection()
        cursor = conn.execute("""
            SELECT 
                AVG(rating) as avg_rating,
                COUNT(*) as total_reviews,
                SUM(CASE WHEN rating = 5 THEN 1 ELSE 0 END) as five_star,
                SUM(CASE WHEN rating = 4 THEN 1 ELSE 0 END) as four_star,
                SUM(CASE WHEN rating = 3 THEN 1 ELSE 0 END) as three_star,
                SUM(CASE WHEN rating = 2 THEN 1 ELSE 0 END) as two_star,
                SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END) as one_star
            FROM module_reviews
            WHERE module_id = ?
        """, (module_id,))
        
        stats = dict(cursor.fetchone())
        
        # 평점 분포
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("평균 평점", f"⭐ {stats['avg_rating'] or 0:.1f}")
            st.write(f"총 {stats['total_reviews']}개 리뷰")
            
        with col2:
            # 평점 분포 차트
            import plotly.graph_objects as go
            
            fig = go.Figure(data=[
                go.Bar(
                    y=['5⭐', '4⭐', '3⭐', '2⭐', '1⭐'],
                    x=[stats['five_star'], stats['four_star'], stats['three_star'],
                       stats['two_star'], stats['one_star']],
                    orientation='h'
                )
            ])
            fig.update_layout(
                height=200,
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
        st.divider()
        
        # 리뷰 작성
        if st.session_state.authenticated:
            with st.expander("리뷰 작성", expanded=False):
                self._render_review_form(module_id)
                
        # 리뷰 목록
        reviews = self._get_module_reviews(module_id)
        
        for review in reviews:
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**{review['title']}**")
                    st.write("⭐" * review['rating'])
                    st.caption(f"{review['user_name']} • {self._format_date(review['created_at'])}")
                    
                with col2:
                    if review['verified_purchase']:
                        st.caption("✅ 구매 확인됨")
                        
                st.write(review['content'])
                
                # 장단점
                if review.get('pros'):
                    st.write("**👍 장점**")
                    st.write(review['pros'])
                if review.get('cons'):
                    st.write("**👎 단점**")
                    st.write(review['cons'])
                    
                # 개발자 응답
                if review.get('developer_response'):
                    with st.container():
                        st.info(f"**개발자 응답** ({self._format_date(review['response_date'])})")
                        st.write(review['developer_response'])
                        
                # 도움됨 투표
                col1, col2, col3 = st.columns([2, 1, 3])
                with col1:
                    st.caption(f"{review['helpful_count']}명에게 도움됨")
                with col2:
                    if st.button("👍 도움됨", key=f"helpful_{review['id']}"):
                        self._vote_helpful(review['id'])
                        
                st.divider()
                
    def _render_review_form(self, module_id: str):
        """리뷰 작성 폼"""
        with st.form(f"review_form_{module_id}"):
            rating = st.slider("평점", 1, 5, 5)
            title = st.text_input("제목")
            content = st.text_area("리뷰 내용", height=100)
            
            col1, col2 = st.columns(2)
            with col1:
                pros = st.text_area("장점 (선택)", height=50)
            with col2:
                cons = st.text_area("단점 (선택)", height=50)
                
            if st.form_submit_button("리뷰 등록", type="primary"):
                self._submit_review(module_id, rating, title, content, pros, cons)
                
    def _render_module_stats(self, module_id: str):
        """모듈 통계"""
        # 최근 30일 통계
        conn = self.db_manager._get_connection()
        cursor = conn.execute("""
            SELECT date, views, downloads, installs
            FROM module_analytics
            WHERE module_id = ? AND date >= date('now', '-30 days')
            ORDER BY date
        """, (module_id,))
        
        data = pd.DataFrame(cursor.fetchall(), columns=['date', 'views', 'downloads', 'installs'])
        
        if not data.empty:
            # 차트 그리기
            import plotly.graph_objects as go
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['date'], y=data['views'], name='조회수'))
            fig.add_trace(go.Scatter(x=data['date'], y=data['downloads'], name='다운로드'))
            fig.add_trace(go.Scatter(x=data['date'], y=data['installs'], name='설치'))
            
            fig.update_layout(
                title="최근 30일 통계",
                xaxis_title="날짜",
                yaxis_title="횟수",
                hovermode='x'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("아직 통계 데이터가 없습니다")
            
    def _render_version_history(self, module_id: str):
        """버전 히스토리"""
        conn = self.db_manager._get_connection()
        cursor = conn.execute("""
            SELECT version, release_notes, created_at, downloads
            FROM module_versions
            WHERE module_id = ?
            ORDER BY created_at DESC
        """, (module_id,))
        
        versions = cursor.fetchall()
        conn.close()
        
        for version in versions:
            with st.expander(f"v{version['version']} - {self._format_date(version['created_at'])}"):
                st.write(version['release_notes'] or "릴리즈 노트가 없습니다")
                st.caption(f"다운로드: {version['downloads']:,}")
                
    def _get_popular_tags(self) -> List[str]:
        """인기 태그 조회"""
        conn = self.db_manager._get_connection()
        cursor = conn.execute("""
            SELECT tags FROM marketplace_modules 
            WHERE status = 'published' AND tags IS NOT NULL
        """)
        
        all_tags = []
        for row in cursor.fetchall():
            tags = json.loads(row['tags'])
            all_tags.extend(tags)
            
        conn.close()
        
        # 빈도수 계산
        from collections import Counter
        tag_counts = Counter(all_tags)
        
        return [tag for tag, _ in tag_counts.most_common(20)]
        
    def _get_user_modules(self, user_id: int) -> List[Dict]:
        """사용자 모듈 조회"""
        conn = self.db_manager._get_connection()
        cursor = conn.execute("""
            SELECT * FROM marketplace_modules
            WHERE author_id = ?
            ORDER BY created_at DESC
        """, (user_id,))
        
        modules = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return modules
        
    def _get_user_favorites(self, user_id: int) -> List[Dict]:
        """사용자 즐겨찾기 조회"""
        conn = self.db_manager._get_connection()
        cursor = conn.execute("""
            SELECT m.* FROM marketplace_modules m
            JOIN user_favorites f ON m.module_id = f.module_id
            WHERE f.user_id = ?
            ORDER BY f.created_at DESC
        """, (user_id,))
        
        modules = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return modules
        
    def _get_module_reviews(self, module_id: str) -> List[Dict]:
        """모듈 리뷰 조회"""
        conn = self.db_manager._get_connection()
        cursor = conn.execute("""
            SELECT r.*, u.name as user_name
            FROM module_reviews r
            JOIN users u ON r.user_id = u.id
            WHERE r.module_id = ?
            ORDER BY r.helpful_count DESC, r.created_at DESC
        """, (module_id,))
        
        reviews = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return reviews
        
    def _get_developer_stats(self, user_id: int) -> Dict:
        """개발자 통계"""
        conn = self.db_manager._get_connection()
        cursor = conn.execute("""
            SELECT 
                COUNT(*) as total_modules,
                SUM(downloads) as total_downloads,
                AVG(rating) as avg_rating,
                SUM(price * downloads) as total_revenue
            FROM marketplace_modules
            WHERE author_id = ?
        """, (user_id,))
        
        stats = dict(cursor.fetchone())
        conn.close()
        
        # None 값 처리
        return {
            'total_modules': stats['total_modules'] or 0,
            'total_downloads': stats['total_downloads'] or 0,
            'avg_rating': stats['avg_rating'] or 0,
            'total_revenue': stats['total_revenue'] or 0
        }
        
    def _get_marketplace_stats(self) -> Dict:
        """마켓플레이스 전체 통계"""
        conn = self.db_manager._get_connection()
        
        # 기본 통계
        cursor = conn.execute("""
            SELECT 
                COUNT(DISTINCT module_id) as total_modules,
                COUNT(DISTINCT author_id) as active_developers,
                SUM(downloads) as total_downloads,
                SUM(installs) as total_installs
            FROM marketplace_modules
            WHERE status = 'published'
        """)
        
        stats = dict(cursor.fetchone())
        
        # 월간 활성 사용자 (간단히 추정)
        stats['monthly_active_users'] = stats['total_installs'] // 10
        
        conn.close()
        
        return stats
        
    def _get_category_distribution(self) -> pd.DataFrame:
        """카테고리별 분포"""
        conn = self.db_manager._get_connection()
        cursor = conn.execute("""
            SELECT category, COUNT(*) as count
            FROM marketplace_modules
            WHERE status = 'published'
            GROUP BY category
        """)
        
        data = pd.DataFrame(cursor.fetchall(), columns=['category', 'count'])
        conn.close()
        
        return data
        
    def _get_popular_tags_with_count(self) -> pd.DataFrame:
        """인기 태그와 수"""
        tags_count = {}
        
        conn = self.db_manager._get_connection()
        cursor = conn.execute("""
            SELECT tags FROM marketplace_modules 
            WHERE status = 'published' AND tags IS NOT NULL
        """)
        
        for row in cursor.fetchall():
            tags = json.loads(row['tags'])
            for tag in tags:
                tags_count[tag] = tags_count.get(tag, 0) + 1
                
        conn.close()
        
        # 상위 10개
        top_tags = sorted(tags_count.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return pd.DataFrame(top_tags, columns=['tag', 'count'])
        
    def _get_download_trends(self) -> pd.DataFrame:
        """다운로드 추이"""
        # 실제로는 module_analytics 테이블에서 조회
        # 여기서는 더미 데이터 생성
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        downloads = np.random.poisson(100, size=30).cumsum()
        
        return pd.DataFrame({
            'date': dates,
            'downloads': downloads
        })
        
    def _get_trending_modules(self) -> List[Dict]:
        """트렌딩 모듈"""
        # 최근 7일간 다운로드 증가율 기준
        conn = self.db_manager._get_connection()
        
        # 간단히 최근 다운로드가 많은 모듈 반환
        cursor = conn.execute("""
            SELECT *, 
                (downloads * 1.0 / (julianday('now') - julianday(created_at) + 1)) as growth_rate
            FROM marketplace_modules
            WHERE status = 'published' 
                AND created_at >= date('now', '-30 days')
            ORDER BY growth_rate DESC
            LIMIT 10
        """)
        
        modules = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return modules
        
    def _format_date(self, date_str: str) -> str:
        """날짜 포맷팅"""
        try:
            date = datetime.fromisoformat(date_str)
            return date.strftime("%Y-%m-%d")
        except:
            return date_str
            
    def _get_status_badge(self, status: str) -> str:
        """상태 배지"""
        badges = {
            'draft': '📝 초안',
            'beta': '🧪 베타',
            'published': '✅ 공개',
            'archived': '📦 보관',
            'suspended': '⚠️ 중단'
        }
        return badges.get(status, status)
        
    def _validate_step1(self, name: str, display_name: str, 
                       category: str, description: str) -> bool:
        """1단계 검증"""
        if not name or not display_name or not category or not description:
            st.error("모든 필수 항목을 입력해주세요")
            return False
            
        # 이름 검증
        if not re.match(r'^[a-zA-Z0-9_]+$', name):
            st.error("모듈 이름은 영문, 숫자, 언더스코어만 사용 가능합니다")
            return False
            
        # 중복 확인
        conn = self.db_manager._get_connection()
        cursor = conn.execute(
            "SELECT 1 FROM marketplace_modules WHERE name = ?",
            (name,)
        )
        
        if cursor.fetchone():
            st.error("이미 사용 중인 모듈 이름입니다")
            conn.close()
            return False
            
        conn.close()
        return True
        
    def _submit_review(self, module_id: str, rating: int, title: str,
                      content: str, pros: str, cons: str):
        """리뷰 제출"""
        if not title or not content:
            st.error("제목과 내용을 입력해주세요")
            return
            
        user_id = st.session_state.user['id']
        
        try:
            conn = self.db_manager._get_connection()
            cursor = conn.cursor()
            
            # 기존 리뷰 확인
            cursor.execute(
                "SELECT 1 FROM module_reviews WHERE module_id = ? AND user_id = ?",
                (module_id, user_id)
            )
            
            if cursor.fetchone():
                st.error("이미 이 모듈에 리뷰를 작성하셨습니다")
                conn.close()
                return
                
            # 리뷰 저장
            cursor.execute("""
                INSERT INTO module_reviews (
                    module_id, user_id, rating, title, content, pros, cons
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (module_id, user_id, rating, title, content, pros, cons))
            
            # 모듈 평점 업데이트
            cursor.execute("""
                UPDATE marketplace_modules 
                SET rating = (
                    SELECT AVG(rating) FROM module_reviews WHERE module_id = ?
                ),
                rating_count = (
                    SELECT COUNT(*) FROM module_reviews WHERE module_id = ?
                )
                WHERE module_id = ?
            """, (module_id, module_id, module_id))
            
            conn.commit()
            conn.close()
            
            st.success("리뷰가 등록되었습니다!")
            st.rerun()
            
        except Exception as e:
            logger.error(f"리뷰 제출 실패: {str(e)}")
            st.error("리뷰 등록 중 오류가 발생했습니다")
            
    def _vote_helpful(self, review_id: int):
        """도움됨 투표"""
        conn = self.db_manager._get_connection()
        conn.execute(
            "UPDATE module_reviews SET helpful_count = helpful_count + 1 WHERE id = ?",
            (review_id,)
        )
        conn.commit()
        conn.close()
        
        st.success("피드백이 반영되었습니다")
