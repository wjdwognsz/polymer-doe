"""
🔍 literature_search.py - 통합 연구 자원 검색
문헌, 실험 프로토콜, 데이터, 오픈 사이언스 리소스를 통합 검색하고 AI로 분석
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import re
import asyncio
from datetime import datetime, timedelta
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple, Any
import base64
from io import BytesIO
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

# 외부 라이브러리
try:
    from scholarly import scholarly
except ImportError:
    scholarly = None
    
try:
    import arxiv
except ImportError:
    arxiv = None
    
try:
    import PyPDF2
    import fitz  # PyMuPDF
except ImportError:
    PyPDF2 = None
    fitz = None
    
try:
    from github import Github
except ImportError:
    Github = None
    
try:
    import bibtexparser
except ImportError:
    bibtexparser = None

# 내부 모듈
from utils.auth_manager import check_authentication, get_current_user
from utils.sheets_manager import GoogleSheetsManager
from utils.api_manager import APIManager
from utils.common_ui import render_header, show_success, show_error, show_info, show_warning
from utils.notification_manager import NotificationManager

class IntegratedResearchManager:
    """통합 연구 자원 관리 클래스"""
    
    def __init__(self):
        """초기화"""
        self.sheets = GoogleSheetsManager()
        self.api = APIManager()
        self.notifier = NotificationManager()
        self.current_user = get_current_user()
        self.project_id = st.session_state.get('current_project', {}).get('id')
        
        # API 클라이언트 초기화
        self._init_api_clients()
        
        # 세션 상태 초기화
        self._initialize_session_state()
        
    def _initialize_session_state(self):
        """세션 상태 초기화"""
        if 'search_results' not in st.session_state:
            st.session_state.search_results = None
        if 'selected_paper' not in st.session_state:
            st.session_state.selected_paper = None
        if 'selected_protocol' not in st.session_state:
            st.session_state.selected_protocol = None
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
        if 'saved_resources' not in st.session_state:
            st.session_state.saved_resources = []
        if 'show_ai_details' not in st.session_state:
            st.session_state.show_ai_details = False
            
    def _init_api_clients(self):
        """외부 API 클라이언트 초기화"""
        self.github_client = None
        if Github and st.secrets.get('github', {}).get('token'):
            try:
                self.github_client = Github(st.secrets.github.token)
            except:
                pass
                
    def render_page(self):
        """통합 검색 페이지 메인 렌더링"""
        render_header("🔍 통합 연구 자원 검색", 
                     "문헌, 프로토콜, 데이터를 한 번에 검색하고 AI로 분석하세요")
        
        # 검색 인터페이스
        self._render_search_interface()
        
        # 검색 결과
        if st.session_state.search_results:
            self._render_search_results()
        else:
            # 최근 검색 및 저장된 자료
            col1, col2 = st.columns(2)
            with col1:
                self._render_recent_searches()
            with col2:
                self._render_saved_resources()
                
    def _render_search_interface(self):
        """검색 인터페이스 렌더링"""
        with st.container():
            # 스마트 검색 바
            col1, col2 = st.columns([4, 1])
            
            with col1:
                search_query = st.text_area(
                    "무엇을 찾고 계신가요?",
                    placeholder="예: PLA 3D 프린팅 최적 조건, PEDOT:PSS 전도도 향상 방법...",
                    height=80,
                    key="search_query"
                )
                
                # 검색 제안
                if search_query and len(search_query) > 3:
                    suggestions = self._get_search_suggestions(search_query)
                    if suggestions:
                        st.info(f"💡 추천 키워드: {', '.join(suggestions)}")
                        
            with col2:
                st.write("")  # 공백
                search_button = st.button(
                    "🔍 통합 검색",
                    type="primary",
                    use_container_width=True,
                    disabled=not search_query
                )
                
            # 고급 검색 옵션
            with st.expander("🔧 고급 검색 옵션", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    resource_types = st.multiselect(
                        "리소스 타입",
                        ["📄 논문", "📋 특허", "🔬 프로토콜", 
                         "📊 실험데이터", "💻 코드", "🧪 재료정보"],
                        default=["📄 논문", "🔬 프로토콜", "📊 실험데이터"]
                    )
                    
                    sources = st.multiselect(
                        "데이터 소스",
                        ["Google Scholar", "arXiv", "Patents", "protocols.io", 
                         "GitHub", "Zenodo", "Materials Project"],
                        default=["Google Scholar", "arXiv", "GitHub"]
                    )
                    
                with col2:
                    date_range = st.date_input(
                        "기간",
                        value=(datetime.now() - timedelta(days=365*2), datetime.now()),
                        format="YYYY-MM-DD"
                    )
                    
                    languages = st.multiselect(
                        "언어",
                        ["영어", "한국어", "일본어", "중국어"],
                        default=["영어", "한국어"]
                    )
                    
                with col3:
                    min_citations = st.number_input(
                        "최소 인용수",
                        min_value=0,
                        value=0,
                        step=10
                    )
                    
                    verified_only = st.checkbox("검증된 자료만")
                    has_raw_data = st.checkbox("원본 데이터 포함")
                    
                with col4:
                    polymer_types = st.multiselect(
                        "고분자 유형",
                        ["열가소성", "열경화성", "엘라스토머", 
                         "바이오폴리머", "전도성", "기능성"],
                        help="관련 고분자 유형 선택"
                    )
                    
                    properties = st.multiselect(
                        "관심 물성",
                        ["기계적", "열적", "전기적", "광학적", 
                         "화학적", "생분해성"],
                        help="관련 물성 데이터 우선 검색"
                    )
                    
        # 검색 실행
        if search_button and search_query:
            self._execute_integrated_search(
                query=search_query,
                resource_types=resource_types,
                sources=sources,
                filters={
                    'date_range': date_range,
                    'languages': languages,
                    'min_citations': min_citations,
                    'verified_only': verified_only,
                    'has_raw_data': has_raw_data,
                    'polymer_types': polymer_types,
                    'properties': properties
                }
            )
            
    def _execute_integrated_search(self, query: str, resource_types: List[str], 
                                  sources: List[str], filters: Dict):
        """통합 검색 실행"""
        with st.spinner("여러 소스에서 자료를 검색하는 중... (15-30초 소요)"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = {
                'papers': [],
                'protocols': [],
                'datasets': [],
                'materials': [],
                'code': [],
                'total_count': 0,
                'search_time': datetime.now()
            }
            
            # 병렬 검색 실행
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {}
                
                # 1. 문헌 검색 (30%)
                if "📄 논문" in resource_types:
                    if "Google Scholar" in sources and scholarly:
                        futures[executor.submit(self._search_google_scholar, query, filters)] = 'scholar'
                    if "arXiv" in sources and arxiv:
                        futures[executor.submit(self._search_arxiv, query, filters)] = 'arxiv'
                        
                if "📋 특허" in resource_types:
                    if "Patents" in sources:
                        futures[executor.submit(self._search_patents, query, filters)] = 'patents'
                        
                progress_bar.progress(30)
                status_text.text("📚 학술 문헌 검색 중...")
                
                # 2. 프로토콜 검색 (50%)
                if "🔬 프로토콜" in resource_types:
                    if "protocols.io" in sources:
                        futures[executor.submit(self._search_protocols_io, query, filters)] = 'protocols'
                    # PDF에서 프로토콜 추출은 논문 검색 후 수행
                    
                progress_bar.progress(50)
                status_text.text("🔬 실험 프로토콜 검색 중...")
                
                # 3. 데이터 및 코드 검색 (70%)
                if "📊 실험데이터" in resource_types or "💻 코드" in resource_types:
                    if "GitHub" in sources and self.github_client:
                        futures[executor.submit(self._search_github, query, filters)] = 'github'
                    if "Zenodo" in sources:
                        futures[executor.submit(self._search_zenodo, query, filters)] = 'zenodo'
                        
                progress_bar.progress(70)
                status_text.text("📊 실험 데이터 검색 중...")
                
                # 4. 재료 데이터베이스 검색 (85%)
                if "🧪 재료정보" in resource_types:
                    if "Materials Project" in sources:
                        futures[executor.submit(self._search_materials_project, query, filters)] = 'materials'
                        
                progress_bar.progress(85)
                status_text.text("🧪 재료 데이터베이스 조회 중...")
                
                # 결과 수집
                for future in as_completed(futures):
                    source = futures[future]
                    try:
                        source_results = future.result()
                        
                        if source in ['scholar', 'arxiv', 'patents']:
                            results['papers'].extend(source_results)
                        elif source == 'protocols':
                            results['protocols'].extend(source_results)
                        elif source in ['github', 'zenodo']:
                            # 데이터와 코드 분류
                            for item in source_results:
                                if item.get('type') == 'code':
                                    results['code'].append(item)
                                else:
                                    results['datasets'].append(item)
                        elif source == 'materials':
                            results['materials'].extend(source_results)
                            
                    except Exception as e:
                        show_warning(f"{source} 검색 중 오류 발생: {str(e)}")
                        
            # 5. AI 통합 분석 (100%)
            progress_bar.progress(90)
            status_text.text("🤖 AI가 결과를 분석하고 연결하는 중...")
            
            # 중복 제거
            results = self._deduplicate_results(results)
            
            # AI 분석 및 연결
            results = self._analyze_and_connect_results(results, query)
            
            # 총 개수
            results['total_count'] = sum(len(v) for v in results.values() if isinstance(v, list))
            
            progress_bar.progress(100)
            status_text.text("✅ 검색 완료!")
            
            # 결과 저장
            st.session_state.search_results = results
            
            # 검색 기록 저장
            self._save_search_history(query, results)
            
    def _render_search_results(self):
        """검색 결과 렌더링"""
        results = st.session_state.search_results
        
        # 결과 요약
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("총 결과", results['total_count'])
        with col2:
            st.metric("논문", len(results.get('papers', [])))
        with col3:
            st.metric("프로토콜", len(results.get('protocols', [])))
        with col4:
            st.metric("데이터셋", len(results.get('datasets', [])))
        with col5:
            st.metric("코드", len(results.get('code', [])))
            
        # AI 인사이트
        if results.get('ai_insights'):
            self._render_ai_insights(results['ai_insights'])
            
        # 결과 탭
        tabs = st.tabs([
            "🔗 통합 뷰", 
            "📄 논문", 
            "🔬 프로토콜", 
            "📊 데이터", 
            "💻 코드", 
            "🧪 재료"
        ])
        
        with tabs[0]:
            self._render_integrated_view(results)
        with tabs[1]:
            self._render_papers_tab(results.get('papers', []))
        with tabs[2]:
            self._render_protocols_tab(results.get('protocols', []))
        with tabs[3]:
            self._render_datasets_tab(results.get('datasets', []))
        with tabs[4]:
            self._render_code_tab(results.get('code', []))
        with tabs[5]:
            self._render_materials_tab(results.get('materials', []))
            
    def _render_integrated_view(self, results: Dict):
        """통합 뷰 렌더링"""
        st.subheader("🔗 연결된 리소스 그룹")
        
        resource_groups = results.get('resource_groups', [])
        
        if not resource_groups:
            show_info("연결된 리소스 그룹이 없습니다. 개별 탭에서 결과를 확인하세요.")
            return
            
        for idx, group in enumerate(resource_groups):
            with st.container():
                # 그룹 헤더
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.markdown(f"### {idx + 1}. {group['title']}")
                    st.caption(group['description'])
                    
                with col2:
                    relevance = group.get('relevance_score', 0)
                    st.metric("관련도", f"{relevance:.0%}")
                    
                # 그룹 내 리소스
                cols = st.columns(3)
                
                # 논문
                if group.get('paper'):
                    with cols[0]:
                        self._render_paper_card_mini(group['paper'])
                        
                # 프로토콜
                if group.get('protocol'):
                    with cols[1]:
                        self._render_protocol_card_mini(group['protocol'])
                        
                # 데이터셋
                if group.get('dataset'):
                    with cols[2]:
                        self._render_dataset_card_mini(group['dataset'])
                        
                # AI 설명
                if group.get('explanation'):
                    with st.expander("🤖 AI 분석 - 리소스 연결 근거"):
                        st.write(group['explanation'])
                        
                st.divider()
                
    def _render_papers_tab(self, papers: List[Dict]):
        """논문 탭 렌더링"""
        if not papers:
            show_info("검색된 논문이 없습니다.")
            return
            
        # 정렬 옵션
        sort_by = st.selectbox(
            "정렬",
            ["관련도순", "최신순", "인용순"],
            key="paper_sort"
        )
        
        sorted_papers = self._sort_results(papers, sort_by)
        
        for paper in sorted_papers:
            self._render_paper_card(paper)
            
    def _render_paper_card(self, paper: Dict):
        """논문 카드 렌더링"""
        with st.container():
            # 제목 및 저자
            st.markdown(f"### {paper['title']}")
            st.caption(f"👥 {', '.join(paper.get('authors', [])[:3])}{'...' if len(paper.get('authors', [])) > 3 else ''}")
            
            # 메타데이터
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.write(f"📅 {paper.get('year', 'N/A')}")
            with col2:
                st.write(f"📖 {paper.get('venue', 'N/A')}")
            with col3:
                st.write(f"📊 인용: {paper.get('citations', 0)}")
            with col4:
                if paper.get('doi'):
                    st.write(f"🔗 DOI: {paper['doi']}")
                    
            # 초록
            if paper.get('abstract'):
                with st.expander("초록 보기"):
                    st.write(paper['abstract'])
                    
            # 액션 버튼
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                if st.button("📄 PDF", key=f"pdf_{paper.get('id', paper['title'][:20])}"):
                    self._open_paper_pdf(paper)
                    
            with col2:
                if st.button("🔬 프로토콜 추출", key=f"extract_{paper.get('id', paper['title'][:20])}"):
                    self._extract_protocol_from_paper(paper)
                    
            with col3:
                if st.button("💾 저장", key=f"save_{paper.get('id', paper['title'][:20])}"):
                    self._save_resource(paper, 'paper')
                    
            with col4:
                if st.button("📎 BibTeX", key=f"bib_{paper.get('id', paper['title'][:20])}"):
                    self._show_bibtex(paper)
                    
            with col5:
                if st.button("🔗 관련", key=f"related_{paper.get('id', paper['title'][:20])}"):
                    self._find_related_resources(paper)
                    
            st.divider()
            
    def _render_protocols_tab(self, protocols: List[Dict]):
        """프로토콜 탭 렌더링"""
        if not protocols:
            show_info("검색된 프로토콜이 없습니다.")
            return
            
        for protocol in protocols:
            self._render_protocol_card(protocol)
            
    def _render_protocol_card(self, protocol: Dict):
        """프로토콜 카드 렌더링"""
        with st.container():
            # 제목
            st.markdown(f"### 🔬 {protocol['title']}")
            
            # 메타데이터
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.write(f"📍 출처: {protocol.get('source', 'N/A')}")
            with col2:
                st.write(f"⏱️ 소요시간: {protocol.get('duration', 'N/A')}")
            with col3:
                st.write(f"💰 비용: {protocol.get('cost', 'N/A')}")
            with col4:
                reproducibility = protocol.get('reproducibility', 0)
                st.write(f"✅ 재현성: {reproducibility:.0%}")
                
            # 재료 및 장비
            col1, col2 = st.columns(2)
            
            with col1:
                if protocol.get('materials'):
                    st.write("**재료:**")
                    for material in protocol['materials'][:5]:
                        st.write(f"• {material}")
                    if len(protocol['materials']) > 5:
                        st.write(f"• ... 외 {len(protocol['materials']) - 5}개")
                        
            with col2:
                if protocol.get('equipment'):
                    st.write("**장비:**")
                    for equipment in protocol['equipment'][:5]:
                        st.write(f"• {equipment}")
                        
            # 절차 요약
            if protocol.get('procedure_summary'):
                with st.expander("절차 요약"):
                    st.write(protocol['procedure_summary'])
                    
            # 액션 버튼
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("📋 상세보기", key=f"detail_{protocol['id']}"):
                    self._show_protocol_detail(protocol)
                    
            with col2:
                if st.button("🧪 실험 설계", key=f"design_{protocol['id']}"):
                    st.session_state.selected_protocol = protocol
                    st.switch_page("pages/experiment_design.py")
                    
            with col3:
                if st.button("💾 저장", key=f"save_prot_{protocol['id']}"):
                    self._save_resource(protocol, 'protocol')
                    
            with col4:
                if st.button("🤝 공유", key=f"share_{protocol['id']}"):
                    self._share_protocol(protocol)
                    
            st.divider()
            
    def _render_datasets_tab(self, datasets: List[Dict]):
        """데이터셋 탭 렌더링"""
        if not datasets:
            show_info("검색된 데이터셋이 없습니다.")
            return
            
        for dataset in datasets:
            self._render_dataset_card(dataset)
            
    def _render_dataset_card(self, dataset: Dict):
        """데이터셋 카드 렌더링"""
        with st.container():
            # 제목
            st.markdown(f"### 📊 {dataset['title']}")
            
            # 메타데이터
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.write(f"📍 출처: {dataset.get('source', 'N/A')}")
            with col2:
                st.write(f"📏 크기: {self._format_file_size(dataset.get('size', 0))}")
            with col3:
                st.write(f"📁 형식: {', '.join(dataset.get('formats', []))}")
            with col4:
                st.write(f"📅 업데이트: {dataset.get('updated', 'N/A')}")
                
            # 설명
            if dataset.get('description'):
                st.write(dataset['description'][:200] + "..." if len(dataset['description']) > 200 else dataset['description'])
                
            # 데이터 구조
            if dataset.get('structure'):
                with st.expander("데이터 구조"):
                    st.json(dataset['structure'])
                    
            # 액션 버튼
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("⬇️ 다운로드", key=f"download_{dataset['id']}"):
                    self._download_dataset(dataset)
                    
            with col2:
                if st.button("📈 분석", key=f"analyze_{dataset['id']}"):
                    st.session_state.selected_dataset = dataset
                    st.switch_page("pages/data_analysis.py")
                    
            with col3:
                if st.button("💾 저장", key=f"save_data_{dataset['id']}"):
                    self._save_resource(dataset, 'dataset')
                    
            with col4:
                if st.button("🔗 메타데이터", key=f"meta_{dataset['id']}"):
                    self._show_metadata(dataset)
                    
            st.divider()
            
    def _render_ai_insights(self, insights: Dict):
        """AI 인사이트 렌더링"""
        with st.container():
            st.subheader("🤖 AI 분석 결과")
            
            # 핵심 발견사항
            if insights.get('key_findings'):
                st.info(f"**핵심 발견:** {insights['key_findings']}")
                
            # 추천 리소스
            if insights.get('recommendations'):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if insights['recommendations'].get('best_paper'):
                        paper = insights['recommendations']['best_paper']
                        st.success(f"**추천 논문:** {paper['title']} (관련도: {paper['score']:.0%})")
                        
                with col2:
                    if insights['recommendations'].get('best_protocol'):
                        protocol = insights['recommendations']['best_protocol']
                        st.success(f"**추천 프로토콜:** {protocol['title']} (재현성: {protocol['reproducibility']:.0%})")
                        
                with col3:
                    if insights['recommendations'].get('best_dataset'):
                        dataset = insights['recommendations']['best_dataset']
                        st.success(f"**추천 데이터셋:** {dataset['title']}")
                        
            # 주의사항
            if insights.get('warnings'):
                for warning in insights['warnings']:
                    st.warning(f"⚠️ {warning}")
                    
            # 상세 분석 토글
            self._render_ai_response(insights, "literature_analysis")
            
    def _render_ai_response(self, response: Dict, response_type: str = "general"):
        """AI 응답 렌더링 (상세도 제어 포함)"""
        # 상세 설명 토글
        show_details = st.session_state.get('show_ai_details', False)
        
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("🔍 상세", key=f"toggle_{response_type}"):
                st.session_state.show_ai_details = not show_details
                st.rerun()
                
        # 상세 설명 (조건부 표시)
        if show_details:
            tabs = st.tabs(["분석 과정", "연결 근거", "신뢰도", "한계점"])
            
            with tabs[0]:
                st.write("**분석 과정:**")
                st.write(response.get('analysis_process', '분석 과정 정보가 없습니다.'))
                
            with tabs[1]:
                st.write("**리소스 연결 근거:**")
                connections = response.get('connection_reasoning', {})
                for conn_type, reasoning in connections.items():
                    st.write(f"• {conn_type}: {reasoning}")
                    
            with tabs[2]:
                st.write("**신뢰도 평가:**")
                confidence = response.get('confidence', {})
                for aspect, score in confidence.items():
                    st.progress(score)
                    st.write(f"{aspect}: {score:.0%}")
                    
            with tabs[3]:
                st.write("**한계점 및 주의사항:**")
                limitations = response.get('limitations', [])
                for limitation in limitations:
                    st.write(f"• {limitation}")
                    
    # === 검색 메서드 ===
    
    def _search_google_scholar(self, query: str, filters: Dict) -> List[Dict]:
        """Google Scholar 검색"""
        results = []
        
        try:
            # scholarly 검색
            search_query = scholarly.search_pubs(query)
            
            for i, paper in enumerate(search_query):
                if i >= 20:  # 최대 20개
                    break
                    
                # 필터 적용
                year = paper.get('bib', {}).get('pub_year', '')
                if year and filters.get('date_range'):
                    try:
                        if int(year) < filters['date_range'][0].year or int(year) > filters['date_range'][1].year:
                            continue
                    except:
                        pass
                        
                results.append({
                    'id': hashlib.md5(paper.get('bib', {}).get('title', '').encode()).hexdigest()[:10],
                    'title': paper.get('bib', {}).get('title', ''),
                    'authors': paper.get('bib', {}).get('author', '').split(' and '),
                    'year': year,
                    'venue': paper.get('bib', {}).get('venue', ''),
                    'abstract': paper.get('bib', {}).get('abstract', ''),
                    'citations': paper.get('num_citations', 0),
                    'url': paper.get('pub_url', ''),
                    'pdf_url': paper.get('eprint_url', ''),
                    'source': 'Google Scholar'
                })
                
        except Exception as e:
            st.error(f"Google Scholar 검색 오류: {str(e)}")
            
        return results
        
    def _search_arxiv(self, query: str, filters: Dict) -> List[Dict]:
        """arXiv 검색"""
        results = []
        
        try:
            # arXiv 검색
            search = arxiv.Search(
                query=query,
                max_results=20,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            for paper in search.results():
                # 필터 적용
                if filters.get('date_range'):
                    if paper.published.date() < filters['date_range'][0] or paper.published.date() > filters['date_range'][1]:
                        continue
                        
                results.append({
                    'id': paper.entry_id.split('/')[-1],
                    'title': paper.title,
                    'authors': [author.name for author in paper.authors],
                    'year': paper.published.year,
                    'abstract': paper.summary,
                    'categories': paper.categories,
                    'url': paper.entry_id,
                    'pdf_url': paper.pdf_url,
                    'source': 'arXiv'
                })
                
        except Exception as e:
            st.error(f"arXiv 검색 오류: {str(e)}")
            
        return results
        
    def _search_github(self, query: str, filters: Dict) -> List[Dict]:
        """GitHub 검색"""
        results = []
        
        if not self.github_client:
            return results
            
        try:
            # 저장소 검색
            search_query = f"{query} polymer in:readme"
            repos = self.github_client.search_repositories(
                query=search_query,
                sort='stars',
                order='desc'
            )
            
            for repo in repos[:10]:
                # 관련 파일 찾기
                data_files = []
                code_files = []
                
                try:
                    contents = repo.get_contents("")
                    while contents:
                        file_content = contents.pop(0)
                        if file_content.type == "dir":
                            contents.extend(repo.get_contents(file_content.path))
                        else:
                            # 데이터 파일
                            if any(file_content.name.endswith(ext) for ext in ['.csv', '.xlsx', '.json', '.hdf5']):
                                data_files.append({
                                    'name': file_content.name,
                                    'path': file_content.path,
                                    'size': file_content.size
                                })
                            # 코드 파일
                            elif any(file_content.name.endswith(ext) for ext in ['.py', '.ipynb', '.m', '.R']):
                                code_files.append({
                                    'name': file_content.name,
                                    'path': file_content.path
                                })
                except:
                    pass
                    
                # 데이터셋으로 분류
                if data_files:
                    results.append({
                        'id': f"github_data_{repo.id}",
                        'type': 'dataset',
                        'title': f"{repo.name} - Dataset",
                        'description': repo.description or '',
                        'source': 'GitHub',
                        'url': repo.html_url,
                        'size': sum(f['size'] for f in data_files),
                        'formats': list(set(f['name'].split('.')[-1] for f in data_files)),
                        'files': data_files,
                        'stars': repo.stargazers_count,
                        'updated': repo.updated_at.strftime('%Y-%m-%d')
                    })
                    
                # 코드로 분류
                if code_files:
                    results.append({
                        'id': f"github_code_{repo.id}",
                        'type': 'code',
                        'title': repo.name,
                        'description': repo.description or '',
                        'source': 'GitHub',
                        'url': repo.html_url,
                        'language': repo.language,
                        'files': code_files,
                        'stars': repo.stargazers_count,
                        'topics': repo.get_topics(),
                        'updated': repo.updated_at.strftime('%Y-%m-%d')
                    })
                    
        except Exception as e:
            st.error(f"GitHub 검색 오류: {str(e)}")
            
        return results
        
    def _extract_protocol_from_paper(self, paper: Dict):
        """논문에서 프로토콜 추출"""
        if not paper.get('pdf_url'):
            show_error("PDF URL이 없습니다.")
            return
            
        with st.spinner("PDF에서 실험 프로토콜을 추출하는 중..."):
            try:
                # PDF 다운로드
                response = requests.get(paper['pdf_url'])
                pdf_file = BytesIO(response.content)
                
                # 텍스트 추출
                text = self._extract_text_from_pdf(pdf_file)
                
                # Methods 섹션 찾기
                methods_text = self._extract_methods_section(text)
                
                if not methods_text:
                    show_warning("Methods 섹션을 찾을 수 없습니다.")
                    return
                    
                # AI로 구조화
                protocol = self._structure_protocol_with_ai(methods_text, paper['title'])
                
                # 프로토콜 표시
                self._show_extracted_protocol(protocol)
                
            except Exception as e:
                show_error(f"프로토콜 추출 실패: {str(e)}")
                
    def _extract_text_from_pdf(self, pdf_file: BytesIO) -> str:
        """PDF에서 텍스트 추출"""
        text = ""
        
        if fitz:
            # PyMuPDF 사용
            with fitz.open(stream=pdf_file, filetype="pdf") as pdf:
                for page in pdf:
                    text += page.get_text()
        elif PyPDF2:
            # PyPDF2 사용
            reader = PyPDF2.PdfReader(pdf_file)
            for page in reader.pages:
                text += page.extract_text()
                
        return text
        
    def _extract_methods_section(self, text: str) -> str:
        """텍스트에서 Methods 섹션 추출"""
        # 다양한 섹션 이름 패턴
        patterns = [
            r'(?i)(experimental|methods|methodology|materials and methods|experimental section)(.*?)(?=references|acknowledgment|conclusion|results)',
            r'(?i)(2\.\s*experimental|3\.\s*experimental|2\.\s*methods|3\.\s*methods)(.*?)(?=\d\.\s*\w+|references)',
            r'(?i)(experimental procedures?|experimental details?)(.*?)(?=references|acknowledgment|conclusion)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(2).strip()
                
        # 키워드 기반 추출
        keywords = ['prepared', 'synthesized', 'mixed', 'dissolved', 'heated', 'cooled', 'stirred']
        paragraphs = text.split('\n\n')
        
        methods_paragraphs = []
        for para in paragraphs:
            if any(keyword in para.lower() for keyword in keywords):
                methods_paragraphs.append(para)
                
        return '\n\n'.join(methods_paragraphs[:10])
        
    def _structure_protocol_with_ai(self, methods_text: str, paper_title: str) -> Dict:
        """AI를 사용하여 프로토콜 구조화"""
        prompt = f"""
        다음 논문의 실험 방법 텍스트에서 구조화된 프로토콜을 추출해주세요:
        
        논문 제목: {paper_title}
        
        Methods 텍스트:
        {methods_text[:3000]}
        
        다음 정보를 JSON 형식으로 추출하세요:
        1. materials: 재료 목록 (이름, 순도, 공급사, 사용량)
        2. equipment: 장비 목록 (종류, 모델, 설정값)
        3. procedure: 실험 절차 (단계별로)
        4. conditions: 공정 조건 (온도, 압력, 시간 등)
        5. characterization: 분석 방법
        """
        
        try:
            response = self.api.generate_response(prompt, json_mode=True)
            protocol = json.loads(response)
            
            # 추가 정보
            protocol['title'] = f"Protocol from: {paper_title}"
            protocol['source'] = 'Extracted from paper'
            protocol['paper_id'] = paper_title[:50]
            
            return protocol
            
        except Exception as e:
            st.error(f"AI 프로토콜 구조화 실패: {str(e)}")
            return {
                'title': f"Protocol from: {paper_title}",
                'raw_text': methods_text,
                'error': str(e)
            }
            
    def _analyze_and_connect_results(self, results: Dict, query: str) -> Dict:
        """AI를 사용한 결과 분석 및 연결"""
        # 리소스 간 연결 찾기
        resource_groups = self._find_resource_connections(results)
        
        # AI 인사이트 생성
        insights = self._generate_integrated_insights(results, query)
        
        # 추천 선정
        recommendations = self._select_best_resources(results, query)
        insights['recommendations'] = recommendations
        
        # 주의사항 식별
        warnings = self._identify_warnings(results)
        insights['warnings'] = warnings
        
        results['resource_groups'] = resource_groups
        results['ai_insights'] = insights
        
        return results
        
    def _find_resource_connections(self, results: Dict) -> List[Dict]:
        """리소스 간 연결 관계 찾기"""
        connections = []
        
        papers = results.get('papers', [])
        protocols = results.get('protocols', [])
        datasets = results.get('datasets', [])
        
        # 논문-프로토콜-데이터 매칭
        for paper in papers:
            paper_authors = set(author.lower() for author in paper.get('authors', []))
            paper_keywords = self._extract_keywords(paper.get('title', '') + ' ' + paper.get('abstract', ''))
            
            matching_protocols = []
            matching_datasets = []
            
            # 프로토콜 매칭
            for protocol in protocols:
                # 저자 매칭
                if protocol.get('authors'):
                    protocol_authors = set(author.lower() for author in protocol['authors'])
                    if paper_authors & protocol_authors:
                        matching_protocols.append(protocol)
                        continue
                        
                # 키워드 매칭
                protocol_keywords = self._extract_keywords(protocol.get('title', '') + ' ' + protocol.get('description', ''))
                if len(paper_keywords & protocol_keywords) >= 3:
                    matching_protocols.append(protocol)
                    
            # 데이터셋 매칭
            for dataset in datasets:
                dataset_keywords = self._extract_keywords(dataset.get('title', '') + ' ' + dataset.get('description', ''))
                if len(paper_keywords & dataset_keywords) >= 3:
                    matching_datasets.append(dataset)
                    
            # 연결 그룹 생성
            if matching_protocols or matching_datasets:
                connections.append({
                    'title': paper['title'][:80] + "...",
                    'relevance_score': 0.8 if matching_protocols and matching_datasets else 0.6,
                    'description': f"논문 + {len(matching_protocols)}개 프로토콜 + {len(matching_datasets)}개 데이터셋",
                    'paper': paper,
                    'protocol': matching_protocols[0] if matching_protocols else None,
                    'dataset': matching_datasets[0] if matching_datasets else None,
                    'explanation': self._generate_connection_explanation(paper, matching_protocols, matching_datasets)
                })
                
        return sorted(connections, key=lambda x: x['relevance_score'], reverse=True)[:10]
        
    def _generate_connection_explanation(self, paper: Dict, protocols: List[Dict], datasets: List[Dict]) -> str:
        """연결 관계 설명 생성"""
        explanation = f"이 논문 '{paper['title']}'은"
        
        if protocols:
            explanation += f" {len(protocols)}개의 관련 프로토콜과 연결되었습니다."
            if protocols[0].get('reproducibility'):
                explanation += f" 가장 관련성 높은 프로토콜의 재현성은 {protocols[0]['reproducibility']:.0%}입니다."
                
        if datasets:
            explanation += f" {len(datasets)}개의 데이터셋이 발견되었습니다."
            total_size = sum(d.get('size', 0) for d in datasets)
            if total_size > 0:
                explanation += f" 총 {self._format_file_size(total_size)}의 데이터를 제공합니다."
                
        return explanation
        
    def _extract_keywords(self, text: str) -> set:
        """텍스트에서 키워드 추출"""
        # 간단한 키워드 추출 (실제로는 더 정교한 NLP 필요)
        words = re.findall(r'\b[a-z]+\b', text.lower())
        
        # 불용어 제거
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'been'}
        
        keywords = set()
        for word in words:
            if len(word) > 3 and word not in stopwords:
                keywords.add(word)
                
        return keywords
        
    def _format_file_size(self, size: int) -> str:
        """파일 크기 포맷팅"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
        
    def _deduplicate_results(self, results: Dict) -> Dict:
        """중복 결과 제거"""
        for key in ['papers', 'protocols', 'datasets', 'materials', 'code']:
            if key in results and isinstance(results[key], list):
                # 제목 기반 중복 제거
                seen_titles = set()
                unique_items = []
                
                for item in results[key]:
                    title = item.get('title', '').lower()
                    if title and title not in seen_titles:
                        seen_titles.add(title)
                        unique_items.append(item)
                        
                results[key] = unique_items
                
        return results
        
    def _save_search_history(self, query: str, results: Dict):
        """검색 기록 저장"""
        history_entry = {
            'query': query,
            'timestamp': datetime.now(),
            'result_count': results['total_count'],
            'user_id': self.current_user['id']
        }
        
        st.session_state.search_history.append(history_entry)
        
        # 데이터베이스에도 저장
        try:
            self.sheets.create_data('SearchHistory', history_entry)
        except:
            pass


def render():
    """페이지 렌더링 함수"""
    # 인증 확인
    if not check_authentication():
        st.error("로그인이 필요합니다.")
        st.stop()
        
    # 연구 자원 관리자 초기화 및 렌더링
    manager = IntegratedResearchManager()
    manager.render_page()


if __name__ == "__main__":
    render()
