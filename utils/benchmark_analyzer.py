"""
벤치마크 분석기 - 실험 결과와 문헌 데이터 비교 분석
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import json
from collections import defaultdict
import hashlib

# Plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# 프로젝트 모듈
from config.app_config import get_config
from utils.error_handler import handle_errors
from utils.api_manager import get_api_manager
from utils.data_processor import DataProcessor

logger = logging.getLogger(__name__)

# ===========================================================================
# 🎯 데이터 모델
# ===========================================================================

@dataclass
class BenchmarkMetrics:
    """성능 평가 지표"""
    percentile_rank: float  # 백분위 순위 (0-100)
    z_score: float  # 표준화 점수
    relative_performance: Dict[str, float]  # 항목별 상대 성능
    improvement_suggestions: List[str]  # 개선 제안
    similar_works: List[Dict[str, Any]]  # 유사 연구
    confidence_score: float = 0.0  # 신뢰도 점수
    sample_size: int = 0  # 비교 데이터 수

@dataclass
class LiteratureData:
    """문헌 데이터"""
    source: str  # 출처 (OpenAlex, Crossref 등)
    title: str
    authors: List[str]
    year: int
    doi: Optional[str] = None
    url: Optional[str] = None
    abstract: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    materials: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    relevance_score: float = 0.0

@dataclass
class ComparisonResult:
    """비교 분석 결과"""
    experiment_id: str
    timestamp: datetime
    metrics: BenchmarkMetrics
    literature_data: List[LiteratureData]
    visualizations: Dict[str, Any]
    report: Dict[str, Any]

# ===========================================================================
# 🔍 벤치마크 분석기
# ===========================================================================

class BenchmarkAnalyzer:
    """실험 결과와 문헌 데이터 비교 분석"""
    
    def __init__(self):
        self.config = get_config()
        self.api_manager = get_api_manager()
        self.data_processor = DataProcessor()
        self.cache = {}
        self.cache_ttl = timedelta(hours=24)
        
    # ===== 메인 분석 함수 =====
    
    @handle_errors
    async def analyze_benchmark(
        self,
        experiment_data: Dict[str, Any],
        search_params: Optional[Dict[str, Any]] = None,
        analysis_type: str = 'comprehensive'
    ) -> ComparisonResult:
        """실험 결과의 벤치마크 분석 수행"""
        
        logger.info(f"Starting benchmark analysis for experiment: {experiment_data.get('name', 'Unknown')}")
        
        # 1. 검색 파라미터 준비
        if not search_params:
            search_params = self._prepare_search_params(experiment_data)
            
        # 2. 문헌 데이터 수집 (비동기)
        literature_data = await self._collect_literature_data(search_params)
        
        # 3. 데이터 매칭 및 필터링
        matched_data = self._match_and_filter(experiment_data, literature_data)
        
        # 4. 메트릭 계산
        metrics = self._calculate_metrics(experiment_data, matched_data)
        
        # 5. 시각화 생성
        visualizations = self._create_visualizations(experiment_data, matched_data, metrics)
        
        # 6. 리포트 생성
        report = self._generate_report(experiment_data, matched_data, metrics)
        
        # 결과 반환
        result = ComparisonResult(
            experiment_id=experiment_data.get('id', 'unknown'),
            timestamp=datetime.now(),
            metrics=metrics,
            literature_data=matched_data,
            visualizations=visualizations,
            report=report
        )
        
        logger.info(f"Benchmark analysis completed. Found {len(matched_data)} comparable studies.")
        
        return result
        
    # ===== 데이터 수집 =====
    
    async def _collect_literature_data(
        self, 
        search_params: Dict[str, Any]
    ) -> List[LiteratureData]:
        """여러 소스에서 문헌 데이터 수집"""
        
        # 캐시 확인
        cache_key = self._get_cache_key(search_params)
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < self.cache_ttl:
                logger.info("Using cached literature data")
                return cached_data
                
        # 비동기 작업 목록
        tasks = []
        
        # OpenAlex 검색
        if self.config.get('apis', {}).get('openalex', {}).get('enabled', True):
            tasks.append(self._search_openalex(search_params))
            
        # Crossref 검색
        if self.config.get('apis', {}).get('crossref', {}).get('enabled', True):
            tasks.append(self._search_crossref(search_params))
            
        # PubMed 검색
        if self.config.get('apis', {}).get('pubmed', {}).get('enabled', True):
            tasks.append(self._search_pubmed(search_params))
            
        # Materials Project 검색
        if self.config.get('apis', {}).get('materials_project', {}).get('enabled', True):
            tasks.append(self._search_materials_project(search_params))
            
        # 병렬 실행
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 결과 통합
        all_literature = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error in data collection: {result}")
            else:
                all_literature.extend(result)
                
        # 중복 제거
        unique_literature = self._deduplicate_literature(all_literature)
        
        # 캐시 저장
        self.cache[cache_key] = (unique_literature, datetime.now())
        
        return unique_literature
        
    async def _search_openalex(self, params: Dict[str, Any]) -> List[LiteratureData]:
        """OpenAlex API 검색"""
        try:
            query = params.get('keywords', [])
            if isinstance(query, list):
                query = ' '.join(query)
                
            # API 호출
            response = await self.api_manager.search_literature(
                query=query,
                source='openalex',
                filters=params.get('filters', {}),
                limit=params.get('limit', 50)
            )
            
            # 결과 파싱
            literature = []
            for work in response.get('results', []):
                lit_data = LiteratureData(
                    source='OpenAlex',
                    title=work.get('title', ''),
                    authors=[author.get('author', {}).get('display_name', '') 
                            for author in work.get('authorships', [])],
                    year=work.get('publication_year', 0),
                    doi=work.get('doi', ''),
                    abstract=work.get('abstract', ''),
                    metrics=self._extract_metrics_from_text(
                        work.get('title', '') + ' ' + work.get('abstract', '')
                    )
                )
                literature.append(lit_data)
                
            return literature
            
        except Exception as e:
            logger.error(f"OpenAlex search error: {e}")
            return []
            
    async def _search_crossref(self, params: Dict[str, Any]) -> List[LiteratureData]:
        """Crossref API 검색"""
        try:
            query = params.get('keywords', [])
            if isinstance(query, list):
                query = ' '.join(query)
                
            # API 호출
            response = await self.api_manager.search_literature(
                query=query,
                source='crossref',
                filters=params.get('filters', {}),
                limit=params.get('limit', 50)
            )
            
            # 결과 파싱
            literature = []
            for item in response.get('message', {}).get('items', []):
                lit_data = LiteratureData(
                    source='Crossref',
                    title=item.get('title', [''])[0],
                    authors=[f"{author.get('given', '')} {author.get('family', '')}"
                            for author in item.get('author', [])],
                    year=item.get('published-print', {}).get('date-parts', [[0]])[0][0],
                    doi=item.get('DOI', ''),
                    url=item.get('URL', '')
                )
                literature.append(lit_data)
                
            return literature
            
        except Exception as e:
            logger.error(f"Crossref search error: {e}")
            return []
            
    async def _search_pubmed(self, params: Dict[str, Any]) -> List[LiteratureData]:
        """PubMed API 검색"""
        try:
            query = params.get('keywords', [])
            if isinstance(query, list):
                query = ' '.join(query)
                
            # API 호출
            response = await self.api_manager.search_literature(
                query=query,
                source='pubmed',
                filters=params.get('filters', {}),
                limit=params.get('limit', 50)
            )
            
            # 결과 파싱
            literature = []
            for article in response.get('articles', []):
                lit_data = LiteratureData(
                    source='PubMed',
                    title=article.get('title', ''),
                    authors=article.get('authors', []),
                    year=article.get('year', 0),
                    doi=article.get('doi', ''),
                    abstract=article.get('abstract', ''),
                    metrics=self._extract_metrics_from_text(article.get('abstract', ''))
                )
                literature.append(lit_data)
                
            return literature
            
        except Exception as e:
            logger.error(f"PubMed search error: {e}")
            return []
            
    async def _search_materials_project(self, params: Dict[str, Any]) -> List[LiteratureData]:
        """Materials Project 데이터베이스 검색"""
        try:
            # 재료 관련 검색
            materials = params.get('materials', [])
            if not materials:
                return []
                
            # API 호출
            response = await self.api_manager.search_materials_data(
                materials=materials,
                properties=params.get('properties', [])
            )
            
            # 결과 파싱
            literature = []
            for material in response.get('data', []):
                lit_data = LiteratureData(
                    source='Materials Project',
                    title=f"{material.get('formula', '')} - {material.get('material_id', '')}",
                    authors=['Materials Project'],
                    year=datetime.now().year,
                    url=f"https://materialsproject.org/materials/{material.get('material_id', '')}",
                    metrics=material.get('properties', {}),
                    materials=[material.get('formula', '')]
                )
                literature.append(lit_data)
                
            return literature
            
        except Exception as e:
            logger.error(f"Materials Project search error: {e}")
            return []
            
    # ===== 데이터 매칭 =====
    
    def _match_and_filter(
        self,
        experiment_data: Dict[str, Any],
        literature_data: List[LiteratureData]
    ) -> List[LiteratureData]:
        """실험과 관련된 문헌 매칭 및 필터링"""
        
        matched = []
        
        for lit in literature_data:
            # 관련성 점수 계산
            relevance_score = self._calculate_relevance(experiment_data, lit)
            lit.relevance_score = relevance_score
            
            # 임계값 이상만 포함
            if relevance_score >= 0.5:
                matched.append(lit)
                
        # 관련성 순으로 정렬
        matched.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # 상위 N개만 반환
        max_results = self.config.get('benchmark', {}).get('max_comparison_items', 100)
        return matched[:max_results]
        
    def _calculate_relevance(
        self,
        experiment: Dict[str, Any],
        literature: LiteratureData
    ) -> float:
        """실험과 문헌의 관련성 점수 계산"""
        
        score = 0.0
        weights = {
            'materials': 0.4,
            'conditions': 0.3,
            'keywords': 0.2,
            'year': 0.1
        }
        
        # 재료 유사도
        exp_materials = set(experiment.get('materials', []))
        lit_materials = set(literature.materials)
        if exp_materials and lit_materials:
            material_similarity = len(exp_materials & lit_materials) / len(exp_materials | lit_materials)
            score += material_similarity * weights['materials']
            
        # 조건 유사도
        exp_conditions = experiment.get('conditions', {})
        lit_conditions = literature.conditions
        if exp_conditions and lit_conditions:
            condition_similarity = self._calculate_condition_similarity(exp_conditions, lit_conditions)
            score += condition_similarity * weights['conditions']
            
        # 키워드 유사도
        exp_keywords = set(experiment.get('keywords', []))
        lit_keywords = set(self._extract_keywords(literature.title + ' ' + (literature.abstract or '')))
        if exp_keywords and lit_keywords:
            keyword_similarity = len(exp_keywords & lit_keywords) / len(exp_keywords | lit_keywords)
            score += keyword_similarity * weights['keywords']
            
        # 연도 가중치 (최신 논문 선호)
        if literature.year:
            year_diff = datetime.now().year - literature.year
            year_score = max(0, 1 - (year_diff / 10))  # 10년 이내 선호
            score += year_score * weights['year']
            
        return score
        
    def _calculate_condition_similarity(
        self,
        cond1: Dict[str, Any],
        cond2: Dict[str, Any]
    ) -> float:
        """실험 조건 유사도 계산"""
        
        if not cond1 or not cond2:
            return 0.0
            
        common_params = set(cond1.keys()) & set(cond2.keys())
        if not common_params:
            return 0.0
            
        similarities = []
        for param in common_params:
            val1 = cond1[param]
            val2 = cond2[param]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # 수치 비교
                if val1 == 0 and val2 == 0:
                    sim = 1.0
                else:
                    diff = abs(val1 - val2)
                    avg = (abs(val1) + abs(val2)) / 2
                    sim = max(0, 1 - (diff / avg))
            else:
                # 범주형 비교
                sim = 1.0 if val1 == val2 else 0.0
                
            similarities.append(sim)
            
        return sum(similarities) / len(similarities)
        
    # ===== 메트릭 계산 =====
    
    def _calculate_metrics(
        self,
        experiment_data: Dict[str, Any],
        matched_literature: List[LiteratureData]
    ) -> BenchmarkMetrics:
        """벤치마크 메트릭 계산"""
        
        if not matched_literature:
            return BenchmarkMetrics(
                percentile_rank=50.0,
                z_score=0.0,
                relative_performance={},
                improvement_suggestions=['문헌 데이터가 부족하여 비교가 제한적입니다.'],
                similar_works=[],
                confidence_score=0.0,
                sample_size=0
            )
            
        # 주요 메트릭 추출
        metric_values = defaultdict(list)
        for lit in matched_literature:
            for metric, value in lit.metrics.items():
                if isinstance(value, (int, float)):
                    metric_values[metric].append(value)
                    
        # 각 메트릭별 분석
        relative_performance = {}
        percentiles = []
        z_scores = []
        
        for metric, values in metric_values.items():
            if metric in experiment_data.get('results', {}):
                exp_value = experiment_data['results'][metric]
                if isinstance(exp_value, (int, float)) and len(values) > 1:
                    # 백분위 계산
                    percentile = self._calculate_percentile(exp_value, values)
                    percentiles.append(percentile)
                    
                    # Z-score 계산
                    z_score = self._calculate_z_score(exp_value, values)
                    z_scores.append(z_score)
                    
                    # 상대 성능
                    relative_performance[metric] = {
                        'value': exp_value,
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'percentile': percentile,
                        'z_score': z_score,
                        'sample_size': len(values)
                    }
                    
        # 전체 메트릭
        overall_percentile = np.mean(percentiles) if percentiles else 50.0
        overall_z_score = np.mean(z_scores) if z_scores else 0.0
        
        # 유사 연구 선정
        similar_works = self._find_similar_works(experiment_data, matched_literature)
        
        # 개선 제안 생성
        suggestions = self._generate_improvement_suggestions(
            experiment_data,
            relative_performance,
            similar_works
        )
        
        # 신뢰도 계산
        confidence = self._calculate_confidence_score(matched_literature, relative_performance)
        
        return BenchmarkMetrics(
            percentile_rank=overall_percentile,
            z_score=overall_z_score,
            relative_performance=relative_performance,
            improvement_suggestions=suggestions,
            similar_works=similar_works,
            confidence_score=confidence,
            sample_size=len(matched_literature)
        )
        
    def _calculate_percentile(self, value: float, values: List[float]) -> float:
        """백분위 계산"""
        return (sum(1 for v in values if v <= value) / len(values)) * 100
        
    def _calculate_z_score(self, value: float, values: List[float]) -> float:
        """Z-score 계산"""
        mean = np.mean(values)
        std = np.std(values)
        return (value - mean) / std if std > 0 else 0.0
        
    def _find_similar_works(
        self,
        experiment: Dict[str, Any],
        literature: List[LiteratureData]
    ) -> List[Dict[str, Any]]:
        """가장 유사한 연구 찾기"""
        
        # 관련성 높은 상위 5개
        top_similar = sorted(literature, key=lambda x: x.relevance_score, reverse=True)[:5]
        
        similar_works = []
        for lit in top_similar:
            work = {
                'title': lit.title,
                'authors': lit.authors[:3],  # 첫 3명만
                'year': lit.year,
                'doi': lit.doi,
                'relevance_score': lit.relevance_score,
                'key_metrics': lit.metrics,
                'differences': self._identify_differences(experiment, lit)
            }
            similar_works.append(work)
            
        return similar_works
        
    def _identify_differences(
        self,
        experiment: Dict[str, Any],
        literature: LiteratureData
    ) -> Dict[str, Any]:
        """주요 차이점 식별"""
        
        differences = {
            'materials': [],
            'conditions': [],
            'performance': []
        }
        
        # 재료 차이
        exp_materials = set(experiment.get('materials', []))
        lit_materials = set(literature.materials)
        differences['materials'] = {
            'unique_to_experiment': list(exp_materials - lit_materials),
            'unique_to_literature': list(lit_materials - exp_materials)
        }
        
        # 조건 차이
        exp_conditions = experiment.get('conditions', {})
        lit_conditions = literature.conditions
        for param in set(exp_conditions.keys()) | set(lit_conditions.keys()):
            if param in exp_conditions and param in lit_conditions:
                exp_val = exp_conditions[param]
                lit_val = lit_conditions[param]
                if exp_val != lit_val:
                    differences['conditions'].append({
                        'parameter': param,
                        'experiment': exp_val,
                        'literature': lit_val
                    })
                    
        return differences
        
    # ===== 개선 제안 =====
    
    def _generate_improvement_suggestions(
        self,
        experiment: Dict[str, Any],
        performance: Dict[str, Any],
        similar_works: List[Dict[str, Any]]
    ) -> List[str]:
        """개선 제안 생성"""
        
        suggestions = []
        
        # 성능 기반 제안
        for metric, perf in performance.items():
            if perf['percentile'] < 50:
                suggestions.append(
                    f"{metric}이(가) 평균보다 낮습니다 (하위 {perf['percentile']:.0f}%). "
                    f"상위 연구들의 평균값은 {perf['mean']:.2f}입니다."
                )
                
        # 유사 연구 기반 제안
        if similar_works:
            top_work = similar_works[0]
            if top_work['differences']['materials']['unique_to_literature']:
                materials = top_work['differences']['materials']['unique_to_literature'][:3]
                suggestions.append(
                    f"상위 연구에서 사용된 추가 재료를 고려해보세요: {', '.join(materials)}"
                )
                
        # 조건 최적화 제안
        condition_suggestions = self._suggest_condition_optimization(experiment, performance, similar_works)
        suggestions.extend(condition_suggestions)
        
        # AI 기반 제안 (선택적)
        if self.config.get('benchmark', {}).get('use_ai_suggestions', True):
            ai_suggestions = self._get_ai_suggestions(experiment, performance, similar_works)
            suggestions.extend(ai_suggestions[:2])  # 최대 2개
            
        return suggestions[:5]  # 최대 5개 제안
        
    def _suggest_condition_optimization(
        self,
        experiment: Dict[str, Any],
        performance: Dict[str, Any],
        similar_works: List[Dict[str, Any]]
    ) -> List[str]:
        """조건 최적화 제안"""
        
        suggestions = []
        
        # 상위 연구들의 조건 분석
        if similar_works:
            condition_stats = defaultdict(list)
            
            for work in similar_works[:3]:  # 상위 3개
                for diff in work['differences']['conditions']:
                    param = diff['parameter']
                    lit_val = diff['literature']
                    if isinstance(lit_val, (int, float)):
                        condition_stats[param].append(lit_val)
                        
            # 평균값과 비교
            exp_conditions = experiment.get('conditions', {})
            for param, values in condition_stats.items():
                if param in exp_conditions:
                    exp_val = exp_conditions[param]
                    avg_val = np.mean(values)
                    if isinstance(exp_val, (int, float)):
                        diff_percent = abs(exp_val - avg_val) / avg_val * 100
                        if diff_percent > 20:  # 20% 이상 차이
                            suggestions.append(
                                f"{param}을(를) {avg_val:.1f}(으)로 조정해보세요 "
                                f"(현재: {exp_val:.1f}, 차이: {diff_percent:.0f}%)"
                            )
                            
        return suggestions
        
    def _get_ai_suggestions(
        self,
        experiment: Dict[str, Any],
        performance: Dict[str, Any],
        similar_works: List[Dict[str, Any]]
    ) -> List[str]:
        """AI 기반 개선 제안"""
        
        try:
            # AI 프롬프트 생성
            prompt = f"""
            실험 결과: {json.dumps(experiment.get('results', {}), indent=2)}
            성능 분석: {json.dumps(performance, indent=2)}
            유사 연구: {json.dumps(similar_works[:2], indent=2)}
            
            이 실험을 개선하기 위한 구체적인 제안 2개를 제시해주세요.
            각 제안은 한 문장으로 작성하세요.
            """
            
            # API 호출 (동기)
            response = self.api_manager.generate_text('gemini', prompt)
            
            # 제안 추출
            suggestions = []
            if response:
                lines = response.strip().split('\n')
                for line in lines:
                    if line.strip() and not line.startswith('#'):
                        suggestions.append(line.strip())
                        
            return suggestions[:2]
            
        except Exception as e:
            logger.error(f"AI suggestion error: {e}")
            return []
            
    # ===== 시각화 =====
    
    def _create_visualizations(
        self,
        experiment: Dict[str, Any],
        literature: List[LiteratureData],
        metrics: BenchmarkMetrics
    ) -> Dict[str, Any]:
        """벤치마크 시각화 생성"""
        
        visualizations = {}
        
        # 1. 백분위 게이지 차트
        visualizations['percentile_gauge'] = self._create_percentile_gauge(metrics.percentile_rank)
        
        # 2. 성능 비교 레이더 차트
        visualizations['performance_radar'] = self._create_performance_radar(
            experiment,
            metrics.relative_performance
        )
        
        # 3. 히스토그램 (분포 비교)
        visualizations['distribution_histogram'] = self._create_distribution_histogram(
            experiment,
            literature,
            metrics.relative_performance
        )
        
        # 4. 시계열 트렌드
        visualizations['timeline_trend'] = self._create_timeline_trend(literature)
        
        # 5. 히트맵 (상관관계)
        visualizations['correlation_heatmap'] = self._create_correlation_heatmap(
            experiment,
            literature
        )
        
        return visualizations
        
    def _create_percentile_gauge(self, percentile: float) -> Dict[str, Any]:
        """백분위 게이지 차트"""
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=percentile,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "전체 성능 백분위"},
            delta={'reference': 50, 'increasing': {'color': "green"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 25], 'color': '#ff4444'},
                    {'range': [25, 50], 'color': '#ffaa00'},
                    {'range': [50, 75], 'color': '#44ff44'},
                    {'range': [75, 100], 'color': '#00aa00'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            paper_bgcolor="white",
            font={'color': "darkblue", 'family': "Arial"},
            height=300
        )
        
        return fig.to_dict()
        
    def _create_performance_radar(
        self,
        experiment: Dict[str, Any],
        performance: Dict[str, Any]
    ) -> Dict[str, Any]:
        """성능 레이더 차트"""
        
        if not performance:
            return {}
            
        # 메트릭 추출
        metrics = list(performance.keys())[:8]  # 최대 8개
        exp_values = []
        lit_values = []
        
        for metric in metrics:
            perf = performance[metric]
            exp_values.append(perf['value'])
            lit_values.append(perf['mean'])
            
        # 정규화 (0-100)
        max_values = [max(exp_values[i], lit_values[i]) for i in range(len(metrics))]
        exp_normalized = [(v / m * 100) if m > 0 else 0 for v, m in zip(exp_values, max_values)]
        lit_normalized = [(v / m * 100) if m > 0 else 0 for v, m in zip(lit_values, max_values)]
        
        fig = go.Figure()
        
        # 실험 데이터
        fig.add_trace(go.Scatterpolar(
            r=exp_normalized,
            theta=metrics,
            fill='toself',
            name='내 실험',
            line_color='blue'
        ))
        
        # 문헌 평균
        fig.add_trace(go.Scatterpolar(
            r=lit_normalized,
            theta=metrics,
            fill='toself',
            name='문헌 평균',
            line_color='red',
            opacity=0.6
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            title="성능 비교 (정규화)",
            height=400
        )
        
        return fig.to_dict()
        
    def _create_distribution_histogram(
        self,
        experiment: Dict[str, Any],
        literature: List[LiteratureData],
        performance: Dict[str, Any]
    ) -> Dict[str, Any]:
        """분포 히스토그램"""
        
        if not performance:
            return {}
            
        # 가장 중요한 메트릭 선택
        metric_name = list(performance.keys())[0]
        metric_data = performance[metric_name]
        
        # 문헌 값 수집
        lit_values = []
        for lit in literature:
            if metric_name in lit.metrics:
                value = lit.metrics[metric_name]
                if isinstance(value, (int, float)):
                    lit_values.append(value)
                    
        if not lit_values:
            return {}
            
        fig = go.Figure()
        
        # 히스토그램
        fig.add_trace(go.Histogram(
            x=lit_values,
            name='문헌 분포',
            nbinsx=20,
            opacity=0.7
        ))
        
        # 내 실험 값 표시
        exp_value = metric_data['value']
        fig.add_vline(
            x=exp_value,
            line_dash="dash",
            line_color="red",
            annotation_text=f"내 실험: {exp_value:.2f}"
        )
        
        # 평균 표시
        mean_value = np.mean(lit_values)
        fig.add_vline(
            x=mean_value,
            line_dash="dot",
            line_color="green",
            annotation_text=f"평균: {mean_value:.2f}"
        )
        
        fig.update_layout(
            title=f"{metric_name} 분포",
            xaxis_title=metric_name,
            yaxis_title="빈도",
            showlegend=True,
            height=300
        )
        
        return fig.to_dict()
        
    def _create_timeline_trend(self, literature: List[LiteratureData]) -> Dict[str, Any]:
        """시계열 트렌드"""
        
        # 연도별 데이터 수집
        year_data = defaultdict(list)
        
        for lit in literature:
            if lit.year and lit.metrics:
                for metric, value in lit.metrics.items():
                    if isinstance(value, (int, float)):
                        year_data[lit.year].append(value)
                        
        if not year_data:
            return {}
            
        # 연도별 평균 계산
        years = sorted(year_data.keys())
        averages = [np.mean(year_data[year]) for year in years]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=years,
            y=averages,
            mode='lines+markers',
            name='연도별 평균 성능',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))
        
        # 추세선
        if len(years) > 2:
            z = np.polyfit(years, averages, 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=years,
                y=p(years),
                mode='lines',
                name='추세선',
                line=dict(color='red', dash='dash')
            ))
            
        fig.update_layout(
            title="연구 동향",
            xaxis_title="연도",
            yaxis_title="평균 성능",
            showlegend=True,
            height=300
        )
        
        return fig.to_dict()
        
    def _create_correlation_heatmap(
        self,
        experiment: Dict[str, Any],
        literature: List[LiteratureData]
    ) -> Dict[str, Any]:
        """상관관계 히트맵"""
        
        # 데이터 수집
        data_matrix = []
        metric_names = set()
        
        for lit in literature[:20]:  # 최대 20개
            if lit.metrics:
                row = {}
                for metric, value in lit.metrics.items():
                    if isinstance(value, (int, float)):
                        row[metric] = value
                        metric_names.add(metric)
                if row:
                    data_matrix.append(row)
                    
        if len(data_matrix) < 3 or len(metric_names) < 2:
            return {}
            
        # DataFrame 생성
        df = pd.DataFrame(data_matrix)
        df = df[list(metric_names)]
        df = df.dropna(axis=1, how='all')
        
        # 상관관계 계산
        corr = df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title="메트릭 간 상관관계",
            height=400,
            width=400
        )
        
        return fig.to_dict()
        
    # ===== 리포트 생성 =====
    
    def _generate_report(
        self,
        experiment: Dict[str, Any],
        literature: List[LiteratureData],
        metrics: BenchmarkMetrics
    ) -> Dict[str, Any]:
        """종합 리포트 생성"""
        
        report = {
            'summary': self._generate_summary(experiment, metrics),
            'detailed_analysis': self._generate_detailed_analysis(experiment, literature, metrics),
            'recommendations': metrics.improvement_suggestions,
            'similar_works': metrics.similar_works,
            'methodology': self._generate_methodology_section(),
            'limitations': self._identify_limitations(literature, metrics),
            'next_steps': self._suggest_next_steps(experiment, metrics)
        }
        
        return report
        
    def _generate_summary(
        self,
        experiment: Dict[str, Any],
        metrics: BenchmarkMetrics
    ) -> Dict[str, Any]:
        """요약 생성"""
        
        # 성능 수준 판정
        if metrics.percentile_rank >= 90:
            level = "최상위"
            emoji = "🏆"
        elif metrics.percentile_rank >= 75:
            level = "상위"
            emoji = "🥇"
        elif metrics.percentile_rank >= 50:
            level = "평균 이상"
            emoji = "🥈"
        elif metrics.percentile_rank >= 25:
            level = "평균 이하"
            emoji = "🥉"
        else:
            level = "하위"
            emoji = "📊"
            
        summary = {
            'overall_rating': f"{emoji} {level}",
            'percentile': f"상위 {100 - metrics.percentile_rank:.1f}%",
            'confidence': f"{metrics.confidence_score:.0%}",
            'sample_size': metrics.sample_size,
            'key_findings': self._extract_key_findings(metrics)
        }
        
        return summary
        
    def _generate_detailed_analysis(
        self,
        experiment: Dict[str, Any],
        literature: List[LiteratureData],
        metrics: BenchmarkMetrics
    ) -> Dict[str, Any]:
        """상세 분석"""
        
        analysis = {
            'performance_breakdown': {},
            'strength_areas': [],
            'weakness_areas': [],
            'unique_aspects': [],
            'trend_position': ""
        }
        
        # 성능 분석
        for metric, perf in metrics.relative_performance.items():
            analysis['performance_breakdown'][metric] = {
                'percentile': perf['percentile'],
                'vs_average': f"{((perf['value'] / perf['mean']) - 1) * 100:+.1f}%",
                'interpretation': self._interpret_performance(perf)
            }
            
            # 강점/약점 분류
            if perf['percentile'] >= 75:
                analysis['strength_areas'].append(metric)
            elif perf['percentile'] <= 25:
                analysis['weakness_areas'].append(metric)
                
        # 독특한 측면 식별
        analysis['unique_aspects'] = self._identify_unique_aspects(experiment, literature)
        
        # 트렌드 포지션
        analysis['trend_position'] = self._analyze_trend_position(experiment, literature)
        
        return analysis
        
    def _interpret_performance(self, perf: Dict[str, Any]) -> str:
        """성능 해석"""
        
        if perf['z_score'] > 2:
            return "매우 우수 (2σ 이상)"
        elif perf['z_score'] > 1:
            return "우수 (1-2σ)"
        elif perf['z_score'] > -1:
            return "평균 수준 (-1σ ~ +1σ)"
        elif perf['z_score'] > -2:
            return "평균 이하 (-2σ ~ -1σ)"
        else:
            return "매우 낮음 (-2σ 이하)"
            
    def _extract_key_findings(self, metrics: BenchmarkMetrics) -> List[str]:
        """핵심 발견사항 추출"""
        
        findings = []
        
        # 전체 성능
        findings.append(
            f"전체 성능은 {metrics.percentile_rank:.0f} 백분위로, "
            f"비교 대상 {metrics.sample_size}개 연구 중 상위 {100 - metrics.percentile_rank:.0f}%에 해당합니다."
        )
        
        # 강점 분야
        strong_metrics = [
            m for m, p in metrics.relative_performance.items()
            if p['percentile'] >= 75
        ]
        if strong_metrics:
            findings.append(
                f"특히 {', '.join(strong_metrics[:3])} 분야에서 우수한 성능을 보였습니다."
            )
            
        # 개선 필요 분야
        weak_metrics = [
            m for m, p in metrics.relative_performance.items()
            if p['percentile'] <= 25
        ]
        if weak_metrics:
            findings.append(
                f"{', '.join(weak_metrics[:3])} 분야는 개선이 필요합니다."
            )
            
        return findings
        
    def _identify_unique_aspects(
        self,
        experiment: Dict[str, Any],
        literature: List[LiteratureData]
    ) -> List[str]:
        """독특한 측면 식별"""
        
        unique_aspects = []
        
        # 독특한 재료
        all_materials = set()
        for lit in literature:
            all_materials.update(lit.materials)
            
        exp_materials = set(experiment.get('materials', []))
        unique_materials = exp_materials - all_materials
        
        if unique_materials:
            unique_aspects.append(
                f"독특한 재료 사용: {', '.join(list(unique_materials)[:3])}"
            )
            
        # 독특한 조건
        # ... (구현 생략)
        
        return unique_aspects
        
    def _analyze_trend_position(
        self,
        experiment: Dict[str, Any],
        literature: List[LiteratureData]
    ) -> str:
        """트렌드 위치 분석"""
        
        # 최근 연구 비중
        recent_years = [lit.year for lit in literature if lit.year >= datetime.now().year - 2]
        recent_ratio = len(recent_years) / len(literature) if literature else 0
        
        if recent_ratio > 0.5:
            return "활발히 연구되는 최신 분야"
        elif recent_ratio > 0.2:
            return "지속적으로 연구되는 분야"
        else:
            return "성숙한 연구 분야"
            
    def _generate_methodology_section(self) -> Dict[str, Any]:
        """방법론 섹션"""
        
        return {
            'data_sources': [
                "OpenAlex - 학술 논문 메타데이터",
                "Crossref - DOI 기반 서지정보",
                "PubMed - 생의학 문헌",
                "Materials Project - 재료 물성 데이터"
            ],
            'matching_algorithm': "코사인 유사도 기반 다차원 매칭",
            'statistical_methods': [
                "백분위 순위 계산",
                "Z-score 표준화",
                "피어슨 상관관계 분석"
            ],
            'confidence_calculation': "샘플 크기, 관련성 점수, 데이터 완전성 기반"
        }
        
    def _identify_limitations(
        self,
        literature: List[LiteratureData],
        metrics: BenchmarkMetrics
    ) -> List[str]:
        """분석 한계점 식별"""
        
        limitations = []
        
        if metrics.sample_size < 10:
            limitations.append("비교 가능한 문헌이 적어 통계적 신뢰도가 제한적입니다.")
            
        if metrics.confidence_score < 0.7:
            limitations.append("데이터 매칭 신뢰도가 낮아 해석에 주의가 필요합니다.")
            
        # 시간적 편향
        years = [lit.year for lit in literature if lit.year]
        if years:
            avg_year = np.mean(years)
            if datetime.now().year - avg_year > 5:
                limitations.append("비교 문헌이 다소 오래되어 최신 동향을 반영하지 못할 수 있습니다.")
                
        return limitations
        
    def _suggest_next_steps(
        self,
        experiment: Dict[str, Any],
        metrics: BenchmarkMetrics
    ) -> List[str]:
        """다음 단계 제안"""
        
        next_steps = []
        
        # 성능 수준에 따른 제안
        if metrics.percentile_rank >= 80:
            next_steps.append("우수한 결과를 논문으로 발표하는 것을 고려하세요.")
            next_steps.append("특허 출원 가능성을 검토하세요.")
        elif metrics.percentile_rank >= 50:
            next_steps.append("상위 연구의 방법론을 참고하여 추가 실험을 계획하세요.")
            next_steps.append("개선된 조건으로 재현 실험을 수행하세요.")
        else:
            next_steps.append("근본적인 접근 방법 재검토가 필요할 수 있습니다.")
            next_steps.append("전문가 자문을 구하는 것을 고려하세요.")
            
        # 약점 기반 제안
        if metrics.improvement_suggestions:
            next_steps.append("제안된 개선사항을 우선순위에 따라 적용하세요.")
            
        return next_steps
        
    # ===== 유틸리티 함수 =====
    
    def _prepare_search_params(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """검색 파라미터 준비"""
        
        params = {
            'keywords': [],
            'materials': experiment.get('materials', []),
            'filters': {},
            'limit': 100
        }
        
        # 키워드 추출
        if 'name' in experiment:
            params['keywords'].extend(self._extract_keywords(experiment['name']))
        if 'description' in experiment:
            params['keywords'].extend(self._extract_keywords(experiment['description']))
            
        # 실험 유형
        if 'type' in experiment:
            params['filters']['type'] = experiment['type']
            
        # 날짜 범위 (최근 10년)
        params['filters']['from_year'] = datetime.now().year - 10
        
        return params
        
    def _extract_keywords(self, text: str) -> List[str]:
        """텍스트에서 키워드 추출"""
        
        # 간단한 키워드 추출 (실제로는 NLP 사용 권장)
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        words = text.lower().split()
        keywords = [w for w in words if len(w) > 3 and w not in stopwords]
        
        return keywords[:10]  # 최대 10개
        
    def _extract_metrics_from_text(self, text: str) -> Dict[str, Any]:
        """텍스트에서 메트릭 추출"""
        
        metrics = {}
        
        # 숫자 패턴 찾기 (간단한 예시)
        import re
        
        # 백분율
        percent_pattern = r'(\d+\.?\d*)\s*%'
        percents = re.findall(percent_pattern, text)
        if percents:
            metrics['percentage'] = float(percents[0])
            
        # 온도
        temp_pattern = r'(\d+\.?\d*)\s*°C'
        temps = re.findall(temp_pattern, text)
        if temps:
            metrics['temperature'] = float(temps[0])
            
        # 수율
        yield_pattern = r'yield[:\s]+(\d+\.?\d*)'
        yields = re.findall(yield_pattern, text, re.IGNORECASE)
        if yields:
            metrics['yield'] = float(yields[0])
            
        return metrics
        
    def _deduplicate_literature(self, literature: List[LiteratureData]) -> List[LiteratureData]:
        """중복 문헌 제거"""
        
        seen = set()
        unique = []
        
        for lit in literature:
            # DOI 기반 중복 체크
            if lit.doi and lit.doi in seen:
                continue
                
            # 제목 기반 중복 체크
            title_key = lit.title.lower().strip()
            if title_key in seen:
                continue
                
            seen.add(lit.doi if lit.doi else title_key)
            unique.append(lit)
            
        return unique
        
    def _get_cache_key(self, params: Dict[str, Any]) -> str:
        """캐시 키 생성"""
        
        # 파라미터를 정렬하여 일관된 키 생성
        sorted_params = json.dumps(params, sort_keys=True)
        return hashlib.md5(sorted_params.encode()).hexdigest()
        
    def _calculate_confidence_score(
        self,
        literature: List[LiteratureData],
        performance: Dict[str, Any]
    ) -> float:
        """신뢰도 점수 계산"""
        
        score = 0.0
        weights = {
            'sample_size': 0.3,
            'relevance': 0.3,
            'data_completeness': 0.2,
            'recency': 0.2
        }
        
        # 샘플 크기
        if len(literature) >= 50:
            score += weights['sample_size']
        elif len(literature) >= 20:
            score += weights['sample_size'] * 0.7
        elif len(literature) >= 10:
            score += weights['sample_size'] * 0.5
        else:
            score += weights['sample_size'] * 0.3
            
        # 관련성
        avg_relevance = np.mean([lit.relevance_score for lit in literature]) if literature else 0
        score += weights['relevance'] * avg_relevance
        
        # 데이터 완전성
        if performance:
            completeness = len([p for p in performance.values() if p['sample_size'] > 5]) / len(performance)
            score += weights['data_completeness'] * completeness
            
        # 최신성
        if literature:
            recent_count = len([lit for lit in literature if lit.year >= datetime.now().year - 3])
            recency_ratio = recent_count / len(literature)
            score += weights['recency'] * recency_ratio
            
        return min(score, 1.0)

# ===========================================================================
# 🔧 헬퍼 함수
# ===========================================================================

async def quick_benchmark(
    experiment_data: Dict[str, Any],
    metric_name: str = None
) -> Dict[str, Any]:
    """빠른 벤치마크 분석"""
    
    analyzer = BenchmarkAnalyzer()
    
    # 특정 메트릭만 분석
    if metric_name:
        experiment_data = {
            'name': experiment_data.get('name', 'Quick Analysis'),
            'results': {metric_name: experiment_data['results'].get(metric_name)}
        }
        
    result = await analyzer.analyze_benchmark(
        experiment_data,
        analysis_type='quick'
    )
    
    return {
        'percentile': result.metrics.percentile_rank,
        'sample_size': result.metrics.sample_size,
        'suggestions': result.metrics.improvement_suggestions[:3]
    }
