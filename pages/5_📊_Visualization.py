"""
데이터 시각화 페이지
실험 데이터와 결과를 다양한 형태로 시각화하고 대화형 대시보드 생성
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from typing import Dict, List, Any, Optional, Tuple, Union
import json
from datetime import datetime
import io
import base64
from pathlib import Path

# 프로젝트 임포트
from config.theme_config import COLORS, FONTS, apply_theme
from utils.common_ui import get_common_ui
from utils.data_processor import DataProcessor
from modules.base_module import ExperimentDesign
from utils.api_manager import get_api_manager


class ChartType:
    """차트 타입 정의"""
    # 기본 차트
    SCATTER = "scatter"
    LINE = "line"
    BAR = "bar"
    HISTOGRAM = "histogram"
    BOX = "box"
    VIOLIN = "violin"
    
    # 고급 차트
    HEATMAP = "heatmap"
    CONTOUR = "contour"
    SURFACE_3D = "surface_3d"
    SCATTER_3D = "scatter_3d"
    PARALLEL_COORDINATES = "parallel_coordinates"
    RADAR = "radar"
    SUNBURST = "sunburst"
    TREEMAP = "treemap"
    SANKEY = "sankey"
    
    # 통계 차트
    PARETO = "pareto"
    CONTROL_CHART = "control_chart"
    RESIDUAL_PLOT = "residual_plot"
    QQ_PLOT = "qq_plot"
    
    # 실험 설계 전용
    DESIGN_SPACE = "design_space"
    RESPONSE_SURFACE = "response_surface"
    INTERACTION_PLOT = "interaction_plot"
    MAIN_EFFECTS = "main_effects"


class VisualizationEngine:
    """시각화 엔진"""
    
    def __init__(self):
        self.ui = get_common_ui()
        self.data_processor = DataProcessor()
        self.api_manager = None  # 지연 로딩
        
        # 차트 레지스트리
        self.chart_registry = self._init_chart_registry()
        
        # Plotly 테마 설정
        self._setup_plotly_theme()
    
    def _setup_plotly_theme(self):
        """Plotly 테마 설정"""
        pio.templates["doe_theme"] = go.layout.Template(
            layout=go.Layout(
                font=dict(family=FONTS['body'], color=COLORS['text_primary']),
                plot_bgcolor=COLORS['surface'],
                paper_bgcolor=COLORS['background'],
                colorway=[
                    COLORS['primary'], COLORS['secondary'], 
                    COLORS['accent'], COLORS['warning'], 
                    COLORS['info'], COLORS['success']
                ],
                hovermode='closest',
                hoverlabel=dict(
                    bgcolor=COLORS['surface'],
                    font_size=14,
                    font_family=FONTS['body']
                )
            )
        )
        pio.templates.default = "doe_theme"
    
    def _init_chart_registry(self) -> Dict[str, Dict[str, Any]]:
        """차트 레지스트리 초기화"""
        return {
            ChartType.SCATTER: {
                'name': '산점도',
                'icon': '⚪',
                'description': '두 변수 간의 관계를 점으로 표시',
                'data_requirements': {'min_vars': 2, 'types': ['numeric']},
                'builder': self._build_scatter
            },
            ChartType.LINE: {
                'name': '선 그래프',
                'icon': '📈',
                'description': '시간에 따른 변화나 연속적인 데이터 표시',
                'data_requirements': {'min_vars': 2, 'types': ['numeric', 'datetime']},
                'builder': self._build_line
            },
            ChartType.BAR: {
                'name': '막대 그래프',
                'icon': '📊',
                'description': '범주별 값을 비교',
                'data_requirements': {'min_vars': 1, 'types': ['numeric', 'categorical']},
                'builder': self._build_bar
            },
            ChartType.HEATMAP: {
                'name': '히트맵',
                'icon': '🟥',
                'description': '행렬 형태의 데이터를 색상으로 표현',
                'data_requirements': {'min_vars': 2, 'types': ['numeric']},
                'builder': self._build_heatmap
            },
            ChartType.SURFACE_3D: {
                'name': '3D 표면도',
                'icon': '🏔️',
                'description': '3차원 공간의 표면을 시각화',
                'data_requirements': {'min_vars': 3, 'types': ['numeric']},
                'builder': self._build_surface_3d
            },
            ChartType.SCATTER_3D: {
                'name': '3D 산점도',
                'icon': '🎯',
                'description': '3차원 공간에 데이터 포인트 표시',
                'data_requirements': {'min_vars': 3, 'types': ['numeric']},
                'builder': self._build_scatter_3d
            },
            ChartType.PARALLEL_COORDINATES: {
                'name': '평행 좌표계',
                'icon': '〰️',
                'description': '다차원 데이터를 평행선으로 표현',
                'data_requirements': {'min_vars': 3, 'types': ['numeric']},
                'builder': self._build_parallel_coordinates
            },
            ChartType.BOX: {
                'name': '박스 플롯',
                'icon': '📦',
                'description': '데이터의 분포와 이상치 표시',
                'data_requirements': {'min_vars': 1, 'types': ['numeric']},
                'builder': self._build_box
            },
            ChartType.VIOLIN: {
                'name': '바이올린 플롯',
                'icon': '🎻',
                'description': '분포의 형태를 자세히 표시',
                'data_requirements': {'min_vars': 1, 'types': ['numeric']},
                'builder': self._build_violin
            },
            ChartType.CONTOUR: {
                'name': '등고선도',
                'icon': '🗺️',
                'description': '2차원 평면에 3차원 데이터 표현',
                'data_requirements': {'min_vars': 3, 'types': ['numeric']},
                'builder': self._build_contour
            },
            ChartType.RADAR: {
                'name': '레이더 차트',
                'icon': '🎯',
                'description': '다변량 데이터를 방사형으로 표시',
                'data_requirements': {'min_vars': 3, 'types': ['numeric']},
                'builder': self._build_radar
            },
            ChartType.PARETO: {
                'name': '파레토 차트',
                'icon': '📉',
                'description': '요인별 기여도와 누적 비율 표시',
                'data_requirements': {'min_vars': 2, 'types': ['numeric', 'categorical']},
                'builder': self._build_pareto
            },
            ChartType.RESPONSE_SURFACE: {
                'name': '반응표면도',
                'icon': '🌋',
                'description': '실험 설계의 반응표면 모델 시각화',
                'data_requirements': {'min_vars': 3, 'types': ['numeric']},
                'builder': self._build_response_surface
            },
            ChartType.INTERACTION_PLOT: {
                'name': '교호작용도',
                'icon': '🔀',
                'description': '요인 간 교호작용 효과 표시',
                'data_requirements': {'min_vars': 3, 'types': ['numeric']},
                'builder': self._build_interaction_plot
            },
            ChartType.MAIN_EFFECTS: {
                'name': '주효과도',
                'icon': '📊',
                'description': '각 요인의 주효과 표시',
                'data_requirements': {'min_vars': 2, 'types': ['numeric']},
                'builder': self._build_main_effects
            }
        }
    
    def recommend_charts(self, df: pd.DataFrame, 
                        show_details: bool = None) -> List[Dict[str, Any]]:
        """데이터에 적합한 차트 추천 (AI 지원)"""
        recommendations = []
        
        # 데이터 특성 분석
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        n_numeric = len(numeric_cols)
        n_categorical = len(categorical_cols)
        n_rows = len(df)
        
        # 규칙 기반 추천
        if n_numeric >= 2:
            recommendations.append({
                'type': ChartType.SCATTER,
                'score': 0.9,
                'reason': '연속형 변수 간 관계 파악에 적합'
            })
            
            if n_numeric >= 3:
                recommendations.append({
                    'type': ChartType.SCATTER_3D,
                    'score': 0.85,
                    'reason': '3차원 공간에서 패턴 발견 가능'
                })
                
                recommendations.append({
                    'type': ChartType.PARALLEL_COORDINATES,
                    'score': 0.8,
                    'reason': '다차원 데이터의 패턴 비교에 유용'
                })
        
        if n_categorical >= 1 and n_numeric >= 1:
            recommendations.append({
                'type': ChartType.BAR,
                'score': 0.85,
                'reason': '범주별 값 비교에 최적'
            })
            
            recommendations.append({
                'type': ChartType.BOX,
                'score': 0.8,
                'reason': '그룹별 분포 비교에 효과적'
            })
        
        # AI 기반 추천 (온라인 시)
        if self._should_use_ai() and st.session_state.get('show_ai_details', show_details):
            ai_recommendations = self._get_ai_recommendations(df)
            if ai_recommendations:
                recommendations.extend(ai_recommendations)
        
        # 점수 기준 정렬
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations[:5]  # 상위 5개 추천
    
    def _should_use_ai(self) -> bool:
        """AI 사용 가능 여부 확인"""
        if not self.api_manager:
            self.api_manager = get_api_manager()
        return self.api_manager and self.api_manager.is_available()
    
    def _get_ai_recommendations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """AI 기반 차트 추천"""
        try:
            # 데이터 요약
            data_summary = {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict(),
                'sample': df.head(5).to_dict()
            }
            
            prompt = f"""
            다음 데이터에 적합한 시각화 차트를 추천해주세요:
            
            데이터 요약:
            {json.dumps(data_summary, indent=2, default=str)}
            
            응답 형식:
            1. 추천 차트 타입 (최대 3개)
            2. 각 차트 선택 이유
            3. 대안 차트와 비교
            4. 시각화 시 주의사항
            
            차트 타입은 다음 중에서 선택:
            {', '.join(self.chart_registry.keys())}
            """
            
            response = self.api_manager.query(prompt, "visualization_recommendation")
            
            # AI 응답 파싱 (실제 구현 시 더 정교하게)
            ai_recommendations = []
            # ... AI 응답 파싱 로직
            
            return ai_recommendations
            
        except Exception as e:
            st.warning(f"AI 추천을 가져올 수 없습니다: {str(e)}")
            return []
    
    def create_chart(self, chart_type: str, df: pd.DataFrame, 
                    config: Dict[str, Any]) -> go.Figure:
        """차트 생성"""
        if chart_type not in self.chart_registry:
            raise ValueError(f"Unknown chart type: {chart_type}")
        
        builder = self.chart_registry[chart_type]['builder']
        return builder(df, config)
    
    # 차트 빌더 메서드들
    def _build_scatter(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """산점도 생성"""
        fig = px.scatter(
            df,
            x=config.get('x'),
            y=config.get('y'),
            color=config.get('color'),
            size=config.get('size'),
            hover_data=config.get('hover_data', []),
            title=config.get('title', '산점도'),
            labels=config.get('labels', {})
        )
        
        # 추세선 추가 옵션
        if config.get('trendline'):
            fig = px.scatter(
                df,
                x=config.get('x'),
                y=config.get('y'),
                color=config.get('color'),
                size=config.get('size'),
                trendline="ols",
                title=config.get('title', '산점도')
            )
        
        return self._apply_layout_updates(fig, config)
    
    def _build_line(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """선 그래프 생성"""
        fig = px.line(
            df,
            x=config.get('x'),
            y=config.get('y'),
            color=config.get('color'),
            line_dash=config.get('line_dash'),
            title=config.get('title', '선 그래프'),
            labels=config.get('labels', {})
        )
        
        # 마커 추가 옵션
        if config.get('markers'):
            fig.update_traces(mode='lines+markers')
        
        return self._apply_layout_updates(fig, config)
    
    def _build_bar(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """막대 그래프 생성"""
        orientation = config.get('orientation', 'v')
        
        if orientation == 'v':
            fig = px.bar(
                df,
                x=config.get('x'),
                y=config.get('y'),
                color=config.get('color'),
                title=config.get('title', '막대 그래프'),
                labels=config.get('labels', {})
            )
        else:
            fig = px.bar(
                df,
                x=config.get('y'),
                y=config.get('x'),
                color=config.get('color'),
                orientation='h',
                title=config.get('title', '막대 그래프'),
                labels=config.get('labels', {})
            )
        
        return self._apply_layout_updates(fig, config)
    
    def _build_heatmap(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """히트맵 생성"""
        # 상관관계 히트맵
        if config.get('correlation'):
            corr_matrix = df[config.get('columns', df.columns)].corr()
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
        else:
            fig = px.imshow(
                df.pivot_table(
                    index=config.get('y'),
                    columns=config.get('x'),
                    values=config.get('values')
                ),
                title=config.get('title', '히트맵'),
                labels=config.get('labels', {})
            )
        
        return self._apply_layout_updates(fig, config)
    
    def _build_surface_3d(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """3D 표면도 생성"""
        # 격자 데이터 생성
        x = df[config.get('x')].unique()
        y = df[config.get('y')].unique()
        z = df.pivot_table(
            index=config.get('y'),
            columns=config.get('x'),
            values=config.get('z')
        ).values
        
        fig = go.Figure(data=[go.Surface(
            x=x,
            y=y,
            z=z,
            colorscale=config.get('colorscale', 'Viridis'),
            showscale=True
        )])
        
        fig.update_layout(
            title=config.get('title', '3D 표면도'),
            scene=dict(
                xaxis_title=config.get('x'),
                yaxis_title=config.get('y'),
                zaxis_title=config.get('z')
            )
        )
        
        return self._apply_layout_updates(fig, config)
    
    def _build_scatter_3d(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """3D 산점도 생성"""
        fig = px.scatter_3d(
            df,
            x=config.get('x'),
            y=config.get('y'),
            z=config.get('z'),
            color=config.get('color'),
            size=config.get('size'),
            hover_data=config.get('hover_data', []),
            title=config.get('title', '3D 산점도'),
            labels=config.get('labels', {})
        )
        
        return self._apply_layout_updates(fig, config)
    
    def _build_parallel_coordinates(self, df: pd.DataFrame, 
                                   config: Dict[str, Any]) -> go.Figure:
        """평행 좌표계 생성"""
        dimensions = []
        
        for col in config.get('columns', df.select_dtypes(include=[np.number]).columns):
            dimensions.append(
                dict(
                    label=col,
                    values=df[col],
                    range=[df[col].min(), df[col].max()]
                )
            )
        
        fig = go.Figure(data=go.Parcoords(
            line=dict(
                color=df[config.get('color')] if config.get('color') else None,
                colorscale=config.get('colorscale', 'Viridis')
            ),
            dimensions=dimensions
        ))
        
        fig.update_layout(title=config.get('title', '평행 좌표계'))
        
        return self._apply_layout_updates(fig, config)
    
    def _build_box(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """박스 플롯 생성"""
        fig = px.box(
            df,
            x=config.get('x'),
            y=config.get('y'),
            color=config.get('color'),
            notched=config.get('notched', False),
            title=config.get('title', '박스 플롯'),
            labels=config.get('labels', {})
        )
        
        # 포인트 표시 옵션
        if config.get('points'):
            fig.update_traces(boxpoints='all', jitter=0.3)
        
        return self._apply_layout_updates(fig, config)
    
    def _build_violin(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """바이올린 플롯 생성"""
        fig = px.violin(
            df,
            x=config.get('x'),
            y=config.get('y'),
            color=config.get('color'),
            box=config.get('box', True),
            title=config.get('title', '바이올린 플롯'),
            labels=config.get('labels', {})
        )
        
        return self._apply_layout_updates(fig, config)
    
    def _build_contour(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """등고선도 생성"""
        fig = go.Figure(data=go.Contour(
            x=df[config.get('x')],
            y=df[config.get('y')],
            z=df[config.get('z')],
            colorscale=config.get('colorscale', 'Viridis'),
            showscale=True,
            contours=dict(
                showlabels=config.get('show_labels', True),
                labelfont=dict(size=12, color='white')
            )
        ))
        
        fig.update_layout(
            title=config.get('title', '등고선도'),
            xaxis_title=config.get('x'),
            yaxis_title=config.get('y')
        )
        
        return self._apply_layout_updates(fig, config)
    
    def _build_radar(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """레이더 차트 생성"""
        categories = config.get('categories', df.columns.tolist())
        
        fig = go.Figure()
        
        for idx, row in df.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=row[categories].values,
                theta=categories,
                fill='toself',
                name=str(row.get(config.get('name_column', 'index'), idx))
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, df[categories].max().max()]
                )
            ),
            showlegend=True,
            title=config.get('title', '레이더 차트')
        )
        
        return self._apply_layout_updates(fig, config)
    
    def _build_pareto(self, df: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
        """파레토 차트 생성"""
        # 데이터 정렬
        sorted_df = df.sort_values(config.get('value'), ascending=False)
        
        # 누적 비율 계산
        total = sorted_df[config.get('value')].sum()
        sorted_df['cumulative_percent'] = (
            sorted_df[config.get('value')].cumsum() / total * 100
        )
        
        # 복합 차트 생성
        fig = make_subplots(
            specs=[[{"secondary_y": True}]],
            subplot_titles=[config.get('title', '파레토 차트')]
        )
        
        # 막대 그래프
        fig.add_trace(
            go.Bar(
                x=sorted_df[config.get('category')],
                y=sorted_df[config.get('value')],
                name='값',
                marker_color=COLORS['primary']
            ),
            secondary_y=False
        )
        
        # 누적 선 그래프
        fig.add_trace(
            go.Scatter(
                x=sorted_df[config.get('category')],
                y=sorted_df['cumulative_percent'],
                name='누적 %',
                mode='lines+markers',
                marker_color=COLORS['accent']
            ),
            secondary_y=True
        )
        
        # 80% 기준선
        fig.add_hline(
            y=80, 
            line_dash="dash", 
            line_color="gray",
            secondary_y=True,
            annotation_text="80%"
        )
        
        fig.update_yaxes(title_text="값", secondary_y=False)
        fig.update_yaxes(title_text="누적 %", secondary_y=True, range=[0, 100])
        
        return self._apply_layout_updates(fig, config)
    
    def _build_response_surface(self, df: pd.DataFrame, 
                               config: Dict[str, Any]) -> go.Figure:
        """반응표면도 생성 (실험 설계 전용)"""
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        
        # 입력 변수
        X = df[[config.get('x1'), config.get('x2')]].values
        y = df[config.get('response')].values
        
        # 2차 다항식 모델 피팅
        poly = PolynomialFeatures(degree=2, include_bias=True)
        X_poly = poly.fit_transform(X)
        
        model = LinearRegression()
        model.fit(X_poly, y)
        
        # 예측을 위한 격자 생성
        x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 50)
        x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 50)
        x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
        
        X_grid = np.column_stack([x1_grid.ravel(), x2_grid.ravel()])
        X_grid_poly = poly.transform(X_grid)
        y_pred = model.predict(X_grid_poly).reshape(x1_grid.shape)
        
        # 3D 표면도
        fig = go.Figure()
        
        # 반응표면
        fig.add_trace(go.Surface(
            x=x1_range,
            y=x2_range,
            z=y_pred,
            colorscale='Viridis',
            opacity=0.8,
            name='반응표면'
        ))
        
        # 실험 데이터 포인트
        fig.add_trace(go.Scatter3d(
            x=X[:, 0],
            y=X[:, 1],
            z=y,
            mode='markers',
            marker=dict(size=8, color=COLORS['accent']),
            name='실험 데이터'
        ))
        
        fig.update_layout(
            title=config.get('title', '반응표면도'),
            scene=dict(
                xaxis_title=config.get('x1'),
                yaxis_title=config.get('x2'),
                zaxis_title=config.get('response')
            )
        )
        
        return self._apply_layout_updates(fig, config)
    
    def _build_interaction_plot(self, df: pd.DataFrame, 
                               config: Dict[str, Any]) -> go.Figure:
        """교호작용도 생성"""
        factor1 = config.get('factor1')
        factor2 = config.get('factor2')
        response = config.get('response')
        
        # 평균값 계산
        interaction_data = df.groupby([factor1, factor2])[response].mean().reset_index()
        
        fig = go.Figure()
        
        # 각 factor2 수준별로 선 그리기
        for level in interaction_data[factor2].unique():
            data_subset = interaction_data[interaction_data[factor2] == level]
            fig.add_trace(go.Scatter(
                x=data_subset[factor1],
                y=data_subset[response],
                mode='lines+markers',
                name=f'{factor2}={level}'
            ))
        
        fig.update_layout(
            title=config.get('title', f'{factor1} × {factor2} 교호작용'),
            xaxis_title=factor1,
            yaxis_title=f'{response} (평균)',
            hovermode='x unified'
        )
        
        return self._apply_layout_updates(fig, config)
    
    def _build_main_effects(self, df: pd.DataFrame, 
                           config: Dict[str, Any]) -> go.Figure:
        """주효과도 생성"""
        factors = config.get('factors', [])
        response = config.get('response')
        
        n_factors = len(factors)
        fig = make_subplots(
            rows=1, cols=n_factors,
            subplot_titles=factors,
            horizontal_spacing=0.1
        )
        
        for i, factor in enumerate(factors, 1):
            # 각 수준별 평균 계산
            means = df.groupby(factor)[response].mean().reset_index()
            
            fig.add_trace(
                go.Scatter(
                    x=means[factor],
                    y=means[response],
                    mode='lines+markers',
                    name=factor,
                    showlegend=False
                ),
                row=1, col=i
            )
            
            # 전체 평균선 추가
            overall_mean = df[response].mean()
            fig.add_hline(
                y=overall_mean, 
                line_dash="dash", 
                line_color="gray",
                row=1, col=i
            )
        
        fig.update_layout(
            title=config.get('title', '주효과도'),
            height=400
        )
        
        # Y축 범위 통일
        y_min = df[response].min() * 0.95
        y_max = df[response].max() * 1.05
        for i in range(1, n_factors + 1):
            fig.update_yaxes(range=[y_min, y_max], row=1, col=i)
            if i == 1:
                fig.update_yaxes(title_text=response, row=1, col=i)
        
        return self._apply_layout_updates(fig, config)
    
    def _apply_layout_updates(self, fig: go.Figure, 
                             config: Dict[str, Any]) -> go.Figure:
        """공통 레이아웃 업데이트"""
        # 크기 설정
        if config.get('height'):
            fig.update_layout(height=config['height'])
        if config.get('width'):
            fig.update_layout(width=config['width'])
        
        # 여백 설정
        fig.update_layout(
            margin=dict(l=40, r=40, t=60, b=40),
            autosize=True
        )
        
        # 그리드 설정
        if config.get('show_grid', True):
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        # 범례 위치
        if config.get('legend_position'):
            fig.update_layout(legend=dict(
                orientation=config['legend_position'].get('orientation', 'v'),
                yanchor=config['legend_position'].get('yanchor', 'top'),
                y=config['legend_position'].get('y', 1),
                xanchor=config['legend_position'].get('xanchor', 'left'),
                x=config['legend_position'].get('x', 1.02)
            ))
        
        return fig
    
    def export_chart(self, fig: go.Figure, format: str, 
                    filename: str = None) -> Union[bytes, str]:
        """차트 내보내기"""
        if filename is None:
            filename = f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if format == 'png':
            return fig.to_image(format='png', width=1200, height=800)
        elif format == 'svg':
            return fig.to_image(format='svg')
        elif format == 'pdf':
            return fig.to_image(format='pdf')
        elif format == 'html':
            return fig.to_html(include_plotlyjs='cdn')
        elif format == 'json':
            return fig.to_json()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def create_dashboard(self, charts: List[Tuple[go.Figure, Dict[str, Any]]]) -> go.Figure:
        """대시보드 생성"""
        n_charts = len(charts)
        
        # 레이아웃 계산
        if n_charts <= 2:
            rows, cols = 1, n_charts
        elif n_charts <= 4:
            rows, cols = 2, 2
        elif n_charts <= 6:
            rows, cols = 2, 3
        else:
            rows, cols = 3, 3
        
        # 서브플롯 생성
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[c[1].get('title', f'Chart {i+1}') 
                           for i, c in enumerate(charts[:rows*cols])],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # 차트 추가
        for idx, (chart, config) in enumerate(charts[:rows*cols]):
            row = idx // cols + 1
            col = idx % cols + 1
            
            for trace in chart.data:
                fig.add_trace(trace, row=row, col=col)
        
        # 레이아웃 업데이트
        fig.update_layout(
            height=300 * rows,
            showlegend=False,
            title_text="대시보드",
            title_x=0.5
        )
        
        return fig


def render():
    """시각화 페이지 렌더링"""
    st.title("📊 데이터 시각화")
    st.markdown("실험 데이터를 다양한 차트로 시각화하고 인사이트를 발견하세요")
    
    # 테마 적용
    apply_theme()
    
    # 시각화 엔진 초기화
    viz_engine = VisualizationEngine()
    ui = get_common_ui()
    
    # 사이드바 - 데이터 선택
    with st.sidebar:
        st.subheader("📁 데이터 선택")
        
        data_source = st.radio(
            "데이터 소스",
            ["파일 업로드", "프로젝트 데이터", "예제 데이터"]
        )
        
        df = None
        
        if data_source == "파일 업로드":
            uploaded_file = st.file_uploader(
                "데이터 파일 선택",
                type=['csv', 'xlsx', 'xls', 'parquet']
            )
            
            if uploaded_file:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                        df = pd.read_excel(uploaded_file)
                    elif uploaded_file.name.endswith('.parquet'):
                        df = pd.read_parquet(uploaded_file)
                    
                    st.success(f"✅ 데이터 로드 완료 ({df.shape[0]} × {df.shape[1]})")
                except Exception as e:
                    st.error(f"데이터 로드 실패: {str(e)}")
        
        elif data_source == "프로젝트 데이터":
            # 프로젝트 선택
            project = st.selectbox(
                "프로젝트 선택",
                ["프로젝트 1", "프로젝트 2", "프로젝트 3"]  # 실제 구현 시 DB에서 가져옴
            )
            
            # 실험 데이터 로드 (예시)
            # df = load_project_data(project)
            st.info("프로젝트 데이터 로드 기능은 구현 중입니다")
        
        else:  # 예제 데이터
            example_type = st.selectbox(
                "예제 데이터 선택",
                ["화학 합성 실험", "재료 특성 분석", "공정 최적화"]
            )
            
            # 예제 데이터 생성
            np.random.seed(42)
            if example_type == "화학 합성 실험":
                df = pd.DataFrame({
                    '온도': np.random.uniform(20, 100, 50),
                    '시간': np.random.uniform(30, 180, 50),
                    '농도': np.random.uniform(0.1, 2.0, 50),
                    'pH': np.random.uniform(4, 10, 50),
                    '수율': np.random.uniform(40, 95, 50),
                    '순도': np.random.uniform(80, 99, 50),
                    '촉매': np.random.choice(['A', 'B', 'C'], 50)
                })
            
            st.success("✅ 예제 데이터 로드 완료")
    
    # 메인 영역
    if df is not None:
        # AI 설명 모드 토글
        col1, col2 = st.columns([4, 1])
        with col2:
            show_ai_details = st.checkbox(
                "🔍 AI 설명",
                value=st.session_state.get('show_ai_details', True),
                help="차트 추천 시 AI의 추론 과정을 표시합니다"
            )
            st.session_state.show_ai_details = show_ai_details
        
        # 탭 구성
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 차트 갤러리", 
            "🎯 차트 추천", 
            "🎨 대시보드 빌더",
            "📈 실험 설계 전용",
            "💾 내보내기"
        ])
        
        with tab1:
            st.subheader("차트 갤러리")
            
            # 차트 타입 선택
            chart_type = st.selectbox(
                "차트 타입 선택",
                options=list(viz_engine.chart_registry.keys()),
                format_func=lambda x: f"{viz_engine.chart_registry[x]['icon']} {viz_engine.chart_registry[x]['name']}"
            )
            
            # 차트 설정
            st.write("### 차트 설정")
            
            chart_info = viz_engine.chart_registry[chart_type]
            config = {}
            
            # 데이터 컬럼 선택
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            all_cols = df.columns.tolist()
            
            col1, col2 = st.columns(2)
            
            with col1:
                if chart_type in [ChartType.SCATTER, ChartType.LINE, ChartType.BAR]:
                    config['x'] = st.selectbox("X축", all_cols)
                    config['y'] = st.selectbox("Y축", numeric_cols)
                    
                elif chart_type in [ChartType.SCATTER_3D, ChartType.SURFACE_3D]:
                    config['x'] = st.selectbox("X축", numeric_cols)
                    config['y'] = st.selectbox("Y축", numeric_cols)
                    config['z'] = st.selectbox("Z축", numeric_cols)
                
                elif chart_type == ChartType.HEATMAP:
                    if st.checkbox("상관관계 히트맵"):
                        config['correlation'] = True
                        config['columns'] = st.multiselect(
                            "변수 선택", numeric_cols, default=numeric_cols[:5]
                        )
                    else:
                        config['x'] = st.selectbox("X축", all_cols)
                        config['y'] = st.selectbox("Y축", all_cols)
                        config['values'] = st.selectbox("값", numeric_cols)
            
            with col2:
                # 공통 옵션
                config['title'] = st.text_input(
                    "차트 제목", 
                    value=f"{chart_info['name']} - {datetime.now().strftime('%Y-%m-%d')}"
                )
                
                if categorical_cols:
                    config['color'] = st.selectbox(
                        "색상 구분", 
                        [None] + categorical_cols + numeric_cols
                    )
                
                # 차트별 특수 옵션
                if chart_type == ChartType.SCATTER:
                    config['trendline'] = st.checkbox("추세선 표시")
                    config['size'] = st.selectbox("크기", [None] + numeric_cols)
                
                elif chart_type == ChartType.BOX:
                    config['points'] = st.checkbox("데이터 포인트 표시")
                    config['notched'] = st.checkbox("노치 표시")
            
            # 고급 옵션
            with st.expander("고급 옵션"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    config['height'] = st.slider("높이", 400, 1000, 600)
                with col2:
                    config['show_grid'] = st.checkbox("그리드 표시", True)
                with col3:
                    config['colorscale'] = st.selectbox(
                        "색상 스케일",
                        ['Viridis', 'Plasma', 'Inferno', 'RdBu', 'Blues', 'Reds']
                    )
            
            # 차트 생성 및 표시
            try:
                with st.spinner("차트 생성 중..."):
                    fig = viz_engine.create_chart(chart_type, df, config)
                    st.plotly_chart(fig, use_container_width=True)
                
                # 차트 설명
                st.info(f"💡 {chart_info['description']}")
                
            except Exception as e:
                st.error(f"차트 생성 실패: {str(e)}")
        
        with tab2:
            st.subheader("🎯 AI 기반 차트 추천")
            
            # 차트 추천
            with st.spinner("데이터 분석 및 차트 추천 중..."):
                recommendations = viz_engine.recommend_charts(df, show_ai_details)
            
            if recommendations:
                st.write("### 추천 차트")
                
                for i, rec in enumerate(recommendations):
                    chart_info = viz_engine.chart_registry.get(rec['type'])
                    if chart_info:
                        col1, col2, col3 = st.columns([1, 3, 1])
                        
                        with col1:
                            st.metric("추천도", f"{rec['score']*100:.0f}%")
                        
                        with col2:
                            st.write(f"**{chart_info['icon']} {chart_info['name']}**")
                            st.caption(rec['reason'])
                        
                        with col3:
                            if st.button("생성", key=f"create_{i}"):
                                st.session_state.selected_chart = rec['type']
                                st.rerun()
                
                # AI 상세 설명 (조건부)
                if show_ai_details and viz_engine._should_use_ai():
                    with st.expander("🤖 AI 추천 상세 설명", expanded=True):
                        st.write("#### 추론 과정")
                        st.write("""
                        1. **데이터 특성 분석**
                           - 숫자형 변수: X개
                           - 범주형 변수: Y개
                           - 데이터 크기: N개 행
                        
                        2. **패턴 인식**
                           - 변수 간 상관관계 확인
                           - 분포 형태 분석
                           - 시계열 패턴 탐지
                        
                        3. **차트 매칭**
                           - 데이터 특성과 차트 요구사항 비교
                           - 시각화 목적 고려
                           - 사용자 선호도 반영
                        """)
                        
                        st.write("#### 대안 차트")
                        st.write("다른 가능한 시각화 옵션들도 고려해보세요:")
                        # ... 대안 차트 목록
        
        with tab3:
            st.subheader("🎨 대시보드 빌더")
            st.info("여러 차트를 조합하여 대시보드를 만들어보세요")
            
            # 대시보드 차트 목록
            if 'dashboard_charts' not in st.session_state:
                st.session_state.dashboard_charts = []
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write("### 대시보드 구성")
                
                if st.session_state.dashboard_charts:
                    # 대시보드 미리보기
                    dashboard_config = []
                    for chart_config in st.session_state.dashboard_charts:
                        fig = viz_engine.create_chart(
                            chart_config['type'],
                            df,
                            chart_config['config']
                        )
                        dashboard_config.append((fig, chart_config['config']))
                    
                    dashboard_fig = viz_engine.create_dashboard(dashboard_config)
                    st.plotly_chart(dashboard_fig, use_container_width=True)
                else:
                    st.info("차트를 추가하여 대시보드를 구성하세요")
            
            with col2:
                st.write("### 차트 추가")
                
                # 빠른 차트 추가
                quick_chart = st.selectbox(
                    "차트 타입",
                    options=list(viz_engine.chart_registry.keys()),
                    format_func=lambda x: viz_engine.chart_registry[x]['name']
                )
                
                if st.button("➕ 추가"):
                    # 기본 설정으로 차트 추가
                    st.session_state.dashboard_charts.append({
                        'type': quick_chart,
                        'config': {
                            'title': f"차트 {len(st.session_state.dashboard_charts) + 1}"
                        }
                    })
                    st.rerun()
                
                if st.button("🗑️ 초기화"):
                    st.session_state.dashboard_charts = []
                    st.rerun()
        
        with tab4:
            st.subheader("📈 실험 설계 전용 차트")
            
            # 실험 설계 전용 차트 타입
            doe_charts = [
                ChartType.RESPONSE_SURFACE,
                ChartType.INTERACTION_PLOT,
                ChartType.MAIN_EFFECTS,
                ChartType.PARETO
            ]
            
            doe_chart_type = st.selectbox(
                "차트 타입 선택",
                options=doe_charts,
                format_func=lambda x: viz_engine.chart_registry[x]['name']
            )
            
            # 실험 설계 차트 설정
            doe_config = {'title': viz_engine.chart_registry[doe_chart_type]['name']}
            
            if doe_chart_type == ChartType.RESPONSE_SURFACE:
                col1, col2 = st.columns(2)
                with col1:
                    doe_config['x1'] = st.selectbox("요인 1", numeric_cols)
                    doe_config['x2'] = st.selectbox("요인 2", numeric_cols)
                with col2:
                    doe_config['response'] = st.selectbox("반응변수", numeric_cols)
            
            elif doe_chart_type == ChartType.INTERACTION_PLOT:
                col1, col2 = st.columns(2)
                with col1:
                    doe_config['factor1'] = st.selectbox("요인 1", all_cols)
                    doe_config['factor2'] = st.selectbox("요인 2", all_cols)
                with col2:
                    doe_config['response'] = st.selectbox("반응변수", numeric_cols)
            
            elif doe_chart_type == ChartType.MAIN_EFFECTS:
                doe_config['factors'] = st.multiselect("요인 선택", all_cols)
                doe_config['response'] = st.selectbox("반응변수", numeric_cols)
            
            elif doe_chart_type == ChartType.PARETO:
                doe_config['category'] = st.selectbox("범주", all_cols)
                doe_config['value'] = st.selectbox("값", numeric_cols)
            
            # 차트 생성
            if st.button("차트 생성", key="create_doe"):
                try:
                    with st.spinner("실험 설계 차트 생성 중..."):
                        fig = viz_engine.create_chart(doe_chart_type, df, doe_config)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 차트 해석 (AI 지원)
                        if show_ai_details:
                            with st.expander("📊 차트 해석", expanded=True):
                                st.write("""
                                **주요 발견사항:**
                                - 요인 A의 주효과가 가장 크게 나타남
                                - 요인 B와 C 간 교호작용 존재
                                - 최적 조건: A=높음, B=중간, C=낮음
                                
                                **권장사항:**
                                - 추가 실험으로 최적 영역 정밀 탐색
                                - 재현성 확인 실험 수행
                                """)
                
                except Exception as e:
                    st.error(f"차트 생성 실패: {str(e)}")
        
        with tab5:
            st.subheader("💾 내보내기")
            
            # 내보내기 설정
            col1, col2 = st.columns(2)
            
            with col1:
                export_format = st.selectbox(
                    "내보내기 형식",
                    ['png', 'svg', 'pdf', 'html', 'json'],
                    format_func=lambda x: {
                        'png': 'PNG 이미지',
                        'svg': 'SVG 벡터',
                        'pdf': 'PDF 문서',
                        'html': 'HTML (대화형)',
                        'json': 'JSON 데이터'
                    }[x]
                )
                
                filename = st.text_input(
                    "파일명",
                    value=f"visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
            
            with col2:
                st.write("### 내보내기 옵션")
                
                if export_format in ['png', 'pdf']:
                    width = st.number_input("너비 (px)", 800, 3000, 1200)
                    height = st.number_input("높이 (px)", 600, 2000, 800)
                
                include_watermark = st.checkbox("워터마크 포함")
            
            # 내보내기 실행
            if st.button("📥 내보내기", type="primary"):
                # 현재 표시된 차트 가져오기 (예시)
                # 실제 구현 시 현재 활성 차트 추적 필요
                sample_fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[4, 5, 6])])
                
                try:
                    export_data = viz_engine.export_chart(
                        sample_fig, 
                        export_format,
                        filename
                    )
                    
                    if export_format in ['png', 'svg', 'pdf']:
                        st.download_button(
                            label=f"⬇️ {export_format.upper()} 다운로드",
                            data=export_data,
                            file_name=f"{filename}.{export_format}",
                            mime=f"image/{export_format}"
                        )
                    else:
                        st.download_button(
                            label=f"⬇️ {export_format.upper()} 다운로드",
                            data=export_data,
                            file_name=f"{filename}.{export_format}",
                            mime="text/plain"
                        )
                    
                    st.success("✅ 내보내기 준비 완료!")
                    
                except Exception as e:
                    st.error(f"내보내기 실패: {str(e)}")
    
    else:
        # 데이터가 없을 때
        ui.render_empty_state(
            "데이터를 선택하거나 업로드하여 시작하세요",
            "📊"
        )
        
        # 도움말
        with st.expander("🔍 시작하기"):
            st.write("""
            ### 데이터 시각화 시작하기
            
            1. **데이터 준비**
               - CSV, Excel 파일 업로드
               - 프로젝트 데이터 선택
               - 예제 데이터로 연습
            
            2. **차트 선택**
               - 차트 갤러리에서 직접 선택
               - AI 추천 기능 활용
               - 실험 설계 전용 차트 사용
            
            3. **커스터마이징**
               - 색상, 크기, 스타일 조정
               - 제목과 라벨 편집
               - 대화형 기능 추가
            
            4. **공유 및 내보내기**
               - 다양한 형식으로 저장
               - 대시보드 생성
               - 보고서에 삽입
            """)


if __name__ == "__main__":
    render()
