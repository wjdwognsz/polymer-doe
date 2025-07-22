"""
utils/data_processor.py - 데이터 처리 엔진
Universal DOE Platform의 핵심 데이터 처리 및 분석 엔진
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import warnings
import logging
from abc import ABC, abstractmethod
from enum import Enum
import copy
from functools import lru_cache

# 과학 계산
from scipy import stats, optimize, signal
from scipy.spatial.distance import cdist
from scipy.stats import f_oneway, ttest_ind, chi2_contingency
from scipy.optimize import minimize, differential_evolution
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.diagnostic import het_breuschpagan

# 기계학습
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    PolynomialFeatures, PowerTransformer
)
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression

# 실험 설계
import pyDOE2 as doe
from pyDOE2 import *

# 최적화
try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize as pymoo_minimize
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

# 시각화 지원
import matplotlib.pyplot as plt
import seaborn as sns

# 내부 모듈
from utils.api_manager import APIManager
from utils.common_ui import show_error, show_warning, show_info

# 로깅 설정
logger = logging.getLogger(__name__)


# ============= 데이터 구조 정의 =============

@dataclass
class Factor:
    """실험 인자 정의"""
    name: str
    type: str  # 'continuous', 'categorical', 'discrete'
    low: Optional[float] = None
    high: Optional[float] = None
    levels: Optional[List[Any]] = None
    unit: Optional[str] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        if self.type == 'continuous' and (self.low is None or self.high is None):
            raise ValueError(f"연속형 인자 '{self.name}'는 low와 high 값이 필요합니다")
        if self.type == 'categorical' and not self.levels:
            raise ValueError(f"범주형 인자 '{self.name}'는 levels가 필요합니다")


@dataclass
class Response:
    """반응변수 정의"""
    name: str
    unit: Optional[str] = None
    goal: str = 'maximize'  # 'maximize', 'minimize', 'target'
    target: Optional[float] = None
    lower_limit: Optional[float] = None
    upper_limit: Optional[float] = None
    description: Optional[str] = None


@dataclass
class ExperimentDesign:
    """실험 설계 결과"""
    design_matrix: pd.DataFrame
    design_type: str
    factors: List[Factor]
    responses: List[Response]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_run_count(self) -> int:
        return len(self.design_matrix)
    
    def to_dict(self) -> Dict:
        return {
            'design_matrix': self.design_matrix.to_dict(),
            'design_type': self.design_type,
            'factors': [vars(f) for f in self.factors],
            'responses': [vars(r) for r in self.responses],
            'metadata': self.metadata
        }


@dataclass
class AnalysisResult:
    """분석 결과 구조"""
    result_type: str
    data: Dict[str, Any]
    statistics: Dict[str, float]
    plots: Optional[Dict[str, Any]] = None
    ai_insights: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_summary(self) -> str:
        """결과 요약 반환"""
        summary = f"분석 유형: {self.result_type}\n"
        summary += f"분석 시간: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        if self.statistics:
            summary += "\n주요 통계:\n"
            for key, value in self.statistics.items():
                summary += f"  - {key}: {value:.4f}\n"
                
        return summary


# ============= 기본 클래스 =============

class DataValidator:
    """데이터 검증 클래스"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None) -> Tuple[bool, List[str]]:
        """데이터프레임 검증"""
        errors = []
        
        # 빈 데이터프레임 확인
        if df.empty:
            errors.append("데이터가 비어있습니다")
            return False, errors
            
        # 필수 컬럼 확인
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                errors.append(f"필수 컬럼이 없습니다: {missing_cols}")
                
        # 데이터 타입 확인
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            errors.append("숫자형 데이터가 없습니다")
            
        # 결측치 확인
        na_counts = df.isna().sum()
        high_na_cols = na_counts[na_counts > len(df) * 0.5]
        if not high_na_cols.empty:
            errors.append(f"결측치가 50% 이상인 컬럼: {high_na_cols.index.tolist()}")
            
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_design_constraints(factors: List[Factor], constraints: Dict) -> Tuple[bool, List[str]]:
        """설계 제약조건 검증"""
        errors = []
        
        # 기본 검증
        if not factors:
            errors.append("인자가 정의되지 않았습니다")
            
        # 제약조건 검증
        if 'min_runs' in constraints:
            min_theoretical = 2 ** len([f for f in factors if f.type == 'continuous'])
            if constraints['min_runs'] < min_theoretical:
                errors.append(f"최소 실험 횟수는 {min_theoretical} 이상이어야 합니다")
                
        return len(errors) == 0, errors


class TransformationEngine:
    """데이터 변환 엔진"""
    
    def __init__(self):
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        self._fitted_scalers = {}
        
    def transform(self, data: pd.DataFrame, method: str = 'standard', 
                 columns: List[str] = None) -> pd.DataFrame:
        """데이터 변환"""
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
            
        transformed_data = data.copy()
        
        if method in self.scalers:
            scaler = self.scalers[method]
            transformed_data[columns] = scaler.fit_transform(data[columns])
            self._fitted_scalers[method] = scaler
        elif method == 'log':
            # 로그 변환 (양수 확인)
            for col in columns:
                if (data[col] > 0).all():
                    transformed_data[col] = np.log(data[col])
                else:
                    logger.warning(f"{col}에 0 이하의 값이 있어 로그 변환을 건너뜁니다")
        elif method == 'sqrt':
            # 제곱근 변환
            for col in columns:
                if (data[col] >= 0).all():
                    transformed_data[col] = np.sqrt(data[col])
                else:
                    logger.warning(f"{col}에 음수가 있어 제곱근 변환을 건너뜁니다")
        elif method == 'box-cox':
            # Box-Cox 변환
            pt = PowerTransformer(method='box-cox', standardize=True)
            for col in columns:
                if (data[col] > 0).all():
                    transformed_data[col] = pt.fit_transform(data[[col]])
                else:
                    logger.warning(f"{col}에 0 이하의 값이 있어 Box-Cox 변환을 건너뜁니다")
                    
        return transformed_data
    
    def inverse_transform(self, data: pd.DataFrame, method: str = 'standard',
                         columns: List[str] = None) -> pd.DataFrame:
        """역변환"""
        if method not in self._fitted_scalers:
            raise ValueError(f"{method} 스케일러가 학습되지 않았습니다")
            
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
            
        transformed_data = data.copy()
        scaler = self._fitted_scalers[method]
        transformed_data[columns] = scaler.inverse_transform(data[columns])
        
        return transformed_data


# ============= 실험 설계 엔진 =============

class ExperimentDesignEngine:
    """실험 설계 생성 엔진"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
        
    def create_design(self, factors: List[Factor], design_type: str,
                     **kwargs) -> ExperimentDesign:
        """실험 설계 생성"""
        
        # 설계 유형별 처리
        if design_type == 'full_factorial':
            design_matrix = self._create_full_factorial(factors, **kwargs)
        elif design_type == 'fractional_factorial':
            design_matrix = self._create_fractional_factorial(factors, **kwargs)
        elif design_type == 'ccd':
            design_matrix = self._create_ccd(factors, **kwargs)
        elif design_type == 'box_behnken':
            design_matrix = self._create_box_behnken(factors, **kwargs)
        elif design_type == 'plackett_burman':
            design_matrix = self._create_plackett_burman(factors, **kwargs)
        elif design_type == 'lhs':
            design_matrix = self._create_lhs(factors, **kwargs)
        elif design_type == 'mixture':
            design_matrix = self._create_mixture_design(factors, **kwargs)
        elif design_type == 'taguchi':
            design_matrix = self._create_taguchi_design(factors, **kwargs)
        else:
            raise ValueError(f"지원하지 않는 설계 유형: {design_type}")
            
        # 실제 값으로 변환
        design_df = self._coded_to_actual(design_matrix, factors)
        
        # 실험 순서 랜덤화
        if kwargs.get('randomize', True):
            design_df = design_df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
            
        # 실험 설계 객체 생성
        return ExperimentDesign(
            design_matrix=design_df,
            design_type=design_type,
            factors=factors,
            responses=kwargs.get('responses', []),
            metadata={
                'created_at': datetime.now().isoformat(),
                'random_state': self.random_state,
                'design_params': kwargs
            }
        )
    
    def _create_full_factorial(self, factors: List[Factor], **kwargs) -> np.ndarray:
        """완전요인설계"""
        n_levels = kwargs.get('n_levels', 2)
        continuous_factors = [f for f in factors if f.type == 'continuous']
        n_factors = len(continuous_factors)
        
        if n_levels == 2:
            return doe.ff2n(n_factors)
        else:
            levels = [n_levels] * n_factors
            return doe.fullfact(levels)
    
    def _create_fractional_factorial(self, factors: List[Factor], **kwargs) -> np.ndarray:
        """부분요인설계"""
        continuous_factors = [f for f in factors if f.type == 'continuous']
        n_factors = len(continuous_factors)
        
        # 해상도 기반 generator 선택
        resolution = kwargs.get('resolution', 3)
        
        if n_factors <= 3:
            return self._create_full_factorial(factors, **kwargs)
        elif n_factors == 4:
            gen = "D = A*B*C" if resolution >= 4 else "D = A*B"
        elif n_factors == 5:
            gen = "E = A*B*C*D" if resolution >= 5 else "D = A*B; E = A*C"
        else:
            # 자동 generator 생성 로직
            gen = self._generate_fractional_generators(n_factors, resolution)
            
        return doe.fracfact(gen)
    
    def _create_ccd(self, factors: List[Factor], **kwargs) -> np.ndarray:
        """중심합성설계 (Central Composite Design)"""
        continuous_factors = [f for f in factors if f.type == 'continuous']
        n_factors = len(continuous_factors)
        
        center = kwargs.get('center', (4, 4))
        alpha = kwargs.get('alpha', 'orthogonal')
        face = kwargs.get('face', 'circumscribed')
        
        return doe.ccdesign(n_factors, center=center, alpha=alpha, face=face)
    
    def _create_box_behnken(self, factors: List[Factor], **kwargs) -> np.ndarray:
        """Box-Behnken 설계"""
        continuous_factors = [f for f in factors if f.type == 'continuous']
        n_factors = len(continuous_factors)
        
        if n_factors < 3:
            raise ValueError("Box-Behnken 설계는 최소 3개의 인자가 필요합니다")
            
        center = kwargs.get('center', 3)
        return doe.bbdesign(n_factors, center=center)
    
    def _create_plackett_burman(self, factors: List[Factor], **kwargs) -> np.ndarray:
        """Plackett-Burman 설계"""
        continuous_factors = [f for f in factors if f.type == 'continuous']
        n_factors = len(continuous_factors)
        
        return doe.pbdesign(n_factors)
    
    def _create_lhs(self, factors: List[Factor], **kwargs) -> np.ndarray:
        """Latin Hypercube Sampling"""
        continuous_factors = [f for f in factors if f.type == 'continuous']
        n_factors = len(continuous_factors)
        n_samples = kwargs.get('n_samples', 10 * n_factors)
        
        # 0-1 범위로 생성 후 -1~1로 변환
        lhs_design = doe.lhs(n_factors, samples=n_samples, criterion='maximin')
        return 2 * lhs_design - 1
    
    def _create_mixture_design(self, factors: List[Factor], **kwargs) -> np.ndarray:
        """혼합물 설계"""
        n_components = len(factors)
        design_type = kwargs.get('mixture_type', 'simplex_lattice')
        
        if design_type == 'simplex_lattice':
            degree = kwargs.get('degree', 2)
            # pyDOE2에는 mixture design이 없으므로 직접 구현
            return self._simplex_lattice(n_components, degree)
        elif design_type == 'simplex_centroid':
            return self._simplex_centroid(n_components)
        else:
            raise ValueError(f"지원하지 않는 혼합물 설계: {design_type}")
    
    def _create_taguchi_design(self, factors: List[Factor], **kwargs) -> np.ndarray:
        """다구치 설계"""
        n_factors = len(factors)
        n_levels = kwargs.get('n_levels', 3)
        
        # 적절한 직교배열 선택
        if n_factors <= 4 and n_levels == 2:
            return doe.fracfact('a b c d')
        elif n_factors <= 7 and n_levels == 2:
            return doe.fracfact('a b c d e f g')
        else:
            # L9, L18, L27 등의 배열 구현
            return self._taguchi_array(n_factors, n_levels)
    
    def _coded_to_actual(self, coded_matrix: np.ndarray, 
                        factors: List[Factor]) -> pd.DataFrame:
        """코드화된 값을 실제 값으로 변환"""
        df = pd.DataFrame(coded_matrix)
        continuous_factors = [f for f in factors if f.type == 'continuous']
        
        # 연속형 인자 변환
        for i, factor in enumerate(continuous_factors):
            if i < df.shape[1]:
                # -1 ~ 1 코드를 실제 값으로 변환
                df[factor.name] = factor.low + (df.iloc[:, i] + 1) / 2 * (factor.high - factor.low)
        
        # 범주형 인자 추가
        categorical_factors = [f for f in factors if f.type == 'categorical']
        for factor in categorical_factors:
            # 랜덤하게 레벨 할당 (실제로는 더 체계적인 방법 필요)
            n_runs = len(df)
            df[factor.name] = np.random.choice(factor.levels, size=n_runs)
            
        return df
    
    def _generate_fractional_generators(self, n_factors: int, resolution: int) -> str:
        """부분요인설계 generator 자동 생성"""
        # 간단한 구현 - 실제로는 더 복잡한 로직 필요
        base_factors = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'][:n_factors]
        
        if n_factors <= 4:
            return ' '.join(base_factors)
        else:
            # 기본 generator 패턴
            gens = []
            for i in range(4, n_factors):
                if resolution >= 4:
                    gens.append(f"{base_factors[i]} = {base_factors[0]}*{base_factors[1]}*{base_factors[2]}")
                else:
                    gens.append(f"{base_factors[i]} = {base_factors[0]}*{base_factors[1]}")
            return '; '.join(gens)
    
    def _simplex_lattice(self, n_components: int, degree: int) -> np.ndarray:
        """Simplex Lattice 설계 구현"""
        # 간단한 구현
        points = []
        
        # 정점
        for i in range(n_components):
            point = np.zeros(n_components)
            point[i] = 1.0
            points.append(point)
            
        # 중점 (degree >= 2)
        if degree >= 2:
            for i in range(n_components):
                for j in range(i+1, n_components):
                    point = np.zeros(n_components)
                    point[i] = point[j] = 0.5
                    points.append(point)
                    
        return np.array(points)
    
    def _simplex_centroid(self, n_components: int) -> np.ndarray:
        """Simplex Centroid 설계 구현"""
        points = []
        
        # 모든 부분집합의 중심점
        for i in range(1, 2**n_components):
            indices = [j for j in range(n_components) if i & (1 << j)]
            if indices:
                point = np.zeros(n_components)
                for idx in indices:
                    point[idx] = 1.0 / len(indices)
                points.append(point)
                
        return np.array(points)
    
    def _taguchi_array(self, n_factors: int, n_levels: int) -> np.ndarray:
        """다구치 직교배열 생성"""
        # 간단한 L9 (3^4) 구현
        if n_levels == 3 and n_factors <= 4:
            L9 = np.array([
                [0, 0, 0, 0],
                [0, 1, 1, 1],
                [0, 2, 2, 2],
                [1, 0, 1, 2],
                [1, 1, 2, 0],
                [1, 2, 0, 1],
                [2, 0, 2, 1],
                [2, 1, 0, 2],
                [2, 2, 1, 0]
            ])
            return L9[:, :n_factors]
        else:
            # 기본 full factorial로 대체
            return doe.fullfact([n_levels] * n_factors)


# ============= 통계 분석 엔진 =============

class StatisticalAnalyzer:
    """통계 분석 엔진"""
    
    def __init__(self, api_manager: Optional[APIManager] = None):
        self.api_manager = api_manager
        self._ai_detail_level = 'auto'
        
    def set_ai_detail_level(self, level: str):
        """AI 설명 상세도 설정"""
        self._ai_detail_level = level
        
    def descriptive_statistics(self, data: pd.DataFrame, 
                             columns: List[str] = None) -> AnalysisResult:
        """기술통계 분석"""
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
            
        stats_dict = {}
        for col in columns:
            stats_dict[col] = {
                'mean': data[col].mean(),
                'std': data[col].std(),
                'min': data[col].min(),
                'q1': data[col].quantile(0.25),
                'median': data[col].median(),
                'q3': data[col].quantile(0.75),
                'max': data[col].max(),
                'skewness': stats.skew(data[col].dropna()),
                'kurtosis': stats.kurtosis(data[col].dropna()),
                'cv': data[col].std() / data[col].mean() if data[col].mean() != 0 else np.nan
            }
            
        # AI 인사이트 생성
        ai_insights = self._generate_descriptive_insights(stats_dict, data)
        
        return AnalysisResult(
            result_type='descriptive_statistics',
            data={'statistics': stats_dict},
            statistics={
                'n_variables': len(columns),
                'n_observations': len(data),
                'missing_rate': data[columns].isna().sum().sum() / (len(data) * len(columns))
            },
            ai_insights=ai_insights
        )
    
    def correlation_analysis(self, data: pd.DataFrame,
                           method: str = 'pearson') -> AnalysisResult:
        """상관관계 분석"""
        numeric_data = data.select_dtypes(include=[np.number])
        
        # 상관계수 계산
        corr_matrix = numeric_data.corr(method=method)
        
        # p-value 계산
        p_values = pd.DataFrame(
            np.zeros_like(corr_matrix),
            columns=corr_matrix.columns,
            index=corr_matrix.index
        )
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                
                if method == 'pearson':
                    _, p_val = stats.pearsonr(numeric_data[col1].dropna(), 
                                             numeric_data[col2].dropna())
                elif method == 'spearman':
                    _, p_val = stats.spearmanr(numeric_data[col1].dropna(),
                                              numeric_data[col2].dropna())
                elif method == 'kendall':
                    _, p_val = stats.kendalltau(numeric_data[col1].dropna(),
                                               numeric_data[col2].dropna())
                    
                p_values.loc[col1, col2] = p_val
                p_values.loc[col2, col1] = p_val
                
        # 중요 상관관계 추출
        significant_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_val = corr_matrix.loc[col1, col2]
                p_val = p_values.loc[col1, col2]
                
                if abs(corr_val) > 0.3 and p_val < 0.05:
                    significant_corrs.append({
                        'var1': col1,
                        'var2': col2,
                        'correlation': corr_val,
                        'p_value': p_val,
                        'strength': self._interpret_correlation(corr_val)
                    })
                    
        # AI 인사이트 생성
        ai_insights = self._generate_correlation_insights(corr_matrix, significant_corrs)
        
        return AnalysisResult(
            result_type='correlation_analysis',
            data={
                'correlation_matrix': corr_matrix.to_dict(),
                'p_values': p_values.to_dict(),
                'significant_correlations': significant_corrs
            },
            statistics={
                'n_variables': len(corr_matrix.columns),
                'n_significant': len(significant_corrs),
                'avg_abs_correlation': corr_matrix.abs().mean().mean()
            },
            ai_insights=ai_insights
        )
    
    def anova_analysis(self, data: pd.DataFrame, 
                      factor_cols: List[str],
                      response_col: str,
                      interaction: bool = True) -> AnalysisResult:
        """분산분석 (ANOVA)"""
        # 데이터 준비
        clean_data = data[factor_cols + [response_col]].dropna()
        
        # 모델 공식 생성
        if len(factor_cols) == 1:
            # 일원분산분석
            formula = f"{response_col} ~ C({factor_cols[0]})"
        else:
            # 다원분산분석
            main_effects = ' + '.join([f"C({col})" for col in factor_cols])
            formula = f"{response_col} ~ {main_effects}"
            
            if interaction:
                # 상호작용 항 추가
                interactions = []
                for i in range(len(factor_cols)):
                    for j in range(i+1, len(factor_cols)):
                        interactions.append(f"C({factor_cols[i]}):C({factor_cols[j]})")
                formula += ' + ' + ' + '.join(interactions)
                
        # ANOVA 수행
        model = ols(formula, data=clean_data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        # 사후검정 (일원분산분석인 경우)
        posthoc_results = None
        if len(factor_cols) == 1:
            factor = factor_cols[0]
            posthoc = pairwise_tukeyhsd(
                clean_data[response_col],
                clean_data[factor],
                alpha=0.05
            )
            posthoc_results = {
                'summary': str(posthoc),
                'reject': posthoc.reject.tolist(),
                'meandiffs': posthoc.meandiffs.tolist(),
                'p_values': posthoc.pvalues.tolist()
            }
            
        # 효과 크기 계산
        eta_squared = {}
        for source in anova_table.index[:-1]:  # Residual 제외
            eta_squared[source] = (anova_table.loc[source, 'sum_sq'] / 
                                 anova_table['sum_sq'].sum())
            
        # AI 인사이트 생성
        ai_insights = self._generate_anova_insights(
            anova_table, eta_squared, posthoc_results
        )
        
        return AnalysisResult(
            result_type='anova',
            data={
                'anova_table': anova_table.to_dict(),
                'model_summary': str(model.summary()),
                'posthoc': posthoc_results,
                'eta_squared': eta_squared
            },
            statistics={
                'r_squared': model.rsquared,
                'r_squared_adj': model.rsquared_adj,
                'f_statistic': model.fvalue,
                'p_value': model.f_pvalue,
                'n_observations': len(clean_data)
            },
            ai_insights=ai_insights
        )
    
    def regression_analysis(self, data: pd.DataFrame,
                          predictors: List[str],
                          response: str,
                          model_type: str = 'linear',
                          polynomial_degree: int = 2) -> AnalysisResult:
        """회귀분석"""
        # 데이터 준비
        X = data[predictors].dropna()
        y = data.loc[X.index, response]
        
        # 다항식 변환 (필요시)
        if model_type == 'polynomial':
            poly = PolynomialFeatures(degree=polynomial_degree, include_bias=False)
            X_poly = poly.fit_transform(X)
            feature_names = poly.get_feature_names_out(predictors)
            X_model = pd.DataFrame(X_poly, columns=feature_names, index=X.index)
        else:
            X_model = X
            
        # 상수항 추가
        X_model = sm.add_constant(X_model)
        
        # 회귀모델 적합
        model = sm.OLS(y, X_model).fit()
        
        # 잔차 분석
        residuals = model.resid
        fitted_values = model.fittedvalues
        
        # 잔차 정규성 검정
        _, normality_p = stats.shapiro(residuals)
        
        # 등분산성 검정
        _, bp_p_value, _, _ = het_breuschpagan(residuals, X_model)
        
        # VIF 계산 (다중공선성)
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X_model.columns
        vif_data["VIF"] = [variance_inflation_factor(X_model.values, i) 
                          for i in range(X_model.shape[1])]
        
        # 예측 성능
        y_pred = model.predict(X_model)
        mse = np.mean((y - y_pred) ** 2)
        mae = np.mean(np.abs(y - y_pred))
        mape = np.mean(np.abs((y - y_pred) / y)) * 100
        
        # AI 인사이트 생성
        ai_insights = self._generate_regression_insights(
            model, vif_data, normality_p, bp_p_value
        )
        
        return AnalysisResult(
            result_type='regression',
            data={
                'coefficients': model.params.to_dict(),
                'p_values': model.pvalues.to_dict(),
                'confidence_intervals': model.conf_int().to_dict(),
                'vif': vif_data.to_dict(),
                'residuals': {
                    'values': residuals.tolist(),
                    'fitted': fitted_values.tolist()
                }
            },
            statistics={
                'r_squared': model.rsquared,
                'r_squared_adj': model.rsquared_adj,
                'f_statistic': model.fvalue,
                'f_p_value': model.f_pvalue,
                'aic': model.aic,
                'bic': model.bic,
                'mse': mse,
                'mae': mae,
                'mape': mape,
                'normality_p': normality_p,
                'homoscedasticity_p': bp_p_value
            },
            ai_insights=ai_insights
        )
    
    def rsm_analysis(self, data: pd.DataFrame,
                    factors: List[str],
                    response: str,
                    model_type: str = 'quadratic') -> AnalysisResult:
        """반응표면분석 (Response Surface Methodology)"""
        # 데이터 준비
        X = data[factors]
        y = data[response]
        
        # 모델 유형에 따른 변환
        if model_type == 'linear':
            X_model = X
        elif model_type == 'quadratic':
            # 2차 모델: 주효과 + 제곱항 + 교호작용
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = poly.fit_transform(X)
            feature_names = poly.get_feature_names_out(factors)
            X_model = pd.DataFrame(X_poly, columns=feature_names)
        elif model_type == 'cubic':
            # 3차 모델
            poly = PolynomialFeatures(degree=3, include_bias=False)
            X_poly = poly.fit_transform(X)
            feature_names = poly.get_feature_names_out(factors)
            X_model = pd.DataFrame(X_poly, columns=feature_names)
            
        # 회귀분석
        X_model = sm.add_constant(X_model)
        model = sm.OLS(y, X_model).fit()
        
        # 최적점 찾기
        if model_type == 'quadratic' and len(factors) <= 3:
            optimal_point = self._find_rsm_optimum(model, factors, X)
        else:
            optimal_point = None
            
        # 반응표면 예측을 위한 그리드 생성
        if len(factors) == 2:
            grid_data = self._create_2d_grid(X[factors[0]], X[factors[1]])
            predictions = self._predict_rsm_surface(model, grid_data, factors)
        else:
            grid_data = None
            predictions = None
            
        # AI 인사이트 생성
        ai_insights = self._generate_rsm_insights(
            model, factors, optimal_point
        )
        
        return AnalysisResult(
            result_type='rsm',
            data={
                'model': {
                    'coefficients': model.params.to_dict(),
                    'p_values': model.pvalues.to_dict(),
                    'summary': str(model.summary())
                },
                'optimal_point': optimal_point,
                'surface_data': {
                    'grid': grid_data,
                    'predictions': predictions
                } if grid_data is not None else None
            },
            statistics={
                'r_squared': model.rsquared,
                'r_squared_adj': model.rsquared_adj,
                'lack_of_fit_p': self._calculate_lack_of_fit(data, model, factors, response),
                'prediction_variance': np.var(model.predict())
            },
            ai_insights=ai_insights
        )
    
    # ===== 보조 메서드들 =====
    
    def _interpret_correlation(self, corr: float) -> str:
        """상관관계 강도 해석"""
        abs_corr = abs(corr)
        if abs_corr < 0.3:
            return "약한"
        elif abs_corr < 0.7:
            return "중간"
        else:
            return "강한"
            
    def _find_rsm_optimum(self, model, factors: List[str], 
                         X_data: pd.DataFrame) -> Dict:
        """RSM 최적점 찾기"""
        # 초기값 설정
        x0 = X_data[factors].mean().values
        
        # 경계 설정
        bounds = [(X_data[f].min(), X_data[f].max()) for f in factors]
        
        # 목적함수 (예측값 최대화)
        def objective(x):
            # 2차 항 생성
            poly = PolynomialFeatures(degree=2, include_bias=False)
            x_poly = poly.fit_transform(x.reshape(1, -1))
            x_model = np.concatenate([[1], x_poly[0]])  # 상수항 추가
            return -model.predict(x_model)[0]  # 최소화 문제로 변환
            
        # 최적화
        result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            optimal_x = result.x
            optimal_y = -result.fun
            
            return {
                'location': {factors[i]: optimal_x[i] for i in range(len(factors))},
                'predicted_value': optimal_y,
                'optimization_success': True
            }
        else:
            return {
                'optimization_success': False,
                'message': result.message
            }
            
    def _create_2d_grid(self, x1: pd.Series, x2: pd.Series, 
                       n_points: int = 50) -> Dict:
        """2D 그리드 생성"""
        x1_range = np.linspace(x1.min(), x1.max(), n_points)
        x2_range = np.linspace(x2.min(), x2.max(), n_points)
        X1, X2 = np.meshgrid(x1_range, x2_range)
        
        return {
            'x1': X1,
            'x2': X2,
            'x1_flat': X1.flatten(),
            'x2_flat': X2.flatten()
        }
        
    def _predict_rsm_surface(self, model, grid_data: Dict, 
                           factors: List[str]) -> np.ndarray:
        """RSM 표면 예측"""
        # 그리드 포인트를 모델 입력 형태로 변환
        X_grid = np.column_stack([grid_data['x1_flat'], grid_data['x2_flat']])
        
        # 다항식 변환
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X_grid)
        X_model = np.column_stack([np.ones(len(X_poly)), X_poly])
        
        # 예측
        predictions = model.predict(X_model)
        
        # 원래 그리드 형태로 변환
        return predictions.reshape(grid_data['x1'].shape)
        
    def _calculate_lack_of_fit(self, data: pd.DataFrame, model,
                              factors: List[str], response: str) -> float:
        """적합결여 검정"""
        # 반복 실험이 있는 경우에만 계산 가능
        # 여기서는 간단히 0.5 반환 (실제로는 더 복잡한 계산 필요)
        return 0.5
        
    # ===== AI 인사이트 생성 메서드들 =====
    
    def _generate_descriptive_insights(self, stats_dict: Dict, 
                                     data: pd.DataFrame) -> Dict:
        """기술통계 AI 인사이트 생성"""
        insights = {
            'summary': "데이터의 기본적인 통계적 특성을 분석했습니다.",
            'key_findings': [],
            'recommendations': []
        }
        
        # 주요 발견사항
        for col, stats in stats_dict.items():
            # 왜도 확인
            if abs(stats['skewness']) > 1:
                direction = "양" if stats['skewness'] > 0 else "음"
                insights['key_findings'].append(
                    f"{col}은 {direction}의 왜도를 보입니다 (skewness={stats['skewness']:.2f})"
                )
                
            # 변동계수 확인
            if stats['cv'] > 0.5:
                insights['key_findings'].append(
                    f"{col}의 변동성이 큽니다 (CV={stats['cv']:.2f})"
                )
                
        # 추천사항
        if any(abs(stats['skewness']) > 1 for stats in stats_dict.values()):
            insights['recommendations'].append(
                "왜도가 큰 변수들은 로그 또는 Box-Cox 변환을 고려하세요"
            )
            
        # 상세 설명 (AI 상세도 레벨에 따라)
        if self._ai_detail_level in ['detailed', 'always_detailed']:
            insights['detailed_explanation'] = self._get_detailed_descriptive_explanation(stats_dict)
            
        return insights
        
    def _generate_correlation_insights(self, corr_matrix: pd.DataFrame,
                                     significant_corrs: List[Dict]) -> Dict:
        """상관관계 AI 인사이트 생성"""
        insights = {
            'summary': f"{len(significant_corrs)}개의 유의미한 상관관계를 발견했습니다.",
            'key_findings': [],
            'recommendations': [],
            'warnings': []
        }
        
        # 주요 발견사항
        for corr in significant_corrs[:5]:  # 상위 5개
            insights['key_findings'].append(
                f"{corr['var1']}와 {corr['var2']} 간 {corr['strength']} "
                f"{'양' if corr['correlation'] > 0 else '음'}의 상관관계 "
                f"(r={corr['correlation']:.3f}, p={corr['p_value']:.3f})"
            )
            
        # 다중공선성 경고
        high_corr_pairs = [c for c in significant_corrs if abs(c['correlation']) > 0.8]
        if high_corr_pairs:
            insights['warnings'].append(
                f"{len(high_corr_pairs)}개 변수 쌍에서 높은 상관관계가 발견되어 "
                "다중공선성 문제가 있을 수 있습니다"
            )
            
        # 상세 설명
        if self._ai_detail_level in ['detailed', 'always_detailed']:
            insights['detailed_explanation'] = {
                'interpretation': "상관계수의 의미와 해석 방법...",
                'assumptions': "상관분석의 가정사항...",
                'limitations': "상관관계는 인과관계를 의미하지 않습니다..."
            }
            
        return insights
        
    def _generate_anova_insights(self, anova_table: pd.DataFrame,
                                eta_squared: Dict,
                                posthoc_results: Dict) -> Dict:
        """ANOVA AI 인사이트 생성"""
        insights = {
            'summary': "분산분석을 통해 그룹 간 차이를 검정했습니다.",
            'key_findings': [],
            'recommendations': [],
            'effect_sizes': []
        }
        
        # 유의미한 효과 찾기
        for source in anova_table.index[:-1]:
            p_value = anova_table.loc[source, 'PR(>F)']
            if p_value < 0.05:
                effect_size = eta_squared.get(source, 0)
                insights['key_findings'].append(
                    f"{source}의 효과가 통계적으로 유의합니다 "
                    f"(p={p_value:.4f}, η²={effect_size:.3f})"
                )
                
                # 효과 크기 해석
                if effect_size > 0.14:
                    size_interpretation = "큰"
                elif effect_size > 0.06:
                    size_interpretation = "중간"
                else:
                    size_interpretation = "작은"
                    
                insights['effect_sizes'].append(
                    f"{source}는 {size_interpretation} 효과 크기를 보입니다"
                )
                
        # 사후검정 결과
        if posthoc_results and any(posthoc_results['reject']):
            insights['key_findings'].append(
                "Tukey HSD 사후검정 결과 일부 그룹 간 유의한 차이가 있습니다"
            )
            
        # 상세 설명
        if self._ai_detail_level in ['detailed', 'always_detailed']:
            insights['detailed_explanation'] = {
                'method': "ANOVA는 세 개 이상의 그룹 평균을 비교하는 통계 기법입니다...",
                'assumptions': "정규성, 등분산성, 독립성 가정이 필요합니다...",
                'interpretation': "p-value < 0.05는 적어도 하나의 그룹이 다름을 의미합니다..."
            }
            
        return insights
        
    def _generate_regression_insights(self, model, vif_data: pd.DataFrame,
                                    normality_p: float, bp_p_value: float) -> Dict:
        """회귀분석 AI 인사이트 생성"""
        insights = {
            'summary': f"R² = {model.rsquared:.3f}로 모델이 데이터의 "
                      f"{model.rsquared*100:.1f}%를 설명합니다.",
            'key_findings': [],
            'recommendations': [],
            'diagnostics': []
        }
        
        # 유의미한 예측변수
        significant_vars = model.pvalues[model.pvalues < 0.05].index.tolist()
        if significant_vars:
            insights['key_findings'].append(
                f"유의미한 예측변수: {', '.join(significant_vars)}"
            )
            
        # 모델 진단
        if normality_p < 0.05:
            insights['diagnostics'].append(
                "잔차의 정규성 가정이 위배되었습니다 (변환 고려)"
            )
            
        if bp_p_value < 0.05:
            insights['diagnostics'].append(
                "등분산성 가정이 위배되었습니다 (가중회귀 고려)"
            )
            
        # VIF 확인
        high_vif = vif_data[vif_data['VIF'] > 10]
        if not high_vif.empty:
            insights['diagnostics'].append(
                f"다중공선성 문제: {high_vif['Variable'].tolist()}"
            )
            
        # 상세 설명
        if self._ai_detail_level in ['detailed', 'always_detailed']:
            insights['detailed_explanation'] = {
                'coefficients': "회귀계수의 해석 방법...",
                'assumptions': "선형성, 독립성, 등분산성, 정규성...",
                'improvements': "모델 개선 방법..."
            }
            
        return insights
        
    def _generate_rsm_insights(self, model, factors: List[str],
                              optimal_point: Dict) -> Dict:
        """RSM AI 인사이트 생성"""
        insights = {
            'summary': "반응표면분석을 통해 최적 조건을 탐색했습니다.",
            'key_findings': [],
            'recommendations': [],
            'optimization': []
        }
        
        # 모델 적합도
        insights['key_findings'].append(
            f"모델 적합도: R² = {model.rsquared:.3f}"
        )
        
        # 최적점
        if optimal_point and optimal_point.get('optimization_success'):
            opt_conditions = optimal_point['location']
            opt_value = optimal_point['predicted_value']
            
            insights['optimization'].append(
                f"최적 조건: {', '.join([f'{k}={v:.3f}' for k, v in opt_conditions.items()])}"
            )
            insights['optimization'].append(
                f"예측 최적값: {opt_value:.3f}"
            )
            
        # 유의미한 항
        significant_terms = model.pvalues[model.pvalues < 0.05].index.tolist()
        
        # 2차 항 확인
        quadratic_terms = [t for t in significant_terms if '^2' in t]
        if quadratic_terms:
            insights['key_findings'].append(
                f"곡률 효과가 유의합니다: {', '.join(quadratic_terms)}"
            )
            
        # 교호작용 확인  
        interaction_terms = [t for t in significant_terms if ' ' in t and '^' not in t]
        if interaction_terms:
            insights['key_findings'].append(
                f"교호작용이 유의합니다: {', '.join(interaction_terms)}"
            )
            
        # 상세 설명
        if self._ai_detail_level in ['detailed', 'always_detailed']:
            insights['detailed_explanation'] = {
                'method': "RSM은 반응변수와 여러 설명변수 간의 관계를 모델링합니다...",
                'optimization': "경사상승법을 사용하여 최적점을 찾습니다...",
                'validation': "추가 실험을 통해 최적점을 검증하세요..."
            }
            
        return insights
        
    def _get_detailed_descriptive_explanation(self, stats_dict: Dict) -> Dict:
        """기술통계 상세 설명"""
        return {
            'measures': {
                'mean': "산술평균은 모든 값의 합을 개수로 나눈 값입니다",
                'std': "표준편차는 데이터의 산포도를 나타냅니다",
                'skewness': "왜도는 분포의 비대칭성을 측정합니다 (0=대칭)",
                'kurtosis': "첨도는 분포의 뾰족함을 측정합니다 (3=정규분포)",
                'cv': "변동계수는 상대적 변동성을 나타냅니다 (std/mean)"
            },
            'interpretation_guide': {
                'skewness': {
                    'high_positive': "오른쪽으로 긴 꼬리 (이상치가 큰 값)",
                    'high_negative': "왼쪽으로 긴 꼬리 (이상치가 작은 값)",
                    'near_zero': "대칭적 분포"
                },
                'cv': {
                    'low': "CV < 0.3: 낮은 변동성",
                    'medium': "0.3 ≤ CV < 0.6: 중간 변동성", 
                    'high': "CV ≥ 0.6: 높은 변동성"
                }
            }
        }


# ============= 최적화 엔진 =============

class OptimizationEngine:
    """최적화 엔진"""
    
    def __init__(self, api_manager: Optional[APIManager] = None):
        self.api_manager = api_manager
        self._ai_detail_level = 'auto'
        
    def single_objective_optimization(self, 
                                    objective_func: Callable,
                                    bounds: List[Tuple[float, float]],
                                    method: str = 'differential_evolution',
                                    constraints: List[Dict] = None) -> AnalysisResult:
        """단일 목적 최적화"""
        
        # 최적화 방법 선택
        if method == 'differential_evolution':
            result = differential_evolution(
                objective_func, 
                bounds,
                seed=42,
                maxiter=1000,
                popsize=15
            )
        elif method == 'scipy_minimize':
            x0 = [(b[0] + b[1]) / 2 for b in bounds]
            result = minimize(
                objective_func,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000}
            )
        elif method == 'bayesian' and SKOPT_AVAILABLE:
            result = gp_minimize(
                objective_func,
                bounds,
                n_calls=50,
                n_initial_points=10,
                random_state=42
            )
        else:
            raise ValueError(f"지원하지 않는 최적화 방법: {method}")
            
        # 결과 정리
        optimal_x = result.x
        optimal_value = result.fun if hasattr(result, 'fun') else result.fun()
        
        # 민감도 분석
        sensitivity = self._sensitivity_analysis(objective_func, optimal_x, bounds)
        
        # AI 인사이트 생성
        ai_insights = self._generate_optimization_insights(
            result, optimal_x, optimal_value, sensitivity
        )
        
        return AnalysisResult(
            result_type='single_objective_optimization',
            data={
                'optimal_solution': optimal_x.tolist(),
                'optimal_value': optimal_value,
                'convergence': result.success if hasattr(result, 'success') else True,
                'n_iterations': result.nit if hasattr(result, 'nit') else None,
                'sensitivity': sensitivity
            },
            statistics={
                'improvement': None,  # 초기값 대비 개선율
                'function_calls': result.nfev if hasattr(result, 'nfev') else None
            },
            ai_insights=ai_insights
        )
        
    def multi_objective_optimization(self,
                                   objectives: List[Callable],
                                   bounds: List[Tuple[float, float]],
                                   n_objectives: int,
                                   method: str = 'nsga2') -> AnalysisResult:
        """다중 목적 최적화"""
        
        if not PYMOO_AVAILABLE:
            raise ImportError("pymoo가 설치되지 않았습니다. pip install pymoo")
            
        # 문제 정의
        class MultiObjectiveProblem(Problem):
            def __init__(self, objectives, bounds):
                super().__init__(
                    n_var=len(bounds),
                    n_obj=n_objectives,
                    n_constr=0,
                    xl=np.array([b[0] for b in bounds]),
                    xu=np.array([b[1] for b in bounds])
                )
                self.objectives = objectives
                
            def _evaluate(self, x, out, *args, **kwargs):
                # 각 목적함수 평가
                f_values = np.zeros((x.shape[0], self.n_obj))
                for i in range(x.shape[0]):
                    for j, obj_func in enumerate(self.objectives):
                        f_values[i, j] = obj_func(x[i])
                out["F"] = f_values
                
        # 최적화 실행
        problem = MultiObjectiveProblem(objectives, bounds)
        algorithm = NSGA2(pop_size=100)
        
        res = pymoo_minimize(
            problem,
            algorithm,
            ('n_gen', 200),
            seed=42,
            verbose=False
        )
        
        # Pareto front 추출
        pareto_front = res.F
        pareto_set = res.X
        
        # 대표 솔루션 선택 (무릎점)
        knee_point_idx = self._find_knee_point(pareto_front)
        
        # AI 인사이트 생성
        ai_insights = self._generate_multi_objective_insights(
            pareto_front, pareto_set, knee_point_idx
        )
        
        return AnalysisResult(
            result_type='multi_objective_optimization',
            data={
                'pareto_front': pareto_front.tolist(),
                'pareto_set': pareto_set.tolist(),
                'knee_point': {
                    'solution': pareto_set[knee_point_idx].tolist(),
                    'objectives': pareto_front[knee_point_idx].tolist()
                },
                'n_solutions': len(pareto_front)
            },
            statistics={
                'hypervolume': self._calculate_hypervolume(pareto_front),
                'spread': self._calculate_spread(pareto_front)
            },
            ai_insights=ai_insights
        )
        
    def _sensitivity_analysis(self, objective_func: Callable,
                            optimal_x: np.ndarray,
                            bounds: List[Tuple[float, float]],
                            delta: float = 0.01) -> Dict:
        """민감도 분석"""
        sensitivity = {}
        base_value = objective_func(optimal_x)
        
        for i in range(len(optimal_x)):
            # 변수별 민감도
            x_plus = optimal_x.copy()
            x_minus = optimal_x.copy()
            
            # 범위의 1% 변화
            change = (bounds[i][1] - bounds[i][0]) * delta
            x_plus[i] = min(optimal_x[i] + change, bounds[i][1])
            x_minus[i] = max(optimal_x[i] - change, bounds[i][0])
            
            f_plus = objective_func(x_plus)
            f_minus = objective_func(x_minus)
            
            # 민감도 계산
            sensitivity[f'x{i}'] = {
                'derivative': (f_plus - f_minus) / (2 * change),
                'relative_change': abs((f_plus - f_minus) / base_value) * 100
            }
            
        return sensitivity
        
    def _find_knee_point(self, pareto_front: np.ndarray) -> int:
        """Pareto front에서 무릎점 찾기"""
        # 정규화
        pf_norm = (pareto_front - pareto_front.min(axis=0)) / \
                  (pareto_front.max(axis=0) - pareto_front.min(axis=0))
                  
        # 이상점 (0, 0, ..., 0)으로부터의 거리
        distances = np.sqrt(np.sum(pf_norm ** 2, axis=1))
        
        # 최소 거리 점이 무릎점
        return np.argmin(distances)
        
    def _calculate_hypervolume(self, pareto_front: np.ndarray) -> float:
        """Hypervolume 계산 (간단한 구현)"""
        # 참조점 설정 (최악의 경우)
        ref_point = pareto_front.max(axis=0) * 1.1
        
        # 2D인 경우만 구현
        if pareto_front.shape[1] == 2:
            # 정렬
            sorted_pf = pareto_front[pareto_front[:, 0].argsort()]
            
            hv = 0
            prev_x = 0
            for point in sorted_pf:
                hv += (point[0] - prev_x) * (ref_point[1] - point[1])
                prev_x = point[0]
                
            return hv
        else:
            # 고차원은 근사값 반환
            return np.prod(ref_point - pareto_front.min(axis=0))
            
    def _calculate_spread(self, pareto_front: np.ndarray) -> float:
        """Spread 지표 계산"""
        # 각 차원별 범위
        ranges = pareto_front.max(axis=0) - pareto_front.min(axis=0)
        
        # 정규화된 spread
        return np.mean(ranges / (pareto_front.max(axis=0) + 1e-10))
        
    def _generate_optimization_insights(self, result, optimal_x: np.ndarray,
                                      optimal_value: float,
                                      sensitivity: Dict) -> Dict:
        """최적화 AI 인사이트 생성"""
        insights = {
            'summary': f"최적해를 찾았습니다. 목적함수 값: {optimal_value:.4f}",
            'key_findings': [],
            'recommendations': [],
            'sensitivity_analysis': []
        }
        
        # 수렴 상태
        if hasattr(result, 'success') and result.success:
            insights['key_findings'].append("최적화가 성공적으로 수렴했습니다")
        else:
            insights['key_findings'].append("최적화가 완전히 수렴하지 않았을 수 있습니다")
            
        # 민감도 분석 결과
        most_sensitive = max(sensitivity.items(), 
                           key=lambda x: x[1]['relative_change'])
        insights['sensitivity_analysis'].append(
            f"가장 민감한 변수: {most_sensitive[0]} "
            f"(1% 변화 시 {most_sensitive[1]['relative_change']:.2f}% 변화)"
        )
        
        # 상세 설명
        if self._ai_detail_level in ['detailed', 'always_detailed']:
            insights['detailed_explanation'] = {
                'method': f"{result.method if hasattr(result, 'method') else 'differential_evolution'} 알고리즘 사용",
                'convergence': "수렴 기준과 반복 횟수...",
                'validation': "최적해 검증 방법..."
            }
            
        return insights
        
    def _generate_multi_objective_insights(self, pareto_front: np.ndarray,
                                         pareto_set: np.ndarray,
                                         knee_point_idx: int) -> Dict:
        """다중 목적 최적화 AI 인사이트 생성"""
        insights = {
            'summary': f"{len(pareto_front)}개의 Pareto 최적해를 찾았습니다",
            'key_findings': [],
            'recommendations': [],
            'trade_offs': []
        }
        
        # 목적함수 간 상충관계
        if pareto_front.shape[1] == 2:
            correlation = np.corrcoef(pareto_front[:, 0], pareto_front[:, 1])[0, 1]
            if correlation < -0.5:
                insights['trade_offs'].append(
                    "두 목적함수 간 강한 상충관계가 있습니다"
                )
                
        # 무릎점 추천
        knee_solution = pareto_set[knee_point_idx]
        knee_objectives = pareto_front[knee_point_idx]
        insights['recommendations'].append(
            f"균형잡힌 해(무릎점): {knee_objectives.tolist()}"
        )
        
        # Pareto front 특성
        spread = self._calculate_spread(pareto_front)
        insights['key_findings'].append(
            f"해의 다양성(spread): {spread:.3f}"
        )
        
        # 상세 설명
        if self._ai_detail_level in ['detailed', 'always_detailed']:
            insights['detailed_explanation'] = {
                'pareto_optimality': "Pareto 최적성의 개념...",
                'selection_criteria': "해 선택 기준...",
                'visualization': "Pareto front 시각화 방법..."
            }
            
        return insights


# ============= 메인 데이터 처리 클래스 =============

class DataProcessor:
    """통합 데이터 처리 엔진"""
    
    def __init__(self, api_manager: Optional[APIManager] = None):
        self.api_manager = api_manager
        self.validator = DataValidator()
        self.transformer = TransformationEngine()
        self.design_engine = ExperimentDesignEngine()
        self.stat_analyzer = StatisticalAnalyzer(api_manager)
        self.optimizer = OptimizationEngine(api_manager)
        
        # AI 상세도 레벨
        self._ai_detail_level = 'auto'
        
        logger.info("DataProcessor 초기화 완료")
        
    def set_ai_detail_level(self, level: str):
        """AI 설명 상세도 설정"""
        valid_levels = ['auto', 'simple', 'detailed', 'always_detailed']
        if level in valid_levels:
            self._ai_detail_level = level
            self.stat_analyzer.set_ai_detail_level(level)
            self.optimizer._ai_detail_level = level
            logger.info(f"AI 상세도 레벨 설정: {level}")
        else:
            logger.warning(f"잘못된 AI 상세도 레벨: {level}")
            
    # ===== 데이터 전처리 =====
    
    def preprocess_data(self, data: pd.DataFrame,
                       handle_missing: str = 'drop',
                       handle_outliers: str = 'keep',
                       transform_method: str = None) -> Tuple[pd.DataFrame, Dict]:
        """데이터 전처리 통합 파이프라인"""
        processing_log = {
            'original_shape': data.shape,
            'steps': [],
            'warnings': []
        }
        
        # 1. 데이터 검증
        is_valid, errors = self.validator.validate_dataframe(data)
        if not is_valid:
            processing_log['warnings'].extend(errors)
            
        # 2. 결측치 처리
        processed_data = self._handle_missing_values(data, handle_missing)
        processing_log['steps'].append({
            'step': 'missing_values',
            'method': handle_missing,
            'before': data.isna().sum().sum(),
            'after': processed_data.isna().sum().sum()
        })
        
        # 3. 이상치 처리
        processed_data, outlier_info = self._handle_outliers(processed_data, handle_outliers)
        processing_log['steps'].append({
            'step': 'outliers',
            'method': handle_outliers,
            'outliers_found': outlier_info
        })
        
        # 4. 데이터 변환
        if transform_method:
            processed_data = self.transformer.transform(processed_data, transform_method)
            processing_log['steps'].append({
                'step': 'transformation',
                'method': transform_method
            })
            
        processing_log['final_shape'] = processed_data.shape
        
        return processed_data, processing_log
        
    def _handle_missing_values(self, data: pd.DataFrame, 
                             method: str = 'drop') -> pd.DataFrame:
        """결측치 처리"""
        if method == 'drop':
            return data.dropna()
        elif method == 'mean':
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
            return data
        elif method == 'median':
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
            return data
        elif method == 'forward_fill':
            return data.fillna(method='ffill')
        elif method == 'interpolate':
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = data[numeric_cols].interpolate()
            return data
        else:
            return data
            
    def _handle_outliers(self, data: pd.DataFrame, 
                        method: str = 'keep') -> Tuple[pd.DataFrame, Dict]:
        """이상치 처리"""
        outlier_info = {}
        
        if method == 'keep':
            return data, outlier_info
            
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
                outlier_info[col] = len(outliers)
                
                if method == 'remove':
                    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
                elif method == 'cap':
                    data[col] = data[col].clip(lower_bound, upper_bound)
                    
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(data[col].dropna()))
                outliers = data[col][z_scores > 3]
                outlier_info[col] = len(outliers)
                
                if method == 'remove':
                    data = data[z_scores <= 3]
                    
        return data, outlier_info
        
    # ===== 실험 설계 =====
    
    def create_experiment_design(self, factors: List[Dict],
                               responses: List[Dict],
                               design_type: str,
                               **kwargs) -> ExperimentDesign:
        """실험 설계 생성"""
        # Factor 객체 생성
        factor_objects = [Factor(**f) for f in factors]
        response_objects = [Response(**r) for r in responses]
        
        # 설계 제약조건 검증
        constraints = kwargs.get('constraints', {})
        is_valid, errors = self.validator.validate_design_constraints(
            factor_objects, constraints
        )
        
        if not is_valid:
            raise ValueError(f"설계 제약조건 오류: {errors}")
            
        # 설계 생성
        kwargs['responses'] = response_objects
        design = self.design_engine.create_design(
            factor_objects, design_type, **kwargs
        )
        
        logger.info(f"{design_type} 설계 생성 완료: {design.get_run_count()}개 실험")
        
        return design
        
    # ===== 통계 분석 =====
    
    def analyze_data(self, data: pd.DataFrame,
                    analysis_type: str,
                    **kwargs) -> AnalysisResult:
        """데이터 분석 실행"""
        
        analysis_methods = {
            'descriptive': self.stat_analyzer.descriptive_statistics,
            'correlation': self.stat_analyzer.correlation_analysis,
            'anova': self.stat_analyzer.anova_analysis,
            'regression': self.stat_analyzer.regression_analysis,
            'rsm': self.stat_analyzer.rsm_analysis
        }
        
        if analysis_type not in analysis_methods:
            raise ValueError(f"지원하지 않는 분석 유형: {analysis_type}")
            
        # 분석 실행
        method = analysis_methods[analysis_type]
        result = method(data, **kwargs)
        
        logger.info(f"{analysis_type} 분석 완료")
        
        return result
        
    # ===== 최적화 =====
    
    def optimize(self, optimization_type: str,
                **kwargs) -> AnalysisResult:
        """최적화 실행"""
        
        if optimization_type == 'single':
            result = self.optimizer.single_objective_optimization(**kwargs)
        elif optimization_type == 'multi':
            result = self.optimizer.multi_objective_optimization(**kwargs)
        else:
            raise ValueError(f"지원하지 않는 최적화 유형: {optimization_type}")
            
        logger.info(f"{optimization_type} 최적화 완료")
        
        return result
        
    # ===== 통합 파이프라인 =====
    
    def run_complete_pipeline(self, raw_data: pd.DataFrame,
                            factors: List[Dict],
                            responses: List[Dict],
                            design_type: str = 'ccd',
                            optimization_goal: str = 'maximize') -> Dict:
        """전체 분석 파이프라인 실행"""
        
        results = {
            'preprocessing': None,
            'design': None,
            'analysis': {},
            'optimization': None,
            'summary': {}
        }
        
        try:
            # 1. 데이터 전처리
            processed_data, processing_log = self.preprocess_data(
                raw_data,
                handle_missing='interpolate',
                handle_outliers='cap',
                transform_method='standard'
            )
            results['preprocessing'] = processing_log
            
            # 2. 실험 설계 (필요시)
            if not raw_data.empty:
                # 기존 데이터가 있으면 분석만
                pass
            else:
                # 새로운 실험 설계 생성
                design = self.create_experiment_design(
                    factors, responses, design_type
                )
                results['design'] = design.to_dict()
                
            # 3. 통계 분석
            # 기술통계
            results['analysis']['descriptive'] = self.analyze_data(
                processed_data, 'descriptive'
            )
            
            # 상관분석
            results['analysis']['correlation'] = self.analyze_data(
                processed_data, 'correlation'
            )
            
            # ANOVA (범주형 변수가 있는 경우)
            categorical_cols = processed_data.select_dtypes(
                include=['object', 'category']
            ).columns.tolist()
            
            if categorical_cols and len(responses) > 0:
                results['analysis']['anova'] = self.analyze_data(
                    processed_data,
                    'anova',
                    factor_cols=categorical_cols[:1],
                    response_col=responses[0]['name']
                )
                
            # 회귀분석
            numeric_factors = [f['name'] for f in factors if f['type'] == 'continuous']
            if numeric_factors and len(responses) > 0:
                results['analysis']['regression'] = self.analyze_data(
                    processed_data,
                    'regression',
                    predictors=numeric_factors,
                    response=responses[0]['name']
                )
                
            # 4. 최적화 (회귀모델 기반)
            if 'regression' in results['analysis']:
                # 회귀모델을 목적함수로 사용
                reg_model = results['analysis']['regression']
                
                def objective(x):
                    # 회귀모델 예측값 (최대화 문제로 변환)
                    pred_data = pd.DataFrame([x], columns=numeric_factors)
                    pred_data = sm.add_constant(pred_data)
                    prediction = reg_model.data.get('model', {}).get('predict', lambda x: 0)(pred_data)
                    return -prediction[0] if optimization_goal == 'maximize' else prediction[0]
                    
                bounds = [(f['low'], f['high']) for f in factors if f['type'] == 'continuous']
                
                results['optimization'] = self.optimize(
                    'single',
                    objective_func=objective,
                    bounds=bounds
                )
                
            # 5. 요약
            results['summary'] = self._generate_pipeline_summary(results)
            
        except Exception as e:
            logger.error(f"파이프라인 실행 중 오류: {str(e)}")
            results['error'] = str(e)
            
        return results
        
    def _generate_pipeline_summary(self, results: Dict) -> Dict:
        """파이프라인 결과 요약"""
        summary = {
            'preprocessing': {
                'original_samples': results['preprocessing']['original_shape'][0],
                'final_samples': results['preprocessing']['final_shape'][0],
                'steps_applied': len(results['preprocessing']['steps'])
            },
            'analyses_performed': list(results['analysis'].keys()),
            'optimization_success': False
        }
        
        # 주요 통계 결과
        if 'descriptive' in results['analysis']:
            desc_stats = results['analysis']['descriptive'].statistics
            summary['data_quality'] = {
                'missing_rate': desc_stats.get('missing_rate', 0),
                'n_variables': desc_stats.get('n_variables', 0)
            }
            
        # 최적화 결과
        if results.get('optimization'):
            opt_data = results['optimization'].data
            summary['optimization_success'] = opt_data.get('convergence', False)
            summary['optimal_value'] = opt_data.get('optimal_value')
            
        return summary
        
    # ===== 유틸리티 메서드 =====
    
    def export_results(self, results: Dict, format: str = 'excel',
                      filename: str = 'analysis_results') -> str:
        """결과 내보내기"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        full_filename = f"{filename}_{timestamp}"
        
        if format == 'excel':
            # Excel 파일로 내보내기
            with pd.ExcelWriter(f"{full_filename}.xlsx", engine='openpyxl') as writer:
                # 각 분석 결과를 시트로 저장
                for analysis_name, result in results.get('analysis', {}).items():
                    if hasattr(result, 'data') and isinstance(result.data, dict):
                        for key, value in result.data.items():
                            if isinstance(value, pd.DataFrame):
                                sheet_name = f"{analysis_name}_{key}"[:31]
                                value.to_excel(writer, sheet_name=sheet_name)
                            elif isinstance(value, dict) and key != 'model':
                                df = pd.DataFrame([value])
                                sheet_name = f"{analysis_name}_{key}"[:31]
                                df.to_excel(writer, sheet_name=sheet_name)
                                
                # 요약 정보
                if 'summary' in results:
                    summary_df = pd.DataFrame([results['summary']])
                    summary_df.to_excel(writer, sheet_name='Summary')
                    
            return f"{full_filename}.xlsx"
            
        elif format == 'json':
            # JSON 파일로 내보내기
            import json
            
            # DataFrame을 dict로 변환
            json_results = self._convert_results_to_json(results)
            
            with open(f"{full_filename}.json", 'w', encoding='utf-8') as f:
                json.dump(json_results, f, ensure_ascii=False, indent=2)
                
            return f"{full_filename}.json"
            
        else:
            raise ValueError(f"지원하지 않는 내보내기 형식: {format}")
            
    def _convert_results_to_json(self, results: Dict) -> Dict:
        """결과를 JSON 직렬화 가능한 형태로 변환"""
        json_results = {}
        
        for key, value in results.items():
            if isinstance(value, pd.DataFrame):
                json_results[key] = value.to_dict('records')
            elif isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, AnalysisResult):
                json_results[key] = {
                    'result_type': value.result_type,
                    'data': self._convert_results_to_json(value.data),
                    'statistics': value.statistics,
                    'ai_insights': value.ai_insights,
                    'timestamp': value.timestamp.isoformat()
                }
            elif isinstance(value, dict):
                json_results[key] = self._convert_results_to_json(value)
            elif isinstance(value, (list, tuple)):
                json_results[key] = [self._convert_results_to_json(item) 
                                   if isinstance(item, (dict, pd.DataFrame)) 
                                   else item for item in value]
            else:
                json_results[key] = value
                
        return json_results


# ============= 싱글톤 인스턴스 =============

_data_processor = None

def get_data_processor(api_manager: Optional[APIManager] = None) -> DataProcessor:
    """DataProcessor 싱글톤 인스턴스 반환"""
    global _data_processor
    if _data_processor is None:
        _data_processor = DataProcessor(api_manager)
    return _data_processor
