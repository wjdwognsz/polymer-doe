"""
데이터 처리 엔진 - Universal DOE Platform의 핵심 분석 시스템
실험 설계, 데이터 전처리, 통계 분석, 최적화를 담당하는 통합 처리 엔진
Version: Ultimate (1번 + 2번 코드 통합)
"""

# 표준 라이브러리
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
import warnings
import logging
from abc import ABC, abstractmethod
from enum import Enum
import copy
from functools import lru_cache
import time
import re
import hashlib

# 과학 계산
from scipy import stats, optimize, signal
from scipy.spatial.distance import cdist
from scipy.stats import (
    f_oneway, ttest_ind, chi2_contingency, 
    normaltest, shapiro, kstest, boxcox
)
from scipy.optimize import minimize, differential_evolution, shgo, dual_annealing
from scipy.interpolate import interp1d, griddata
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 기계학습
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    PolynomialFeatures, PowerTransformer
)
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 실험 설계
try:
    import pyDOE3 as doe
    from pyDOE3 import *
except ImportError:
    try:
        import pyDOE2 as doe
        from pyDOE2 import *
    except ImportError:
        import pyDOE as doe
        from pyDOE import *

# 최적화
try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize as pymoo_minimize
    from pymoo.termination import get_termination
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    logging.warning("pymoo not available, multi-objective optimization limited")

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    logging.warning("scikit-optimize not available, Bayesian optimization limited")

# 시각화 지원
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# 내부 모듈
try:
    from utils.api_manager import APIManager
    from utils.common_ui import show_error, show_warning, show_info
    from config.app_config import APP_CONFIG, EXPERIMENT_DEFAULTS
except ImportError:
    APIManager = None
    APP_CONFIG = {}
    EXPERIMENT_DEFAULTS = {}

# 로깅 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# 전역 상수
RANDOM_STATE = 42
CONFIDENCE_LEVEL = 0.95
MAX_ITERATIONS = 1000
CONVERGENCE_TOL = 1e-6


# === Enums and Constants ===

class TransformMethod(Enum):
    """변환 방법"""
    NONE = "none"
    STANDARDIZE = "standardize"
    NORMALIZE = "normalize"
    LOG = "log"
    SQRT = "sqrt"
    RECIPROCAL = "reciprocal"
    BOX_COX = "box_cox"
    YEO_JOHNSON = "yeo_johnson"
    QUANTILE = "quantile"
    ROBUST_SCALE = "robust_scale"


class DesignType(Enum):
    """실험 설계 유형"""
    FACTORIAL = "factorial"
    FRACTIONAL_FACTORIAL = "fractional_factorial"
    CENTRAL_COMPOSITE = "central_composite"
    BOX_BEHNKEN = "box_behnken"
    LATIN_HYPERCUBE = "latin_hypercube"
    PLACKETT_BURMAN = "plackett_burman"
    MIXTURE = "mixture"
    TAGUCHI = "taguchi"
    D_OPTIMAL = "d_optimal"
    CUSTOM = "custom"


class OptimizationType(Enum):
    """최적화 유형"""
    SINGLE = "single_objective"
    MULTI = "multi_objective"
    BAYESIAN = "bayesian"
    ROBUST = "robust"


# === Data Classes ===

@dataclass
class Factor:
    """실험 인자"""
    name: str
    type: str  # continuous, categorical, discrete
    low: Optional[float] = None
    high: Optional[float] = None
    levels: Optional[List[Any]] = None
    center: Optional[float] = None
    units: Optional[str] = None
    
    def __post_init__(self):
        if self.type == 'continuous' and self.center is None:
            if self.low is not None and self.high is not None:
                self.center = (self.low + self.high) / 2


@dataclass
class Response:
    """반응 변수"""
    name: str
    units: Optional[str] = None
    target: Optional[float] = None
    lower_limit: Optional[float] = None
    upper_limit: Optional[float] = None
    weight: float = 1.0
    goal: str = "maximize"  # maximize, minimize, target


@dataclass
class ValidationReport:
    """데이터 검증 보고서"""
    valid: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class ProcessedData:
    """처리된 데이터"""
    data: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)
    transformations: List[Dict[str, Any]] = field(default_factory=list)
    validation_report: Optional[ValidationReport] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AnalysisResult:
    """분석 결과"""
    analysis_type: str
    results: Dict[str, Any]
    statistics: Dict[str, Any] = field(default_factory=dict)
    visualizations: Dict[str, Any] = field(default_factory=dict)
    interpretation: Optional[str] = None
    ai_insights: Optional[List[str]] = None
    confidence_level: float = 0.95
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OptimizationResult:
    """최적화 결과"""
    optimal_values: Dict[str, float]
    optimal_responses: Dict[str, float]
    convergence_history: Optional[List[float]] = None
    pareto_front: Optional[pd.DataFrame] = None
    sensitivity: Optional[Dict[str, float]] = None
    robustness: Optional[Dict[str, Any]] = None
    confidence_region: Optional[Dict] = None
    constraints_satisfied: bool = True
    timestamp: datetime = field(default_factory=datetime.now)


# === Data Validator ===

class DataValidator:
    """데이터 검증기"""
    
    def __init__(self):
        self.validation_rules = {
            'min_samples': 3,
            'max_missing_ratio': 0.3,
            'outlier_threshold': 3,  # Z-score
            'vif_threshold': 10,  # Variance Inflation Factor
            'correlation_threshold': 0.95
        }
        
        self.checks = {
            'basic': self._check_basic_validity,
            'missing': self._check_missing_values,
            'outliers': self._check_outliers,
            'types': self._check_data_types,
            'distribution': self._check_statistical_assumptions,
            'design': self._check_design_validity
        }
    
    def validate(self, data: pd.DataFrame, 
                 factors: Optional[List[Factor]] = None,
                 responses: Optional[List[Response]] = None,
                 design_type: Optional[DesignType] = None,
                 checks: Optional[List[str]] = None) -> ValidationReport:
        """종합 데이터 검증"""
        report = ValidationReport(valid=True)
        
        if checks is None:
            checks = ['basic', 'missing', 'outliers', 'types', 'distribution']
        
        # 각 검사 수행
        for check_name in checks:
            if check_name in self.checks:
                try:
                    if check_name == 'design':
                        self._check_design_validity(data, design_type, factors, report)
                    else:
                        self.checks[check_name](data, report)
                except Exception as e:
                    logger.error(f"검증 중 오류 ({check_name}): {e}")
                    report.issues.append(f"{check_name} 검증 실패")
        
        # 최종 판정
        report.valid = len(report.issues) == 0
        
        return report
    
    def _check_basic_validity(self, data: pd.DataFrame, report: ValidationReport):
        """기본 유효성 검사"""
        n_rows, n_cols = data.shape
        
        report.statistics['n_samples'] = n_rows
        report.statistics['n_variables'] = n_cols
        
        if n_rows < self.validation_rules['min_samples']:
            report.issues.append(f"샘플 수가 너무 적습니다 ({n_rows} < {self.validation_rules['min_samples']})")
        
        if n_rows < n_cols:
            report.warnings.append("샘플 수가 변수 수보다 적습니다")
        
        # 중복 검사
        n_duplicates = data.duplicated().sum()
        if n_duplicates > 0:
            report.warnings.append(f"{n_duplicates}개의 중복 행 발견")
            report.recommendations.append("중복 데이터 제거를 고려하세요")
    
    def _check_missing_values(self, data: pd.DataFrame, report: ValidationReport):
        """결측치 검사"""
        missing_counts = data.isnull().sum()
        missing_ratio = missing_counts / len(data)
        
        report.statistics['missing_values'] = missing_counts.to_dict()
        
        for col, ratio in missing_ratio.items():
            if ratio > self.validation_rules['max_missing_ratio']:
                report.issues.append(f"{col}: 결측치 비율 과다 ({ratio:.1%})")
            elif ratio > 0:
                report.warnings.append(f"{col}: {missing_counts[col]}개 결측치 ({ratio:.1%})")
        
        # 결측치 패턴 분석
        if missing_counts.sum() > 0:
            missing_pattern = self._analyze_missing_pattern(data)
            if missing_pattern['type'] != 'random':
                report.warnings.append(f"결측치 패턴: {missing_pattern['type']}")
                report.recommendations.append(f"결측치 처리 방법: {missing_pattern['recommendation']}")
    
    def _check_outliers(self, data: pd.DataFrame, report: ValidationReport):
        """이상치 검사"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for col in numeric_cols:
            # Z-score 방법
            z_scores = np.abs(stats.zscore(data[col].dropna()))
            outliers = z_scores > self.validation_rules['outlier_threshold']
            n_outliers = outliers.sum()
            
            # IQR 방법도 확인
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            iqr_outliers = ((data[col] < Q1 - 1.5 * IQR) | (data[col] > Q3 + 1.5 * IQR)).sum()
            
            if n_outliers > 0 or iqr_outliers > 0:
                outlier_info[col] = {
                    'z_score_count': int(n_outliers),
                    'iqr_count': int(iqr_outliers),
                    'percentage': max(n_outliers, iqr_outliers) / len(data) * 100
                }
                
                if outlier_info[col]['percentage'] > 10:
                    report.warnings.append(
                        f"{col}: 이상치 과다 ({outlier_info[col]['percentage']:.1f}%)"
                    )
        
        report.statistics['outliers'] = outlier_info
    
    def _check_data_types(self, data: pd.DataFrame, report: ValidationReport):
        """데이터 타입 검사"""
        for col in data.columns:
            if data[col].dtype == 'object':
                try:
                    pd.to_numeric(data[col], errors='raise')
                    report.recommendations.append(f"{col}: 숫자형으로 변환 가능")
                except:
                    # 범주형 변수 확인
                    n_unique = data[col].nunique()
                    if n_unique < 10:
                        report.statistics[f'{col}_categories'] = data[col].unique().tolist()
    
    def _check_statistical_assumptions(self, data: pd.DataFrame, report: ValidationReport):
        """통계적 가정 검사"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        # 정규성 검정
        normality_results = {}
        for col in numeric_cols:
            if len(data[col].dropna()) >= 8:
                _, p_value = stats.shapiro(data[col].dropna())
                normality_results[col] = p_value
                
                if p_value < 0.05:
                    report.warnings.append(f"{col}: 정규성 가정 위반 (p={p_value:.3f})")
                    report.recommendations.append(f"{col}: 변환을 고려하세요 (Box-Cox, log 등)")
        
        report.statistics['normality_tests'] = normality_results
        
        # 다중공선성 검사 (VIF)
        if len(numeric_cols) > 1:
            vif_results = self._calculate_vif(data[numeric_cols].dropna())
            high_vif = {k: v for k, v in vif_results.items() if v > self.validation_rules['vif_threshold']}
            
            if high_vif:
                report.warnings.append(f"다중공선성 발견: {list(high_vif.keys())}")
                report.statistics['vif'] = vif_results
    
    def _check_design_validity(self, data: pd.DataFrame, design_type: Optional[DesignType], 
                              factors: Optional[List[Factor]], report: ValidationReport):
        """실험 설계 유효성 검사"""
        if not design_type or not factors:
            return
        
        if design_type == DesignType.FACTORIAL:
            # 완전요인설계 검사
            expected_runs = 1
            for factor in factors:
                if factor.type == 'continuous':
                    expected_runs *= 2  # 2수준 가정
                elif factor.levels:
                    expected_runs *= len(factor.levels)
            
            if len(data) < expected_runs:
                report.warnings.append(f"불완전한 요인설계 (실행 수: {len(data)}/{expected_runs})")
        
        elif design_type == DesignType.CENTRAL_COMPOSITE:
            # 중심합성설계 검사
            n_factors = len([f for f in factors if f.type == 'continuous'])
            expected_runs = 2**n_factors + 2*n_factors + 1  # 최소 중심점 1개
            
            if len(data) < expected_runs:
                report.warnings.append(f"CCD 실행 수 부족 ({len(data)}/{expected_runs})")
    
    def _analyze_missing_pattern(self, data: pd.DataFrame) -> Dict[str, str]:
        """결측치 패턴 분석"""
        missing_mask = data.isnull()
        
        # MCAR (Missing Completely At Random) 테스트
        correlations = []
        for col in missing_mask.columns:
            if missing_mask[col].sum() > 0:
                for other_col in data.columns:
                    if col != other_col and data[other_col].dtype in [np.number]:
                        corr = missing_mask[col].astype(int).corr(data[other_col].fillna(0))
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
        
        avg_corr = np.mean(correlations) if correlations else 0
        
        if avg_corr < 0.1:
            return {
                'type': 'random',
                'recommendation': '평균값 또는 중앙값 대체'
            }
        else:
            return {
                'type': 'systematic',
                'recommendation': '다중 대체 또는 모델 기반 대체'
            }
    
    def _calculate_vif(self, data: pd.DataFrame) -> Dict[str, float]:
        """VIF (Variance Inflation Factor) 계산"""
        vif_data = {}
        
        for i, col in enumerate(data.columns):
            try:
                vif_data[col] = variance_inflation_factor(data.values, i)
            except:
                vif_data[col] = np.nan
        
        return vif_data


# === Transformation Engine ===

class TransformationEngine:
    """데이터 변환 엔진"""
    
    def __init__(self):
        self.scalers = {}
        self.transformers = {}
        self.transformation_history = []
    
    def auto_transform(self, data: pd.DataFrame, 
                      target_col: Optional[str] = None) -> Tuple[pd.DataFrame, List[Dict]]:
        """자동 변환 적용"""
        transformed_data = data.copy()
        transformations = []
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col == target_col:
                continue
            
            # 최적 변환 찾기
            best_transform = self._find_best_transform(data[col])
            
            if best_transform['method'] != TransformMethod.NONE:
                transformed_col = self.apply_transform(
                    data[col], 
                    best_transform['method'],
                    best_transform['params']
                )
                
                transformed_data[col] = transformed_col
                transformations.append({
                    'column': col,
                    'method': best_transform['method'].value,
                    'params': best_transform['params'],
                    'improvement': best_transform['improvement']
                })
        
        return transformed_data, transformations
    
    def apply_transform(self, series: pd.Series, 
                       method: Union[str, TransformMethod],
                       params: Optional[Dict] = None) -> pd.Series:
        """변환 적용"""
        if isinstance(method, str):
            method = TransformMethod(method)
        
        clean_series = series.dropna()
        
        if method == TransformMethod.STANDARDIZE:
            scaler = StandardScaler()
            transformed = scaler.fit_transform(clean_series.values.reshape(-1, 1)).flatten()
            self.scalers[f"{series.name}_standard"] = scaler
            
        elif method == TransformMethod.NORMALIZE:
            scaler = MinMaxScaler()
            transformed = scaler.fit_transform(clean_series.values.reshape(-1, 1)).flatten()
            self.scalers[f"{series.name}_minmax"] = scaler
            
        elif method == TransformMethod.ROBUST_SCALE:
            scaler = RobustScaler()
            transformed = scaler.fit_transform(clean_series.values.reshape(-1, 1)).flatten()
            self.scalers[f"{series.name}_robust"] = scaler
            
        elif method == TransformMethod.LOG:
            # 음수 처리
            min_val = clean_series.min()
            if min_val <= 0:
                shift = abs(min_val) + 1
                transformed = np.log(clean_series + shift)
                params = {'shift': shift}
            else:
                transformed = np.log(clean_series)
                
        elif method == TransformMethod.SQRT:
            min_val = clean_series.min()
            if min_val < 0:
                shift = abs(min_val)
                transformed = np.sqrt(clean_series + shift)
                params = {'shift': shift}
            else:
                transformed = np.sqrt(clean_series)
                
        elif method == TransformMethod.RECIPROCAL:
            # 0 방지
            transformed = 1 / (clean_series + 1e-8)
            
        elif method == TransformMethod.BOX_COX:
            if clean_series.min() <= 0:
                shift = abs(clean_series.min()) + 1
                transformed, lambda_param = stats.boxcox(clean_series + shift)
                params = {'lambda': lambda_param, 'shift': shift}
            else:
                transformed, lambda_param = stats.boxcox(clean_series)
                params = {'lambda': lambda_param}
            self.transformers[f"{series.name}_boxcox"] = params
                
        elif method == TransformMethod.YEO_JOHNSON:
            pt = PowerTransformer(method='yeo-johnson')
            transformed = pt.fit_transform(clean_series.values.reshape(-1, 1)).flatten()
            self.transformers[f"{series.name}_yeojohnson"] = pt
            
        elif method == TransformMethod.QUANTILE:
            pt = PowerTransformer(method='quantile')
            transformed = pt.fit_transform(clean_series.values.reshape(-1, 1)).flatten()
            self.transformers[f"{series.name}_quantile"] = pt
            
        else:
            transformed = clean_series.values
        
        # 결과 시리즈 생성
        result = pd.Series(index=series.index, dtype=float)
        result[clean_series.index] = transformed
        
        # 변환 기록
        self.transformation_history.append({
            'column': series.name,
            'method': method.value,
            'params': params,
            'timestamp': datetime.now()
        })
        
        return result
    
    def _find_best_transform(self, series: pd.Series) -> Dict:
        """최적 변환 방법 찾기"""
        clean_series = series.dropna()
        
        if len(clean_series) < 8:
            return {'method': TransformMethod.NONE, 'params': None, 'improvement': 0}
        
        # 원본 정규성과 왜도
        _, original_p = stats.shapiro(clean_series)
        original_skew = abs(clean_series.skew())
        
        # 이미 정규분포에 가까우면 변환 불필요
        if original_p > 0.1 and original_skew < 0.5:
            return {'method': TransformMethod.NONE, 'params': None, 'improvement': 0}
        
        best_method = TransformMethod.NONE
        best_score = original_p
        best_params = None
        
        # 각 변환 방법 시도
        methods_to_try = [
            TransformMethod.LOG, 
            TransformMethod.SQRT,
            TransformMethod.BOX_COX, 
            TransformMethod.YEO_JOHNSON
        ]
        
        for method in methods_to_try:
            try:
                transformed = self.apply_transform(series, method)
                _, p_value = stats.shapiro(transformed.dropna())
                
                if p_value > best_score:
                    best_method = method
                    best_score = p_value
                    best_params = self.transformers.get(f"{series.name}_{method.value}")
                    
            except Exception as e:
                logger.debug(f"Transform {method} failed: {e}")
                continue
        
        improvement = (best_score - original_p) / (original_p + 1e-10) * 100
        
        return {
            'method': best_method,
            'params': best_params,
            'improvement': improvement
        }
    
    def inverse_transform(self, series: pd.Series, 
                         column_name: str,
                         method: TransformMethod,
                         params: Optional[Dict] = None) -> pd.Series:
        """역변환"""
        if method == TransformMethod.STANDARDIZE:
            scaler = self.scalers.get(f"{column_name}_standard")
            if scaler:
                return pd.Series(
                    scaler.inverse_transform(series.values.reshape(-1, 1)).flatten(),
                    index=series.index
                )
                
        elif method == TransformMethod.NORMALIZE:
            scaler = self.scalers.get(f"{column_name}_minmax")
            if scaler:
                return pd.Series(
                    scaler.inverse_transform(series.values.reshape(-1, 1)).flatten(),
                    index=series.index
                )
                
        elif method == TransformMethod.ROBUST_SCALE:
            scaler = self.scalers.get(f"{column_name}_robust")
            if scaler:
                return pd.Series(
                    scaler.inverse_transform(series.values.reshape(-1, 1)).flatten(),
                    index=series.index
                )
                
        elif method == TransformMethod.LOG:
            result = np.exp(series)
            if params and 'shift' in params:
                result -= params['shift']
            return result
            
        elif method == TransformMethod.SQRT:
            result = series ** 2
            if params and 'shift' in params:
                result -= params['shift']
            return result
            
        elif method == TransformMethod.RECIPROCAL:
            return 1 / series
            
        elif method == TransformMethod.BOX_COX:
            params = params or self.transformers.get(f"{column_name}_boxcox")
            if params and 'lambda' in params:
                from scipy.special import inv_boxcox
                result = inv_boxcox(series, params['lambda'])
                if 'shift' in params:
                    result -= params['shift']
                return pd.Series(result, index=series.index)
                
        elif method in [TransformMethod.YEO_JOHNSON, TransformMethod.QUANTILE]:
            transformer_key = f"{column_name}_{method.value}"
            pt = self.transformers.get(transformer_key)
            if pt:
                return pd.Series(
                    pt.inverse_transform(series.values.reshape(-1, 1)).flatten(),
                    index=series.index
                )
        
        return series


# === Experiment Design Engine ===

class ExperimentDesignEngine:
    """실험 설계 엔진 (통합 버전)"""
    
    def __init__(self):
        self.design_functions = {
            DesignType.FACTORIAL: self._create_factorial,
            DesignType.FRACTIONAL_FACTORIAL: self._create_fractional_factorial,
            DesignType.CENTRAL_COMPOSITE: self._create_ccd,
            DesignType.BOX_BEHNKEN: self._create_box_behnken,
            DesignType.LATIN_HYPERCUBE: self._create_latin_hypercube,
            DesignType.PLACKETT_BURMAN: self._create_plackett_burman,
            DesignType.MIXTURE: self._create_mixture,
            DesignType.TAGUCHI: self._create_taguchi,
            DesignType.D_OPTIMAL: self._create_d_optimal
        }
    
    def create_design(self, 
                     factors: List[Factor],
                     design_type: DesignType,
                     **kwargs) -> pd.DataFrame:
        """실험 설계 생성"""
        if design_type not in self.design_functions:
            raise ValueError(f"지원하지 않는 설계 유형: {design_type}")
        
        # 연속형 인자만 추출
        continuous_factors = [f for f in factors if f.type == 'continuous']
        categorical_factors = [f for f in factors if f.type == 'categorical']
        
        if not continuous_factors and design_type not in [DesignType.TAGUCHI, DesignType.FACTORIAL]:
            raise ValueError("연속형 인자가 필요합니다")
        
        # 설계 생성
        design_matrix = self.design_functions[design_type](
            continuous_factors, 
            categorical_factors,
            **kwargs
        )
        
        return design_matrix
    
    def _create_factorial(self, continuous_factors: List[Factor], 
                         categorical_factors: List[Factor],
                         **kwargs) -> pd.DataFrame:
        """완전요인설계"""
        n_continuous = len(continuous_factors)
        n_levels = kwargs.get('n_levels', 2)
        
        if n_continuous > 0:
            # 연속형 인자 설계
            if n_levels == 2:
                design = doe.ff2n(n_continuous)
            else:
                levels = [n_levels] * n_continuous
                design = doe.fullfact(levels)
                # 정규화
                for i in range(n_continuous):
                    design[:, i] = design[:, i] / (n_levels - 1)
                    design[:, i] = design[:, i] * 2 - 1
        else:
            design = np.array([[]])
        
        # 범주형 인자 추가
        if categorical_factors:
            cat_design = self._add_categorical_factors(design, categorical_factors)
            design = cat_design
        
        # 실제 값으로 변환
        df = self._coded_to_actual(design, continuous_factors + categorical_factors)
        
        # 중심점 추가
        n_center = kwargs.get('n_center', 0)
        if n_center > 0 and n_continuous > 0:
            center_runs = self._create_center_runs(continuous_factors, categorical_factors, n_center)
            df = pd.concat([df, center_runs], ignore_index=True)
        
        # 랜덤화
        if kwargs.get('randomize', True):
            df = df.sample(frac=1, random_state=kwargs.get('random_state', RANDOM_STATE))
            df.index = range(1, len(df) + 1)
        
        return df
    
    def _create_fractional_factorial(self, continuous_factors: List[Factor],
                                   categorical_factors: List[Factor],
                                   **kwargs) -> pd.DataFrame:
        """부분요인설계"""
        n_factors = len(continuous_factors)
        resolution = kwargs.get('resolution', 3)
        
        if n_factors < 3:
            # 인자가 적으면 완전요인설계
            return self._create_factorial(continuous_factors, categorical_factors, **kwargs)
        
        # 해상도에 따른 생성자 선택
        if n_factors <= 7:
            generators = self._get_fractional_generators(n_factors, resolution)
            design = doe.fracfact(generators)
        else:
            # Plackett-Burman for screening
            design = doe.pbdesign(n_factors)
            if len(design) > n_factors + 1:
                design = design[:, :n_factors]
        
        # 실제 값으로 변환
        df = self._coded_to_actual(design, continuous_factors)
        
        # 범주형 인자 추가
        if categorical_factors:
            df = self._add_categorical_to_dataframe(df, categorical_factors)
        
        return df
    
    def _create_ccd(self, continuous_factors: List[Factor],
                   categorical_factors: List[Factor],
                   **kwargs) -> pd.DataFrame:
        """중심합성설계 (Central Composite Design)"""
        n_factors = len(continuous_factors)
        
        # CCD 옵션
        alpha = kwargs.get('alpha', 'rotatable')
        center = kwargs.get('center', [4, 4])
        face = kwargs.get('face', 'circumscribed')
        
        # 알파 값 계산
        if alpha == 'rotatable':
            alpha_value = (2 ** n_factors) ** 0.25
        elif alpha == 'orthogonal':
            alpha_value = np.sqrt(n_factors)
        elif alpha == 'face':
            alpha_value = 1
        else:
            alpha_value = float(alpha)
        
        # 설계 생성
        design = doe.ccdesign(n_factors, center=center, alpha=alpha_value, face=face)
        
        # 실제 값으로 변환
        df = self._coded_to_actual(design, continuous_factors)
        
        # 범주형 인자 추가
        if categorical_factors:
            df = self._add_categorical_to_dataframe(df, categorical_factors)
        
        # 블록 정보 추가
        n_factorial = 2 ** n_factors
        n_axial = 2 * n_factors
        blocks = (['factorial'] * n_factorial + 
                  ['axial'] * n_axial + 
                  ['center'] * (len(df) - n_factorial - n_axial))
        df['Block'] = blocks[:len(df)]
        
        return df
    
    def _create_box_behnken(self, continuous_factors: List[Factor],
                           categorical_factors: List[Factor],
                           **kwargs) -> pd.DataFrame:
        """Box-Behnken 설계"""
        n_factors = len(continuous_factors)
        
        if n_factors < 3:
            raise ValueError("Box-Behnken 설계는 최소 3개의 인자가 필요합니다")
        
        # 설계 생성
        center = kwargs.get('center', 3)
        design = doe.bbdesign(n_factors, center=center)
        
        # 실제 값으로 변환
        df = self._coded_to_actual(design, continuous_factors)
        
        # 범주형 인자 추가
        if categorical_factors:
            df = self._add_categorical_to_dataframe(df, categorical_factors)
        
        return df
    
    def _create_latin_hypercube(self, continuous_factors: List[Factor],
                               categorical_factors: List[Factor],
                               **kwargs) -> pd.DataFrame:
        """라틴 하이퍼큐브 설계"""
        n_factors = len(continuous_factors)
        n_samples = kwargs.get('n_samples', max(10, 2 * n_factors))
        
        # LHS 생성
        design = doe.lhs(n_factors, samples=n_samples, random_state=kwargs.get('random_state', RANDOM_STATE))
        
        # [-1, 1]로 스케일링
        design = 2 * design - 1
        
        # 실제 값으로 변환
        df = self._coded_to_actual(design, continuous_factors)
        
        # 범주형 인자 추가
        if categorical_factors:
            df = self._add_categorical_to_dataframe(df, categorical_factors)
        
        return df
    
    def _create_plackett_burman(self, continuous_factors: List[Factor],
                               categorical_factors: List[Factor],
                               **kwargs) -> pd.DataFrame:
        """Plackett-Burman 설계"""
        n_factors = len(continuous_factors) + len(categorical_factors)
        
        # PB 설계 생성
        design = doe.pbdesign(n_factors)
        
        # 연속형과 범주형 분리
        n_continuous = len(continuous_factors)
        continuous_design = design[:, :n_continuous]
        
        # 실제 값으로 변환
        df = self._coded_to_actual(continuous_design, continuous_factors)
        
        # 범주형 인자 처리
        if categorical_factors:
            cat_design = design[:, n_continuous:]
            for i, factor in enumerate(categorical_factors):
                # -1, 1을 범주 레벨로 매핑
                df[factor.name] = [factor.levels[0] if x < 0 else factor.levels[1] 
                                  for x in cat_design[:, i]]
        
        return df
    
    def _create_mixture(self, continuous_factors: List[Factor],
                       categorical_factors: List[Factor],
                       **kwargs) -> pd.DataFrame:
        """혼합물 설계"""
        n_components = len(continuous_factors)
        
        if n_components < 2:
            raise ValueError("혼합물 설계는 최소 2개의 성분이 필요합니다")
        
        # 설계 유형
        design_type = kwargs.get('mixture_type', 'simplex_lattice')
        degree = kwargs.get('degree', 2)
        
        if design_type == 'simplex_lattice':
            design = self._simplex_lattice(n_components, degree)
        elif design_type == 'simplex_centroid':
            design = self._simplex_centroid(n_components)
        else:
            design = self._simplex_lattice(n_components, degree)
        
        # DataFrame 생성
        df = pd.DataFrame(design, columns=[f.name for f in continuous_factors])
        
        # 제약조건 확인
        constraints = kwargs.get('constraints', {})
        if constraints:
            df = self._apply_mixture_constraints(df, constraints)
        
        # 반복 추가
        n_replicates = kwargs.get('n_replicates', 1)
        if n_replicates > 1:
            df = pd.concat([df] * n_replicates, ignore_index=True)
        
        return df
    
    def _create_taguchi(self, continuous_factors: List[Factor],
                       categorical_factors: List[Factor],
                       **kwargs) -> pd.DataFrame:
        """다구치 설계"""
        all_factors = continuous_factors + categorical_factors
        n_factors = len(all_factors)
        
        # 각 인자의 수준 수
        levels = []
        for factor in all_factors:
            if factor.type == 'continuous':
                levels.append(kwargs.get('n_levels', 3))
            else:
                levels.append(len(factor.levels))
        
        # 적절한 직교배열 선택
        oa_type = self._select_taguchi_array(n_factors, levels)
        
        # 직교배열 생성
        design = self._generate_taguchi_array(oa_type, n_factors)
        
        # DataFrame 생성
        df = pd.DataFrame()
        
        for i, factor in enumerate(all_factors):
            if i < design.shape[1]:
                if factor.type == 'continuous':
                    # 연속형: 수준을 실제 값으로 매핑
                    n_levels = int(design[:, i].max() + 1)
                    level_values = np.linspace(factor.low, factor.high, n_levels)
                    df[factor.name] = [level_values[int(x)] for x in design[:, i]]
                else:
                    # 범주형: 직접 매핑
                    df[factor.name] = [factor.levels[int(x) % len(factor.levels)] 
                                      for x in design[:, i]]
        
        # 외부 배열 추가 (노이즈 인자)
        if kwargs.get('outer_array', False):
            df = self._add_outer_array(df, kwargs.get('noise_factors', []))
        
        return df
    
    def _create_d_optimal(self, continuous_factors: List[Factor],
                         categorical_factors: List[Factor],
                         **kwargs) -> pd.DataFrame:
        """D-최적 설계"""
        n_runs = kwargs.get('n_runs', len(continuous_factors) * 4)
        model_type = kwargs.get('model', 'linear')
        
        # 후보 점 생성
        if model_type == 'linear':
            candidates = self._create_factorial(continuous_factors, categorical_factors, n_levels=2)
        elif model_type == 'quadratic':
            candidates = self._create_ccd(continuous_factors, categorical_factors)
        else:
            candidates = self._create_latin_hypercube(
                continuous_factors, categorical_factors, 
                n_samples=n_runs * 10
            )
        
        # 모델 행렬 생성
        X_candidates = self._create_model_matrix(candidates, continuous_factors, model_type)
        
        # D-최적 알고리즘
        selected_indices = self._d_optimal_selection(X_candidates, n_runs)
        
        # 선택된 설계
        optimal_design = candidates.iloc[selected_indices].reset_index(drop=True)
        
        return optimal_design
    
    # === 헬퍼 메서드 ===
    
    def _coded_to_actual(self, coded_design: np.ndarray, 
                        factors: List[Factor]) -> pd.DataFrame:
        """코드화된 설계를 실제 값으로 변환"""
        df = pd.DataFrame()
        
        continuous_idx = 0
        for factor in factors:
            if factor.type == 'continuous':
                if continuous_idx < coded_design.shape[1]:
                    # -1 to 1 -> actual values
                    df[factor.name] = (coded_design[:, continuous_idx] + 1) / 2 * (factor.high - factor.low) + factor.low
                    continuous_idx += 1
        
        return df
    
    def _create_center_runs(self, continuous_factors: List[Factor],
                           categorical_factors: List[Factor],
                           n_center: int) -> pd.DataFrame:
        """중심점 실행 생성"""
        center_data = {}
        
        for factor in continuous_factors:
            center_data[factor.name] = [factor.center] * n_center
        
        # 범주형은 모든 조합 또는 특정 레벨
        for factor in categorical_factors:
            if len(factor.levels) > 0:
                center_data[factor.name] = [factor.levels[0]] * n_center
        
        return pd.DataFrame(center_data)
    
    def _add_categorical_factors(self, continuous_design: np.ndarray,
                                categorical_factors: List[Factor]) -> np.ndarray:
        """범주형 인자를 설계에 추가"""
        if not categorical_factors:
            return continuous_design
        
        # 모든 범주형 조합 생성
        cat_levels = [factor.levels for factor in categorical_factors]
        import itertools
        cat_combinations = list(itertools.product(*cat_levels))
        
        # 연속형 설계와 범주형 조합의 전체 조합
        n_continuous_runs = len(continuous_design) if continuous_design.size > 0 else 1
        n_cat_combinations = len(cat_combinations)
        
        full_design = []
        for cont_row in continuous_design:
            for cat_combo in cat_combinations:
                if continuous_design.size > 0:
                    full_design.append(np.concatenate([cont_row, cat_combo]))
                else:
                    full_design.append(cat_combo)
        
        return np.array(full_design)
    
    def _add_categorical_to_dataframe(self, df: pd.DataFrame,
                                     categorical_factors: List[Factor]) -> pd.DataFrame:
        """DataFrame에 범주형 인자 추가"""
        # 각 범주형 인자에 대해 랜덤 할당
        for factor in categorical_factors:
            n_runs = len(df)
            # 균형잡힌 할당
            n_levels = len(factor.levels)
            assignments = factor.levels * (n_runs // n_levels + 1)
            np.random.shuffle(assignments)
            df[factor.name] = assignments[:n_runs]
        
        return df
    
    def _get_fractional_generators(self, n_factors: int, resolution: int) -> str:
        """부분요인설계 생성자 결정"""
        # 해상도에 따른 생성자
        generators_map = {
            3: {  # Resolution III
                4: 'a b c abc',
                5: 'a b c d abcd',
                6: 'a b c d e bcde',
                7: 'a b c d e f abcdef'
            },
            4: {  # Resolution IV
                4: 'a b c d',
                5: 'a b c d ab',
                6: 'a b c d e ace',
                7: 'a b c d e f abf'
            },
            5: {  # Resolution V
                5: 'a b c d e',
                6: 'a b c d e abc',
                7: 'a b c d e f abcf'
            }
        }
        
        if resolution in generators_map and n_factors in generators_map[resolution]:
            return generators_map[resolution][n_factors]
        else:
            # 기본값
            return 'a b c d e f g h'[:n_factors*2-1]
    
    def _simplex_lattice(self, n_components: int, degree: int) -> np.ndarray:
        """Simplex lattice 설계 생성"""
        points = []
        
        def generate_points(n, d, current=[]):
            if n == 1:
                current.append(d)
                points.append(current[:])
                current.pop()
            else:
                for i in range(d + 1):
                    current.append(i)
                    generate_points(n - 1, d - i, current)
                    current.pop()
        
        generate_points(n_components, degree, [])
        
        # 정규화
        design = np.array(points) / degree
        
        return design
    
    def _simplex_centroid(self, n_components: int) -> np.ndarray:
        """Simplex centroid 설계 생성"""
        points = []
        
        # 단일 성분 점
        for i in range(n_components):
            point = np.zeros(n_components)
            point[i] = 1.0
            points.append(point)
        
        # 이진 혼합물
        for i in range(n_components):
            for j in range(i + 1, n_components):
                point = np.zeros(n_components)
                point[i] = point[j] = 0.5
                points.append(point)
        
        # 삼진 혼합물 (n >= 3)
        if n_components >= 3:
            for i in range(n_components):
                for j in range(i + 1, n_components):
                    for k in range(j + 1, n_components):
                        point = np.zeros(n_components)
                        point[i] = point[j] = point[k] = 1/3
                        points.append(point)
        
        # 중심점
        center = np.ones(n_components) / n_components
        points.append(center)
        
        return np.array(points)
    
    def _apply_mixture_constraints(self, df: pd.DataFrame, 
                                  constraints: Dict) -> pd.DataFrame:
        """혼합물 제약조건 적용"""
        valid_rows = []
        
        for idx, row in df.iterrows():
            valid = True
            for component, (low, high) in constraints.items():
                if component in row:
                    if row[component] < low or row[component] > high:
                        valid = False
                        break
            if valid:
                valid_rows.append(idx)
        
        return df.loc[valid_rows].reset_index(drop=True)
    
    def _select_taguchi_array(self, n_factors: int, levels: List[int]) -> str:
        """적절한 다구치 직교배열 선택"""
        max_level = max(levels)
        
        if max_level == 2:
            if n_factors <= 3:
                return 'L4'
            elif n_factors <= 7:
                return 'L8'
            elif n_factors <= 15:
                return 'L16'
        elif max_level == 3:
            if n_factors <= 4:
                return 'L9'
            elif n_factors <= 13:
                return 'L27'
        
        return 'L16'
    
    def _generate_taguchi_array(self, oa_type: str, n_factors: int) -> np.ndarray:
        """다구치 직교배열 생성"""
        if oa_type == 'L4':
            design = np.array([[0,0,0], [0,1,1], [1,0,1], [1,1,0]])
        elif oa_type == 'L8':
            design = doe.fracfact('a b c d=abc')
            design = (design + 1) / 2  # 0, 1로 변환
        elif oa_type == 'L9':
            design = np.array([
                [0,0,0,0], [0,1,1,1], [0,2,2,2],
                [1,0,1,2], [1,1,2,0], [1,2,0,1],
                [2,0,2,1], [2,1,0,2], [2,2,1,0]
            ])
        elif oa_type == 'L16':
            design = doe.fracfact('a b c d e=abcd')
            design = (design + 1) / 2
        elif oa_type == 'L27':
            # 간단한 L27 구현
            design = self._generate_l27_array()
        else:
            # 기본값: L8
            design = doe.fracfact('a b c')
            design = (design + 1) / 2
        
        # 필요한 열만 선택
        return design[:, :n_factors]
    
    def _generate_l27_array(self) -> np.ndarray:
        """L27 직교배열 생성"""
        # 3^13 설계의 부분 (간단한 구현)
        l27 = []
        for i in range(27):
            row = []
            temp = i
            for j in range(13):
                row.append(temp % 3)
                temp //= 3
            l27.append(row)
        
        return np.array(l27)
    
    def _add_outer_array(self, inner_array: pd.DataFrame, 
                        noise_factors: List[Factor]) -> pd.DataFrame:
        """외부 배열 추가 (다구치 방법)"""
        if not noise_factors:
            return inner_array
        
        # 노이즈 인자에 대한 간단한 2수준 설계
        n_noise = len(noise_factors)
        if n_noise <= 3:
            noise_design = doe.ff2n(n_noise)
        else:
            noise_design = doe.pbdesign(n_noise)[:, :n_noise]
        
        # 내부 배열과 외부 배열 조합
        result_dfs = []
        
        for i in range(len(noise_design)):
            df_copy = inner_array.copy()
            # 노이즈 조건 추가
            for j, factor in enumerate(noise_factors):
                level = 'low' if noise_design[i, j] < 0 else 'high'
                df_copy[f'{factor.name}_noise'] = level
            df_copy['noise_condition'] = i + 1
            result_dfs.append(df_copy)
        
        return pd.concat(result_dfs, ignore_index=True)
    
    def _create_model_matrix(self, df: pd.DataFrame, 
                           continuous_factors: List[Factor],
                           model_type: str) -> np.ndarray:
        """모델 행렬 생성"""
        factor_names = [f.name for f in continuous_factors]
        X = df[factor_names].values
        
        if model_type == 'linear':
            # 절편 + 주효과
            X_model = np.column_stack([np.ones(len(X)), X])
        elif model_type == 'quadratic':
            # 절편 + 주효과 + 제곱항 + 교호작용
            poly = PolynomialFeatures(degree=2, include_bias=True)
            X_model = poly.fit_transform(X)
        else:
            # 3차 이상
            poly = PolynomialFeatures(degree=3, include_bias=True)
            X_model = poly.fit_transform(X)
        
        return X_model
    
    def _d_optimal_selection(self, X: np.ndarray, n_runs: int) -> List[int]:
        """D-최적 설계 선택 알고리즘"""
        n_candidates, n_terms = X.shape
        
        if n_runs >= n_candidates:
            return list(range(n_candidates))
        
        # 초기 설계: 랜덤 선택
        selected = np.random.choice(n_candidates, n_runs, replace=False).tolist()
        
        # 교환 알고리즘
        max_iterations = 100
        for _ in range(max_iterations):
            improved = False
            
            # 현재 설계의 D-효율성
            X_current = X[selected]
            try:
                current_det = np.linalg.det(X_current.T @ X_current)
            except:
                current_det = 0
            
            # 각 점을 교환해보기
            for i in range(n_runs):
                for j in range(n_candidates):
                    if j not in selected:
                        # 교환
                        new_selected = selected.copy()
                        new_selected[i] = j
                        
                        X_new = X[new_selected]
                        try:
                            new_det = np.linalg.det(X_new.T @ X_new)
                        except:
                            new_det = 0
                        
                        # 개선되면 업데이트
                        if new_det > current_det:
                            selected = new_selected
                            current_det = new_det
                            improved = True
                            break
                
                if improved:
                    break
            
            if not improved:
                break
        
        return selected


# === Statistical Analyzer (계속) ===

class StatisticalAnalyzer:
    """통계 분석 엔진 (통합 버전)"""
    
    def __init__(self, api_manager: Optional[APIManager] = None):
        self.api_manager = api_manager
        self._ai_detail_level = 'auto'
    
    def set_ai_detail_level(self, level: str):
        """AI 설명 상세도 설정"""
        self._ai_detail_level = level
    
    def analyze(self, data: pd.DataFrame, 
               analysis_type: str,
               **kwargs) -> AnalysisResult:
        """통계 분석 수행"""
        
        analysis_functions = {
            'descriptive': self._descriptive_analysis,
            'correlation': self._correlation_analysis,
            'anova': self._anova_analysis,
            'regression': self._regression_analysis,
            'rsm': self._rsm_analysis,
            'pca': self._pca_analysis,
            'time_series': self._time_series_analysis
        }
        
        if analysis_type not in analysis_functions:
            raise ValueError(f"지원하지 않는 분석 유형: {analysis_type}")
        
        # 분석 수행
        results = analysis_functions[analysis_type](data, **kwargs)
        
        # AI 인사이트 생성
        if self.api_manager and kwargs.get('use_ai', True):
            ai_insights = self._generate_ai_insights(results, analysis_type)
            results.ai_insights = ai_insights
        
        return results
    
    def _descriptive_analysis(self, data: pd.DataFrame, **kwargs) -> AnalysisResult:
        """기술통계 분석"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        results = {
            'summary': data[numeric_cols].describe().to_dict(),
            'skewness': data[numeric_cols].skew().to_dict(),
            'kurtosis': data[numeric_cols].kurtosis().to_dict()
        }
        
        # 정규성 검정
        normality_tests = {}
        for col in numeric_cols:
            if len(data[col].dropna()) >= 8:
                stat, p_value = stats.shapiro(data[col].dropna())
                normality_tests[col] = {
                    'statistic': stat,
                    'p_value': p_value,
                    'is_normal': p_value > 0.05
                }
        
        results['normality_tests'] = normality_tests
        
        # 시각화 데이터
        visualizations = {
            'histogram_data': {col: data[col].dropna().tolist() for col in numeric_cols},
            'box_plot_data': {col: {
                'y': data[col].dropna().tolist(),
                'name': col
            } for col in numeric_cols}
        }
        
        return AnalysisResult(
            analysis_type='descriptive',
            results=results,
            statistics={
                'n_samples': len(data),
                'n_variables': len(numeric_cols),
                'missing_values': data[numeric_cols].isnull().sum().to_dict()
            },
            visualizations=visualizations
        )
    
    def _correlation_analysis(self, data: pd.DataFrame, **kwargs) -> AnalysisResult:
        """상관관계 분석"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        method = kwargs.get('method', 'pearson')
        
        # 상관계수 행렬
        corr_matrix = data[numeric_cols].corr(method=method)
        
        # p-값 계산
        p_values = pd.DataFrame(np.zeros_like(corr_matrix), 
                               columns=corr_matrix.columns, 
                               index=corr_matrix.index)
        
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                col1, col2 = numeric_cols[i], numeric_cols[j]
                if method == 'pearson':
                    r, p = stats.pearsonr(data[col1].dropna(), data[col2].dropna())
                elif method == 'spearman':
                    r, p = stats.spearmanr(data[col1].dropna(), data[col2].dropna())
                else:
                    r, p = stats.kendalltau(data[col1].dropna(), data[col2].dropna())
                
                p_values.loc[col1, col2] = p
                p_values.loc[col2, col1] = p
        
        # 유의한 상관관계 찾기
        significant_correlations = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                if p_values.iloc[i, j] < 0.05:
                    significant_correlations.append({
                        'var1': numeric_cols[i],
                        'var2': numeric_cols[j],
                        'correlation': corr_matrix.iloc[i, j],
                        'p_value': p_values.iloc[i, j]
                    })
        
        # 시각화 데이터
        visualizations = {
            'heatmap': {
                'z': corr_matrix.values.tolist(),
                'x': corr_matrix.columns.tolist(),
                'y': corr_matrix.index.tolist()
            }
        }
        
        return AnalysisResult(
            analysis_type='correlation',
            results={
                'correlation_matrix': corr_matrix.to_dict(),
                'p_values': p_values.to_dict(),
                'significant_correlations': significant_correlations
            },
            statistics={
                'method': method,
                'n_variables': len(numeric_cols),
                'n_significant': len(significant_correlations)
            },
            visualizations=visualizations
        )
    
    def _anova_analysis(self, data: pd.DataFrame, **kwargs) -> AnalysisResult:
        """분산분석 (ANOVA)"""
        factor_cols = kwargs.get('factor_cols', [])
        response_col = kwargs.get('response_col')
        
        if not factor_cols or not response_col:
            raise ValueError("factor_cols와 response_col이 필요합니다")
        
        results = {}
        
        # One-way ANOVA
        if len(factor_cols) == 1:
            groups = []
            factor = factor_cols[0]
            
            for level in data[factor].unique():
                group_data = data[data[factor] == level][response_col].dropna()
                groups.append(group_data)
            
            f_stat, p_value = f_oneway(*groups)
            
            results['one_way_anova'] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
            
            # 사후 검정 (Tukey HSD)
            if p_value < 0.05:
                tukey_result = pairwise_tukeyhsd(
                    data[response_col].dropna(),
                    data[factor].dropna()
                )
                results['post_hoc'] = {
                    'method': 'Tukey HSD',
                    'results': str(tukey_result)
                }
        
        # Two-way or multi-way ANOVA
        else:
            # 모델 구성
            formula_parts = [f"C({factor})" for factor in factor_cols]
            
            # 주효과
            formula = f"{response_col} ~ " + " + ".join(formula_parts)
            
            # 교호작용 (2-way interactions)
            if kwargs.get('include_interactions', True) and len(factor_cols) > 1:
                interactions = []
                for i in range(len(factor_cols)):
                    for j in range(i+1, len(factor_cols)):
                        interactions.append(f"C({factor_cols[i]}):C({factor_cols[j]})")
                formula += " + " + " + ".join(interactions)
            
            model = ols(formula, data=data).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            results['anova_table'] = anova_table.to_dict()
            
            # 주효과와 교호작용 검정
            results['main_effects'] = {}
            results['interactions'] = {}
            
            for index, row in anova_table.iterrows():
                if ':' not in index and index != 'Residual':
                    # 주효과
                    factor_name = index.replace('C(', '').replace(')', '')
                    results['main_effects'][factor_name] = {
                        'f_statistic': row['F'],
                        'p_value': row['PR(>F)'],
                        'significant': row['PR(>F)'] < 0.05
                    }
                elif ':' in index:
                    # 교호작용
                    results['interactions'][index] = {
                        'f_statistic': row['F'],
                        'p_value': row['PR(>F)'],
                        'significant': row['PR(>F)'] < 0.05
                    }
        
        # 시각화 데이터
        visualizations = {}
        
        if len(factor_cols) == 1:
            # Box plot
            box_data = []
            for level in data[factor_cols[0]].unique():
                box_data.append({
                    'y': data[data[factor_cols[0]] == level][response_col].dropna().tolist(),
                    'name': str(level)
                })
            visualizations['box_plot'] = box_data
        
        return AnalysisResult(
            analysis_type='anova',
            results=results,
            statistics={
                'n_factors': len(factor_cols),
                'n_observations': len(data),
                'response_variable': response_col
            },
            visualizations=visualizations
        )
    
    def _regression_analysis(self, data: pd.DataFrame, **kwargs) -> AnalysisResult:
        """회귀분석"""
        predictors = kwargs.get('predictors', [])
        response = kwargs.get('response')
        poly_degree = kwargs.get('poly_degree', 1)
        
        if not predictors or not response:
            raise ValueError("predictors와 response가 필요합니다")
        
        # 데이터 준비
        X = data[predictors].dropna()
        y = data[response].dropna()
        
        # 인덱스 정렬
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        
        # 다항식 특징 추가
        if poly_degree > 1:
            poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
            X_poly = poly.fit_transform(X)
            feature_names = poly.get_feature_names_out(predictors)
            X = pd.DataFrame(X_poly, columns=feature_names, index=X.index)
        
        # 회귀 모델
        X_with_const = sm.add_constant(X)
        model = sm.OLS(y, X_with_const).fit()
        
        # 잔차 분석
        residuals = model.resid
        fitted_values = model.fittedvalues
        
        # 잔차 정규성 검정
        _, residual_normality_p = stats.shapiro(residuals)
        
        # 등분산성 검정
        _, bp_p_value, _, _ = het_breuschpagan(residuals, X_with_const)
        
        # VIF 계산
        vif_data = {}
        if len(predictors) > 1:
            for i, col in enumerate(X.columns):
                try:
                    vif_data[col] = variance_inflation_factor(X.values, i)
                except:
                    vif_data[col] = np.nan
        
        results = {
            'model_summary': {
                'r_squared': model.rsquared,
                'adj_r_squared': model.rsquared_adj,
                'f_statistic': model.fvalue,
                'f_pvalue': model.f_pvalue,
                'aic': model.aic,
                'bic': model.bic
            },
            'coefficients': model.params.to_dict(),
            'p_values': model.pvalues.to_dict(),
            'confidence_intervals': model.conf_int().to_dict(),
            'residual_analysis': {
                'normality_p_value': residual_normality_p,
                'homoscedasticity_p_value': bp_p_value,
                'residuals_normal': residual_normality_p > 0.05,
                'homoscedastic': bp_p_value > 0.05
            },
            'vif': vif_data,
            'model': model  # 모델 객체 저장
        }
        
        # 시각화 데이터
        visualizations = {
            'residual_plot': {
                'x': fitted_values.tolist(),
                'y': residuals.tolist()
            },
            'qq_plot': {
                'theoretical': stats.probplot(residuals, dist="norm")[0][0].tolist(),
                'sample': stats.probplot(residuals, dist="norm")[0][1].tolist()
            }
        }
        
        # 예측 구간
        if len(predictors) == 1:
            x_range = np.linspace(X[predictors[0]].min(), X[predictors[0]].max(), 100)
            X_pred = pd.DataFrame({predictors[0]: x_range})
            if poly_degree > 1:
                X_pred = pd.DataFrame(
                    poly.transform(X_pred),
                    columns=feature_names
                )
            X_pred_with_const = sm.add_constant(X_pred)
            
            predictions = model.get_prediction(X_pred_with_const)
            pred_summary = predictions.summary_frame(alpha=0.05)
            
            visualizations['fitted_line'] = {
                'x': x_range.tolist(),
                'y': pred_summary['mean'].tolist(),
                'lower': pred_summary['mean_ci_lower'].tolist(),
                'upper': pred_summary['mean_ci_upper'].tolist()
            }
        
        return AnalysisResult(
            analysis_type='regression',
            results=results,
            statistics={
                'n_observations': len(y),
                'n_predictors': len(predictors),
                'polynomial_degree': poly_degree
            },
            visualizations=visualizations
        )
    
    def _rsm_analysis(self, data: pd.DataFrame, **kwargs) -> AnalysisResult:
        """반응표면분석 (Response Surface Methodology)"""
        factors = kwargs.get('factors', [])
        response = kwargs.get('response')
        
        if len(factors) < 2:
            raise ValueError("RSM은 최소 2개의 인자가 필요합니다")
        
        # 2차 회귀모델
        results = self._regression_analysis(
            data, 
            predictors=factors, 
            response=response,
            poly_degree=2
        )
        
        # 정준 분석 (Canonical Analysis)
        model = results.results['model']
        
        # 정상점 찾기
        stationary_point = self._find_stationary_point(model, factors)
        
        # 고유값 분석
        eigenanalysis = self._canonical_analysis(model, factors)
        
        # 능선 분석 (Ridge Analysis)
        ridge_analysis = self._ridge_analysis(data, factors, response)
        
        # RSM 특화 결과 추가
        rsm_results = {
            'regression': results.results,
            'stationary_point': stationary_point,
            'eigenanalysis': eigenanalysis,
            'ridge_analysis': ridge_analysis,
            'surface_type': self._determine_surface_type(eigenanalysis)
        }
        
        # 3D 표면 플롯 데이터
        if len(factors) == 2:
            surface_data = self._generate_surface_data(model, factors, data)
            results.visualizations['surface_plot'] = surface_data
        
        results.results = rsm_results
        results.analysis_type = 'rsm'
        
        return results
    
    def _pca_analysis(self, data: pd.DataFrame, **kwargs) -> AnalysisResult:
        """주성분 분석 (PCA)"""
        numeric_cols = kwargs.get('variables', data.select_dtypes(include=[np.number]).columns)
        n_components = kwargs.get('n_components', min(len(numeric_cols), len(data)))
        
        # 데이터 준비
        X = data[numeric_cols].dropna()
        
        # 표준화
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # 결과 정리
        results = {
            'explained_variance': pca.explained_variance_.tolist(),
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'components': pca.components_.tolist(),
            'loadings': pd.DataFrame(
                pca.components_.T,
                index=numeric_cols,
                columns=[f'PC{i+1}' for i in range(n_components)]
            ).to_dict(),
            'scores': pd.DataFrame(
                X_pca,
                index=X.index,
                columns=[f'PC{i+1}' for i in range(n_components)]
            ).to_dict()
        }
        
        # Kaiser 기준으로 주성분 수 추천
        n_kaiser = sum(pca.explained_variance_ > 1)
        
        # 시각화 데이터
        visualizations = {
            'scree_plot': {
                'x': list(range(1, n_components + 1)),
                'y': pca.explained_variance_.tolist()
            },
            'biplot': {
                'scores': X_pca[:, :2].tolist() if n_components >= 2 else [],
                'loadings': (pca.components_[:2].T * np.sqrt(pca.explained_variance_[:2])).tolist()
                           if n_components >= 2 else [],
                'labels': numeric_cols.tolist()
            }
        }
        
        return AnalysisResult(
            analysis_type='pca',
            results=results,
            statistics={
                'n_variables': len(numeric_cols),
                'n_observations': len(X),
                'n_components': n_components,
                'n_components_kaiser': n_kaiser,
                'total_variance_explained': sum(pca.explained_variance_ratio_[:n_kaiser])
            },
            visualizations=visualizations
        )
    
    def _time_series_analysis(self, data: pd.DataFrame, **kwargs) -> AnalysisResult:
        """시계열 분석"""
        time_col = kwargs.get('time_column')
        value_col = kwargs.get('value_column')
        
        if not time_col or not value_col:
            raise ValueError("time_column과 value_column이 필요합니다")
        
        # 시계열 데이터 준비
        ts_data = data[[time_col, value_col]].dropna()
        ts_data = ts_data.sort_values(time_col)
        ts_data.set_index(time_col, inplace=True)
        
        # 기본 통계
        results = {
            'summary': {
                'mean': ts_data[value_col].mean(),
                'std': ts_data[value_col].std(),
                'min': ts_data[value_col].min(),
                'max': ts_data[value_col].max(),
                'trend': self._detect_trend(ts_data[value_col])
            }
        }
        
        # 자기상관 분석
        acf_values = sm.tsa.acf(ts_data[value_col], nlags=20)
        pacf_values = sm.tsa.pacf(ts_data[value_col], nlags=20)
        
        results['autocorrelation'] = {
            'acf': acf_values.tolist(),
            'pacf': pacf_values.tolist()
        }
        
        # Ljung-Box 검정
        lb_result = acorr_ljungbox(ts_data[value_col], lags=10, return_df=True)
        results['ljung_box_test'] = lb_result.to_dict()
        
        # 이동평균
        if kwargs.get('moving_average'):
            window = kwargs.get('window_size', 3)
            results['moving_average'] = ts_data[value_col].rolling(window=window).mean().to_dict()
        
        # 시각화 데이터
        visualizations = {
            'time_series_plot': {
                'x': ts_data.index.tolist(),
                'y': ts_data[value_col].tolist()
            },
            'acf_plot': {
                'lags': list(range(len(acf_values))),
                'values': acf_values.tolist()
            },
            'pacf_plot': {
                'lags': list(range(len(pacf_values))),
                'values': pacf_values.tolist()
            }
        }
        
        return AnalysisResult(
            analysis_type='time_series',
            results=results,
            statistics={
                'n_observations': len(ts_data),
                'time_range': f"{ts_data.index.min()} to {ts_data.index.max()}"
            },
            visualizations=visualizations
        )
    
    # === RSM 헬퍼 메서드 ===
    
    def _find_stationary_point(self, model, factors: List[str]) -> Dict:
        """정상점 찾기"""
        # 회귀계수 추출
        params = model.params
        
        # 1차 및 2차 계수 행렬 구성
        n_factors = len(factors)
        b = np.zeros(n_factors)  # 1차 계수
        B = np.zeros((n_factors, n_factors))  # 2차 계수
        
        for i, factor in enumerate(factors):
            if factor in params.index:
                b[i] = params[factor]
            
            # 제곱항
            squared_term = f'{factor}^2'
            if squared_term in params.index:
                B[i, i] = params[squared_term]
            
            # 교호작용항
            for j, other_factor in enumerate(factors):
                if i < j:
                    interaction = f'{factor}:{other_factor}'
                    if interaction in params.index:
                        B[i, j] = B[j, i] = params[interaction] / 2
        
        # 정상점 계산: x_s = -0.5 * B^(-1) * b
        try:
            B_inv = np.linalg.inv(B)
            stationary_point = -0.5 * B_inv @ b
            
            # 정상점에서의 예측값
            X_s = np.ones(len(params))
            X_s[0] = 1  # 절편
            for i, factor in enumerate(factors):
                idx = list(params.index).index(factor)
                X_s[idx] = stationary_point[i]
            
            y_s = X_s @ params.values
            
            return {
                'coordinates': {factors[i]: stationary_point[i] for i in range(n_factors)},
                'predicted_response': y_s,
                'type': self._classify_stationary_point(B)
            }
        except np.linalg.LinAlgError:
            return {
                'coordinates': None,
                'predicted_response': None,
                'type': 'undefined'
            }
    
    def _canonical_analysis(self, model, factors: List[str]) -> Dict:
        """정준 분석"""
        # 2차 계수 행렬 구성
        params = model.params
        n_factors = len(factors)
        B = np.zeros((n_factors, n_factors))
        
        for i, factor in enumerate(factors):
            # 제곱항
            squared_term = f'{factor}^2'
            if squared_term in params.index:
                B[i, i] = params[squared_term]
            
            # 교호작용항
            for j, other_factor in enumerate(factors):
                if i < j:
                    interaction = f'{factor}:{other_factor}'
                    if interaction in params.index:
                        B[i, j] = B[j, i] = params[interaction] / 2
        
        # 고유값과 고유벡터
        eigenvalues, eigenvectors = np.linalg.eig(B)
        
        # 정렬 (절대값 기준)
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        return {
            'eigenvalues': eigenvalues.tolist(),
            'eigenvectors': eigenvectors.tolist(),
            'principal_axes': [
                {factors[j]: eigenvectors[j, i] for j in range(n_factors)}
                for i in range(n_factors)
            ]
        }
    
    def _ridge_analysis(self, data: pd.DataFrame, 
                       factors: List[str], 
                       response: str) -> Dict:
        """능선 분석"""
        # 중심점에서 시작
        center = {factor: data[factor].mean() for factor in factors}
        
        # 여러 반경에서 최적점 찾기
        radii = np.linspace(0, 2, 11)[1:]  # 0 제외
        ridge_path = []
        
        for radius in radii:
            # 제약 최적화: maximize y subject to ||x - center|| = radius
            def objective(x):
                # 예측값 계산 (간단히 2차 모델 가정)
                return -self._predict_response(x, factors, data, response)
            
            def constraint(x):
                return np.sum([(x[i] - center[factors[i]])**2 for i in range(len(factors))]) - radius**2
            
            # 최적화
            x0 = np.array([center[f] for f in factors])
            bounds = [(data[f].min(), data[f].max()) for f in factors]
            
            result = optimize.minimize(
                objective, x0, 
                method='SLSQP',
                bounds=bounds,
                constraints={'type': 'eq', 'fun': constraint}
            )
            
            if result.success:
                ridge_path.append({
                    'radius': radius,
                    'coordinates': {factors[i]: result.x[i] for i in range(len(factors))},
                    'predicted_response': -result.fun
                })
        
        return {
            'center': center,
            'ridge_path': ridge_path
        }
    
    def _determine_surface_type(self, eigenanalysis: Dict) -> str:
        """표면 유형 결정"""
        eigenvalues = eigenanalysis['eigenvalues']
        
        if all(e > 0 for e in eigenvalues):
            return 'minimum'
        elif all(e < 0 for e in eigenvalues):
            return 'maximum'
        elif any(e > 0 for e in eigenvalues) and any(e < 0 for e in eigenvalues):
            return 'saddle_point'
        else:
            return 'stationary_ridge'
    
    def _classify_stationary_point(self, B: np.ndarray) -> str:
        """정상점 분류"""
        eigenvalues = np.linalg.eigvals(B)
        
        if all(e > 0 for e in eigenvalues):
            return 'minimum'
        elif all(e < 0 for e in eigenvalues):
            return 'maximum'
        else:
            return 'saddle_point'
    
    def _generate_surface_data(self, model, factors: List[str], data: pd.DataFrame) -> Dict:
        """3D 표면 데이터 생성"""
        if len(factors) != 2:
            return {}
        
        # 격자 생성
        x_range = np.linspace(data[factors[0]].min(), data[factors[0]].max(), 50)
        y_range = np.linspace(data[factors[1]].min(), data[factors[1]].max(), 50)
        X_grid, Y_grid = np.meshgrid(x_range, y_range)
        
        # 예측값 계산
        Z_grid = np.zeros_like(X_grid)
        
        for i in range(len(x_range)):
            for j in range(len(y_range)):
                # 2차 모델로 예측
                x_point = pd.DataFrame({
                    factors[0]: [X_grid[j, i]],
                    factors[1]: [Y_grid[j, i]]
                })
                
                # 다항식 특징 생성
                poly = PolynomialFeatures(degree=2, include_bias=False)
                x_poly = poly.fit_transform(x_point)
                x_poly_df = pd.DataFrame(
                    x_poly,
                    columns=poly.get_feature_names_out(factors)
                )
                x_with_const = sm.add_constant(x_poly_df)
                
                # 예측
                Z_grid[j, i] = model.predict(x_with_const)[0]
        
        return {
            'x': x_range.tolist(),
            'y': y_range.tolist(),
            'z': Z_grid.tolist()
        }
    
    def _predict_response(self, x: np.ndarray, factors: List[str], 
                         data: pd.DataFrame, response: str) -> float:
        """주어진 점에서 반응값 예측"""
        # 간단한 2차 모델 가정
        prediction = data[response].mean()
        
        # 주효과
        for i, factor in enumerate(factors):
            prediction += x[i] * data[response].corr(data[factor]) * data[response].std()
        
        # 제곱효과
        for i, factor in enumerate(factors):
            prediction -= (x[i] - data[factor].mean())**2 * 0.05 * data[response].std()
        
        return prediction
    
    def _detect_trend(self, series: pd.Series) -> str:
        """추세 감지"""
        x = np.arange(len(series))
        y = series.values
        
        # 선형 회귀
        slope, intercept = np.polyfit(x, y, 1)
        
        # 추세 판단
        if abs(slope) < 0.01 * np.std(y):
            return 'stationary'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    def _generate_ai_insights(self, results: AnalysisResult, analysis_type: str) -> List[str]:
        """AI 기반 인사이트 생성"""
        if not self.api_manager:
            return []
        
        insights = []
        
        # 분석 유형별 프롬프트 생성
        prompt_functions = {
            'descriptive': self._create_descriptive_prompt,
            'correlation': self._create_correlation_prompt,
            'anova': self._create_anova_prompt,
            'regression': self._create_regression_prompt,
            'rsm': self._create_rsm_prompt,
            'pca': self._create_pca_prompt,
            'time_series': self._create_time_series_prompt
        }
        
        if analysis_type in prompt_functions:
            prompt = prompt_functions[analysis_type](results)
            
            # AI 호출
            try:
                ai_response = self.api_manager.generate_ai_response(
                    prompt=prompt,
                    model='gemini',
                    system_prompt="당신은 전문 통계 분석가입니다. 간결하고 실용적인 인사이트를 제공하세요."
                )
                
                # 인사이트 파싱
                insights = self._parse_ai_insights(ai_response)
                
            except Exception as e:
                logger.error(f"AI 인사이트 생성 실패: {e}")
        
        return insights
    
    def _create_descriptive_prompt(self, results: AnalysisResult) -> str:
        """기술통계 프롬프트"""
        stats = results.results
        
        prompt = f"""
        다음 기술통계 분석 결과를 해석하고 주요 인사이트를 제공해주세요:
        
        변수별 요약:
        {json.dumps(stats.get('summary', {}), indent=2)}
        
        정규성 검정:
        {json.dumps(stats.get('normality_tests', {}), indent=2)}
        
        왜도: {json.dumps(stats.get('skewness', {}), indent=2)}
        첨도: {json.dumps(stats.get('kurtosis', {}), indent=2)}
        
        다음 관점에서 분석해주세요:
        1. 데이터의 전반적인 분포 특성
        2. 이상치나 극단값의 가능성
        3. 변수 변환의 필요성
        4. 추가 분석을 위한 권장사항
        
        {self._get_detail_instruction()}
        """
        
        return prompt
    
    def _create_correlation_prompt(self, results: AnalysisResult) -> str:
        """상관분석 프롬프트"""
        corr_data = results.results
        
        prompt = f"""
        상관관계 분석 결과를 해석해주세요:
        
        유의한 상관관계:
        {json.dumps(corr_data.get('significant_correlations', []), indent=2)}
        
        분석 방법: {results.statistics.get('method', 'pearson')}
        
        다음을 포함해서 설명해주세요:
        1. 강한 상관관계를 보이는 변수들
        2. 다중공선성 위험이 있는 변수들
        3. 예상외의 상관관계
        4. 인과관계 해석 시 주의사항
        
        {self._get_detail_instruction()}
        """
        
        return prompt
    
    def _create_anova_prompt(self, results: AnalysisResult) -> str:
        """ANOVA 프롬프트"""
        anova_results = results.results
        
        prompt = f"""
        분산분석(ANOVA) 결과를 해석해주세요:
        
        {json.dumps(anova_results, indent=2)}
        
        다음 내용을 포함해주세요:
        1. 주효과의 유의성과 실제적 의미
        2. 교호작용 효과 (있는 경우)
        3. 사후검정 결과 해석
        4. 실험 설계 개선을 위한 제안
        
        {self._get_detail_instruction()}
        """
        
        return prompt
    
    def _create_regression_prompt(self, results: AnalysisResult) -> str:
        """회귀분석 프롬프트"""
        reg_results = results.results
        
        prompt = f"""
        회귀분석 결과를 해석해주세요:
        
        모델 요약:
        {json.dumps(reg_results.get('model_summary', {}), indent=2)}
        
        회귀계수:
        {json.dumps(reg_results.get('coefficients', {}), indent=2)}
        
        잔차 분석:
        {json.dumps(reg_results.get('residual_analysis', {}), indent=2)}
        
        다음을 분석해주세요:
        1. 모델의 전반적인 적합도
        2. 중요한 예측변수와 그 영향
        3. 모델 가정의 충족 여부
        4. 예측력 향상을 위한 제안
        
        {self._get_detail_instruction()}
        """
        
        return prompt
    
    def _create_rsm_prompt(self, results: AnalysisResult) -> str:
        """RSM 프롬프트"""
        rsm_results = results.results
        
        prompt = f"""
        반응표면분석(RSM) 결과를 해석해주세요:
        
        정상점: {json.dumps(rsm_results.get('stationary_point', {}), indent=2)}
        표면 유형: {rsm_results.get('surface_type', '')}
        
        고유값 분석:
        {json.dumps(rsm_results.get('eigenanalysis', {}), indent=2)}
        
        다음을 포함해서 설명해주세요:
        1. 최적 조건과 그 신뢰성
        2. 반응표면의 형태와 특성
        3. 강건성(robustness) 평가
        4. 추가 실험 권장 영역
        
        {self._get_detail_instruction()}
        """
        
        return prompt
    
    def _create_pca_prompt(self, results: AnalysisResult) -> str:
        """PCA 프롬프트"""
        pca_results = results.results
        
        prompt = f"""
        주성분 분석(PCA) 결과를 해석해주세요:
        
        설명된 분산:
        {json.dumps(pca_results.get('explained_variance_ratio', []), indent=2)}
        
        주성분 수: {results.statistics.get('n_components_kaiser')}개 권장 (Kaiser 기준)
        
        다음을 분석해주세요:
        1. 차원 축소의 효과성
        2. 주요 주성분의 해석
        3. 원래 변수들의 기여도
        4. 추가 분석 방향
        
        {self._get_detail_instruction()}
        """
        
        return prompt
    
    def _create_time_series_prompt(self, results: AnalysisResult) -> str:
        """시계열 분석 프롬프트"""
        ts_results = results.results
        
        prompt = f"""
        시계열 분석 결과를 해석해주세요:
        
        기본 통계:
        {json.dumps(ts_results.get('summary', {}), indent=2)}
        
        자기상관:
        {json.dumps(ts_results.get('autocorrelation', {}), indent=2)}
        
        다음을 포함해서 설명해주세요:
        1. 시계열의 추세와 패턴
        2. 계절성 또는 주기성
        3. 정상성 여부
        4. 예측 모델 추천
        
        {self._get_detail_instruction()}
        """
        
        return prompt
    
    def _get_detail_instruction(self) -> str:
        """AI 상세도 지시문"""
        if self._ai_detail_level == 'simple':
            return "핵심 내용만 간단히 설명해주세요."
        elif self._ai_detail_level == 'detailed':
            return "통계적 근거와 함께 상세히 설명해주세요."
        elif self._ai_detail_level == 'always_detailed':
            return "모든 통계적 세부사항과 이론적 배경을 포함해 매우 상세히 설명해주세요."
        else:  # auto
            return "사용자의 통계 지식 수준을 고려해 적절한 수준으로 설명해주세요."
    
    def _parse_ai_insights(self, ai_response: str) -> List[str]:
        """AI 응답을 인사이트 리스트로 파싱"""
        insights = []
        
        # 번호나 불릿 포인트로 구분된 인사이트 추출
        patterns = [
            r'^\d+\.\s*(.+)$',  # 1. insight
            r'^[-•]\s*(.+)$',   # - insight or • insight
            r'^\*\s*(.+)$',     # * insight
        ]
        
        lines = ai_response.strip().split('\n')
        current_insight = []
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_insight:
                    insights.append(' '.join(current_insight))
                    current_insight = []
                continue
            
            # 새로운 인사이트 시작 확인
            is_new_insight = False
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    if current_insight:
                        insights.append(' '.join(current_insight))
                    current_insight = [match.group(1)]
                    is_new_insight = True
                    break
            
            if not is_new_insight and current_insight:
                current_insight.append(line)
            elif not is_new_insight and not insights:
                # 첫 번째 인사이트
                current_insight = [line]
        
        if current_insight:
            insights.append(' '.join(current_insight))
        
        # 빈 인사이트 제거
        insights = [i.strip() for i in insights if i.strip()]
        
        return insights[:10]  # 최대 10개


# === Optimization Engine ===

class OptimizationEngine:
    """최적화 엔진 (통합 버전)"""
    
    def __init__(self, api_manager: Optional[APIManager] = None):
        self.api_manager = api_manager
        self._ai_detail_level = 'auto'
    
    def optimize(self, 
                optimization_type: OptimizationType,
                objective_function: Optional[Callable] = None,
                constraints: Optional[List[Dict]] = None,
                bounds: Optional[List[Tuple]] = None,
                **kwargs) -> OptimizationResult:
        """최적화 수행"""
        
        if optimization_type == OptimizationType.SINGLE:
            return self._single_objective_optimization(
                objective_function, constraints, bounds, **kwargs
            )
        elif optimization_type == OptimizationType.MULTI:
            return self._multi_objective_optimization(
                objective_function, constraints, bounds, **kwargs
            )
        elif optimization_type == OptimizationType.BAYESIAN:
            return self._bayesian_optimization(
                objective_function, bounds, **kwargs
            )
        elif optimization_type == OptimizationType.ROBUST:
            return self._robust_optimization(
                objective_function, constraints, bounds, **kwargs
            )
        else:
            raise ValueError(f"지원하지 않는 최적화 유형: {optimization_type}")
    
    def _single_objective_optimization(self,
                                     objective_function: Callable,
                                     constraints: Optional[List[Dict]],
                                     bounds: Optional[List[Tuple]],
                                     **kwargs) -> OptimizationResult:
        """단일 목적 최적화"""
        method = kwargs.get('method', 'SLSQP')
        x0 = kwargs.get('initial_guess')
        
        if x0 is None and bounds:
            # 초기값 생성
            x0 = [(b[0] + b[1]) / 2 for b in bounds]
        
        # 제약조건 처리
        scipy_constraints = []
        if constraints:
            for constraint in constraints:
                if 'fun' in constraint:
                    scipy_constraints.append(constraint)
                elif 'expression' in constraint:
                    # 문자열 표현식을 함수로 변환
                    def make_constraint_func(expr):
                        def constraint_func(x):
                            local_vars = {f'x{i}': x[i] for i in range(len(x))}
                            return eval(expr, {"__builtins__": {}}, local_vars)
                        return constraint_func
                    
                    scipy_constraints.append({
                        'type': constraint.get('type', 'ineq'),
                        'fun': make_constraint_func(constraint['expression'])
                    })
        
        # 최적화 수행
        if method in ['differential_evolution', 'shgo', 'dual_annealing']:
            # 전역 최적화
            if method == 'differential_evolution':
                result = differential_evolution(
                    objective_function,
                    bounds,
                    constraints=scipy_constraints,
                    seed=kwargs.get('random_state', RANDOM_STATE),
                    maxiter=kwargs.get('max_iterations', MAX_ITERATIONS),
                    tol=kwargs.get('tolerance', CONVERGENCE_TOL)
                )
            elif method == 'shgo':
                result = shgo(
                    objective_function,
                    bounds,
                    constraints=scipy_constraints,
                    options={'maxiter': kwargs.get('max_iterations', MAX_ITERATIONS)}
                )
            elif method == 'dual_annealing':
                result = dual_annealing(
                    objective_function,
                    bounds,
                    maxiter=kwargs.get('max_iterations', MAX_ITERATIONS)
                )
        else:
            # 국소 최적화
            result = minimize(
                objective_function,
                x0,
                method=method,
                bounds=bounds,
                constraints=scipy_constraints,
                options={
                    'maxiter': kwargs.get('max_iterations', MAX_ITERATIONS),
                    'ftol': kwargs.get('tolerance', CONVERGENCE_TOL)
                }
            )
        
        # 민감도 분석
        sensitivity = None
        if kwargs.get('sensitivity_analysis', False):
            sensitivity = self._sensitivity_analysis(
                objective_function,
                result.x,
                bounds
            )
        
        # 신뢰구간 계산
        confidence_region = None
        if kwargs.get('confidence_region', False):
            confidence_region = self._calculate_confidence_region(
                objective_function,
                result.x,
                bounds
            )
        
        # 결과 정리
        optimal_values = {f'x{i}': val for i, val in enumerate(result.x)}
        
        return OptimizationResult(
            optimal_values=optimal_values,
            optimal_responses={'objective': result.fun},
            convergence_history=[result.fun],
            sensitivity=sensitivity,
            confidence_region=confidence_region,
            constraints_satisfied=all(c['fun'](result.x) >= 0 for c in scipy_constraints if c['type'] == 'ineq')
        )
    
    def _multi_objective_optimization(self,
                                    objective_functions: Union[Callable, List[Callable]],
                                    constraints: Optional[List[Dict]],
                                    bounds: Optional[List[Tuple]],
                                    **kwargs) -> OptimizationResult:
        """다중 목적 최적화"""
        if isinstance(objective_functions, Callable):
            # 단일 함수가 여러 목적값을 반환하는 경우
            objectives_list = objective_functions
            n_objectives = kwargs.get('n_objectives', 2)
        else:
            objectives_list = objective_functions
            n_objectives = len(objectives_list)
        
        method = kwargs.get('method', 'weighted_sum')
        
        if method == 'weighted_sum':
            # 가중합 방법
            weights = kwargs.get('weights', [1.0] * n_objectives)
            
            if isinstance(objectives_list, list):
                def combined_objective(x):
                    return sum(w * obj(x) for w, obj in zip(weights, objectives_list))
            else:
                def combined_objective(x):
                    obj_values = objectives_list(x)
                    return sum(w * v for w, v in zip(weights, obj_values))
            
            # 단일 목적 최적화로 변환
            result = self._single_objective_optimization(
                combined_objective, constraints, bounds, **kwargs
            )
            
            return result
        
        elif method == 'pareto' and PYMOO_AVAILABLE:
            # Pareto 최적화
            return self._pareto_optimization(objectives_list, constraints, bounds, n_objectives, **kwargs)
        
        else:
            raise ValueError(f"지원하지 않는 다중 목적 최적화 방법: {method}")
    
    def _bayesian_optimization(self,
                              objective_function: Callable,
                              bounds: List[Tuple],
                              **kwargs) -> OptimizationResult:
        """베이지안 최적화"""
        if SKOPT_AVAILABLE:
            # scikit-optimize 사용
            dimensions = [Real(low, high) for low, high in bounds]
            
            result = gp_minimize(
                func=objective_function,
                dimensions=dimensions,
                n_calls=kwargs.get('n_calls', 50),
                n_initial_points=kwargs.get('n_initial_points', 10),
                acq_func=kwargs.get('acquisition_function', 'EI'),
                random_state=kwargs.get('random_state', RANDOM_STATE)
            )
            
            optimal_values = {f'x{i}': val for i, val in enumerate(result.x)}
            
            return OptimizationResult(
                optimal_values=optimal_values,
                optimal_responses={'objective': result.fun},
                convergence_history=result.func_vals.tolist()
            )
        else:
            # 대체 구현: Gaussian Process 기반 간단한 베이지안 최적화
            return self._simple_bayesian_optimization(
                objective_function, bounds, **kwargs
            )
    
    def _simple_bayesian_optimization(self,
                                    objective_function: Callable,
                                    bounds: List[Tuple],
                                    **kwargs) -> OptimizationResult:
        """간단한 베이지안 최적화 구현"""
        n_calls = kwargs.get('n_calls', 50)
        n_initial = kwargs.get('n_initial_points', 10)
        
        # 초기 샘플링
        X_sample = []
        y_sample = []
        
        for _ in range(n_initial):
            x = [np.random.uniform(low, high) for low, high in bounds]
            y = objective_function(x)
            X_sample.append(x)
            y_sample.append(y)
        
        # Gaussian Process 모델
        kernel = ConstantKernel() * Matern(length_scale=1.0, nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
        
        # 반복 최적화
        for _ in range(n_calls - n_initial):
            # GP 모델 학습
            gp.fit(X_sample, y_sample)
            
            # 획득 함수 최적화 (Expected Improvement)
            best_y = min(y_sample)
            
            def acquisition(x):
                mu, sigma = gp.predict([x], return_std=True)
                if sigma == 0:
                    return 0
                
                Z = (best_y - mu) / sigma
                ei = sigma * (Z * stats.norm.cdf(Z) + stats.norm.pdf(Z))
                return -ei[0]  # 최대화를 위해 음수
            
            # 다음 샘플 위치 찾기
            result = minimize(
                acquisition,
                x0=[np.random.uniform(low, high) for low, high in bounds],
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            # 새로운 샘플 평가
            next_x = result.x
            next_y = objective_function(next_x)
            
            X_sample.append(next_x.tolist())
            y_sample.append(next_y)
        
        # 최적값 찾기
        best_idx = np.argmin(y_sample)
        optimal_x = X_sample[best_idx]
        optimal_y = y_sample[best_idx]
        
        optimal_values = {f'x{i}': val for i, val in enumerate(optimal_x)}
        
        return OptimizationResult(
            optimal_values=optimal_values,
            optimal_responses={'objective': optimal_y},
            convergence_history=y_sample
        )
    
    def _robust_optimization(self,
                           objective_function: Callable,
                           constraints: Optional[List[Dict]],
                           bounds: Optional[List[Tuple]],
                           **kwargs) -> OptimizationResult:
        """강건 최적화"""
        uncertainty_params = kwargs.get('uncertainty', {})
        n_scenarios = kwargs.get('n_scenarios', 100)
        risk_measure = kwargs.get('risk_measure', 'worst_case')  # worst_case, cvar, mean_std
        
        # 불확실성을 고려한 목적함수
        def robust_objective(x):
            # 몬테카를로 시뮬레이션
            values = []
            
            for _ in range(n_scenarios):
                # 불확실성 추가
                x_perturbed = x.copy()
                for i, (low, high) in enumerate(bounds):
                    if f'x{i}' in uncertainty_params:
                        std = uncertainty_params[f'x{i}']
                        noise = np.random.normal(0, std)
                        x_perturbed[i] = np.clip(x[i] + noise, low, high)
                
                values.append(objective_function(x_perturbed))
            
            values = np.array(values)
            
            # 위험 척도에 따른 목적함수 값
            if risk_measure == 'worst_case':
                return np.max(values)  # 최악의 경우
            elif risk_measure == 'cvar':
                # Conditional Value at Risk (95%)
                percentile = np.percentile(values, 95)
                return np.mean(values[values >= percentile])
            else:  # mean_std
                # 평균 + 표준편차
                return np.mean(values) + kwargs.get('risk_weight', 1.0) * np.std(values)
        
        # 일반 최적화로 해결
        result = self._single_objective_optimization(
            robust_objective,
            constraints,
            bounds,
            **kwargs
        )
        
        # 강건성 분석
        optimal_x = list(result.optimal_values.values())
        robustness_analysis = self._analyze_robustness(
            objective_function,
            optimal_x,
            bounds,
            uncertainty_params,
            n_scenarios
        )
        
        result.robustness = robustness_analysis
        
        return result
    
    def _pareto_optimization(self, objectives_list, constraints, bounds, n_objectives, **kwargs):
        """Pareto 최적화 (PyMoo 사용)"""
        n_variables = len(bounds) if bounds else kwargs.get('n_variables', 2)
        
        # PyMoo Problem 정의
        class MultiObjectiveProblem(Problem):
            def __init__(self):
                super().__init__(
                    n_var=n_variables,
                    n_obj=n_objectives,
                    n_constr=len(constraints) if constraints else 0,
                    xl=[b[0] for b in bounds],
                    xu=[b[1] for b in bounds]
                )
            
            def _evaluate(self, x, out, *args, **kwargs):
                # 목적함수 평가
                f = np.zeros((x.shape[0], n_objectives))
                
                if isinstance(objectives_list, list):
                    for i in range(x.shape[0]):
                        for j, obj_func in enumerate(objectives_list):
                            f[i, j] = obj_func(x[i])
                else:
                    for i in range(x.shape[0]):
                        f[i] = objectives_list(x[i])
                
                out["F"] = f
                
                # 제약조건 평가
                if constraints:
                    g = np.zeros((x.shape[0], len(constraints)))
                    for i in range(x.shape[0]):
                        for j, constraint in enumerate(constraints):
                            if constraint['type'] == 'ineq':
                                g[i, j] = -constraint['fun'](x[i])
                            else:
                                g[i, j] = abs(constraint['fun'](x[i]))
                    out["G"] = g
        
        problem = MultiObjectiveProblem()
        
        # 알고리즘 선택
        algorithm_name = kwargs.get('algorithm', 'NSGA2')
        pop_size = kwargs.get('pop_size', 100)
        n_gen = kwargs.get('n_gen', 100)
        
        if algorithm_name == 'NSGA2':
            algorithm = NSGA2(pop_size=pop_size)
        elif algorithm_name == 'NSGA3':
            from pymoo.factory import get_reference_directions
            ref_dirs = get_reference_directions("das-dennis", n_objectives, n_partitions=12)
            algorithm = NSGA3(ref_dirs=ref_dirs, pop_size=pop_size)
        
        # 최적화 실행
        res = pymoo_minimize(
            problem,
            algorithm,
            termination=get_termination("n_gen", n_gen),
            seed=kwargs.get('seed', RANDOM_STATE),
            verbose=kwargs.get('verbose', False)
        )
        
        # Pareto front 추출
        pareto_front = pd.DataFrame(
            res.F,
            columns=[f'f{i+1}' for i in range(n_objectives)]
        )
        
        pareto_solutions = pd.DataFrame(
            res.X,
            columns=[f'x{i+1}' for i in range(n_variables)]
        )
        
        pareto_front = pd.concat([pareto_solutions, pareto_front], axis=1)
        
        # 대표 솔루션 선택 (중심점에 가장 가까운 해)
        ideal_point = pareto_front[[f'f{i+1}' for i in range(n_objectives)]].min()
        distances = np.sqrt(((pareto_front[[f'f{i+1}' for i in range(n_objectives)]] - ideal_point) ** 2).sum(axis=1))
        best_idx = distances.argmin()
        
        optimal_values = {f'x{i}': res.X[best_idx, i] for i in range(n_variables)}
        optimal_responses = {f'f{i+1}': res.F[best_idx, i] for i in range(n_objectives)}
        
        return OptimizationResult(
            optimal_values=optimal_values,
            optimal_responses=optimal_responses,
            pareto_front=pareto_front
        )
    
    def _sensitivity_analysis(self,
                            objective_function: Callable,
                            optimal_point: np.ndarray,
                            bounds: List[Tuple],
                            delta: float = 0.01) -> Dict[str, float]:
        """민감도 분석"""
        n_vars = len(optimal_point)
        sensitivity = {}
        
        base_value = objective_function(optimal_point)
        
        for i in range(n_vars):
            # 작은 변화량
            range_i = bounds[i][1] - bounds[i][0]
            delta_i = range_i * delta
            
            # 양의 방향
            x_plus = optimal_point.copy()
            x_plus[i] = min(optimal_point[i] + delta_i, bounds[i][1])
            value_plus = objective_function(x_plus)
            
            # 음의 방향
            x_minus = optimal_point.copy()
            x_minus[i] = max(optimal_point[i] - delta_i, bounds[i][0])
            value_minus = objective_function(x_minus)
            
            # 민감도 계산
            actual_delta = x_plus[i] - x_minus[i]
            if actual_delta > 0:
                sensitivity[f'x{i}'] = abs((value_plus - value_minus) / actual_delta)
            else:
                sensitivity[f'x{i}'] = 0
        
        # 정규화
        total_sensitivity = sum(sensitivity.values())
        if total_sensitivity > 0:
            sensitivity = {k: v/total_sensitivity * 100 for k, v in sensitivity.items()}
        
        return sensitivity
    
    def _calculate_confidence_region(self,
                                   objective_function: Callable,
                                   optimal_point: np.ndarray,
                                   bounds: List[Tuple],
                                   confidence_level: float = 0.95) -> Dict:
        """신뢰구간 계산"""
        confidence_region = {}
        
        # 간단한 구현 - 실제로는 Hessian 행렬 기반 계산 필요
        for i, (low, high) in enumerate(bounds):
            # 표준편차 추정
            std_estimate = (high - low) * 0.05
            
            # 신뢰구간
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            margin = z_score * std_estimate
            
            confidence_region[f'x{i}'] = {
                'lower': max(optimal_point[i] - margin, low),
                'upper': min(optimal_point[i] + margin, high),
                'point_estimate': optimal_point[i]
            }
        
        return confidence_region
    
    def _analyze_robustness(self,
                          objective_function: Callable,
                          optimal_point: List[float],
                          bounds: List[Tuple],
                          uncertainty_params: Dict,
                          n_scenarios: int) -> Dict:
        """강건성 분석"""
        results = []
        
        for _ in range(n_scenarios):
            # 불확실성 시나리오 생성
            x_scenario = optimal_point.copy()
            
            for i, (low, high) in enumerate(bounds):
                if f'x{i}' in uncertainty_params:
                    std = uncertainty_params[f'x{i}']
                    noise = np.random.normal(0, std)
                    x_scenario[i] = np.clip(optimal_point[i] + noise, low, high)
            
            results.append(objective_function(x_scenario))
        
        results = np.array(results)
        
        return {
            'mean': float(np.mean(results)),
            'std': float(np.std(results)),
            'min': float(np.min(results)),
            'max': float(np.max(results)),
            'cv': float(np.std(results) / np.mean(results)) if np.mean(results) != 0 else 0,
            'percentiles': {
                '5%': float(np.percentile(results, 5)),
                '25%': float(np.percentile(results, 25)),
                '50%': float(np.percentile(results, 50)),
                '75%': float(np.percentile(results, 75)),
                '95%': float(np.percentile(results, 95))
            },
            'reliability': float(np.mean(results <= optimal_point[0] * 1.1))  # 10% 마진 내 비율
        }


# === Main DataProcessor Class ===

class DataProcessor:
    """데이터 처리 메인 엔진 (통합 버전)"""
    
    def __init__(self, api_manager: Optional[APIManager] = None):
        self.api_manager = api_manager
        self.validator = DataValidator()
        self.transformer = TransformationEngine()
        self.design_engine = ExperimentDesignEngine()
        self.analyzer = StatisticalAnalyzer(api_manager)
        self.optimizer = OptimizationEngine(api_manager)
        
        # 처리 이력
        self.processing_history = []
        
        # AI 상세도
        self._ai_detail_level = 'auto'
        
        logger.info("DataProcessor initialized successfully")
    
    def set_ai_detail_level(self, level: str):
        """AI 설명 상세도 설정"""
        valid_levels = ['auto', 'simple', 'detailed', 'always_detailed']
        if level in valid_levels:
            self._ai_detail_level = level
            self.analyzer.set_ai_detail_level(level)
            self.optimizer._ai_detail_level = level
    
    # === 데이터 전처리 ===
    
    def preprocess_data(self,
                       data: pd.DataFrame,
                       missing_method: str = 'drop',
                       outlier_method: str = 'keep',
                       transformations: Optional[List[str]] = None,
                       selected_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """데이터 전처리 파이프라인"""
        processed_data = data.copy()
        
        # 컬럼 선택
        if selected_columns:
            processed_data = processed_data[selected_columns]
        
        # 결측치 처리
        if missing_method == 'drop':
            processed_data = processed_data.dropna()
        elif missing_method == 'mean':
            numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
            processed_data[numeric_cols] = processed_data[numeric_cols].fillna(
                processed_data[numeric_cols].mean()
            )
        elif missing_method == 'median':
            numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
            processed_data[numeric_cols] = processed_data[numeric_cols].fillna(
                processed_data[numeric_cols].median()
            )
        elif missing_method == 'interpolate':
            numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
            processed_data[numeric_cols] = processed_data[numeric_cols].interpolate()
        elif missing_method == 'forward_fill':
            processed_data = processed_data.fillna(method='ffill')
        elif missing_method == 'backward_fill':
            processed_data = processed_data.fillna(method='bfill')
        
        # 이상치 처리
        if outlier_method == 'iqr':
            numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                Q1 = processed_data[col].quantile(0.25)
                Q3 = processed_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                processed_data[col] = processed_data[col].clip(lower_bound, upper_bound)
                
        elif outlier_method == 'zscore':
            numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                z_scores = np.abs(stats.zscore(processed_data[col].dropna()))
                threshold = 3
                mask = z_scores < threshold
                processed_data.loc[processed_data[col].notna(), col] = processed_data.loc[
                    processed_data[col].notna(), col
                ][mask]
                
        elif outlier_method == 'isolation_forest':
            numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                iso_forest = IsolationForest(contamination=0.1, random_state=RANDOM_STATE)
                outlier_mask = iso_forest.fit_predict(processed_data[numeric_cols].fillna(0))
                processed_data = processed_data[outlier_mask == 1]
        
        # 변환 적용
        if transformations:
            numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
            
            for transform in transformations:
                if transform == 'auto':
                    processed_data, _ = self.transformer.auto_transform(processed_data)
                elif transform in ['정규화', 'normalize']:
                    scaler = MinMaxScaler()
                    processed_data[numeric_cols] = scaler.fit_transform(processed_data[numeric_cols])
                elif transform in ['표준화', 'standardize']:
                    scaler = StandardScaler()
                    processed_data[numeric_cols] = scaler.fit_transform(processed_data[numeric_cols])
                elif transform in ['로그 변환', 'log']:
                    for col in numeric_cols:
                        processed_data[col] = self.transformer.apply_transform(
                            processed_data[col], TransformMethod.LOG
                        )
                elif transform in ['제곱근 변환', 'sqrt']:
                    for col in numeric_cols:
                        processed_data[col] = self.transformer.apply_transform(
                            processed_data[col], TransformMethod.SQRT
                        )
                elif transform in ['Box-Cox', 'box_cox']:
                    for col in numeric_cols:
                        processed_data[col] = self.transformer.apply_transform(
                            processed_data[col], TransformMethod.BOX_COX
                        )
        
        # 처리 기록
        self._record_processing({
            'timestamp': datetime.now(),
            'operation': 'preprocess',
            'parameters': {
                'missing_method': missing_method,
                'outlier_method': outlier_method,
                'transformations': transformations,
                'selected_columns': selected_columns
            },
            'input_shape': data.shape,
            'output_shape': processed_data.shape
        })
        
        return processed_data
    
    # === 실험 설계 ===
    
    def create_experiment_design(self,
                               factors: List[Dict],
                               design_type: str = 'factorial',
                               **kwargs) -> pd.DataFrame:
        """실험 설계 생성"""
        # Factor 객체로 변환
        factor_objects = []
        for f in factors:
            factor = Factor(
                name=f['name'],
                type=f.get('type', 'continuous'),
                low=f.get('low'),
                high=f.get('high'),
                levels=f.get('levels'),
                units=f.get('units')
            )
            factor_objects.append(factor)
        
        # 설계 유형 변환
        design_type_enum = DesignType(design_type)
        
        # 설계 생성
        design = self.design_engine.create_design(
            factor_objects,
            design_type_enum,
            **kwargs
        )
        
        # 처리 기록
        self._record_processing({
            'timestamp': datetime.now(),
            'operation': 'create_design',
            'parameters': {
                'factors': factors,
                'design_type': design_type,
                'kwargs': kwargs
            },
            'output_shape': design.shape
        })
        
        return design
    
    # === 통계 분석 ===
    
    def analyze_data(self,
                    data: pd.DataFrame,
                    analysis_type: str,
                    **kwargs) -> Dict[str, Any]:
        """데이터 분석"""
        result = self.analyzer.analyze(data, analysis_type, **kwargs)
        
        # 처리 기록
        self._record_processing({
            'timestamp': datetime.now(),
            'operation': 'analyze',
            'parameters': {
                'analysis_type': analysis_type,
                'kwargs': kwargs
            },
            'input_shape': data.shape
        })
        
        # 간단한 형식으로 변환
        return {
            'analysis_type': result.analysis_type,
            'results': result.results,
            'statistics': result.statistics,
            'visualizations': result.visualizations,
            'ai_insights': result.ai_insights,
            'interpretation': result.interpretation
        }
    
    # === 최적화 ===
    
    def optimize(self,
                optimization_type: str,
                objective_func: Optional[Callable] = None,
                bounds: Optional[List[Tuple]] = None,
                **kwargs) -> Dict[str, Any]:
        """최적화 수행"""
        # OptimizationType으로 변환
        opt_type = OptimizationType(optimization_type)
        
        result = self.optimizer.optimize(
            opt_type,
            objective_func,
            bounds=bounds,
            **kwargs
        )
        
        # 처리 기록
        self._record_processing({
            'timestamp': datetime.now(),
            'operation': 'optimize',
            'parameters': {
                'optimization_type': optimization_type,
                'bounds': bounds,
                'kwargs': kwargs
            }
        })
        
        # 간단한 형식으로 변환
        return {
            'optimal_values': result.optimal_values,
            'optimal_responses': result.optimal_responses,
            'sensitivity': result.sensitivity,
            'robustness': result.robustness,
            'confidence_region': result.confidence_region,
            'pareto_front': result.pareto_front.to_dict() if result.pareto_front is not None else None
        }
    
    # === 통합 파이프라인 ===
    
    def run_complete_analysis(self,
                            raw_data: Optional[pd.DataFrame] = None,
                            factors: Optional[List[Dict]] = None,
                            responses: Optional[List[Dict]] = None,
                            design_type: str = 'factorial',
                            optimization_goal: str = 'maximize') -> Dict[str, Any]:
        """완전한 분석 파이프라인 실행"""
        results = {
            'design': None,
            'preprocessed_data': None,
            'validation': None,
            'analysis': {},
            'optimization': None,
            'recommendations': []
        }
        
        try:
            # 1. 데이터 검증
            if raw_data is not None and not raw_data.empty:
                validation_report = self.validator.validate(raw_data)
                results['validation'] = {
                    'valid': validation_report.valid,
                    'issues': validation_report.issues,
                    'warnings': validation_report.warnings,
                    'statistics': validation_report.statistics,
                    'recommendations': validation_report.recommendations
                }
                
                # 전처리
                processed_data = self.preprocess_data(
                    raw_data,
                    missing_method='mean',
                    outlier_method='iqr'
                )
                results['preprocessed_data'] = processed_data
                
            else:
                # 2. 실험 설계 생성
                if factors:
                    design = self.create_experiment_design(
                        factors,
                        design_type=design_type
                    )
                    results['design'] = design
                    processed_data = design
                else:
                    raise ValueError("데이터 또는 실험 인자가 필요합니다")
            
            # 3. 통계 분석
            if processed_data is not None and not processed_data.empty:
                # 기술통계
                results['analysis']['descriptive'] = self.analyze_data(
                    processed_data,
                    'descriptive'
                )
                
                # 상관분석
                results['analysis']['correlation'] = self.analyze_data(
                    processed_data,
                    'correlation'
                )
                
                # ANOVA (범주형 변수가 있는 경우)
                categorical_cols = processed_data.select_dtypes(
                    include=['object', 'category']
                ).columns.tolist()
                
                if categorical_cols and responses and len(responses) > 0:
                    try:
                        results['analysis']['anova'] = self.analyze_data(
                            processed_data,
                            'anova',
                            factor_cols=categorical_cols[:2],  # 최대 2개
                            response_col=responses[0]['name']
                        )
                    except Exception as e:
                        logger.warning(f"ANOVA 분석 실패: {e}")
                
                # 회귀분석
                if factors and responses and len(responses) > 0:
                    numeric_factors = [f['name'] for f in factors if f.get('type', 'continuous') == 'continuous']
                    if numeric_factors and responses[0]['name'] in processed_data.columns:
                        try:
                            results['analysis']['regression'] = self.analyze_data(
                                processed_data,
                                'regression',
                                predictors=numeric_factors[:3],  # 최대 3개 인자
                                response=responses[0]['name']
                            )
                            
                            # RSM 분석 (인자가 2개 이상인 경우)
                            if len(numeric_factors) >= 2:
                                results['analysis']['rsm'] = self.analyze_data(
                                    processed_data,
                                    'rsm',
                                    factors=numeric_factors[:3],
                                    response=responses[0]['name']
                                )
                        except Exception as e:
                            logger.warning(f"회귀/RSM 분석 실패: {e}")
                
                # PCA (변수가 많은 경우)
                numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 5:
                    try:
                        results['analysis']['pca'] = self.analyze_data(
                            processed_data,
                            'pca',
                            variables=numeric_cols.tolist()
                        )
                    except Exception as e:
                        logger.warning(f"PCA 분석 실패: {e}")
            
            # 4. 최적화
            if factors and responses and 'regression' in results['analysis']:
                try:
                    # 회귀모델 기반 최적화
                    reg_model = results['analysis']['regression']['results'].get('model')
                    
                    if reg_model:
                        numeric_factors = [f['name'] for f in factors if f.get('type', 'continuous') == 'continuous']
                        
                        def objective(x):
                            # 회귀모델을 사용한 예측
                            x_df = pd.DataFrame([x], columns=numeric_factors[:len(x)])
                            
                            # 다항식 특징 추가 (회귀 분석에서 사용한 경우)
                            poly_degree = results['analysis']['regression']['statistics'].get('polynomial_degree', 1)
                            if poly_degree > 1:
                                poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
                                x_poly = poly.fit_transform(x_df)
                                x_df = pd.DataFrame(x_poly, columns=poly.get_feature_names_out(numeric_factors[:len(x)]))
                            
                            x_with_const = sm.add_constant(x_df)
                            
                            try:
                                prediction = reg_model.predict(x_with_const)[0]
                                return -prediction if optimization_goal == 'maximize' else prediction
                            except:
                                return float('inf')
                        
                        bounds = [(f['low'], f['high']) for f in factors if f.get('type', 'continuous') == 'continuous'][:3]
                        
                        if bounds:
                            results['optimization'] = self.optimize(
                                'single_objective',
                                objective_func=objective,
                                bounds=bounds,
                                method='differential_evolution',
                                sensitivity_analysis=True,
                                confidence_region=True
                            )
                except Exception as e:
                    logger.warning(f"최적화 실패: {e}")
            
            # 5. 권장사항 생성
            results['recommendations'] = self._generate_recommendations(results)
            
        except Exception as e:
            logger.error(f"분석 파이프라인 오류: {e}")
            results['error'] = str(e)
        
        return results
    
    # === 프로토콜 처리 (v9.1 기능) ===
    
    def process_extracted_protocol(self, raw_protocol: Dict) -> Dict:
        """추출된 프로토콜 데이터 정제 및 표준화"""
        
        processed = {
            'materials': self._standardize_materials(raw_protocol.get('materials', [])),
            'conditions': self._normalize_conditions(raw_protocol.get('conditions', {})),
            'procedure': self._structure_steps(raw_protocol.get('procedure', [])),
            'metadata': {
                'extraction_date': datetime.now(),
                'confidence_score': self._calculate_confidence(raw_protocol),
                'completeness': self._assess_completeness(raw_protocol),
                'processing_version': '1.0'
            }
        }
        
        # AI를 통한 보완
        if self.api_manager and processed['metadata']['completeness'] < 0.8:
            processed = self._enhance_protocol_with_ai(processed)
        
        return processed
    
    def merge_protocols(self, protocols: List[Dict]) -> Dict:
        """여러 소스에서 추출된 프로토콜 병합"""
        if not protocols:
            return {}
        
        if len(protocols) == 1:
            return protocols[0]
        
        # 기준 프로토콜 선택 (가장 완성도 높은 것)
        base_protocol = max(protocols, 
                           key=lambda p: p.get('metadata', {}).get('completeness', 0))
        
        merged = copy.deepcopy(base_protocol)
        
        # 각 섹션별 병합
        for protocol in protocols:
            if protocol is base_protocol:
                continue
            
            # 재료 병합
            merged['materials'] = self._merge_materials(
                merged.get('materials', {}), 
                protocol.get('materials', {})
            )
            
            # 조건 병합
            merged['conditions'] = self._merge_conditions(
                merged.get('conditions', {}),
                protocol.get('conditions', {})
            )
            
            # 절차 병합
            merged['procedure'] = self._merge_procedures(
                merged.get('procedure', []),
                protocol.get('procedure', [])
            )
        
        # 병합 후 재평가
        merged['metadata']['completeness'] = self._assess_completeness(merged)
        merged['metadata']['merge_count'] = len(protocols)
        
        return merged
    
    # === 시각화 데이터 준비 ===
    
    def prepare_visualization_data(self,
                                 data: pd.DataFrame,
                                 viz_type: str,
                                 **kwargs) -> Dict[str, Any]:
        """시각화를 위한 데이터 준비"""
        viz_functions = {
            'scatter': self._prepare_scatter,
            'scatter_matrix': self._prepare_scatter_matrix,
            'line': self._prepare_line,
            'bar': self._prepare_bar,
            'box': self._prepare_box_plot,
            'heatmap': self._prepare_heatmap,
            'surface_3d': self._prepare_surface_3d,
            'contour': self._prepare_contour,
            'pareto': self._prepare_pareto_chart,
            'parallel_coordinates': self._prepare_parallel_coordinates
        }
        
        if viz_type not in viz_functions:
            raise ValueError(f"지원하지 않는 시각화 유형: {viz_type}")
        
        return viz_functions[viz_type](data, **kwargs)
    
    def _prepare_scatter(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """산점도 데이터 준비"""
        x_col = kwargs.get('x')
        y_col = kwargs.get('y')
        color_col = kwargs.get('color')
        size_col = kwargs.get('size')
        
        if not x_col or not y_col:
            raise ValueError("x와 y 컬럼을 지정해야 합니다")
        
        plot_data = {
            'x': data[x_col].tolist(),
            'y': data[y_col].tolist(),
            'x_label': x_col,
            'y_label': y_col,
            'type': 'scatter'
        }
        
        if color_col and color_col in data.columns:
            plot_data['color'] = data[color_col].tolist()
            plot_data['color_label'] = color_col
        
        if size_col and size_col in data.columns:
            plot_data['size'] = data[size_col].tolist()
            plot_data['size_label'] = size_col
        
        return plot_data
    
    def _prepare_scatter_matrix(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """산점도 행렬 데이터 준비"""
        numeric_cols = kwargs.get('columns', data.select_dtypes(include=[np.number]).columns.tolist())
        
        plot_data = []
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i != j:
                    plot_data.append({
                        'x': data[col1].tolist(),
                        'y': data[col2].tolist(),
                        'x_label': col1,
                        'y_label': col2,
                        'row': i,
                        'col': j
                    })
        
        return {
            'type': 'scatter_matrix',
            'data': plot_data,
            'n_vars': len(numeric_cols),
            'variables': numeric_cols
        }
    
    def _prepare_line(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """선 그래프 데이터 준비"""
        x_col = kwargs.get('x')
        y_cols = kwargs.get('y', [])
        
        if not x_col:
            x_data = list(range(len(data)))
            x_label = 'Index'
        else:
            x_data = data[x_col].tolist()
            x_label = x_col
        
        if not y_cols:
            y_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if x_col in y_cols:
                y_cols.remove(x_col)
        
        lines = []
        for y_col in y_cols:
            if y_col in data.columns:
                lines.append({
                    'name': y_col,
                    'y': data[y_col].tolist()
                })
        
        return {
            'type': 'line',
            'x': x_data,
            'x_label': x_label,
            'lines': lines
        }
    
    def _prepare_bar(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """막대 그래프 데이터 준비"""
        category_col = kwargs.get('category')
        value_col = kwargs.get('value')
        agg_func = kwargs.get('agg_func', 'mean')
        
        if category_col and value_col:
            grouped = data.groupby(category_col)[value_col].agg(agg_func)
            
            return {
                'type': 'bar',
                'categories': grouped.index.tolist(),
                'values': grouped.values.tolist(),
                'category_label': category_col,
                'value_label': f"{agg_func}({value_col})"
            }
        else:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            means = data[numeric_cols].mean()
            
            return {
                'type': 'bar',
                'categories': means.index.tolist(),
                'values': means.values.tolist(),
                'category_label': 'Variables',
                'value_label': 'Mean Value'
            }
    
    def _prepare_box_plot(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """박스플롯 데이터 준비"""
        group_col = kwargs.get('group')
        value_col = kwargs.get('value')
        
        plot_data = []
        
        if group_col and value_col:
            for group in data[group_col].unique():
                group_data = data[data[group_col] == group][value_col].dropna()
                plot_data.append({
                    'name': str(group),
                    'y': group_data.tolist(),
                    'boxpoints': 'outliers'
                })
        else:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                plot_data.append({
                    'name': col,
                    'y': data[col].dropna().tolist(),
                    'boxpoints': 'outliers'
                })
        
        return {
            'type': 'box',
            'data': plot_data
        }
    
    def _prepare_heatmap(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """히트맵 데이터 준비"""
        method = kwargs.get('method', 'correlation')
        
        if method == 'correlation':
            numeric_data = data.select_dtypes(include=[np.number])
            matrix = numeric_data.corr()
        else:
            matrix = data
        
        return {
            'type': 'heatmap',
            'z': matrix.values.tolist(),
            'x': matrix.columns.tolist(),
            'y': matrix.index.tolist(),
            'colorscale': kwargs.get('colorscale', 'RdBu')
        }
    
    def _prepare_surface_3d(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """3D 표면 플롯 데이터 준비"""
        x_col = kwargs.get('x')
        y_col = kwargs.get('y')
        z_col = kwargs.get('z')
        
        if not all([x_col, y_col, z_col]):
            raise ValueError("x, y, z 컬럼이 필요합니다")
        
        # 격자 생성
        x_unique = np.sort(data[x_col].unique())
        y_unique = np.sort(data[y_col].unique())
        
        X, Y = np.meshgrid(x_unique, y_unique)
        Z = np.zeros_like(X)
        
        # Z 값 채우기
        for i, x in enumerate(x_unique):
            for j, y in enumerate(y_unique):
                mask = (data[x_col] == x) & (data[y_col] == y)
                if mask.any():
                    Z[j, i] = data.loc[mask, z_col].iloc[0]
                else:
                    # 보간
                    points = data[[x_col, y_col]].values
                    values = data[z_col].values
                    Z[j, i] = griddata(points, values, (x, y), method='linear')
        
        return {
            'type': 'surface',
            'x': x_unique.tolist(),
            'y': y_unique.tolist(),
            'z': Z.tolist(),
            'x_label': x_col,
            'y_label': y_col,
            'z_label': z_col
        }
    
    def _prepare_contour(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """등고선 플롯 데이터 준비"""
        surface_data = self._prepare_surface_3d(data, **kwargs)
        
        return {
            'type': 'contour',
            'x': surface_data['x'],
            'y': surface_data['y'],
            'z': surface_data['z'],
            'x_label': surface_data['x_label'],
            'y_label': surface_data['y_label'],
            'colorscale': kwargs.get('colorscale', 'Viridis')
        }
    
    def _prepare_pareto_chart(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """파레토 차트 데이터 준비"""
        category_col = kwargs.get('category')
        value_col = kwargs.get('value')
        
        if not category_col or not value_col:
            raise ValueError("category와 value 컬럼이 필요합니다")
        
        # 집계 및 정렬
        grouped = data.groupby(category_col)[value_col].sum().sort_values(ascending=False)
        
        # 누적 백분율
        cumsum = grouped.cumsum()
        total = grouped.sum()
        cumulative_percent = (cumsum / total * 100).round(2)
        
        return {
            'type': 'pareto',
            'categories': grouped.index.tolist(),
            'values': grouped.values.tolist(),
            'cumulative_percent': cumulative_percent.tolist(),
            'threshold_80': 80
        }
    
    def _prepare_parallel_coordinates(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """평행 좌표 플롯 데이터 준비"""
        numeric_cols = kwargs.get('columns', data.select_dtypes(include=[np.number]).columns.tolist())
        
        # 정규화
        normalized_data = data[numeric_cols].copy()
        for col in numeric_cols:
            col_min = normalized_data[col].min()
            col_max = normalized_data[col].max()
            if col_max > col_min:
                normalized_data[col] = (normalized_data[col] - col_min) / (col_max - col_min)
        
        dimensions = []
        for col in numeric_cols:
            dimensions.append({
                'label': col,
                'values': normalized_data[col].tolist(),
                'range': [data[col].min(), data[col].max()]
            })
        
        return {
            'type': 'parallel_coordinates',
            'dimensions': dimensions
        }
    
    # === 유틸리티 메서드 ===
    
    def _record_processing(self, record: Dict):
        """처리 기록 저장"""
        self.processing_history.append(record)
        
        # 최대 100개까지만 유지
        if len(self.processing_history) > 100:
            self.processing_history.pop(0)
    
    def _generate_recommendations(self, analysis_results: Dict) -> List[str]:
        """분석 결과 기반 권장사항 생성"""
        recommendations = []
        
        # 검증 결과 기반
        if analysis_results.get('validation'):
            validation = analysis_results['validation']
            if not validation['valid']:
                recommendations.append("데이터 품질 개선이 필요합니다.")
            
            for issue in validation.get('issues', []):
                if '결측치' in issue:
                    recommendations.append("결측치 처리 전략을 재검토하세요.")
                if '이상치' in issue:
                    recommendations.append("이상치 제거 또는 변환을 고려하세요.")
            
            # 검증 권장사항 추가
            recommendations.extend(validation.get('recommendations', []))
        
        # 통계 분석 기반
        if 'descriptive' in analysis_results.get('analysis', {}):
            desc = analysis_results['analysis']['descriptive']
            if 'normality_tests' in desc.get('results', {}):
                non_normal = [k for k, v in desc['results']['normality_tests'].items() 
                             if not v.get('is_normal', True)]
                if non_normal:
                    recommendations.append(f"변환 필요: {', '.join(non_normal[:3])}")
        
        # 상관분석 기반
        if 'correlation' in analysis_results.get('analysis', {}):
            corr = analysis_results['analysis']['correlation']
            high_corr = [f"{c['var1']}-{c['var2']}" 
                        for c in corr.get('results', {}).get('significant_correlations', [])
                        if abs(c.get('correlation', 0)) > 0.8]
            if high_corr:
                recommendations.append(f"다중공선성 주의: {', '.join(high_corr[:3])}")
        
        # 회귀분석 기반
        if 'regression' in analysis_results.get('analysis', {}):
            reg = analysis_results['analysis']['regression']
            r_squared = reg.get('results', {}).get('model_summary', {}).get('r_squared', 0)
            if r_squared < 0.5:
                recommendations.append("모델 설명력이 낮습니다. 추가 변수나 변환을 고려하세요.")
            
            # VIF 확인
            vif_data = reg.get('results', {}).get('vif', {})
            high_vif_vars = [var for var, vif in vif_data.items() if vif > 10]
            if high_vif_vars:
                recommendations.append(f"다중공선성 제거 필요: {', '.join(high_vif_vars)}")
        
        # RSM 분석 기반
        if 'rsm' in analysis_results.get('analysis', {}):
            rsm = analysis_results['analysis']['rsm']
            surface_type = rsm.get('results', {}).get('surface_type', '')
            if surface_type == 'saddle_point':
                recommendations.append("안장점 발견 - 추가 실험으로 최적 영역 탐색 필요")
            
            stationary_point = rsm.get('results', {}).get('stationary_point', {})
            if stationary_point.get('coordinates'):
                coords = stationary_point['coordinates']
                out_of_bounds = any(v < -1 or v > 1 for v in coords.values())
                if out_of_bounds:
                    recommendations.append("정상점이 실험 영역 밖에 있습니다 - 실험 영역 확장 고려")
        
        # 최적화 기반
        if analysis_results.get('optimization'):
            opt = analysis_results['optimization']
            if opt.get('sensitivity'):
                sensitive_vars = sorted(opt['sensitivity'].items(), 
                                      key=lambda x: x[1], reverse=True)[:2]
                if sensitive_vars:
                    recommendations.append(
                        f"민감 변수 정밀 제어: {', '.join([v[0] for v in sensitive_vars])}"
                    )
            
            if opt.get('robustness'):
                cv = opt['robustness'].get('cv', 0)
                if cv > 0.2:
                    recommendations.append("최적 조건의 강건성이 낮습니다 - 불확실성 관리 필요")
        
        # AI 인사이트 추가
        for analysis in analysis_results.get('analysis', {}).values():
            if isinstance(analysis, dict) and 'ai_insights' in analysis:
                recommendations.extend(analysis['ai_insights'][:2])
        
        # 중복 제거 및 제한
        unique_recommendations = []
        for rec in recommendations:
            if rec not in unique_recommendations:
                unique_recommendations.append(rec)
        
        return unique_recommendations[:10]  # 최대 10개
    
    # === 프로토콜 처리 헬퍼 메서드 ===
    
    def _standardize_materials(self, materials: Union[List, Dict]) -> Dict:
        """재료 정보 표준화"""
        standardized = {
            'polymers': [],
            'solvents': [],
            'additives': [],
            'catalysts': [],
            'others': []
        }
        
        if isinstance(materials, list):
            for material in materials:
                category = self._classify_material(material)
                standardized[category].append(self._parse_material_info(material))
        
        elif isinstance(materials, dict):
            for category, items in materials.items():
                if category in standardized:
                    standardized[category] = [self._parse_material_info(item) for item in items]
        
        return standardized
    
    def _normalize_conditions(self, conditions: Dict) -> Dict:
        """실험 조건 정규화"""
        normalized = {}
        
        # 표준 조건 매핑
        condition_map = {
            'temp': ['temperature', 'temp', 'T', '온도'],
            'time': ['time', 'duration', 't', '시간'],
            'pressure': ['pressure', 'P', '압력'],
            'speed': ['speed', 'rpm', 'stirring', '속도', '교반'],
            'concentration': ['concentration', 'conc', 'C', '농도']
        }
        
        for key, value in conditions.items():
            # 표준 키 찾기
            standard_key = key
            for std_key, variations in condition_map.items():
                if key.lower() in [v.lower() for v in variations]:
                    standard_key = std_key
                    break
            
            # 값 파싱
            normalized[standard_key] = self._parse_condition_value(value)
        
        return normalized
    
    def _structure_steps(self, procedure: Union[List, str]) -> List[Dict]:
        """실험 절차 구조화"""
        if isinstance(procedure, str):
            steps = self._split_procedure_text(procedure)
        else:
            steps = procedure
        
        structured = []
        for i, step in enumerate(steps):
            structured.append({
                'step_number': i + 1,
                'description': step if isinstance(step, str) else step.get('description', ''),
                'duration': self._extract_duration(step),
                'temperature': self._extract_temperature(step),
                'critical': self._is_critical_step(step)
            })
        
        return structured
    
    def _calculate_confidence(self, protocol: Dict) -> float:
        """프로토콜 신뢰도 계산"""
        score = 0.0
        weights = {
            'materials': 0.3,
            'conditions': 0.3,
            'procedure': 0.4
        }
        
        # 재료 정보 평가
        if 'materials' in protocol:
            materials = protocol['materials']
            if isinstance(materials, dict):
                total_materials = sum(len(v) for v in materials.values() if isinstance(v, list))
                if total_materials > 0:
                    score += weights['materials'] * min(total_materials / 5, 1.0)
        
        # 조건 정보 평가
        if 'conditions' in protocol:
            n_conditions = len(protocol['conditions'])
            score += weights['conditions'] * min(n_conditions / 4, 1.0)
        
        # 절차 정보 평가
        if 'procedure' in protocol:
            n_steps = len(protocol['procedure'])
            score += weights['procedure'] * min(n_steps / 5, 1.0)
        
        return round(score, 2)
    
    def _assess_completeness(self, protocol: Dict) -> float:
        """프로토콜 완성도 평가"""
        required_elements = {
            'materials': ['polymers', 'solvents'],
            'conditions': ['temp', 'time'],
            'procedure': 3  # 최소 단계 수
        }
        
        completeness = 0.0
        
        # 재료 확인
        if 'materials' in protocol:
            materials = protocol['materials']
            if isinstance(materials, dict):
                for req in required_elements['materials']:
                    if req in materials and materials[req]:
                        completeness += 0.2
        
        # 조건 확인
        if 'conditions' in protocol:
            conditions = protocol['conditions']
            for req in required_elements['conditions']:
                if req in conditions:
                    completeness += 0.2
        
        # 절차 확인
        if 'procedure' in protocol:
            if len(protocol['procedure']) >= required_elements['procedure']:
                completeness += 0.2
        
        return min(completeness, 1.0)
    
    def _classify_material(self, material: Union[str, Dict]) -> str:
        """재료 분류"""
        if isinstance(material, dict):
            material_name = material.get('name', '').lower()
        else:
            material_name = str(material).lower()
        
        # 키워드 기반 분류
        polymer_keywords = ['polymer', 'poly', 'resin', 'plastic', '폴리머', '수지']
        solvent_keywords = ['solvent', 'solution', 'alcohol', 'water', 'acetone', '용매', '용액']
        catalyst_keywords = ['catalyst', 'initiator', 'accelerator', '촉매', '개시제']
        additive_keywords = ['additive', 'stabilizer', 'plasticizer', 'filler', '첨가제', '안정제']
        
        for keyword in polymer_keywords:
            if keyword in material_name:
                return 'polymers'
        
        for keyword in solvent_keywords:
            if keyword in material_name:
                return 'solvents'
        
        for keyword in catalyst_keywords:
            if keyword in material_name:
                return 'catalysts'
        
        for keyword in additive_keywords:
            if keyword in material_name:
                return 'additives'
        
        return 'others'
    
    def _parse_material_info(self, material: Union[str, Dict]) -> Dict:
        """재료 정보 파싱"""
        if isinstance(material, dict):
            return material
        
        # 문자열에서 정보 추출
        info = {'name': str(material)}
        
        # 농도 추출 (예: "10% solution")
        conc_match = re.search(r'(\d+(?:\.\d+)?)\s*%', str(material))
        if conc_match:
            info['concentration'] = float(conc_match.group(1))
        
        # 무게/부피 추출 (예: "5g", "10mL")
        amount_match = re.search(r'(\d+(?:\.\d+)?)\s*(g|mg|kg|mL|L)', str(material))
        if amount_match:
            info['amount'] = float(amount_match.group(1))
            info['unit'] = amount_match.group(2)
        
        return info
    
    def _parse_condition_value(self, value: Union[str, float, Dict]) -> Dict:
        """조건 값 파싱"""
        if isinstance(value, dict):
            return value
        
        result = {}
        
        if isinstance(value, (int, float)):
            result['value'] = float(value)
        else:
            # 문자열에서 숫자와 단위 추출
            match = re.search(r'(\d+(?:\.\d+)?)\s*([°℃CKFfahrenheit|min|h|hr|hours|bar|atm|psi|Pa|rpm]*)', str(value))
            if match:
                result['value'] = float(match.group(1))
                if match.group(2):
                    result['unit'] = match.group(2)
        
        return result
    
    def _split_procedure_text(self, text: str) -> List[str]:
        """절차 텍스트를 단계로 분리"""
        # 번호나 불릿으로 분리
        steps = re.split(r'\n\s*(?:\d+[\.\)]\s*|\*\s*|-\s*|•\s*)', text)
        
        # 빈 단계 제거
        steps = [step.strip() for step in steps if step.strip()]
        
        # 문장 단위로도 분리 시도
        if len(steps) <= 1:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            steps = [s.strip() for s in sentences if s.strip() and len(s) > 20]
        
        return steps
    
    def _extract_duration(self, step: Union[str, Dict]) -> Optional[float]:
        """단계에서 시간 정보 추출"""
        if isinstance(step, dict):
            text = step.get('description', '')
        else:
            text = str(step)
        
        # 시간 패턴 찾기
        time_patterns = [
            (r'(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h|시간)', 60),  # 시간 → 분
            (r'(\d+(?:\.\d+)?)\s*(?:minutes?|mins?|m|분)', 1),  # 분
            (r'(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s|초)', 1/60),  # 초 → 분
        ]
        
        for pattern, multiplier in time_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1)) * multiplier
        
        return None
    
    def _extract_temperature(self, step: Union[str, Dict]) -> Optional[float]:
        """단계에서 온도 정보 추출"""
        if isinstance(step, dict):
            text = step.get('description', '')
        else:
            text = str(step)
        
        # 온도 패턴 찾기
        temp_patterns = [
            r'(\d+(?:\.\d+)?)\s*°C',
            r'(\d+(?:\.\d+)?)\s*℃',
            r'(\d+(?:\.\d+)?)\s*degrees?\s*C',
            r'(\d+(?:\.\d+)?)\s*celsius'
        ]
        
        for pattern in temp_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        
        return None
    
    def _is_critical_step(self, step: Union[str, Dict]) -> bool:
        """중요 단계 여부 판단"""
        if isinstance(step, dict):
            text = step.get('description', '').lower()
        else:
            text = str(step).lower()
        
        # 중요 키워드
        critical_keywords = [
            'critical', 'important', 'crucial', 'essential',
            'must', 'ensure', 'carefully', 'caution',
            'do not', "don't", 'avoid', 'never',
            '중요', '필수', '주의', '반드시', '절대'
        ]
        
        return any(keyword in text for keyword in critical_keywords)
    
    def _enhance_protocol_with_ai(self, protocol: Dict) -> Dict:
        """AI를 통한 프로토콜 보완"""
        if not self.api_manager:
            return protocol
        
        prompt = f"""
        다음 실험 프로토콜을 검토하고 누락된 정보를 보완해주세요:
        
        {json.dumps(protocol, indent=2, ensure_ascii=False)}
        
        특히 다음 사항을 확인하고 추가해주세요:
        1. 누락된 필수 재료나 조건
        2. 안전 주의사항
        3. 일반적인 실험 조건 (명시되지 않은 경우)
        4. 예상되는 문제점과 해결 방법
        
        JSON 형식으로 응답해주세요.
        """
        
        try:
            response = self.api_manager.generate_ai_response(
                prompt=prompt,
                model='gemini',
                system_prompt="당신은 고분자 실험 전문가입니다."
            )
            
            # 응답 파싱
            enhanced = json.loads(response)
            
            # 원본과 병합
            return self.merge_protocols([protocol, enhanced])
            
        except Exception as e:
            logger.error(f"AI 프로토콜 보완 실패: {e}")
            return protocol
    
    def _merge_materials(self, materials1: Dict, materials2: Dict) -> Dict:
        """재료 정보 병합"""
        merged = copy.deepcopy(materials1)
        
        for category, items in materials2.items():
            if category not in merged:
                merged[category] = items
            else:
                # 중복 제거하며 병합
                existing_names = {item.get('name', str(item)).lower() for item in merged[category]}
                
                for item in items:
                    item_name = item.get('name', str(item)).lower()
                    if item_name not in existing_names:
                        merged[category].append(item)
        
        return merged
    
    def _merge_conditions(self, conditions1: Dict, conditions2: Dict) -> Dict:
        """조건 정보 병합"""
        merged = copy.deepcopy(conditions1)
        
        for key, value in conditions2.items():
            if key not in merged:
                merged[key] = value
            elif isinstance(value, dict) and isinstance(merged[key], dict):
                # 더 상세한 정보 우선
                if len(value) > len(merged[key]):
                    merged[key] = value
        
        return merged
    
    def _merge_procedures(self, procedure1: List, procedure2: List) -> List:
        """절차 정보 병합"""
        if not procedure1:
            return procedure2
        if not procedure2:
            return procedure1
        
        # 더 상세한 절차 선택
        if len(procedure1) >= len(procedure2):
            return procedure1
        else:
            return procedure2
    
    # === 편의 메서드 ===
    
    def quick_analysis(self, data: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        """빠른 데이터 분석"""
        results = {}
        
        # 데이터 검증
        validation = self.validator.validate(data)
        results['validation'] = {
            'valid': validation.valid,
            'issues': validation.issues,
            'warnings': validation.warnings
        }
        
        # 기술통계
        results['descriptive'] = self.analyze_data(data, 'descriptive')
        
        # 상관관계
        results['correlation'] = self.analyze_data(data, 'correlation')
        
        # 타겟 컬럼이 있으면 회귀분석
        if target_column and target_column in data.columns:
            predictors = [col for col in data.select_dtypes(include=[np.number]).columns 
                         if col != target_column][:5]  # 최대 5개
            
            if predictors:
                results['regression'] = self.analyze_data(
                    data, 'regression', 
                    predictors=predictors,
                    response=target_column
                )
        
        return results
    
    def export_results(self, results: Union[Dict, pd.DataFrame], 
                      filename: str, 
                      format: str = 'excel') -> str:
        """분석 결과 내보내기"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename}_{timestamp}"
        
        try:
            if format == 'excel':
                with pd.ExcelWriter(f"{filename}.xlsx", engine='openpyxl') as writer:
                    # DataFrame인 경우
                    if isinstance(results, pd.DataFrame):
                        results.to_excel(writer, sheet_name='Data', index=False)
                    
                    # Dict인 경우
                    elif isinstance(results, dict):
                        # 설계 데이터
                        if 'design' in results and results['design'] is not None:
                            results['design'].to_excel(writer, sheet_name='Design', index=False)
                        
                        # 전처리 데이터
                        if 'preprocessed_data' in results and isinstance(results['preprocessed_data'], pd.DataFrame):
                            results['preprocessed_data'].to_excel(writer, sheet_name='Processed_Data', index=False)
                        
                        # 분석 결과
                        for analysis_name, analysis_data in results.get('analysis', {}).items():
                            if isinstance(analysis_data, dict):
                                # 요약 데이터
                                summary_data = {
                                    'Metric': [],
                                    'Value': []
                                }
                                
                                # 통계량 추출
                                if 'statistics' in analysis_data:
                                    for k, v in analysis_data['statistics'].items():
                                        if isinstance(v, (int, float, str)):
                                            summary_data['Metric'].append(k)
                                            summary_data['Value'].append(v)
                                
                                if summary_data['Metric']:
                                    pd.DataFrame(summary_data).to_excel(
                                        writer, 
                                        sheet_name=f'{analysis_name[:20]}_summary',
                                        index=False
                                    )
                        
                        # 최적화 결과
                        if 'optimization' in results and results['optimization']:
                            opt_data = []
                            opt = results['optimization']
                            
                            # 최적값
                            opt_row = {'Type': 'Optimal Values'}
                            opt_row.update(opt.get('optimal_values', {}))
                            opt_data.append(opt_row)
                            
                            # 최적 반응
                            resp_row = {'Type': 'Optimal Responses'}
                            resp_row.update(opt.get('optimal_responses', {}))
                            opt_data.append(resp_row)
                            
                            # 민감도
                            if opt.get('sensitivity'):
                                sens_row = {'Type': 'Sensitivity'}
                                sens_row.update(opt['sensitivity'])
                                opt_data.append(sens_row)
                            
                            pd.DataFrame(opt_data).to_excel(
                                writer, 
                                sheet_name='Optimization',
                                index=False
                            )
                        
                        # 권장사항
                        if 'recommendations' in results and results['recommendations']:
                            rec_df = pd.DataFrame({
                                'Recommendations': results['recommendations']
                            })
                            rec_df.to_excel(writer, sheet_name='Recommendations', index=False)
            
            elif format == 'csv':
                if isinstance(results, pd.DataFrame):
                    results.to_csv(f"{filename}.csv", index=False)
                else:
                    # 주요 데이터만 CSV로
                    if 'preprocessed_data' in results and isinstance(results['preprocessed_data'], pd.DataFrame):
                        results['preprocessed_data'].to_csv(f"{filename}.csv", index=False)
                    elif 'design' in results and results['design'] is not None:
                        results['design'].to_csv(f"{filename}.csv", index=False)
            
            elif format == 'json':
                # JSON으로 저장
                def convert_to_serializable(obj):
                    if isinstance(obj, pd.DataFrame):
                        return obj.to_dict('records')
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, (datetime, pd.Timestamp)):
                        return obj.isoformat()
                    elif isinstance(obj, (np.integer, np.floating)):
                        return float(obj)
                    elif hasattr(obj, '__dict__'):
                        return str(obj)
                    return obj
                
                with open(f"{filename}.json", 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2, default=convert_to_serializable)
            
            logger.info(f"결과를 {filename}.{format}로 저장했습니다")
            return f"{filename}.{format}"
            
        except Exception as e:
            logger.error(f"결과 저장 실패: {e}")
            raise
    
    def get_processing_history(self) -> List[Dict]:
        """처리 이력 조회"""
        return self.processing_history
    
    def clear_history(self):
        """처리 이력 초기화"""
        self.processing_history = []
        logger.info("처리 이력이 초기화되었습니다")
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """데이터 유효성 검사 (간단한 인터페이스)"""
        report = self.validator.validate(data)
        issues = report.issues + report.warnings
        return report.valid, issues
    
    def suggest_design(self, 
                      n_factors: int, 
                      n_runs_available: int,
                      screening: bool = False) -> str:
        """실험 설계 추천"""
        if screening:
            if n_factors <= 7:
                return 'plackett_burman'
            else:
                return 'fractional_factorial'
        else:
            if n_runs_available >= 2**n_factors + 2*n_factors + 3:
                return 'central_composite'
            elif n_runs_available >= 2**n_factors:
                return 'factorial'
            elif n_factors >= 3 and n_runs_available >= 3*n_factors:
                return 'box_behnken'
            else:
                return 'latin_hypercube'


# === 싱글톤 인스턴스 ===

_data_processor: Optional[DataProcessor] = None

def get_data_processor(api_manager: Optional[APIManager] = None) -> DataProcessor:
    """DataProcessor 싱글톤 인스턴스 반환"""
    global _data_processor
    if _data_processor is None:
        _data_processor = DataProcessor(api_manager)
    return _data_processor


# === 편의 함수 ===

def quick_analysis(data: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
    """빠른 데이터 분석"""
    processor = get_data_processor()
    return processor.quick_analysis(data, target_column)


def create_doe_design(factors: List[Dict], design_type: str = 'auto', n_runs: Optional[int] = None) -> pd.DataFrame:
    """DOE 설계 생성 헬퍼"""
    processor = get_data_processor()
    
    # 자동 설계 선택
    if design_type == 'auto':
        n_factors = len([f for f in factors if f.get('type', 'continuous') == 'continuous'])
        if n_runs:
            design_type = processor.suggest_design(n_factors, n_runs)
        else:
            design_type = 'factorial' if n_factors <= 4 else 'fractional_factorial'
    
    return processor.create_experiment_design(factors, design_type)


# === 테스트 코드 ===

if __name__ == "__main__":
    # 테스트 데이터 생성
    np.random.seed(42)
    test_data = pd.DataFrame({
        'Temperature': np.random.uniform(20, 80, 50),
        'Pressure': np.random.uniform(1, 5, 50),
        'Time': np.random.uniform(10, 60, 50),
        'Yield': 50 + 0.5*np.random.uniform(20, 80, 50) + 2*np.random.uniform(1, 5, 50) + np.random.normal(0, 5, 50)
    })
    
    # 프로세서 생성
    processor = get_data_processor()
    
    # 1. 데이터 검증
    print("=== 데이터 검증 ===")
    is_valid, issues = processor.validate_data(test_data)
    print(f"유효성: {is_valid}")
    print(f"이슈: {issues}")
    
    # 2. 실험 설계
    print("\n=== 실험 설계 ===")
    factors = [
        {'name': 'Temperature', 'type': 'continuous', 'low': 20, 'high': 80},
        {'name': 'Pressure', 'type': 'continuous', 'low': 1, 'high': 5}
    ]
    
    design = processor.create_experiment_design(factors, 'central_composite')
    print(f"설계 크기: {design.shape}")
    print(design.head())
    
    # 3. 통계 분석
    print("\n=== 통계 분석 ===")
    analysis = processor.analyze_data(test_data, 'descriptive')
    print(f"기술통계 완료: {len(analysis['statistics'])} 항목")
    
    # 4. 빠른 분석
    print("\n=== 빠른 분석 ===")
    quick_results = quick_analysis(test_data, 'Yield')
    print(f"분석 완료: {list(quick_results.keys())}")
    
    print("\n테스트 완료!")
            std_estimate =
