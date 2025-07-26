# modules/core/general_experiment.py
"""
Universal DOE Platform - General Experiment Module
범용 실험 설계 모듈 - 모든 연구 분야를 지원하는 핵심 모듈
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
import json
from datetime import datetime
import traceback
import logging
from itertools import product, combinations
from scipy import stats
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# DOE 관련 라이브러리 - 선택적 의존성
try:
    from pyDOE3 import (
        fullfact, fracfact, pbdesign, ccdesign, bbdesign,
        lhs
    )
    PYDOE_AVAILABLE = True
except ImportError:
    PYDOE_AVAILABLE = False
    # pyDOE3가 없어도 기본 기능은 작동

# 부모 클래스 임포트
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from modules.base_module import BaseExperimentModule
from utils.data_processor import DataProcessor

logger = logging.getLogger(__name__)


# ==================== 데이터 클래스 ====================

@dataclass
class Factor:
    """실험 요인 정의"""
    name: str
    display_name: str
    type: str  # 'continuous', 'discrete', 'categorical'
    unit: str = ""
    
    # 연속형 요인
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    # 이산형/범주형 요인
    levels: Optional[List[Any]] = None
    
    # 메타데이터
    description: str = ""
    importance: str = "medium"  # 'high', 'medium', 'low'
    controllability: str = "full"  # 'full', 'partial', 'noise'
    measurement_precision: Optional[float] = None
    cost_per_level: Optional[float] = None
    
    # 제약사항
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[Dict[str, Any]] = field(default_factory=list)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """요인 유효성 검증"""
        errors = []
        
        if not self.name:
            errors.append("요인 이름이 필요합니다")
            
        if self.type == 'continuous':
            if self.min_value is None or self.max_value is None:
                errors.append(f"{self.name}: 연속형 요인은 최소/최대값이 필요합니다")
            elif self.min_value >= self.max_value:
                errors.append(f"{self.name}: 최소값은 최대값보다 작아야 합니다")
        
        elif self.type in ['discrete', 'categorical']:
            if not self.levels or len(self.levels) < 2:
                errors.append(f"{self.name}: 최소 2개 이상의 수준이 필요합니다")
                
        return len(errors) == 0, errors
    
    def get_coded_levels(self, n_levels: int = 2) -> List[float]:
        """코딩된 수준 반환"""
        if self.type == 'continuous':
            if n_levels == 2:
                return [-1, 1]
            else:
                return np.linspace(-1, 1, n_levels).tolist()
        else:
            return list(range(len(self.levels)))
    
    def decode_value(self, coded_value: float) -> Any:
        """코딩된 값을 실제 값으로 변환"""
        if self.type == 'continuous':
            # -1 to 1 => min to max
            return self.min_value + (coded_value + 1) * (self.max_value - self.min_value) / 2
        else:
            # 인덱스로 수준 선택
            idx = int(round(coded_value))
            return self.levels[idx] if 0 <= idx < len(self.levels) else self.levels[0]


@dataclass
class Response:
    """반응변수 정의"""
    name: str
    display_name: str
    unit: str = ""
    
    # 최적화 목표
    optimization_type: str = "maximize"  # 'maximize', 'minimize', 'target'
    target_value: Optional[float] = None
    lower_limit: Optional[float] = None
    upper_limit: Optional[float] = None
    
    # 메타데이터
    description: str = ""
    measurement_method: str = ""
    importance_weight: float = 1.0
    measurement_cost: Optional[float] = None
    measurement_time: Optional[float] = None
    
    def validate(self) -> Tuple[bool, List[str]]:
        """반응변수 유효성 검증"""
        errors = []
        
        if not self.name:
            errors.append("반응변수 이름이 필요합니다")
            
        if self.optimization_type == 'target' and self.target_value is None:
            errors.append(f"{self.name}: 목표값 최적화는 목표값이 필요합니다")
            
        if self.importance_weight <= 0:
            errors.append(f"{self.name}: 중요도 가중치는 양수여야 합니다")
            
        return len(errors) == 0, errors


@dataclass
class DesignResult:
    """실험 설계 결과"""
    design_matrix: pd.DataFrame
    design_type: str
    n_runs: int
    n_factors: int
    
    # 설계 속성
    resolution: Optional[str] = None
    orthogonality: Optional[float] = None
    d_efficiency: Optional[float] = None
    g_efficiency: Optional[float] = None
    condition_number: Optional[float] = None
    
    # 메타데이터
    creation_time: datetime = field(default_factory=datetime.now)
    design_parameters: Dict[str, Any] = field(default_factory=dict)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    
    # 추가 정보
    blocked: bool = False
    randomized: bool = True
    center_points: int = 0
    
    def get_summary(self) -> Dict[str, Any]:
        """설계 요약 정보"""
        return {
            'design_type': self.design_type,
            'n_runs': self.n_runs,
            'n_factors': self.n_factors,
            'resolution': self.resolution,
            'd_efficiency': f"{self.d_efficiency:.1f}%" if self.d_efficiency else "N/A",
            'orthogonality': f"{self.orthogonality:.3f}" if self.orthogonality else "N/A",
            'center_points': self.center_points,
            'blocked': self.blocked,
            'randomized': self.randomized
        }


# ==================== 설계 엔진 ====================

class DesignEngine:
    """실험 설계 생성 엔진"""
    
    def __init__(self):
        self.methods = {
            'full_factorial': self.generate_full_factorial,
            'fractional_factorial': self.generate_fractional_factorial,
            'plackett_burman': self.generate_plackett_burman,
            'ccd': self.generate_ccd,
            'box_behnken': self.generate_box_behnken,
            'd_optimal': self.generate_d_optimal,
            'latin_hypercube': self.generate_latin_hypercube,
            'custom': self.generate_custom
        }
    
    def generate_design(self, factors: List[Factor], method: str, 
                       params: Dict[str, Any]) -> DesignResult:
        """통합 설계 생성 메서드"""
        if method not in self.methods:
            raise ValueError(f"지원하지 않는 설계 방법: {method}")
            
        return self.methods[method](factors, params)
    
    def generate_full_factorial(self, factors: List[Factor], 
                               params: Dict[str, Any]) -> DesignResult:
        """완전요인설계 생성"""
        levels_per_factor = []
        factor_names = []
        
        for factor in factors:
            factor_names.append(factor.name)
            if factor.type == 'continuous':
                n_levels = params.get(f'{factor.name}_levels', 2)
                levels = factor.get_coded_levels(n_levels)
            else:
                levels = list(range(len(factor.levels)))
            levels_per_factor.append(levels)
        
        # 모든 조합 생성
        all_combinations = list(product(*levels_per_factor))
        design_array = np.array(all_combinations)
        
        # 중심점 추가
        n_center_points = params.get('center_points', 0)
        if n_center_points > 0:
            continuous_factors = [i for i, f in enumerate(factors) 
                                if f.type == 'continuous']
            if continuous_factors:
                center_point = np.zeros(len(factors))
                center_runs = np.tile(center_point, (n_center_points, 1))
                design_array = np.vstack([design_array, center_runs])
        
        # DataFrame 생성
        design_df = pd.DataFrame(design_array, columns=factor_names)
        
        # 실제 값으로 디코딩
        for i, factor in enumerate(factors):
            if factor.type == 'continuous':
                design_df[f'{factor.name}_actual'] = design_df[factor.name].apply(
                    lambda x: factor.decode_value(x)
                )
            else:
                design_df[f'{factor.name}_actual'] = design_df[factor.name].apply(
                    lambda x: factor.levels[int(x)] if 0 <= int(x) < len(factor.levels) else factor.levels[0]
                )
        
        # 랜덤화
        if params.get('randomize', True):
            design_df = design_df.sample(frac=1).reset_index(drop=True)
            design_df.insert(0, 'RunOrder', range(1, len(design_df) + 1))
        
        # 결과 생성
        result = DesignResult(
            design_matrix=design_df,
            design_type='Full Factorial',
            n_runs=len(design_df),
            n_factors=len(factors),
            center_points=n_center_points,
            randomized=params.get('randomize', True)
        )
        
        # 설계 품질 계산
        self._calculate_design_quality(result, factors)
        
        return result
    
    def generate_fractional_factorial(self, factors: List[Factor], 
                                    params: Dict[str, Any]) -> DesignResult:
        """부분요인설계 생성"""
        n_factors = len(factors)
        resolution = params.get('resolution', 3)
        
        # 생성기 결정
        if n_factors <= 3:
            # 작은 설계는 완전요인설계 사용
            return self.generate_full_factorial(factors, params)
        
        # 부분요인설계 생성 (간단한 구현)
        if n_factors == 4:
            # 2^(4-1) 설계
            base_design = self._generate_2level_factorial(3)
            # 네 번째 열은 처음 세 열의 곱
            col4 = base_design[:, 0] * base_design[:, 1] * base_design[:, 2]
            design_array = np.column_stack([base_design, col4])
        elif n_factors == 5:
            # 2^(5-2) 설계
            base_design = self._generate_2level_factorial(3)
            col4 = base_design[:, 0] * base_design[:, 1]
            col5 = base_design[:, 0] * base_design[:, 2]
            design_array = np.column_stack([base_design, col4, col5])
        else:
            # 더 큰 설계는 기본 패턴 사용
            n_runs = 2 ** (n_factors - (n_factors // 3))
            design_array = self._generate_hadamard_design(n_runs, n_factors)
        
        # DataFrame 생성
        factor_names = [f.name for f in factors]
        design_df = pd.DataFrame(design_array, columns=factor_names)
        
        # 디코딩 및 랜덤화
        for i, factor in enumerate(factors):
            design_df[f'{factor.name}_actual'] = design_df[factor.name].apply(
                lambda x: factor.decode_value(x)
            )
        
        if params.get('randomize', True):
            design_df = design_df.sample(frac=1).reset_index(drop=True)
            design_df.insert(0, 'RunOrder', range(1, len(design_df) + 1))
        
        result = DesignResult(
            design_matrix=design_df,
            design_type='Fractional Factorial',
            n_runs=len(design_df),
            n_factors=n_factors,
            resolution=f"Resolution {resolution}",
            randomized=params.get('randomize', True)
        )
        
        self._calculate_design_quality(result, factors)
        
        return result
    
    def generate_plackett_burman(self, factors: List[Factor], 
                                params: Dict[str, Any]) -> DesignResult:
        """Plackett-Burman 설계 생성"""
        n_factors = len(factors)
        
        # 실행 수 결정 (4의 배수)
        n_runs = 4
        while n_runs <= n_factors:
            n_runs += 4
        
        # 기본 Plackett-Burman 설계 생성
        if n_runs == 8:
            # 8-run 설계
            base_row = [1, 1, 1, -1, 1, -1, -1]
        elif n_runs == 12:
            # 12-run 설계
            base_row = [1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1]
        elif n_runs == 16:
            # 16-run 설계
            base_row = [1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1]
        else:
            # 일반적인 Hadamard 행렬 기반
            base_row = self._generate_hadamard_row(n_runs - 1)
        
        # 순환 행렬 생성
        design_list = []
        for i in range(n_runs - 1):
            row = base_row[i:] + base_row[:i]
            design_list.append(row[:n_factors])
        
        # 마지막 행은 모두 -1
        design_list.append([-1] * n_factors)
        
        design_array = np.array(design_list)
        
        # DataFrame 생성
        factor_names = [f.name for f in factors]
        design_df = pd.DataFrame(design_array, columns=factor_names)
        
        # 디코딩
        for i, factor in enumerate(factors):
            design_df[f'{factor.name}_actual'] = design_df[factor.name].apply(
                lambda x: factor.decode_value(x)
            )
        
        # 랜덤화
        if params.get('randomize', True):
            design_df = design_df.sample(frac=1).reset_index(drop=True)
            design_df.insert(0, 'RunOrder', range(1, len(design_df) + 1))
        
        result = DesignResult(
            design_matrix=design_df,
            design_type='Plackett-Burman',
            n_runs=n_runs,
            n_factors=n_factors,
            randomized=params.get('randomize', True)
        )
        
        self._calculate_design_quality(result, factors)
        
        return result
    
    def generate_ccd(self, factors: List[Factor], 
                    params: Dict[str, Any]) -> DesignResult:
        """중심합성설계 (CCD) 생성"""
        continuous_factors = [f for f in factors if f.type == 'continuous']
        n_continuous = len(continuous_factors)
        
        if n_continuous < 2:
            raise ValueError("CCD는 최소 2개 이상의 연속형 요인이 필요합니다")
        
        # 설계 유형
        ccd_type = params.get('ccd_type', 'ccc')  # ccc, cci, ccf
        alpha = params.get('alpha', 'rotatable')
        
        # 1. 요인점 (2^k 또는 2^(k-p))
        if n_continuous <= 5:
            factorial_array = self._generate_2level_factorial(n_continuous)
        else:
            # 큰 설계는 부분요인설계 사용
            resolution = params.get('resolution', 5)
            fraction = max(1, n_continuous - resolution + 1)
            n_runs = 2 ** (n_continuous - fraction)
            factorial_array = self._generate_hadamard_design(n_runs, n_continuous)[:, :n_continuous]
        
        # 2. 축점 (2k)
        axial_array = []
        for i in range(n_continuous):
            point_plus = np.zeros(n_continuous)
            point_minus = np.zeros(n_continuous)
            
            # alpha 값 결정
            if alpha == 'rotatable':
                alpha_value = (len(factorial_array)) ** 0.25
            elif alpha == 'orthogonal':
                alpha_value = np.sqrt(n_continuous)
            elif alpha == 'face':
                alpha_value = 1.0
            else:
                alpha_value = float(alpha)
            
            point_plus[i] = alpha_value
            point_minus[i] = -alpha_value
            
            axial_array.extend([point_plus, point_minus])
        
        axial_array = np.array(axial_array)
        
        # 3. 중심점
        n_center = params.get('center_points', 4)
        center_array = np.zeros((n_center, n_continuous))
        
        # 전체 설계 조합
        design_array = np.vstack([factorial_array, axial_array, center_array])
        
        # 범주형 요인 처리
        if len(factors) > n_continuous:
            categorical_factors = [f for f in factors if f.type != 'continuous']
            # 범주형 요인은 중심점 수준으로 고정
            cat_values = []
            for f in categorical_factors:
                cat_values.append([0] * len(design_array))
            
            cat_array = np.array(cat_values).T
            design_array = np.hstack([design_array, cat_array])
        
        # DataFrame 생성
        factor_names = [f.name for f in factors]
        design_df = pd.DataFrame(design_array, columns=factor_names)
        
        # 디코딩
        for i, factor in enumerate(factors):
            design_df[f'{factor.name}_actual'] = design_df[factor.name].apply(
                lambda x: factor.decode_value(x)
            )
        
        # 블록 지정
        design_df['Block'] = ['Factorial'] * len(factorial_array) + \
                           ['Axial'] * len(axial_array) + \
                           ['Center'] * n_center
        
        # 랜덤화
        if params.get('randomize', True):
            design_df = design_df.sample(frac=1).reset_index(drop=True)
            design_df.insert(0, 'RunOrder', range(1, len(design_df) + 1))
        
        result = DesignResult(
            design_matrix=design_df,
            design_type='Central Composite',
            n_runs=len(design_df),
            n_factors=len(factors),
            center_points=n_center,
            blocked=True,
            randomized=params.get('randomize', True),
            design_parameters={'alpha': alpha_value, 'type': ccd_type}
        )
        
        self._calculate_design_quality(result, continuous_factors)
        
        return result
    
    def generate_box_behnken(self, factors: List[Factor], 
                           params: Dict[str, Any]) -> DesignResult:
        """Box-Behnken 설계 생성"""
        continuous_factors = [f for f in factors if f.type == 'continuous']
        k = len(continuous_factors)
        
        if k < 3:
            raise ValueError("Box-Behnken 설계는 최소 3개 이상의 연속형 요인이 필요합니다")
        
        # Box-Behnken 설계 생성
        design_list = []
        
        # 2요인 조합에 대해 ±1 수준
        for i in range(k):
            for j in range(i + 1, k):
                for sign1 in [-1, 1]:
                    for sign2 in [-1, 1]:
                        point = [0] * k
                        point[i] = sign1
                        point[j] = sign2
                        design_list.append(point)
        
        design_array = np.array(design_list)
        
        # 중심점 추가
        n_center = params.get('center_points', 3)
        center_array = np.zeros((n_center, k))
        design_array = np.vstack([design_array, center_array])
        
        # 범주형 요인 처리
        if len(factors) > k:
            categorical_factors = [f for f in factors if f.type != 'continuous']
            cat_values = []
            for f in categorical_factors:
                cat_values.append([0] * len(design_array))
            
            cat_array = np.array(cat_values).T
            full_design_array = np.hstack([design_array, cat_array])
        else:
            full_design_array = design_array
        
        # DataFrame 생성
        factor_names = [f.name for f in factors]
        design_df = pd.DataFrame(full_design_array, columns=factor_names)
        
        # 디코딩
        for i, factor in enumerate(factors):
            design_df[f'{factor.name}_actual'] = design_df[factor.name].apply(
                lambda x: factor.decode_value(x)
            )
        
        # 랜덤화
        if params.get('randomize', True):
            design_df = design_df.sample(frac=1).reset_index(drop=True)
            design_df.insert(0, 'RunOrder', range(1, len(design_df) + 1))
        
        result = DesignResult(
            design_matrix=design_df,
            design_type='Box-Behnken',
            n_runs=len(design_df),
            n_factors=len(factors),
            center_points=n_center,
            randomized=params.get('randomize', True)
        )
        
        self._calculate_design_quality(result, continuous_factors)
        
        return result
    
    def generate_d_optimal(self, factors: List[Factor], 
                         params: Dict[str, Any]) -> DesignResult:
        """D-최적 설계 생성"""
        n_runs = params.get('n_runs', 20)
        
        # 후보점 생성
        candidate_set = self._generate_candidate_set(factors, params)
        
        # 초기 설계 선택 (랜덤)
        n_factors = len(factors)
        initial_indices = np.random.choice(len(candidate_set), 
                                         size=min(n_runs, len(candidate_set)), 
                                         replace=False)
        current_design = candidate_set[initial_indices]
        
        # 교환 알고리즘
        max_iter = params.get('max_iterations', 100)
        for iteration in range(max_iter):
            improved = False
            
            for i in range(len(current_design)):
                best_criterion = self._calculate_d_criterion(current_design)
                best_j = -1
                
                # 각 후보점에 대해 교환 시도
                for j in range(len(candidate_set)):
                    if j not in initial_indices:
                        # 교환
                        temp_design = current_design.copy()
                        temp_design[i] = candidate_set[j]
                        
                        # D-criterion 계산
                        criterion = self._calculate_d_criterion(temp_design)
                        
                        if criterion > best_criterion:
                            best_criterion = criterion
                            best_j = j
                            improved = True
                
                # 최선의 교환 수행
                if best_j >= 0:
                    current_design[i] = candidate_set[best_j]
                    initial_indices[i] = best_j
            
            if not improved:
                break
        
        # DataFrame 생성
        factor_names = [f.name for f in factors]
        design_df = pd.DataFrame(current_design, columns=factor_names)
        
        # 디코딩
        for i, factor in enumerate(factors):
            if factor.type == 'continuous':
                design_df[f'{factor.name}_actual'] = design_df[factor.name].apply(
                    lambda x: factor.decode_value(x)
                )
            else:
                design_df[f'{factor.name}_actual'] = design_df[factor.name].apply(
                    lambda x: factor.levels[int(x)] if 0 <= int(x) < len(factor.levels) else factor.levels[0]
                )
        
        # 랜덤화
        if params.get('randomize', True):
            design_df = design_df.sample(frac=1).reset_index(drop=True)
            design_df.insert(0, 'RunOrder', range(1, len(design_df) + 1))
        
        result = DesignResult(
            design_matrix=design_df,
            design_type='D-Optimal',
            n_runs=n_runs,
            n_factors=n_factors,
            randomized=params.get('randomize', True),
            d_efficiency=self._calculate_d_efficiency(current_design) * 100
        )
        
        self._calculate_design_quality(result, factors)
        
        return result
    
    def generate_latin_hypercube(self, factors: List[Factor], 
                                params: Dict[str, Any]) -> DesignResult:
        """Latin Hypercube 설계 생성"""
        n_runs = params.get('n_runs', 10)
        n_factors = len(factors)
        
        # 기본 Latin Hypercube 생성
        design_array = np.zeros((n_runs, n_factors))
        
        for i in range(n_factors):
            # 각 요인에 대해 순열 생성
            perm = np.random.permutation(n_runs)
            
            if factors[i].type == 'continuous':
                # 연속형: 균등 분할
                design_array[:, i] = (perm + np.random.rand(n_runs)) / n_runs * 2 - 1
            else:
                # 범주형: 수준 할당
                n_levels = len(factors[i].levels)
                level_size = n_runs // n_levels
                remainder = n_runs % n_levels
                
                levels = []
                for j in range(n_levels):
                    count = level_size + (1 if j < remainder else 0)
                    levels.extend([j] * count)
                
                np.random.shuffle(levels)
                design_array[:, i] = levels
        
        # 최적화 (선택적)
        if params.get('optimize', True):
            design_array = self._optimize_lhs(design_array, factors)
        
        # DataFrame 생성
        factor_names = [f.name for f in factors]
        design_df = pd.DataFrame(design_array, columns=factor_names)
        
        # 디코딩
        for i, factor in enumerate(factors):
            if factor.type == 'continuous':
                design_df[f'{factor.name}_actual'] = design_df[factor.name].apply(
                    lambda x: factor.decode_value(x)
                )
            else:
                design_df[f'{factor.name}_actual'] = design_df[factor.name].apply(
                    lambda x: factor.levels[int(x)] if 0 <= int(x) < len(factor.levels) else factor.levels[0]
                )
        
        # 이미 랜덤화됨
        design_df.insert(0, 'RunOrder', range(1, len(design_df) + 1))
        
        result = DesignResult(
            design_matrix=design_df,
            design_type='Latin Hypercube',
            n_runs=n_runs,
            n_factors=n_factors,
            randomized=True
        )
        
        self._calculate_design_quality(result, factors)
        
        return result
    
    def generate_custom(self, factors: List[Factor], 
                       params: Dict[str, Any]) -> DesignResult:
        """사용자 정의 설계"""
        custom_matrix = params.get('design_matrix')
        
        if custom_matrix is None:
            raise ValueError("사용자 정의 설계는 design_matrix가 필요합니다")
        
        # numpy array 또는 list를 DataFrame으로 변환
        if isinstance(custom_matrix, (np.ndarray, list)):
            factor_names = [f.name for f in factors]
            design_df = pd.DataFrame(custom_matrix, columns=factor_names)
        else:
            design_df = custom_matrix.copy()
        
        # 디코딩
        for i, factor in enumerate(factors):
            if f'{factor.name}_actual' not in design_df.columns:
                if factor.type == 'continuous':
                    design_df[f'{factor.name}_actual'] = design_df[factor.name].apply(
                        lambda x: factor.decode_value(x)
                    )
                else:
                    design_df[f'{factor.name}_actual'] = design_df[factor.name].apply(
                        lambda x: factor.levels[int(x)] if 0 <= int(x) < len(factor.levels) else factor.levels[0]
                    )
        
        # 랜덤화
        if params.get('randomize', True) and 'RunOrder' not in design_df.columns:
            design_df = design_df.sample(frac=1).reset_index(drop=True)
            design_df.insert(0, 'RunOrder', range(1, len(design_df) + 1))
        
        result = DesignResult(
            design_matrix=design_df,
            design_type='Custom',
            n_runs=len(design_df),
            n_factors=len(factors),
            randomized=params.get('randomize', True)
        )
        
        self._calculate_design_quality(result, factors)
        
        return result
    
    # ========== 헬퍼 메서드 ==========
    
    def _generate_2level_factorial(self, n_factors: int) -> np.ndarray:
        """2수준 완전요인설계 생성"""
        n_runs = 2 ** n_factors
        design = np.zeros((n_runs, n_factors))
        
        for i in range(n_factors):
            level_repeat = 2 ** (n_factors - i - 1)
            pattern = np.array([-1] * level_repeat + [1] * level_repeat)
            design[:, i] = np.tile(pattern, 2 ** i)
        
        return design
    
    def _generate_hadamard_design(self, n_runs: int, n_factors: int) -> np.ndarray:
        """Hadamard 기반 설계 생성"""
        # 간단한 구현 - 실제로는 더 정교한 알고리즘 필요
        design = np.random.choice([-1, 1], size=(n_runs, n_factors))
        return design
    
    def _generate_hadamard_row(self, length: int) -> List[int]:
        """Hadamard 행 생성"""
        # 간단한 패턴
        if length <= 7:
            return [1, -1, 1, 1, -1, -1, 1][:length]
        else:
            # 랜덤 패턴
            return list(np.random.choice([-1, 1], size=length))
    
    def _generate_candidate_set(self, factors: List[Factor], 
                               params: Dict[str, Any]) -> np.ndarray:
        """D-optimal을 위한 후보점 집합 생성"""
        grid_levels = params.get('grid_levels', 5)
        
        candidate_list = []
        
        # 각 요인에 대한 수준 생성
        factor_levels = []
        for factor in factors:
            if factor.type == 'continuous':
                levels = np.linspace(-1, 1, grid_levels)
            else:
                levels = list(range(len(factor.levels)))
            factor_levels.append(levels)
        
        # 전체 격자점 생성 (너무 크면 샘플링)
        total_points = np.prod([len(levels) for levels in factor_levels])
        
        if total_points > 10000:
            # 랜덤 샘플링
            n_samples = min(5000, total_points)
            candidates = []
            for _ in range(n_samples):
                point = []
                for levels in factor_levels:
                    point.append(np.random.choice(levels))
                candidates.append(point)
            return np.array(candidates)
        else:
            # 전체 격자
            return np.array(list(product(*factor_levels)))
    
    def _calculate_d_criterion(self, design: np.ndarray) -> float:
        """D-criterion 계산"""
        try:
            X = np.column_stack([np.ones(len(design)), design])
            XtX = X.T @ X
            return np.linalg.det(XtX) ** (1 / X.shape[1])
        except:
            return 0
    
    def _calculate_d_efficiency(self, design: np.ndarray) -> float:
        """D-efficiency 계산"""
        n, p = design.shape
        actual_det = self._calculate_d_criterion(design)
        
        # 이론적 최대값 (정규직교 설계)
        theoretical_max = n ** (p / n)
        
        return (actual_det / theoretical_max) if theoretical_max > 0 else 0
    
    def _optimize_lhs(self, design: np.ndarray, factors: List[Factor]) -> np.ndarray:
        """Latin Hypercube 최적화 (거리 최대화)"""
        n_runs, n_factors = design.shape
        continuous_indices = [i for i, f in enumerate(factors) if f.type == 'continuous']
        
        if not continuous_indices:
            return design
        
        # 간단한 교환 알고리즘
        for _ in range(100):
            i1, i2 = np.random.choice(n_runs, size=2, replace=False)
            col = np.random.choice(continuous_indices)
            
            # 교환 전후 최소 거리 비교
            original_min_dist = self._min_distance(design[:, continuous_indices])
            
            design[i1, col], design[i2, col] = design[i2, col], design[i1, col]
            new_min_dist = self._min_distance(design[:, continuous_indices])
            
            if new_min_dist < original_min_dist:
                # 원래대로 복구
                design[i1, col], design[i2, col] = design[i2, col], design[i1, col]
        
        return design
    
    def _min_distance(self, points: np.ndarray) -> float:
        """점들 간 최소 거리 계산"""
        n = len(points)
        min_dist = np.inf
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(points[i] - points[j])
                min_dist = min(min_dist, dist)
        
        return min_dist
    
    def _calculate_design_quality(self, result: DesignResult, 
                                 factors: List[Factor]) -> None:
        """설계 품질 지표 계산"""
        continuous_cols = []
        for factor in factors:
            if factor.type == 'continuous' and factor.name in result.design_matrix.columns:
                continuous_cols.append(factor.name)
        
        if not continuous_cols:
            return
        
        X = result.design_matrix[continuous_cols].values
        
        # 정규화
        X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        
        # D-efficiency
        try:
            result.d_efficiency = self._calculate_d_efficiency(X_norm) * 100
        except:
            result.d_efficiency = 0
        
        # Orthogonality
        try:
            corr_matrix = np.corrcoef(X_norm.T)
            off_diagonal = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            result.orthogonality = 1 - np.mean(np.abs(off_diagonal))
        except:
            result.orthogonality = 0
        
        # Condition number
        try:
            result.condition_number = np.linalg.cond(X_norm.T @ X_norm)
        except:
            result.condition_number = np.inf


# ==================== 분석 엔진 ====================

class AnalysisEngine:
    """실험 결과 분석 엔진"""
    
    def analyze(self, design: pd.DataFrame, results: pd.DataFrame, 
                factors: List[Factor], responses: List[Response],
                analysis_type: str = 'full') -> Dict[str, Any]:
        """통합 분석 메서드"""
        
        analysis_results = {
            'summary_statistics': self._calculate_summary_stats(results, responses),
            'main_effects': {},
            'interactions': {},
            'anova_tables': {},
            'regression_models': {},
            'optimal_conditions': {},
            'visualizations': {}
        }
        
        # 각 반응변수에 대해 분석
        for response in responses:
            if response.name not in results.columns:
                continue
            
            y = results[response.name].values
            
            # 주효과 분석
            main_effects = self._analyze_main_effects(design, y, factors)
            analysis_results['main_effects'][response.name] = main_effects
            
            # 교호작용 분석
            if analysis_type == 'full':
                interactions = self._analyze_interactions(design, y, factors)
                analysis_results['interactions'][response.name] = interactions
            
            # ANOVA
            anova_table = self._perform_anova(design, y, factors)
            analysis_results['anova_tables'][response.name] = anova_table
            
            # 회귀 모델
            regression_model = self._fit_regression_model(design, y, factors)
            analysis_results['regression_models'][response.name] = regression_model
            
            # 최적 조건
            optimal = self._find_optimal_conditions(
                regression_model, factors, response
            )
            analysis_results['optimal_conditions'][response.name] = optimal
        
        return analysis_results
    
    def _calculate_summary_stats(self, results: pd.DataFrame, 
                                responses: List[Response]) -> Dict[str, Any]:
        """기술통계량 계산"""
        stats = {}
        
        for response in responses:
            if response.name not in results.columns:
                continue
            
            data = results[response.name].dropna()
            
            stats[response.name] = {
                'count': len(data),
                'mean': data.mean(),
                'std': data.std(),
                'min': data.min(),
                'max': data.max(),
                'q1': data.quantile(0.25),
                'median': data.median(),
                'q3': data.quantile(0.75),
                'cv': (data.std() / data.mean() * 100) if data.mean() != 0 else np.inf
            }
        
        return stats
    
    def _analyze_main_effects(self, design: pd.DataFrame, y: np.ndarray,
                            factors: List[Factor]) -> Dict[str, Any]:
        """주효과 분석"""
        effects = {}
        
        for factor in factors:
            if factor.name not in design.columns:
                continue
            
            if factor.type == 'continuous':
                # 연속형: 선형 효과
                x = design[factor.name].values
                if len(np.unique(x)) > 1:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    effects[factor.name] = {
                        'type': 'linear',
                        'slope': slope,
                        'p_value': p_value,
                        'r_squared': r_value ** 2,
                        'significant': p_value < 0.05
                    }
            else:
                # 범주형: 수준별 평균
                x = design[factor.name].values
                levels = np.unique(x)
                level_means = {}
                
                for level in levels:
                    level_means[level] = y[x == level].mean()
                
                # ANOVA F-test
                groups = [y[x == level] for level in levels]
                if len(groups) > 1 and all(len(g) > 0 for g in groups):
                    f_stat, p_value = stats.f_oneway(*groups)
                    
                    effects[factor.name] = {
                        'type': 'categorical',
                        'level_means': level_means,
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
        
        return effects
    
    def _analyze_interactions(self, design: pd.DataFrame, y: np.ndarray,
                            factors: List[Factor]) -> Dict[str, Any]:
        """교호작용 분석"""
        interactions = {}
        
        # 2차 교호작용만 분석
        for i in range(len(factors)):
            for j in range(i + 1, len(factors)):
                f1, f2 = factors[i], factors[j]
                
                if f1.name not in design.columns or f2.name not in design.columns:
                    continue
                
                if f1.type == 'continuous' and f2.type == 'continuous':
                    # 연속형 x 연속형: 교호작용 항의 회귀계수
                    x1 = design[f1.name].values
                    x2 = design[f2.name].values
                    x12 = x1 * x2
                    
                    # 다중회귀
                    X = np.column_stack([np.ones_like(x1), x1, x2, x12])
                    try:
                        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                        
                        # 교호작용 항의 t-test
                        residuals = y - X @ coeffs
                        mse = np.mean(residuals ** 2)
                        se = np.sqrt(mse * np.diag(np.linalg.inv(X.T @ X)))
                        t_stat = coeffs[3] / se[3]
                        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(y) - 4))
                        
                        interactions[f"{f1.name}*{f2.name}"] = {
                            'coefficient': coeffs[3],
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        }
                    except:
                        pass
        
        return interactions
    
    def _perform_anova(self, design: pd.DataFrame, y: np.ndarray,
                      factors: List[Factor]) -> pd.DataFrame:
        """ANOVA 수행"""
        # 간단한 ANOVA 테이블
        anova_data = []
        
        # 전체 평균
        grand_mean = y.mean()
        sst = np.sum((y - grand_mean) ** 2)
        
        # 각 요인에 대한 제곱합
        total_ss_explained = 0
        
        for factor in factors:
            if factor.name not in design.columns:
                continue
            
            x = design[factor.name].values
            
            if factor.type == 'categorical' or len(np.unique(x)) < 10:
                # 범주형으로 처리
                levels = np.unique(x)
                ss = 0
                for level in levels:
                    level_mean = y[x == level].mean()
                    n_level = np.sum(x == level)
                    ss += n_level * (level_mean - grand_mean) ** 2
                
                df = len(levels) - 1
                ms = ss / df if df > 0 else 0
                
                anova_data.append({
                    'Source': factor.name,
                    'DF': df,
                    'SS': ss,
                    'MS': ms,
                    'F': 0,  # 나중에 계산
                    'p-value': 0
                })
                
                total_ss_explained += ss
        
        # 잔차
        sse = sst - total_ss_explained
        dfe = len(y) - sum(row['DF'] for row in anova_data) - 1
        mse = sse / dfe if dfe > 0 else 0
        
        # F 통계량과 p-value 계산
        for row in anova_data:
            if mse > 0:
                row['F'] = row['MS'] / mse
                row['p-value'] = 1 - stats.f.cdf(row['F'], row['DF'], dfe)
        
        # 잔차 행 추가
        anova_data.append({
            'Source': 'Error',
            'DF': dfe,
            'SS': sse,
            'MS': mse,
            'F': np.nan,
            'p-value': np.nan
        })
        
        # 전체 행 추가
        anova_data.append({
            'Source': 'Total',
            'DF': len(y) - 1,
            'SS': sst,
            'MS': np.nan,
            'F': np.nan,
            'p-value': np.nan
        })
        
        return pd.DataFrame(anova_data)
    
    def _fit_regression_model(self, design: pd.DataFrame, y: np.ndarray,
                            factors: List[Factor]) -> Dict[str, Any]:
        """회귀 모델 적합"""
        # 설계 행렬 구성
        X_list = [np.ones(len(y))]  # 절편
        feature_names = ['Intercept']
        
        # 주효과
        for factor in factors:
            if factor.name in design.columns:
                x = design[factor.name].values
                X_list.append(x)
                feature_names.append(factor.name)
                
                # 연속형 요인에 대해 2차 항 추가
                if factor.type == 'continuous':
                    X_list.append(x ** 2)
                    feature_names.append(f"{factor.name}^2")
        
        X = np.column_stack(X_list)
        
        # 회귀 계수 추정
        try:
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            
            # 잔차 계산
            y_pred = X @ coeffs
            residuals = y - y_pred
            
            # R-squared
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Adjusted R-squared
            n = len(y)
            p = X.shape[1]
            adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p)
            
            model = {
                'coefficients': dict(zip(feature_names, coeffs)),
                'r_squared': r_squared,
                'adj_r_squared': adj_r_squared,
                'rmse': np.sqrt(np.mean(residuals ** 2)),
                'predictions': y_pred,
                'residuals': residuals
            }
            
            return model
            
        except:
            return {
                'coefficients': {},
                'r_squared': 0,
                'adj_r_squared': 0,
                'rmse': np.inf,
                'predictions': np.zeros_like(y),
                'residuals': y
            }
    
    def _find_optimal_conditions(self, model: Dict[str, Any],
                               factors: List[Factor],
                               response: Response) -> Dict[str, Any]:
        """최적 조건 찾기"""
        if not model['coefficients']:
            return {}
        
        # 연속형 요인만 최적화
        continuous_factors = [f for f in factors if f.type == 'continuous']
        
        if not continuous_factors:
            return {}
        
        # 목적 함수 정의
        def objective(x):
            value = model['coefficients'].get('Intercept', 0)
            
            for i, factor in enumerate(continuous_factors):
                if factor.name in model['coefficients']:
                    value += model['coefficients'][factor.name] * x[i]
                
                if f"{factor.name}^2" in model['coefficients']:
                    value += model['coefficients'][f"{factor.name}^2"] * x[i] ** 2
            
            # 최적화 방향에 따라 부호 변경
            if response.optimization_type == 'minimize':
                return value
            elif response.optimization_type == 'maximize':
                return -value
            else:  # target
                return abs(value - response.target_value)
        
        # 초기값과 경계
        x0 = np.zeros(len(continuous_factors))
        bounds = [(-1, 1) for _ in continuous_factors]
        
        # 최적화 수행
        result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
        
        # 결과 정리
        optimal_conditions = {}
        for i, factor in enumerate(continuous_factors):
            optimal_conditions[factor.name] = {
                'coded': result.x[i],
                'actual': factor.decode_value(result.x[i])
            }
        
        # 예측값
        predicted_value = -result.fun if response.optimization_type == 'maximize' else result.fun
        
        return {
            'conditions': optimal_conditions,
            'predicted_value': predicted_value,
            'optimization_success': result.success
        }


# ==================== 메인 모듈 클래스 ====================

class GeneralExperimentModule(BaseExperimentModule):
    """범용 실험 설계 모듈"""
    
    def __init__(self):
        """모듈 초기화"""
        super().__init__()
        
        # 메타데이터 설정
        self.metadata.update({
            'module_id': 'general_experiment_v2',
            'name': '범용 실험 설계',
            'version': '2.0.0',
            'author': 'Universal DOE Platform Team',
            'description': '모든 연구 분야를 위한 범용 실험 설계 모듈',
            'category': 'core',
            'tags': ['general', 'universal', 'flexible', 'all-purpose', 'doe'],
            'icon': '🌐',
            'color': '#0066cc'
        })
        
        # 설정
        self.config = {
            'supported_designs': [
                'full_factorial', 'fractional_factorial', 
                'plackett_burman', 'ccd', 'box_behnken',
                'd_optimal', 'latin_hypercube', 'custom'
            ],
            'max_factors': 50,
            'max_runs': 10000,
            'allow_custom_factors': True,
            'allow_constraints': True,
            'validation_level': 'standard'
        }
        
        # 내부 엔진
        self.design_engine = DesignEngine()
        self.analysis_engine = AnalysisEngine()
        
        # 사용자 정의 요인/반응변수 저장
        self._custom_factors = []
        self._custom_responses = []
        
        self._initialized = True
    
    # ==================== 필수 메서드 구현 ====================
    
    def get_factors(self) -> List[Factor]:
        """실험 요인 목록 반환"""
        # 기본 템플릿 + 사용자 정의 요인
        default_factors = [
            Factor(
                name="temperature",
                display_name="온도",
                type="continuous",
                unit="°C",
                min_value=20,
                max_value=100,
                description="반응 온도",
                importance="high"
            ),
            Factor(
                name="time",
                display_name="시간",
                type="continuous",
                unit="min",
                min_value=10,
                max_value=120,
                description="반응 시간",
                importance="high"
            ),
            Factor(
                name="pressure",
                display_name="압력",
                type="continuous",
                unit="bar",
                min_value=1,
                max_value=10,
                description="반응 압력",
                importance="medium"
            ),
            Factor(
                name="catalyst",
                display_name="촉매",
                type="categorical",
                levels=["A", "B", "C", "None"],
                description="촉매 종류",
                importance="high"
            ),
            Factor(
                name="ph",
                display_name="pH",
                type="continuous",
                unit="",
                min_value=3,
                max_value=11,
                description="용액 pH",
                importance="medium"
            ),
            Factor(
                name="concentration",
                display_name="농도",
                type="continuous",
                unit="M",
                min_value=0.1,
                max_value=2.0,
                description="반응물 농도",
                importance="high"
            )
        ]
        
        return default_factors + self._custom_factors
    
    def get_responses(self) -> List[Response]:
        """반응변수 목록 반환"""
        default_responses = [
            Response(
                name="yield",
                display_name="수율",
                unit="%",
                optimization_type="maximize",
                description="생성물 수율",
                importance_weight=1.0
            ),
            Response(
                name="purity",
                display_name="순도",
                unit="%",
                optimization_type="maximize",
                description="생성물 순도",
                importance_weight=0.8
            ),
            Response(
                name="cost",
                display_name="비용",
                unit="$/kg",
                optimization_type="minimize",
                description="생산 비용",
                importance_weight=0.6
            ),
            Response(
                name="selectivity",
                display_name="선택성",
                unit="%",
                optimization_type="maximize",
                description="반응 선택성",
                importance_weight=0.7
            ),
            Response(
                name="conversion",
                display_name="전환율",
                unit="%",
                optimization_type="maximize",
                description="반응물 전환율",
                importance_weight=0.9
            )
        ]
        
        return default_responses + self._custom_responses
    
    def validate_design(self, design_params: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """설계 유효성 검증"""
        errors = []
        warnings = []
        
        # 요인 검증
        factors = design_params.get('factors', [])
        if not factors:
            errors.append("최소 1개 이상의 요인이 필요합니다")
        else:
            for factor in factors:
                valid, factor_errors = factor.validate()
                if not valid:
                    errors.extend(factor_errors)
        
        # 반응변수 검증
        responses = design_params.get('responses', [])
        if not responses:
            warnings.append("반응변수가 정의되지 않았습니다")
        else:
            for response in responses:
                valid, response_errors = response.validate()
                if not valid:
                    errors.extend(response_errors)
        
        # 설계 방법 검증
        design_type = design_params.get('design_type')
        if design_type not in self.config['supported_designs']:
            errors.append(f"지원하지 않는 설계 유형: {design_type}")
        
        # 실행 수 검증
        n_runs = design_params.get('n_runs', 0)
        if design_type == 'd_optimal' or design_type == 'latin_hypercube':
            if n_runs < len(factors) + 1:
                errors.append(f"최소 {len(factors) + 1}개의 실행이 필요합니다")
            elif n_runs > self.config['max_runs']:
                errors.append(f"최대 {self.config['max_runs']}개까지 실행 가능합니다")
        
        # 특정 설계별 검증
        if design_type == 'ccd':
            continuous_factors = [f for f in factors if f.type == 'continuous']
            if len(continuous_factors) < 2:
                errors.append("CCD는 최소 2개 이상의 연속형 요인이 필요합니다")
        
        elif design_type == 'box_behnken':
            continuous_factors = [f for f in factors if f.type == 'continuous']
            if len(continuous_factors) < 3:
                errors.append("Box-Behnken 설계는 최소 3개 이상의 연속형 요인이 필요합니다")
        
        # 경고 사항
        if len(factors) > 10:
            warnings.append("요인이 10개를 초과하면 실행 수가 많아질 수 있습니다")
        
        if design_type == 'full_factorial':
            estimated_runs = 1
            for factor in factors:
                if factor.type == 'continuous':
                    estimated_runs *= design_params.get(f'{factor.name}_levels', 2)
                else:
                    estimated_runs *= len(factor.levels)
            
            if estimated_runs > 100:
                warnings.append(f"완전요인설계로 약 {estimated_runs}개의 실행이 필요합니다. "
                              "부분요인설계를 고려해보세요.")
        
        return len(errors) == 0, errors, warnings
    
    def generate_design(self, design_params: Dict[str, Any]) -> Dict[str, Any]:
        """실험 설계 생성"""
        try:
            # 파라미터 추출
            factors = design_params.get('factors', [])
            design_type = design_params.get('design_type', 'full_factorial')
            
            # 설계 생성
            result = self.design_engine.generate_design(
                factors=factors,
                method=design_type,
                params=design_params
            )
            
            # 시각화 생성
            visualizations = self._create_design_visualizations(result, factors)
            
            return {
                'success': True,
                'design': result,
                'visualizations': visualizations,
                'summary': result.get_summary()
            }
            
        except Exception as e:
            logger.error(f"설계 생성 실패: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def analyze_results(self, design: pd.DataFrame, results: pd.DataFrame, 
                       analysis_params: Dict[str, Any]) -> Dict[str, Any]:
        """실험 결과 분석"""
        try:
            # 파라미터 추출
            factors = analysis_params.get('factors', [])
            responses = analysis_params.get('responses', [])
            analysis_type = analysis_params.get('analysis_type', 'full')
            
            # 분석 수행
            analysis_results = self.analysis_engine.analyze(
                design=design,
                results=results,
                factors=factors,
                responses=responses,
                analysis_type=analysis_type
            )
            
            # 시각화 생성
            visualizations = self._create_analysis_visualizations(
                design, results, analysis_results, factors, responses
            )
            analysis_results['visualizations'] = visualizations
            
            # 권장사항 생성
            recommendations = self._generate_recommendations(
                analysis_results, factors, responses
            )
            analysis_results['recommendations'] = recommendations
            
            return {
                'success': True,
                'results': analysis_results
            }
            
        except Exception as e:
            logger.error(f"분석 실패: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    # ==================== 추가 공개 메서드 ====================
    
    def add_custom_factor(self, factor: Factor) -> bool:
        """사용자 정의 요인 추가"""
        try:
            valid, errors = factor.validate()
            if valid:
                self._custom_factors.append(factor)
                return True
            else:
                logger.error(f"요인 검증 실패: {errors}")
                return False
        except Exception as e:
            logger.error(f"요인 추가 실패: {str(e)}")
            return False
    
    def add_custom_response(self, response: Response) -> bool:
        """사용자 정의 반응변수 추가"""
        try:
            valid, errors = response.validate()
            if valid:
                self._custom_responses.append(response)
                return True
            else:
                logger.error(f"반응변수 검증 실패: {errors}")
                return False
        except Exception as e:
            logger.error(f"반응변수 추가 실패: {str(e)}")
            return False
    
    def clear_custom_variables(self) -> None:
        """사용자 정의 변수 초기화"""
        self._custom_factors.clear()
        self._custom_responses.clear()
    
    def get_design_recommendations(self, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """요구사항에 따른 설계 추천"""
        recommendations = []
        
        n_factors = requirements.get('n_factors', 0)
        n_runs_budget = requirements.get('max_runs', np.inf)
        objective = requirements.get('objective', 'screening')  # screening, optimization, robustness
        
        # 스크리닝
        if objective == 'screening':
            if n_factors <= 7:
                recommendations.append({
                    'design_type': 'fractional_factorial',
                    'description': '부분요인설계 (Resolution IV)',
                    'estimated_runs': 2 ** (n_factors - 1),
                    'pros': ['적은 실행 수', '주효과 추정 가능'],
                    'cons': ['일부 교호작용 교락']
                })
            else:
                recommendations.append({
                    'design_type': 'plackett_burman',
                    'description': 'Plackett-Burman 설계',
                    'estimated_runs': 4 * ((n_factors // 4) + 1),
                    'pros': ['매우 효율적', '많은 요인 처리'],
                    'cons': ['교호작용 추정 불가']
                })
        
        # 최적화
        elif objective == 'optimization':
            if n_factors <= 4:
                recommendations.append({
                    'design_type': 'ccd',
                    'description': '중심합성설계',
                    'estimated_runs': 2 ** n_factors + 2 * n_factors + 4,
                    'pros': ['2차 모델 적합', '순차적 실험 가능'],
                    'cons': ['실행 수 많음']
                })
                
                recommendations.append({
                    'design_type': 'box_behnken',
                    'description': 'Box-Behnken 설계',
                    'estimated_runs': 2 * n_factors * (n_factors - 1) + 3,
                    'pros': ['CCD보다 적은 실행', '극단 조건 회피'],
                    'cons': ['꼭짓점 제외']
                })
            
            recommendations.append({
                'design_type': 'd_optimal',
                'description': 'D-최적 설계',
                'estimated_runs': min(n_runs_budget, 3 * n_factors),
                'pros': ['실행 수 자유 설정', '제약조건 처리'],
                'cons': ['계산 복잡도 높음']
            })
        
        # 강건성
        elif objective == 'robustness':
            recommendations.append({
                'design_type': 'latin_hypercube',
                'description': 'Latin Hypercube 설계',
                'estimated_runs': n_runs_budget,
                'pros': ['공간 충진성 좋음', '유연한 실행 수'],
                'cons': ['모델 적합 어려움']
            })
        
        # 예산에 맞게 필터링
        recommendations = [r for r in recommendations 
                         if r['estimated_runs'] <= n_runs_budget]
        
        # 우선순위 정렬
        recommendations.sort(key=lambda x: x['estimated_runs'])
        
        return recommendations
    
    # ==================== 비공개 헬퍼 메서드 ====================
    
    def _create_design_visualizations(self, design_result: DesignResult,
                                    factors: List[Factor]) -> Dict[str, Any]:
        """설계 시각화 생성"""
        visualizations = {}
        
        # 1. 설계 매트릭스 히트맵
        if len(factors) <= 10:
            continuous_factors = [f for f in factors if f.type == 'continuous']
            if continuous_factors:
                factor_names = [f.name for f in continuous_factors]
                design_matrix = design_result.design_matrix[factor_names].values
                
                fig = go.Figure(data=go.Heatmap(
                    z=design_matrix.T,
                    x=[f"Run {i+1}" for i in range(len(design_matrix))],
                    y=factor_names,
                    colorscale='RdBu',
                    zmid=0
                ))
                
                fig.update_layout(
                    title="설계 매트릭스",
                    xaxis_title="실험 런",
                    yaxis_title="요인",
                    height=400
                )
                
                visualizations['design_matrix'] = fig
        
        # 2. 3D 산점도 (3개 요인인 경우)
        continuous_factors = [f for f in factors if f.type == 'continuous']
        if len(continuous_factors) >= 3:
            f1, f2, f3 = continuous_factors[:3]
            
            fig = go.Figure(data=go.Scatter3d(
                x=design_result.design_matrix[f1.name],
                y=design_result.design_matrix[f2.name],
                z=design_result.design_matrix[f3.name],
                mode='markers',
                marker=dict(
                    size=8,
                    color=design_result.design_matrix.get('RunOrder', range(len(design_result.design_matrix))),
                    colorscale='Viridis',
                    showscale=True
                )
            ))
            
            fig.update_layout(
                title="설계 공간 (3D)",
                scene=dict(
                    xaxis_title=f1.display_name,
                    yaxis_title=f2.display_name,
                    zaxis_title=f3.display_name
                ),
                height=500
            )
            
            visualizations['design_space_3d'] = fig
        
        # 3. 페어플롯 (2D 투영)
        if 2 <= len(continuous_factors) <= 5:
            from plotly.subplots import make_subplots
            
            n_factors = len(continuous_factors)
            fig = make_subplots(
                rows=n_factors, cols=n_factors,
                shared_xaxes=True, shared_yaxes=True
            )
            
            for i in range(n_factors):
                for j in range(n_factors):
                    if i == j:
                        # 대각선: 히스토그램
                        values = design_result.design_matrix[continuous_factors[i].name]
                        fig.add_trace(
                            go.Histogram(x=values, showlegend=False),
                            row=i+1, col=j+1
                        )
                    else:
                        # 산점도
                        fig.add_trace(
                            go.Scatter(
                                x=design_result.design_matrix[continuous_factors[j].name],
                                y=design_result.design_matrix[continuous_factors[i].name],
                                mode='markers',
                                showlegend=False
                            ),
                            row=i+1, col=j+1
                        )
            
            # 축 레이블
            for i in range(n_factors):
                fig.update_xaxes(title_text=continuous_factors[i].display_name, 
                               row=n_factors, col=i+1)
                fig.update_yaxes(title_text=continuous_factors[i].display_name, 
                               row=i+1, col=1)
            
            fig.update_layout(title="설계 공간 투영", height=600)
            visualizations['pairplot'] = fig
        
        return visualizations
    
    def _create_analysis_visualizations(self, design: pd.DataFrame,
                                      results: pd.DataFrame,
                                      analysis_results: Dict[str, Any],
                                      factors: List[Factor],
                                      responses: List[Response]) -> Dict[str, Any]:
        """분석 결과 시각화"""
        visualizations = {}
        
        for response in responses:
            if response.name not in results.columns:
                continue
            
            response_data = results[response.name]
            
            # 1. 주효과 플롯
            main_effects = analysis_results['main_effects'].get(response.name, {})
            if main_effects:
                fig = make_subplots(
                    rows=1, cols=len(main_effects),
                    subplot_titles=list(main_effects.keys())
                )
                
                col = 1
                for factor_name, effect_data in main_effects.items():
                    factor = next((f for f in factors if f.name == factor_name), None)
                    if not factor:
                        continue
                    
                    if effect_data['type'] == 'linear':
                        # 연속형: 회귀선
                        x = design[factor_name]
                        y = response_data
                        
                        fig.add_trace(
                            go.Scatter(x=x, y=y, mode='markers', name='Data'),
                            row=1, col=col
                        )
                        
                        # 회귀선
                        x_range = np.linspace(x.min(), x.max(), 100)
                        y_pred = effect_data['slope'] * x_range + y.mean() - effect_data['slope'] * x.mean()
                        
                        fig.add_trace(
                            go.Scatter(x=x_range, y=y_pred, mode='lines', name='Fit'),
                            row=1, col=col
                        )
                    
                    else:
                        # 범주형: 막대 그래프
                        levels = list(effect_data['level_means'].keys())
                        means = list(effect_data['level_means'].values())
                        
                        fig.add_trace(
                            go.Bar(x=levels, y=means),
                            row=1, col=col
                        )
                    
                    col += 1
                
                fig.update_layout(
                    title=f"주효과 플롯 - {response.display_name}",
                    showlegend=False,
                    height=400
                )
                
                visualizations[f'main_effects_{response.name}'] = fig
            
            # 2. 잔차 플롯
            model = analysis_results['regression_models'].get(response.name, {})
            if 'residuals' in model and 'predictions' in model:
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=['잔차 vs 적합값', '정규 Q-Q', '히스토그램', '순서 플롯']
                )
                
                residuals = model['residuals']
                predictions = model['predictions']
                
                # 잔차 vs 적합값
                fig.add_trace(
                    go.Scatter(x=predictions, y=residuals, mode='markers'),
                    row=1, col=1
                )
                fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
                
                # Q-Q 플롯
                sorted_residuals = np.sort(residuals)
                theoretical_quantiles = stats.norm.ppf(
                    np.linspace(0.01, 0.99, len(residuals))
                )
                
                fig.add_trace(
                    go.Scatter(x=theoretical_quantiles, y=sorted_residuals, mode='markers'),
                    row=1, col=2
                )
                fig.add_trace(
                    go.Scatter(x=[-3, 3], y=[-3, 3], mode='lines', line=dict(dash='dash')),
                    row=1, col=2
                )
                
                # 히스토그램
                fig.add_trace(
                    go.Histogram(x=residuals, nbinsx=20),
                    row=2, col=1
                )
                
                # 순서 플롯
                fig.add_trace(
                    go.Scatter(y=residuals, mode='markers+lines'),
                    row=2, col=2
                )
                fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)
                
                fig.update_layout(
                    title=f"잔차 진단 - {response.display_name}",
                    showlegend=False,
                    height=600
                )
                
                visualizations[f'residuals_{response.name}'] = fig
            
            # 3. 반응표면 (2요인)
            continuous_factors = [f for f in factors if f.type == 'continuous']
            if len(continuous_factors) >= 2 and response.name in results.columns:
                f1, f2 = continuous_factors[:2]
                
                # 격자 생성
                x_range = np.linspace(-1, 1, 50)
                y_range = np.linspace(-1, 1, 50)
                X_grid, Y_grid = np.meshgrid(x_range, y_range)
                
                # 모델 예측 (간단한 2차 모델)
                coeffs = model.get('coefficients', {})
                
                Z_grid = np.zeros_like(X_grid)
                Z_grid += coeffs.get('Intercept', 0)
                Z_grid += coeffs.get(f1.name, 0) * X_grid
                Z_grid += coeffs.get(f2.name, 0) * Y_grid
                Z_grid += coeffs.get(f'{f1.name}^2', 0) * X_grid**2
                Z_grid += coeffs.get(f'{f2.name}^2', 0) * Y_grid**2
                
                fig = go.Figure(data=[
                    go.Surface(x=x_range, y=y_range, z=Z_grid, colorscale='Viridis'),
                    go.Scatter3d(
                        x=design[f1.name],
                        y=design[f2.name],
                        z=response_data,
                        mode='markers',
                        marker=dict(size=8, color='red')
                    )
                ])
                
                fig.update_layout(
                    title=f"반응표면 - {response.display_name}",
                    scene=dict(
                        xaxis_title=f1.display_name,
                        yaxis_title=f2.display_name,
                        zaxis_title=response.display_name
                    ),
                    height=500
                )
                
                visualizations[f'response_surface_{response.name}'] = fig
        
        return visualizations
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any],
                                factors: List[Factor],
                                responses: List[Response]) -> List[str]:
        """분석 결과 기반 권장사항 생성"""
        recommendations = []
        
        # 모델 적합도 체크
        for response in responses:
            model = analysis_results['regression_models'].get(response.name, {})
            r_squared = model.get('r_squared', 0)
            
            if r_squared < 0.7:
                recommendations.append(
                    f"⚠️ {response.display_name}의 R² = {r_squared:.3f}로 낮습니다. "
                    "추가 요인을 고려하거나 비선형 모델을 시도해보세요."
                )
            elif r_squared > 0.95:
                recommendations.append(
                    f"✅ {response.display_name}의 모델 적합도가 매우 우수합니다 (R² = {r_squared:.3f})"
                )
        
        # 유의한 요인 식별
        for response in responses:
            main_effects = analysis_results['main_effects'].get(response.name, {})
            significant_factors = [
                factor_name for factor_name, effect in main_effects.items()
                if effect.get('significant', False)
            ]
            
            if significant_factors:
                recommendations.append(
                    f"📊 {response.display_name}에 유의한 영향을 미치는 요인: "
                    f"{', '.join(significant_factors)}"
                )
        
        # 최적 조건 제시
        for response in responses:
            optimal = analysis_results['optimal_conditions'].get(response.name, {})
            if optimal and optimal.get('optimization_success'):
                conditions = optimal['conditions']
                pred_value = optimal['predicted_value']
                
                condition_str = ", ".join([
                    f"{name}: {values['actual']:.2f}"
                    for name, values in conditions.items()
                ])
                
                recommendations.append(
                    f"🎯 {response.display_name} 최적 조건: {condition_str} "
                    f"(예측값: {pred_value:.2f})"
                )
        
        # 추가 실험 제안
        recommendations.append(
            "💡 검증 실험: 최적 조건에서 3-5회 반복 실험을 수행하여 재현성을 확인하세요."
        )
        
        return recommendations


# ==================== 모듈 등록 ====================

# 모듈이 임포트될 때 자동으로 레지스트리에 등록되도록 설정
if __name__ != "__main__":
    try:
        from modules.module_registry import get_module_registry
        registry = get_module_registry()
        registry.register(GeneralExperimentModule)
    except ImportError:
        # 레지스트리를 찾을 수 없는 경우 무시
        pass
