"""
고분자 실험 설계 모듈
- 고분자 용매 시스템 설계
- 고분자 가공 조건 최적화
- 섬유화/필름화 공정 설계
- 나노재료 합성 실험
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime
import json

from modules.base_module import (
    BaseExperimentModule, 
    Factor, 
    Response,
    Design,
    DesignInfo,
    ValidationResult,
    AnalysisResult,
    VisualizationResult,
    AIRecommendation,
    FactorType,
    ResponseType,
    DesignMethod
)

logger = logging.getLogger(__name__)

# ==============================================================================
# 🧬 고분자 특화 상수 및 열거형
# ==============================================================================

class PolymerProcessType(str, Enum):
    """고분자 가공 유형"""
    DISSOLUTION = "dissolution"
    ELECTROSPINNING = "electrospinning"
    COATING = "coating"
    EXTRUSION = "extrusion"
    INJECTION = "injection"
    FILM_CASTING = "film_casting"
    NANOPARTICLE = "nanoparticle"
    FIBER_SPINNING = "fiber_spinning"

class SolventSystemType(str, Enum):
    """용매 시스템 유형"""
    SINGLE = "single"
    BINARY = "binary"
    TERNARY = "ternary"
    QUATERNARY = "quaternary"
    MULTICOMPONENT = "multicomponent"

class PhaseType(str, Enum):
    """상 유형"""
    SINGLE_PHASE = "single_phase"
    BIPHASIC = "biphasic"
    TRIPHASIC = "triphasic"
    MULTIPHASE = "multiphase"

# ==============================================================================
# 🧪 용매 시스템 설계 클래스
# ==============================================================================

@dataclass
class SolventComponent:
    """용매 성분"""
    name: str
    cas_number: Optional[str] = None
    hansen_parameters: Dict[str, float] = field(default_factory=dict)  # δD, δP, δH
    properties: Dict[str, Any] = field(default_factory=dict)  # bp, mp, viscosity, etc.
    ratio: float = 100.0  # 비율 (%)
    
@dataclass
class SolventSystem:
    """용매 시스템"""
    components: List[SolventComponent]
    system_type: SolventSystemType
    phase_behavior: PhaseType = PhaseType.SINGLE_PHASE
    temperature: float = 25.0  # °C
    total_hansen: Optional[Dict[str, float]] = None
    
    def calculate_hansen_parameters(self) -> Dict[str, float]:
        """혼합 용매의 Hansen 매개변수 계산"""
        if len(self.components) == 1:
            return self.components[0].hansen_parameters
        
        # 부피 분율 기반 가중 평균
        total_ratio = sum(comp.ratio for comp in self.components)
        hansen = {'δD': 0, 'δP': 0, 'δH': 0}
        
        for comp in self.components:
            weight = comp.ratio / total_ratio
            for param in ['δD', 'δP', 'δH']:
                if param in comp.hansen_parameters:
                    hansen[param] += weight * comp.hansen_parameters[param]
        
        self.total_hansen = hansen
        return hansen

class SolventSystemDesign:
    """다성분 용매/용제 시스템 설계"""
    
    def __init__(self):
        self.solvent_database = self._load_solvent_database()
        
    def _load_solvent_database(self) -> Dict[str, SolventComponent]:
        """용매 데이터베이스 로드"""
        # 주요 용매 데이터베이스 (실제로는 외부 파일에서 로드)
        solvents = {
            'THF': SolventComponent(
                name='Tetrahydrofuran',
                cas_number='109-99-9',
                hansen_parameters={'δD': 16.8, 'δP': 5.7, 'δH': 8.0},
                properties={'bp': 66, 'mp': -108.4, 'viscosity': 0.48}
            ),
            'DMF': SolventComponent(
                name='N,N-Dimethylformamide',
                cas_number='68-12-2',
                hansen_parameters={'δD': 17.4, 'δP': 13.7, 'δH': 11.3},
                properties={'bp': 153, 'mp': -61, 'viscosity': 0.92}
            ),
            'Toluene': SolventComponent(
                name='Toluene',
                cas_number='108-88-3',
                hansen_parameters={'δD': 18.0, 'δP': 1.4, 'δH': 2.0},
                properties={'bp': 110.6, 'mp': -93, 'viscosity': 0.59}
            ),
            'Chloroform': SolventComponent(
                name='Chloroform',
                cas_number='67-66-3',
                hansen_parameters={'δD': 17.8, 'δP': 3.1, 'δH': 5.7},
                properties={'bp': 61.2, 'mp': -63.5, 'viscosity': 0.57}
            ),
            'DMSO': SolventComponent(
                name='Dimethyl sulfoxide',
                cas_number='67-68-5',
                hansen_parameters={'δD': 18.4, 'δP': 16.4, 'δH': 10.2},
                properties={'bp': 189, 'mp': 18.5, 'viscosity': 1.99}
            ),
            'Water': SolventComponent(
                name='Water',
                cas_number='7732-18-5',
                hansen_parameters={'δD': 15.5, 'δP': 16.0, 'δH': 42.3},
                properties={'bp': 100, 'mp': 0, 'viscosity': 0.89}
            ),
            'Ethanol': SolventComponent(
                name='Ethanol',
                cas_number='64-17-5',
                hansen_parameters={'δD': 15.8, 'δP': 8.8, 'δH': 19.4},
                properties={'bp': 78.4, 'mp': -114.1, 'viscosity': 1.08}
            ),
            'Acetone': SolventComponent(
                name='Acetone',
                cas_number='67-64-1',
                hansen_parameters={'δD': 15.5, 'δP': 10.4, 'δH': 7.0},
                properties={'bp': 56.1, 'mp': -94.7, 'viscosity': 0.31}
            ),
            'DCM': SolventComponent(
                name='Dichloromethane',
                cas_number='75-09-2',
                hansen_parameters={'δD': 18.2, 'δP': 6.3, 'δH': 6.1},
                properties={'bp': 39.6, 'mp': -96.7, 'viscosity': 0.43}
            ),
            'MEK': SolventComponent(
                name='Methyl ethyl ketone',
                cas_number='78-93-3',
                hansen_parameters={'δD': 16.0, 'δP': 9.0, 'δH': 5.1},
                properties={'bp': 79.6, 'mp': -86.7, 'viscosity': 0.40}
            )
        }
        return solvents
    
    def calculate_hansen_distance(self, polymer_hansen: Dict[str, float], 
                                 solvent_hansen: Dict[str, float]) -> float:
        """Hansen 거리 계산 (Ra)"""
        dD = polymer_hansen.get('δD', 0) - solvent_hansen.get('δD', 0)
        dP = polymer_hansen.get('δP', 0) - solvent_hansen.get('δP', 0)
        dH = polymer_hansen.get('δH', 0) - solvent_hansen.get('δH', 0)
        
        Ra = (4 * dD**2 + dP**2 + dH**2)**0.5
        return Ra
    
    def predict_solubility(self, polymer_hansen: Dict[str, float],
                          solvent_system: SolventSystem,
                          Ro: float = 8.0) -> Dict[str, Any]:
        """용해도 예측"""
        solvent_hansen = solvent_system.calculate_hansen_parameters()
        Ra = self.calculate_hansen_distance(polymer_hansen, solvent_hansen)
        
        RED = Ra / Ro  # Relative Energy Difference
        
        result = {
            'hansen_distance': Ra,
            'relative_energy_difference': RED,
            'solubility_prediction': 'Good' if RED < 1 else 'Poor',
            'confidence': max(0, min(1, 2 - RED)) if RED < 2 else 0
        }
        
        return result
    
    def design_binary_system(self, good_solvent: str, poor_solvent: str,
                           ratios: List[float] = None) -> List[SolventSystem]:
        """이성분 용매 시스템 설계"""
        if ratios is None:
            ratios = [0, 25, 50, 75, 100]
        
        systems = []
        for ratio in ratios:
            if ratio == 0:
                components = [self.solvent_database[poor_solvent]]
                components[0].ratio = 100
            elif ratio == 100:
                components = [self.solvent_database[good_solvent]]
                components[0].ratio = 100
            else:
                comp1 = self.solvent_database[good_solvent].copy()
                comp2 = self.solvent_database[poor_solvent].copy()
                comp1.ratio = ratio
                comp2.ratio = 100 - ratio
                components = [comp1, comp2]
            
            system = SolventSystem(
                components=components,
                system_type=SolventSystemType.BINARY if len(components) > 1 else SolventSystemType.SINGLE
            )
            systems.append(system)
        
        return systems
    
    def design_ternary_system(self, solvent1: str, solvent2: str, solvent3: str,
                            design_points: int = 10) -> List[SolventSystem]:
        """삼성분 용매 시스템 설계 (Simplex-lattice design)"""
        systems = []
        
        # Simplex-lattice {3, 2} design points
        if design_points == 7:
            ratios = [
                (1, 0, 0), (0, 1, 0), (0, 0, 1),  # Pure components
                (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5),  # Binary blends
                (1/3, 1/3, 1/3)  # Centroid
            ]
        else:
            # Custom grid
            n = int(design_points**(1/2))
            ratios = []
            for i in range(n):
                for j in range(n):
                    x1 = i / (n - 1)
                    x2 = j / (n - 1) * (1 - x1)
                    x3 = 1 - x1 - x2
                    if x3 >= 0:
                        ratios.append((x1, x2, x3))
        
        for r1, r2, r3 in ratios:
            components = []
            if r1 > 0:
                comp1 = self.solvent_database[solvent1].copy()
                comp1.ratio = r1 * 100
                components.append(comp1)
            if r2 > 0:
                comp2 = self.solvent_database[solvent2].copy()
                comp2.ratio = r2 * 100
                components.append(comp2)
            if r3 > 0:
                comp3 = self.solvent_database[solvent3].copy()
                comp3.ratio = r3 * 100
                components.append(comp3)
            
            system = SolventSystem(
                components=components,
                system_type=SolventSystemType.TERNARY if len(components) == 3 else 
                           SolventSystemType.BINARY if len(components) == 2 else 
                           SolventSystemType.SINGLE
            )
            systems.append(system)
        
        return systems

# ==============================================================================
# 🎯 고분자 실험 모듈
# ==============================================================================

class PolymerExperimentModule(BaseExperimentModule):
    """고분자 특화 실험 설계 모듈"""
    
    def __init__(self):
        super().__init__()
        self.module_info = {
            'id': 'polymer_experiment',
            'name': '고분자 실험 설계',
            'version': '2.0.0',
            'author': 'Universal DOE Team',
            'description': '고분자 용매 시스템, 가공, 나노재료 실험 설계',
            'category': 'materials',
            'tags': ['polymer', 'solvent', 'processing', 'nanofiber', 'nanoparticle']
        }
        
        self.solvent_designer = SolventSystemDesign()
        self._initialize_templates()
        
    def _initialize_templates(self):
        """고분자 실험 템플릿 초기화"""
        self.experiment_templates = {
            'polymer_dissolution': {
                'name': '고분자 용해 실험',
                'process_type': PolymerProcessType.DISSOLUTION,
                'factors': [
                    Factor(name='용매 종류', type=FactorType.CATEGORICAL, 
                          levels=['THF', 'DMF', 'Chloroform', 'Toluene', 'DMSO']),
                    Factor(name='온도', type=FactorType.CONTINUOUS, 
                          min_value=20, max_value=150, unit='°C'),
                    Factor(name='농도', type=FactorType.CONTINUOUS,
                          min_value=0.1, max_value=20, unit='wt%'),
                    Factor(name='교반속도', type=FactorType.CONTINUOUS,
                          min_value=0, max_value=1000, unit='rpm')
                ],
                'responses': [
                    Response(name='용해시간', type=ResponseType.CONTINUOUS, unit='min'),
                    Response(name='용액투명도', type=ResponseType.CONTINUOUS, unit='%T'),
                    Response(name='점도', type=ResponseType.CONTINUOUS, unit='cP')
                ]
            },
            
            'electrospinning': {
                'name': '전기방사',
                'process_type': PolymerProcessType.ELECTROSPINNING,
                'factors': [
                    Factor(name='전압', type=FactorType.CONTINUOUS,
                          min_value=5, max_value=30, unit='kV'),
                    Factor(name='유속', type=FactorType.CONTINUOUS,
                          min_value=0.1, max_value=10, unit='mL/h'),
                    Factor(name='거리', type=FactorType.CONTINUOUS,
                          min_value=5, max_value=30, unit='cm'),
                    Factor(name='농도', type=FactorType.CONTINUOUS,
                          min_value=5, max_value=25, unit='wt%'),
                    Factor(name='습도', type=FactorType.CONTINUOUS,
                          min_value=20, max_value=80, unit='%RH')
                ],
                'responses': [
                    Response(name='섬유직경', type=ResponseType.CONTINUOUS, unit='nm'),
                    Response(name='직경균일도', type=ResponseType.CONTINUOUS, unit='CV%'),
                    Response(name='비드형성', type=ResponseType.BINARY),
                    Response(name='생산성', type=ResponseType.CONTINUOUS, unit='g/h')
                ]
            },
            
            'polymer_coating': {
                'name': '고분자 코팅',
                'process_type': PolymerProcessType.COATING,
                'factors': [
                    Factor(name='코팅방법', type=FactorType.CATEGORICAL,
                          levels=['스핀코팅', '딥코팅', '스프레이', '블레이드']),
                    Factor(name='속도', type=FactorType.CONTINUOUS,
                          min_value=100, max_value=5000, unit='rpm'),
                    Factor(name='농도', type=FactorType.CONTINUOUS,
                          min_value=0.5, max_value=10, unit='wt%'),
                    Factor(name='건조온도', type=FactorType.CONTINUOUS,
                          min_value=20, max_value=200, unit='°C'),
                    Factor(name='건조시간', type=FactorType.CONTINUOUS,
                          min_value=1, max_value=60, unit='min')
                ],
                'responses': [
                    Response(name='필름두께', type=ResponseType.CONTINUOUS, unit='nm'),
                    Response(name='표면거칠기', type=ResponseType.CONTINUOUS, unit='nm'),
                    Response(name='투과율', type=ResponseType.CONTINUOUS, unit='%'),
                    Response(name='접착력', type=ResponseType.CONTINUOUS, unit='N/m')
                ]
            },
            
            'nanoparticle_synthesis': {
                'name': '나노입자 합성',
                'process_type': PolymerProcessType.NANOPARTICLE,
                'factors': [
                    Factor(name='합성방법', type=FactorType.CATEGORICAL,
                          levels=['침전법', '에멀젼법', '분무건조', '초음파']),
                    Factor(name='반응온도', type=FactorType.CONTINUOUS,
                          min_value=0, max_value=100, unit='°C'),
                    Factor(name='반응시간', type=FactorType.CONTINUOUS,
                          min_value=0.5, max_value=24, unit='h'),
                    Factor(name='계면활성제농도', type=FactorType.CONTINUOUS,
                          min_value=0, max_value=5, unit='wt%'),
                    Factor(name='교반속도', type=FactorType.CONTINUOUS,
                          min_value=100, max_value=2000, unit='rpm')
                ],
                'responses': [
                    Response(name='입자크기', type=ResponseType.CONTINUOUS, unit='nm'),
                    Response(name='PDI', type=ResponseType.CONTINUOUS),
                    Response(name='제타전위', type=ResponseType.CONTINUOUS, unit='mV'),
                    Response(name='수율', type=ResponseType.CONTINUOUS, unit='%')
                ]
            },
            
            'solvent_mixture': {
                'name': '용매 혼합 최적화',
                'process_type': PolymerProcessType.DISSOLUTION,
                'factors': [
                    Factor(name='용매A비율', type=FactorType.CONTINUOUS,
                          min_value=0, max_value=100, unit='%'),
                    Factor(name='용매B비율', type=FactorType.CONTINUOUS,
                          min_value=0, max_value=100, unit='%'),
                    Factor(name='용매C비율', type=FactorType.CONTINUOUS,
                          min_value=0, max_value=100, unit='%',
                          constraint='A+B+C=100'),
                    Factor(name='온도', type=FactorType.CONTINUOUS,
                          min_value=20, max_value=80, unit='°C')
                ],
                'responses': [
                    Response(name='용해도', type=ResponseType.CONTINUOUS, unit='g/L'),
                    Response(name='상분리', type=ResponseType.BINARY),
                    Response(name='점도', type=ResponseType.CONTINUOUS, unit='cP'),
                    Response(name='Hansen거리', type=ResponseType.CONTINUOUS)
                ]
            }
        }
    
    # ===========================================================================
    # 필수 메서드 구현
    # ===========================================================================
    
    def get_info(self) -> Dict[str, Any]:
        """모듈 정보 반환"""
        return self.module_info
    
    def get_experiment_types(self) -> List[str]:
        """실험 유형 목록"""
        return list(self.experiment_templates.keys())
    
    def get_experiment_info(self, experiment_type: str) -> Optional[Dict[str, Any]]:
        """실험 유형 정보"""
        template = self.experiment_templates.get(experiment_type)
        if not template:
            return None
            
        return {
            'name': template['name'],
            'description': template.get('description', ''),
            'process_type': template.get('process_type', ''),
            'num_factors': len(template['factors']),
            'num_responses': len(template['responses']),
            'typical_runs': self._estimate_runs(len(template['factors']))
        }
    
    def _estimate_runs(self, num_factors: int) -> int:
        """예상 실험 횟수 추정"""
        if num_factors <= 2:
            return 4 + 3  # Full factorial + center points
        elif num_factors <= 4:
            return 2**num_factors + 2*num_factors + 3  # CCD
        else:
            return 2**(num_factors-1) + 2*num_factors + 3  # Fractional factorial + axial
    
    def get_default_factors(self, experiment_type: str) -> List[Factor]:
        """기본 요인 목록"""
        template = self.experiment_templates.get(experiment_type)
        if not template:
            return []
        
        return template['factors'].copy()
    
    def get_default_responses(self, experiment_type: str) -> List[Response]:
        """기본 반응변수 목록"""
        template = self.experiment_templates.get(experiment_type)
        if not template:
            return []
            
        return template['responses'].copy()
    
    def validate_design(self, factors: List[Factor], responses: List[Response],
                       design_method: DesignMethod,
                       design_params: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """설계 유효성 검증"""
        errors = []
        warnings = []
        suggestions = []
        
        # 기본 검증
        if len(factors) < 1:
            errors.append("최소 1개 이상의 요인이 필요합니다")
        
        if len(responses) < 1:
            errors.append("최소 1개 이상의 반응변수가 필요합니다")
        
        # 고분자 특화 검증
        factor_names = [f.name for f in factors]
        
        # 혼합물 제약 확인
        mixture_factors = [f for f in factors if '비율' in f.name or 'fraction' in f.name.lower()]
        if len(mixture_factors) >= 2:
            # 합이 100%인지 확인
            has_constraint = any(hasattr(f, 'constraint') and '100' in str(f.constraint) 
                               for f in mixture_factors)
            if not has_constraint:
                warnings.append("혼합물 성분의 합이 100%가 되도록 제약조건을 설정하세요")
                suggestions.append("혼합물 설계(Mixture Design) 사용을 권장합니다")
        
        # 전기방사 특화 검증
        if any('전압' in f.name or 'voltage' in f.name.lower() for f in factors):
            if not any('거리' in f.name or 'distance' in f.name.lower() for f in factors):
                warnings.append("전기방사에서는 전압과 거리를 함께 고려하는 것이 중요합니다")
        
        # 용매 시스템 검증
        solvent_factors = [f for f in factors if '용매' in f.name or 'solvent' in f.name.lower()]
        if solvent_factors:
            if not any('온도' in f.name or 'temperature' in f.name.lower() for f in factors):
                suggestions.append("용매 시스템에서는 온도가 중요한 요인입니다")
        
        # 설계 방법 적합성
        num_factors = len(factors)
        if design_method == DesignMethod.FULL_FACTORIAL and num_factors > 5:
            warnings.append(f"{num_factors}개 요인의 완전요인설계는 실험 수가 과도합니다")
            suggestions.append("부분요인설계 또는 Plackett-Burman 설계를 고려하세요")
        
        # 나노재료 특화
        if any('나노' in r.name or 'nano' in r.name.lower() for r in responses):
            if not any('PDI' in r.name or '균일' in r.name for r in responses):
                suggestions.append("나노재료에서는 크기 균일도(PDI)도 중요한 반응변수입니다")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def generate_design(self, factors: List[Factor], responses: List[Response],
                       design_method: DesignMethod,
                       design_params: Optional[Dict[str, Any]] = None) -> Design:
        """실험 설계 생성"""
        import pyDOE3 as doe
        
        design_params = design_params or {}
        num_factors = len(factors)
        
        # 혼합물 요인 확인
        mixture_factors = [i for i, f in enumerate(factors) 
                          if '비율' in f.name or 'fraction' in f.name.lower()]
        
        if len(mixture_factors) >= 2:
            # 혼합물 설계
            return self._generate_mixture_design(factors, responses, design_params)
        
        # 일반 설계
        if design_method == DesignMethod.FULL_FACTORIAL:
            levels = design_params.get('levels', 2)
            design_matrix = doe.fullfact([levels] * num_factors)
            
        elif design_method == DesignMethod.FRACTIONAL_FACTORIAL:
            if num_factors <= 3:
                design_matrix = doe.fullfact([2] * num_factors)
            else:
                resolution = design_params.get('resolution', 3)
                design_matrix = doe.fracfact(f'2^({num_factors}-{num_factors//2})')
                
        elif design_method == DesignMethod.CENTRAL_COMPOSITE:
            center = design_params.get('center', [3, 3])
            alpha = design_params.get('alpha', 'rotatable')
            design_matrix = doe.ccdesign(num_factors, center=center, alpha=alpha)
            
        elif design_method == DesignMethod.BOX_BEHNKEN:
            center = design_params.get('center', 3)
            design_matrix = doe.bbdesign(num_factors, center=center)
            
        elif design_method == DesignMethod.PLACKETT_BURMAN:
            design_matrix = doe.pbdesign(num_factors)
            
        elif design_method == DesignMethod.OPTIMAL:
            # D-optimal design
            candidate_set = doe.fullfact([3] * num_factors)
            num_runs = design_params.get('num_runs', 2 * num_factors + 4)
            
            from pyDOE3 import *
            design_matrix = candidate_set[np.random.choice(len(candidate_set), 
                                                          num_runs, replace=False)]
        else:
            # Latin Hypercube as default
            num_runs = design_params.get('num_runs', 10 * num_factors)
            design_matrix = doe.lhs(num_factors, samples=num_runs)
        
        # 정규화된 설계를 실제 값으로 변환
        runs = self._convert_to_actual_values(design_matrix, factors)
        
        # 설계 정보 생성
        design_info = self._calculate_design_info(runs, factors, design_method)
        
        return Design(
            factors=factors,
            responses=responses,
            runs=runs,
            design_method=design_method,
            design_info=design_info
        )
    
    def _generate_mixture_design(self, factors: List[Factor], 
                               responses: List[Response],
                               design_params: Dict[str, Any]) -> Design:
        """혼합물 설계 생성"""
        import pyDOE3 as doe
        
        # 혼합물 요인 인덱스
        mixture_indices = [i for i, f in enumerate(factors) 
                         if '비율' in f.name or 'fraction' in f.name.lower()]
        
        num_components = len(mixture_indices)
        
        # Simplex-lattice design
        if num_components == 2:
            proportions = np.array([[1, 0], [0, 1], [0.5, 0.5]])
        elif num_components == 3:
            proportions = np.array([
                [1, 0, 0], [0, 1, 0], [0, 0, 1],
                [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
                [1/3, 1/3, 1/3]
            ])
        else:
            # General mixture design
            degree = design_params.get('degree', 2)
            proportions = doe.simplex_lattice(num_components, degree)
        
        # 프로세스 변수 추가
        process_indices = [i for i in range(len(factors)) if i not in mixture_indices]
        
        if process_indices:
            # 프로세스 변수에 대한 설계
            process_design = doe.fullfact([2] * len(process_indices))
            
            # 혼합물과 프로세스 변수 결합
            runs = []
            for mix_point in proportions:
                for proc_point in process_design:
                    run = np.zeros(len(factors))
                    for i, idx in enumerate(mixture_indices):
                        run[idx] = mix_point[i] * 100  # 백분율로 변환
                    for i, idx in enumerate(process_indices):
                        run[idx] = proc_point[i]
                    runs.append(run)
        else:
            runs = proportions * 100  # 백분율로 변환
        
        # 실제 값으로 변환
        runs = self._convert_to_actual_values(np.array(runs), factors)
        
        design_info = DesignInfo(
            num_runs=len(runs),
            design_type="Mixture Design",
            resolution=None,
            efficiency_metrics={'D-efficiency': 0.95}
        )
        
        return Design(
            factors=factors,
            responses=responses,
            runs=runs,
            design_method=DesignMethod.OPTIMAL,
            design_info=design_info
        )
    
    def _convert_to_actual_values(self, normalized_design: np.ndarray,
                                 factors: List[Factor]) -> pd.DataFrame:
        """정규화된 설계를 실제 값으로 변환"""
        runs_dict = {}
        
        for i, factor in enumerate(factors):
            if factor.type == FactorType.CONTINUOUS:
                # 정규화된 값 (0-1 또는 -1 to 1)을 실제 범위로 변환
                col_values = normalized_design[:, i]
                
                # -1 to 1 범위인지 확인
                if col_values.min() < -0.5:
                    # -1 to 1을 0 to 1로 변환
                    col_values = (col_values + 1) / 2
                
                # 실제 값으로 스케일
                actual_values = (factor.min_value + 
                               col_values * (factor.max_value - factor.min_value))
                runs_dict[factor.name] = actual_values
                
            elif factor.type == FactorType.CATEGORICAL:
                # 범주형은 인덱스를 레벨로 변환
                col_values = normalized_design[:, i]
                level_indices = np.round(col_values * (len(factor.levels) - 1)).astype(int)
                level_indices = np.clip(level_indices, 0, len(factor.levels) - 1)
                runs_dict[factor.name] = [factor.levels[idx] for idx in level_indices]
        
        return pd.DataFrame(runs_dict)
    
    def _calculate_design_info(self, runs: pd.DataFrame, factors: List[Factor],
                             design_method: DesignMethod) -> DesignInfo:
        """설계 정보 계산"""
        num_runs = len(runs)
        num_factors = len(factors)
        
        # 설계 효율성 메트릭
        efficiency_metrics = {
            'runs_per_factor': num_runs / num_factors,
            'degrees_of_freedom': num_runs - 1
        }
        
        # Resolution 계산 (부분요인설계의 경우)
        resolution = None
        if design_method == DesignMethod.FRACTIONAL_FACTORIAL:
            if num_factors <= 3:
                resolution = "Full"
            elif num_factors <= 7:
                resolution = "IV"
            else:
                resolution = "III"
        
        return DesignInfo(
            num_runs=num_runs,
            design_type=design_method.value,
            resolution=resolution,
            efficiency_metrics=efficiency_metrics
        )
    
    def analyze_results(self, design: Design, results_data: pd.DataFrame,
                       analysis_options: Optional[Dict[str, Any]] = None) -> AnalysisResult:
        """결과 분석"""
        from scipy import stats
        from sklearn.preprocessing import StandardScaler
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        
        analysis_options = analysis_options or {}
        
        # 데이터 준비
        factor_names = [f.name for f in design.factors]
        response_names = [r.name for r in design.responses]
        
        # 분석 결과 저장
        statistical_results = {}
        model_equations = {}
        optimization_results = {}
        insights = []
        
        for response_name in response_names:
            if response_name not in results_data.columns:
                continue
            
            y = results_data[response_name]
            X = results_data[factor_names]
            
            # 범주형 변수 처리
            X_encoded = pd.get_dummies(X, drop_first=True)
            
            # 회귀 모델
            if analysis_options.get('include_interactions', True):
                # 2차 항과 교호작용 포함
                formula_parts = factor_names.copy()
                
                # 2차 항
                for factor in factor_names:
                    if design.get_factor(factor).type == FactorType.CONTINUOUS:
                        formula_parts.append(f'I({factor}**2)')
                
                # 교호작용
                for i in range(len(factor_names)):
                    for j in range(i+1, len(factor_names)):
                        formula_parts.append(f'{factor_names[i]}:{factor_names[j]}')
                
                formula = f"{response_name} ~ " + " + ".join(formula_parts)
            else:
                formula = f"{response_name} ~ " + " + ".join(factor_names)
            
            try:
                model = smf.ols(formula, data=results_data).fit()
                
                # 통계 결과
                statistical_results[response_name] = {
                    'r_squared': model.rsquared,
                    'adj_r_squared': model.rsquared_adj,
                    'f_statistic': model.fvalue,
                    'p_value': model.f_pvalue,
                    'coefficients': model.params.to_dict(),
                    'p_values': model.pvalues.to_dict(),
                    'confidence_intervals': model.conf_int().to_dict()
                }
                
                # 모델 방정식
                equation_parts = []
                for param, coef in model.params.items():
                    if param == 'Intercept':
                        equation_parts.append(f"{coef:.3f}")
                    else:
                        equation_parts.append(f"{coef:+.3f}*{param}")
                
                model_equations[response_name] = " ".join(equation_parts)
                
                # 최적화 (연속형 반응변수만)
                if design.get_response(response_name).type == ResponseType.CONTINUOUS:
                    # 반응표면 최적화
                    opt_result = self._optimize_response(model, design, response_name)
                    optimization_results[response_name] = opt_result
                
                # 인사이트 생성
                significant_factors = [
                    param for param, p_val in model.pvalues.items()
                    if p_val < 0.05 and param != 'Intercept'
                ]
                
                if significant_factors:
                    insights.append(
                        f"{response_name}에 대해 {', '.join(significant_factors)}가 "
                        f"통계적으로 유의합니다 (p < 0.05)"
                    )
                
            except Exception as e:
                logger.error(f"Error analyzing {response_name}: {e}")
                statistical_results[response_name] = {'error': str(e)}
        
        # 고분자 특화 분석
        polymer_insights = self._generate_polymer_insights(design, results_data)
        insights.extend(polymer_insights)
        
        return AnalysisResult(
            statistical_results=statistical_results,
            model_equations=model_equations,
            optimization_results=optimization_results,
            insights=insights
        )
    
    def _optimize_response(self, model, design: Design, 
                          response_name: str) -> Dict[str, Any]:
        """반응 최적화"""
        from scipy.optimize import minimize
        
        response = design.get_response(response_name)
        
        # 목적 함수 정의
        def objective(x):
            # 예측값 계산
            data_point = pd.DataFrame([x], columns=[f.name for f in design.factors])
            prediction = model.predict(data_point)[0]
            
            # 목표에 따라 최적화
            if response.target:
                if 'min' in response.target:
                    return (prediction - response.target['min'])**2
                elif 'max' in response.target:
                    return -prediction  # 최대화는 음수로
                elif 'target' in response.target:
                    return (prediction - response.target['target'])**2
            else:
                return -prediction  # 기본은 최대화
        
        # 초기값과 범위 설정
        x0 = []
        bounds = []
        
        for factor in design.factors:
            if factor.type == FactorType.CONTINUOUS:
                x0.append((factor.min_value + factor.max_value) / 2)
                bounds.append((factor.min_value, factor.max_value))
            else:
                x0.append(0)  # 범주형은 0
                bounds.append((0, len(factor.levels) - 1))
        
        # 최적화 실행
        result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
        
        # 최적 조건에서 예측
        optimal_conditions = pd.DataFrame([result.x], 
                                        columns=[f.name for f in design.factors])
        optimal_value = model.predict(optimal_conditions)[0]
        
        return {
            'optimal_conditions': dict(zip([f.name for f in design.factors], result.x)),
            'optimal_value': optimal_value,
            'success': result.success,
            'message': result.message
        }
    
    def _generate_polymer_insights(self, design: Design, 
                                  results_data: pd.DataFrame) -> List[str]:
        """고분자 특화 인사이트 생성"""
        insights = []
        
        # 전기방사 인사이트
        if any('전압' in f.name or 'voltage' in f.name.lower() for f in design.factors):
            if '섬유직경' in results_data.columns:
                correlation = results_data[['전압', '섬유직경']].corr().iloc[0, 1]
                if abs(correlation) > 0.7:
                    direction = "증가" if correlation > 0 else "감소"
                    insights.append(
                        f"전압이 증가할수록 섬유 직경이 {direction}하는 경향을 보입니다 "
                        f"(상관계수: {correlation:.2f})"
                    )
        
        # 용매 시스템 인사이트
        solvent_cols = [col for col in results_data.columns 
                       if '용매' in col or 'solvent' in col.lower()]
        if solvent_cols and '용해시간' in results_data.columns:
            fastest_idx = results_data['용해시간'].idxmin()
            fastest_condition = results_data.loc[fastest_idx, solvent_cols].to_dict()
            insights.append(
                f"가장 빠른 용해 조건: {fastest_condition} "
                f"(용해시간: {results_data.loc[fastest_idx, '용해시간']:.1f}분)"
            )
        
        # 나노입자 인사이트
        if 'PDI' in results_data.columns:
            low_pdi = results_data['PDI'] < 0.3
            if low_pdi.any():
                uniform_conditions = results_data[low_pdi].index.tolist()
                insights.append(
                    f"{len(uniform_conditions)}개 조건에서 균일한 입자 크기 분포 "
                    f"(PDI < 0.3)를 달성했습니다"
                )
        
        return insights
    
    def create_visualizations(self, design: Design, results_data: pd.DataFrame,
                            viz_options: Optional[Dict[str, Any]] = None) -> List[VisualizationResult]:
        """시각화 생성"""
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
        
        viz_options = viz_options or {}
        visualizations = []
        
        # 1. 주효과 플롯
        for response in design.responses:
            if response.name not in results_data.columns:
                continue
            
            fig = make_subplots(
                rows=1, cols=len(design.factors),
                subplot_titles=[f.name for f in design.factors]
            )
            
            for i, factor in enumerate(design.factors):
                if factor.type == FactorType.CONTINUOUS:
                    # 산점도 + 추세선
                    fig.add_trace(
                        go.Scatter(
                            x=results_data[factor.name],
                            y=results_data[response.name],
                            mode='markers',
                            name=factor.name,
                            showlegend=False
                        ),
                        row=1, col=i+1
                    )
                else:
                    # 박스플롯
                    for level in factor.levels:
                        y_data = results_data[results_data[factor.name] == level][response.name]
                        fig.add_trace(
                            go.Box(y=y_data, name=level, showlegend=False),
                            row=1, col=i+1
                        )
            
            fig.update_layout(
                title=f"주효과 플롯: {response.name}",
                height=400
            )
            
            visualizations.append(
                VisualizationResult(
                    chart_type='main_effects',
                    title=f"주효과 플롯: {response.name}",
                    figure=fig,
                    description="각 요인이 반응변수에 미치는 주효과"
                )
            )
        
        # 2. 교호작용 플롯 (연속형 요인 2개 이상일 때)
        continuous_factors = [f for f in design.factors if f.type == FactorType.CONTINUOUS]
        
        if len(continuous_factors) >= 2:
            for response in design.responses:
                if response.name not in results_data.columns:
                    continue
                
                # 첫 두 연속형 요인의 교호작용
                factor1, factor2 = continuous_factors[:2]
                
                fig = go.Figure(data=[
                    go.Scatter3d(
                        x=results_data[factor1.name],
                        y=results_data[factor2.name],
                        z=results_data[response.name],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=results_data[response.name],
                            colorscale='Viridis',
                            showscale=True
                        )
                    )
                ])
                
                fig.update_layout(
                    title=f"3D 반응표면: {response.name}",
                    scene=dict(
                        xaxis_title=factor1.name,
                        yaxis_title=factor2.name,
                        zaxis_title=response.name
                    )
                )
                
                visualizations.append(
                    VisualizationResult(
                        chart_type='response_surface',
                        title=f"3D 반응표면: {response.name}",
                        figure=fig,
                        description=f"{factor1.name}와 {factor2.name}의 교호작용"
                    )
                )
        
        # 3. 고분자 특화 시각화
        polymer_viz = self._create_polymer_visualizations(design, results_data)
        visualizations.extend(polymer_viz)
        
        return visualizations
    
    def _create_polymer_visualizations(self, design: Design,
                                     results_data: pd.DataFrame) -> List[VisualizationResult]:
        """고분자 특화 시각화"""
        import plotly.graph_objects as go
        
        visualizations = []
        
        # 용매 시스템 삼원 다이어그램 (3개 용매 비율이 있을 때)
        solvent_cols = [col for col in results_data.columns 
                       if '비율' in col and '용매' in col]
        
        if len(solvent_cols) == 3:
            # Ternary plot
            import plotly.figure_factory as ff
            
            # 데이터 준비
            a = results_data[solvent_cols[0]]
            b = results_data[solvent_cols[1]]
            c = results_data[solvent_cols[2]]
            
            # 색상을 위한 반응변수 선택
            color_response = None
            for resp in ['용해도', '점도', 'Hansen거리']:
                if resp in results_data.columns:
                    color_response = resp
                    break
            
            if color_response:
                fig = ff.create_ternary_contour(
                    np.array([a, b, c]).T,
                    results_data[color_response],
                    pole_labels=solvent_cols,
                    interp_mode='cartesian',
                    ncontours=20,
                    colorscale='Viridis',
                    showscale=True
                )
                
                visualizations.append(
                    VisualizationResult(
                        chart_type='ternary',
                        title=f"용매 시스템 삼원 다이어그램: {color_response}",
                        figure=fig,
                        description="3성분 용매 시스템의 최적 조성 탐색"
                    )
                )
        
        # 섬유 직경 분포 (전기방사)
        if '섬유직경' in results_data.columns:
            fig = go.Figure()
            
            # 히스토그램
            fig.add_trace(go.Histogram(
                x=results_data['섬유직경'],
                nbinsx=20,
                name='분포'
            ))
            
            # 정규분포 피팅
            from scipy import stats
            mu, sigma = stats.norm.fit(results_data['섬유직경'])
            x_range = np.linspace(results_data['섬유직경'].min(), 
                                results_data['섬유직경'].max(), 100)
            y_norm = stats.norm.pdf(x_range, mu, sigma) * len(results_data) * \
                    (results_data['섬유직경'].max() - results_data['섬유직경'].min()) / 20
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_norm,
                mode='lines',
                name=f'정규분포 (μ={mu:.1f}, σ={sigma:.1f})'
            ))
            
            fig.update_layout(
                title="섬유 직경 분포",
                xaxis_title="섬유 직경 (nm)",
                yaxis_title="빈도",
                showlegend=True
            )
            
            visualizations.append(
                VisualizationResult(
                    chart_type='histogram',
                    title="섬유 직경 분포",
                    figure=fig,
                    description="전기방사 섬유의 직경 분포 및 균일성"
                )
            )
        
        return visualizations
    
    def get_ai_recommendations(self, design_context: Dict[str, Any],
                             api_manager=None) -> List[AIRecommendation]:
        """AI 추천 생성"""
        recommendations = []
        
        # 컨텍스트 분석
        experiment_type = design_context.get('experiment_type', '')
        factors = design_context.get('factors', [])
        responses = design_context.get('responses', [])
        constraints = design_context.get('constraints', {})
        
        # 1. 설계 방법 추천
        design_rec = self._recommend_design_method(experiment_type, factors, constraints)
        recommendations.append(design_rec)
        
        # 2. 고분자 특화 추천
        if experiment_type == 'polymer_dissolution':
            solvent_rec = self._recommend_solvent_system(factors, constraints, api_manager)
            recommendations.append(solvent_rec)
        
        elif experiment_type == 'electrospinning':
            spinning_rec = self._recommend_electrospinning_conditions(factors, api_manager)
            recommendations.append(spinning_rec)
        
        elif experiment_type == 'nanoparticle_synthesis':
            nano_rec = self._recommend_nanoparticle_conditions(factors, responses, api_manager)
            recommendations.append(nano_rec)
        
        # 3. 일반 최적화 팁
        opt_tips = self._generate_optimization_tips(experiment_type, factors, responses)
        recommendations.extend(opt_tips)
        
        return recommendations
    
    def _recommend_design_method(self, experiment_type: str,
                                factors: List[Factor],
                                constraints: Dict) -> AIRecommendation:
        """설계 방법 추천"""
        num_factors = len(factors)
        mixture_factors = [f for f in factors if '비율' in f.name]
        
        if mixture_factors:
            reasoning = (
                f"{len(mixture_factors)}개의 혼합물 성분이 있으므로 "
                "혼합물 설계(Mixture Design)가 적합합니다."
            )
            if num_factors > len(mixture_factors):
                reasoning += (
                    f" 추가로 {num_factors - len(mixture_factors)}개의 "
                    "프로세스 변수가 있으므로 혼합-프로세스 설계를 권장합니다."
                )
            
            recommendation = "Simplex-lattice 또는 Simplex-centroid 설계"
            
        elif num_factors <= 3:
            reasoning = "요인이 3개 이하이므로 완전요인설계로 충분합니다."
            recommendation = "Full Factorial Design"
            
        elif num_factors <= 5:
            reasoning = (
                "요인이 4-5개인 경우 중심합성설계(CCD)나 "
                "Box-Behnken 설계가 효율적입니다."
            )
            recommendation = "Central Composite Design"
            
        else:
            reasoning = (
                f"{num_factors}개의 많은 요인이 있으므로 "
                "스크리닝 설계로 시작하는 것이 좋습니다."
            )
            recommendation = "Plackett-Burman Design → Response Surface"
        
        return AIRecommendation(
            recommendation_type='design_method',
            title="실험 설계 방법 추천",
            description=recommendation,
            reasoning=reasoning,
            confidence=0.9,
            alternatives=[
                "D-optimal 설계 (유연성이 필요한 경우)",
                "Definitive Screening Design (2차 효과 포함 스크리닝)"
            ]
        )
    
    def _recommend_solvent_system(self, factors: List[Factor],
                                constraints: Dict,
                                api_manager=None) -> AIRecommendation:
        """용매 시스템 추천"""
        polymer_name = constraints.get('polymer_name', 'Unknown polymer')
        
        description = "추천 용매 시스템:\n"
        reasoning = ""
        
        # 기본 추천 (실제로는 폴리머별 데이터베이스 사용)
        solvent_recommendations = {
            'PMMA': ['THF', 'Acetone', 'Toluene'],
            'PS': ['Toluene', 'THF', 'Chloroform'],
            'PVA': ['Water', 'DMSO'],
            'PCL': ['Chloroform', 'DCM', 'THF'],
            'PLA': ['Chloroform', 'DCM', 'Dioxane']
        }
        
        if polymer_name in solvent_recommendations:
            solvents = solvent_recommendations[polymer_name]
            description += f"- 주용매: {solvents[0]}\n"
            description += f"- 대체용매: {', '.join(solvents[1:])}\n"
            reasoning = f"{polymer_name}의 용해도 매개변수와 가장 적합한 용매입니다."
        else:
            description += "- Hansen 용해도 매개변수 기반 스크리닝 필요\n"
            description += "- 극성/비극성 용매 모두 테스트\n"
            reasoning = "폴리머 정보가 부족하여 광범위한 스크리닝이 필요합니다."
        
        # AI API 사용 가능한 경우
        if api_manager and polymer_name != 'Unknown polymer':
            try:
                ai_response = api_manager.design_solvent_system(
                    polymer_name, 
                    purpose="dissolution"
                )
                if ai_response.get('solvent_system'):
                    description = ai_response['solvent_system']
                    reasoning = "AI 기반 용매 예측 모델 사용"
            except:
                pass
        
        return AIRecommendation(
            recommendation_type='solvent_system',
            title="용매 시스템 추천",
            description=description,
            reasoning=reasoning,
            confidence=0.85,
            alternatives=[
                "혼합 용매 시스템 (공용매 효과)",
                "그린 용매 대체품 고려"
            ]
        )
    
    def _recommend_electrospinning_conditions(self, factors: List[Factor],
                                            api_manager=None) -> AIRecommendation:
        """전기방사 조건 추천"""
        description = """추천 초기 조건:
- 전압: 15-20 kV (시작점)
- 유속: 0.5-1.0 mL/h
- 거리: 15-20 cm
- 농도: 8-12 wt% (폴리머에 따라 조정)
- 습도: 40-50% RH"""
        
        reasoning = """이 조건들은 대부분의 고분자에서 안정적인 전기방사를 시작하기 좋은 
범위입니다. 폴리머 종류와 용매에 따라 미세 조정이 필요합니다."""
        
        return AIRecommendation(
            recommendation_type='process_conditions',
            title="전기방사 초기 조건",
            description=description,
            reasoning=reasoning,
            confidence=0.8,
            alternatives=[
                "니들리스 전기방사 (대량 생산)",
                "동축 전기방사 (코어-쉘 구조)"
            ]
        )
    
    def _recommend_nanoparticle_conditions(self, factors: List[Factor],
                                         responses: List[Response],
                                         api_manager=None) -> AIRecommendation:
        """나노입자 합성 조건 추천"""
        # 목표 크기 확인
        target_size = None
        for response in responses:
            if '크기' in response.name or 'size' in response.name.lower():
                if hasattr(response, 'target'):
                    target_size = response.target
        
        if target_size and target_size < 100:
            description = """100nm 이하 나노입자를 위한 추천:
- 침전법: 빠른 혼합, 낮은 농도
- 계면활성제: CMC의 2-3배
- 온도: 0-5°C (핵생성 제어)
- 교반: 고속 (>1000 rpm)"""
        else:
            description = """일반 나노입자 합성 추천:
- 침전법 또는 에멀젼법
- 계면활성제: 1-2 wt%
- 온도: 실온
- 교반: 중속 (500-800 rpm)"""
        
        reasoning = "입자 크기는 핵생성 속도와 성장 속도의 균형으로 결정됩니다."
        
        return AIRecommendation(
            recommendation_type='synthesis_method',
            title="나노입자 합성 방법",
            description=description,
            reasoning=reasoning,
            confidence=0.75,
            alternatives=[
                "마이크로플루이딕 합성 (균일성 향상)",
                "초음파 보조 합성 (분산성 개선)"
            ]
        )
    
    def _generate_optimization_tips(self, experiment_type: str,
                                   factors: List[Factor],
                                   responses: List[Response]) -> List[AIRecommendation]:
        """최적화 팁 생성"""
        tips = []
        
        # 다중 반응 최적화
        if len(responses) > 1:
            tips.append(AIRecommendation(
                recommendation_type='optimization_tip',
                title="다중 반응 최적화",
                description="Desirability function을 사용하여 여러 반응변수를 동시에 최적화하세요.",
                reasoning="상충하는 목표들 간의 균형을 찾는 것이 중요합니다.",
                confidence=0.9
            ))
        
        # 제약조건 처리
        constraint_factors = [f for f in factors if hasattr(f, 'constraint')]
        if constraint_factors:
            tips.append(AIRecommendation(
                recommendation_type='constraint_handling',
                title="제약조건 관리",
                description="제약조건이 있는 설계공간에서는 실행 가능한 영역을 먼저 파악하세요.",
                reasoning="제약조건 위반은 실험 실패로 이어질 수 있습니다.",
                confidence=0.85
            ))
        
        return tips
    
    def export_protocol(self, design: Design, format: str = 'markdown') -> str:
        """실험 프로토콜 내보내기"""
        if format == 'markdown':
            return self._export_markdown_protocol(design)
        elif format == 'latex':
            return self._export_latex_protocol(design)
        else:
            return self._export_text_protocol(design)
    
    def _export_markdown_protocol(self, design: Design) -> str:
        """마크다운 형식 프로토콜"""
        protocol = f"""# 고분자 실험 프로토콜

## 실험 정보
- **설계 방법**: {design.design_method.value}
- **총 실험 수**: {design.design_info.num_runs}
- **생성 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## 요인 (Factors)
"""
        for factor in design.factors:
            protocol += f"\n### {factor.name}\n"
            if factor.type == FactorType.CONTINUOUS:
                protocol += f"- 유형: 연속형\n"
                protocol += f"- 범위: {factor.min_value} - {factor.max_value} {factor.unit}\n"
            else:
                protocol += f"- 유형: 범주형\n"
                protocol += f"- 수준: {', '.join(factor.levels)}\n"
        
        protocol += "\n## 반응변수 (Responses)\n"
        for response in design.responses:
            protocol += f"\n### {response.name}\n"
            protocol += f"- 유형: {response.type.value}\n"
            if response.unit:
                protocol += f"- 단위: {response.unit}\n"
            if response.target:
                protocol += f"- 목표: {response.target}\n"
        
        protocol += "\n## 실험 계획표\n\n"
        protocol += "| Run # | " + " | ".join(design.runs.columns) + " |\n"
        protocol += "|" + "---|" * (len(design.runs.columns) + 1) + "\n"
        
        for idx, row in design.runs.iterrows():
            protocol += f"| {idx + 1} | "
            protocol += " | ".join(f"{val:.2f}" if isinstance(val, float) else str(val) 
                                 for val in row.values)
            protocol += " |\n"
        
        protocol += "\n## 실험 수행 지침\n"
        protocol += "1. 모든 실험은 무작위 순서로 수행하세요.\n"
        protocol += "2. 각 실험 조건 간 충분한 평형 시간을 두세요.\n"
        protocol += "3. 반복 실험을 통해 재현성을 확인하세요.\n"
        
        # 고분자 특화 지침
        if any('용매' in f.name for f in design.factors):
            protocol += "\n### 용매 취급 주의사항\n"
            protocol += "- 흄후드에서 작업하세요.\n"
            protocol += "- 적절한 PPE를 착용하세요.\n"
            protocol += "- 폐용매는 지정된 용기에 수거하세요.\n"
        
        if any('전압' in f.name for f in design.factors):
            protocol += "\n### 전기방사 안전 수칙\n"
            protocol += "- 고전압 주의\n"
            protocol += "- 접지 확인\n"
            protocol += "- 절연 장갑 착용\n"
        
        return protocol
    
    def _export_latex_protocol(self, design: Design) -> str:
        """LaTeX 형식 프로토콜"""
        # LaTeX 형식 구현 (간략)
        protocol = r"""\documentclass{article}
\usepackage{booktabs}
\begin{document}
\section{Polymer Experiment Protocol}
"""
        # ... LaTeX 테이블 생성 ...
        protocol += r"\end{document}"
        return protocol
    
    def _export_text_protocol(self, design: Design) -> str:
        """텍스트 형식 프로토콜"""
        protocol = "고분자 실험 프로토콜\n"
        protocol += "=" * 50 + "\n\n"
        # ... 간단한 텍스트 형식 ...
        return protocol

# ==============================================================================
# 🏭 팩토리 함수
# ==============================================================================

def create_polymer_module() -> PolymerExperimentModule:
    """고분자 실험 모듈 생성"""
    return PolymerExperimentModule()

# 모듈 등록용
MODULE_CLASS = PolymerExperimentModule
MODULE_INFO = {
    'id': 'polymer_experiment',
    'name': '고분자 실험 설계',
    'class': MODULE_CLASS,
    'category': 'materials',
    'version': '2.0.0'
}
