# 📚 사용자 정의 실험 모듈 개발 가이드

## 🎯 소개

Universal DOE Platform에 오신 것을 환영합니다! 이 가이드는 여러분이 자신만의 실험 설계 모듈을 만들어 플랫폼의 기능을 확장할 수 있도록 도와드립니다.

### 왜 커스텀 모듈인가?

- **특화된 실험 설계**: 여러분의 연구 분야에 특화된 실험 설계법 구현
- **커스터마이징**: 조직이나 연구실만의 고유한 프로토콜 반영
- **공유와 협업**: 다른 연구자들과 모듈 공유
- **지속적 개선**: 커뮤니티 피드백을 통한 모듈 개선

---

## 🚀 빠른 시작

### 1단계: 기본 모듈 구조

모든 커스텀 모듈은 `BaseExperimentModule`을 상속받아야 합니다:

```python
from modules.base_module import (
    BaseExperimentModule, Factor, Response, ExperimentDesign, 
    AnalysisResult, ValidationResult
)
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

class MyCustomModule(BaseExperimentModule):
    """나만의 실험 모듈"""
    
    def __init__(self):
        """모듈 초기화"""
        super().__init__()
        self.metadata.update({
            'name': '내 커스텀 실험 모듈',
            'version': '1.0.0',
            'author': '홍길동',
            'description': '특수한 용도의 실험 설계 모듈',
            'category': 'custom',
            'tags': ['polymer', 'optimization', 'custom'],
            'icon': '🔬',
            'color': '#FF6B6B'
        })
```

### 2단계: 필수 메서드 구현

모든 모듈은 다음 추상 메서드들을 반드시 구현해야 합니다:

```python
def get_factors(self) -> List[Factor]:
    """실험 요인 목록 반환"""
    return [
        Factor(
            name='온도',
            type='continuous',
            unit='°C',
            min_value=20,
            max_value=200,
            description='반응 온도'
        ),
        Factor(
            name='시간',
            type='continuous',
            unit='min',
            min_value=10,
            max_value=120,
            description='반응 시간'
        ),
        Factor(
            name='촉매',
            type='categorical',
            levels=['A', 'B', 'C'],
            description='촉매 종류'
        )
    ]

def get_responses(self) -> List[Response]:
    """반응변수 목록 반환"""
    return [
        Response(
            name='수율',
            unit='%',
            goal='maximize',
            target_value=95,
            lower_limit=0,
            upper_limit=100
        ),
        Response(
            name='순도',
            unit='%',
            goal='maximize',
            target_value=99,
            lower_limit=0,
            upper_limit=100
        )
    ]

def validate_input(self, inputs: Dict[str, Any]) -> ValidationResult:
    """입력값 검증"""
    result = ValidationResult()
    
    # 요인 검증
    factors = inputs.get('factors', [])
    if len(factors) < 2:
        result.add_error("최소 2개 이상의 요인이 필요합니다.")
    
    # 실험 횟수 검증
    n_runs = inputs.get('n_runs', 0)
    if n_runs < 4:
        result.add_warning("신뢰할 수 있는 결과를 위해 최소 4회 이상의 실험을 권장합니다.")
    
    return result

def generate_design(self, inputs: Dict[str, Any]) -> ExperimentDesign:
    """실험 설계 생성"""
    from pyDOE3 import fullfact, ccdesign, bbdesign
    
    design_type = inputs.get('design_type', 'full_factorial')
    factors = inputs.get('factors', [])
    
    # 설계 매트릭스 생성
    if design_type == 'full_factorial':
        levels = [f.get('levels', 2) for f in factors]
        design_matrix = fullfact(levels)
    elif design_type == 'central_composite':
        n_factors = len(factors)
        design_matrix = ccdesign(n_factors)
    else:
        # 기본값: Box-Behnken
        n_factors = len(factors)
        design_matrix = bbdesign(n_factors)
    
    # ExperimentDesign 객체 생성
    design = ExperimentDesign(
        design_type=design_type,
        factors=[Factor(**f) for f in factors],
        responses=self.get_responses(),
        runs=pd.DataFrame(design_matrix, columns=[f['name'] for f in factors]),
        metadata={'generated_by': self.metadata['name']}
    )
    
    return design

def analyze_results(self, design: ExperimentDesign, 
                   results_data: pd.DataFrame) -> AnalysisResult:
    """결과 분석"""
    import statsmodels.api as sm
    from scipy import stats
    
    # 기본 통계 분석
    summary_stats = results_data.describe()
    
    # 회귀 분석 (예시)
    X = design.runs
    y = results_data.iloc[:, 0]  # 첫 번째 반응변수
    
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()
    
    # 분석 결과 객체 생성
    analysis = AnalysisResult(
        summary_statistics=summary_stats.to_dict(),
        regression_results={
            'coefficients': model.params.to_dict(),
            'p_values': model.pvalues.to_dict(),
            'r_squared': model.rsquared
        },
        anova_results=None,  # 필요시 구현
        optimization_results=None,  # 필요시 구현
        predictions=None,  # 필요시 구현
        recommendations=["더 많은 중심점을 추가하여 곡률을 확인하세요."]
    )
    
    return analysis
```

---

## 📋 상세 개발 가이드

### 1. 데이터 모델 이해하기

#### Factor (요인) 클래스
```python
Factor(
    name: str,              # 요인 이름 (필수)
    type: str,              # 'continuous' 또는 'categorical' (필수)
    unit: Optional[str],    # 단위 (예: '°C', 'min', 'g/L')
    min_value: Optional[float],  # 최소값 (연속형)
    max_value: Optional[float],  # 최대값 (연속형)
    levels: Optional[List],      # 수준 목록 (범주형)
    description: Optional[str],  # 설명
    constraints: Optional[Dict]  # 제약조건
)
```

#### Response (반응변수) 클래스
```python
Response(
    name: str,              # 반응변수 이름 (필수)
    unit: Optional[str],    # 단위
    goal: str,              # 'maximize', 'minimize', 'target'
    target_value: Optional[float],  # 목표값
    lower_limit: Optional[float],   # 하한
    upper_limit: Optional[float],   # 상한
    weight: float = 1.0,    # 중요도 가중치
    description: Optional[str]      # 설명
)
```

#### ExperimentDesign (실험 설계) 클래스
```python
ExperimentDesign(
    design_type: str,       # 설계 유형
    factors: List[Factor],  # 요인 목록
    responses: List[Response],  # 반응변수 목록
    runs: pd.DataFrame,     # 실험 런 테이블
    metadata: Optional[Dict],   # 메타데이터
    constraints: Optional[Dict], # 제약조건
    blocks: Optional[List],     # 블록 정보
    center_points: int = 0      # 중심점 개수
)
```

### 2. 고급 기능 구현

#### 2.1 AI 통합
```python
def get_ai_recommendations(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """AI 기반 추천 생성"""
    from utils.api_manager import APIManager
    
    api_manager = APIManager()
    
    prompt = f"""
    다음 실험 조건에 대한 최적의 설계를 추천해주세요:
    - 요인: {inputs.get('factors')}
    - 목표: {inputs.get('objectives')}
    - 제약사항: {inputs.get('constraints')}
    
    다음을 포함해서 답변해주세요:
    1. 추천 설계 방법 및 이유
    2. 예상 실험 횟수
    3. 주의사항
    """
    
    response = api_manager.query_ai(prompt, service='google_gemini')
    
    return {
        'recommendations': response,
        'confidence': 0.85
    }
```

#### 2.2 사용자 정의 설계 방법
```python
def _custom_design_method(self, factors: List[Factor], 
                         n_runs: int) -> np.ndarray:
    """사용자 정의 설계 방법 구현"""
    # 여기에 독자적인 설계 알고리즘 구현
    # 예: 적응적 설계, 순차적 설계 등
    
    # D-optimal 설계 예시
    from pyDOE3 import doe_lhs
    
    n_factors = len(factors)
    # Latin Hypercube Sampling으로 초기 설계
    initial_design = doe_lhs.lhs(n_factors, samples=n_runs)
    
    # 설계 최적화 (D-optimality)
    optimized_design = self._optimize_design(initial_design, factors)
    
    return optimized_design
```

#### 2.3 시각화 통합
```python
def create_visualization(self, design: ExperimentDesign) -> Dict[str, Any]:
    """설계 시각화 생성"""
    import plotly.graph_objects as go
    
    figures = {}
    
    # 2D 설계 공간 (처음 두 요인)
    if len(design.factors) >= 2:
        factor1 = design.factors[0]
        factor2 = design.factors[1]
        
        fig = go.Figure(data=[go.Scatter(
            x=design.runs[factor1.name],
            y=design.runs[factor2.name],
            mode='markers',
            marker=dict(
                size=10,
                color='blue',
                line=dict(width=1, color='darkblue')
            ),
            text=[f"Run {i+1}" for i in range(len(design.runs))],
            hovertemplate='%{text}<br>%{x}<br>%{y}<extra></extra>'
        )])
        
        fig.update_layout(
            title='실험 설계 공간',
            xaxis_title=f"{factor1.name} ({factor1.unit})" if factor1.unit else factor1.name,
            yaxis_title=f"{factor2.name} ({factor2.unit})" if factor2.unit else factor2.name,
            hovermode='closest'
        )
        
        figures['design_space_2d'] = fig
    
    # 3D 설계 공간 (처음 세 요인)
    if len(design.factors) >= 3:
        factor3 = design.factors[2]
        
        fig3d = go.Figure(data=[go.Scatter3d(
            x=design.runs[factor1.name],
            y=design.runs[factor2.name],
            z=design.runs[factor3.name],
            mode='markers',
            marker=dict(size=8, color='red')
        )])
        
        fig3d.update_layout(
            title='3D 실험 설계 공간',
            scene=dict(
                xaxis_title=factor1.name,
                yaxis_title=factor2.name,
                zaxis_title=factor3.name
            )
        )
        
        figures['design_space_3d'] = fig3d
    
    return figures
```

### 3. 고분자 특화 모듈 예제

```python
class PolymerDissolutionModule(BaseExperimentModule):
    """고분자 용해 실험 모듈"""
    
    def __init__(self):
        super().__init__()
        self.metadata.update({
            'name': '고분자 용해 최적화',
            'version': '1.0.0',
            'author': '폴리머랩',
            'description': '고분자 용매 시스템 최적화를 위한 실험 설계',
            'category': 'polymer',
            'tags': ['polymer', 'dissolution', 'solvent', 'optimization'],
            'icon': '🧪',
            'color': '#9C27B0'
        })
    
    def get_factors(self) -> List[Factor]:
        """고분자 용해 관련 요인"""
        return [
            Factor(
                name='주용매',
                type='categorical',
                levels=['THF', 'CHCl3', 'DMF', 'DMSO'],
                description='주 용매 선택'
            ),
            Factor(
                name='보조용매_비율',
                type='continuous',
                unit='%',
                min_value=0,
                max_value=50,
                description='보조용매 혼합 비율'
            ),
            Factor(
                name='온도',
                type='continuous',
                unit='°C',
                min_value=20,
                max_value=80,
                description='용해 온도'
            ),
            Factor(
                name='교반속도',
                type='continuous',
                unit='rpm',
                min_value=100,
                max_value=1000,
                description='교반 속도'
            ),
            Factor(
                name='용해시간',
                type='continuous',
                unit='h',
                min_value=0.5,
                max_value=24,
                description='용해 시간'
            )
        ]
    
    def get_responses(self) -> List[Response]:
        """고분자 용해 관련 반응변수"""
        return [
            Response(
                name='용해도',
                unit='g/L',
                goal='maximize',
                target_value=100,
                lower_limit=0,
                description='고분자 용해도'
            ),
            Response(
                name='용액점도',
                unit='cP',
                goal='target',
                target_value=1000,
                lower_limit=100,
                upper_limit=5000,
                description='용액 점도'
            ),
            Response(
                name='상분리시간',
                unit='h',
                goal='maximize',
                lower_limit=24,
                description='용액 안정성 (상분리까지 시간)'
            )
        ]
    
    def validate_input(self, inputs: Dict[str, Any]) -> ValidationResult:
        """고분자 특화 검증"""
        result = super().validate_input(inputs)
        
        # 온도-용매 호환성 검증
        factors = {f['name']: f for f in inputs.get('factors', [])}
        
        if '주용매' in factors and '온도' in factors:
            solvent = factors['주용매'].get('selected_level')
            temp = factors['온도'].get('max_value', 0)
            
            # 용매별 끓는점 체크
            boiling_points = {
                'THF': 66,
                'CHCl3': 61,
                'DMF': 153,
                'DMSO': 189
            }
            
            if solvent in boiling_points and temp > boiling_points[solvent] - 10:
                result.add_warning(
                    f"{solvent}의 끓는점({boiling_points[solvent]}°C)에 가까운 온도입니다. "
                    "환류 냉각기 사용을 권장합니다."
                )
        
        return result
    
    def _suggest_solvent_system(self, polymer_type: str) -> List[str]:
        """고분자 종류에 따른 용매 시스템 제안"""
        solvent_database = {
            'PS': ['THF', 'toluene', 'CHCl3'],
            'PMMA': ['acetone', 'THF', 'CHCl3'],
            'PLA': ['CHCl3', 'DCM', 'dioxane'],
            'Nylon': ['formic acid', 'TFA', 'phenol'],
            'PEO': ['water', 'CHCl3', 'acetonitrile']
        }
        
        return solvent_database.get(polymer_type, ['THF', 'CHCl3', 'DMF'])
```

---

## 🔒 보안 가이드라인

### 금지된 작업

커스텀 모듈에서는 다음 작업이 제한됩니다:

```python
# ❌ 금지된 예시
import os
os.system('command')  # 시스템 명령 실행 금지

import subprocess  # 프로세스 생성 금지

eval('code')  # 동적 코드 실행 금지
exec('code')

# 파일 시스템 직접 접근 제한
open('/etc/passwd', 'r')
```

### 허용된 작업

```python
# ✅ 허용된 예시
import numpy as np  # 수치 계산
import pandas as pd  # 데이터 처리
import scipy  # 과학 계산
from pyDOE3 import *  # 실험 설계

# 플랫폼 제공 API 사용
from utils.data_processor import DataProcessor
from utils.api_manager import APIManager
```

---

## 🧪 테스트 가이드

### 단위 테스트 작성

```python
# test_my_module.py

import unittest
import pandas as pd
import numpy as np
from modules.user_modules.my_custom_module import MyCustomModule

class TestMyCustomModule(unittest.TestCase):
    
    def setUp(self):
        """테스트 준비"""
        self.module = MyCustomModule()
        self.test_inputs = {
            'design_type': 'full_factorial',
            'factors': [
                {
                    'name': '온도',
                    'type': 'continuous',
                    'min_value': 20,
                    'max_value': 100,
                    'levels': 3
                },
                {
                    'name': '시간',
                    'type': 'continuous',
                    'min_value': 10,
                    'max_value': 60,
                    'levels': 3
                }
            ],
            'responses': [
                {'name': '수율', 'goal': 'maximize'}
            ],
            'n_runs': 9
        }
    
    def test_initialization(self):
        """모듈 초기화 테스트"""
        self.assertIsNotNone(self.module.metadata)
        self.assertEqual(self.module.metadata['name'], '내 커스텀 실험 모듈')
        self.assertIn('version', self.module.metadata)
    
    def test_factor_generation(self):
        """요인 생성 테스트"""
        factors = self.module.get_factors()
        self.assertIsInstance(factors, list)
        self.assertGreater(len(factors), 0)
        
        # 첫 번째 요인 검증
        first_factor = factors[0]
        self.assertIn('name', first_factor.__dict__)
        self.assertIn('type', first_factor.__dict__)
    
    def test_input_validation(self):
        """입력 검증 테스트"""
        # 정상 입력
        result = self.module.validate_input(self.test_inputs)
        self.assertTrue(result.is_valid)
        
        # 비정상 입력 (요인 부족)
        invalid_inputs = {
            'factors': [{'name': '온도', 'type': 'continuous'}],
            'n_runs': 2
        }
        result = self.module.validate_input(invalid_inputs)
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.errors), 0)
    
    def test_design_generation(self):
        """설계 생성 테스트"""
        design = self.module.generate_design(self.test_inputs)
        
        self.assertIsNotNone(design)
        self.assertIsInstance(design.runs, pd.DataFrame)
        self.assertEqual(len(design.runs), 9)  # 3x3 완전요인설계
        self.assertEqual(len(design.runs.columns), 2)  # 2개 요인
    
    def test_results_analysis(self):
        """결과 분석 테스트"""
        # 설계 생성
        design = self.module.generate_design(self.test_inputs)
        
        # 가상의 결과 데이터 생성
        np.random.seed(42)
        results_data = pd.DataFrame({
            '수율': np.random.normal(80, 10, len(design.runs))
        })
        
        # 분석 실행
        analysis = self.module.analyze_results(design, results_data)
        
        self.assertIsNotNone(analysis)
        self.assertIn('summary_statistics', analysis.dict())
        self.assertIn('regression_results', analysis.dict())
    
    def test_edge_cases(self):
        """엣지 케이스 테스트"""
        # 빈 입력
        empty_result = self.module.validate_input({})
        self.assertFalse(empty_result.is_valid)
        
        # 극단적인 값
        extreme_inputs = {
            'factors': [
                {
                    'name': 'A',
                    'type': 'continuous',
                    'min_value': -1000,
                    'max_value': 1000
                }
            ] * 10,  # 10개 요인
            'n_runs': 1000  # 많은 실험 횟수
        }
        result = self.module.validate_input(extreme_inputs)
        self.assertGreater(len(result.warnings), 0)

if __name__ == '__main__':
    unittest.main()
```

### 통합 테스트

플랫폼과의 통합을 테스트하려면:

1. 모듈을 `modules/user_modules/{your_user_id}/` 폴더에 저장
2. 플랫폼을 재시작하거나 핫 리로드 기능 사용
3. UI에서 모듈이 나타나는지 확인
4. 실제 실험 설계 수행 테스트

---

## 📦 배포 및 공유

### 1. 로컬 설치

```bash
# 모듈 파일을 사용자 모듈 디렉토리에 복사
cp my_custom_module.py ~/UniversalDOE/modules/user_modules/{user_id}/
```

### 2. 모듈 패키징

```yaml
# module.yaml
name: My Custom Module
version: 1.0.0
author: Hong Gil Dong
email: hong@example.com
files:
  - my_custom_module.py
  - helpers.py
  - data/templates.json
dependencies:
  - numpy>=1.20.0
  - scipy>=1.7.0
  - pyDOE3>=1.0.0
test_files:
  - test_my_module.py
documentation:
  - README.md
  - examples/
license: MIT
```

### 3. 마켓플레이스 업로드

1. 플랫폼 UI에서 "모듈 업로드" 선택
2. 모듈 파일 및 메타데이터 제공
3. 자동 검증 통과
4. 커뮤니티 리뷰 기간 (선택적)
5. 공개 및 공유

---

## 💡 모범 사례

### 1. 명확한 문서화

```python
def my_method(self, param1: float, param2: str) -> Dict[str, Any]:
    """
    메서드의 기능을 명확히 설명
    
    Parameters
    ----------
    param1 : float
        첫 번째 매개변수 설명 (범위: 0-100)
    param2 : str
        두 번째 매개변수 설명 (옵션: 'A', 'B', 'C')
        
    Returns
    -------
    Dict[str, Any]
        반환값 설명
        - 'result': 계산 결과 (float)
        - 'status': 상태 코드 (str)
        
    Raises
    ------
    ValueError
        param1이 범위를 벗어난 경우
        
    Examples
    --------
    >>> module.my_method(50.0, "A")
    {'result': 100.0, 'status': 'success'}
    """
    # 구현
    pass
```

### 2. 강건한 에러 처리

```python
def risky_operation(self, data: pd.DataFrame) -> Optional[Dict]:
    """위험할 수 있는 작업 수행"""
    try:
        # 입력 검증
        if data.empty:
            self.logger.warning("빈 데이터프레임이 입력되었습니다.")
            return None
        
        # 계산 수행
        result = self._complex_calculation(data)
        
        # 결과 검증
        if not self._validate_result(result):
            raise ValueError("계산 결과가 유효하지 않습니다.")
        
        return result
        
    except ValueError as e:
        # 예상된 에러 처리
        self.logger.error(f"계산 오류: {str(e)}")
        return None
        
    except Exception as e:
        # 예상치 못한 에러
        self.logger.error(f"예상치 못한 오류: {str(e)}")
        raise
```

### 3. 성능 최적화

```python
def process_large_dataset(self, data: pd.DataFrame, chunk_size: int = 1000):
    """대용량 데이터셋 처리"""
    # 메모리 효율적인 청크 처리
    results = []
    
    with tqdm(total=len(data), desc="데이터 처리 중") as pbar:
        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i:i+chunk_size]
            
            # 청크별 처리
            chunk_result = self._process_chunk(chunk)
            results.append(chunk_result)
            
            # 진행률 업데이트
            pbar.update(len(chunk))
            
            # 메모리 정리
            if i % (chunk_size * 10) == 0:
                import gc
                gc.collect()
    
    # 결과 통합
    return pd.concat(results, ignore_index=True)
```

### 4. 설정 가능한 모듈

```python
class ConfigurableModule(BaseExperimentModule):
    """설정 가능한 모듈 예제"""
    
    def __init__(self):
        super().__init__()
        # 기본 설정
        self.config = {
            'default_design_type': 'central_composite',
            'min_runs': 4,
            'max_runs': 1000,
            'optimization_algorithm': 'sequential',
            'confidence_level': 0.95,
            'allow_constraints': True,
            'enable_ai_assist': True
        }
    
    def update_config(self, new_config: Dict[str, Any]):
        """설정 업데이트"""
        # 유효한 설정만 업데이트
        valid_keys = set(self.config.keys())
        for key, value in new_config.items():
            if key in valid_keys:
                self.config[key] = value
                self.logger.info(f"설정 업데이트: {key} = {value}")
            else:
                self.logger.warning(f"알 수 없는 설정 키: {key}")
    
    def get_config_schema(self) -> Dict[str, Any]:
        """설정 스키마 반환 (UI 생성용)"""
        return {
            'default_design_type': {
                'type': 'select',
                'options': ['full_factorial', 'central_composite', 'box_behnken'],
                'default': 'central_composite',
                'description': '기본 실험 설계 방법'
            },
            'confidence_level': {
                'type': 'slider',
                'min': 0.8,
                'max': 0.99,
                'step': 0.01,
                'default': 0.95,
                'description': '통계적 신뢰 수준'
            }
        }
```

---

## 🆘 문제 해결

### 자주 발생하는 문제

1. **ModuleNotFoundError**
   ```python
   # 문제: ImportError: No module named 'my_helper'
   # 해결: 상대 경로 사용
   from .my_helper import helper_function  # 동일 디렉토리
   ```

2. **ValidationError**
   ```python
   # 문제: "필수 메서드 'get_factors'가 구현되지 않았습니다"
   # 해결: 모든 추상 메서드 구현 확인
   def get_factors(self) -> List[Factor]:
       return []  # 최소한 빈 리스트라도 반환
   ```

3. **SecurityError**
   ```python
   # 문제: "금지된 모듈 'os'를 import할 수 없습니다"
   # 해결: 플랫폼 제공 API 사용
   from utils.file_manager import save_file  # os 대신 사용
   ```

### 디버깅 팁

```python
# 디버그 모드 활성화
import logging
logging.basicConfig(level=logging.DEBUG)

# 모듈 내부에서 로깅
class MyModule(BaseExperimentModule):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.debug("모듈 초기화됨")
    
    def generate_design(self, inputs: Dict[str, Any]) -> ExperimentDesign:
        self.logger.debug(f"입력값: {inputs}")
        
        # 중간 결과 출력
        factors = inputs.get('factors', [])
        self.logger.info(f"요인 개수: {len(factors)}")
        
        # 조건부 디버깅
        if self.config.get('debug_mode', False):
            import pdb; pdb.set_trace()  # 개발 중에만 사용
```

---

## 📚 추가 리소스

### 예제 모듈

1. **[기본 예제](examples/basic_module.py)** - 간단한 2요인 실험
2. **[고급 예제](examples/advanced_module.py)** - 복잡한 제약조건 처리
3. **[고분자 예제](examples/polymer_module.py)** - 고분자 특화 기능
4. **[AI 통합 예제](examples/ai_integrated_module.py)** - AI 추천 시스템

### 참고 문서

- [BaseExperimentModule API 문서](../base_module.py)
- [데이터 모델 상세 설명](../../docs/data_models.md)
- [플랫폼 API 레퍼런스](../../docs/api_reference.md)
- [실험 설계 이론](../../docs/doe_theory.md)

### 유용한 라이브러리

- **[pyDOE3](https://github.com/relf/pyDOE3)** - 실험 설계 생성
- **[statsmodels](https://www.statsmodels.org/)** - 통계 분석
- **[scikit-learn](https://scikit-learn.org/)** - 머신러닝
- **[plotly](https://plotly.com/python/)** - 대화형 시각화

### 커뮤니티

- **질문과 답변**: [포럼](https://forum.universaldoe.com)
- **버그 리포트**: [GitHub Issues](https://github.com/universaldoe/issues)
- **모듈 공유**: [Module Marketplace](https://universaldoe.com/marketplace)
- **개발자 채팅**: [Discord](https://discord.gg/universaldoe)

---

## 🎉 축하합니다!

이제 여러분은 Universal DOE Platform을 위한 커스텀 모듈을 만들 준비가 되었습니다. 
여러분의 창의적인 아이디어와 전문 지식이 전 세계 연구자들에게 도움이 될 것입니다.

**궁금한 점이 있으신가요?** 
- 📧 이메일: support@universaldoe.com
- 💬 실시간 채팅: 플랫폼 내 지원 버튼

**Happy Experimenting! 🧪✨**
