```markdown
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
from modules.base_module import BaseExperimentModule, Factor, Response, ExperimentDesign, ValidationResult

class MyCustomModule(BaseExperimentModule):
    """나만의 실험 모듈"""
    
    def _initialize(self):
        """모듈 초기화"""
        self.metadata = {
            'name': '내 커스텀 실험 모듈',
            'version': '1.0.0',
            'author': '홍길동',
            'description': '특수한 용도의 실험 설계 모듈',
            'category': 'custom',
            'tags': ['polymer', 'optimization', 'custom']
        }
```

### 2단계: 필수 메서드 구현

```python
def get_experiment_types(self) -> List[str]:
    """지원하는 실험 유형 목록"""
    return ['타입1', '타입2', '타입3']

def get_factors(self, experiment_type: str) -> List[Factor]:
    """실험 유형별 요인 목록"""
    return [
        Factor(
            name='온도',
            type='continuous',
            unit='°C',
            min_value=20,
            max_value=200
        ),
        # 더 많은 요인들...
    ]

def get_responses(self, experiment_type: str) -> List[Response]:
    """실험 유형별 반응변수 목록"""
    return [
        Response(
            name='수율',
            unit='%',
            goal='maximize'
        ),
        # 더 많은 반응변수들...
    ]
```

---

## 📋 상세 개발 가이드

### 1. 모듈 메타데이터 정의

메타데이터는 모듈의 신원과 특성을 정의합니다:

```python
self.metadata = {
    'name': str,              # 모듈 이름 (필수)
    'version': str,           # 시맨틱 버전 (필수) 예: '1.0.0'
    'author': str,            # 작성자 이름 (필수)
    'email': str,             # 연락처 이메일 (선택)
    'description': str,       # 모듈 설명 (필수)
    'category': str,          # 카테고리 (필수)
    'subcategory': str,       # 세부 카테고리 (선택)
    'tags': List[str],        # 검색 태그 (선택)
    'documentation_url': str, # 문서 링크 (선택)
    'license': str,           # 라이선스 (선택) 예: 'MIT'
    'dependencies': List[str] # 필요한 패키지 (선택)
}
```

### 2. 데이터 모델 활용

#### Factor (요인) 정의

```python
from modules.base_module import Factor

# 연속형 요인
temperature = Factor(
    name='반응온도',
    type='continuous',
    unit='°C',
    min_value=20.0,
    max_value=100.0,
    description='반응이 일어나는 온도'
)

# 범주형 요인
catalyst = Factor(
    name='촉매종류',
    type='categorical',
    levels=['A', 'B', 'C', 'None'],
    description='사용할 촉매 종류'
)

# 이산형 요인 (정수값)
cycles = Factor(
    name='반복횟수',
    type='continuous',  # 이산형도 continuous로 처리
    unit='회',
    min_value=1,
    max_value=10,
    description='공정 반복 횟수'
)
```

#### Response (반응변수) 정의

```python
from modules.base_module import Response

# 최대화 목표
yield_response = Response(
    name='수율',
    unit='%',
    goal='maximize',
    description='생성물의 수율'
)

# 최소화 목표
cost_response = Response(
    name='비용',
    unit='원/kg',
    goal='minimize',
    description='단위 생산 비용'
)

# 목표값 달성
purity_response = Response(
    name='순도',
    unit='%',
    goal='target',
    target_value=99.5,
    description='목표 순도 달성'
)
```

### 3. 실험 설계 생성

#### 기본 구현 예제

```python
def generate_design(self, inputs: Dict[str, Any]) -> ExperimentDesign:
    """실험 설계 생성"""
    
    # 입력값에서 정보 추출
    design_type = inputs.get('design_type', 'full_factorial')
    factors = [Factor(**f) for f in inputs['factors']]
    responses = [Response(**r) for r in inputs['responses']]
    
    # 설계 매트릭스 생성 (예: 2수준 완전요인설계)
    if design_type == 'full_factorial':
        import numpy as np
        from pyDOE3 import ff2n
        
        # 연속형 요인만 추출
        continuous_factors = [f for f in factors if f.type == 'continuous']
        n_factors = len(continuous_factors)
        
        # 코드화된 설계 매트릭스 생성 (-1, 1)
        coded_matrix = ff2n(n_factors)
        
        # 실제 값으로 변환
        runs_data = {}
        for i, factor in enumerate(continuous_factors):
            low = factor.min_value
            high = factor.max_value
            coded_values = coded_matrix[:, i]
            real_values = low + (coded_values + 1) / 2 * (high - low)
            runs_data[factor.name] = real_values
        
        # DataFrame 생성
        runs_df = pd.DataFrame(runs_data)
        
        # 범주형 요인 처리
        categorical_factors = [f for f in factors if f.type == 'categorical']
        for factor in categorical_factors:
            # 예: 랜덤 할당 (실제로는 더 정교한 방법 사용)
            n_runs = len(runs_df)
            runs_df[factor.name] = np.random.choice(factor.levels, n_runs)
        
        # 반응변수 열 추가 (빈 값)
        for response in responses:
            runs_df[response.name] = np.nan
        
        # 실험 순서 랜덤화
        runs_df = runs_df.sample(frac=1).reset_index(drop=True)
        runs_df.index = range(1, len(runs_df) + 1)
        runs_df.index.name = 'Run'
        
        return ExperimentDesign(
            design_type=design_type,
            runs=runs_df,
            factors=factors,
            responses=responses,
            metadata={
                'created_by': self.__class__.__name__,
                'design_info': {
                    'n_runs': len(runs_df),
                    'n_factors': len(factors),
                    'n_responses': len(responses)
                }
            }
        )
```

### 4. 입력값 검증

```python
def validate_input(self, inputs: Dict[str, Any]) -> ValidationResult:
    """입력값 검증"""
    result = ValidationResult(is_valid=True)
    
    # 필수 필드 확인
    if 'factors' not in inputs or not inputs['factors']:
        result.is_valid = False
        result.errors.append("최소 1개 이상의 요인이 필요합니다")
    
    # 요인 검증
    if 'factors' in inputs:
        for i, factor in enumerate(inputs['factors']):
            # 이름 확인
            if not factor.get('name'):
                result.errors.append(f"요인 {i+1}의 이름이 없습니다")
                
            # 범위 확인 (연속형)
            if factor.get('type') == 'continuous':
                min_val = factor.get('min_value', 0)
                max_val = factor.get('max_value', 1)
                if min_val >= max_val:
                    result.errors.append(
                        f"요인 '{factor.get('name')}'의 최소값이 최대값보다 크거나 같습니다"
                    )
            
            # 수준 확인 (범주형)
            elif factor.get('type') == 'categorical':
                if not factor.get('levels') or len(factor['levels']) < 2:
                    result.errors.append(
                        f"범주형 요인 '{factor.get('name')}'은 최소 2개 이상의 수준이 필요합니다"
                    )
    
    # 실험 횟수 경고
    if 'design_type' in inputs:
        estimated_runs = self._estimate_runs(inputs)
        if estimated_runs > 100:
            result.warnings.append(
                f"예상 실험 횟수가 {estimated_runs}개로 많습니다. "
                "실험 계획을 재검토해보세요."
            )
    
    # 개선 제안
    if len(inputs.get('factors', [])) > 7:
        result.suggestions.append(
            "요인이 7개를 초과합니다. 스크리닝 설계를 먼저 수행하여 "
            "중요한 요인을 선별하는 것을 권장합니다."
        )
    
    return result
```

### 5. 결과 분석

```python
def analyze_results(self, design: ExperimentDesign, 
                   data: pd.DataFrame) -> Dict[str, Any]:
    """실험 결과 분석"""
    analysis = {
        'summary': {},
        'factor_effects': {},
        'models': {},
        'optimization': {}
    }
    
    # 기본 통계
    for response in design.responses:
        if response.name in data.columns:
            response_data = data[response.name].dropna()
            
            analysis['summary'][response.name] = {
                'n': len(response_data),
                'mean': response_data.mean(),
                'std': response_data.std(),
                'min': response_data.min(),
                'max': response_data.max(),
                'cv': response_data.std() / response_data.mean() * 100
            }
    
    # 주효과 분석 (연속형 요인)
    continuous_factors = [f for f in design.factors if f.type == 'continuous']
    
    for response in design.responses:
        if response.name not in data.columns:
            continue
            
        effects = {}
        for factor in continuous_factors:
            if factor.name in data.columns:
                # 간단한 상관관계 (실제로는 더 정교한 분석 필요)
                correlation = data[factor.name].corr(data[response.name])
                effects[factor.name] = {
                    'correlation': correlation,
                    'significance': abs(correlation) > 0.5  # 단순 기준
                }
        
        analysis['factor_effects'][response.name] = effects
    
    # 회귀 모델 (예시)
    try:
        from sklearn.linear_model import LinearRegression
        
        X = data[[f.name for f in continuous_factors]].dropna()
        
        for response in design.responses:
            if response.name in data.columns:
                y = data[response.name].dropna()
                
                # 인덱스 정렬
                common_idx = X.index.intersection(y.index)
                X_fit = X.loc[common_idx]
                y_fit = y.loc[common_idx]
                
                if len(X_fit) > len(continuous_factors):
                    model = LinearRegression()
                    model.fit(X_fit, y_fit)
                    
                    analysis['models'][response.name] = {
                        'coefficients': dict(zip(X_fit.columns, model.coef_)),
                        'intercept': model.intercept_,
                        'r_squared': model.score(X_fit, y_fit)
                    }
    except Exception as e:
        analysis['models']['error'] = str(e)
    
    return analysis
```

---

## 🧪 고급 기능

### 1. AI 통합

```python
def get_ai_recommendations(self, context: Dict) -> Dict:
    """AI 기반 추천 제공"""
    # API Manager를 통해 AI 호출
    from utils.api_manager import get_api_manager
    
    api_manager = get_api_manager()
    
    prompt = f"""
    다음 실험 설계에 대한 추천을 제공해주세요:
    - 분야: {context.get('field')}
    - 목적: {context.get('objective')}
    - 제약사항: {context.get('constraints')}
    
    추천 사항:
    1. 적절한 실험 설계 방법
    2. 예상 실험 횟수
    3. 주의사항
    """
    
    response = api_manager.query_ai(prompt, service='google_gemini')
    
    return {
        'recommendations': response,
        'confidence': 0.85
    }
```

### 2. 사용자 정의 설계 방법

```python
def _custom_design_method(self, factors: List[Factor], 
                         n_runs: int) -> np.ndarray:
    """사용자 정의 설계 방법 구현"""
    # 여기에 독자적인 설계 알고리즘 구현
    # 예: 적응적 설계, 순차적 설계 등
    pass
```

### 3. 시각화 통합

```python
def create_visualization(self, design: ExperimentDesign) -> Dict:
    """설계 시각화 생성"""
    import plotly.graph_objects as go
    
    # 2D/3D 산점도, 평행좌표 플롯 등
    figures = {}
    
    # 예: 2요인 설계 공간
    if len(design.factors) >= 2:
        factor1 = design.factors[0]
        factor2 = design.factors[1]
        
        fig = go.Figure(data=[go.Scatter(
            x=design.runs[factor1.name],
            y=design.runs[factor2.name],
            mode='markers',
            marker=dict(size=10)
        )])
        
        fig.update_layout(
            title='실험 설계 공간',
            xaxis_title=factor1.name,
            yaxis_title=factor2.name
        )
        
        figures['design_space'] = fig
    
    return figures
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
from utils.data_processor import process_data
```

---

## 🧑‍🔬 테스트 가이드

### 단위 테스트 작성

```python
# test_my_module.py

import unittest
from modules.user_modules.my_custom_module import MyCustomModule

class TestMyCustomModule(unittest.TestCase):
    
    def setUp(self):
        self.module = MyCustomModule()
    
    def test_initialization(self):
        """모듈 초기화 테스트"""
        self.assertIsNotNone(self.module.metadata)
        self.assertEqual(self.module.metadata['name'], '내 커스텀 실험 모듈')
    
    def test_factor_generation(self):
        """요인 생성 테스트"""
        factors = self.module.get_factors('타입1')
        self.assertGreater(len(factors), 0)
        self.assertIsInstance(factors[0], Factor)
    
    def test_design_generation(self):
        """설계 생성 테스트"""
        inputs = {
            'design_type': 'full_factorial',
            'factors': [
                {'name': '온도', 'type': 'continuous', 
                 'min_value': 20, 'max_value': 100}
            ],
            'responses': [
                {'name': '수율', 'goal': 'maximize'}
            ]
        }
        
        design = self.module.generate_design(inputs)
        self.assertIsInstance(design, ExperimentDesign)
        self.assertGreater(len(design.runs), 0)
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
files:
  - my_custom_module.py
  - helpers.py
  - data/templates.json
dependencies:
  - numpy>=1.20.0
  - scipy>=1.7.0
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
def my_method(self, param1: float, param2: str) -> Dict:
    """
    메서드의 기능을 명확히 설명
    
    Parameters
    ----------
    param1 : float
        첫 번째 매개변수 설명
    param2 : str
        두 번째 매개변수 설명
        
    Returns
    -------
    Dict
        반환값 설명
        
    Examples
    --------
    >>> module.my_method(1.0, "test")
    {'result': 'success'}
    """
```

### 2. 에러 처리

```python
def risky_operation(self):
    try:
        # 위험한 작업
        result = complex_calculation()
    except ValueError as e:
        # 구체적인 에러 처리
        self.logger.error(f"계산 오류: {str(e)}")
        return None
    except Exception as e:
        # 일반적인 에러 처리
        self.logger.error(f"예상치 못한 오류: {str(e)}")
        raise
    
    return result
```

### 3. 성능 최적화

```python
# 큰 데이터셋 처리 시 청크 사용
def process_large_dataset(self, data: pd.DataFrame):
    chunk_size = 1000
    results = []
    
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i+chunk_size]
        result = self._process_chunk(chunk)
        results.append(result)
    
    return pd.concat(results)
```

---

## 🆘 문제 해결

### 자주 발생하는 문제

1. **ModuleNotFoundError**
   - 원인: 의존성 패키지 누락
   - 해결: requirements.txt에 패키지 추가

2. **ValidationError**
   - 원인: 필수 메서드 미구현
   - 해결: BaseExperimentModule의 모든 추상 메서드 구현

3. **SecurityError**
   - 원인: 제한된 작업 시도
   - 해결: 보안 가이드라인 준수

### 디버깅 팁

```python
# 디버그 모드 활성화
import logging
logging.basicConfig(level=logging.DEBUG)

# 중간 결과 출력
print(f"Debug: factors = {factors}")
print(f"Debug: design matrix shape = {matrix.shape}")
```

---

## 📚 추가 리소스

### 예제 모듈

- [기본 예제](examples/basic_module.py)
- [고급 예제](examples/advanced_module.py)
- [특수 분야 예제](examples/specialized_module.py)

### 참고 문서

- [BaseExperimentModule API 문서](../base_module.py)
- [플랫폼 API 레퍼런스](../../docs/api_reference.md)
- [실험 설계 이론](../../docs/doe_theory.md)

### 커뮤니티

- 질문과 답변: [wjdwognsz@gmail.com](wjdwognsz@gmail.com)
- 버그 리포트: [wjdwognsz@gmail.com](wjdwognsz@gmail.com)
- 모듈 공유: [Module Marketplace](https://universaldoe.com/marketplace)

---

## 🎉 축하합니다!

이제 여러분은 Universal DOE Platform을 위한 커스텀 모듈을 만들 준비가 되었습니다. 
여러분의 창의적인 아이디어와 전문 지식이 전 세계 연구자들에게 도움이 될 것입니다.

**Happy Experimenting! 🧪✨**
```

끝났습니다.
