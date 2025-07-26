"""
🤖 Universal DOE Platform - API 통합 관리자
================================================================================
6개 AI 엔진과 과학 데이터베이스를 통합 관리하는 핵심 모듈
오프라인 우선 설계, 캐싱, 폴백 메커니즘 제공
================================================================================
"""

import os
import json
import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from functools import wraps, lru_cache
from enum import Enum
import hashlib
import threading
from pathlib import Path
import base64
from cryptography.fernet import Fernet

# 서드파티 라이브러리
import streamlit as st
import pandas as pd
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
import aiohttp
import httpx

# AI 라이브러리 - 선택적 import
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("Google Gemini API 라이브러리가 설치되지 않았습니다.")

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI API 라이브러리가 설치되지 않았습니다.")

try:
    from groq import AsyncGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logging.warning("Groq API 라이브러리가 설치되지 않았습니다.")

try:
    from transformers import pipeline
    import torch
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    logging.warning("HuggingFace 라이브러리가 설치되지 않았습니다.")

# 과학 데이터베이스 라이브러리
try:
    from mp_api.client import MPRester
    MATERIALS_PROJECT_AVAILABLE = True
except ImportError:
    MATERIALS_PROJECT_AVAILABLE = False
    logging.warning("Materials Project API 라이브러리가 설치되지 않았습니다.")

try:
    import pubchempy as pcp
    PUBCHEM_AVAILABLE = True
except ImportError:
    PUBCHEM_AVAILABLE = False
    logging.warning("PubChemPy 라이브러리가 설치되지 않았습니다.")

try:
    from github import Github
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False
    logging.warning("PyGithub 라이브러리가 설치되지 않았습니다.")

# 로컬 모듈 임포트
try:
    from config.app_config import (
        API_CONFIG, FILE_PROCESSING, PROTOCOL_EXTRACTION,
        get_config
    )
    from config.secrets_config import API_KEY_STRUCTURE
except ImportError:
    # 기본값 설정
    API_CONFIG = {
        'google_gemini': {
            'name': 'Google Gemini 2.0 Flash',
            'model': 'gemini-2.0-flash-exp',
            'required': True,
            'free_tier': True,
            'rate_limit': 60,
            'max_tokens': 1048576
        }
    }
    FILE_PROCESSING = {
        'allowed_extensions': {
            'document': ['.pdf', '.docx', '.doc', '.txt', '.rtf'],
            'markup': ['.html', '.htm', '.md', '.xml'],
            'data': ['.json', '.csv']
        }
    }

# ===========================================================================
# 🔧 로깅 설정
# ===========================================================================

logger = logging.getLogger(__name__)

# ===========================================================================
# 📌 상수 및 Enum 정의
# ===========================================================================

class AIEngineType(Enum):
    """AI 엔진 타입"""
    GEMINI = "gemini"
    GROK = "grok"
    GROQ = "groq"
    SAMBANOVA = "sambanova"
    DEEPSEEK = "deepseek"
    HUGGINGFACE = "huggingface"

class ResponseStatus(Enum):
    """API 응답 상태"""
    SUCCESS = "success"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    CACHED = "cached"
    OFFLINE = "offline"

# Rate Limit 설정
RATE_LIMITS = {
    AIEngineType.GEMINI: {'calls': 60, 'period': 60},  # 60 calls/min
    AIEngineType.GROQ: {'calls': 100, 'period': 60},   # 100 calls/min
    AIEngineType.HUGGINGFACE: {'calls': 1000, 'period': 3600},  # 1000 calls/hour
    'default': {'calls': 30, 'period': 60}
}

# 캐시 TTL 설정 (초)
CACHE_TTL = {
    'ai_response': 3600,      # 1시간
    'material_data': 86400,   # 24시간
    'compound_data': 86400,   # 24시간
    'protocol': 7200,         # 2시간
    'default': 1800           # 30분
}

# API 비용 추정 (1K 토큰당 USD)
API_COSTS = {
    AIEngineType.GEMINI: {'input': 0.0, 'output': 0.0},  # 무료
    AIEngineType.GROQ: {'input': 0.0, 'output': 0.0},    # 무료
    AIEngineType.DEEPSEEK: {'input': 0.001, 'output': 0.002},
    AIEngineType.SAMBANOVA: {'input': 0.0, 'output': 0.0},  # 무료 티어
    'default': {'input': 0.001, 'output': 0.002}
}

# ===========================================================================
# 🔧 프로토콜 추출 프롬프트 템플릿
# ===========================================================================

EXTRACTION_PROMPTS = {
    'pdf_academic': """
학술 논문 PDF에서 실험 프로토콜을 추출합니다.
Methods/Experimental/Materials and Methods 섹션을 중점적으로 분석하세요.

추출해야 할 정보:
1. 재료 및 시약 (이름, 순도, 공급업체)
2. 장비 및 도구 (모델명, 제조사)
3. 실험 조건 (온도, 압력, 시간, pH 등)
4. 실험 절차 (단계별 상세 설명)
5. 주의사항 및 안전 정보

JSON 형식으로 구조화하여 반환하세요.
""",
    
    'text_protocol': """
일반 텍스트에서 실험 절차를 식별합니다.
번호나 불릿으로 구분된 단계를 찾고, 재료와 조건을 구분하세요.

추출 포맷:
{
  "materials": [...],
  "equipment": [...],
  "conditions": {...},
  "procedure": [...],
  "safety": [...]
}
""",
    
    'html_webpage': """
웹페이지에서 프로토콜 정보를 추출합니다.
구조화된 리스트, 테이블, 또는 단계별 설명을 찾으세요.
네비게이션이나 광고 같은 무관한 내용은 제외하세요.
""",
    
    'mixed_format': """
다양한 형식이 혼재된 텍스트를 분석합니다.
재료, 조건, 절차를 구분하여 추출하고, 논리적 순서로 정리하세요.
수량, 단위, 시간 정보를 정확히 파악하세요.
"""
}

# ===========================================================================
# 📊 데이터 클래스
# ===========================================================================

@dataclass
class APIResponse:
    """API 응답 데이터 클래스"""
    status: ResponseStatus
    data: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    cached: bool = False
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class UsageRecord:
    """API 사용량 기록"""
    user_id: str
    api_type: str
    timestamp: datetime
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    success: bool = True
    error: Optional[str] = None

# ===========================================================================
# 🔐 암호화 관리자
# ===========================================================================

class EncryptionManager:
    """API 키 암호화 관리"""
    
    def __init__(self):
        self.key = self._get_or_create_key()
        self.cipher = Fernet(self.key)
    
    def _get_or_create_key(self) -> bytes:
        """암호화 키 가져오기 또는 생성"""
        key_file = Path.home() / '.universaldoe' / 'api_key.key'
        key_file.parent.mkdir(parents=True, exist_ok=True)
        
        if key_file.exists():
            return key_file.read_bytes()
        else:
            key = Fernet.generate_key()
            key_file.write_bytes(key)
            return key
    
    def encrypt(self, value: str) -> str:
        """문자열 암호화"""
        return self.cipher.encrypt(value.encode()).decode()
    
    def decrypt(self, encrypted: str) -> str:
        """문자열 복호화"""
        return self.cipher.decrypt(encrypted.encode()).decode()

# ===========================================================================
# ⏱️ Rate Limiter
# ===========================================================================

class RateLimiter:
    """API Rate Limiting"""
    
    def __init__(self, max_calls: int, period: int):
        self.max_calls = max_calls
        self.period = period  # 초
        self.calls = deque()
        self.lock = threading.Lock()
    
    def check(self) -> bool:
        """호출 가능 여부 확인"""
        with self.lock:
            now = time.time()
            # 기간이 지난 호출 제거
            while self.calls and self.calls[0] < now - self.period:
                self.calls.popleft()
            
            return len(self.calls) < self.max_calls
    
    def record(self):
        """호출 기록"""
        with self.lock:
            self.calls.append(time.time())
    
    def get_remaining(self) -> Dict[str, Any]:
        """남은 호출 수 반환"""
        with self.lock:
            now = time.time()
            while self.calls and self.calls[0] < now - self.period:
                self.calls.popleft()
            
            return {
                'remaining': self.max_calls - len(self.calls),
                'reset_in': int(self.period - (now - self.calls[0])) if self.calls else 0
            }

# ===========================================================================
# 💾 캐시 시스템
# ===========================================================================

class APICache:
    """API 응답 캐싱"""
    
    def __init__(self, ttl: int = 3600):
        self.ttl = ttl
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.lock = threading.Lock()
    
    def _make_key(self, prefix: str, params: Dict) -> str:
        """캐시 키 생성"""
        param_str = json.dumps(params, sort_keys=True)
        return f"{prefix}:{hashlib.md5(param_str.encode()).hexdigest()}"
    
    def get(self, prefix: str, params: Dict) -> Optional[Any]:
        """캐시에서 가져오기"""
        with self.lock:
            key = self._make_key(prefix, params)
            if key in self.cache:
                data, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    return data
                else:
                    del self.cache[key]
            return None
    
    def set(self, prefix: str, params: Dict, data: Any):
        """캐시에 저장"""
        with self.lock:
            key = self._make_key(prefix, params)
            self.cache[key] = (data, time.time())
    
    def clear(self, prefix: Optional[str] = None):
        """캐시 초기화"""
        with self.lock:
            if prefix:
                keys_to_remove = [k for k in self.cache.keys() if k.startswith(prefix)]
                for key in keys_to_remove:
                    del self.cache[key]
            else:
                self.cache.clear()

# ===========================================================================
# 📊 사용량 추적
# ===========================================================================

class UsageTracker:
    """API 사용량 추적"""
    
    def __init__(self):
        self.records: List[UsageRecord] = []
        self.lock = threading.Lock()
    
    def record(self, record: UsageRecord):
        """사용량 기록"""
        with self.lock:
            self.records.append(record)
            # 7일 이상 된 기록 제거
            cutoff = datetime.now() - timedelta(days=7)
            self.records = [r for r in self.records if r.timestamp > cutoff]
    
    def get_summary(self, user_id: Optional[str] = None, 
                   period: str = 'day') -> Dict[str, Any]:
        """사용량 요약"""
        with self.lock:
            # 기간 설정
            if period == 'day':
                start = datetime.now() - timedelta(days=1)
            elif period == 'week':
                start = datetime.now() - timedelta(days=7)
            elif period == 'month':
                start = datetime.now() - timedelta(days=30)
            else:
                start = datetime.min
            
            # 필터링
            filtered = [r for r in self.records if r.timestamp >= start]
            if user_id:
                filtered = [r for r in filtered if r.user_id == user_id]
            
            # 집계
            summary = defaultdict(lambda: {
                'calls': 0,
                'tokens': 0,
                'cost': 0.0,
                'errors': 0
            })
            
            for record in filtered:
                api_summary = summary[record.api_type]
                api_summary['calls'] += 1
                api_summary['tokens'] += record.tokens_used or 0
                api_summary['cost'] += record.cost or 0
                if not record.success:
                    api_summary['errors'] += 1
            
            return dict(summary)

# ===========================================================================
# 🤖 AI 엔진 기본 클래스
# ===========================================================================

class BaseAIEngine:
    """AI 엔진 추상 클래스"""
    
    def __init__(self, engine_type: AIEngineType, model_name: str):
        self.engine_type = engine_type
        self.model_name = model_name
        self.api_key = self._get_api_key()
        self.is_available = self._check_availability()
        
        # Rate Limiter
        limits = RATE_LIMITS.get(engine_type, RATE_LIMITS['default'])
        self.rate_limiter = RateLimiter(limits['calls'], limits['period'])
        
        # Cache
        ttl = CACHE_TTL.get('ai_response', CACHE_TTL['default'])
        self.cache = APICache(ttl)
        
        # 비용 정보
        self.costs = API_COSTS.get(engine_type, API_COSTS['default'])
    
    def _get_api_key(self) -> Optional[str]:
        """API 키 가져오기"""
        # 1. 세션 상태 확인
        if hasattr(st, 'session_state') and 'api_keys' in st.session_state:
            key = st.session_state.api_keys.get(self.engine_type.value)
            if key:
                return key
        
        # 2. Streamlit secrets 확인
        if hasattr(st, 'secrets'):
            try:
                key = st.secrets.get(f"{self.engine_type.value}_api_key")
                if key:
                    return key
            except:
                pass
        
        # 3. 환경 변수 확인
        env_key = f"{self.engine_type.value.upper()}_API_KEY"
        return os.getenv(env_key)
    
    def _check_availability(self) -> bool:
        """사용 가능 여부 확인"""
        return bool(self.api_key)
    
    async def generate(self, prompt: str, user_id: str = "anonymous", 
                      **kwargs) -> APIResponse:
        """텍스트 생성 (구현 필요)"""
        raise NotImplementedError
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """비용 추정"""
        input_cost = (input_tokens / 1000) * self.costs['input']
        output_cost = (output_tokens / 1000) * self.costs['output']
        return input_cost + output_cost

# ===========================================================================
# 🌟 Google Gemini 엔진
# ===========================================================================

class GeminiEngine(BaseAIEngine):
    """Google Gemini AI 엔진"""
    
    def __init__(self):
        super().__init__(AIEngineType.GEMINI, "gemini-2.0-flash-exp")
        if self.is_available and GEMINI_AVAILABLE:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            self.chat = None
    
    async def generate(self, prompt: str, user_id: str = "anonymous", 
                      **kwargs) -> APIResponse:
        """Gemini로 텍스트 생성"""
        # 캐시 확인
        cache_params = {'prompt': prompt, **kwargs}
        cached = self.cache.get('gemini', cache_params)
        if cached:
            return APIResponse(
                status=ResponseStatus.CACHED,
                data=cached,
                cached=True
            )
        
        # Rate limit 확인
        if not self.rate_limiter.check():
            return APIResponse(
                status=ResponseStatus.RATE_LIMITED,
                error="Rate limit exceeded"
            )
        
        try:
            # 동기 호출을 비동기로 변환
            response = await asyncio.to_thread(
                self.model.generate_content, prompt
            )
            
            # Rate limit 기록
            self.rate_limiter.record()
            
            # 응답 처리
            result = response.text
            
            # 캐시 저장
            self.cache.set('gemini', cache_params, result)
            
            # 사용량 기록 (추후 토큰 계산 추가)
            # tokens = response.usage_metadata...
            
            return APIResponse(
                status=ResponseStatus.SUCCESS,
                data=result,
                metadata={'model': self.model_name}
            )
            
        except Exception as e:
            logger.error(f"Gemini 에러: {str(e)}")
            return APIResponse(
                status=ResponseStatus.ERROR,
                error=str(e)
            )
    
    async def extract_protocol(self, text: str, file_type: str = "mixed_format",
                             user_id: str = "anonymous") -> APIResponse:
        """프로토콜 추출 특화 기능"""
        # 파일 타입별 프롬프트 선택
        base_prompt = EXTRACTION_PROMPTS.get(file_type, EXTRACTION_PROMPTS['mixed_format'])
        
        # 전체 프롬프트 구성
        full_prompt = f"""
{base_prompt}

텍스트:
{text[:10000]}  # 길이 제한

반드시 다음 JSON 형식으로만 응답하세요:
{{
  "materials": [
    {{"name": "재료명", "amount": "양", "purity": "순도", "supplier": "공급업체"}}
  ],
  "equipment": [
    {{"name": "장비명", "model": "모델", "manufacturer": "제조사"}}
  ],
  "conditions": {{
    "temperature": "온도",
    "pressure": "압력",
    "time": "시간",
    "ph": "pH",
    "other": {{}}
  }},
  "procedure": [
    {{"step": 1, "action": "동작", "details": "상세 설명", "duration": "소요 시간"}}
  ],
  "safety": ["주의사항1", "주의사항2"]
}}
"""
        
        response = await self.generate(full_prompt, user_id)
        
        if response.status == ResponseStatus.SUCCESS:
            try:
                # JSON 파싱
                protocol_data = json.loads(response.data)
                response.data = protocol_data
            except json.JSONDecodeError:
                # JSON 파싱 실패 시 텍스트 그대로 반환
                logger.warning("프로토콜 JSON 파싱 실패")
        
        return response

# ===========================================================================
# ⚡ Groq 엔진
# ===========================================================================

class GroqEngine(BaseAIEngine):
    """Groq 초고속 추론 엔진"""
    
    def __init__(self):
        super().__init__(AIEngineType.GROQ, "llama-3.3-70b-versatile")
        if self.is_available and GROQ_AVAILABLE:
            self.client = AsyncGroq(api_key=self.api_key)
    
    async def generate(self, prompt: str, user_id: str = "anonymous", 
                      **kwargs) -> APIResponse:
        """Groq로 텍스트 생성"""
        # 캐시 확인
        cache_params = {'prompt': prompt, **kwargs}
        cached = self.cache.get('groq', cache_params)
        if cached:
            return APIResponse(
                status=ResponseStatus.CACHED,
                data=cached,
                cached=True
            )
        
        # Rate limit 확인
        if not self.rate_limiter.check():
            return APIResponse(
                status=ResponseStatus.RATE_LIMITED,
                error="Rate limit exceeded"
            )
        
        try:
            # API 호출
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert in materials science and chemistry."},
                    {"role": "user", "content": prompt}
                ],
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 4096)
            )
            
            # Rate limit 기록
            self.rate_limiter.record()
            
            # 응답 처리
            result = response.choices[0].message.content
            
            # 캐시 저장
            self.cache.set('groq', cache_params, result)
            
            # 사용량 기록
            if hasattr(response, 'usage'):
                tokens = response.usage.total_tokens
                # 토큰 기록...
            
            return APIResponse(
                status=ResponseStatus.SUCCESS,
                data=result,
                metadata={'model': self.model_name}
            )
            
        except Exception as e:
            logger.error(f"Groq 에러: {str(e)}")
            return APIResponse(
                status=ResponseStatus.ERROR,
                error=str(e)
            )

# ===========================================================================
# 🤗 HuggingFace 엔진
# ===========================================================================

class HuggingFaceEngine(BaseAIEngine):
    """HuggingFace 특수 모델 엔진"""
    
    def __init__(self):
        super().__init__(AIEngineType.HUGGINGFACE, "microsoft/BioGPT-Large")
        if self.is_available and HUGGINGFACE_AVAILABLE:
            # 로컬 모델 또는 API 사용
            self.use_local = kwargs.get('use_local', False)
            if self.use_local:
                self._init_local_model()
            else:
                self._init_api_client()
    
    def _init_local_model(self):
        """로컬 모델 초기화"""
        try:
            self.pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            logger.error(f"로컬 모델 로드 실패: {e}")
            self.is_available = False
    
    def _init_api_client(self):
        """HuggingFace API 클라이언트 초기화"""
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
    
    async def generate(self, prompt: str, user_id: str = "anonymous", 
                      **kwargs) -> APIResponse:
        """HuggingFace로 텍스트 생성"""
        # 캐시 확인
        cache_params = {'prompt': prompt, **kwargs}
        cached = self.cache.get('huggingface', cache_params)
        if cached:
            return APIResponse(
                status=ResponseStatus.CACHED,
                data=cached,
                cached=True
            )
        
        try:
            if self.use_local and hasattr(self, 'pipeline'):
                # 로컬 추론
                result = await asyncio.to_thread(
                    self.pipeline,
                    prompt,
                    max_length=kwargs.get('max_length', 512),
                    temperature=kwargs.get('temperature', 0.7)
                )
                text = result[0]['generated_text']
            else:
                # API 호출
                async with aiohttp.ClientSession() as session:
                    payload = {
                        "inputs": prompt,
                        "parameters": {
                            "max_length": kwargs.get('max_length', 512),
                            "temperature": kwargs.get('temperature', 0.7)
                        }
                    }
                    async with session.post(
                        self.api_url,
                        headers=self.headers,
                        json=payload
                    ) as response:
                        result = await response.json()
                        text = result[0]['generated_text']
            
            # 캐시 저장
            self.cache.set('huggingface', cache_params, text)
            
            return APIResponse(
                status=ResponseStatus.SUCCESS,
                data=text,
                metadata={'model': self.model_name}
            )
            
        except Exception as e:
            logger.error(f"HuggingFace 에러: {str(e)}")
            return APIResponse(
                status=ResponseStatus.ERROR,
                error=str(e)
            )

# ===========================================================================
# 🔄 OpenAI 호환 엔진 (Grok, DeepSeek, SambaNova)
# ===========================================================================

class OpenAICompatibleEngine(BaseAIEngine):
    """OpenAI API 호환 엔진"""
    
    def __init__(self, engine_type: AIEngineType, model_name: str, 
                 base_url: str):
        super().__init__(engine_type, model_name)
        self.base_url = base_url
        if self.is_available and OPENAI_AVAILABLE:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=base_url
            )
    
    async def generate(self, prompt: str, user_id: str = "anonymous", 
                      **kwargs) -> APIResponse:
        """OpenAI 호환 API로 텍스트 생성"""
        # 캐시 확인
        cache_params = {'prompt': prompt, **kwargs}
        cached = self.cache.get(self.engine_type.value, cache_params)
        if cached:
            return APIResponse(
                status=ResponseStatus.CACHED,
                data=cached,
                cached=True
            )
        
        # Rate limit 확인
        if not self.rate_limiter.check():
            return APIResponse(
                status=ResponseStatus.RATE_LIMITED,
                error="Rate limit exceeded"
            )
        
        try:
            # API 호출
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 4096)
            )
            
            # Rate limit 기록
            self.rate_limiter.record()
            
            # 응답 처리
            result = response.choices[0].message.content
            
            # 캐시 저장
            self.cache.set(self.engine_type.value, cache_params, result)
            
            # 사용량 기록
            if hasattr(response, 'usage'):
                tokens = response.usage.total_tokens
                cost = self.estimate_cost(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens
                )
                # 기록...
            
            return APIResponse(
                status=ResponseStatus.SUCCESS,
                data=result,
                metadata={'model': self.model_name}
            )
            
        except Exception as e:
            logger.error(f"{self.engine_type.value} 에러: {str(e)}")
            return APIResponse(
                status=ResponseStatus.ERROR,
                error=str(e)
            )

# ===========================================================================
# 🔬 과학 데이터베이스 클라이언트
# ===========================================================================

class ScienceDBClient:
    """과학 데이터베이스 통합 클라이언트"""
    
    def __init__(self):
        self.clients = {}
        self.cache = APICache(CACHE_TTL.get('material_data', 86400))
        self._init_clients()
    
    def _init_clients(self):
        """클라이언트 초기화"""
        # Materials Project
        if MATERIALS_PROJECT_AVAILABLE:
            mp_key = self._get_api_key('materials_project')
            if mp_key:
                self.clients['materials_project'] = MPRester(mp_key)
        
        # PubChem (API 키 불필요)
        if PUBCHEM_AVAILABLE:
            self.clients['pubchem'] = pcp
        
        # GitHub
        if GITHUB_AVAILABLE:
            github_token = self._get_api_key('github')
            if github_token:
                self.clients['github'] = Github(github_token)
    
    def _get_api_key(self, service: str) -> Optional[str]:
        """API 키 가져오기"""
        # 세션, secrets, 환경변수 순으로 확인
        if hasattr(st, 'session_state') and 'api_keys' in st.session_state:
            key = st.session_state.api_keys.get(service)
            if key:
                return key
        
        if hasattr(st, 'secrets'):
            try:
                key = st.secrets.get(f"{service}_api_key")
                if key:
                    return key
            except:
                pass
        
        return os.getenv(f"{service.upper()}_API_KEY")
    
    async def search_materials(self, formula: Optional[str] = None,
                             **criteria) -> List[Dict]:
        """재료 검색"""
        if 'materials_project' not in self.clients:
            return []
        
        # 캐시 확인
        cache_params = {'formula': formula, **criteria}
        cached = self.cache.get('materials', cache_params)
        if cached:
            return cached
        
        try:
            mp = self.clients['materials_project']
            # 동기 호출을 비동기로 변환
            results = await asyncio.to_thread(
                mp.materials.search,
                formula=formula,
                **criteria
            )
            
            # 결과 변환
            materials = []
            for material in results:
                materials.append({
                    'material_id': material.material_id,
                    'formula': material.formula,
                    'energy': material.energy_per_atom,
                    'band_gap': material.band_gap,
                    'density': material.density,
                    'crystal_system': material.symmetry.crystal_system
                })
            
            # 캐시 저장
            self.cache.set('materials', cache_params, materials)
            
            return materials
            
        except Exception as e:
            logger.error(f"Materials Project 검색 에러: {str(e)}")
            return []
    
    async def search_compounds(self, name: Optional[str] = None,
                             formula: Optional[str] = None) -> List[Dict]:
        """화합물 검색"""
        if 'pubchem' not in self.clients:
            return []
        
        # 캐시 확인
        cache_params = {'name': name, 'formula': formula}
        cached = self.cache.get('compounds', cache_params)
        if cached:
            return cached
        
        try:
            compounds = []
            
            if name:
                # 이름으로 검색
                results = await asyncio.to_thread(
                    pcp.get_compounds, name, 'name'
                )
            elif formula:
                # 분자식으로 검색
                results = await asyncio.to_thread(
                    pcp.get_compounds, formula, 'formula'
                )
            else:
                return []
            
            # 결과 변환
            for compound in results[:10]:  # 최대 10개
                compounds.append({
                    'cid': compound.cid,
                    'iupac_name': compound.iupac_name,
                    'molecular_formula': compound.molecular_formula,
                    'molecular_weight': compound.molecular_weight,
                    'smiles': compound.canonical_smiles,
                    'synonyms': compound.synonyms[:5] if compound.synonyms else []
                })
            
            # 캐시 저장
            self.cache.set('compounds', cache_params, compounds)
            
            return compounds
            
        except Exception as e:
            logger.error(f"PubChem 검색 에러: {str(e)}")
            return []

# ===========================================================================
# 🎯 메인 API 관리자
# ===========================================================================

class APIManager:
    """통합 API 관리자"""
    
    def __init__(self):
        self.encryption_manager = EncryptionManager()
        self.ai_engines: Dict[str, BaseAIEngine] = {}
        self.db_client = ScienceDBClient()
        self.usage_tracker = UsageTracker()
        self._init_engines()
        
        logger.info("API Manager 초기화 완료")
    
    def _init_engines(self):
        """AI 엔진 초기화"""
        # Gemini
        if GEMINI_AVAILABLE:
            try:
                engine = GeminiEngine()
                if engine.is_available:
                    self.ai_engines['gemini'] = engine
                    logger.info("Gemini 엔진 초기화 성공")
            except Exception as e:
                logger.error(f"Gemini 초기화 실패: {e}")
        
        # Groq
        if GROQ_AVAILABLE:
            try:
                engine = GroqEngine()
                if engine.is_available:
                    self.ai_engines['groq'] = engine
                    logger.info("Groq 엔진 초기화 성공")
            except Exception as e:
                logger.error(f"Groq 초기화 실패: {e}")
        
        # HuggingFace
        if HUGGINGFACE_AVAILABLE:
            try:
                engine = HuggingFaceEngine()
                if engine.is_available:
                    self.ai_engines['huggingface'] = engine
                    logger.info("HuggingFace 엔진 초기화 성공")
            except Exception as e:
                logger.error(f"HuggingFace 초기화 실패: {e}")
        
        # OpenAI 호환 엔진들 (나중에 API 제공시 활성화)
        # self.ai_engines['grok'] = OpenAICompatibleEngine(
        #     AIEngineType.GROK, "grok-beta", "https://api.x.ai/v1"
        # )
    
    def get_available_engines(self) -> List[str]:
        """사용 가능한 AI 엔진 목록"""
        return list(self.ai_engines.keys())
    
    def set_api_key(self, service: str, api_key: str) -> bool:
        """API 키 설정"""
        try:
            # 세션 상태에 저장
            if 'api_keys' not in st.session_state:
                st.session_state.api_keys = {}
            
            st.session_state.api_keys[service] = api_key
            
            # 엔진 재초기화
            self._init_engines()
            
            return True
        except Exception as e:
            logger.error(f"API 키 설정 실패: {e}")
            return False
    
    async def generate_text(self, engine_id: str, prompt: str,
                          user_id: str = "anonymous", **kwargs) -> APIResponse:
        """AI 텍스트 생성"""
        # 오프라인 체크
        if not self._check_connection():
            # 캐시 확인
            for engine in self.ai_engines.values():
                cache_params = {'prompt': prompt, **kwargs}
                cached = engine.cache.get(engine_id, cache_params)
                if cached:
                    return APIResponse(
                        status=ResponseStatus.OFFLINE,
                        data=cached,
                        cached=True,
                        metadata={'offline_mode': True}
                    )
            
            return APIResponse(
                status=ResponseStatus.OFFLINE,
                error="오프라인 모드: 캐시된 응답이 없습니다"
            )
        
        # 엔진 확인
        if engine_id not in self.ai_engines:
            # 폴백: 사용 가능한 첫 번째 엔진 사용
            if self.ai_engines:
                engine_id = list(self.ai_engines.keys())[0]
                logger.warning(f"요청된 엔진 없음, {engine_id}로 폴백")
            else:
                return APIResponse(
                    status=ResponseStatus.ERROR,
                    error="사용 가능한 AI 엔진이 없습니다"
                )
        
        engine = self.ai_engines[engine_id]
        
        # 생성 요청
        response = await engine.generate(prompt, user_id, **kwargs)
        
        # 사용량 기록
        if response.status == ResponseStatus.SUCCESS:
            record = UsageRecord(
                user_id=user_id,
                api_type=engine_id,
                timestamp=datetime.now(),
                tokens_used=kwargs.get('tokens', 0),
                cost=0.0,  # 추후 계산
                success=True
            )
            self.usage_tracker.record(record)
        
        return response
    
    async def extract_protocol(self, text: str, file_type: str = "mixed_format",
                             user_id: str = "anonymous") -> APIResponse:
        """프로토콜 추출"""
        # Gemini 우선 사용 (프로토콜 추출에 최적화)
        if 'gemini' in self.ai_engines:
            engine = self.ai_engines['gemini']
            if hasattr(engine, 'extract_protocol'):
                return await engine.extract_protocol(text, file_type, user_id)
        
        # 다른 엔진으로 폴백
        prompt = f"""
{EXTRACTION_PROMPTS.get(file_type, EXTRACTION_PROMPTS['mixed_format'])}

텍스트:
{text[:10000]}

JSON 형식으로 프로토콜 정보를 추출하세요.
"""
        
        return await self.generate_text(
            list(self.ai_engines.keys())[0] if self.ai_engines else 'gemini',
            prompt,
            user_id
        )
    
    async def analyze_experiment(self, engine_id: str, experiment_data: Dict,
                               user_id: str = "anonymous") -> APIResponse:
        """실험 데이터 분석"""
        prompt = f"""
다음 실험 데이터를 분석하고 인사이트를 제공하세요:

실험 정보:
{json.dumps(experiment_data, indent=2, ensure_ascii=False)}

다음 항목들을 포함하여 분석하세요:
1. 주요 발견사항
2. 통계적 유의성
3. 개선 제안
4. 다음 실험 추천
"""
        
        return await self.generate_text(engine_id, prompt, user_id)
    
    def get_usage_summary(self, user_id: Optional[str] = None,
                         period: str = 'day') -> Dict[str, Any]:
        """사용량 요약"""
        return self.usage_tracker.get_summary(user_id, period)
    
    def get_api_status(self) -> Dict[str, Any]:
        """API 상태 확인"""
        status = {
            'ai_engines': {},
            'databases': {},
            'total_available': 0
        }
        
        # AI 엔진 상태
        for engine_id, engine in self.ai_engines.items():
            status['ai_engines'][engine_id] = {
                'available': engine.is_available,
                'model': engine.model_name,
                'rate_limit': engine.rate_limiter.get_remaining()
            }
            if engine.is_available:
                status['total_available'] += 1
        
        # 데이터베이스 상태
        for db_name, client in self.db_client.clients.items():
            status['databases'][db_name] = {
                'available': client is not None
            }
        
        return status
    
    def _check_connection(self) -> bool:
        """인터넷 연결 확인"""
        try:
            import requests
            response = requests.get('https://www.google.com', timeout=3)
            return response.status_code == 200
        except:
            return False
    
    def clear_cache(self, cache_type: Optional[str] = None):
        """캐시 초기화"""
        for engine in self.ai_engines.values():
            engine.cache.clear(cache_type)
        
        self.db_client.cache.clear(cache_type)
        
        logger.info(f"캐시 초기화 완료: {cache_type or '전체'}")

# ===========================================================================
# 🔧 싱글톤 인스턴스
# ===========================================================================

_api_manager: Optional[APIManager] = None

def get_api_manager() -> APIManager:
    """API Manager 싱글톤 인스턴스 반환"""
    global _api_manager
    
    if _api_manager is None:
        _api_manager = APIManager()
    
    return _api_manager

# ===========================================================================
# 🎯 헬퍼 함수
# ===========================================================================

async def ask_ai(prompt: str, engine: str = "gemini",
                user_id: str = "anonymous", **kwargs) -> str:
    """간편 AI 질문 함수"""
    manager = get_api_manager()
    response = await manager.generate_text(engine, prompt, user_id, **kwargs)
    
    if response.status in [ResponseStatus.SUCCESS, ResponseStatus.CACHED]:
        return response.data
    else:
        return f"오류: {response.error}"

def get_available_ai_engines() -> List[str]:
    """사용 가능한 AI 엔진 목록"""
    manager = get_api_manager()
    return manager.get_available_engines()

# ===========================================================================
# 🧪 테스트 코드
# ===========================================================================

if __name__ == "__main__":
    import asyncio
    
    async def test_api_manager():
        """API Manager 테스트"""
        manager = get_api_manager()
        
        # 상태 확인
        print("API 상태:", manager.get_api_status())
        
        # 텍스트 생성 테스트
        if manager.get_available_engines():
            engine = manager.get_available_engines()[0]
            response = await manager.generate_text(
                engine,
                "고분자 합성의 기본 원리를 설명하세요.",
                "test_user"
            )
            print(f"\n{engine} 응답:", response.data[:200] if response.data else response.error)
        
        # 프로토콜 추출 테스트
        test_text = """
        Materials: Polymer A (98% purity, Sigma-Aldrich), Solvent B
        
        Procedure:
        1. Dissolve 5g of Polymer A in 100mL of Solvent B
        2. Heat to 80°C for 2 hours
        3. Cool to room temperature
        """
        
        protocol_response = await manager.extract_protocol(test_text)
        print("\n프로토콜 추출:", protocol_response.data)
    
    # 테스트 실행
    asyncio.run(test_api_manager())
