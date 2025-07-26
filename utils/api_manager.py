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
import requests

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

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logging.warning("Anthropic API 라이브러리가 설치되지 않았습니다.")

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

try:
    from pyalex import Works, Authors, Institutions
    OPENALEX_AVAILABLE = True
except ImportError:
    OPENALEX_AVAILABLE = False
    logging.warning("PyAlex 라이브러리가 설치되지 않았습니다.")

try:
    from crossref.restful import Works as CrossrefWorks
    CROSSREF_AVAILABLE = True
except ImportError:
    CROSSREF_AVAILABLE = False
    logging.warning("Crossref 라이브러리가 설치되지 않았습니다.")

try:
    import arxiv
    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False
    logging.warning("arXiv 라이브러리가 설치되지 않았습니다.")

try:
    from Bio import Entrez
    PUBMED_AVAILABLE = True
except ImportError:
    PUBMED_AVAILABLE = False
    logging.warning("Biopython 라이브러리가 설치되지 않았습니다.")

# 로컬 모듈 임포트
try:
    from config.app_config import (
        API_CONFIG, FILE_PROCESSING, PROTOCOL_EXTRACTION,
        get_config, LITERATURE_APIS, DATABASE_APIS
    )
    from config.secrets_config import API_KEY_STRUCTURE
    from config.error_config import ERROR_CODES, USER_FRIENDLY_MESSAGES
    from utils.error_handler import handle_api_error
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
        },
        'xai_grok': {
            'name': 'xAI Grok',
            'model': 'grok-beta',
            'required': False,
            'free_tier': False,
            'rate_limit': 60
        },
        'groq': {
            'name': 'Groq',
            'model': 'llama-3.3-70b-versatile',
            'required': False,
            'free_tier': True,
            'rate_limit': 100
        },
        'sambanova': {
            'name': 'SambaNova',
            'model': 'llama-3.1-405b',
            'required': False,
            'free_tier': True,
            'rate_limit': 60
        },
        'deepseek': {
            'name': 'DeepSeek',
            'model': 'deepseek-chat',
            'required': False,
            'free_tier': False,
            'rate_limit': 60
        },
        'huggingface': {
            'name': 'HuggingFace',
            'model': 'microsoft/BioGPT-Large',
            'required': False,
            'free_tier': True,
            'rate_limit': 1000
        }
    }
    ERROR_CODES = {}
    USER_FRIENDLY_MESSAGES = {}

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
    FALLBACK = "fallback"

class DatabaseType(Enum):
    """데이터베이스 타입"""
    MATERIALS_PROJECT = "materials_project"
    PUBCHEM = "pubchem"
    GITHUB = "github"
    OPENALEX = "openalex"
    CROSSREF = "crossref"
    ARXIV = "arxiv"
    PUBMED = "pubmed"
    ZENODO = "zenodo"
    PROTOCOLS_IO = "protocols_io"
    FIGSHARE = "figshare"

# Rate Limit 설정
RATE_LIMITS = {
    AIEngineType.GEMINI: {'calls': 60, 'period': 60},  # 60 calls/min
    AIEngineType.GROK: {'calls': 60, 'period': 60},
    AIEngineType.GROQ: {'calls': 100, 'period': 60},   # 100 calls/min
    AIEngineType.SAMBANOVA: {'calls': 60, 'period': 60},
    AIEngineType.DEEPSEEK: {'calls': 60, 'period': 60},
    AIEngineType.HUGGINGFACE: {'calls': 1000, 'period': 3600},  # 1000 calls/hour
    # 데이터베이스
    DatabaseType.OPENALEX: {'calls': 100, 'period': 60},  # Polite pool
    DatabaseType.CROSSREF: {'calls': 50, 'period': 60},
    DatabaseType.ARXIV: {'calls': 3, 'period': 10},  # 3초 간격
    DatabaseType.PUBMED: {'calls': 10, 'period': 1},  # 10/sec
    'default': {'calls': 30, 'period': 60}
}

# 캐시 TTL 설정 (초)
CACHE_TTL = {
    'ai_response': 3600,      # 1시간
    'material_data': 86400,   # 24시간
    'compound_data': 86400,   # 24시간
    'protocol': 7200,         # 2시간
    'literature': 21600,      # 6시간
    'benchmark': 43200,       # 12시간
    'default': 1800           # 30분
}

# API 비용 추정 (1K 토큰당 USD)
API_COSTS = {
    AIEngineType.GEMINI: {'input': 0.0, 'output': 0.0},  # 무료
    AIEngineType.GROK: {'input': 0.005, 'output': 0.015},
    AIEngineType.GROQ: {'input': 0.0, 'output': 0.0},    # 무료
    AIEngineType.DEEPSEEK: {'input': 0.001, 'output': 0.002},
    AIEngineType.SAMBANOVA: {'input': 0.0, 'output': 0.0},  # 무료 티어
    AIEngineType.HUGGINGFACE: {'input': 0.0, 'output': 0.0},  # 무료
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
    
    'polymer_specific': """
고분자 실험 프로토콜을 추출합니다.
특히 다음 사항에 주의하세요:
- 용매 시스템 (단일/이성분/삼성분)
- 고분자 농도 및 분자량
- 가공 조건 (온도, 압력, 시간)
- 특수 장비 (전기방사, 압출기 등)
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
    error_code: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    cached: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    fallback_used: Optional[str] = None

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
    response_time: Optional[float] = None

@dataclass
class ProtocolData:
    """추출된 프로토콜 데이터"""
    materials: List[Dict[str, Any]]
    equipment: List[Dict[str, Any]]
    conditions: Dict[str, Any]
    procedure: List[Dict[str, Any]]
    safety: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0
    source: Optional[str] = None

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
                'reset_in': int(self.period - (now - self.calls[0])) if self.calls else 0,
                'total': self.max_calls
            }
    
    def wait_if_needed(self) -> float:
        """필요시 대기 시간 반환"""
        with self.lock:
            if not self.check():
                now = time.time()
                oldest_call = self.calls[0]
                wait_time = self.period - (now - oldest_call) + 0.1
                return wait_time
            return 0

# ===========================================================================
# 💾 캐시 시스템
# ===========================================================================

class APICache:
    """API 응답 캐싱"""
    
    def __init__(self, ttl: int = 3600):
        self.ttl = ttl
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.lock = threading.Lock()
        self.stats = {'hits': 0, 'misses': 0, 'evictions': 0}
    
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
                    self.stats['hits'] += 1
                    return data
                else:
                    del self.cache[key]
                    self.stats['evictions'] += 1
            self.stats['misses'] += 1
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
    
    def get_stats(self) -> Dict[str, int]:
        """캐시 통계"""
        with self.lock:
            return self.stats.copy()

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
                'errors': 0,
                'avg_response_time': 0.0
            })
            
            response_times = defaultdict(list)
            
            for record in filtered:
                api_summary = summary[record.api_type]
                api_summary['calls'] += 1
                api_summary['tokens'] += record.tokens_used or 0
                api_summary['cost'] += record.cost or 0
                if not record.success:
                    api_summary['errors'] += 1
                if record.response_time:
                    response_times[record.api_type].append(record.response_time)
            
            # 평균 응답 시간 계산
            for api_type, times in response_times.items():
                if times:
                    summary[api_type]['avg_response_time'] = sum(times) / len(times)
            
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
    
    def estimate_tokens(self, text: str) -> int:
        """토큰 수 추정 (대략적)"""
        # 한글은 평균 2-3토큰, 영어는 평균 1.3토큰 per word
        korean_chars = len([c for c in text if ord(c) >= 0xAC00 and ord(c) <= 0xD7AF])
        english_words = len(text.split()) - korean_chars // 3
        return int(korean_chars * 2.5 + english_words * 1.3)

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
        start_time = time.time()
        
        # 캐시 확인
        cache_params = {'prompt': prompt, **kwargs}
        cached = self.cache.get('gemini', cache_params)
        if cached:
            return APIResponse(
                status=ResponseStatus.CACHED,
                data=cached,
                cached=True,
                metadata={'response_time': 0.0}
            )
        
        # Rate limit 확인
        wait_time = self.rate_limiter.wait_if_needed()
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        
        try:
            # 동기 호출을 비동기로 변환
            response = await asyncio.to_thread(
                self.model.generate_content, prompt
            )
            
            # Rate limit 기록
            self.rate_limiter.record()
            
            # 응답 처리
            result = response.text
            response_time = time.time() - start_time
            
            # 캐시 저장
            self.cache.set('gemini', cache_params, result)
            
            # 사용량 기록
            estimated_tokens = self.estimate_tokens(prompt + result)
            record = UsageRecord(
                user_id=user_id,
                api_type='gemini',
                timestamp=datetime.now(),
                tokens_used=estimated_tokens,
                cost=0.0,  # 무료
                success=True,
                response_time=response_time
            )
            
            return APIResponse(
                status=ResponseStatus.SUCCESS,
                data=result,
                metadata={
                    'model': self.model_name,
                    'response_time': response_time,
                    'tokens': estimated_tokens
                }
            )
            
        except Exception as e:
            logger.error(f"Gemini 에러: {str(e)}")
            return APIResponse(
                status=ResponseStatus.ERROR,
                error=str(e),
                error_code='3001'  # AI API 에러
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
                response.data = ProtocolData(
                    materials=protocol_data.get('materials', []),
                    equipment=protocol_data.get('equipment', []),
                    conditions=protocol_data.get('conditions', {}),
                    procedure=protocol_data.get('procedure', []),
                    safety=protocol_data.get('safety', []),
                    confidence_score=0.9,
                    source='gemini'
                )
            except json.JSONDecodeError:
                # JSON 파싱 실패 시 텍스트 그대로 반환
                logger.warning("프로토콜 JSON 파싱 실패")
                response.error_code = '4202'  # 프로토콜 추출 실패
        
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
        start_time = time.time()
        
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
        wait_time = self.rate_limiter.wait_if_needed()
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        
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
            response_time = time.time() - start_time
            
            # 캐시 저장
            self.cache.set('groq', cache_params, result)
            
            # 사용량 기록
            tokens = response.usage.total_tokens if hasattr(response, 'usage') else 0
            
            return APIResponse(
                status=ResponseStatus.SUCCESS,
                data=result,
                metadata={
                    'model': self.model_name,
                    'response_time': response_time,
                    'tokens': tokens
                }
            )
            
        except Exception as e:
            logger.error(f"Groq 에러: {str(e)}")
            return APIResponse(
                status=ResponseStatus.ERROR,
                error=str(e),
                error_code='3002'
            )

# ===========================================================================
# 🤗 HuggingFace 엔진
# ===========================================================================

class HuggingFaceEngine(BaseAIEngine):
    """HuggingFace 특수 모델 엔진"""
    
    def __init__(self, use_local: bool = False):
        super().__init__(AIEngineType.HUGGINGFACE, "microsoft/BioGPT-Large")
        self.use_local = use_local
        if self.is_available and HUGGINGFACE_AVAILABLE:
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
        start_time = time.time()
        
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
            
            response_time = time.time() - start_time
            
            # 캐시 저장
            self.cache.set('huggingface', cache_params, text)
            
            return APIResponse(
                status=ResponseStatus.SUCCESS,
                data=text,
                metadata={
                    'model': self.model_name,
                    'response_time': response_time
                }
            )
            
        except Exception as e:
            logger.error(f"HuggingFace 에러: {str(e)}")
            return APIResponse(
                status=ResponseStatus.ERROR,
                error=str(e),
                error_code='3006'
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
        start_time = time.time()
        
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
        wait_time = self.rate_limiter.wait_if_needed()
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        
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
            response_time = time.time() - start_time
            
            # 캐시 저장
            self.cache.set(self.engine_type.value, cache_params, result)
            
            # 사용량 기록
            tokens = 0
            cost = 0.0
            if hasattr(response, 'usage'):
                tokens = response.usage.total_tokens
                cost = self.estimate_cost(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens
                )
            
            return APIResponse(
                status=ResponseStatus.SUCCESS,
                data=result,
                metadata={
                    'model': self.model_name,
                    'response_time': response_time,
                    'tokens': tokens,
                    'cost': cost
                }
            )
            
        except Exception as e:
            logger.error(f"{self.engine_type.value} 에러: {str(e)}")
            error_code = {
                AIEngineType.GROK: '3003',
                AIEngineType.DEEPSEEK: '3004',
                AIEngineType.SAMBANOVA: '3005'
            }.get(self.engine_type, '3000')
            
            return APIResponse(
                status=ResponseStatus.ERROR,
                error=str(e),
                error_code=error_code
            )

# ===========================================================================
# 🧬 SambaNova 엔진 (Claude API)
# ===========================================================================

class SambaNovaEngine(BaseAIEngine):
    """SambaNova Claude 호환 엔진"""
    
    def __init__(self):
        super().__init__(AIEngineType.SAMBANOVA, "llama-3.1-405b")
        if self.is_available and ANTHROPIC_AVAILABLE:
            self.client = anthropic.AsyncAnthropic(
                api_key=self.api_key,
                base_url="https://api.sambanova.ai/v1"
            )
    
    async def generate(self, prompt: str, user_id: str = "anonymous", 
                      **kwargs) -> APIResponse:
        """SambaNova로 텍스트 생성"""
        start_time = time.time()
        
        # 캐시 확인
        cache_params = {'prompt': prompt, **kwargs}
        cached = self.cache.get('sambanova', cache_params)
        if cached:
            return APIResponse(
                status=ResponseStatus.CACHED,
                data=cached,
                cached=True
            )
        
        try:
            # API 호출
            response = await self.client.messages.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get('max_tokens', 4096),
                temperature=kwargs.get('temperature', 0.7)
            )
            
            # 응답 처리
            result = response.content[0].text
            response_time = time.time() - start_time
            
            # 캐시 저장
            self.cache.set('sambanova', cache_params, result)
            
            return APIResponse(
                status=ResponseStatus.SUCCESS,
                data=result,
                metadata={
                    'model': self.model_name,
                    'response_time': response_time
                }
            )
            
        except Exception as e:
            logger.error(f"SambaNova 에러: {str(e)}")
            return APIResponse(
                status=ResponseStatus.ERROR,
                error=str(e),
                error_code='3005'
            )

# ===========================================================================
# 🔬 과학 데이터베이스 클라이언트
# ===========================================================================

class ScienceDBClient:
    """과학 데이터베이스 통합 클라이언트"""
    
    def __init__(self):
        self.clients = {}
        self.cache = APICache(CACHE_TTL.get('material_data', 86400))
        self.rate_limiters = {}
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
        
        # OpenAlex (Polite pool)
        if OPENALEX_AVAILABLE:
            email = os.getenv('OPENALEX_EMAIL', 'your-email@example.com')
            Works().config.email = email
            self.clients['openalex'] = Works()
            limits = RATE_LIMITS.get(DatabaseType.OPENALEX, RATE_LIMITS['default'])
            self.rate_limiters['openalex'] = RateLimiter(limits['calls'], limits['period'])
        
        # Crossref
        if CROSSREF_AVAILABLE:
            self.clients['crossref'] = CrossrefWorks()
            limits = RATE_LIMITS.get(DatabaseType.CROSSREF, RATE_LIMITS['default'])
            self.rate_limiters['crossref'] = RateLimiter(limits['calls'], limits['period'])
        
        # arXiv
        if ARXIV_AVAILABLE:
            self.clients['arxiv'] = arxiv
            limits = RATE_LIMITS.get(DatabaseType.ARXIV, RATE_LIMITS['default'])
            self.rate_limiters['arxiv'] = RateLimiter(limits['calls'], limits['period'])
        
        # PubMed
        if PUBMED_AVAILABLE:
            Entrez.email = os.getenv('PUBMED_EMAIL', 'your-email@example.com')
            Entrez.tool = os.getenv('PUBMED_TOOL', 'UniversalDOE')
            self.clients['pubmed'] = Entrez
            limits = RATE_LIMITS.get(DatabaseType.PUBMED, RATE_LIMITS['default'])
            self.rate_limiters['pubmed'] = RateLimiter(limits['calls'], limits['period'])
    
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
    
    async def search_literature(self, query: str, source: str = 'openalex',
                              limit: int = 10) -> List[Dict]:
        """문헌 검색"""
        if source not in self.clients:
            return []
        
        # 캐시 확인
        cache_params = {'query': query, 'source': source, 'limit': limit}
        cached = self.cache.get('literature', cache_params)
        if cached:
            return cached
        
        # Rate limit 확인
        if source in self.rate_limiters:
            wait_time = self.rate_limiters[source].wait_if_needed()
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        try:
            results = []
            
            if source == 'openalex':
                # OpenAlex 검색
                works = await asyncio.to_thread(
                    self.clients['openalex'].search,
                    query
                )
                for work in works[:limit]:
                    results.append({
                        'title': work.get('title'),
                        'doi': work.get('doi'),
                        'year': work.get('publication_year'),
                        'authors': [a.get('author', {}).get('display_name') 
                                   for a in work.get('authorships', [])],
                        'abstract': work.get('abstract'),
                        'source': 'OpenAlex'
                    })
            
            elif source == 'crossref':
                # Crossref 검색
                works = self.clients['crossref'].query(query).sort('relevance')
                for work in works[:limit]:
                    results.append({
                        'title': work.get('title', [''])[0],
                        'doi': work.get('DOI'),
                        'year': work.get('published-print', {}).get('date-parts', [[None]])[0][0],
                        'authors': [f"{a.get('given', '')} {a.get('family', '')}" 
                                   for a in work.get('author', [])],
                        'abstract': work.get('abstract'),
                        'source': 'Crossref'
                    })
            
            elif source == 'arxiv':
                # arXiv 검색
                search = arxiv.Search(
                    query=query,
                    max_results=limit,
                    sort_by=arxiv.SortCriterion.Relevance
                )
                for paper in search.results():
                    results.append({
                        'title': paper.title,
                        'doi': paper.doi,
                        'year': paper.published.year,
                        'authors': [author.name for author in paper.authors],
                        'abstract': paper.summary,
                        'pdf_url': paper.pdf_url,
                        'source': 'arXiv'
                    })
            
            elif source == 'pubmed':
                # PubMed 검색
                handle = Entrez.esearch(db="pubmed", term=query, retmax=limit)
                record = Entrez.read(handle)
                handle.close()
                
                if record['IdList']:
                    handle = Entrez.efetch(
                        db="pubmed",
                        id=record['IdList'],
                        rettype="abstract",
                        retmode="xml"
                    )
                    papers = Entrez.read(handle)
                    handle.close()
                    
                    for paper in papers['PubmedArticle']:
                        article = paper['MedlineCitation']['Article']
                        results.append({
                            'title': article.get('ArticleTitle', ''),
                            'doi': None,  # DOI 추출 로직 필요
                            'year': article['Journal']['JournalIssue'].get('PubDate', {}).get('Year'),
                            'authors': [f"{author.get('ForeName', '')} {author.get('LastName', '')}"
                                       for author in article.get('AuthorList', [])],
                            'abstract': article.get('Abstract', {}).get('AbstractText', [''])[0],
                            'source': 'PubMed'
                        })
            
            # Rate limit 기록
            if source in self.rate_limiters:
                self.rate_limiters[source].record()
            
            # 캐시 저장
            self.cache.set('literature', cache_params, results)
            
            return results
            
        except Exception as e:
            logger.error(f"{source} 검색 에러: {str(e)}")
            return []

# ===========================================================================
# 🔄 폴백 메커니즘
# ===========================================================================

class FallbackChain:
    """AI 엔진 폴백 체인"""
    
    def __init__(self, engines: List[BaseAIEngine]):
        self.engines = engines
    
    async def generate(self, prompt: str, user_id: str = "anonymous",
                      **kwargs) -> APIResponse:
        """폴백 체인으로 생성"""
        errors = []
        
        for i, engine in enumerate(self.engines):
            if not engine.is_available:
                continue
            
            try:
                response = await engine.generate(prompt, user_id, **kwargs)
                
                if response.status == ResponseStatus.SUCCESS:
                    if i > 0:  # 폴백 사용
                        response.fallback_used = engine.engine_type.value
                    return response
                else:
                    errors.append({
                        'engine': engine.engine_type.value,
                        'error': response.error
                    })
                    
            except Exception as e:
                errors.append({
                    'engine': engine.engine_type.value,
                    'error': str(e)
                })
        
        # 모든 엔진 실패
        return APIResponse(
            status=ResponseStatus.ERROR,
            error="모든 AI 엔진 호출 실패",
            error_code='3100',
            metadata={'errors': errors}
        )

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
        self.fallback_chains: Dict[str, FallbackChain] = {}
        self._init_engines()
        self._init_fallback_chains()
        
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
        
        # xAI Grok
        if OPENAI_AVAILABLE:
            try:
                engine = OpenAICompatibleEngine(
                    AIEngineType.GROK,
                    "grok-beta",
                    "https://api.x.ai/v1"
                )
                if engine.is_available:
                    self.ai_engines['grok'] = engine
                    logger.info("xAI Grok 엔진 초기화 성공")
            except Exception as e:
                logger.error(f"Grok 초기화 실패: {e}")
        
        # DeepSeek
        if OPENAI_AVAILABLE:
            try:
                engine = OpenAICompatibleEngine(
                    AIEngineType.DEEPSEEK,
                    "deepseek-chat",
                    "https://api.deepseek.com/v1"
                )
                if engine.is_available:
                    self.ai_engines['deepseek'] = engine
                    logger.info("DeepSeek 엔진 초기화 성공")
            except Exception as e:
                logger.error(f"DeepSeek 초기화 실패: {e}")
        
        # SambaNova
        if ANTHROPIC_AVAILABLE:
            try:
                engine = SambaNovaEngine()
                if engine.is_available:
                    self.ai_engines['sambanova'] = engine
                    logger.info("SambaNova 엔진 초기화 성공")
            except Exception as e:
                logger.error(f"SambaNova 초기화 실패: {e}")
    
    def _init_fallback_chains(self):
        """폴백 체인 초기화"""
        # 무료 엔진 우선 체인
        free_engines = []
        for engine_id in ['gemini', 'groq', 'sambanova', 'huggingface']:
            if engine_id in self.ai_engines:
                free_engines.append(self.ai_engines[engine_id])
        
        if free_engines:
            self.fallback_chains['free'] = FallbackChain(free_engines)
        
        # 전체 엔진 체인
        all_engines = list(self.ai_engines.values())
        if all_engines:
            self.fallback_chains['all'] = FallbackChain(all_engines)
    
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
            self._init_fallback_chains()
            
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
                error="오프라인 모드: 캐시된 응답이 없습니다",
                error_code='5001'  # 네트워크 에러
            )
        
        # 엔진 확인
        if engine_id not in self.ai_engines:
            # 폴백 체인 사용
            if 'free' in self.fallback_chains:
                response = await self.fallback_chains['free'].generate(
                    prompt, user_id, **kwargs
                )
                if response.status == ResponseStatus.SUCCESS:
                    response.status = ResponseStatus.FALLBACK
                return response
            else:
                return APIResponse(
                    status=ResponseStatus.ERROR,
                    error="사용 가능한 AI 엔진이 없습니다",
                    error_code='3000'
                )
        
        engine = self.ai_engines[engine_id]
        
        # 생성 요청
        response = await engine.generate(prompt, user_id, **kwargs)
        
        # 사용량 기록
        if response.status in [ResponseStatus.SUCCESS, ResponseStatus.CACHED]:
            record = UsageRecord(
                user_id=user_id,
                api_type=engine_id,
                timestamp=datetime.now(),
                tokens_used=response.metadata.get('tokens', 0),
                cost=response.metadata.get('cost', 0.0),
                success=True,
                response_time=response.metadata.get('response_time', 0.0)
            )
            self.usage_tracker.record(record)
        
        return response
    
    async def extract_protocol(self, text: str, file_type: str = "mixed_format",
                             user_id: str = "anonymous", 
                             preferred_engine: str = "gemini") -> APIResponse:
        """프로토콜 추출"""
        # Gemini 우선 사용 (프로토콜 추출에 최적화)
        if preferred_engine in self.ai_engines:
            engine = self.ai_engines[preferred_engine]
            if hasattr(engine, 'extract_protocol'):
                return await engine.extract_protocol(text, file_type, user_id)
        
        # 다른 엔진으로 폴백
        prompt = f"""
{EXTRACTION_PROMPTS.get(file_type, EXTRACTION_PROMPTS['mixed_format'])}

텍스트:
{text[:10000]}

JSON 형식으로 프로토콜 정보를 추출하세요.
"""
        
        # 사용 가능한 엔진으로 시도
        if self.ai_engines:
            return await self.generate_text(
                list(self.ai_engines.keys())[0],
                prompt,
                user_id
            )
        else:
            return APIResponse(
                status=ResponseStatus.ERROR,
                error="프로토콜 추출을 위한 AI 엔진이 없습니다",
                error_code='4202'
            )
    
    async def analyze_experiment(self, experiment_data: Dict,
                               user_id: str = "anonymous",
                               engine_id: Optional[str] = None) -> APIResponse:
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
5. 문헌과의 비교 (가능한 경우)
"""
        
        # 엔진 선택
        if not engine_id:
            # 분석에 적합한 엔진 우선순위
            for preferred in ['gemini', 'sambanova', 'groq']:
                if preferred in self.ai_engines:
                    engine_id = preferred
                    break
        
        return await self.generate_text(engine_id or 'gemini', prompt, user_id)
    
    async def search_benchmark_data(self, query: str, 
                                  limit: int = 20) -> List[Dict]:
        """벤치마크 데이터 검색"""
        results = []
        
        # 여러 소스에서 병렬 검색
        tasks = []
        for source in ['openalex', 'crossref', 'arxiv']:
            if source in self.db_client.clients:
                task = self.db_client.search_literature(query, source, limit)
                tasks.append(task)
        
        if tasks:
            all_results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in all_results:
                if isinstance(result, list):
                    results.extend(result)
        
        # 중복 제거 및 관련성 정렬
        seen = set()
        unique_results = []
        for item in results:
            title = item.get('title', '').lower()
            if title and title not in seen:
                seen.add(title)
                unique_results.append(item)
        
        return unique_results[:limit]
    
    def get_usage_summary(self, user_id: Optional[str] = None,
                         period: str = 'day') -> Dict[str, Any]:
        """사용량 요약"""
        return self.usage_tracker.get_summary(user_id, period)
    
    def get_api_status(self) -> Dict[str, Any]:
        """API 상태 확인"""
        status = {
            'ai_engines': {},
            'databases': {},
            'fallback_chains': list(self.fallback_chains.keys()),
            'total_available': 0,
            'cache_stats': {}
        }
        
        # AI 엔진 상태
        for engine_id, engine in self.ai_engines.items():
            status['ai_engines'][engine_id] = {
                'available': engine.is_available,
                'model': engine.model_name,
                'rate_limit': engine.rate_limiter.get_remaining(),
                'cache_stats': engine.cache.get_stats()
            }
            if engine.is_available:
                status['total_available'] += 1
        
        # 데이터베이스 상태
        for db_name, client in self.db_client.clients.items():
            status['databases'][db_name] = {
                'available': client is not None
            }
        
        # 전체 캐시 통계
        total_cache_stats = {'hits': 0, 'misses': 0, 'evictions': 0}
        for engine in self.ai_engines.values():
            stats = engine.cache.get_stats()
            for key, value in stats.items():
                total_cache_stats[key] += value
        
        status['cache_stats'] = total_cache_stats
        
        return status
    
    def _check_connection(self) -> bool:
        """인터넷 연결 확인"""
        try:
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
    
    async def multi_engine_consensus(self, prompt: str, 
                                   engines: Optional[List[str]] = None,
                                   user_id: str = "anonymous") -> APIResponse:
        """여러 AI 엔진의 합의 응답 생성"""
        if not engines:
            engines = list(self.ai_engines.keys())[:3]  # 최대 3개
        
        # 병렬로 여러 엔진 호출
        tasks = []
        for engine_id in engines:
            if engine_id in self.ai_engines:
                task = self.generate_text(engine_id, prompt, user_id)
                tasks.append((engine_id, task))
        
        if not tasks:
            return APIResponse(
                status=ResponseStatus.ERROR,
                error="사용 가능한 엔진이 없습니다",
                error_code='3000'
            )
        
        # 결과 수집
        results = []
        for engine_id, task in tasks:
            try:
                response = await task
                if response.status == ResponseStatus.SUCCESS:
                    results.append({
                        'engine': engine_id,
                        'response': response.data
                    })
            except Exception as e:
                logger.error(f"{engine_id} 합의 생성 실패: {e}")
        
        if not results:
            return APIResponse(
                status=ResponseStatus.ERROR,
                error="모든 엔진 호출 실패",
                error_code='3100'
            )
        
        # 결과 통합
        consensus_data = {
            'responses': results,
            'consensus': self._generate_consensus(results),
            'engines_used': [r['engine'] for r in results]
        }
        
        return APIResponse(
            status=ResponseStatus.SUCCESS,
            data=consensus_data,
            metadata={'consensus_count': len(results)}
        )
    
    def _generate_consensus(self, results: List[Dict]) -> str:
        """응답들로부터 합의 생성"""
        if len(results) == 1:
            return results[0]['response']
        
        # 간단한 합의: 공통 내용 추출
        # 실제로는 더 복잡한 알고리즘 필요
        consensus = "다음은 여러 AI 엔진의 종합적인 견해입니다:\n\n"
        for i, result in enumerate(results, 1):
            consensus += f"{i}. {result['engine']} 의견:\n"
            consensus += f"{result['response'][:200]}...\n\n"
        
        return consensus

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

async def ask_ai(prompt: str, engine: str = "auto",
                user_id: str = "anonymous", **kwargs) -> str:
    """간편 AI 질문 함수"""
    manager = get_api_manager()
    
    if engine == "auto":
        # 자동 엔진 선택
        available = manager.get_available_engines()
        if available:
            engine = available[0]
        else:
            return "오류: 사용 가능한 AI 엔진이 없습니다"
    
    response = await manager.generate_text(engine, prompt, user_id, **kwargs)
    
    if response.status in [ResponseStatus.SUCCESS, ResponseStatus.CACHED, 
                          ResponseStatus.FALLBACK]:
        return response.data
    else:
        return f"오류: {response.error}"

def get_available_ai_engines() -> List[str]:
    """사용 가능한 AI 엔진 목록"""
    manager = get_api_manager()
    return manager.get_available_engines()

async def search_materials(formula: str = None, **criteria) -> List[Dict]:
    """재료 검색 헬퍼"""
    manager = get_api_manager()
    return await manager.db_client.search_materials(formula, **criteria)

async def search_compounds(name: str = None, formula: str = None) -> List[Dict]:
    """화합물 검색 헬퍼"""
    manager = get_api_manager()
    return await manager.db_client.search_compounds(name, formula)

async def search_literature(query: str, sources: List[str] = None, 
                          limit: int = 10) -> List[Dict]:
    """문헌 검색 헬퍼"""
    manager = get_api_manager()
    
    if not sources:
        sources = ['openalex', 'crossref']
    
    all_results = []
    for source in sources:
        results = await manager.db_client.search_literature(query, source, limit)
        all_results.extend(results)
    
    return all_results

# ===========================================================================
# 🧪 테스트 코드
# ===========================================================================

if __name__ == "__main__":
    import asyncio
    
    async def test_api_manager():
        """API Manager 테스트"""
        manager = get_api_manager()
        
        # 상태 확인
        print("API 상태:", json.dumps(manager.get_api_status(), indent=2))
        
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
        
        # 문헌 검색 테스트
        if 'openalex' in manager.db_client.clients:
            papers = await manager.db_client.search_literature(
                "polymer synthesis electrospinning",
                "openalex",
                5
            )
            print(f"\n문헌 검색 결과: {len(papers)}개")
            if papers:
                print(f"첫 번째 논문: {papers[0].get('title')}")
        
        # 사용량 요약
        usage = manager.get_usage_summary("test_user", "day")
        print("\n사용량 요약:", json.dumps(usage, indent=2))
    
    # 테스트 실행
    asyncio.run(test_api_manager())
