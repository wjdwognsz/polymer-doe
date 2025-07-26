# utils/api_manager.py
"""
AI API 통합 관리자
6개 AI 엔진과 다양한 과학 데이터베이스를 중앙에서 관리합니다.
오프라인 우선 설계로 캐싱과 폴백 메커니즘을 제공합니다.
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

# 내부 모듈
from utils.database_manager import get_database_manager
from utils.common_ui import show_error, show_warning, show_info, show_success

# 로깅 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ============================================================================
# Enums 및 데이터 클래스
# ============================================================================

class AIEngineType(Enum):
    """AI 엔진 타입"""
    GEMINI = "gemini"
    GROK = "grok"
    GROQ = "groq"
    SAMBANOVA = "sambanova"
    DEEPSEEK = "deepseek"
    HUGGINGFACE = "huggingface"


class APICategory(Enum):
    """API 카테고리"""
    AI_ENGINE = "ai_engine"
    SCIENCE_DB = "science_db"
    DATA_REPO = "data_repo"


class ResponseStatus(Enum):
    """API 응답 상태"""
    SUCCESS = "success"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    CACHED = "cached"


@dataclass
class APIConfig:
    """API 설정"""
    engine_type: AIEngineType
    model_name: str
    api_key_id: str
    max_tokens: int = 2048
    temperature: float = 0.7
    timeout: int = 30
    cache_ttl: int = 3600
    features: List[str] = field(default_factory=list)


@dataclass
class APIResponse:
    """API 응답"""
    status: ResponseStatus
    data: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    latency: float = 0.0
    cached: bool = False


@dataclass
class APIUsage:
    """API 사용량 추적"""
    api_id: str
    user_id: str
    timestamp: datetime
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    success: bool = True
    error_type: Optional[str] = None
    latency: float = 0.0


# ============================================================================
# 유틸리티 클래스
# ============================================================================

class EncryptionManager:
    """API 키 암호화 관리"""
    
    def __init__(self):
        self.key = self._get_or_create_key()
        self.cipher = Fernet(self.key)
    
    def _get_or_create_key(self) -> bytes:
        """암호화 키 생성 또는 로드"""
        key_file = Path.home() / '.polymer_doe' / 'api_key.key'
        key_file.parent.mkdir(exist_ok=True)
        
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


class RateLimiter:
    """API 호출 제한 관리"""
    
    def __init__(self, calls_per_minute: int = 60, calls_per_day: Optional[int] = None):
        self.calls_per_minute = calls_per_minute
        self.calls_per_day = calls_per_day
        self.minute_calls = deque()
        self.day_calls = deque()
        self.lock = threading.Lock()
    
    def check_and_add(self) -> Tuple[bool, Optional[float]]:
        """호출 가능 여부 확인 및 기록"""
        with self.lock:
            now = time.time()
            
            # 1분 제한 확인
            minute_ago = now - 60
            self.minute_calls = deque(t for t in self.minute_calls if t > minute_ago)
            
            if len(self.minute_calls) >= self.calls_per_minute:
                wait_time = 60 - (now - self.minute_calls[0])
                return False, wait_time
            
            # 일일 제한 확인
            if self.calls_per_day:
                day_ago = now - 86400
                self.day_calls = deque(t for t in self.day_calls if t > day_ago)
                
                if len(self.day_calls) >= self.calls_per_day:
                    wait_time = 86400 - (now - self.day_calls[0])
                    return False, wait_time
            
            # 호출 기록
            self.minute_calls.append(now)
            if self.calls_per_day:
                self.day_calls.append(now)
            
            return True, None
    
    def get_remaining(self) -> Dict[str, int]:
        """남은 호출 횟수"""
        with self.lock:
            now = time.time()
            minute_ago = now - 60
            
            minute_calls = sum(1 for t in self.minute_calls if t > minute_ago)
            remaining_minute = max(0, self.calls_per_minute - minute_calls)
            
            result = {'per_minute': remaining_minute}
            
            if self.calls_per_day:
                day_ago = now - 86400
                day_calls = sum(1 for t in self.day_calls if t > day_ago)
                result['per_day'] = max(0, self.calls_per_day - day_calls)
            
            return result


class ResponseCache:
    """API 응답 캐시"""
    
    def __init__(self, default_ttl: int = 3600):
        self.default_ttl = default_ttl
        self.cache = {}
        self.timestamps = {}
        self.lock = threading.Lock()
        self.db_manager = get_database_manager()
    
    def _generate_key(self, api_type: str, params: Dict) -> str:
        """캐시 키 생성"""
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(f"{api_type}:{param_str}".encode()).hexdigest()
    
    def get(self, api_type: str, params: Dict) -> Optional[Any]:
        """캐시에서 응답 조회"""
        key = self._generate_key(api_type, params)
        
        with self.lock:
            # 메모리 캐시 확인
            if key in self.cache:
                timestamp = self.timestamps.get(key, 0)
                if time.time() - timestamp < self.default_ttl:
                    return self.cache[key]
                else:
                    # 만료된 캐시 삭제
                    del self.cache[key]
                    del self.timestamps[key]
            
            # DB 캐시 확인
            cached = self.db_manager.cache_get(f"api:{key}")
            if cached:
                self.cache[key] = cached
                self.timestamps[key] = time.time()
                return cached
        
        return None
    
    def set(self, api_type: str, params: Dict, response: Any, 
            ttl: Optional[int] = None):
        """응답 캐시에 저장"""
        key = self._generate_key(api_type, params)
        ttl = ttl or self.default_ttl
        
        with self.lock:
            # 메모리 캐시 저장
            self.cache[key] = response
            self.timestamps[key] = time.time()
            
            # DB 캐시 저장
            self.db_manager.cache_set(f"api:{key}", response, ttl)
    
    def clear(self, api_type: Optional[str] = None):
        """캐시 초기화"""
        with self.lock:
            if api_type:
                # 특정 API 타입만 삭제
                keys_to_remove = [
                    k for k in self.cache.keys() 
                    if k.startswith(f"{api_type}:")
                ]
                for key in keys_to_remove:
                    del self.cache[key]
                    del self.timestamps[key]
            else:
                # 전체 캐시 삭제
                self.cache.clear()
                self.timestamps.clear()


class UsageTracker:
    """API 사용량 추적"""
    
    def __init__(self):
        self.usage_data = defaultdict(list)
        self.lock = threading.Lock()
        self.db_manager = get_database_manager()
    
    async def track(self, usage: APIUsage):
        """사용량 기록"""
        with self.lock:
            self.usage_data[usage.api_id].append(usage)
            
            # DB에 저장
            try:
                self.db_manager.insert('api_usage', {
                    'api_id': usage.api_id,
                    'user_id': usage.user_id,
                    'timestamp': usage.timestamp.isoformat(),
                    'tokens_used': usage.tokens_used,
                    'cost': usage.cost,
                    'success': usage.success,
                    'error_type': usage.error_type,
                    'latency': usage.latency
                })
            except Exception as e:
                logger.error(f"사용량 추적 DB 저장 실패: {e}")
    
    def get_usage_summary(self, user_id: Optional[str] = None, 
                         period: str = 'day') -> Dict[str, Any]:
        """사용량 요약"""
        now = datetime.now()
        
        if period == 'day':
            start_time = now - timedelta(days=1)
        elif period == 'week':
            start_time = now - timedelta(weeks=1)
        elif period == 'month':
            start_time = now - timedelta(days=30)
        else:
            start_time = datetime.min
        
        # 필터링
        summary = defaultdict(lambda: {
            'requests': 0, 'tokens': 0, 'cost': 0.0, 
            'errors': 0, 'success_rate': 0.0
        })
        
        for api_id, usages in self.usage_data.items():
            filtered = [u for u in usages if u.timestamp >= start_time]
            if user_id:
                filtered = [u for u in filtered if u.user_id == user_id]
            
            if filtered:
                total = len(filtered)
                success = sum(1 for u in filtered if u.success)
                
                summary[api_id] = {
                    'requests': total,
                    'tokens': sum(u.tokens_used or 0 for u in filtered),
                    'cost': sum(u.cost or 0 for u in filtered),
                    'errors': total - success,
                    'success_rate': (success / total * 100) if total > 0 else 0
                }
        
        return dict(summary)


# ============================================================================
# AI 엔진 기본 클래스
# ============================================================================

class BaseAIEngine:
    """모든 AI 엔진의 기본 클래스"""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.client = None
        self.rate_limiter = None
        self.cache = ResponseCache(config.cache_ttl)
        self.is_available = False
        self.usage_tracker = UsageTracker()
        self._initialize()
    
    def _initialize(self):
        """엔진 초기화"""
        # API 키 확인
        if not self.config.api_key_id:
            self.config.api_key_id = self._get_api_key()
        
        if not self.config.api_key_id:
            logger.warning(f"{self.config.engine_type.value} API 키가 없습니다.")
            return
        
        # Rate limiter 설정
        limits = self._get_rate_limits()
        self.rate_limiter = RateLimiter(
            calls_per_minute=limits.get('per_minute', 60),
            calls_per_day=limits.get('per_day')
        )
        
        # 클라이언트 초기화
        try:
            self._init_client()
            self.is_available = True
            logger.info(f"{self.config.engine_type.value} 엔진 초기화 성공")
        except Exception as e:
            logger.error(f"{self.config.engine_type.value} 초기화 실패: {str(e)}")
            self.is_available = False
    
    def _get_api_key(self) -> Optional[str]:
        """API 키 가져오기"""
        # 1. 환경 변수
        key = os.environ.get(f"{self.config.engine_type.value.upper()}_API_KEY")
        if key:
            return key
        
        # 2. Streamlit secrets
        if hasattr(st, 'secrets'):
            try:
                key = st.secrets.get(f"{self.config.engine_type.value}_api_key")
                if key:
                    return key
            except:
                pass
        
        # 3. 세션 상태
        if hasattr(st, 'session_state') and 'api_keys' in st.session_state:
            key = st.session_state.api_keys.get(self.config.engine_type.value)
            if key:
                return key
        
        return None
    
    def _get_rate_limits(self) -> Dict[str, int]:
        """Rate limit 설정 가져오기"""
        # 기본 제한
        defaults = {
            'gemini': {'per_minute': 60, 'per_day': 1500},
            'grok': {'per_minute': 60},
            'groq': {'per_minute': 30, 'per_day': 14400},
            'sambanova': {'per_minute': 60},
            'deepseek': {'per_minute': 60},
            'huggingface': {'per_minute': 100}
        }
        return defaults.get(self.config.engine_type.value, {'per_minute': 60})
    
    def _init_client(self):
        """클라이언트 초기화 (서브클래스에서 구현)"""
        raise NotImplementedError
    
    async def generate(self, prompt: str, user_id: str = "anonymous", 
                      **kwargs) -> APIResponse:
        """텍스트 생성"""
        start_time = time.time()
        
        # 캐시 확인
        use_cache = kwargs.pop('use_cache', True)
        cache_params = {'prompt': prompt, **kwargs}
        
        if use_cache:
            cached = self.cache.get(self.config.engine_type.value, cache_params)
            if cached:
                return APIResponse(
                    status=ResponseStatus.CACHED,
                    data=cached,
                    cached=True,
                    latency=time.time() - start_time
                )
        
        # Rate limit 확인
        can_call, wait_time = self.rate_limiter.check_and_add()
        if not can_call:
            return APIResponse(
                status=ResponseStatus.RATE_LIMITED,
                error=f"Rate limit 초과. {wait_time:.1f}초 후 재시도하세요.",
                metadata={'wait_time': wait_time}
            )
        
        # API 호출
        try:
            response = await self._generate_with_retry(prompt, **kwargs)
            
            # 캐시 저장
            if use_cache:
                self.cache.set(
                    self.config.engine_type.value,
                    cache_params,
                    response
                )
            
            # 사용량 추적
            tokens_estimate = len(prompt.split()) + len(response.split())
            await self.usage_tracker.track(APIUsage(
                api_id=self.config.engine_type.value,
                user_id=user_id,
                timestamp=datetime.now(),
                tokens_used=tokens_estimate,
                success=True,
                latency=time.time() - start_time
            ))
            
            return APIResponse(
                status=ResponseStatus.SUCCESS,
                data=response,
                latency=time.time() - start_time,
                metadata={
                    'model': self.config.model_name,
                    'temperature': kwargs.get('temperature', self.config.temperature)
                }
            )
            
        except Exception as e:
            logger.error(f"{self.config.engine_type.value} 오류: {e}")
            
            # 사용량 추적 (실패)
            await self.usage_tracker.track(APIUsage(
                api_id=self.config.engine_type.value,
                user_id=user_id,
                timestamp=datetime.now(),
                success=False,
                error_type=type(e).__name__,
                latency=time.time() - start_time
            ))
            
            return APIResponse(
                status=ResponseStatus.ERROR,
                error=str(e),
                latency=time.time() - start_time
            )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def _generate_with_retry(self, prompt: str, **kwargs) -> str:
        """재시도 로직이 포함된 API 호출"""
        return await self._generate_response(prompt, **kwargs)
    
    async def _generate_response(self, prompt: str, **kwargs) -> str:
        """실제 API 호출 (서브클래스에서 구현)"""
        raise NotImplementedError


# ============================================================================
# AI 엔진 구현체
# ============================================================================

class GeminiEngine(BaseAIEngine):
    """Google Gemini AI 엔진"""
    
    def __init__(self):
        super().__init__(APIConfig(
            engine_type=AIEngineType.GEMINI,
            model_name="gemini-2.0-flash-exp",
            api_key_id="gemini",
            max_tokens=8192,
            temperature=0.9,
            features=["text", "code", "vision", "function_calling"]
        ))
    
    def _init_client(self):
        """Gemini 클라이언트 초기화"""
        if GEMINI_AVAILABLE:
            genai.configure(api_key=self.config.api_key_id)
            self.client = genai.GenerativeModel(self.config.model_name)
    
    async def _generate_response(self, prompt: str, **kwargs) -> str:
        """Gemini API 호출"""
        if not self.client:
            raise Exception("Gemini 클라이언트가 초기화되지 않았습니다.")
        
        generation_config = genai.types.GenerationConfig(
            temperature=kwargs.get('temperature', self.config.temperature),
            max_output_tokens=kwargs.get('max_tokens', self.config.max_tokens),
            top_p=kwargs.get('top_p', 0.95)
        )
        
        response = await asyncio.to_thread(
            self.client.generate_content,
            prompt,
            generation_config=generation_config
        )
        
        return response.text


class GrokEngine(BaseAIEngine):
    """xAI Grok 엔진"""
    
    def __init__(self):
        super().__init__(APIConfig(
            engine_type=AIEngineType.GROK,
            model_name="grok-2-latest",
            api_key_id="grok",
            max_tokens=131072,
            features=["text", "code", "real_time_info"]
        ))
    
    def _init_client(self):
        """Grok 클라이언트 초기화"""
        if OPENAI_AVAILABLE:
            self.client = AsyncOpenAI(
                api_key=self.config.api_key_id,
                base_url="https://api.x.ai/v1"
            )
    
    async def _generate_response(self, prompt: str, **kwargs) -> str:
        """Grok API 호출"""
        if not self.client:
            raise Exception("Grok 클라이언트가 초기화되지 않았습니다.")
        
        response = await self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": "You are an expert in polymer science and experimental design."},
                {"role": "user", "content": prompt}
            ],
            temperature=kwargs.get('temperature', self.config.temperature),
            max_tokens=kwargs.get('max_tokens', self.config.max_tokens)
        )
        
        return response.choices[0].message.content


class GroqEngine(BaseAIEngine):
    """Groq 초고속 추론 엔진"""
    
    def __init__(self):
        super().__init__(APIConfig(
            engine_type=AIEngineType.GROQ,
            model_name="llama-3.1-70b-versatile",
            api_key_id="groq",
            max_tokens=8192,
            features=["ultra_fast", "streaming", "batch_processing"]
        ))
    
    def _init_client(self):
        """Groq 클라이언트 초기화"""
        if GROQ_AVAILABLE:
            self.client = AsyncGroq(api_key=self.config.api_key_id)
    
    async def _generate_response(self, prompt: str, **kwargs) -> str:
        """Groq API 호출"""
        if not self.client:
            raise Exception("Groq 클라이언트가 초기화되지 않았습니다.")
        
        response = await self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": "You are an expert in polymer science."},
                {"role": "user", "content": prompt}
            ],
            temperature=kwargs.get('temperature', self.config.temperature),
            max_tokens=kwargs.get('max_tokens', self.config.max_tokens)
        )
        
        return response.choices[0].message.content


class SambaNovaEngine(BaseAIEngine):
    """SambaNova 대규모 모델 엔진"""
    
    def __init__(self):
        super().__init__(APIConfig(
            engine_type=AIEngineType.SAMBANOVA,
            model_name="llama-3.1-405b",
            api_key_id="sambanova",
            max_tokens=4096,
            features=["large_scale", "stable", "enterprise"]
        ))
    
    def _init_client(self):
        """SambaNova 클라이언트 초기화"""
        if OPENAI_AVAILABLE:
            self.client = AsyncOpenAI(
                api_key=self.config.api_key_id,
                base_url="https://api.sambanova.ai/v1"
            )
    
    async def _generate_response(self, prompt: str, **kwargs) -> str:
        """SambaNova API 호출"""
        if not self.client:
            raise Exception("SambaNova 클라이언트가 초기화되지 않았습니다.")
        
        response = await self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": "You are an expert in materials science."},
                {"role": "user", "content": prompt}
            ],
            temperature=kwargs.get('temperature', self.config.temperature),
            max_tokens=kwargs.get('max_tokens', self.config.max_tokens)
        )
        
        return response.choices[0].message.content


class DeepSeekEngine(BaseAIEngine):
    """DeepSeek 수학/코드 특화 엔진"""
    
    def __init__(self):
        super().__init__(APIConfig(
            engine_type=AIEngineType.DEEPSEEK,
            model_name="deepseek-chat",
            api_key_id="deepseek",
            max_tokens=16384,
            features=["math", "code", "formula"]
        ))
    
    def _init_client(self):
        """DeepSeek 클라이언트 초기화"""
        if OPENAI_AVAILABLE:
            self.client = AsyncOpenAI(
                api_key=self.config.api_key_id,
                base_url="https://api.deepseek.com/v1"
            )
    
    async def _generate_response(self, prompt: str, **kwargs) -> str:
        """DeepSeek API 호출"""
        if not self.client:
            raise Exception("DeepSeek 클라이언트가 초기화되지 않았습니다.")
        
        response = await self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": "You are an expert in mathematical modeling and polymer calculations."},
                {"role": "user", "content": prompt}
            ],
            temperature=kwargs.get('temperature', self.config.temperature),
            max_tokens=kwargs.get('max_tokens', self.config.max_tokens)
        )
        
        return response.choices[0].message.content


class HuggingFaceEngine(BaseAIEngine):
    """HuggingFace 특수 모델 엔진"""
    
    def __init__(self):
        super().__init__(APIConfig(
            engine_type=AIEngineType.HUGGINGFACE,
            model_name="microsoft/ChemBERTa-77M",
            api_key_id="huggingface",
            features=["chemistry", "materials", "local_inference"]
        ))
        self.pipeline = None
    
    def _init_client(self):
        """HuggingFace 파이프라인 초기화"""
        if HUGGINGFACE_AVAILABLE:
            try:
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.config.model_name,
                    device=0 if torch.cuda.is_available() else -1
                )
            except Exception as e:
                logger.warning(f"로컬 모델 로드 실패, API 사용: {e}")
                # API 폴백
                self.use_api = True
    
    async def _generate_response(self, prompt: str, **kwargs) -> str:
        """HuggingFace 모델 추론"""
        if self.pipeline:
            # 로컬 추론
            result = await asyncio.to_thread(
                self.pipeline,
                prompt,
                max_length=kwargs.get('max_tokens', 512),
                temperature=kwargs.get('temperature', self.config.temperature)
            )
            return result[0]['generated_text']
        else:
            # API 사용
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {self.config.api_key_id}"}
                data = {
                    "inputs": prompt,
                    "parameters": {
                        "max_length": kwargs.get('max_tokens', 512),
                        "temperature": kwargs.get('temperature', self.config.temperature)
                    }
                }
                
                async with session.post(
                    f"https://api-inference.huggingface.co/models/{self.config.model_name}",
                    headers=headers,
                    json=data
                ) as response:
                    result = await response.json()
                    return result[0]['generated_text']


# ============================================================================
# 과학 데이터베이스 클라이언트
# ============================================================================

class ScienceDBClient:
    """과학 데이터베이스 통합 클라이언트"""
    
    def __init__(self):
        self.clients = {}
        self.cache = ResponseCache(ttl=86400)  # 24시간 캐시
        self._initialize_clients()
    
    def _initialize_clients(self):
        """각 DB 클라이언트 초기화"""
        # Materials Project
        if MATERIALS_PROJECT_AVAILABLE:
            mp_key = self._get_api_key('materials_project')
            if mp_key:
                try:
                    self.clients['materials_project'] = MPRester(mp_key)
                    logger.info("Materials Project 클라이언트 초기화 성공")
                except Exception as e:
                    logger.error(f"Materials Project 초기화 실패: {e}")
        
        # PubChem
        if PUBCHEM_AVAILABLE:
            self.clients['pubchem'] = pcp  # API 키 불필요
            logger.info("PubChem 클라이언트 초기화 성공")
        
        # GitHub
        if GITHUB_AVAILABLE:
            github_token = self._get_api_key('github')
            if github_token:
                try:
                    self.clients['github'] = Github(github_token)
                    logger.info("GitHub 클라이언트 초기화 성공")
                except Exception as e:
                    logger.error(f"GitHub 초기화 실패: {e}")
    
    def _get_api_key(self, service: str) -> Optional[str]:
        """API 키 가져오기"""
        # 환경 변수
        key = os.environ.get(f"{service.upper()}_API_KEY")
        if key:
            return key
        
        # Streamlit secrets
        if hasattr(st, 'secrets'):
            try:
                key = st.secrets.get(f"{service}_api_key")
                if key:
                    return key
            except:
                pass
        
        # 세션 상태
        if hasattr(st, 'session_state') and 'api_keys' in st.session_state:
            key = st.session_state.api_keys.get(service)
            if key:
                return key
        
        return None
    
    async def search_materials(self, formula: str = None, 
                             material_id: str = None,
                             properties: List[str] = None) -> List[Dict]:
        """재료 검색"""
        if 'materials_project' not in self.clients:
            return []
        
        # 캐시 확인
        cache_key = {'formula': formula, 'material_id': material_id, 'properties': properties}
        cached = self.cache.get('materials_project', cache_key)
        if cached:
            return cached
        
        try:
            mp = self.clients['materials_project']
            
            # 검색 실행
            if material_id:
                results = [mp.get_structure_by_material_id(material_id)]
            elif formula:
                results = mp.get_structures(formula)
            else:
                return []
            
            # 속성 조회
            materials = []
            for structure in results[:10]:  # 최대 10개
                material = {
                    'formula': structure.composition.reduced_formula,
                    'crystal_system': structure.get_space_group_info()[0],
                    'volume': structure.volume,
                    'density': structure.density
                }
                
                if properties:
                    # 추가 속성 조회
                    props = mp.get_data(structure.composition.reduced_formula, 
                                       prop=properties)
                    if props:
                        material.update(props[0])
                
                materials.append(material)
            
            # 캐시 저장
            self.cache.set('materials_project', cache_key, materials)
            
            return materials
            
        except Exception as e:
            logger.error(f"Materials Project 검색 오류: {e}")
            return []
    
    async def search_compounds(self, name: str = None, 
                             formula: str = None,
                             smiles: str = None) -> List[Dict]:
        """화합물 검색"""
        if 'pubchem' not in self.clients:
            return []
        
        # 캐시 확인
        cache_key = {'name': name, 'formula': formula, 'smiles': smiles}
        cached = self.cache.get('pubchem', cache_key)
        if cached:
            return cached
        
        try:
            compounds = []
            
            if name:
                results = pcp.get_compounds(name, 'name')
            elif formula:
                results = pcp.get_compounds(formula, 'formula')
            elif smiles:
                results = pcp.get_compounds(smiles, 'smiles')
            else:
                return []
            
            for compound in results[:10]:  # 최대 10개
                compounds.append({
                    'cid': compound.cid,
                    'name': compound.iupac_name or compound.synonyms[0] if compound.synonyms else 'Unknown',
                    'formula': compound.molecular_formula,
                    'weight': compound.molecular_weight,
                    'smiles': compound.canonical_smiles,
                    'inchi': compound.inchi
                })
            
            # 캐시 저장
            self.cache.set('pubchem', cache_key, compounds)
            
            return compounds
            
        except Exception as e:
            logger.error(f"PubChem 검색 오류: {e}")
            return []
    
    async def search_github_data(self, query: str, 
                               language: str = "Python") -> List[Dict]:
        """GitHub 과학 데이터 검색"""
        if 'github' not in self.clients:
            return []
        
        try:
            github = self.clients['github']
            
            # 과학 관련 리포지토리 검색
            search_query = f"{query} polymer materials science data language:{language}"
            results = github.search_repositories(query=search_query, sort='stars')
            
            repos = []
            for repo in results[:10]:  # 최대 10개
                repos.append({
                    'name': repo.full_name,
                    'description': repo.description,
                    'url': repo.html_url,
                    'stars': repo.stargazers_count,
                    'language': repo.language,
                    'updated': repo.updated_at.isoformat() if repo.updated_at else None
                })
            
            return repos
            
        except Exception as e:
            logger.error(f"GitHub 검색 오류: {e}")
            return []


# ============================================================================
# 메인 API 관리자
# ============================================================================

class APIManager:
    """통합 API 관리자"""
    
    def __init__(self):
        self.ai_engines = {}
        self.db_client = None
        self.usage_tracker = defaultdict(list)
        self.encryption_manager = EncryptionManager()
        self._initialize()
    
    def _initialize(self):
        """관리자 초기화"""
        # AI 엔진 초기화
        self._init_ai_engines()
        
        # 데이터베이스 클라이언트 초기화
        self.db_client = ScienceDBClient()
        
        logger.info("API Manager 초기화 완료")
    
    def _init_ai_engines(self):
        """AI 엔진 초기화"""
        engines = [
            (AIEngineType.GEMINI, GeminiEngine),
            (AIEngineType.GROK, GrokEngine),
            (AIEngineType.GROQ, GroqEngine),
            (AIEngineType.SAMBANOVA, SambaNovaEngine),
            (AIEngineType.DEEPSEEK, DeepSeekEngine),
            (AIEngineType.HUGGINGFACE, HuggingFaceEngine)
        ]
        
        for engine_type, engine_class in engines:
            try:
                engine = engine_class()
                if engine.is_available:
                    self.ai_engines[engine_type.value] = engine
                    logger.info(f"{engine_type.value} 엔진 초기화 성공")
            except Exception as e:
                logger.warning(f"{engine_type.value} 엔진 초기화 실패: {e}")
    
    # ============================================================================
    # API 키 관리
    # ============================================================================
    
    def set_api_key(self, service: str, key: str) -> bool:
        """API 키 설정"""
        try:
            # 암호화 저장
            encrypted = self.encryption_manager.encrypt(key)
            
            # 세션 상태에 저장
            if 'api_keys' not in st.session_state:
                st.session_state.api_keys = {}
            st.session_state.api_keys[service] = encrypted
            
            # 엔진 재초기화
            if service in [e.value for e in AIEngineType]:
                self._init_ai_engines()
            elif service in ['materials_project', 'github']:
                self.db_client._initialize_clients()
            
            logger.info(f"{service} API 키 설정 완료")
            return True
            
        except Exception as e:
            logger.error(f"API 키 설정 오류: {e}")
            return False
    
    def remove_api_key(self, service: str):
        """API 키 제거"""
        if hasattr(st, 'session_state') and 'api_keys' in st.session_state:
            if service in st.session_state.api_keys:
                del st.session_state.api_keys[service]
    
    def get_configured_services(self) -> Dict[str, bool]:
        """설정된 서비스 목록"""
        services = {}
        
        # AI 엔진
        for engine_type in AIEngineType:
            services[engine_type.value] = engine_type.value in self.ai_engines
        
        # 데이터베이스
        if self.db_client:
            for db_name in ['materials_project', 'pubchem', 'github']:
                services[db_name] = db_name in self.db_client.clients
        
        return services
    
    # ============================================================================
    # AI 엔진 메서드
    # ============================================================================
    
    def get_available_engines(self) -> List[str]:
        """사용 가능한 AI 엔진 목록"""
        return list(self.ai_engines.keys())
    
    async def generate_text(self, engine_id: str, prompt: str, 
                          user_id: str = "anonymous", **kwargs) -> APIResponse:
        """AI 텍스트 생성"""
        if engine_id not in self.ai_engines:
            # 폴백: 사용 가능한 첫 번째 엔진 사용
            available = self.get_available_engines()
            if available:
                engine_id = available[0]
                logger.info(f"{engine_id} 엔진으로 폴백")
            else:
                return APIResponse(
                    status=ResponseStatus.ERROR,
                    error="사용 가능한 AI 엔진이 없습니다."
                )
        
        engine = self.ai_engines[engine_id]
        response = await engine.generate(prompt, user_id, **kwargs)
        
        # 사용량 기록
        self.usage_tracker[engine_id].append({
            'user_id': user_id,
            'timestamp': datetime.now(),
            'success': response.status == ResponseStatus.SUCCESS
        })
        
        return response
    
    async def analyze_experiment(self, engine_id: str, 
                               experiment_data: Dict,
                               user_id: str = "anonymous") -> APIResponse:
        """실험 데이터 분석"""
        prompt = f"""
        다음 실험 데이터를 분석하고 인사이트를 제공해주세요:
        
        실험 유형: {experiment_data.get('type', '알 수 없음')}
        요인: {experiment_data.get('factors', [])}
        반응변수: {experiment_data.get('responses', [])}
        결과: {experiment_data.get('results', {})}
        
        1. 주요 발견사항
        2. 통계적 유의성
        3. 최적 조건
        4. 개선 제안
        """
        
        return await self.generate_text(engine_id, prompt, user_id, 
                                      temperature=0.5)  # 분석은 낮은 온도
    
    async def suggest_next_experiment(self, engine_id: str,
                                    current_results: Dict,
                                    user_id: str = "anonymous") -> APIResponse:
        """다음 실험 제안"""
        prompt = f"""
        현재까지의 실험 결과를 바탕으로 다음 실험을 제안해주세요:
        
        완료된 실험: {current_results.get('completed_runs', 0)}
        현재 최적값: {current_results.get('current_optimum', {})}
        탐색된 영역: {current_results.get('explored_region', [])}
        
        제안 형식:
        1. 추천하는 다음 실험 조건
        2. 기대되는 개선 정도
        3. 위험 요소
        4. 대안적 접근법
        """
        
        return await self.generate_text(engine_id, prompt, user_id,
                                      temperature=0.7)
    
    # ============================================================================
    # 데이터베이스 메서드
    # ============================================================================
    
    async def search_materials(self, **kwargs) -> List[Dict]:
        """재료 검색"""
        if not self.db_client:
            return []
        return await self.db_client.search_materials(**kwargs)
    
    async def search_compounds(self, **kwargs) -> List[Dict]:
        """화합물 검색"""
        if not self.db_client:
            return []
        return await self.db_client.search_compounds(**kwargs)
    
    async def search_github_data(self, **kwargs) -> List[Dict]:
        """GitHub 데이터 검색"""
        if not self.db_client:
            return []
        return await self.db_client.search_github_data(**kwargs)
    
    # ============================================================================
    # 사용량 및 상태 관리
    # ============================================================================
    
    def get_usage_summary(self, user_id: Optional[str] = None,
                         period: str = 'day') -> Dict[str, Dict]:
        """사용량 요약"""
        summary = {}
        
        # AI 엔진별 사용량
        for engine_id, engine in self.ai_engines.items():
            summary[engine_id] = engine.usage_tracker.get_usage_summary(user_id, period)
        
        return summary
    
    def get_rate_limit_status(self) -> Dict[str, Dict]:
        """Rate limit 상태"""
        status = {}
        
        for engine_id, engine in self.ai_engines.items():
            if engine.rate_limiter:
                status[engine_id] = engine.rate_limiter.get_remaining()
        
        return status
    
    def clear_cache(self, cache_type: Optional[str] = None):
        """캐시 초기화"""
        # AI 엔진 캐시
        for engine in self.ai_engines.values():
            engine.cache.clear(cache_type)
        
        # DB 클라이언트 캐시
        if self.db_client:
            self.db_client.cache.clear(cache_type)
        
        logger.info(f"캐시 초기화 완료: {cache_type or '전체'}")
    
    def get_api_status(self) -> Dict[str, Any]:
        """전체 API 상태"""
        return {
            'ai_engines': {
                engine_id: {
                    'available': True,
                    'model': engine.config.model_name,
                    'features': engine.config.features,
                    'rate_limit': engine.rate_limiter.get_remaining() if engine.rate_limiter else None
                }
                for engine_id, engine in self.ai_engines.items()
            },
            'databases': self.get_configured_services(),
            'total_available': len(self.ai_engines) + len(self.db_client.clients if self.db_client else {})
        }


# ============================================================================
# 싱글톤 인스턴스
# ============================================================================

_api_manager: Optional[APIManager] = None


def get_api_manager() -> APIManager:
    """APIManager 싱글톤 인스턴스 반환"""
    global _api_manager
    
    if _api_manager is None:
        _api_manager = APIManager()
    
    return _api_manager


# ============================================================================
# 헬퍼 함수
# ============================================================================

async def ask_ai(prompt: str, engine: str = "gemini", 
                user_id: str = "anonymous", **kwargs) -> str:
    """간편 AI 질문 함수"""
    manager = get_api_manager()
    response = await manager.generate_text(engine, prompt, user_id, **kwargs)
    
    if response.status == ResponseStatus.SUCCESS:
        return response.data
    else:
        return f"오류: {response.error}"


async def analyze_experiment_data(data: Dict, user_id: str = "anonymous") -> Dict:
    """간편 실험 분석 함수"""
    manager = get_api_manager()
    
    # 사용 가능한 첫 번째 엔진 사용
    engines = manager.get_available_engines()
    if not engines:
        return {"error": "사용 가능한 AI 엔진이 없습니다"}
    
    response = await manager.analyze_experiment(engines[0], data, user_id)
    
    if response.status == ResponseStatus.SUCCESS:
        return {"analysis": response.data}
    else:
        return {"error": response.error}


def render_api_status_card():
    """API 상태 카드 렌더링"""
    manager = get_api_manager()
    status = manager.get_api_status()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "활성 AI 엔진",
            len(status['ai_engines']),
            delta=f"/{len(AIEngineType)}"
        )
    
    with col2:
        active_dbs = sum(1 for v in status['databases'].values() if v)
        st.metric(
            "연결된 DB",
            active_dbs,
            delta=f"/{len(status['databases'])}"
        )
    
    with col3:
        st.metric(
            "전체 API",
            status['total_available'],
            delta="활성"
        )


def render_api_configuration():
    """API 설정 UI"""
    st.subheader("🔑 API 설정")
    
    manager = get_api_manager()
    configured = manager.get_configured_services()
    
    # AI 엔진 설정
    with st.expander("AI 엔진 API 키", expanded=False):
        for engine_type in AIEngineType:
            col1, col2 = st.columns([3, 1])
            with col1:
                key = st.text_input(
                    f"{engine_type.value.upper()} API Key",
                    type="password",
                    key=f"api_key_{engine_type.value}",
                    help=f"{engine_type.value} API 키를 입력하세요"
                )
            with col2:
                if st.button("저장", key=f"save_{engine_type.value}"):
                    if key:
                        if manager.set_api_key(engine_type.value, key):
                            show_success(f"{engine_type.value} API 키가 저장되었습니다.")
                        else:
                            show_error("API 키 저장에 실패했습니다.")
                
                status = "✅" if configured.get(engine_type.value) else "❌"
                st.write(f"상태: {status}")
    
    # 데이터베이스 설정
    with st.expander("과학 데이터베이스 API 키", expanded=False):
        db_services = ['materials_project', 'github']
        for service in db_services:
            col1, col2 = st.columns([3, 1])
            with col1:
                key = st.text_input(
                    f"{service.replace('_', ' ').title()} API Key",
                    type="password",
                    key=f"api_key_{service}",
                    help=f"{service} API 키를 입력하세요"
                )
            with col2:
                if st.button("저장", key=f"save_{service}"):
                    if key:
                        if manager.set_api_key(service, key):
                            show_success(f"{service} API 키가 저장되었습니다.")
                        else:
                            show_error("API 키 저장에 실패했습니다.")
                
                status = "✅" if configured.get(service) else "❌"
                st.write(f"상태: {status}")
