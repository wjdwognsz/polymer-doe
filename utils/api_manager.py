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

# 과학 데이터베이스 - 선택적 import
try:
    from mp_api.client import MPRester
    MATERIALS_PROJECT_AVAILABLE = True
except ImportError:
    MATERIALS_PROJECT_AVAILABLE = False

try:
    import pubchempy as pcp
    PUBCHEM_AVAILABLE = True
except ImportError:
    PUBCHEM_AVAILABLE = False

try:
    from github import Github
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False

import pandas as pd
import numpy as np
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from config.app_config import API_CONFIG
from config.local_config import LOCAL_CONFIG
from utils.database_manager import get_database_manager

logger = logging.getLogger(__name__)


# ============================================================================
# 데이터 모델
# ============================================================================

class AIEngineType(Enum):
    """AI 엔진 타입"""
    GEMINI = "gemini"
    GROK = "grok"
    GROQ = "groq"
    DEEPSEEK = "deepseek"
    SAMBANOVA = "sambanova"
    HUGGINGFACE = "huggingface"


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
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 30
    retry_count: int = 3
    cache_ttl: int = 3600  # 1시간


@dataclass
class APIResponse:
    """API 응답"""
    status: ResponseStatus
    data: Optional[Any] = None
    error: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    cached: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    duration: Optional[float] = None


@dataclass
class UsageRecord:
    """사용량 기록"""
    user_id: str
    api_type: str
    endpoint: str
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# 헬퍼 클래스
# ============================================================================

class RateLimiter:
    """API 호출 속도 제한기"""
    
    def __init__(self, calls_per_minute: int = 60, 
                 calls_per_day: Optional[int] = None):
        self.calls_per_minute = calls_per_minute
        self.calls_per_day = calls_per_day
        self.minute_calls = deque()
        self.day_calls = deque()
        self.lock = threading.Lock()
    
    def acquire(self) -> bool:
        """호출 권한 획득"""
        with self.lock:
            now = time.time()
            
            # 분당 제한 확인
            self.minute_calls = deque(
                t for t in self.minute_calls if now - t < 60
            )
            
            if len(self.minute_calls) >= self.calls_per_minute:
                return False
            
            # 일일 제한 확인
            if self.calls_per_day:
                self.day_calls = deque(
                    t for t in self.day_calls if now - t < 86400
                )
                
                if len(self.day_calls) >= self.calls_per_day:
                    return False
            
            # 호출 기록
            self.minute_calls.append(now)
            if self.calls_per_day:
                self.day_calls.append(now)
            
            return True
    
    def get_remaining(self) -> Dict[str, int]:
        """남은 호출 횟수"""
        with self.lock:
            now = time.time()
            
            # 정리
            self.minute_calls = deque(
                t for t in self.minute_calls if now - t < 60
            )
            
            remaining = {
                'per_minute': self.calls_per_minute - len(self.minute_calls)
            }
            
            if self.calls_per_day:
                self.day_calls = deque(
                    t for t in self.day_calls if now - t < 86400
                )
                remaining['per_day'] = self.calls_per_day - len(self.day_calls)
            
            return remaining


class ResponseCache:
    """API 응답 캐시"""
    
    def __init__(self, default_ttl: int = 3600):
        self.cache = {}
        self.timestamps = {}
        self.default_ttl = default_ttl
        self.lock = threading.Lock()
        self.db_manager = get_database_manager()
    
    def _generate_key(self, api_type: str, params: Dict) -> str:
        """캐시 키 생성"""
        # 파라미터를 정렬하여 일관된 키 생성
        sorted_params = json.dumps(params, sort_keys=True)
        key_string = f"{api_type}:{sorted_params}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, api_type: str, params: Dict) -> Optional[Any]:
        """캐시에서 응답 가져오기"""
        key = self._generate_key(api_type, params)
        
        with self.lock:
            # 메모리 캐시 확인
            if key in self.cache:
                timestamp = self.timestamps[key]
                if time.time() - timestamp < self.default_ttl:
                    return self.cache[key]
                else:
                    # 만료된 캐시 제거
                    del self.cache[key]
                    del self.timestamps[key]
            
            # DB 캐시 확인
            cached = self.db_manager.cache_get(f"api:{key}")
            if cached:
                # 메모리 캐시에도 저장
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
        self._initialize()
    
    def _initialize(self):
        """엔진 초기화"""
        # API 키 확인
        if not self.config.api_key:
            self.config.api_key = self._get_api_key()
        
        if not self.config.api_key:
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
        try:
            import streamlit as st
            key_name = f"{self.config.engine_type.value}_key"
            if hasattr(st, 'secrets') and key_name in st.secrets:
                return st.secrets[key_name]
        except:
            pass
        
        return None
    
    def _get_rate_limits(self) -> Dict[str, int]:
        """Rate limit 설정 가져오기"""
        default_limits = {
            AIEngineType.GEMINI: {'per_minute': 60},
            AIEngineType.GROK: {'per_minute': 30},
            AIEngineType.GROQ: {'per_minute': 100},
            AIEngineType.DEEPSEEK: {'per_minute': 60},
            AIEngineType.SAMBANOVA: {'per_minute': 10},
            AIEngineType.HUGGINGFACE: {'per_minute': 100}
        }
        return default_limits.get(self.config.engine_type, {'per_minute': 60})
    
    def _init_client(self):
        """클라이언트 초기화 - 서브클래스에서 구현"""
        raise NotImplementedError
    
    async def generate(self, prompt: str, **kwargs) -> APIResponse:
        """텍스트 생성 - 서브클래스에서 구현"""
        raise NotImplementedError
    
    async def _make_request(self, func: Callable, *args, **kwargs) -> Any:
        """공통 요청 처리"""
        # Rate limit 확인
        if not self.rate_limiter.acquire():
            return APIResponse(
                status=ResponseStatus.RATE_LIMITED,
                error="API 호출 한도 초과"
            )
        
        # 재시도 로직
        @retry(
            stop=stop_after_attempt(self.config.retry_count),
            wait=wait_exponential(multiplier=1, min=4, max=10)
        )
        async def _request():
            return await func(*args, **kwargs)
        
        try:
            start_time = time.time()
            result = await _request()
            duration = time.time() - start_time
            
            return APIResponse(
                status=ResponseStatus.SUCCESS,
                data=result,
                duration=duration
            )
            
        except asyncio.TimeoutError:
            return APIResponse(
                status=ResponseStatus.TIMEOUT,
                error="요청 시간 초과"
            )
        except Exception as e:
            return APIResponse(
                status=ResponseStatus.ERROR,
                error=str(e)
            )


# ============================================================================
# AI 엔진 구현
# ============================================================================

class GeminiEngine(BaseAIEngine):
    """Google Gemini 엔진"""
    
    def __init__(self):
        super().__init__(APIConfig(
            engine_type=AIEngineType.GEMINI,
            model_name="gemini-2.0-flash-exp",
            max_tokens=8192,
            temperature=0.7
        ))
    
    def _init_client(self):
        """Gemini 클라이언트 초기화"""
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai가 설치되지 않았습니다")
        
        genai.configure(api_key=self.config.api_key)
        self.client = genai.GenerativeModel(self.config.model_name)
    
    async def generate(self, prompt: str, **kwargs) -> APIResponse:
        """텍스트 생성"""
        if not self.is_available:
            return APIResponse(
                status=ResponseStatus.ERROR,
                error="Gemini가 초기화되지 않았습니다"
            )
        
        # 캐시 확인
        cache_params = {'prompt': prompt, **kwargs}
        if cached := self.cache.get('gemini', cache_params):
            return APIResponse(
                status=ResponseStatus.CACHED,
                data=cached,
                cached=True
            )
        
        try:
            # API 호출
            start_time = time.time()
            
            # 동기 함수를 비동기로 실행
            response = await asyncio.to_thread(
                self.client.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                    temperature=kwargs.get('temperature', self.config.temperature),
                )
            )
            
            duration = time.time() - start_time
            
            # 응답 처리
            result = {
                'text': response.text,
                'usage': {
                    'prompt_tokens': response.usage_metadata.prompt_token_count,
                    'completion_tokens': response.usage_metadata.candidates_token_count,
                    'total_tokens': response.usage_metadata.total_token_count
                }
            }
            
            # 캐시 저장
            self.cache.set('gemini', cache_params, result['text'])
            
            return APIResponse(
                status=ResponseStatus.SUCCESS,
                data=result['text'],
                usage=result['usage'],
                duration=duration
            )
            
        except Exception as e:
            logger.error(f"Gemini 생성 오류: {str(e)}")
            return APIResponse(
                status=ResponseStatus.ERROR,
                error=str(e)
            )


class GroqEngine(BaseAIEngine):
    """Groq 초고속 추론 엔진"""
    
    def __init__(self):
        super().__init__(APIConfig(
            engine_type=AIEngineType.GROQ,
            model_name="mixtral-8x7b-32768",
            base_url="https://api.groq.com/openai/v1",
            max_tokens=32768
        ))
    
    def _init_client(self):
        """Groq 클라이언트 초기화"""
        if not GROQ_AVAILABLE:
            raise ImportError("groq가 설치되지 않았습니다")
        
        self.client = AsyncGroq(api_key=self.config.api_key)
    
    async def generate(self, prompt: str, **kwargs) -> APIResponse:
        """텍스트 생성"""
        if not self.is_available:
            return APIResponse(
                status=ResponseStatus.ERROR,
                error="Groq가 초기화되지 않았습니다"
            )
        
        try:
            start_time = time.time()
            
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                temperature=kwargs.get('temperature', self.config.temperature),
                stream=kwargs.get('stream', False)
            )
            
            duration = time.time() - start_time
            
            return APIResponse(
                status=ResponseStatus.SUCCESS,
                data=response.choices[0].message.content,
                usage={
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                },
                duration=duration
            )
            
        except Exception as e:
            logger.error(f"Groq 생성 오류: {str(e)}")
            return APIResponse(
                status=ResponseStatus.ERROR,
                error=str(e)
            )


class DeepSeekEngine(BaseAIEngine):
    """DeepSeek 수학/코드 특화 엔진"""
    
    def __init__(self):
        super().__init__(APIConfig(
            engine_type=AIEngineType.DEEPSEEK,
            model_name="deepseek-chat",
            base_url="https://api.deepseek.com/v1"
        ))
    
    def _init_client(self):
        """DeepSeek 클라이언트 초기화"""
        if not OPENAI_AVAILABLE:
            raise ImportError("openai가 설치되지 않았습니다")
        
        self.client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url
        )
    
    async def generate(self, prompt: str, **kwargs) -> APIResponse:
        """텍스트 생성"""
        if not self.is_available:
            return APIResponse(
                status=ResponseStatus.ERROR,
                error="DeepSeek이 초기화되지 않았습니다"
            )
        
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                temperature=kwargs.get('temperature', self.config.temperature)
            )
            
            return APIResponse(
                status=ResponseStatus.SUCCESS,
                data=response.choices[0].message.content,
                usage={
                    'total_tokens': response.usage.total_tokens
                }
            )
            
        except Exception as e:
            return APIResponse(
                status=ResponseStatus.ERROR,
                error=str(e)
            )


class HuggingFaceEngine(BaseAIEngine):
    """HuggingFace 특수 모델 엔진"""
    
    def __init__(self):
        super().__init__(APIConfig(
            engine_type=AIEngineType.HUGGINGFACE,
            model_name="ChemBERTa-77M-MTR",
            cache_ttl=7200  # 2시간
        ))
        self.pipeline = None
    
    def _init_client(self):
        """HuggingFace 파이프라인 초기화"""
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError("transformers가 설치되지 않았습니다")
        
        # 로컬 모델 사용 (오프라인 지원)
        try:
            self.pipeline = pipeline(
                "text-generation",
                model=self.config.model_name,
                device=0 if torch.cuda.is_available() else -1
            )
        except:
            # 온라인 API 사용
            self.use_api = True
    
    async def generate(self, prompt: str, **kwargs) -> APIResponse:
        """텍스트 생성"""
        if not self.is_available:
            return APIResponse(
                status=ResponseStatus.ERROR,
                error="HuggingFace가 초기화되지 않았습니다"
            )
        
        try:
            if self.pipeline:
                # 로컬 모델 사용
                result = await asyncio.to_thread(
                    self.pipeline,
                    prompt,
                    max_length=kwargs.get('max_tokens', 512),
                    temperature=kwargs.get('temperature', 0.7)
                )
                
                return APIResponse(
                    status=ResponseStatus.SUCCESS,
                    data=result[0]['generated_text']
                )
            else:
                # API 사용
                headers = {"Authorization": f"Bearer {self.config.api_key}"}
                response = await asyncio.to_thread(
                    requests.post,
                    f"https://api-inference.huggingface.co/models/{self.config.model_name}",
                    headers=headers,
                    json={"inputs": prompt}
                )
                
                if response.status_code == 200:
                    return APIResponse(
                        status=ResponseStatus.SUCCESS,
                        data=response.json()[0]['generated_text']
                    )
                else:
                    return APIResponse(
                        status=ResponseStatus.ERROR,
                        error=f"API 오류: {response.status_code}"
                    )
                    
        except Exception as e:
            return APIResponse(
                status=ResponseStatus.ERROR,
                error=str(e)
            )


# ============================================================================
# 과학 데이터베이스 클라이언트
# ============================================================================

class ScienceDBClient:
    """과학 데이터베이스 통합 클라이언트"""
    
    def __init__(self):
        self.clients = {}
        self.cache = ResponseCache(default_ttl=86400)  # 24시간
        self._initialize_clients()
    
    def _initialize_clients(self):
        """각 데이터베이스 클라이언트 초기화"""
        # Materials Project
        if MATERIALS_PROJECT_AVAILABLE:
            mp_key = os.environ.get('MATERIALS_PROJECT_API_KEY')
            if mp_key:
                try:
                    self.clients['materials_project'] = MPRester(mp_key)
                    logger.info("Materials Project 클라이언트 초기화 완료")
                except:
                    logger.warning("Materials Project 초기화 실패")
        
        # PubChem (API 키 불필요)
        if PUBCHEM_AVAILABLE:
            self.clients['pubchem'] = pcp
            logger.info("PubChem 클라이언트 초기화 완료")
        
        # GitHub
        if GITHUB_AVAILABLE:
            github_token = os.environ.get('GITHUB_TOKEN')
            if github_token:
                try:
                    self.clients['github'] = Github(github_token)
                    logger.info("GitHub 클라이언트 초기화 완료")
                except:
                    logger.warning("GitHub 초기화 실패")
    
    async def search_materials(self, formula: Optional[str] = None,
                             property_filters: Optional[Dict] = None) -> List[Dict]:
        """Materials Project에서 재료 검색"""
        if 'materials_project' not in self.clients:
            return []
        
        # 캐시 확인
        cache_params = {'formula': formula, 'filters': property_filters}
        if cached := self.cache.get('materials', cache_params):
            return cached
        
        try:
            mp = self.clients['materials_project']
            
            # 검색 실행
            if formula:
                results = await asyncio.to_thread(
                    mp.materials.search,
                    formula=formula,
                    fields=['material_id', 'formula_pretty', 'band_gap', 
                           'density', 'volume', 'spacegroup']
                )
            else:
                results = []
            
            # 결과 변환
            materials = []
            for mat in results[:20]:  # 최대 20개
                materials.append({
                    'id': mat.material_id,
                    'formula': mat.formula_pretty,
                    'band_gap': mat.band_gap,
                    'density': mat.density,
                    'volume': mat.volume,
                    'spacegroup': mat.spacegroup.symbol if mat.spacegroup else None
                })
            
            # 캐시 저장
            self.cache.set('materials', cache_params, materials)
            
            return materials
            
        except Exception as e:
            logger.error(f"Materials Project 검색 오류: {str(e)}")
            return []
    
    async def search_compounds(self, name: Optional[str] = None,
                             formula: Optional[str] = None,
                             smiles: Optional[str] = None) -> List[Dict]:
        """PubChem에서 화합물 검색"""
        if 'pubchem' not in self.clients:
            return []
        
        try:
            compounds = []
            
            if name:
                results = await asyncio.to_thread(
                    pcp.get_compounds, name, 'name'
                )
            elif formula:
                results = await asyncio.to_thread(
                    pcp.get_compounds, formula, 'formula'
                )
            elif smiles:
                results = await asyncio.to_thread(
                    pcp.get_compounds, smiles, 'smiles'
                )
            else:
                return []
            
            for comp in results[:10]:  # 최대 10개
                compounds.append({
                    'cid': comp.cid,
                    'name': comp.iupac_name or (comp.synonyms[0] if comp.synonyms else ''),
                    'formula': comp.molecular_formula,
                    'weight': comp.molecular_weight,
                    'smiles': comp.canonical_smiles,
                    'inchi': comp.inchi
                })
            
            return compounds
            
        except Exception as e:
            logger.error(f"PubChem 검색 오류: {str(e)}")
            return []
    
    async def search_github_repos(self, query: str, 
                                language: str = "python") -> List[Dict]:
        """GitHub에서 저장소 검색"""
        if 'github' not in self.clients:
            return []
        
        try:
            gh = self.clients['github']
            
            # 고분자 관련 저장소 검색
            search_query = f"{query} polymer in:readme language:{language}"
            
            results = await asyncio.to_thread(
                lambda: list(gh.search_repositories(search_query, sort='stars')[:10])
            )
            
            repos = []
            for repo in results:
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
            logger.error(f"GitHub 검색 오류: {str(e)}")
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
            (AIEngineType.GROQ, GroqEngine),
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
                logger.warning(f"{engine_type.value} 엔진 초기화 실패: {str(e)}")
    
    # ============================================================================
    # AI 엔진 메서드
    # ============================================================================
    
    def get_available_engines(self) -> List[str]:
        """사용 가능한 AI 엔진 목록"""
        return list(self.ai_engines.keys())
    
    async def generate_text(self, engine_id: str, prompt: str, 
                          user_id: str, **kwargs) -> APIResponse:
        """AI 텍스트 생성"""
        if engine_id not in self.ai_engines:
            # 폴백: 사용 가능한 첫 번째 엔진 사용
            if self.ai_engines:
                engine_id = list(self.ai_engines.keys())[0]
                logger.info(f"엔진 폴백: {engine_id}")
            else:
                return APIResponse(
                    status=ResponseStatus.ERROR,
                    error="사용 가능한 AI 엔진이 없습니다"
                )
        
        engine = self.ai_engines[engine_id]
        
        # 요청 시작
        start_time = time.time()
        response = await engine.generate(prompt, **kwargs)
        
        # 사용량 기록
        self._record_usage(
            user_id=user_id,
            api_type=engine_id,
            endpoint='generate',
            tokens_used=response.usage.get('total_tokens') if response.usage else None,
            success=response.status == ResponseStatus.SUCCESS,
            error_message=response.error
        )
        
        return response
    
    async def analyze_experiment(self, engine_id: str, experiment_data: Dict,
                               user_id: str) -> APIResponse:
        """실험 데이터 AI 분석"""
        # 프롬프트 생성
        prompt = self._create_experiment_prompt(experiment_data)
        
        # AI 생성
        response = await self.generate_text(
            engine_id, prompt, user_id,
            temperature=0.3,  # 더 일관된 분석을 위해 낮은 온도
            max_tokens=2000
        )
        
        # 응답 파싱
        if response.status == ResponseStatus.SUCCESS:
            try:
                analysis = self._parse_analysis_response(response.data)
                response.data = analysis
            except:
                logger.warning("분석 응답 파싱 실패, 원본 텍스트 반환")
        
        return response
    
    def _create_experiment_prompt(self, experiment_data: Dict) -> str:
        """실험 분석 프롬프트 생성"""
        prompt = f"""
        다음 고분자 실험 데이터를 분석해주세요:
        
        실험 제목: {experiment_data.get('title', 'N/A')}
        실험 유형: {experiment_data.get('type', 'N/A')}
        
        요인(Factors):
        {json.dumps(experiment_data.get('factors', {}), indent=2, ensure_ascii=False)}
        
        반응변수(Responses):
        {json.dumps(experiment_data.get('responses', {}), indent=2, ensure_ascii=False)}
        
        실험 결과:
        {experiment_data.get('results_summary', 'N/A')}
        
        다음을 포함하여 분석해주세요:
        1. 주요 발견사항
        2. 요인 간 상호작용
        3. 최적 조건 추천
        4. 추가 실험 제안
        5. 주의사항
        
        JSON 형식으로 응답해주세요.
        """
        return prompt
    
    def _parse_analysis_response(self, response_text: str) -> Dict:
        """AI 분석 응답 파싱"""
        # JSON 블록 추출 시도
        import re
        json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass
        
        # 직접 JSON 파싱 시도
        try:
            return json.loads(response_text)
        except:
            # 텍스트 응답을 구조화
            return {
                'analysis': response_text,
                'structured': False
            }
    
    # ============================================================================
    # 데이터베이스 메서드
    # ============================================================================
    
    async def search_materials(self, **kwargs) -> List[Dict]:
        """재료 데이터베이스 검색"""
        if not self.db_client:
            return []
        
        results = await self.db_client.search_materials(**kwargs)
        
        # 사용량 기록
        self._record_usage(
            user_id=kwargs.get('user_id', 'anonymous'),
            api_type='materials_project',
            endpoint='search',
            success=bool(results)
        )
        
        return results
    
    async def search_compounds(self, **kwargs) -> List[Dict]:
        """화합물 데이터베이스 검색"""
        if not self.db_client:
            return []
        
        results = await self.db_client.search_compounds(**kwargs)
        
        # 사용량 기록
        self._record_usage(
            user_id=kwargs.get('user_id', 'anonymous'),
            api_type='pubchem',
            endpoint='search',
            success=bool(results)
        )
        
        return results
    
    async def search_github(self, query: str, **kwargs) -> List[Dict]:
        """GitHub 저장소 검색"""
        if not self.db_client:
            return []
        
        results = await self.db_client.search_github_repos(query, **kwargs)
        
        # 사용량 기록
        self._record_usage(
            user_id=kwargs.get('user_id', 'anonymous'),
            api_type='github',
            endpoint='search',
            success=bool(results)
        )
        
        return results
    
    # ============================================================================
    # 사용량 추적
    # ============================================================================
    
    def _record_usage(self, user_id: str, api_type: str, endpoint: str,
                     tokens_used: Optional[int] = None,
                     cost: Optional[float] = None,
                     success: bool = True,
                     error_message: Optional[str] = None):
        """API 사용량 기록"""
        record = UsageRecord(
            user_id=user_id,
            api_type=api_type,
            endpoint=endpoint,
            tokens_used=tokens_used,
            cost=cost or self._calculate_cost(api_type, tokens_used),
            success=success,
            error_message=error_message
        )
        
        # 메모리에 저장
        self.usage_tracker[api_type].append(record)
        
        # DB에 저장 (비동기)
        try:
            db_manager = get_database_manager()
            db_manager.log_api_usage(asdict(record))
        except:
            logger.warning("사용량 DB 저장 실패")
    
    def _calculate_cost(self, api_type: str, tokens: Optional[int]) -> float:
        """API 비용 계산"""
        if not tokens:
            return 0.0
        
        # 1000 토큰당 비용 (USD)
        cost_per_1k = {
            'gemini': 0.0,  # 무료
            'groq': 0.0,     # 무료
            'deepseek': 0.002,
            'sambanova': 0.001,
            'huggingface': 0.0  # 무료
        }
        
        rate = cost_per_1k.get(api_type, 0.0)
        return (tokens / 1000) * rate
    
    def get_usage_summary(self, user_id: Optional[str] = None,
                         period: str = 'day') -> Dict[str, Any]:
        """사용량 요약"""
        # 기간 계산
        now = datetime.now()
        if period == 'day':
            start_time = now - timedelta(days=1)
        elif period == 'week':
            start_time = now - timedelta(weeks=1)
        elif period == 'month':
            start_time = now - timedelta(days=30)
        else:
            start_time = datetime.min
        
        summary = defaultdict(lambda: {
            'requests': 0,
            'tokens': 0,
            'cost': 0.0,
            'errors': 0,
            'success_rate': 0.0
        })
        
        # 집계
        for api_type, records in self.usage_tracker.items():
            filtered = [r for r in records if r.timestamp >= start_time]
            if user_id:
                filtered = [r for r in filtered if r.user_id == user_id]
            
            if filtered:
                total = len(filtered)
                success = sum(1 for r in filtered if r.success)
                
                summary[api_type] = {
                    'requests': total,
                    'tokens': sum(r.tokens_used or 0 for r in filtered),
                    'cost': sum(r.cost or 0 for r in filtered),
                    'errors': total - success,
                    'success_rate': (success / total * 100) if total > 0 else 0
                }
        
        return dict(summary)
    
    # ============================================================================
    # 유틸리티
    # ============================================================================
    
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
                'model': engine.config.model_name,
                'rate_limit': engine.rate_limiter.get_remaining() if engine.rate_limiter else None
            }
            if engine.is_available:
                status['total_available'] += 1
        
        # 데이터베이스 상태
        if self.db_client:
            for db_name in self.db_client.clients:
                status['databases'][db_name] = {'available': True}
                status['total_available'] += 1
        
        return status
    
    def set_api_key(self, api_type: str, api_key: str) -> bool:
        """API 키 동적 설정"""
        # 환경 변수 설정
        os.environ[f"{api_type.upper()}_API_KEY"] = api_key
        
        # 엔진 재초기화
        self._init_ai_engines()
        
        # 성공 여부 확인
        if api_type in ['materials_project', 'github']:
            self.db_client._initialize_clients()
        
        return api_type in self.ai_engines or api_type in self.db_client.clients
    
    def clear_cache(self, cache_type: Optional[str] = None):
        """캐시 초기화"""
        # AI 엔진 캐시
        for engine in self.ai_engines.values():
            engine.cache.clear(cache_type)
        
        # DB 클라이언트 캐시
        if self.db_client:
            self.db_client.cache.clear(cache_type)
        
        logger.info(f"캐시 초기화 완료: {cache_type or '전체'}")


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
        return response.data
    else:
        return {"error": response.error}
