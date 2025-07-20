#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 Module Registry - 실험 모듈 중앙 관리 시스템
================================================================================
모든 실험 모듈의 생명주기를 관리하는 핵심 컴포넌트입니다.
동적 로딩, 검증, 의존성 관리, 캐싱 등을 담당합니다.
================================================================================
"""

import os
import sys
import json
import hashlib
import importlib
import importlib.util
import inspect
import threading
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Type, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor
import logging
import traceback
import tempfile
import shutil

# 로깅 설정
logger = logging.getLogger(__name__)

# ==================== 데이터 모델 ====================

@dataclass
class ModuleDependency:
    """모듈 의존성 정보"""
    module_name: str
    version_spec: str  # ">=1.0.0,<2.0.0"
    optional: bool = False
    purpose: str = ""

@dataclass
class ModuleMetadata:
    """모듈 메타데이터"""
    # 식별 정보
    id: str
    name: str
    version: str
    author: str
    
    # 분류 정보
    category: str
    tags: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    description: str = ""
    
    # 기술 정보
    interface_version: str = "1.0.0"
    dependencies: List[ModuleDependency] = field(default_factory=list)
    python_version: str = ">=3.8"
    
    # 품질 정보
    validation_status: str = "pending"  # pending, passed, failed
    performance_grade: str = "?"  # A, B, C, ?
    test_coverage: float = 0.0
    
    # 사용 통계
    download_count: int = 0
    usage_count: int = 0
    rating: float = 0.0
    
    # 시간 정보
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_validated: Optional[datetime] = None

@dataclass
class ModuleInfo:
    """로드된 모듈 정보"""
    metadata: ModuleMetadata
    module_class: Type
    instance: Optional[Any] = None
    file_path: str = ""
    checksum: str = ""
    load_time: Optional[datetime] = None
    error: Optional[str] = None

@dataclass
class ValidationResult:
    """모듈 검증 결과"""
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

# ==================== 저장소 클래스 ====================

class ModuleStore(ABC):
    """모듈 저장소 추상 클래스"""
    
    def __init__(self, store_path: Path):
        self.store_path = store_path
        self.store_path.mkdir(parents=True, exist_ok=True)
        
    @abstractmethod
    def list_modules(self) -> List[str]:
        """저장소의 모듈 목록 반환"""
        pass
        
    @abstractmethod
    def get_module_path(self, module_name: str) -> Optional[Path]:
        """모듈 파일 경로 반환"""
        pass
        
    @abstractmethod
    def add_module(self, module_path: Path, metadata: ModuleMetadata) -> bool:
        """모듈 추가"""
        pass
        
    @abstractmethod
    def remove_module(self, module_name: str) -> bool:
        """모듈 제거"""
        pass

class CoreModuleStore(ModuleStore):
    """내장 모듈 저장소"""
    
    def list_modules(self) -> List[str]:
        """핵심 모듈 목록"""
        modules = []
        core_path = self.store_path / "core"
        if core_path.exists():
            for module_file in core_path.glob("*.py"):
                if module_file.name != "__init__.py":
                    modules.append(module_file.stem)
        return modules
        
    def get_module_path(self, module_name: str) -> Optional[Path]:
        """모듈 경로 반환"""
        module_path = self.store_path / "core" / f"{module_name}.py"
        return module_path if module_path.exists() else None
        
    def add_module(self, module_path: Path, metadata: ModuleMetadata) -> bool:
        """내장 모듈은 추가 불가"""
        logger.warning("내장 모듈 저장소는 읽기 전용입니다.")
        return False
        
    def remove_module(self, module_name: str) -> bool:
        """내장 모듈은 제거 불가"""
        logger.warning("내장 모듈은 제거할 수 없습니다.")
        return False

class UserModuleStore(ModuleStore):
    """사용자 모듈 저장소"""
    
    def __init__(self, store_path: Path, user_id: str):
        super().__init__(store_path)
        self.user_id = user_id
        self.user_path = self.store_path / "user_modules" / user_id
        self.user_path.mkdir(parents=True, exist_ok=True)
        
    def list_modules(self) -> List[str]:
        """사용자 모듈 목록"""
        modules = []
        for module_file in self.user_path.glob("*.py"):
            if module_file.name != "__init__.py":
                modules.append(module_file.stem)
        return modules
        
    def get_module_path(self, module_name: str) -> Optional[Path]:
        """모듈 경로 반환"""
        module_path = self.user_path / f"{module_name}.py"
        return module_path if module_path.exists() else None
        
    def add_module(self, module_path: Path, metadata: ModuleMetadata) -> bool:
        """모듈 추가"""
        try:
            # 모듈 파일 복사
            dest_path = self.user_path / module_path.name
            shutil.copy2(module_path, dest_path)
            
            # 메타데이터 저장
            meta_path = self.user_path / f"{module_path.stem}.meta.json"
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata.__dict__, f, indent=2, default=str)
                
            return True
        except Exception as e:
            logger.error(f"모듈 추가 실패: {e}")
            return False
            
    def remove_module(self, module_name: str) -> bool:
        """모듈 제거"""
        try:
            module_path = self.user_path / f"{module_name}.py"
            meta_path = self.user_path / f"{module_name}.meta.json"
            
            if module_path.exists():
                module_path.unlink()
            if meta_path.exists():
                meta_path.unlink()
                
            return True
        except Exception as e:
            logger.error(f"모듈 제거 실패: {e}")
            return False

class CommunityModuleStore(ModuleStore):
    """커뮤니티 모듈 저장소"""
    
    def __init__(self, store_path: Path):
        super().__init__(store_path)
        self.community_path = self.store_path / "community"
        self.community_path.mkdir(parents=True, exist_ok=True)
        
    def list_modules(self) -> List[str]:
        """커뮤니티 모듈 목록"""
        modules = []
        for category_dir in self.community_path.iterdir():
            if category_dir.is_dir():
                for module_file in category_dir.glob("*.py"):
                    if module_file.name != "__init__.py":
                        modules.append(f"{category_dir.name}/{module_file.stem}")
        return modules
        
    def get_module_path(self, module_name: str) -> Optional[Path]:
        """모듈 경로 반환"""
        if "/" in module_name:
            category, name = module_name.split("/", 1)
            module_path = self.community_path / category / f"{name}.py"
            return module_path if module_path.exists() else None
        return None
        
    def add_module(self, module_path: Path, metadata: ModuleMetadata) -> bool:
        """커뮤니티 모듈 추가"""
        try:
            category_path = self.community_path / metadata.category
            category_path.mkdir(exist_ok=True)
            
            dest_path = category_path / module_path.name
            shutil.copy2(module_path, dest_path)
            
            meta_path = category_path / f"{module_path.stem}.meta.json"
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata.__dict__, f, indent=2, default=str)
                
            return True
        except Exception as e:
            logger.error(f"커뮤니티 모듈 추가 실패: {e}")
            return False
            
    def remove_module(self, module_name: str) -> bool:
        """커뮤니티 모듈 제거"""
        try:
            if "/" in module_name:
                category, name = module_name.split("/", 1)
                module_path = self.community_path / category / f"{name}.py"
                meta_path = self.community_path / category / f"{name}.meta.json"
                
                if module_path.exists():
                    module_path.unlink()
                if meta_path.exists():
                    meta_path.unlink()
                    
                return True
            return False
        except Exception as e:
            logger.error(f"커뮤니티 모듈 제거 실패: {e}")
            return False

# ==================== 모듈 검증기 ====================

class ModuleValidator:
    """모듈 검증 시스템"""
    
    def __init__(self):
        self.required_methods = [
            'get_factors', 'get_responses', 'validate_input',
            'generate_design', 'analyze_results'
        ]
        
    def validate_module(self, module_class: Type) -> ValidationResult:
        """모듈 전체 검증"""
        result = ValidationResult(passed=True)
        
        # 인터페이스 검증
        interface_result = self._validate_interface(module_class)
        if not interface_result.passed:
            result.passed = False
            result.errors.extend(interface_result.errors)
            
        # 메타데이터 검증
        metadata_result = self._validate_metadata(module_class)
        if not metadata_result.passed:
            result.passed = False
            result.errors.extend(metadata_result.errors)
            
        # 보안 검증
        security_result = self._validate_security(module_class)
        if not security_result.passed:
            result.passed = False
            result.errors.extend(security_result.errors)
            
        # 성능 검증
        performance_result = self._validate_performance(module_class)
        result.performance_metrics = performance_result.performance_metrics
        result.warnings.extend(performance_result.warnings)
        
        return result
        
    def _validate_interface(self, module_class: Type) -> ValidationResult:
        """인터페이스 검증"""
        result = ValidationResult(passed=True)
        
        # 필수 메서드 확인
        for method_name in self.required_methods:
            if not hasattr(module_class, method_name):
                result.passed = False
                result.errors.append(f"필수 메서드 '{method_name}' 없음")
            else:
                method = getattr(module_class, method_name)
                if not callable(method):
                    result.passed = False
                    result.errors.append(f"'{method_name}'은(는) 호출 가능한 메서드여야 함")
                    
        # 상속 확인
        from modules.base_module import BaseExperimentModule
        if not issubclass(module_class, BaseExperimentModule):
            result.passed = False
            result.errors.append("BaseExperimentModule을 상속해야 함")
            
        return result
        
    def _validate_metadata(self, module_class: Type) -> ValidationResult:
        """메타데이터 검증"""
        result = ValidationResult(passed=True)
        
        try:
            instance = module_class()
            metadata = instance.metadata
            
            # 필수 필드 확인
            required_fields = ['name', 'version', 'author', 'category']
            for field in required_fields:
                if not metadata.get(field):
                    result.passed = False
                    result.errors.append(f"메타데이터 필드 '{field}' 없음")
                    
            # 버전 형식 확인
            version = metadata.get('version', '')
            if not self._is_valid_version(version):
                result.passed = False
                result.errors.append(f"잘못된 버전 형식: {version}")
                
        except Exception as e:
            result.passed = False
            result.errors.append(f"메타데이터 검증 중 오류: {str(e)}")
            
        return result
        
    def _validate_security(self, module_class: Type) -> ValidationResult:
        """보안 검증"""
        result = ValidationResult(passed=True)
        
        # 모듈 소스 코드 분석
        try:
            source = inspect.getsource(module_class)
            
            # 위험한 함수 검사
            dangerous_patterns = [
                'exec', 'eval', '__import__',
                'open(', 'file(', 'compile(',
                'os.system', 'subprocess',
                'socket.', 'requests.'
            ]
            
            for pattern in dangerous_patterns:
                if pattern in source:
                    result.warnings.append(f"잠재적으로 위험한 패턴 발견: {pattern}")
                    
        except Exception as e:
            result.warnings.append(f"소스 코드 분석 실패: {str(e)}")
            
        return result
        
    def _validate_performance(self, module_class: Type) -> ValidationResult:
        """성능 검증"""
        result = ValidationResult(passed=True)
        
        try:
            # 간단한 성능 테스트
            import time
            instance = module_class()
            
            # 초기화 시간
            start_time = time.time()
            instance.__init__()
            init_time = (time.time() - start_time) * 1000
            
            result.performance_metrics['init_time_ms'] = init_time
            
            # 성능 등급 결정
            if init_time < 100:
                grade = 'A'
            elif init_time < 500:
                grade = 'B'
            else:
                grade = 'C'
                result.warnings.append(f"초기화 시간이 느림: {init_time:.2f}ms")
                
            result.performance_metrics['grade'] = grade
            
        except Exception as e:
            result.warnings.append(f"성능 테스트 실패: {str(e)}")
            
        return result
        
    def _is_valid_version(self, version: str) -> bool:
        """버전 형식 검증"""
        import re
        pattern = r'^\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$'
        return bool(re.match(pattern, version))

# ==================== 모듈 로더 ====================

class ModuleLoader:
    """모듈 동적 로더"""
    
    def __init__(self):
        self.loaded_modules: Dict[str, Any] = {}
        self.load_lock = threading.Lock()
        
    def load_module(self, module_path: Path, module_name: str) -> Tuple[Optional[Type], Optional[str]]:
        """모듈 동적 로드"""
        try:
            # 이미 로드된 경우
            if module_name in self.loaded_modules:
                return self.loaded_modules[module_name], None
                
            # 모듈 스펙 생성
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                return None, "모듈 스펙 생성 실패"
                
            # 모듈 로드
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # BaseExperimentModule 서브클래스 찾기
            from modules.base_module import BaseExperimentModule
            module_class = None
            
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BaseExperimentModule) and 
                    obj != BaseExperimentModule):
                    module_class = obj
                    break
                    
            if module_class is None:
                return None, "BaseExperimentModule 서브클래스를 찾을 수 없음"
                
            # 캐시에 저장
            with self.load_lock:
                self.loaded_modules[module_name] = module_class
                
            return module_class, None
            
        except Exception as e:
            error_msg = f"모듈 로드 실패: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return None, error_msg
            
    def unload_module(self, module_name: str) -> bool:
        """모듈 언로드"""
        try:
            with self.load_lock:
                if module_name in self.loaded_modules:
                    del self.loaded_modules[module_name]
                    
                if module_name in sys.modules:
                    del sys.modules[module_name]
                    
            return True
        except Exception as e:
            logger.error(f"모듈 언로드 실패: {e}")
            return False
            
    def reload_module(self, module_path: Path, module_name: str) -> Tuple[Optional[Type], Optional[str]]:
        """모듈 리로드"""
        self.unload_module(module_name)
        return self.load_module(module_path, module_name)

# ==================== 메인 레지스트리 ====================

class ModuleRegistry:
    """모듈 레지스트리 - 중앙 관리 시스템"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """싱글톤 패턴"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
        
    def __init__(self):
        """레지스트리 초기화"""
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self.modules: Dict[str, ModuleInfo] = {}
        self.stores: Dict[str, ModuleStore] = {}
        self.loader = ModuleLoader()
        self.validator = ModuleValidator()
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.cache = OrderedDict()
        self.cache_size = 50
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 기본 경로 설정
        self.base_path = Path("modules")
        self.base_path.mkdir(exist_ok=True)
        
        # 저장소 초기화
        self._initialize_stores()
        
        logger.info("모듈 레지스트리 초기화 완료")
        
    def _initialize_stores(self):
        """저장소 초기화"""
        # 내장 모듈 저장소
        self.stores['core'] = CoreModuleStore(self.base_path)
        
        # 커뮤니티 모듈 저장소
        self.stores['community'] = CommunityModuleStore(self.base_path)
        
        logger.info("모듈 저장소 초기화 완료")
        
    def initialize_user_store(self, user_id: str):
        """사용자 저장소 초기화"""
        store_key = f"user_{user_id}"
        if store_key not in self.stores:
            self.stores[store_key] = UserModuleStore(self.base_path, user_id)
            logger.info(f"사용자 {user_id}의 모듈 저장소 초기화")
            
    def discover_modules(self, user_id: Optional[str] = None) -> Dict[str, List[str]]:
        """모든 저장소에서 모듈 발견"""
        discovered = {
            'core': [],
            'user': [],
            'community': []
        }
        
        # 내장 모듈 발견
        try:
            discovered['core'] = self.stores['core'].list_modules()
        except Exception as e:
            logger.error(f"내장 모듈 발견 실패: {e}")
            
        # 사용자 모듈 발견
        if user_id:
            self.initialize_user_store(user_id)
            try:
                discovered['user'] = self.stores[f"user_{user_id}"].list_modules()
            except Exception as e:
                logger.error(f"사용자 모듈 발견 실패: {e}")
                
        # 커뮤니티 모듈 발견
        try:
            discovered['community'] = self.stores['community'].list_modules()
        except Exception as e:
            logger.error(f"커뮤니티 모듈 발견 실패: {e}")
            
        # 자동 등록
        for store_type, modules in discovered.items():
            for module_name in modules:
                self._auto_register(module_name, store_type, user_id)
                
        return discovered
        
    def _auto_register(self, module_name: str, store_type: str, user_id: Optional[str] = None):
        """자동 모듈 등록"""
        try:
            # 이미 등록된 경우 스킵
            full_name = f"{store_type}:{module_name}"
            if full_name in self.modules:
                return
                
            # 모듈 경로 찾기
            if store_type == 'core':
                store = self.stores['core']
            elif store_type == 'user' and user_id:
                store = self.stores[f"user_{user_id}"]
            elif store_type == 'community':
                store = self.stores['community']
            else:
                return
                
            module_path = store.get_module_path(module_name)
            if not module_path:
                return
                
            # 메타데이터 로드
            meta_path = module_path.with_suffix('.meta.json')
            if meta_path.exists():
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta_dict = json.load(f)
                    metadata = ModuleMetadata(**meta_dict)
            else:
                # 기본 메타데이터 생성
                metadata = ModuleMetadata(
                    id=full_name,
                    name=module_name,
                    version="1.0.0",
                    author="Unknown",
                    category=store_type
                )
                
            # 모듈 정보 생성
            module_info = ModuleInfo(
                metadata=metadata,
                module_class=type,  # 나중에 로드
                file_path=str(module_path),
                checksum=self._calculate_checksum(module_path)
            )
            
            self.modules[full_name] = module_info
            
        except Exception as e:
            logger.error(f"자동 등록 실패 {module_name}: {e}")
            
    def register_module(self, module_path: Path, metadata: ModuleMetadata, 
                       store_type: str = 'user', user_id: Optional[str] = None) -> bool:
        """모듈 수동 등록"""
        try:
            # 모듈 로드 및 검증
            temp_name = f"temp_{metadata.name}_{datetime.now().timestamp()}"
            module_class, error = self.loader.load_module(module_path, temp_name)
            
            if error:
                logger.error(f"모듈 로드 실패: {error}")
                return False
                
            # 검증
            validation_result = self.validator.validate_module(module_class)
            if not validation_result.passed:
                logger.error(f"모듈 검증 실패: {validation_result.errors}")
                return False
                
            # 적절한 저장소에 추가
            if store_type == 'user' and user_id:
                self.initialize_user_store(user_id)
                store = self.stores[f"user_{user_id}"]
            elif store_type == 'community':
                store = self.stores['community']
            else:
                logger.error(f"잘못된 저장소 타입: {store_type}")
                return False
                
            # 저장소에 모듈 추가
            if not store.add_module(module_path, metadata):
                return False
                
            # 레지스트리에 등록
            full_name = f"{store_type}:{metadata.name}"
            module_info = ModuleInfo(
                metadata=metadata,
                module_class=module_class,
                file_path=str(module_path),
                checksum=self._calculate_checksum(module_path),
                load_time=datetime.now()
            )
            
            self.modules[full_name] = module_info
            
            # 의존성 그래프 업데이트
            self._update_dependency_graph(full_name, metadata.dependencies)
            
            logger.info(f"모듈 '{full_name}' 등록 완료")
            return True
            
        except Exception as e:
            logger.error(f"모듈 등록 실패: {e}\n{traceback.format_exc()}")
            return False
            
    def unregister_module(self, module_name: str, user_id: Optional[str] = None) -> bool:
        """모듈 등록 해제"""
        try:
            if module_name not in self.modules:
                logger.warning(f"모듈 '{module_name}'을(를) 찾을 수 없음")
                return False
                
            module_info = self.modules[module_name]
            
            # 내장 모듈은 제거 불가
            if module_name.startswith('core:'):
                logger.warning("내장 모듈은 제거할 수 없습니다")
                return False
                
            # 의존하는 모듈이 있는지 확인
            dependents = self._get_dependents(module_name)
            if dependents:
                logger.warning(f"다음 모듈이 '{module_name}'에 의존합니다: {dependents}")
                return False
                
            # 저장소에서 제거
            store_type, name = module_name.split(':', 1)
            if store_type == 'user' and user_id:
                store = self.stores.get(f"user_{user_id}")
            elif store_type == 'community':
                store = self.stores.get('community')
            else:
                return False
                
            if store and not store.remove_module(name):
                return False
                
            # 언로드
            self.loader.unload_module(module_name)
            
            # 레지스트리에서 제거
            del self.modules[module_name]
            
            # 의존성 그래프에서 제거
            self._remove_from_dependency_graph(module_name)
            
            # 캐시에서 제거
            if module_name in self.cache:
                del self.cache[module_name]
                
            logger.info(f"모듈 '{module_name}' 제거 완료")
            return True
            
        except Exception as e:
            logger.error(f"모듈 제거 실패: {e}")
            return False
            
    def get_module(self, module_name: str) -> Optional[Any]:
        """모듈 인스턴스 반환"""
        try:
            # 캐시 확인
            if module_name in self.cache:
                self.cache.move_to_end(module_name)
                return self.cache[module_name]
                
            # 모듈 정보 확인
            if module_name not in self.modules:
                logger.warning(f"모듈 '{module_name}'을(를) 찾을 수 없음")
                return None
                
            module_info = self.modules[module_name]
            
            # 로드되지 않은 경우 로드
            if module_info.instance is None:
                if not isinstance(module_info.module_class, type):
                    # 실제 로드 필요
                    module_class, error = self.loader.load_module(
                        Path(module_info.file_path),
                        module_name
                    )
                    if error:
                        logger.error(f"모듈 로드 실패: {error}")
                        return None
                    module_info.module_class = module_class
                    
                # 인스턴스 생성
                module_info.instance = module_info.module_class()
                module_info.load_time = datetime.now()
                
            # 캐시에 추가
            self._add_to_cache(module_name, module_info.instance)
            
            return module_info.instance
            
        except Exception as e:
            logger.error(f"모듈 가져오기 실패: {e}\n{traceback.format_exc()}")
            return None
            
    def list_modules(self, category: Optional[str] = None, 
                    tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """사용 가능한 모듈 목록"""
        module_list = []
        
        for name, info in self.modules.items():
            # 카테고리 필터
            if category and info.metadata.category != category:
                continue
                
            # 태그 필터
            if tags and not any(tag in info.metadata.tags for tag in tags):
                continue
                
            module_list.append({
                'name': name,
                'display_name': info.metadata.name,
                'version': info.metadata.version,
                'author': info.metadata.author,
                'category': info.metadata.category,
                'description': info.metadata.description,
                'tags': info.metadata.tags,
                'rating': info.metadata.rating,
                'usage_count': info.metadata.usage_count,
                'validation_status': info.metadata.validation_status,
                'performance_grade': info.metadata.performance_grade
            })
            
        # 사용 횟수로 정렬
        module_list.sort(key=lambda x: x['usage_count'], reverse=True)
        
        return module_list
        
    def update_module(self, module_name: str, new_path: Path) -> bool:
        """모듈 업데이트"""
        try:
            if module_name not in self.modules:
                logger.warning(f"모듈 '{module_name}'을(를) 찾을 수 없음")
                return False
                
            # 새 체크섬 계산
            new_checksum = self._calculate_checksum(new_path)
            old_checksum = self.modules[module_name].checksum
            
            if new_checksum == old_checksum:
                logger.info("모듈이 변경되지 않았습니다")
                return True
                
            # 모듈 리로드
            module_class, error = self.loader.reload_module(new_path, module_name)
            if error:
                logger.error(f"모듈 리로드 실패: {error}")
                return False
                
            # 재검증
            validation_result = self.validator.validate_module(module_class)
            if not validation_result.passed:
                logger.error(f"업데이트된 모듈 검증 실패: {validation_result.errors}")
                return False
                
            # 정보 업데이트
            module_info = self.modules[module_name]
            module_info.module_class = module_class
            module_info.instance = None  # 재생성 필요
            module_info.checksum = new_checksum
            module_info.metadata.updated_at = datetime.now()
            module_info.metadata.validation_status = "passed"
            
            # 캐시 무효화
            if module_name in self.cache:
                del self.cache[module_name]
                
            logger.info(f"모듈 '{module_name}' 업데이트 완료")
            return True
            
        except Exception as e:
            logger.error(f"모듈 업데이트 실패: {e}")
            return False
            
    def validate_all(self) -> Dict[str, ValidationResult]:
        """모든 모듈 검증"""
        results = {}
        
        for name, info in self.modules.items():
            try:
                # 모듈 로드
                if not isinstance(info.module_class, type):
                    module_class, error = self.loader.load_module(
                        Path(info.file_path),
                        name
                    )
                    if error:
                        results[name] = ValidationResult(
                            passed=False,
                            errors=[f"로드 실패: {error}"]
                        )
                        continue
                    info.module_class = module_class
                    
                # 검증
                result = self.validator.validate_module(info.module_class)
                results[name] = result
                
                # 메타데이터 업데이트
                info.metadata.validation_status = "passed" if result.passed else "failed"
                info.metadata.last_validated = datetime.now()
                if 'grade' in result.performance_metrics:
                    info.metadata.performance_grade = result.performance_metrics['grade']
                    
            except Exception as e:
                results[name] = ValidationResult(
                    passed=False,
                    errors=[f"검증 중 오류: {str(e)}"]
                )
                
        return results
        
    def get_module_info(self, module_name: str) -> Optional[Dict[str, Any]]:
        """모듈 상세 정보"""
        if module_name not in self.modules:
            return None
            
        info = self.modules[module_name]
        return {
            'metadata': info.metadata.__dict__,
            'file_path': info.file_path,
            'checksum': info.checksum,
            'load_time': info.load_time.isoformat() if info.load_time else None,
            'is_loaded': info.instance is not None,
            'dependencies': [dep.__dict__ for dep in info.metadata.dependencies],
            'dependents': list(self._get_dependents(module_name))
        }
        
    def check_dependencies(self, module_name: str) -> Tuple[bool, List[str]]:
        """의존성 확인"""
        if module_name not in self.modules:
            return False, [f"모듈 '{module_name}'을(를) 찾을 수 없음"]
            
        missing = []
        info = self.modules[module_name]
        
        for dep in info.metadata.dependencies:
            if not self._is_dependency_satisfied(dep):
                missing.append(f"{dep.module_name} {dep.version_spec}")
                
        return len(missing) == 0, missing
        
    def _calculate_checksum(self, file_path: Path) -> str:
        """파일 체크섬 계산"""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
        
    def _add_to_cache(self, key: str, value: Any):
        """캐시에 추가"""
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            self.cache[key] = value
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
                
    def _update_dependency_graph(self, module_name: str, dependencies: List[ModuleDependency]):
        """의존성 그래프 업데이트"""
        self.dependency_graph[module_name] = set()
        for dep in dependencies:
            self.dependency_graph[module_name].add(dep.module_name)
            
    def _remove_from_dependency_graph(self, module_name: str):
        """의존성 그래프에서 제거"""
        if module_name in self.dependency_graph:
            del self.dependency_graph[module_name]
            
        # 다른 모듈의 의존성에서도 제거
        for deps in self.dependency_graph.values():
            deps.discard(module_name)
            
    def _get_dependents(self, module_name: str) -> Set[str]:
        """해당 모듈에 의존하는 모듈 목록"""
        dependents = set()
        for mod, deps in self.dependency_graph.items():
            if module_name in deps:
                dependents.add(mod)
        return dependents
        
    def _is_dependency_satisfied(self, dependency: ModuleDependency) -> bool:
        """의존성 충족 여부 확인"""
        # 간단한 구현 - 모듈 존재 여부만 확인
        # 실제로는 버전 비교도 필요
        return any(dependency.module_name in name for name in self.modules.keys())
        
    def get_statistics(self) -> Dict[str, Any]:
        """레지스트리 통계"""
        total_modules = len(self.modules)
        loaded_modules = sum(1 for info in self.modules.values() if info.instance is not None)
        
        by_category = defaultdict(int)
        by_status = defaultdict(int)
        by_grade = defaultdict(int)
        
        for info in self.modules.values():
            by_category[info.metadata.category] += 1
            by_status[info.metadata.validation_status] += 1
            by_grade[info.metadata.performance_grade] += 1
            
        return {
            'total_modules': total_modules,
            'loaded_modules': loaded_modules,
            'cache_size': len(self.cache),
            'by_category': dict(by_category),
            'by_validation_status': dict(by_status),
            'by_performance_grade': dict(by_grade)
        }

# ==================== 편의 함수 ====================

def get_registry() -> ModuleRegistry:
    """레지스트리 인스턴스 반환"""
    return ModuleRegistry()

def list_available_modules(category: Optional[str] = None) -> List[Dict[str, Any]]:
    """사용 가능한 모듈 목록 조회"""
    registry = get_registry()
    return registry.list_modules(category=category)

def load_module(module_name: str) -> Optional[Any]:
    """모듈 로드"""
    registry = get_registry()
    return registry.get_module(module_name)

# ==================== 테스트 코드 ====================

if __name__ == "__main__":
    # 레지스트리 테스트
    registry = get_registry()
    
    # 모듈 발견
    discovered = registry.discover_modules()
    print(f"발견된 모듈: {discovered}")
    
    # 통계
    stats = registry.get_statistics()
    print(f"레지스트리 통계: {stats}")
