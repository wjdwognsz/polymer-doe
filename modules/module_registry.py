"""
모듈 레지스트리 - 모든 실험 모듈의 중앙 관리 시스템
플러그인 아키텍처의 핵심으로 플랫폼의 무한 확장성을 실현
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
from typing import Dict, List, Optional, Any, Type, Tuple, Set, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import traceback
import shutil
from enum import Enum

from modules.base_module import (
    BaseExperimentModule, ModuleValidationError, 
    ModuleCompatibilityError, check_module_compatibility
)

logger = logging.getLogger(__name__)


# ==================== Enums ====================

class ModuleStatus(str, Enum):
    """모듈 상태"""
    PENDING = "pending"
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"
    DISABLED = "disabled"


class StoreType(str, Enum):
    """저장소 타입"""
    CORE = "core"
    USER = "user"
    COMMUNITY = "community"


class ValidationStatus(str, Enum):
    """검증 상태"""
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


# ==================== 데이터 모델 ====================

@dataclass
class ModuleDependency:
    """모듈 의존성 정보"""
    module_name: str
    version_spec: str  # ">=1.0.0,<2.0.0"
    optional: bool = False
    purpose: str = ""
    
    def is_satisfied_by(self, version: str) -> bool:
        """버전 만족 여부 확인"""
        from packaging import version as pkg_version
        from packaging.specifiers import SpecifierSet
        
        try:
            spec = SpecifierSet(self.version_spec)
            return pkg_version.parse(version) in spec
        except:
            return False


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
    validation_status: ValidationStatus = ValidationStatus.PENDING
    performance_grade: str = "?"  # A, B, C, ?
    test_coverage: float = 0.0
    reliability_score: float = 0.0
    
    # 사용 통계
    download_count: int = 0
    usage_count: int = 0
    rating: float = 0.0
    review_count: int = 0
    
    # 시간 정보
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_validated: Optional[datetime] = None
    
    @classmethod
    def from_module(cls, module: BaseExperimentModule) -> 'ModuleMetadata':
        """모듈 인스턴스에서 메타데이터 추출"""
        module_info = module.get_module_info()
        
        # 의존성 파싱
        deps = []
        for dep_str in module_info.get('dependencies', []):
            if isinstance(dep_str, str) and '==' in dep_str or '>=' in dep_str:
                parts = dep_str.split('>=') if '>=' in dep_str else dep_str.split('==')
                deps.append(ModuleDependency(
                    module_name=parts[0].strip(),
                    version_spec=f">={parts[1].strip()}" if '>=' in dep_str else f"=={parts[1].strip()}"
                ))
        
        return cls(
            id=module_info.get('module_id', ''),
            name=module_info.get('name', ''),
            version=module_info.get('version', '1.0.0'),
            author=module_info.get('author', ''),
            category=module_info.get('category', 'general'),
            tags=module_info.get('tags', []),
            description=module_info.get('description', ''),
            dependencies=deps
        )


@dataclass
class ModuleInfo:
    """모듈 정보"""
    name: str
    metadata: ModuleMetadata
    store_type: StoreType
    file_path: str
    module_class: Optional[Type[BaseExperimentModule]] = None
    instance: Optional[BaseExperimentModule] = None
    status: ModuleStatus = ModuleStatus.PENDING
    load_time: Optional[datetime] = None
    error_message: Optional[str] = None
    cache_key: Optional[str] = None


@dataclass
class ValidationResult:
    """검증 결과"""
    passed: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    security_issues: List[str] = field(default_factory=list)


# ==================== 모듈 저장소 ====================

class ModuleStore:
    """모듈 저장소 기본 클래스"""
    
    def __init__(self, store_type: StoreType, base_path: Path):
        self.store_type = store_type
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.modules: Dict[str, ModuleInfo] = {}
        self._lock = threading.Lock()
    
    def discover(self) -> List[str]:
        """모듈 발견"""
        discovered = []
        
        for item in self.base_path.iterdir():
            if item.is_dir() and not item.name.startswith('_'):
                module_file = item / "__init__.py"
                if not module_file.exists():
                    module_file = item / f"{item.name}.py"
                
                if module_file.exists():
                    discovered.append(item.name)
                    logger.info(f"발견된 모듈: {self.store_type.value}/{item.name}")
        
        return discovered
    
    def get_module_path(self, module_name: str) -> Optional[Path]:
        """모듈 경로 반환"""
        module_dir = self.base_path / module_name
        
        if module_dir.exists():
            init_file = module_dir / "__init__.py"
            if init_file.exists():
                return init_file
            
            module_file = module_dir / f"{module_name}.py"
            if module_file.exists():
                return module_file
        
        # 단일 파일 모듈
        single_file = self.base_path / f"{module_name}.py"
        if single_file.exists():
            return single_file
        
        return None
    
    def add_module(self, module_info: ModuleInfo) -> bool:
        """모듈 추가"""
        with self._lock:
            self.modules[module_info.name] = module_info
            return True
    
    def remove_module(self, module_name: str) -> bool:
        """모듈 제거"""
        with self._lock:
            if module_name in self.modules:
                del self.modules[module_name]
                return True
            return False
    
    def list_modules(self) -> List[ModuleInfo]:
        """모듈 목록"""
        with self._lock:
            return list(self.modules.values())


class CoreModuleStore(ModuleStore):
    """내장 모듈 저장소"""
    
    def __init__(self, base_path: Path):
        super().__init__(StoreType.CORE, base_path / "core")
        self.readonly = True


class UserModuleStore(ModuleStore):
    """사용자 모듈 저장소"""
    
    def __init__(self, base_path: Path, user_id: str):
        super().__init__(StoreType.USER, base_path / "user" / user_id)
        self.user_id = user_id
        self.readonly = False


class CommunityModuleStore(ModuleStore):
    """커뮤니티 모듈 저장소"""
    
    def __init__(self, base_path: Path):
        super().__init__(StoreType.COMMUNITY, base_path / "community")
        self.readonly = False
        self.sync_enabled = True


# ==================== 모듈 검증기 ====================

class ModuleValidator:
    """모듈 검증기"""
    
    def __init__(self):
        self.required_methods = [
            'get_factors', 'get_responses', 'validate_input',
            'generate_design', 'analyze_results'
        ]
        self.dangerous_imports = [
            'os', 'subprocess', 'socket', 'requests',
            '__import__', 'exec', 'eval', 'compile'
        ]
    
    def validate_module(self, module_class: Type, 
                       module_path: Path) -> ValidationResult:
        """모듈 전체 검증"""
        result = ValidationResult()
        
        # 1. 인터페이스 검증
        interface_result = self._validate_interface(module_class)
        result.errors.extend(interface_result.errors)
        result.warnings.extend(interface_result.warnings)
        
        # 2. 메타데이터 검증
        metadata_result = self._validate_metadata(module_class)
        result.errors.extend(metadata_result.errors)
        result.warnings.extend(metadata_result.warnings)
        
        # 3. 보안 검증
        security_result = self._validate_security(module_class, module_path)
        result.errors.extend(security_result.errors)
        result.warnings.extend(security_result.warnings)
        result.security_issues.extend(security_result.security_issues)
        
        # 4. 성능 검증
        performance_result = self._validate_performance(module_class)
        result.performance_metrics.update(performance_result.performance_metrics)
        result.warnings.extend(performance_result.warnings)
        
        # 최종 판정
        result.passed = len(result.errors) == 0
        
        return result
    
    def _validate_interface(self, module_class: Type) -> ValidationResult:
        """인터페이스 검증"""
        result = ValidationResult()
        
        # BaseExperimentModule 상속 확인
        if not issubclass(module_class, BaseExperimentModule):
            result.passed = False
            result.errors.append("BaseExperimentModule을 상속해야 합니다")
            return result
        
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
        
        return result
    
    def _validate_metadata(self, module_class: Type) -> ValidationResult:
        """메타데이터 검증"""
        result = ValidationResult()
        
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
    
    def _validate_security(self, module_class: Type, 
                          module_path: Path) -> ValidationResult:
        """보안 검증"""
        result = ValidationResult()
        
        try:
            # 소스 코드 읽기
            with open(module_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # 위험한 import 검사
            import ast
            tree = ast.parse(source_code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in self.dangerous_imports:
                            result.security_issues.append(
                                f"보안 위험: '{alias.name}' 모듈 사용"
                            )
                            result.warnings.append(
                                f"위험한 모듈 '{alias.name}' 사용 감지"
                            )
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module in self.dangerous_imports:
                        result.security_issues.append(
                            f"보안 위험: '{node.module}' 모듈에서 import"
                        )
                        result.warnings.append(
                            f"위험한 모듈 '{node.module}'에서 import 감지"
                        )
            
            # 위험한 함수 호출 검사
            dangerous_calls = ['eval', 'exec', 'compile', '__import__']
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in dangerous_calls:
                        result.security_issues.append(
                            f"보안 위험: '{node.func.id}' 함수 호출"
                        )
                        result.warnings.append(
                            f"위험한 함수 '{node.func.id}' 호출 감지"
                        )
        
        except Exception as e:
            result.warnings.append(f"보안 검증 중 오류: {str(e)}")
        
        return result
    
    def _validate_performance(self, module_class: Type) -> ValidationResult:
        """성능 검증"""
        result = ValidationResult()
        
        try:
            import time
            import psutil
            import os
            
            # 현재 프로세스
            process = psutil.Process(os.getpid())
            
            # 초기화 성능 측정
            start_time = time.time()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            instance = module_class()
            
            init_time = (time.time() - start_time) * 1000  # ms
            memory_used = (process.memory_info().rss / 1024 / 1024) - start_memory  # MB
            
            result.performance_metrics['init_time_ms'] = init_time
            result.performance_metrics['memory_used_mb'] = memory_used
            
            # 성능 등급 결정
            if init_time < 100:
                grade = 'A'
            elif init_time < 500:
                grade = 'B'
            else:
                grade = 'C'
                result.warnings.append(f"초기화 시간이 느림: {init_time:.2f}ms")
            
            result.performance_metrics['grade'] = grade
            
            # 메모리 사용량 확인
            if memory_used > 50:
                result.warnings.append(f"메모리 사용량이 높음: {memory_used:.2f}MB")
        
        except Exception as e:
            result.warnings.append(f"성능 검증 중 오류: {str(e)}")
            result.performance_metrics['grade'] = '?'
        
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
        self.loaded_modules: Dict[str, Type] = {}
        self.load_lock = threading.Lock()
        self._import_cache: Dict[str, Any] = {}
    
    def load_module(self, module_path: Path, 
                   module_name: str) -> Tuple[Optional[Type], Optional[str]]:
        """모듈 동적 로드"""
        try:
            with self.load_lock:
                # 캐시 확인
                cache_key = f"{module_path}:{module_name}"
                if cache_key in self._import_cache:
                    return self._import_cache[cache_key], None
                
                # 모듈 스펙 생성
                spec = importlib.util.spec_from_file_location(
                    f"dynamic_module_{module_name}", 
                    module_path
                )
                
                if spec is None or spec.loader is None:
                    return None, "모듈 스펙 생성 실패"
                
                # 모듈 로드
                module = importlib.util.module_from_spec(spec)
                
                # 임시로 sys.modules에 추가
                module_key = f"dynamic_module_{module_name}"
                sys.modules[module_key] = module
                
                try:
                    spec.loader.exec_module(module)
                    
                    # BaseExperimentModule 서브클래스 찾기
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
                    self._import_cache[cache_key] = module_class
                    self.loaded_modules[module_name] = module_class
                    
                    return module_class, None
                    
                finally:
                    # 임시 모듈 제거 (선택적)
                    if module_key in sys.modules:
                        del sys.modules[module_key]
        
        except SyntaxError as e:
            return None, f"구문 오류: {str(e)}"
        except ImportError as e:
            return None, f"Import 오류: {str(e)}"
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
                
                # 캐시에서도 제거
                keys_to_remove = [k for k in self._import_cache.keys() 
                                 if k.endswith(f":{module_name}")]
                for key in keys_to_remove:
                    del self._import_cache[key]
                
                return True
        except Exception as e:
            logger.error(f"모듈 언로드 실패: {e}")
            return False
    
    def reload_module(self, module_path: Path, 
                     module_name: str) -> Tuple[Optional[Type], Optional[str]]:
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
        self.stores: Dict[StoreType, ModuleStore] = {}
        self.loader = ModuleLoader()
        self.validator = ModuleValidator()
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.cache = OrderedDict()
        self.cache_size = 50
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 기본 경로 설정
        self.base_path = Path("modules")
        self.base_path.mkdir(exist_ok=True)
        
        # 이벤트 핸들러
        self.event_handlers: Dict[str, List[callable]] = defaultdict(list)
        
        logger.info("ModuleRegistry 초기화 완료")
    
    def initialize(self, config: Optional[Dict[str, Any]] = None):
        """레지스트리 초기화 및 설정"""
        if config:
            self.base_path = Path(config.get('base_path', 'modules'))
            self.cache_size = config.get('cache_size', 50)
        
        # 저장소 초기화
        self._init_stores()
        
        # 핵심 모듈 자동 발견
        self.discover_modules()
        
        logger.info(f"레지스트리 초기화 완료: {len(self.modules)}개 모듈")
    
    def _init_stores(self):
        """저장소 초기화"""
        # Core 저장소
        self.stores[StoreType.CORE] = CoreModuleStore(self.base_path)
        
        # Community 저장소
        self.stores[StoreType.COMMUNITY] = CommunityModuleStore(self.base_path)
        
        logger.info("모듈 저장소 초기화 완료")
    
    def init_user_store(self, user_id: str):
        """사용자 저장소 초기화"""
        if user_id:
            self.stores[StoreType.USER] = UserModuleStore(self.base_path, user_id)
            logger.info(f"사용자 저장소 초기화: {user_id}")
    
    def discover_modules(self, user_id: Optional[str] = None) -> Dict[str, List[str]]:
        """모듈 자동 발견"""
        discovered = {
            'core': [],
            'user': [],
            'community': []
        }
        
        # Core 모듈 발견
        if StoreType.CORE in self.stores:
            core_modules = self.stores[StoreType.CORE].discover()
            for module_name in core_modules:
                if self._register_discovered_module(module_name, StoreType.CORE):
                    discovered['core'].append(module_name)
        
        # User 모듈 발견
        if user_id:
            self.init_user_store(user_id)
            if StoreType.USER in self.stores:
                user_modules = self.stores[StoreType.USER].discover()
                for module_name in user_modules:
                    if self._register_discovered_module(module_name, StoreType.USER):
                        discovered['user'].append(module_name)
        
        # Community 모듈 발견
        if StoreType.COMMUNITY in self.stores:
            community_modules = self.stores[StoreType.COMMUNITY].discover()
            for module_name in community_modules:
                if self._register_discovered_module(module_name, StoreType.COMMUNITY):
                    discovered['community'].append(module_name)
        
        logger.info(f"모듈 발견 완료: Core={len(discovered['core'])}, "
                   f"User={len(discovered['user'])}, Community={len(discovered['community'])}")
        
        return discovered
    
    def _register_discovered_module(self, module_name: str, 
                                  store_type: StoreType) -> bool:
        """발견된 모듈 등록"""
        try:
            store = self.stores[store_type]
            module_path = store.get_module_path(module_name)
            
            if not module_path:
                logger.warning(f"모듈 경로를 찾을 수 없음: {module_name}")
                return False
            
            # 중복 확인
            full_name = f"{store_type.value}.{module_name}"
            if full_name in self.modules:
                logger.debug(f"이미 등록된 모듈: {full_name}")
                return False
            
            # 모듈 정보 생성
            module_info = ModuleInfo(
                name=full_name,
                metadata=ModuleMetadata(
                    id=full_name,
                    name=module_name,
                    version="1.0.0",
                    author="",
                    category=store_type.value
                ),
                store_type=store_type,
                file_path=str(module_path)
            )
            
            # 레지스트리에 추가
            self.modules[full_name] = module_info
            store.add_module(module_info)
            
            # 이벤트 발생
            self._emit_event('module_discovered', {
                'module_name': full_name,
                'store_type': store_type.value
            })
            
            return True
            
        except Exception as e:
            logger.error(f"모듈 등록 실패 ({module_name}): {str(e)}")
            return False
    
    def register_module(self, module_path: Union[str, Path], 
                       store_type: StoreType = StoreType.USER,
                       validate: bool = True) -> Tuple[bool, Optional[str]]:
        """모듈 수동 등록"""
        try:
            module_path = Path(module_path)
            
            if not module_path.exists():
                return False, "모듈 파일이 존재하지 않습니다"
            
            # 모듈 이름 추출
            module_name = module_path.stem
            full_name = f"{store_type.value}.{module_name}"
            
            # 중복 확인
            if full_name in self.modules:
                return False, "이미 등록된 모듈입니다"
            
            # 모듈 로드
            module_class, error = self.loader.load_module(module_path, module_name)
            if error:
                return False, error
            
            # 검증 (선택적)
            if validate:
                validation_result = self.validator.validate_module(
                    module_class, module_path
                )
                if not validation_result.passed:
                    errors = "\n".join(validation_result.errors)
                    return False, f"모듈 검증 실패:\n{errors}"
            
            # 메타데이터 추출
            try:
                instance = module_class()
                metadata = ModuleMetadata.from_module(instance)
                metadata.validation_status = (
                    ValidationStatus.PASSED if validate else ValidationStatus.PENDING
                )
            except Exception as e:
                return False, f"메타데이터 추출 실패: {str(e)}"
            
            # 모듈 정보 생성
            module_info = ModuleInfo(
                name=full_name,
                metadata=metadata,
                store_type=store_type,
                file_path=str(module_path),
                module_class=module_class,
                status=ModuleStatus.LOADED
            )
            
            # 레지스트리에 추가
            self.modules[full_name] = module_info
            
            # 저장소에 추가
            if store_type in self.stores:
                self.stores[store_type].add_module(module_info)
            
            # 의존성 그래프 업데이트
            self._update_dependency_graph(full_name, metadata.dependencies)
            
            # 이벤트 발생
            self._emit_event('module_registered', {
                'module_name': full_name,
                'metadata': metadata
            })
            
            logger.info(f"모듈 등록 성공: {full_name}")
            return True, None
            
        except Exception as e:
            error_msg = f"모듈 등록 중 오류: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def unregister_module(self, module_name: str) -> Tuple[bool, Optional[str]]:
        """모듈 등록 해제"""
        try:
            if module_name not in self.modules:
                return False, "등록되지 않은 모듈입니다"
            
            module_info = self.modules[module_name]
            
            # 의존하는 모듈 확인
            dependents = self._get_dependents(module_name)
            if dependents:
                return False, f"다른 모듈이 의존하고 있습니다: {', '.join(dependents)}"
            
            # 인스턴스 정리
            if module_info.instance:
                try:
                    module_info.instance.cleanup()
                except:
                    pass
            
            # 언로드
            self.loader.unload_module(module_name.split('.')[-1])
            
            # 레지스트리에서 제거
            del self.modules[module_name]
            
            # 저장소에서 제거
            if module_info.store_type in self.stores:
                self.stores[module_info.store_type].remove_module(module_name)
            
            # 의존성 그래프에서 제거
            self._remove_from_dependency_graph(module_name)
            
            # 캐시에서 제거
            self._remove_from_cache(module_name)
            
            # 이벤트 발생
            self._emit_event('module_unregistered', {
                'module_name': module_name
            })
            
            logger.info(f"모듈 등록 해제: {module_name}")
            return True, None
            
        except Exception as e:
            error_msg = f"모듈 등록 해제 중 오류: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def get_module(self, module_name: str) -> Optional[BaseExperimentModule]:
        """모듈 인스턴스 반환"""
        try:
            # 캐시 확인
            if module_name in self.cache:
                # LRU 업데이트
                self.cache.move_to_end(module_name)
                self.modules[module_name].metadata.usage_count += 1
                return self.cache[module_name]
            
            # 모듈 정보 확인
            if module_name not in self.modules:
                logger.warning(f"모듈을 찾을 수 없음: {module_name}")
                return None
            
            module_info = self.modules[module_name]
            
            # 로드되지 않은 경우 로드
            if module_info.module_class is None:
                module_path = Path(module_info.file_path)
                module_class, error = self.loader.load_module(
                    module_path, 
                    module_name.split('.')[-1]
                )
                
                if error:
                    logger.error(f"모듈 로드 실패: {error}")
                    module_info.status = ModuleStatus.FAILED
                    module_info.error_message = error
                    return None
                
                module_info.module_class = module_class
                module_info.status = ModuleStatus.LOADED
                module_info.load_time = datetime.now()
            
            # 인스턴스 생성
            if module_info.instance is None:
                try:
                    module_info.instance = module_info.module_class()
                    module_info.instance.initialize()
                except Exception as e:
                    logger.error(f"모듈 인스턴스 생성 실패: {str(e)}")
                    module_info.status = ModuleStatus.FAILED
                    module_info.error_message = str(e)
                    return None
            
            # 캐시에 추가
            self._add_to_cache(module_name, module_info.instance)
            
            # 사용 횟수 증가
            module_info.metadata.usage_count += 1
            
            # 이벤트 발생
            self._emit_event('module_loaded', {
                'module_name': module_name
            })
            
            return module_info.instance
            
        except Exception as e:
            logger.error(f"모듈 가져오기 실패: {str(e)}")
            return None
    
    def list_modules(self, category: Optional[str] = None,
                    tags: Optional[List[str]] = None,
                    store_type: Optional[StoreType] = None) -> List[Dict[str, Any]]:
        """모듈 목록 조회"""
        result = []
        
        for name, info in self.modules.items():
            # 필터링
            if store_type and info.store_type != store_type:
                continue
            
            if category and info.metadata.category != category:
                continue
            
            if tags and not any(tag in info.metadata.tags for tag in tags):
                continue
            
            # 모듈 정보 구성
            module_dict = {
                'name': name,
                'display_name': info.metadata.name,
                'version': info.metadata.version,
                'author': info.metadata.author,
                'category': info.metadata.category,
                'tags': info.metadata.tags,
                'description': info.metadata.description,
                'status': info.status.value,
                'store_type': info.store_type.value,
                'validation_status': info.metadata.validation_status.value,
                'performance_grade': info.metadata.performance_grade,
                'usage_count': info.metadata.usage_count,
                'rating': info.metadata.rating,
                'created_at': info.metadata.created_at.isoformat()
            }
            
            result.append(module_dict)
        
        # 사용 횟수로 정렬
        result.sort(key=lambda x: x['usage_count'], reverse=True)
        
        return result
    
    def search_modules(self, query: str) -> List[Dict[str, Any]]:
        """모듈 검색"""
        query_lower = query.lower()
        results = []
        
        for name, info in self.modules.items():
            # 검색 대상 필드
            searchable = [
                info.metadata.name.lower(),
                info.metadata.description.lower(),
                ' '.join(info.metadata.tags).lower(),
                ' '.join(info.metadata.keywords).lower()
            ]
            
            # 검색어 매칭
            if any(query_lower in field for field in searchable):
                module_dict = {
                    'name': name,
                    'display_name': info.metadata.name,
                    'description': info.metadata.description,
                    'category': info.metadata.category,
                    'tags': info.metadata.tags,
                    'relevance': self._calculate_relevance(query_lower, searchable)
                }
                results.append(module_dict)
        
        # 관련도순 정렬
        results.sort(key=lambda x: x['relevance'], reverse=True)
        
        return results
    
    def validate_all_modules(self) -> Dict[str, ValidationResult]:
        """모든 모듈 검증"""
        results = {}
        
        # 병렬 검증
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            
            for name, info in self.modules.items():
                if info.module_class:
                    future = executor.submit(
                        self.validator.validate_module,
                        info.module_class,
                        Path(info.file_path)
                    )
                    futures[future] = name
            
            for future in as_completed(futures):
                module_name = futures[future]
                try:
                    result = future.result()
                    results[module_name] = result
                    
                    # 메타데이터 업데이트
                    info = self.modules[module_name]
                    info.metadata.validation_status = (
                        ValidationStatus.PASSED if result.passed 
                        else ValidationStatus.FAILED
                    )
                    info.metadata.last_validated = datetime.now()
                    
                except Exception as e:
                    logger.error(f"모듈 검증 실패 ({module_name}): {str(e)}")
                    results[module_name] = ValidationResult(
                        passed=False,
                        errors=[str(e)]
                    )
        
        return results
    
    def get_module_dependencies(self, module_name: str) -> List[str]:
        """모듈 의존성 조회"""
        if module_name not in self.modules:
            return []
        
        dependencies = []
        module_info = self.modules[module_name]
        
        for dep in module_info.metadata.dependencies:
            if not dep.optional:
                dependencies.append(dep.module_name)
        
        return dependencies
    
    def check_dependencies(self, module_name: str) -> Tuple[bool, List[str]]:
        """의존성 충족 확인"""
        if module_name not in self.modules:
            return False, ["모듈을 찾을 수 없습니다"]
        
        missing = []
        module_info = self.modules[module_name]
        
        for dep in module_info.metadata.dependencies:
            if not dep.optional and not self._is_dependency_satisfied(dep):
                missing.append(f"{dep.module_name} {dep.version_spec}")
        
        return len(missing) == 0, missing
    
    def get_statistics(self) -> Dict[str, Any]:
        """레지스트리 통계"""
        stats = {
            'total_modules': len(self.modules),
            'loaded_modules': sum(1 for info in self.modules.values() 
                                if info.status == ModuleStatus.LOADED),
            'failed_modules': sum(1 for info in self.modules.values() 
                                if info.status == ModuleStatus.FAILED),
            'cache_size': len(self.cache),
            'by_store': {},
            'by_category': {},
            'by_status': {},
            'by_validation': {},
            'performance_grades': {},
            'total_usage': sum(info.metadata.usage_count 
                             for info in self.modules.values())
        }
        
        # 저장소별 통계
        for store_type in StoreType:
            count = sum(1 for info in self.modules.values() 
                       if info.store_type == store_type)
            stats['by_store'][store_type.value] = count
        
        # 카테고리별 통계
        for info in self.modules.values():
            category = info.metadata.category
            stats['by_category'][category] = stats['by_category'].get(category, 0) + 1
        
        # 상태별 통계
        for status in ModuleStatus:
            count = sum(1 for info in self.modules.values() 
                       if info.status == status)
            stats['by_status'][status.value] = count
        
        # 검증 상태별 통계
        for status in ValidationStatus:
            count = sum(1 for info in self.modules.values() 
                       if info.metadata.validation_status == status)
            stats['by_validation'][status.value] = count
        
        # 성능 등급별 통계
        for info in self.modules.values():
            grade = info.metadata.performance_grade
            stats['performance_grades'][grade] = stats['performance_grades'].get(grade, 0) + 1
        
        return stats
    
    # ==================== 캐시 관리 ====================
    
    def _add_to_cache(self, module_name: str, instance: BaseExperimentModule):
        """캐시에 추가"""
        if module_name in self.cache:
            self.cache.move_to_end(module_name)
        else:
            self.cache[module_name] = instance
            
            # 캐시 크기 제한
            if len(self.cache) > self.cache_size:
                oldest = next(iter(self.cache))
                del self.cache[oldest]
    
    def _remove_from_cache(self, module_name: str):
        """캐시에서 제거"""
        if module_name in self.cache:
            del self.cache[module_name]
    
    def clear_cache(self):
        """캐시 전체 삭제"""
        self.cache.clear()
        logger.info("모듈 캐시 삭제됨")
    
    # ==================== 의존성 관리 ====================
    
    def _update_dependency_graph(self, module_name: str, 
                                dependencies: List[ModuleDependency]):
        """의존성 그래프 업데이트"""
        self.dependency_graph[module_name] = set()
        
        for dep in dependencies:
            if not dep.optional:
                self.dependency_graph[module_name].add(dep.module_name)
    
    def _remove_from_dependency_graph(self, module_name: str):
        """의존성 그래프에서 제거"""
        if module_name in self.dependency_graph:
            del self.dependency_graph[module_name]
        
        # 다른 모듈의 의존성에서도 제거
        for deps in self.dependency_graph.values():
            deps.discard(module_name)
    
    def _get_dependents(self, module_name: str) -> Set[str]:
        """특정 모듈에 의존하는 모듈 목록"""
        dependents = set()
        
        for mod, deps in self.dependency_graph.items():
            if module_name in deps:
                dependents.add(mod)
        
        return dependents
    
    def _is_dependency_satisfied(self, dependency: ModuleDependency) -> bool:
        """의존성 충족 여부 확인"""
        # 모듈 존재 확인
        for name, info in self.modules.items():
            if info.metadata.name == dependency.module_name:
                # 버전 확인
                return dependency.is_satisfied_by(info.metadata.version)
        
        return False
    
    def _calculate_relevance(self, query: str, searchable: List[str]) -> float:
        """검색 관련도 계산"""
        relevance = 0.0
        
        for i, field in enumerate(searchable):
            if query in field:
                # 필드별 가중치 (이름 > 태그 > 설명)
                weight = [3.0, 1.0, 2.0, 1.5][i] if i < 4 else 1.0
                relevance += weight
        
        return relevance
    
    # ==================== 이벤트 시스템 ====================
    
    def on(self, event: str, handler: callable):
        """이벤트 핸들러 등록"""
        self.event_handlers[event].append(handler)
    
    def off(self, event: str, handler: callable):
        """이벤트 핸들러 제거"""
        if event in self.event_handlers:
            self.event_handlers[event].remove(handler)
    
    def _emit_event(self, event: str, data: Dict[str, Any]):
        """이벤트 발생"""
        if event in self.event_handlers:
            for handler in self.event_handlers[event]:
                try:
                    handler(data)
                except Exception as e:
                    logger.error(f"이벤트 핸들러 오류 ({event}): {str(e)}")


# ==================== 헬퍼 함수 ====================

def get_registry() -> ModuleRegistry:
    """레지스트리 싱글톤 인스턴스 반환"""
    return ModuleRegistry()


def list_available_modules(category: Optional[str] = None,
                         store_type: Optional[StoreType] = None) -> List[Dict[str, Any]]:
    """사용 가능한 모듈 목록"""
    registry = get_registry()
    return registry.list_modules(category=category, store_type=store_type)


def load_module(module_name: str) -> Optional[BaseExperimentModule]:
    """모듈 로드"""
    registry = get_registry()
    return registry.get_module(module_name)


def register_custom_module(module_path: Union[str, Path],
                         validate: bool = True) -> Tuple[bool, Optional[str]]:
    """커스텀 모듈 등록"""
    registry = get_registry()
    return registry.register_module(module_path, StoreType.USER, validate)
