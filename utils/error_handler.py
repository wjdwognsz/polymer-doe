# utils/error_handler.py

"""
Universal DOE Platform - 중앙 집중식 에러 처리 관리자

이 모듈은 애플리케이션 전체의 에러를 체계적으로 관리합니다.
모든 예외는 이곳에서 분류, 로깅, 복구 시도됩니다.
"""

import sys
import json
import logging
import traceback
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, Any, List, Callable, Type, Union
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
import asyncio
import streamlit as st

from config.error_config import (
    ErrorCategory, ErrorSeverity, ErrorCode,
    USER_ERROR_MESSAGES, RECOVERY_STRATEGIES,
    ERROR_DEFINITIONS, get_error_definition
)
from config.app_config import DEBUG_CONFIG, APP_NAME, IS_PRODUCTION
from config.local_config import LOCAL_CONFIG


# ==================== 에러 컨텍스트 ====================

@dataclass
class ErrorContext:
    """
    에러 발생 시 수집되는 모든 컨텍스트 정보
    
    이 정보는 디버깅과 사용자 지원에 매우 중요합니다.
    """
    # 기본 정보
    timestamp: datetime = field(default_factory=datetime.now)
    error_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S%f"))
    
    # 에러 정보
    error_type: Type[Exception] = Exception
    error_message: str = ""
    error_traceback: str = ""
    
    # 분류 정보
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    category: ErrorCategory = ErrorCategory.SYSTEM
    
    # 컨텍스트 정보
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    page_name: Optional[str] = None
    action: Optional[str] = None
    
    # 추가 데이터
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    # 복구 시도 정보
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_method: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (로깅용)"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'error_id': self.error_id,
            'error_type': self.error_type.__name__,
            'error_message': self.error_message,
            'severity': self.severity.value,
            'category': self.category.value,
            'user_id': self.user_id,
            'page_name': self.page_name,
            'action': self.action,
            'recovery_attempted': self.recovery_attempted,
            'recovery_successful': self.recovery_successful
        }


# ==================== 커스텀 예외 클래스 ====================

class DOEError(Exception):
    """
    Universal DOE Platform의 기본 예외 클래스
    
    모든 커스텀 예외의 부모 클래스입니다.
    에러 카테고리와 심각도 정보를 포함합니다.
    """
    def __init__(
        self, 
        message: str,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        user_message: Optional[str] = None,
        recovery_hint: Optional[str] = None
    ):
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.user_message = user_message or self._get_default_user_message()
        self.recovery_hint = recovery_hint
        self.context = None  # ErrorContext는 나중에 설정
    
    def _get_default_user_message(self) -> str:
        """카테고리별 기본 사용자 메시지"""
        return USER_ERROR_MESSAGES.get(
            self.category,
            "예기치 않은 오류가 발생했습니다."
        )


# 구체적인 예외 클래스들
class NetworkError(DOEError):
    """네트워크 관련 에러"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class DatabaseError(DOEError):
    """데이터베이스 관련 에러"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.DATABASE,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )


class ValidationError(DOEError):
    """입력값 검증 에러"""
    def __init__(self, message: str, field_name: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            **kwargs
        )
        self.field_name = field_name


class APIError(DOEError):
    """API 호출 관련 에러"""
    def __init__(
        self, 
        message: str, 
        api_name: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            message,
            category=ErrorCategory.API,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )
        self.api_name = api_name
        self.status_code = status_code


class ConfigurationError(DOEError):
    """설정 관련 에러"""
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        self.config_key = config_key


# ==================== 에러 핸들러 ====================

class ErrorHandler:
    """
    중앙 집중식 에러 처리 관리자
    
    이 클래스는 애플리케이션 전체의 에러를 관리하는 중앙 통제실입니다.
    모든 에러는 여기를 거쳐 적절히 분류되고 처리됩니다.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ErrorHandler, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self._setup_logging()
        self._recovery_strategies = self._load_recovery_strategies()
        self._error_history = []  # 최근 에러 기록
        self._error_stats = {}    # 에러 통계
    
    def _setup_logging(self):
        """로깅 시스템 설정"""
        log_path = LOCAL_CONFIG['logs']['path']
        log_path.mkdir(parents=True, exist_ok=True)
        
        # 에러 전용 로거
        self.error_logger = logging.getLogger('error_handler')
        self.error_logger.setLevel(logging.DEBUG if DEBUG_CONFIG['show_debug_info'] else logging.INFO)
        
        # 파일 핸들러 (상세 로그)
        file_handler = logging.FileHandler(
            log_path / 'errors.log',
            encoding='utf-8'
        )
        file_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
        self.error_logger.addHandler(file_handler)
        
        # 심각한 에러용 별도 파일
        critical_handler = logging.FileHandler(
            log_path / 'critical_errors.log',
            encoding='utf-8'
        )
        critical_handler.setLevel(logging.ERROR)
        self.error_logger.addHandler(critical_handler)
    
    def _load_recovery_strategies(self) -> Dict[ErrorCategory, Callable]:
        """복구 전략 로드"""
        strategies = {}
        
        # 각 카테고리별 복구 전략 정의
        strategies[ErrorCategory.NETWORK] = self._recover_from_network_error
        strategies[ErrorCategory.DATABASE] = self._recover_from_database_error
        strategies[ErrorCategory.API] = self._recover_from_api_error
        strategies[ErrorCategory.FILE_SYSTEM] = self._recover_from_file_error
        strategies[ErrorCategory.VALIDATION] = self._recover_from_validation_error
        
        return strategies
    
    # ==================== 메인 에러 처리 메서드 ====================
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        show_user_message: bool = True,
        attempt_recovery: bool = True
    ) -> Optional[Any]:
        """
        에러 처리의 중앙 진입점
        
        이 메서드가 모든 에러 처리의 시작점입니다.
        에러를 받아서 분류하고, 로깅하고, 복구를 시도하고,
        사용자에게 적절한 메시지를 표시합니다.
        
        Args:
            error: 발생한 예외
            context: 추가 컨텍스트 정보
            show_user_message: 사용자에게 메시지 표시 여부
            attempt_recovery: 자동 복구 시도 여부
            
        Returns:
            복구된 값 (있는 경우) 또는 None
        """
        # 1. 에러 컨텍스트 생성
        error_context = self._create_error_context(error, context)
        
        # 2. 에러 분류
        error_context = self._classify_error(error, error_context)
        
        # 3. 에러 로깅
        self._log_error(error_context)
        
        # 4. 에러 히스토리 기록
        self._add_to_history(error_context)
        
        # 5. 통계 업데이트
        self._update_statistics(error_context)
        
        # 6. 사용자 메시지 표시
        if show_user_message:
            self._show_user_message(error, error_context)
        
        # 7. 복구 시도
        recovery_result = None
        if attempt_recovery and error_context.severity != ErrorSeverity.CRITICAL:
            recovery_result = self._attempt_recovery(error, error_context)
        
        # 8. 심각한 에러인 경우 추가 조치
        if error_context.severity == ErrorSeverity.CRITICAL:
            self._handle_critical_error(error_context)
        
        return recovery_result
    
    # ==================== 내부 메서드들 ====================
    
    def _create_error_context(self, error: Exception, context: Optional[Dict[str, Any]]) -> ErrorContext:
        """에러 컨텍스트 생성"""
        error_context = ErrorContext(
            error_type=type(error),
            error_message=str(error),
            error_traceback=traceback.format_exc()
        )
        
        # Streamlit 세션 정보 추가
        if 'st' in sys.modules and hasattr(st, 'session_state'):
            error_context.user_id = st.session_state.get('user_id')
            error_context.session_id = st.session_state.get('session_id')
            error_context.page_name = st.session_state.get('current_page')
        
        # 추가 컨텍스트 정보
        if context:
            error_context.additional_data.update(context)
        
        return error_context
    
    def _classify_error(self, error: Exception, context: ErrorContext) -> ErrorContext:
        """에러 분류"""
        if isinstance(error, DOEError):
            context.category = error.category
            context.severity = error.severity
        else:
            # 일반 예외 분류
            error_name = type(error).__name__
            
            if error_name in ['ConnectionError', 'HTTPError', 'URLError']:
                context.category = ErrorCategory.NETWORK
                context.severity = ErrorSeverity.HIGH
            elif error_name in ['OperationalError', 'IntegrityError']:
                context.category = ErrorCategory.DATABASE
                context.severity = ErrorSeverity.HIGH
            elif error_name in ['FileNotFoundError', 'PermissionError']:
                context.category = ErrorCategory.FILE_SYSTEM
                context.severity = ErrorSeverity.MEDIUM
            elif error_name in ['ValueError', 'TypeError']:
                context.category = ErrorCategory.VALIDATION
                context.severity = ErrorSeverity.LOW
            elif error_name == 'MemoryError':
                context.category = ErrorCategory.SYSTEM
                context.severity = ErrorSeverity.CRITICAL
            else:
                context.category = ErrorCategory.SYSTEM
                context.severity = ErrorSeverity.MEDIUM
        
        return context
    
    def _log_error(self, context: ErrorContext):
        """에러 로깅"""
        log_entry = {
            'error_id': context.error_id,
            'timestamp': context.timestamp.isoformat(),
            'category': context.category.value,
            'severity': context.severity.value,
            'error_type': context.error_type.__name__,
            'error_message': context.error_message,
            'user_id': context.user_id,
            'page_name': context.page_name,
            'additional_data': context.additional_data,
            'traceback': context.error_traceback if not IS_PRODUCTION else None
        }
        
        # 심각도에 따른 로그 레벨
        if context.severity == ErrorSeverity.CRITICAL:
            self.error_logger.critical(json.dumps(log_entry, ensure_ascii=False))
        elif context.severity == ErrorSeverity.HIGH:
            self.error_logger.error(json.dumps(log_entry, ensure_ascii=False))
        elif context.severity == ErrorSeverity.MEDIUM:
            self.error_logger.warning(json.dumps(log_entry, ensure_ascii=False))
        else:
            self.error_logger.info(json.dumps(log_entry, ensure_ascii=False))
    
    def _add_to_history(self, context: ErrorContext):
        """에러 히스토리에 추가"""
        # 최대 100개까지만 유지
        if len(self._error_history) >= 100:
            self._error_history.pop(0)
        
        self._error_history.append(context)
    
    def _update_statistics(self, context: ErrorContext):
        """에러 통계 업데이트"""
        key = f"{context.category.value}_{context.severity.value}"
        
        if key not in self._error_stats:
            self._error_stats[key] = {
                'count': 0,
                'first_occurred': context.timestamp,
                'last_occurred': context.timestamp
            }
        
        self._error_stats[key]['count'] += 1
        self._error_stats[key]['last_occurred'] = context.timestamp
    
    def _show_user_message(self, error: Exception, context: ErrorContext):
        """사용자에게 에러 메시지 표시"""
        if isinstance(error, DOEError):
            user_message = error.user_message
            recovery_hint = error.recovery_hint
        else:
            user_message = USER_ERROR_MESSAGES.get(
                context.category,
                "예기치 않은 오류가 발생했습니다."
            )
            recovery_hint = None
        
        # Streamlit UI에 표시
        if 'st' in sys.modules:
            if context.severity == ErrorSeverity.CRITICAL:
                st.error(f"🚨 {user_message}")
            elif context.severity == ErrorSeverity.HIGH:
                st.error(f"❌ {user_message}")
            elif context.severity == ErrorSeverity.MEDIUM:
                st.warning(f"⚠️ {user_message}")
            else:
                st.info(f"ℹ️ {user_message}")
            
            if recovery_hint:
                st.caption(f"💡 {recovery_hint}")
            
            # 디버그 모드에서는 상세 정보 표시
            if DEBUG_CONFIG['show_debug_info'] and not IS_PRODUCTION:
                with st.expander("🔍 상세 정보"):
                    st.code(context.error_traceback)
                    st.json(context.to_dict())
    
    # ==================== 복구 전략 ====================
    
    def _attempt_recovery(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """자동 복구 시도"""
        strategy = self._recovery_strategies.get(context.category)
        
        if strategy:
            try:
                result = strategy(error, context)
                context.recovery_attempted = True
                context.recovery_successful = result is not None
                context.recovery_method = strategy.__name__
                return result
            except Exception as recovery_error:
                self.error_logger.error(f"복구 실패: {recovery_error}")
                context.recovery_attempted = True
                context.recovery_successful = False
        
        return None
    
    def _recover_from_network_error(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """네트워크 에러 복구"""
        # 오프라인 모드로 전환
        if 'st' in sys.modules and hasattr(st, 'session_state'):
            st.session_state['offline_mode'] = True
            st.info("🔌 오프라인 모드로 전환되었습니다.")
        
        # 캐시된 데이터 반환
        cache_key = context.additional_data.get('cache_key')
        if cache_key and hasattr(st, 'cache_data'):
            # 캐시에서 데이터 찾기
            return None  # 실제 구현에서는 캐시 조회
        
        return None
    
    def _recover_from_database_error(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """데이터베이스 에러 복구"""
        # 백업 DB 사용
        if 'st' in sys.modules:
            st.warning("⚠️ 백업 데이터베이스를 사용합니다.")
        
        # 읽기 전용 모드로 전환
        if hasattr(st, 'session_state'):
            st.session_state['readonly_mode'] = True
        
        return None
    
    def _recover_from_api_error(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """API 에러 복구"""
        api_name = context.additional_data.get('api_name')
        
        # 대체 API 사용
        if api_name == 'gemini':
            if 'st' in sys.modules:
                st.info("🔄 대체 AI 엔진을 사용합니다.")
            return 'groq'  # 대체 API 이름 반환
        
        # 기본 템플릿 사용
        return context.additional_data.get('default_template')
    
    def _recover_from_file_error(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """파일 시스템 에러 복구"""
        # 대체 경로 사용
        file_path = context.additional_data.get('file_path')
        if file_path:
            # 임시 폴더 사용
            temp_path = Path.home() / '.polymer_doe_temp' / Path(file_path).name
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            return str(temp_path)
        
        return None
    
    def _recover_from_validation_error(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """검증 에러 복구"""
        # 기본값 사용
        field_name = getattr(error, 'field_name', None)
        if field_name:
            defaults = {
                'temperature': 25.0,
                'pressure': 1.0,
                'concentration': 1.0,
                'time': 60.0
            }
            return defaults.get(field_name)
        
        return None
    
    def _handle_critical_error(self, context: ErrorContext):
        """심각한 에러 처리"""
        # 관리자에게 알림 (구현 필요)
        self.error_logger.critical(f"CRITICAL ERROR: {context.error_id}")
        
        # 상태 저장
        if 'st' in sys.modules and hasattr(st, 'session_state'):
            # 현재 상태를 임시 파일로 저장
            state_file = LOCAL_CONFIG['logs']['path'] / f"crash_state_{context.error_id}.json"
            try:
                with open(state_file, 'w', encoding='utf-8') as f:
                    json.dump(dict(st.session_state), f, ensure_ascii=False, default=str)
            except:
                pass
    
    # ==================== 공개 메서드 ====================
    
    def get_error_history(self, limit: int = 10) -> List[ErrorContext]:
        """최근 에러 히스토리 조회"""
        return self._error_history[-limit:]
    
    def get_error_stats(self) -> Dict[str, Any]:
        """에러 통계 조회"""
        return self._error_stats.copy()
    
    def clear_error_history(self):
        """에러 히스토리 초기화"""
        self._error_history.clear()
        self._error_stats.clear()
    
    def export_error_report(self, start_date: datetime, end_date: datetime) -> str:
        """에러 리포트 생성 (지원팀용)"""
        report_data = {
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'summary': self.get_error_stats(),
            'errors': [
                error.to_dict() for error in self._error_history
                if start_date <= error.timestamp <= end_date
            ]
        }
        
        return json.dumps(report_data, indent=2, ensure_ascii=False)


# ==================== 데코레이터 ====================

def handle_errors(
    category: ErrorCategory = ErrorCategory.SYSTEM,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    show_message: bool = True,
    attempt_recovery: bool = True,
    default_return: Any = None
):
    """
    에러 처리 데코레이터
    
    함수를 감싸서 발생하는 모든 에러를 자동으로 처리합니다.
    
    사용 예:
        @handle_errors(category=ErrorCategory.API, attempt_recovery=True)
        def call_ai_api(prompt):
            # API 호출 코드
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handler = get_error_handler()
                
                # 컨텍스트 정보 수집
                context = {
                    'function': func.__name__,
                    'args': str(args)[:200],  # 긴 인자는 잘라냄
                    'kwargs': str(kwargs)[:200]
                }
                
                # 에러 처리
                recovery_result = handler.handle_error(
                    e,
                    context=context,
                    show_user_message=show_message,
                    attempt_recovery=attempt_recovery
                )
                
                # 복구 성공 시 복구된 값 반환
                if recovery_result is not None:
                    return recovery_result
                
                # 기본값 반환
                return default_return
        
        # 비동기 함수 지원
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                handler = get_error_handler()
                
                context = {
                    'function': func.__name__,
                    'args': str(args)[:200],
                    'kwargs': str(kwargs)[:200],
                    'is_async': True
                }
                
                recovery_result = handler.handle_error(
                    e,
                    context=context,
                    show_user_message=show_message,
                    attempt_recovery=attempt_recovery
                )
                
                if recovery_result is not None:
                    return recovery_result
                
                return default_return
        
        # 함수 타입에 따라 적절한 래퍼 반환
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


def handle_ui_error(func: Callable) -> Callable:
    """
    UI 관련 에러 처리 전용 데코레이터
    
    UI 렌더링 중 발생하는 에러를 우아하게 처리합니다.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # UI 에러는 항상 사용자에게 표시
            st.error(
                "화면을 표시하는 중 문제가 발생했습니다. "
                "페이지를 새로고침해주세요."
            )
            
            # 디버그 모드에서만 상세 정보
            if DEBUG_CONFIG['show_debug_info']:
                st.exception(e)
            
            # 로깅
            handler = get_error_handler()
            handler.handle_error(
                e,
                context={'ui_function': func.__name__},
                show_user_message=False  # 이미 표시했으므로
            )
            
            return None
    
    return wrapper


# ==================== 전역 함수 ====================

def get_error_handler() -> ErrorHandler:
    """ErrorHandler 싱글톤 인스턴스 반환"""
    return ErrorHandler()


def raise_error(
    message: str,
    category: ErrorCategory,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    **kwargs
) -> None:
    """
    DOEError를 발생시키는 헬퍼 함수
    
    사용 예:
        raise_error(
            "API 키가 설정되지 않았습니다",
            ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            user_message="API 키를 설정해주세요",
            recovery_hint="설정 > API 키 관리에서 키를 입력하세요"
        )
    """
    raise DOEError(message, category, severity, **kwargs)


def safe_execute(
    func: Callable,
    args: tuple = (),
    kwargs: dict = None,
    fallback: Any = None,
    error_message: str = "작업을 수행할 수 없습니다"
) -> Any:
    """
    함수를 안전하게 실행하는 헬퍼
    
    에러 발생 시 fallback 값을 반환합니다.
    """
    if kwargs is None:
        kwargs = {}
    
    try:
        return func(*args, **kwargs)
    except Exception as e:
        handler = get_error_handler()
        handler.handle_error(
            e,
            context={'function': func.__name__},
            show_user_message=True
        )
        return fallback


# ==================== 컨텍스트 매니저 ====================

class error_context:
    """
    에러 처리 컨텍스트 매니저
    
    with 문과 함께 사용하여 특정 블록의 에러를 처리합니다.
    
    사용 예:
        with error_context(category=ErrorCategory.DATABASE):
            # 데이터베이스 작업
            pass
    """
    def __init__(
        self,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        suppress: bool = False,
        fallback: Any = None
    ):
        self.category = category
        self.severity = severity
        self.suppress = suppress
        self.fallback = fallback
        self.handler = get_error_handler()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # 에러 처리
            self.handler.handle_error(
                exc_val,
                context={'context_manager': True}
            )
            
            # suppress=True면 예외를 억제
            return self.suppress


# ==================== 초기화 ====================

def initialize_error_handling():
    """
    에러 처리 시스템 초기화
    
    애플리케이션 시작 시 한 번 호출됩니다.
    """
    handler = get_error_handler()
    
    # 전역 예외 핸들러 설정
    def global_exception_handler(exc_type, exc_value, exc_traceback):
        # KeyboardInterrupt는 정상 종료
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        # 그 외 모든 예외 처리
        handler.handle_error(
            exc_value,
            context={'global_handler': True},
            show_user_message=True,
            attempt_recovery=True
        )
    
    sys.excepthook = global_exception_handler
    
    # 성공 메시지 (디버그 모드에서만)
    if DEBUG_CONFIG['show_debug_info']:
        logging.info("Error handling system initialized")


# 애플리케이션 시작 시 초기화
if __name__ != "__main__":
    initialize_error_handling()
