"""
🚨 Universal DOE Platform - 에러 핸들러
================================================================================
error_config.py의 설정을 기반으로 실제 에러 처리를 수행하는 핸들러
자동 복구, 사용자 알림, 로깅 등을 통합 관리
================================================================================
"""

import streamlit as st
import logging
import traceback
import time
from typing import Optional, Dict, Any, List, Callable, Union
from datetime import datetime
from contextlib import contextmanager
from functools import wraps
import json
import sys
from pathlib import Path

# error_config 임포트
try:
    from config.error_config import (
        ErrorDefinition, ErrorCategory, ErrorSeverity, RecoveryStrategy,
        ERROR_CODES, ERROR_SEVERITY_CONFIG, RECOVERY_CONFIG,
        ERROR_MESSAGE_TEMPLATES, ERROR_RECOVERY_STRATEGIES,
        get_error_definition, format_error_message, should_auto_recover,
        get_recovery_actions, log_error, get_user_friendly_message,
        get_error_color, RECOVERY_ACTIONS
    )
except ImportError:
    # 개발 중 임포트 오류 처리
    print("Warning: error_config not found, using fallback")
    ERROR_CODES = {}
    ErrorSeverity = type('ErrorSeverity', (), {'ERROR': 'error'})

# ============================================================================
# 🎯 에러 핸들러 클래스
# ============================================================================

class ErrorHandler:
    """중앙집중식 에러 처리 관리자"""
    
    def __init__(self):
        """에러 핸들러 초기화"""
        self.error_history: List[Dict[str, Any]] = []
        self.recovery_attempts: Dict[str, int] = {}
        self.suppressed_errors: set = set()
        self.custom_handlers: Dict[str, Callable] = {}
        self.setup_logging()
        
    def setup_logging(self):
        """로깅 설정"""
        log_dir = Path("./data/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'errors.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('ErrorHandler')
    
    # ========================================================================
    # 🔧 에러 처리 메인 메서드
    # ========================================================================
    
    def handle_error(
        self,
        error_code: str,
        context: Dict[str, Any] = None,
        exception: Exception = None,
        show_notification: bool = True,
        auto_recover: bool = True
    ) -> bool:
        """
        에러 처리 메인 함수
        
        Args:
            error_code: 에러 코드
            context: 에러 컨텍스트 정보
            exception: 발생한 예외 객체
            show_notification: 사용자 알림 표시 여부
            auto_recover: 자동 복구 시도 여부
            
        Returns:
            bool: 에러 처리 성공 여부
        """
        context = context or {}
        
        # 에러 정의 가져오기
        error_def = get_error_definition(error_code)
        if not error_def:
            self.logger.error(f"Unknown error code: {error_code}")
            return False
        
        # 에러 기록
        self._record_error(error_code, context, exception)
        
        # 로깅
        log_error(error_code, context, exception)
        
        # 사용자 알림
        if show_notification and error_def.notify_user:
            self._show_notification(error_def, context)
        
        # 자동 복구 시도
        if auto_recover and error_def.auto_recoverable:
            return self._attempt_recovery(error_def, context)
        
        # 수동 개입 필요
        if error_def.recovery_strategy == RecoveryStrategy.USER_INTERVENTION:
            self._show_intervention_dialog(error_def, context)
        
        return not error_def.can_continue
    
    # ========================================================================
    # 🔄 자동 복구 메서드
    # ========================================================================
    
    def _attempt_recovery(self, error_def: ErrorDefinition, context: Dict[str, Any]) -> bool:
        """자동 복구 시도"""
        error_code = error_def.code
        
        # 재시도 횟수 확인
        attempt_key = f"{error_code}_{json.dumps(context, sort_keys=True)}"
        current_attempts = self.recovery_attempts.get(attempt_key, 0)
        
        if current_attempts >= error_def.max_retries:
            self.logger.warning(f"Max recovery attempts reached for {error_code}")
            return False
        
        self.recovery_attempts[attempt_key] = current_attempts + 1
        
        # 복구 액션 실행
        recovery_actions = get_recovery_actions(error_code)
        if not recovery_actions:
            recovery_actions = error_def.recovery_actions
        
        for action in recovery_actions:
            try:
                if self._execute_recovery_action(action, context):
                    self.logger.info(f"Recovery successful for {error_code} using {action}")
                    self.recovery_attempts[attempt_key] = 0  # 성공 시 리셋
                    return True
            except Exception as e:
                self.logger.error(f"Recovery action failed: {action} - {str(e)}")
                continue
        
        # 복구 전략별 처리
        return self._apply_recovery_strategy(error_def, context)
    
    def _execute_recovery_action(self, action: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """복구 액션 실행"""
        action_name = action.get('action')
        params = action.get('params')
        
        # 내장 복구 함수들 직접 실행
        if action_name == 'clear_cache':
            return self._clear_cache_action(params, context)
        elif action_name == 'switch_to_offline':
            return self._switch_to_offline_action(params, context)
        elif action_name == 'manual_input_prompt':
            return self._show_manual_input(params, context)
        
        # 외부 모듈 함수 호출
        action_info = RECOVERY_ACTIONS.get(action_name, {})
        function_path = action_info.get('function')
        
        if not function_path:
            self.logger.error(f"Unknown recovery action: {action_name}")
            return False
        
        try:
            # protocol_extractor나 다른 모듈의 함수 호출
            module_path, function_name = function_path.rsplit('.', 1)
            
            # 이미 로드된 모듈 확인 (순환 임포트 방지)
            if module_path == 'utils.protocol_extractor':
                from utils.protocol_extractor import (
                    try_multiple_encodings, detect_encoding, 
                    read_as_binary, enhance_image, try_multiple_ocr
                )
                func_map = {
                    'try_multiple_encodings': try_multiple_encodings,
                    'detect_encoding': detect_encoding,
                    'read_as_binary': read_as_binary,
                    'enhance_image': enhance_image,
                    'try_multiple_ocr': try_multiple_ocr
                }
                func = func_map.get(function_name)
            else:
                # 다른 모듈은 동적 임포트
                module = __import__(module_path, fromlist=[function_name])
                func = getattr(module, function_name)
            
            # 파라미터와 함께 함수 호출
            if params:
                result = func(*params, **context)
            else:
                result = func(**context)
            
            return bool(result)
            
        except Exception as e:
            self.logger.error(f"Failed to execute recovery action {action_name}: {str(e)}")
            return False
    
    def _clear_cache_action(self, params: Optional[List[str]], context: Dict[str, Any]) -> bool:
        """캐시 정리 액션"""
        try:
            cache_types = params or ['temp', 'api_responses']
            if 'cache' in st.session_state:
                for cache_type in cache_types:
                    if cache_type in st.session_state.cache:
                        st.session_state.cache[cache_type].clear()
            return True
        except Exception as e:
            self.logger.error(f"Cache clear failed: {str(e)}")
            return False
    
    def _switch_to_offline_action(self, params: Optional[List], context: Dict[str, Any]) -> bool:
        """오프라인 모드 전환 액션"""
        try:
            st.session_state['offline_mode'] = True
            st.session_state['offline_reason'] = context.get('error_code', 'network_error')
            return True
        except Exception as e:
            self.logger.error(f"Offline switch failed: {str(e)}")
            return False
    
    def _show_manual_input(self, params: Optional[List], context: Dict[str, Any]) -> bool:
        """수동 입력 다이얼로그 표시"""
        try:
            st.session_state['show_manual_input'] = True
            st.session_state['manual_input_context'] = context
            return True
        except Exception as e:
            self.logger.error(f"Manual input dialog failed: {str(e)}")
            return False
    
    def _apply_recovery_strategy(self, error_def: ErrorDefinition, context: Dict[str, Any]) -> bool:
        """복구 전략 적용"""
        strategy = error_def.recovery_strategy
        config = RECOVERY_CONFIG.get(strategy, {})
        
        if strategy == RecoveryStrategy.RETRY:
            return self._retry_operation(error_def, context, config)
        elif strategy == RecoveryStrategy.FALLBACK:
            return self._use_fallback(error_def, context)
        elif strategy == RecoveryStrategy.CACHE:
            return self._use_cached_data(error_def, context)
        elif strategy == RecoveryStrategy.DEFAULT:
            return self._use_default_value(error_def, context)
        elif strategy == RecoveryStrategy.AUTO_FIX:
            return self._auto_fix(error_def, context)
        else:
            return False
    
    def _retry_operation(self, error_def: ErrorDefinition, context: Dict[str, Any], config: Dict) -> bool:
        """작업 재시도"""
        max_attempts = config.get('max_attempts', 3)
        delay = config.get('delay', error_def.retry_delay)
        backoff_factor = config.get('backoff_factor', 2.0)
        
        for attempt in range(max_attempts):
            if attempt > 0:
                wait_time = delay.total_seconds() * (backoff_factor ** (attempt - 1))
                time.sleep(wait_time)
            
            try:
                # 원래 작업 재실행 (context에 저장된 함수)
                original_func = context.get('_original_function')
                if original_func:
                    result = original_func(**context.get('_original_args', {}))
                    return True
            except Exception as e:
                self.logger.warning(f"Retry {attempt + 1}/{max_attempts} failed: {str(e)}")
                continue
        
        return False
    
    def _use_fallback(self, error_def: ErrorDefinition, context: Dict[str, Any]) -> bool:
        """대체 방법 사용"""
        fallback_func = context.get('_fallback_function')
        if fallback_func:
            try:
                fallback_func(**context.get('_fallback_args', {}))
                return True
            except Exception as e:
                self.logger.error(f"Fallback failed: {str(e)}")
        return False
    
    def _use_cached_data(self, error_def: ErrorDefinition, context: Dict[str, Any]) -> bool:
        """캐시된 데이터 사용"""
        cache_key = context.get('_cache_key')
        if cache_key and 'cache' in st.session_state:
            cached_value = st.session_state.cache.get(cache_key)
            if cached_value:
                context['_result'] = cached_value
                return True
        return False
    
    def _use_default_value(self, error_def: ErrorDefinition, context: Dict[str, Any]) -> bool:
        """기본값 사용"""
        default_value = context.get('_default_value')
        if default_value is not None:
            context['_result'] = default_value
            return True
        return False
    
    def _auto_fix(self, error_def: ErrorDefinition, context: Dict[str, Any]) -> bool:
        """자동 수정 시도"""
        # 에러별 특화된 자동 수정 로직
        if error_def.code == '4201':  # 인코딩 오류
            return self._fix_encoding(context)
        elif error_def.code == '4205':  # OCR 오류
            return self._fix_ocr(context)
        return False
    
    def _fix_encoding(self, context: Dict[str, Any]) -> bool:
        """인코딩 자동 수정"""
        file_path = context.get('filename')
        if not file_path:
            return False
        
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'cp949', 'gbk']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                context['_result'] = content
                context['_fixed_encoding'] = encoding
                return True
            except:
                continue
        return False
    
    def _fix_ocr(self, context: Dict[str, Any]) -> bool:
        """OCR 자동 개선"""
        # 이미지 개선 로직
        return False
    
    # ========================================================================
    # 🔔 알림 및 UI 메서드
    # ========================================================================
    
    def _show_notification(self, error_def: ErrorDefinition, context: Dict[str, Any]):
        """사용자 알림 표시"""
        severity_config = ERROR_SEVERITY_CONFIG[error_def.severity]
        message = format_error_message(error_def.code, context)
        
        # Streamlit 알림 표시
        if error_def.severity == ErrorSeverity.CRITICAL:
            st.error(message, icon=severity_config['icon'])
        elif error_def.severity == ErrorSeverity.ERROR:
            st.error(message, icon=severity_config['icon'])
        elif error_def.severity == ErrorSeverity.WARNING:
            st.warning(message, icon=severity_config['icon'])
        elif error_def.severity == ErrorSeverity.INFO:
            st.info(message, icon=severity_config['icon'])
        
        # 도움말 링크 추가
        if error_def.documentation_url:
            st.markdown(f"[🔗 자세한 도움말]({error_def.documentation_url})")
    
    def _show_intervention_dialog(self, error_def: ErrorDefinition, context: Dict[str, Any]):
        """사용자 개입 다이얼로그"""
        with st.expander("🛠️ 문제 해결 도우미", expanded=True):
            st.write(f"**{error_def.name}**")
            st.write(error_def.user_message.format(**context))
            
            st.write("**해결 방법:**")
            for i, suggestion in enumerate(error_def.recovery_suggestions, 1):
                st.write(f"{i}. {suggestion}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("다시 시도", type="primary"):
                    st.session_state['retry_requested'] = True
                    st.rerun()
            with col2:
                if st.button("건너뛰기"):
                    st.session_state['skip_error'] = True
    
    # ========================================================================
    # 📊 에러 기록 및 분석
    # ========================================================================
    
    def _record_error(self, error_code: str, context: Dict[str, Any], exception: Exception = None):
        """에러 기록"""
        error_record = {
            'timestamp': datetime.now().isoformat(),
            'code': error_code,
            'context': context,
            'exception': str(exception) if exception else None,
            'traceback': traceback.format_exc() if exception else None,
            'session_id': st.session_state.get('session_id', 'unknown'),
            'user_id': st.session_state.get('user_id', 'anonymous')
        }
        
        self.error_history.append(error_record)
        
        # 메모리 관리 (최대 1000개 유지)
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """에러 통계 반환"""
        if not self.error_history:
            return {}
        
        stats = {
            'total_errors': len(self.error_history),
            'by_category': {},
            'by_severity': {},
            'most_common': {},
            'recent_errors': self.error_history[-10:]
        }
        
        # 카테고리별 집계
        for record in self.error_history:
            error_def = get_error_definition(record['code'])
            if error_def:
                category = error_def.category.value
                severity = error_def.severity.value
                
                stats['by_category'][category] = stats['by_category'].get(category, 0) + 1
                stats['by_severity'][severity] = stats['by_severity'].get(severity, 0) + 1
                stats['most_common'][record['code']] = stats['most_common'].get(record['code'], 0) + 1
        
        # 가장 빈번한 에러 정렬
        stats['most_common'] = dict(sorted(
            stats['most_common'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5])
        
        return stats
    
    # ========================================================================
    # 🎨 컨텍스트 매니저 및 데코레이터
    # ========================================================================
    
    @contextmanager
    def error_context(self, operation_name: str, **kwargs):
        """에러 처리 컨텍스트 매니저"""
        context = {
            'operation': operation_name,
            'start_time': datetime.now(),
            **kwargs
        }
        
        try:
            yield context
        except Exception as e:
            # 예외 타입에 따라 에러 코드 매핑
            error_code = self._map_exception_to_error_code(e, context)
            self.handle_error(error_code, context, e)
            
            # 에러 정의에 따라 재발생 여부 결정
            error_def = get_error_definition(error_code)
            if error_def and not error_def.can_continue:
                raise
    
    def _map_exception_to_error_code(self, exception: Exception, context: Dict[str, Any]) -> str:
        """예외를 에러 코드로 매핑"""
        exception_type = type(exception).__name__
        exception_msg = str(exception).lower()
        
        # 파일 관련 예외
        if isinstance(exception, FileNotFoundError):
            return '4001'
        elif isinstance(exception, PermissionError):
            return '1003'
        elif isinstance(exception, UnicodeDecodeError):
            return '4201'
        elif isinstance(exception, MemoryError):
            return '1001'
        elif isinstance(exception, ConnectionError):
            return '3001'
        elif isinstance(exception, TimeoutError):
            return '3002'
        
        # 메시지 기반 매핑
        if 'encoding' in exception_msg or 'decode' in exception_msg:
            return '4201'
        elif 'timeout' in exception_msg:
            return '4208'
        elif 'api' in exception_msg and 'key' in exception_msg:
            return '5001'
        elif 'rate limit' in exception_msg:
            return '5002'
        
        # 기본값
        return '1000'  # 일반 시스템 오류
    
    def error_handler(self, error_mappings: Dict[type, str] = None):
        """에러 처리 데코레이터"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                operation_name = func.__name__
                context = {
                    'function': operation_name,
                    'args': args,
                    'kwargs': kwargs,
                    '_original_function': func,
                    '_original_args': kwargs
                }
                
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # 커스텀 매핑 확인
                    if error_mappings and type(e) in error_mappings:
                        error_code = error_mappings[type(e)]
                    else:
                        error_code = self._map_exception_to_error_code(e, context)
                    
                    # 에러 처리
                    handled = self.handle_error(error_code, context, e)
                    
                    # 처리 결과에 따라 행동
                    error_def = get_error_definition(error_code)
                    if error_def and error_def.can_continue and handled:
                        # 기본값이나 복구된 값 반환
                        return context.get('_result', None)
                    else:
                        raise
            
            return wrapper
        return decorator
    
    # ========================================================================
    # 🔧 유틸리티 메서드
    # ========================================================================
    
    def register_custom_handler(self, error_code: str, handler: Callable):
        """커스텀 에러 핸들러 등록"""
        self.custom_handlers[error_code] = handler
    
    def suppress_error(self, error_code: str):
        """특정 에러 코드 억제 (알림 안 함)"""
        self.suppressed_errors.add(error_code)
    
    def clear_error_history(self):
        """에러 기록 초기화"""
        self.error_history.clear()
        self.recovery_attempts.clear()
    
    def export_error_report(self, filepath: str = None) -> str:
        """에러 리포트 내보내기"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'statistics': self.get_error_statistics(),
            'error_history': self.error_history[-100:],  # 최근 100개
            'recovery_attempts': dict(self.recovery_attempts)
        }
        
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            return filepath
        else:
            return json.dumps(report, indent=2, ensure_ascii=False)


# ============================================================================
# 🎯 글로벌 에러 핸들러 인스턴스
# ============================================================================

# 싱글톤 패턴으로 전역 핸들러 생성
_error_handler_instance = None

def get_error_handler() -> ErrorHandler:
    """전역 에러 핸들러 인스턴스 반환"""
    global _error_handler_instance
    if _error_handler_instance is None:
        _error_handler_instance = ErrorHandler()
    return _error_handler_instance


# ============================================================================
# 🛠️ 편의 함수들
# ============================================================================

def handle_error(error_code: str, **kwargs) -> bool:
    """에러 처리 편의 함수"""
    handler = get_error_handler()
    return handler.handle_error(error_code, **kwargs)


def with_error_handling(operation_name: str, **kwargs):
    """에러 처리 컨텍스트 매니저 편의 함수"""
    handler = get_error_handler()
    return handler.error_context(operation_name, **kwargs)


def error_handler_decorator(error_mappings: Dict[type, str] = None):
    """에러 처리 데코레이터 편의 함수"""
    handler = get_error_handler()
    return handler.error_handler(error_mappings)


# ============================================================================
# 🔄 Streamlit 특화 기능
# ============================================================================

def show_error_in_sidebar():
    """사이드바에 최근 에러 표시"""
    handler = get_error_handler()
    if handler.error_history:
        with st.sidebar:
            st.subheader("🚨 최근 오류")
            recent_error = handler.error_history[-1]
            error_def = get_error_definition(recent_error['code'])
            if error_def:
                severity_config = ERROR_SEVERITY_CONFIG[error_def.severity]
                st.write(f"{severity_config['icon']} {error_def.name}")
                st.caption(f"시간: {recent_error['timestamp']}")


def show_error_statistics():
    """에러 통계 대시보드"""
    handler = get_error_handler()
    stats = handler.get_error_statistics()
    
    if stats:
        st.subheader("📊 에러 통계")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("전체 에러", stats['total_errors'])
        with col2:
            st.metric("카테고리", len(stats['by_category']))
        with col3:
            st.metric("심각도", len(stats['by_severity']))
        
        # 차트 표시
        if stats['most_common']:
            st.bar_chart(stats['most_common'])


# ============================================================================
# 📤 Public API
# ============================================================================

__all__ = [
    'ErrorHandler',
    'get_error_handler',
    'handle_error',
    'with_error_handling',
    'error_handler_decorator',
    'show_error_in_sidebar',
    'show_error_statistics'
]
