"""
config/error_config.py
에러 처리 설정 - 에러 분류, 메시지, 복구 전략 정의

이 파일은 Universal DOE Platform의 모든 에러를 체계적으로 관리합니다.
병원의 응급 매뉴얼처럼, 각 오류 상황에 대한 명확한 진단과 처치 방법을 정의합니다.
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import traceback
import sys


# ============================================================================
# 에러 분류 체계
# ============================================================================

class ErrorCategory(Enum):
    """에러 카테고리 - 처리 방식을 결정"""
    SYSTEM = "system"              # 시스템 레벨 오류
    USER_INPUT = "user_input"      # 사용자 입력 오류
    DATA = "data"                  # 데이터 관련 오류
    NETWORK = "network"            # 네트워크 오류
    API = "api"                    # API 호출 오류
    AUTH = "auth"                  # 인증/권한 오류
    RESOURCE = "resource"          # 리소스 부족
    CALCULATION = "calculation"    # 계산/분석 오류
    MODULE = "module"              # 모듈 관련 오류
    UNKNOWN = "unknown"            # 알 수 없는 오류


class ErrorSeverity(Enum):
    """에러 심각도 - 의료 시스템의 중증도 분류를 참고"""
    DEBUG = "debug"        # 디버그 정보
    INFO = "info"          # 정보성 메시지
    WARNING = "warning"    # 경고 (계속 진행 가능)
    ERROR = "error"        # 오류 (일부 기능 제한)
    CRITICAL = "critical"  # 치명적 (프로그램 중단 필요)


class RecoveryStrategy(Enum):
    """복구 전략"""
    RETRY = "retry"              # 재시도
    FALLBACK = "fallback"        # 대체 방법 사용
    CACHE = "cache"              # 캐시된 데이터 사용
    DEFAULT = "default"          # 기본값 사용
    USER_INTERVENTION = "user"   # 사용자 개입 필요
    ABORT = "abort"              # 작업 중단
    IGNORE = "ignore"            # 무시하고 계속


# ============================================================================
# 에러 정의 구조
# ============================================================================

@dataclass
class ErrorDefinition:
    """에러 정의 - 각 에러의 완전한 정보"""
    # 기본 정보
    code: str                      # 에러 코드 (예: E001)
    name: str                      # 에러 이름
    category: ErrorCategory        # 카테고리
    severity: ErrorSeverity        # 심각도
    
    # 메시지
    user_message: str              # 사용자용 메시지 (한국어, 친화적)
    technical_message: str         # 개발자용 메시지 (영어, 기술적)
    
    # 복구 정보
    recovery_strategy: RecoveryStrategy     # 복구 전략
    recovery_suggestions: List[str]         # 복구 제안사항
    
    # 추가 정보
    documentation_url: Optional[str] = None # 문서 링크
    can_continue: bool = True              # 계속 진행 가능 여부
    auto_recoverable: bool = False         # 자동 복구 가능 여부
    max_retries: int = 3                   # 최대 재시도 횟수
    
    # 컨텍스트 정보
    preserve_context: bool = True          # 작업 컨텍스트 보존
    log_full_trace: bool = True            # 전체 스택 트레이스 로깅
    notify_user: bool = True               # 사용자 알림 여부
    
    # 커스텀 핸들러
    custom_handler: Optional[Callable] = None


# ============================================================================
# 시스템 에러 정의
# ============================================================================

SYSTEM_ERRORS = {
    'E001': ErrorDefinition(
        code='E001',
        name='메모리 부족',
        category=ErrorCategory.SYSTEM,
        severity=ErrorSeverity.CRITICAL,
        user_message="시스템 메모리가 부족합니다. 다른 프로그램을 종료하고 다시 시도해주세요.",
        technical_message="Available memory below threshold: {available_mb}MB < {required_mb}MB",
        recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
        recovery_suggestions=[
            "실행 중인 다른 프로그램을 종료하세요",
            "대용량 데이터를 작은 단위로 나누어 처리하세요",
            "시스템 메모리를 증설하는 것을 고려하세요"
        ],
        can_continue=False,
        documentation_url="https://docs.universaldoe.com/errors/E001"
    ),
    
    'E002': ErrorDefinition(
        code='E002',
        name='디스크 공간 부족',
        category=ErrorCategory.SYSTEM,
        severity=ErrorSeverity.ERROR,
        user_message="디스크 공간이 부족합니다. 최소 {required_mb}MB의 여유 공간이 필요합니다.",
        technical_message="Insufficient disk space: {available_mb}MB available, {required_mb}MB required",
        recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
        recovery_suggestions=[
            "불필요한 파일을 삭제하세요",
            "임시 파일을 정리하세요 (설정 > 유지보수)",
            "다른 드라이브를 사용하세요"
        ],
        can_continue=True,
        auto_recoverable=False
    ),
    
    'E003': ErrorDefinition(
        code='E003',
        name='파일 접근 권한 없음',
        category=ErrorCategory.SYSTEM,
        severity=ErrorSeverity.ERROR,
        user_message="파일 '{filename}'에 접근할 수 없습니다. 파일 권한을 확인해주세요.",
        technical_message="Permission denied: {filepath} (mode: {mode})",
        recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
        recovery_suggestions=[
            "파일이 다른 프로그램에서 사용 중인지 확인하세요",
            "파일 권한을 확인하세요 (읽기/쓰기)",
            "관리자 권한으로 실행해보세요"
        ],
        can_continue=True
    ),
    
    'E004': ErrorDefinition(
        code='E004',
        name='시스템 리소스 부족',
        category=ErrorCategory.SYSTEM,
        severity=ErrorSeverity.ERROR,
        user_message="시스템 리소스가 부족합니다. 잠시 후 다시 시도해주세요.",
        technical_message="System resource exhausted: {resource_type}",
        recovery_strategy=RecoveryStrategy.RETRY,
        recovery_suggestions=[
            "잠시 기다린 후 다시 시도하세요",
            "불필요한 프로세스를 종료하세요",
            "시스템을 재시작하는 것을 고려하세요"
        ],
        can_continue=False,
        auto_recoverable=True,
        max_retries=3
    )
}


# ============================================================================
# 사용자 입력 에러 정의
# ============================================================================

USER_INPUT_ERRORS = {
    'E101': ErrorDefinition(
        code='E101',
        name='잘못된 숫자 형식',
        category=ErrorCategory.USER_INPUT,
        severity=ErrorSeverity.WARNING,
        user_message="입력하신 '{value}'는 올바른 숫자가 아닙니다. 숫자만 입력해주세요.",
        technical_message="Invalid number format: '{value}' cannot be parsed as {expected_type}",
        recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
        recovery_suggestions=[
            "숫자만 입력하세요 (예: 123, 45.6)",
            "소수점은 점(.)을 사용하세요",
            "천 단위 구분 기호(,)는 제거하세요"
        ],
        can_continue=True,
        preserve_context=True
    ),
    
    'E102': ErrorDefinition(
        code='E102',
        name='범위 초과',
        category=ErrorCategory.USER_INPUT,
        severity=ErrorSeverity.WARNING,
        user_message="입력값 {value}이(가) 허용 범위 [{min_value}, {max_value}]를 벗어났습니다.",
        technical_message="Value out of range: {value} not in [{min_value}, {max_value}]",
        recovery_strategy=RecoveryStrategy.DEFAULT,
        recovery_suggestions=[
            "값을 {min_value}에서 {max_value} 사이로 입력하세요",
            "단위를 확인하세요 (예: mg, mL, °C)"
        ],
        can_continue=True,
        auto_recoverable=True
    ),
    
    'E103': ErrorDefinition(
        code='E103',
        name='필수 입력 누락',
        category=ErrorCategory.USER_INPUT,
        severity=ErrorSeverity.ERROR,
        user_message="필수 항목 '{field_name}'이(가) 입력되지 않았습니다.",
        technical_message="Required field missing: {field_name} in {context}",
        recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
        recovery_suggestions=[
            "빨간색으로 표시된 필수 항목을 모두 입력하세요",
            "기본값을 사용하려면 '기본값 채우기' 버튼을 클릭하세요"
        ],
        can_continue=False,
        preserve_context=True
    ),
    
    'E104': ErrorDefinition(
        code='E104',
        name='잘못된 형식',
        category=ErrorCategory.USER_INPUT,
        severity=ErrorSeverity.WARNING,
        user_message="입력 형식이 올바르지 않습니다. 예시: {example}",
        technical_message="Invalid format: {value} does not match pattern {pattern}",
        recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
        recovery_suggestions=[
            "제시된 예시를 참고하여 입력하세요",
            "특수문자나 공백에 주의하세요"
        ],
        can_continue=True,
        preserve_context=True
    )
}


# ============================================================================
# 데이터 에러 정의
# ============================================================================

DATA_ERRORS = {
    'E201': ErrorDefinition(
        code='E201',
        name='데이터 형식 오류',
        category=ErrorCategory.DATA,
        severity=ErrorSeverity.ERROR,
        user_message="파일 '{filename}'의 데이터 형식을 인식할 수 없습니다.",
        technical_message="Unsupported data format: {format} in file {filepath}",
        recovery_strategy=RecoveryStrategy.FALLBACK,
        recovery_suggestions=[
            "지원되는 형식: CSV, Excel (xlsx/xls), JSON, Parquet",
            "파일이 손상되지 않았는지 확인하세요",
            "다른 형식으로 변환 후 다시 시도하세요"
        ],
        can_continue=True
    ),
    
    'E202': ErrorDefinition(
        code='E202',
        name='데이터 무결성 오류',
        category=ErrorCategory.DATA,
        severity=ErrorSeverity.ERROR,
        user_message="데이터에 일관성 문제가 발견되었습니다: {issue}",
        technical_message="Data integrity violation: {constraint} failed on {data_element}",
        recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
        recovery_suggestions=[
            "데이터를 검토하고 수정하세요",
            "데이터 검증 도구를 사용하세요 (도구 > 데이터 검증)",
            "원본 데이터를 다시 확인하세요"
        ],
        can_continue=False,
        log_full_trace=True
    ),
    
    'E203': ErrorDefinition(
        code='E203',
        name='데이터 읽기 실패',
        category=ErrorCategory.DATA,
        severity=ErrorSeverity.ERROR,
        user_message="파일 '{filename}'을(를) 읽을 수 없습니다.",
        technical_message="Failed to read file: {filepath} - {error_detail}",
        recovery_strategy=RecoveryStrategy.RETRY,
        recovery_suggestions=[
            "파일이 열려있지 않은지 확인하세요",
            "파일 경로가 올바른지 확인하세요",
            "파일 크기가 너무 크지 않은지 확인하세요"
        ],
        can_continue=True,
        auto_recoverable=True,
        max_retries=2
    )
}


# ============================================================================
# 네트워크 에러 정의
# ============================================================================

NETWORK_ERRORS = {
    'E301': ErrorDefinition(
        code='E301',
        name='인터넷 연결 없음',
        category=ErrorCategory.NETWORK,
        severity=ErrorSeverity.WARNING,
        user_message="인터넷 연결을 확인할 수 없습니다. 오프라인 모드로 전환합니다.",
        technical_message="Network unreachable: {endpoint} (timeout: {timeout}s)",
        recovery_strategy=RecoveryStrategy.FALLBACK,
        recovery_suggestions=[
            "인터넷 연결을 확인하세요",
            "방화벽 설정을 확인하세요",
            "오프라인 모드에서는 일부 기능이 제한됩니다"
        ],
        can_continue=True,
        auto_recoverable=True
    ),
    
    'E302': ErrorDefinition(
        code='E302',
        name='서버 응답 없음',
        category=ErrorCategory.NETWORK,
        severity=ErrorSeverity.ERROR,
        user_message="서버가 응답하지 않습니다. 잠시 후 다시 시도해주세요.",
        technical_message="Server timeout: {url} after {timeout}s (attempt {attempt}/{max_attempts})",
        recovery_strategy=RecoveryStrategy.RETRY,
        recovery_suggestions=[
            "잠시 후 다시 시도하세요",
            "서비스 상태를 확인하세요: https://status.universaldoe.com",
            "VPN을 사용 중이라면 연결을 확인하세요"
        ],
        can_continue=True,
        auto_recoverable=True,
        max_retries=3
    ),
    
    'E303': ErrorDefinition(
        code='E303',
        name='연결 거부',
        category=ErrorCategory.NETWORK,
        severity=ErrorSeverity.ERROR,
        user_message="서버 연결이 거부되었습니다.",
        technical_message="Connection refused: {host}:{port}",
        recovery_strategy=RecoveryStrategy.FALLBACK,
        recovery_suggestions=[
            "프록시 설정을 확인하세요",
            "방화벽이 연결을 차단하지 않는지 확인하세요",
            "오프라인 모드로 전환하여 계속 작업하세요"
        ],
        can_continue=True
    )
}


# ============================================================================
# API 에러 정의
# ============================================================================

API_ERRORS = {
    'E401': ErrorDefinition(
        code='E401',
        name='API 키 유효하지 않음',
        category=ErrorCategory.API,
        severity=ErrorSeverity.ERROR,
        user_message="{service} API 키가 유효하지 않습니다. 설정에서 확인해주세요.",
        technical_message="Invalid API key for {service}: {response_code} - {response_message}",
        recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
        recovery_suggestions=[
            "설정 > API 키에서 {service} 키를 확인하세요",
            "API 키가 만료되지 않았는지 확인하세요",
            "올바른 API 키를 복사했는지 확인하세요"
        ],
        can_continue=False,
        documentation_url="https://docs.universaldoe.com/api-keys"
    ),
    
    'E402': ErrorDefinition(
        code='E402',
        name='API 할당량 초과',
        category=ErrorCategory.API,
        severity=ErrorSeverity.WARNING,
        user_message="{service} API 사용량이 한도를 초과했습니다. {reset_time}에 초기화됩니다.",
        technical_message="API rate limit exceeded: {current}/{limit} requests, resets at {reset_time}",
        recovery_strategy=RecoveryStrategy.FALLBACK,
        recovery_suggestions=[
            "다른 AI 엔진을 사용하세요",
            "잠시 기다린 후 다시 시도하세요",
            "API 플랜을 업그레이드하는 것을 고려하세요"
        ],
        can_continue=True,
        auto_recoverable=True
    ),
    
    'E403': ErrorDefinition(
        code='E403',
        name='API 응답 오류',
        category=ErrorCategory.API,
        severity=ErrorSeverity.ERROR,
        user_message="{service}에서 예상치 못한 응답을 받았습니다.",
        technical_message="Unexpected API response from {service}: {status_code} - {error_detail}",
        recovery_strategy=RecoveryStrategy.RETRY,
        recovery_suggestions=[
            "다시 시도하세요",
            "입력 데이터를 확인하세요",
            "다른 AI 엔진을 사용해보세요"
        ],
        can_continue=True,
        max_retries=2
    ),
    
    'E404': ErrorDefinition(
        code='E404',
        name='API 서비스 불가',
        category=ErrorCategory.API,
        severity=ErrorSeverity.ERROR,
        user_message="{service} 서비스를 사용할 수 없습니다.",
        technical_message="API service unavailable: {service} - {reason}",
        recovery_strategy=RecoveryStrategy.FALLBACK,
        recovery_suggestions=[
            "다른 AI 엔진을 선택하세요",
            "서비스 상태를 확인하세요",
            "오프라인 기능을 사용하세요"
        ],
        can_continue=True,
        auto_recoverable=True
    )
}


# ============================================================================
# 인증/권한 에러 정의
# ============================================================================

AUTH_ERRORS = {
    'E501': ErrorDefinition(
        code='E501',
        name='인증 필요',
        category=ErrorCategory.AUTH,
        severity=ErrorSeverity.ERROR,
        user_message="이 기능을 사용하려면 로그인이 필요합니다.",
        technical_message="Authentication required for {resource}",
        recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
        recovery_suggestions=[
            "로그인 페이지로 이동하여 로그인하세요",
            "계정이 없다면 회원가입을 진행하세요"
        ],
        can_continue=False
    ),
    
    'E502': ErrorDefinition(
        code='E502',
        name='권한 부족',
        category=ErrorCategory.AUTH,
        severity=ErrorSeverity.ERROR,
        user_message="이 작업을 수행할 권한이 없습니다.",
        technical_message="Insufficient permissions: {required_permission} required for {action}",
        recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
        recovery_suggestions=[
            "프로젝트 소유자에게 권한을 요청하세요",
            "올바른 계정으로 로그인했는지 확인하세요"
        ],
        can_continue=False
    ),
    
    'E503': ErrorDefinition(
        code='E503',
        name='세션 만료',
        category=ErrorCategory.AUTH,
        severity=ErrorSeverity.WARNING,
        user_message="세션이 만료되었습니다. 다시 로그인해주세요.",
        technical_message="Session expired after {duration} minutes",
        recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
        recovery_suggestions=[
            "다시 로그인하세요",
            "작업 내용은 자동으로 저장되었습니다"
        ],
        can_continue=False,
        preserve_context=True
    )
}


# ============================================================================
# 계산/분석 에러 정의
# ============================================================================

CALCULATION_ERRORS = {
    'E601': ErrorDefinition(
        code='E601',
        name='수치 계산 오류',
        category=ErrorCategory.CALCULATION,
        severity=ErrorSeverity.ERROR,
        user_message="계산 중 오류가 발생했습니다: {error_type}",
        technical_message="Numerical error in {function}: {error_detail}",
        recovery_strategy=RecoveryStrategy.FALLBACK,
        recovery_suggestions=[
            "입력 데이터의 범위를 확인하세요",
            "극단적인 값(0, 무한대 등)이 없는지 확인하세요",
            "다른 계산 방법을 시도하세요"
        ],
        can_continue=True,
        preserve_context=True
    ),
    
    'E602': ErrorDefinition(
        code='E602',
        name='통계 분석 실패',
        category=ErrorCategory.CALCULATION,
        severity=ErrorSeverity.ERROR,
        user_message="통계 분석을 수행할 수 없습니다: {reason}",
        technical_message="Statistical analysis failed: {method} - {error_detail}",
        recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
        recovery_suggestions=[
            "데이터가 분석 요구사항을 충족하는지 확인하세요",
            "최소 {min_samples}개 이상의 샘플이 필요합니다",
            "결측값을 처리한 후 다시 시도하세요"
        ],
        can_continue=False,
        log_full_trace=True
    ),
    
    'E603': ErrorDefinition(
        code='E603',
        name='최적화 실패',
        category=ErrorCategory.CALCULATION,
        severity=ErrorSeverity.WARNING,
        user_message="최적화 알고리즘이 수렴하지 않았습니다.",
        technical_message="Optimization failed to converge: {algorithm} after {iterations} iterations",
        recovery_strategy=RecoveryStrategy.FALLBACK,
        recovery_suggestions=[
            "초기값을 변경해보세요",
            "제약조건을 완화해보세요",
            "다른 최적화 방법을 시도하세요"
        ],
        can_continue=True,
        auto_recoverable=True
    )
}


# ============================================================================
# 모듈 에러 정의
# ============================================================================

MODULE_ERRORS = {
    'E701': ErrorDefinition(
        code='E701',
        name='모듈 로드 실패',
        category=ErrorCategory.MODULE,
        severity=ErrorSeverity.ERROR,
        user_message="모듈 '{module_name}'을(를) 불러올 수 없습니다.",
        technical_message="Failed to load module: {module_path} - {error_detail}",
        recovery_strategy=RecoveryStrategy.FALLBACK,
        recovery_suggestions=[
            "모듈이 올바르게 설치되었는지 확인하세요",
            "모듈 파일이 손상되지 않았는지 확인하세요",
            "기본 모듈을 사용하여 계속 진행하세요"
        ],
        can_continue=True
    ),
    
    'E702': ErrorDefinition(
        code='E702',
        name='모듈 호환성 오류',
        category=ErrorCategory.MODULE,
        severity=ErrorSeverity.ERROR,
        user_message="모듈 '{module_name}'이(가) 현재 버전과 호환되지 않습니다.",
        technical_message="Module compatibility error: {module_name} requires version {required_version}, current: {current_version}",
        recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
        recovery_suggestions=[
            "모듈을 최신 버전으로 업데이트하세요",
            "호환되는 다른 모듈을 사용하세요",
            "개발자에게 문의하세요"
        ],
        can_continue=True
    )
}


# ============================================================================
# 에러 처리 설정
# ============================================================================

@dataclass
class ErrorHandlingConfig:
    """에러 처리 전역 설정"""
    # 표시 설정
    show_error_codes: bool = True          # 에러 코드 표시
    show_technical_details: bool = False   # 기술적 세부사항 표시 (개발 모드)
    show_stack_trace: bool = False         # 스택 트레이스 표시
    
    # 로깅 설정
    log_all_errors: bool = True            # 모든 에러 로깅
    log_warnings: bool = True              # 경고 로깅
    log_stack_traces: bool = True          # 스택 트레이스 로깅
    
    # 알림 설정
    notify_critical_errors: bool = True    # 치명적 에러 알림
    email_on_critical: bool = False        # 이메일 알림
    
    # 복구 설정
    auto_recovery_enabled: bool = True     # 자동 복구 활성화
    max_recovery_attempts: int = 3         # 최대 복구 시도
    recovery_delay_seconds: float = 1.0    # 재시도 간격
    
    # UI 설정
    error_display_duration: int = 10       # 에러 메시지 표시 시간 (초)
    group_similar_errors: bool = True      # 유사 에러 그룹화
    max_errors_displayed: int = 5          # 최대 표시 에러 수
    
    # 개발자 설정
    debug_mode: bool = False               # 디버그 모드
    break_on_error: bool = False           # 에러 시 중단 (디버깅용)
    collect_error_context: bool = True     # 에러 컨텍스트 수집


# ============================================================================
# 에러 메시지 포맷터
# ============================================================================

class ErrorMessageFormatter:
    """에러 메시지 포맷팅"""
    
    @staticmethod
    def format_user_message(error_def: ErrorDefinition, context: Dict[str, Any]) -> str:
        """사용자용 메시지 포맷"""
        try:
            message = error_def.user_message.format(**context)
        except:
            message = error_def.user_message
        
        # 이모지 추가
        emoji_map = {
            ErrorSeverity.INFO: "ℹ️",
            ErrorSeverity.WARNING: "⚠️",
            ErrorSeverity.ERROR: "❌",
            ErrorSeverity.CRITICAL: "🚨",
            ErrorSeverity.DEBUG: "🐛"
        }
        
        emoji = emoji_map.get(error_def.severity, "❓")
        
        # 에러 코드 추가 (설정에 따라)
        config = get_error_config()
        
        if config.show_error_codes:
            return f"{emoji} [{error_def.code}] {message}"
        else:
            return f"{emoji} {message}"
    
    @staticmethod
    def format_technical_message(error_def: ErrorDefinition, 
                                context: Dict[str, Any],
                                exception: Optional[Exception] = None) -> str:
        """기술적 메시지 포맷"""
        parts = []
        
        # 기본 정보
        parts.append(f"Error Code: {error_def.code}")
        parts.append(f"Category: {error_def.category.value}")
        parts.append(f"Severity: {error_def.severity.value}")
        
        # 기술적 메시지
        try:
            tech_msg = error_def.technical_message.format(**context)
        except:
            tech_msg = error_def.technical_message
        parts.append(f"Details: {tech_msg}")
        
        # 예외 정보
        if exception:
            parts.append(f"Exception Type: {type(exception).__name__}")
            parts.append(f"Exception Message: {str(exception)}")
        
        # 컨텍스트
        if context:
            parts.append(f"Context: {context}")
        
        return "\n".join(parts)
    
    @staticmethod
    def format_recovery_suggestions(suggestions: List[str]) -> str:
        """복구 제안사항 포맷"""
        if not suggestions:
            return ""
        
        formatted = ["💡 해결 방법:"]
        for i, suggestion in enumerate(suggestions, 1):
            formatted.append(f"  {i}. {suggestion}")
        
        return "\n".join(formatted)


# ============================================================================
# 에러 통계 수집
# ============================================================================

@dataclass
class ErrorStatistics:
    """에러 통계"""
    error_counts: Dict[str, int] = field(default_factory=dict)
    last_errors: Dict[str, float] = field(default_factory=dict)  # 타임스탬프
    recovery_success: Dict[str, int] = field(default_factory=dict)
    recovery_failures: Dict[str, int] = field(default_factory=dict)
    
    def record_error(self, error_code: str):
        """에러 기록"""
        import time
        self.error_counts[error_code] = self.error_counts.get(error_code, 0) + 1
        self.last_errors[error_code] = time.time()
    
    def record_recovery(self, error_code: str, success: bool):
        """복구 시도 기록"""
        if success:
            self.recovery_success[error_code] = self.recovery_success.get(error_code, 0) + 1
        else:
            self.recovery_failures[error_code] = self.recovery_failures.get(error_code, 0) + 1
    
    def get_error_frequency(self, error_code: str, window_seconds: int = 3600) -> int:
        """특정 시간 내 에러 빈도"""
        import time
        current_time = time.time()
        last_time = self.last_errors.get(error_code, 0)
        
        if current_time - last_time <= window_seconds:
            return self.error_counts.get(error_code, 0)
        return 0
    
    def get_recovery_rate(self, error_code: str) -> float:
        """복구 성공률"""
        success = self.recovery_success.get(error_code, 0)
        failure = self.recovery_failures.get(error_code, 0)
        total = success + failure
        
        return (success / total * 100) if total > 0 else 0.0


# ============================================================================
# 에러 컨텍스트 관리
# ============================================================================

class ErrorContext:
    """에러 발생 컨텍스트"""
    
    def __init__(self):
        self.user_actions: List[str] = []      # 사용자 액션 히스토리
        self.system_state: Dict[str, Any] = {} # 시스템 상태
        self.data_snapshot: Optional[Any] = None # 데이터 스냅샷
        self.timestamp: float = 0
        self.session_id: Optional[str] = None
        
    def add_user_action(self, action: str):
        """사용자 액션 추가"""
        import time
        self.user_actions.append(f"{time.strftime('%H:%M:%S')} - {action}")
        
        # 최대 20개 유지
        if len(self.user_actions) > 20:
            self.user_actions.pop(0)
    
    def capture_system_state(self):
        """시스템 상태 캡처"""
        import psutil
        import time
        
        self.timestamp = time.time()
        try:
            self.system_state = {
                'memory_percent': psutil.virtual_memory().percent,
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'disk_usage': psutil.disk_usage('/').percent,
                'python_version': sys.version,
                'platform': sys.platform
            }
        except:
            # psutil이 없는 경우
            self.system_state = {
                'python_version': sys.version,
                'platform': sys.platform
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'user_actions': self.user_actions[-5:],  # 최근 5개만
            'system_state': self.system_state,
            'timestamp': self.timestamp,
            'session_id': self.session_id
        }


# ============================================================================
# 전역 에러 레지스트리
# ============================================================================

class ErrorRegistry:
    """모든 에러 정의 관리"""
    
    def __init__(self):
        self.errors: Dict[str, ErrorDefinition] = {}
        self._load_all_errors()
        
    def _load_all_errors(self):
        """모든 에러 정의 로드"""
        # 각 카테고리별 에러 추가
        for error_dict in [SYSTEM_ERRORS, USER_INPUT_ERRORS, DATA_ERRORS, 
                          NETWORK_ERRORS, API_ERRORS, AUTH_ERRORS,
                          CALCULATION_ERRORS, MODULE_ERRORS]:
            self.errors.update(error_dict)
    
    def get_error(self, code: str) -> Optional[ErrorDefinition]:
        """에러 정의 조회"""
        return self.errors.get(code)
    
    def get_errors_by_category(self, category: ErrorCategory) -> List[ErrorDefinition]:
        """카테고리별 에러 목록"""
        return [
            error for error in self.errors.values()
            if error.category == category
        ]
    
    def search_errors(self, keyword: str) -> List[ErrorDefinition]:
        """에러 검색"""
        keyword = keyword.lower()
        results = []
        
        for error in self.errors.values():
            if (keyword in error.name.lower() or 
                keyword in error.user_message.lower() or
                keyword in error.code.lower()):
                results.append(error)
                
        return results


# ============================================================================
# 에러 복구 관리자
# ============================================================================

class ErrorRecoveryManager:
    """에러 복구 전략 실행"""
    
    def __init__(self, config: ErrorHandlingConfig):
        self.config = config
        self.recovery_handlers = {
            RecoveryStrategy.RETRY: self._retry_recovery,
            RecoveryStrategy.FALLBACK: self._fallback_recovery,
            RecoveryStrategy.CACHE: self._cache_recovery,
            RecoveryStrategy.DEFAULT: self._default_recovery,
            RecoveryStrategy.USER_INTERVENTION: self._user_intervention,
            RecoveryStrategy.ABORT: self._abort_recovery,
            RecoveryStrategy.IGNORE: self._ignore_recovery
        }
    
    async def attempt_recovery(self, error_def: ErrorDefinition, 
                             context: Dict[str, Any],
                             operation: Callable) -> tuple[bool, Any]:
        """복구 시도"""
        if not self.config.auto_recovery_enabled:
            return False, None
            
        if not error_def.auto_recoverable:
            return False, None
            
        handler = self.recovery_handlers.get(error_def.recovery_strategy)
        if handler:
            return await handler(error_def, context, operation)
            
        return False, None
    
    async def _retry_recovery(self, error_def: ErrorDefinition,
                            context: Dict[str, Any],
                            operation: Callable) -> tuple[bool, Any]:
        """재시도 복구"""
        import asyncio
        
        for attempt in range(error_def.max_retries):
            try:
                await asyncio.sleep(self.config.recovery_delay_seconds * (attempt + 1))
                result = await operation()
                return True, result
            except Exception as e:
                if attempt == error_def.max_retries - 1:
                    return False, None
                continue
                
        return False, None
    
    async def _fallback_recovery(self, error_def: ErrorDefinition,
                               context: Dict[str, Any],
                               operation: Callable) -> tuple[bool, Any]:
        """대체 방법 사용"""
        # 컨텍스트에 fallback_operation이 있으면 실행
        fallback_op = context.get('fallback_operation')
        if fallback_op:
            try:
                result = await fallback_op()
                return True, result
            except:
                pass
        
        return False, None
    
    async def _cache_recovery(self, error_def: ErrorDefinition,
                            context: Dict[str, Any],
                            operation: Callable) -> tuple[bool, Any]:
        """캐시 사용"""
        cache_key = context.get('cache_key')
        cache_manager = context.get('cache_manager')
        
        if cache_key and cache_manager:
            cached_value = cache_manager.get(cache_key)
            if cached_value is not None:
                return True, cached_value
                
        return False, None
    
    async def _default_recovery(self, error_def: ErrorDefinition,
                              context: Dict[str, Any],
                              operation: Callable) -> tuple[bool, Any]:
        """기본값 사용"""
        default_value = context.get('default_value')
        if default_value is not None:
            return True, default_value
            
        return False, None
    
    async def _user_intervention(self, error_def: ErrorDefinition,
                                context: Dict[str, Any],
                                operation: Callable) -> tuple[bool, Any]:
        """사용자 개입 필요"""
        # 사용자 개입이 필요하므로 False 반환
        return False, None
    
    async def _abort_recovery(self, error_def: ErrorDefinition,
                            context: Dict[str, Any],
                            operation: Callable) -> tuple[bool, Any]:
        """작업 중단"""
        return False, None
    
    async def _ignore_recovery(self, error_def: ErrorDefinition,
                             context: Dict[str, Any],
                             operation: Callable) -> tuple[bool, Any]:
        """무시하고 계속"""
        return True, None


# ============================================================================
# 전역 인스턴스 및 헬퍼 함수
# ============================================================================

# 싱글톤 인스턴스들
_error_registry: Optional[ErrorRegistry] = None
_error_stats: Optional[ErrorStatistics] = None
_error_context: Optional[ErrorContext] = None
_error_config: Optional[ErrorHandlingConfig] = None
_recovery_manager: Optional[ErrorRecoveryManager] = None


def get_error_registry() -> ErrorRegistry:
    """에러 레지스트리 싱글톤"""
    global _error_registry
    if _error_registry is None:
        _error_registry = ErrorRegistry()
    return _error_registry


def get_error_statistics() -> ErrorStatistics:
    """에러 통계 싱글톤"""
    global _error_stats
    if _error_stats is None:
        _error_stats = ErrorStatistics()
    return _error_stats


def get_error_context() -> ErrorContext:
    """에러 컨텍스트 싱글톤"""
    global _error_context
    if _error_context is None:
        _error_context = ErrorContext()
    return _error_context


def get_error_config() -> ErrorHandlingConfig:
    """에러 설정 싱글톤"""
    global _error_config
    if _error_config is None:
        import os
        is_dev = os.getenv('STREAMLIT_ENV', 'development') == 'development'
        _error_config = ErrorHandlingConfig(
            show_technical_details=is_dev,
            show_stack_trace=is_dev,
            debug_mode=is_dev
        )
    return _error_config


def get_recovery_manager() -> ErrorRecoveryManager:
    """복구 관리자 싱글톤"""
    global _recovery_manager
    if _recovery_manager is None:
        _recovery_manager = ErrorRecoveryManager(get_error_config())
    return _recovery_manager


def format_error_message(error_code: str, context: Dict[str, Any] = None) -> str:
    """에러 메시지 포맷"""
    registry = get_error_registry()
    error_def = registry.get_error(error_code)
    
    if not error_def:
        return f"알 수 없는 에러: {error_code}"
    
    formatter = ErrorMessageFormatter()
    return formatter.format_user_message(error_def, context or {})


def get_recovery_suggestions(error_code: str) -> List[str]:
    """복구 제안사항 조회"""
    registry = get_error_registry()
    error_def = registry.get_error(error_code)
    
    if error_def:
        return error_def.recovery_suggestions
    return []


def record_user_action(action: str):
    """사용자 액션 기록 (에러 컨텍스트용)"""
    context = get_error_context()
    context.add_user_action(action)


# ============================================================================
# 에러 보고서 생성
# ============================================================================

def generate_error_report(error_code: str, 
                         exception: Optional[Exception] = None,
                         context: Dict[str, Any] = None) -> Dict[str, Any]:
    """상세 에러 보고서 생성"""
    registry = get_error_registry()
    error_def = registry.get_error(error_code)
    
    if not error_def:
        return {
            'error_code': error_code,
            'message': '알 수 없는 에러',
            'timestamp': datetime.now().isoformat()
        }
    
    # 에러 컨텍스트 캡처
    error_context = get_error_context()
    error_context.capture_system_state()
    
    # 통계 기록
    stats = get_error_statistics()
    stats.record_error(error_code)
    
    report = {
        'error_code': error_code,
        'error_name': error_def.name,
        'category': error_def.category.value,
        'severity': error_def.severity.value,
        'timestamp': datetime.now().isoformat(),
        'user_message': ErrorMessageFormatter.format_user_message(error_def, context or {}),
        'recovery_suggestions': error_def.recovery_suggestions,
        'can_continue': error_def.can_continue,
        'context': error_context.to_dict()
    }
    
    # 기술적 세부사항 (디버그 모드)
    config = get_error_config()
    if config.show_technical_details:
        report['technical_details'] = {
            'message': error_def.technical_message,
            'exception': str(exception) if exception else None,
            'traceback': traceback.format_exc() if exception else None,
            'error_frequency': stats.get_error_frequency(error_code),
            'recovery_rate': stats.get_recovery_rate(error_code)
        }
    
    return report


# ============================================================================
# Export
# ============================================================================

__all__ = [
    # Enums
    'ErrorCategory',
    'ErrorSeverity', 
    'RecoveryStrategy',
    
    # Classes
    'ErrorDefinition',
    'ErrorHandlingConfig',
    'ErrorMessageFormatter',
    'ErrorStatistics',
    'ErrorContext',
    'ErrorRegistry',
    'ErrorRecoveryManager',
    
    # Error Dictionaries
    'SYSTEM_ERRORS',
    'USER_INPUT_ERRORS',
    'DATA_ERRORS',
    'NETWORK_ERRORS',
    'API_ERRORS',
    'AUTH_ERRORS',
    'CALCULATION_ERRORS',
    'MODULE_ERRORS',
    
    # Functions
    'get_error_registry',
    'get_error_statistics',
    'get_error_context',
    'get_error_config',
    'get_recovery_manager',
    'format_error_message',
    'get_recovery_suggestions',
    'record_user_action',
    'generate_error_report'
]
