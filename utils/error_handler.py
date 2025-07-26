"""
🚨 Universal DOE Platform - 에러 처리 설정
================================================================================
모든 에러 코드, 메시지, 복구 전략을 중앙에서 관리
사용자 친화적이고 해결 지향적인 에러 처리 시스템
================================================================================
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import timedelta
import traceback
import logging
from pathlib import Path

# ============================================================================
# 🎯 에러 분류 체계
# ============================================================================

class ErrorCategory(Enum):
    """에러 카테고리 - 처리 방식 결정"""
    SYSTEM = "system"              # 시스템 레벨 오류
    USER_INPUT = "user_input"      # 사용자 입력 오류
    DATA = "data"                  # 데이터 관련 오류
    NETWORK = "network"            # 네트워크 오류
    API = "api"                    # API 호출 오류
    AUTH = "auth"                  # 인증/권한 오류
    FILE = "file"                  # 파일 처리 오류
    DATABASE = "database"          # 데이터베이스 오류
    CALCULATION = "calculation"    # 계산/분석 오류
    MODULE = "module"              # 모듈 관련 오류
    PROTOCOL = "protocol"          # 프로토콜 추출 오류
    UNKNOWN = "unknown"            # 알 수 없는 오류


class ErrorSeverity(Enum):
    """에러 심각도 - 표시 방식과 로깅 레벨 결정"""
    DEBUG = "debug"        # 디버그 정보 (개발자용)
    INFO = "info"          # 정보성 메시지
    WARNING = "warning"    # 경고 (계속 진행 가능)
    ERROR = "error"        # 오류 (일부 기능 제한)
    CRITICAL = "critical"  # 치명적 (프로그램 중단 위험)


class RecoveryStrategy(Enum):
    """복구 전략 - 에러 발생 시 대응 방법"""
    RETRY = "retry"              # 재시도
    FALLBACK = "fallback"        # 대체 방법 사용
    CACHE = "cache"              # 캐시된 데이터 사용
    DEFAULT = "default"          # 기본값 사용
    USER_INTERVENTION = "user"   # 사용자 개입 필요
    ABORT = "abort"              # 작업 중단
    IGNORE = "ignore"            # 무시하고 계속
    AUTO_FIX = "auto_fix"        # 자동 수정 시도


# ============================================================================
# 🏗️ 에러 정의 구조
# ============================================================================

@dataclass
class ErrorDefinition:
    """에러 정의 - 각 에러의 모든 정보를 담는 컨테이너"""
    # 기본 정보
    code: str                      # 에러 코드 (예: 1001)
    name: str                      # 에러 이름
    category: ErrorCategory        # 카테고리
    severity: ErrorSeverity        # 심각도
    
    # 메시지
    user_message: str              # 사용자용 메시지 (친화적)
    technical_message: str         # 개발자용 메시지 (상세)
    
    # 복구 정보
    recovery_strategy: RecoveryStrategy     # 복구 전략
    recovery_suggestions: List[str]         # 해결 제안사항
    recovery_actions: List[Dict[str, Any]] = field(default_factory=list)  # 자동 복구 액션
    
    # 추가 정보
    documentation_url: Optional[str] = None # 도움말 링크
    can_continue: bool = True              # 계속 진행 가능 여부
    auto_recoverable: bool = False         # 자동 복구 가능 여부
    max_retries: int = 3                   # 최대 재시도 횟수
    retry_delay: timedelta = timedelta(seconds=1)  # 재시도 간격
    
    # 컨텍스트 정보
    preserve_context: bool = True          # 작업 컨텍스트 보존
    log_full_trace: bool = True            # 전체 스택 트레이스 로깅
    notify_user: bool = True               # 사용자 알림 여부
    
    # 커스텀 핸들러
    custom_handler: Optional[Callable] = None
    
    # 관련 에러 코드
    related_errors: List[str] = field(default_factory=list)


# ============================================================================
# 🔧 자동 복구 액션 정의
# ============================================================================

RECOVERY_ACTIONS = {
    'try_encodings': {
        'function': 'utils.file_handler.try_multiple_encodings',
        'params': ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'cp949', 'gbk'],
        'description': '여러 인코딩으로 파일 읽기 시도'
    },
    'use_chardet': {
        'function': 'utils.file_handler.detect_encoding',
        'params': None,
        'description': 'chardet 라이브러리로 인코딩 자동 감지'
    },
    'fallback_binary': {
        'function': 'utils.file_handler.read_as_binary',
        'params': None,
        'description': '바이너리 모드로 읽기 시도'
    },
    'enhance_image': {
        'function': 'utils.ocr_handler.enhance_image',
        'params': ['contrast', 'sharpness', 'denoise'],
        'description': '이미지 품질 개선'
    },
    'try_different_ocr': {
        'function': 'utils.ocr_handler.try_multiple_engines',
        'params': ['tesseract', 'easyocr', 'pytesseract'],
        'description': '다른 OCR 엔진 시도'
    },
    'manual_input_prompt': {
        'function': 'ui.dialogs.show_manual_input',
        'params': None,
        'description': '수동 입력 다이얼로그 표시'
    },
    'clear_cache': {
        'function': 'utils.cache_manager.clear_cache',
        'params': ['temp', 'api_responses'],
        'description': '캐시 정리'
    },
    'switch_to_offline': {
        'function': 'config.offline_config.enable_offline_mode',
        'params': None,
        'description': '오프라인 모드로 전환'
    }
}


# ============================================================================
# 1️⃣ 시스템 에러 (1000-1999)
# ============================================================================

SYSTEM_ERRORS = {
    '1001': ErrorDefinition(
        code='1001',
        name='메모리 부족',
        category=ErrorCategory.SYSTEM,
        severity=ErrorSeverity.CRITICAL,
        user_message="😰 시스템 메모리가 부족합니다. 일부 프로그램을 종료 후 다시 시도해주세요.",
        technical_message="System memory exhausted: Available {available_mb}MB < Required {required_mb}MB",
        recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
        recovery_suggestions=[
            "다른 프로그램을 종료하세요",
            "대용량 파일을 닫으세요",
            "시스템을 재시작하세요",
            "더 작은 데이터로 시도하세요"
        ],
        recovery_actions=[
            {'action': 'clear_cache', 'params': None}
        ],
        can_continue=False,
        auto_recoverable=False
    ),
    
    '1002': ErrorDefinition(
        code='1002',
        name='디스크 공간 부족',
        category=ErrorCategory.SYSTEM,
        severity=ErrorSeverity.ERROR,
        user_message="💾 저장 공간이 부족합니다. {required_mb}MB의 공간이 필요합니다.",
        technical_message="Insufficient disk space: {available_mb}MB available, {required_mb}MB required",
        recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
        recovery_suggestions=[
            "불필요한 파일을 삭제하세요",
            "다른 드라이브를 선택하세요",
            "임시 파일을 정리하세요",
            "클라우드 저장소를 사용하세요"
        ],
        recovery_actions=[
            {'action': 'clear_cache', 'params': ['temp', 'old_backups']}
        ],
        can_continue=False,
        auto_recoverable=False
    ),
    
    '1003': ErrorDefinition(
        code='1003',
        name='권한 없음',
        category=ErrorCategory.SYSTEM,
        severity=ErrorSeverity.ERROR,
        user_message="🔒 이 작업을 수행할 권한이 없습니다.",
        technical_message="Permission denied: {operation} requires {permission} permission",
        recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
        recovery_suggestions=[
            "관리자 권한으로 실행하세요",
            "파일/폴더 권한을 확인하세요",
            "IT 관리자에게 문의하세요"
        ],
        can_continue=False
    )
}

# ============================================================================
# 2️⃣ 사용자 입력 에러 (2000-2999)
# ============================================================================

USER_INPUT_ERRORS = {
    '2001': ErrorDefinition(
        code='2001',
        name='잘못된 입력값',
        category=ErrorCategory.USER_INPUT,
        severity=ErrorSeverity.WARNING,
        user_message="📝 입력값이 올바르지 않습니다. {field_name}은(는) {constraint}이어야 합니다.",
        technical_message="Invalid input for {field_name}: {value} does not meet constraint {constraint}",
        recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
        recovery_suggestions=[
            "입력값을 다시 확인하세요",
            "예시: {example}",
            "도움말을 참조하세요"
        ],
        can_continue=True,
        auto_recoverable=False
    ),
    
    '2002': ErrorDefinition(
        code='2002',
        name='필수 입력 누락',
        category=ErrorCategory.USER_INPUT,
        severity=ErrorSeverity.WARNING,
        user_message="⚠️ 필수 항목이 입력되지 않았습니다: {field_names}",
        technical_message="Required fields missing: {field_names}",
        recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
        recovery_suggestions=[
            "빨간색으로 표시된 필수 항목을 입력하세요",
            "모든 필수 항목(*)을 확인하세요"
        ],
        can_continue=True
    ),
    
    '2003': ErrorDefinition(
        code='2003',
        name='범위 초과',
        category=ErrorCategory.USER_INPUT,
        severity=ErrorSeverity.WARNING,
        user_message="📏 입력값이 허용 범위를 벗어났습니다. {min_value}에서 {max_value} 사이의 값을 입력하세요.",
        technical_message="Value out of range: {value} not in [{min_value}, {max_value}]",
        recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
        recovery_suggestions=[
            "허용 범위 내의 값을 입력하세요",
            "단위를 확인하세요",
            "기본값을 사용하시겠습니까?"
        ],
        can_continue=True
    )
}

# ============================================================================
# 3️⃣ 네트워크 에러 (3000-3999)
# ============================================================================

NETWORK_ERRORS = {
    '3001': ErrorDefinition(
        code='3001',
        name='인터넷 연결 없음',
        category=ErrorCategory.NETWORK,
        severity=ErrorSeverity.ERROR,
        user_message="🌐 인터넷에 연결할 수 없습니다. 오프라인 모드로 전환합니다.",
        technical_message="Network unreachable: Connection timeout after {timeout}s",
        recovery_strategy=RecoveryStrategy.FALLBACK,
        recovery_suggestions=[
            "인터넷 연결을 확인하세요",
            "방화벽 설정을 확인하세요",
            "오프라인 모드에서 계속 작업하세요"
        ],
        recovery_actions=[
            {'action': 'switch_to_offline', 'params': None}
        ],
        can_continue=True,
        auto_recoverable=True
    ),
    
    '3002': ErrorDefinition(
        code='3002',
        name='서버 응답 없음',
        category=ErrorCategory.NETWORK,
        severity=ErrorSeverity.ERROR,
        user_message="⏱️ 서버가 응답하지 않습니다. 잠시 후 다시 시도해주세요.",
        technical_message="Server timeout: No response from {server} after {timeout}s",
        recovery_strategy=RecoveryStrategy.RETRY,
        recovery_suggestions=[
            "잠시 기다린 후 다시 시도하세요",
            "서버 상태를 확인하세요",
            "다른 서버를 선택하세요"
        ],
        can_continue=True,
        auto_recoverable=True,
        max_retries=3,
        retry_delay=timedelta(seconds=5)
    )
}

# ============================================================================
# 4️⃣ 파일 처리 에러 (4000-4999)
# ============================================================================

FILE_ERRORS = {
    '4001': ErrorDefinition(
        code='4001',
        name='파일을 찾을 수 없음',
        category=ErrorCategory.FILE,
        severity=ErrorSeverity.ERROR,
        user_message="📁 파일을 찾을 수 없습니다: {filename}",
        technical_message="File not found: {filepath}",
        recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
        recovery_suggestions=[
            "파일 경로를 확인하세요",
            "파일이 이동되었는지 확인하세요",
            "다른 파일을 선택하세요"
        ],
        can_continue=True
    ),
    
    '4002': ErrorDefinition(
        code='4002',
        name='파일 읽기 실패',
        category=ErrorCategory.FILE,
        severity=ErrorSeverity.ERROR,
        user_message="📖 파일을 읽을 수 없습니다. 파일이 손상되었거나 권한이 없을 수 있습니다.",
        technical_message="File read error: {error_detail}",
        recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
        recovery_suggestions=[
            "파일 권한을 확인하세요",
            "파일이 다른 프로그램에서 사용 중인지 확인하세요",
            "파일을 다시 다운로드하세요"
        ],
        can_continue=True
    ),
    
    # === 프로토콜 추출 관련 에러 (4200-4299) ===
    '4200': ErrorDefinition(
        code='4200',
        name='지원하지 않는 파일 형식',
        category=ErrorCategory.PROTOCOL,
        severity=ErrorSeverity.WARNING,
        user_message="😕 이 파일 형식은 아직 지원하지 않아요. PDF나 Word 파일로 변환해주세요!",
        technical_message="Unsupported file format: {file_type}. Supported: PDF, TXT, DOCX, HTML, MD, RTF",
        recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
        recovery_suggestions=[
            "지원 형식: PDF, TXT, DOCX, HTML, MD, RTF",
            "파일을 다른 형식으로 변환 후 시도하세요",
            "텍스트를 직접 복사-붙여넣기 하세요"
        ],
        can_continue=True,
        related_errors=['4201', '4202']
    ),
    
    '4201': ErrorDefinition(
        code='4201',
        name='파일 인코딩 감지 실패',
        category=ErrorCategory.PROTOCOL,
        severity=ErrorSeverity.ERROR,
        user_message="🤔 파일의 텍스트를 읽을 수 없어요. 메모장에서 UTF-8로 저장 후 다시 시도해주세요.",
        technical_message="Encoding detection failed for file: {filename}",
        recovery_strategy=RecoveryStrategy.AUTO_FIX,
        recovery_suggestions=[
            "메모장에서 파일을 열고 'UTF-8'로 다시 저장하세요",
            "다른 텍스트 에디터를 사용해보세요",
            "파일 내용을 복사하여 직접 입력하세요"
        ],
        recovery_actions=[
            {'action': 'try_encodings', 'params': None},
            {'action': 'use_chardet', 'params': None},
            {'action': 'fallback_binary', 'params': None}
        ],
        can_continue=True,
        auto_recoverable=True,
        max_retries=5
    ),
    
    '4202': ErrorDefinition(
        code='4202',
        name='프로토콜 추출 실패',
        category=ErrorCategory.PROTOCOL,
        severity=ErrorSeverity.ERROR,
        user_message="🔍 실험 방법을 찾을 수 없어요. Methods나 Experimental 섹션이 있는지 확인해주세요.",
        technical_message="Protocol extraction failed: No methods section found in document",
        recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
        recovery_suggestions=[
            "문서에 'Methods', 'Experimental', 'Procedure' 섹션이 있는지 확인하세요",
            "해당 섹션만 선택하여 다시 시도하세요",
            "프로토콜 템플릿을 사용해보세요",
            "수동으로 프로토콜 정보를 입력하세요"
        ],
        can_continue=True,
        documentation_url="https://docs.universaldoe.com/protocol-extraction"
    ),
    
    '4203': ErrorDefinition(
        code='4203',
        name='텍스트가 너무 김',
        category=ErrorCategory.PROTOCOL,
        severity=ErrorSeverity.WARNING,
        user_message="📏 텍스트가 너무 길어요. 중요한 부분만 선택해서 다시 시도해주세요.",
        technical_message="Text too long: {length} characters exceeds maximum {max_length}",
        recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
        recovery_suggestions=[
            "Methods 섹션만 선택하세요",
            "여러 부분으로 나누어 처리하세요",
            "불필요한 부분을 제거하세요",
            "요약된 버전을 사용하세요"
        ],
        can_continue=True
    ),
    
    '4204': ErrorDefinition(
        code='4204',
        name='문서 구조 오류',
        category=ErrorCategory.PROTOCOL,
        severity=ErrorSeverity.ERROR,
        user_message="📄 문서 구조가 올바르지 않습니다. 파일이 손상되었을 수 있습니다.",
        technical_message="Invalid document structure: {error_detail}",
        recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
        recovery_suggestions=[
            "파일을 다시 생성하거나 다운로드하세요",
            "다른 프로그램에서 열어 다시 저장하세요",
            "PDF를 텍스트로 변환 후 시도하세요",
            "구조화된 템플릿을 사용하세요"
        ],
        can_continue=True
    ),
    
    '4205': ErrorDefinition(
        code='4205',
        name='OCR 처리 오류',
        category=ErrorCategory.PROTOCOL,
        severity=ErrorSeverity.ERROR,
        user_message="👁️ 스캔된 이미지에서 텍스트를 추출할 수 없습니다. 텍스트 파일을 사용해주세요.",
        technical_message="OCR processing failed: {error_detail}",
        recovery_strategy=RecoveryStrategy.AUTO_FIX,
        recovery_suggestions=[
            "이미지 품질을 개선하세요 (300DPI 이상)",
            "텍스트가 선명한 페이지만 스캔하세요",
            "텍스트를 직접 입력하세요",
            "다른 OCR 소프트웨어를 사용하세요"
        ],
        recovery_actions=[
            {'action': 'enhance_image', 'params': None},
            {'action': 'try_different_ocr', 'params': None},
            {'action': 'manual_input_prompt', 'params': None}
        ],
        can_continue=True,
        auto_recoverable=True
    ),
    
    '4206': ErrorDefinition(
        code='4206',
        name='다중 파일 처리 오류',
        category=ErrorCategory.PROTOCOL,
        severity=ErrorSeverity.WARNING,
        user_message="📚 일부 파일 처리에 실패했습니다. 성공한 파일들로 계속 진행합니다.",
        technical_message="Multi-file processing error: {failed_count} of {total_count} files failed",
        recovery_strategy=RecoveryStrategy.FALLBACK,
        recovery_suggestions=[
            "실패한 파일을 개별적으로 처리하세요",
            "파일 형식을 통일하세요",
            "한 번에 처리하는 파일 수를 줄이세요"
        ],
        can_continue=True,
        auto_recoverable=True,
        preserve_context=True
    ),
    
    '4207': ErrorDefinition(
        code='4207',
        name='URL 콘텐츠 가져오기 실패',
        category=ErrorCategory.PROTOCOL,
        severity=ErrorSeverity.ERROR,
        user_message="🔗 웹페이지에 접근할 수 없습니다. URL을 확인하거나 파일로 다운로드하세요.",
        technical_message="URL fetch failed: {url} - {status_code}",
        recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
        recovery_suggestions=[
            "URL이 정확한지 확인하세요",
            "웹페이지를 PDF로 저장 후 업로드하세요",
            "다른 브라우저에서 시도하세요",
            "VPN이나 프록시 설정을 확인하세요"
        ],
        can_continue=True
    ),
    
    '4208': ErrorDefinition(
        code='4208',
        name='프로토콜 분석 시간 초과',
        category=ErrorCategory.PROTOCOL,
        severity=ErrorSeverity.ERROR,
        user_message="⏰ 프로토콜 분석이 너무 오래 걸립니다. 더 작은 문서로 시도해주세요.",
        technical_message="Protocol parsing timeout: Exceeded {timeout}s limit",
        recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
        recovery_suggestions=[
            "문서를 여러 부분으로 나누세요",
            "Methods 섹션만 추출하세요",
            "간단한 형식의 문서를 사용하세요",
            "텍스트 형식으로 변환 후 시도하세요"
        ],
        can_continue=True
    ),
    
    '4209': ErrorDefinition(
        code='4209',
        name='프로토콜 정보 부족',
        category=ErrorCategory.PROTOCOL,
        severity=ErrorSeverity.WARNING,
        user_message="📋 프로토콜 정보가 부족해요. 최소한 재료와 실험 절차는 포함되어야 해요.",
        technical_message="Insufficient protocol data: Missing required sections {missing_sections}",
        recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
        recovery_suggestions=[
            "재료(Materials) 섹션을 추가하세요",
            "실험 절차(Procedure)를 상세히 작성하세요",
            "템플릿을 참고하여 작성하세요",
            "예제 프로토콜을 참조하세요"
        ],
        can_continue=True,
        documentation_url="https://docs.universaldoe.com/protocol-template"
    )
}

# ============================================================================
# 5️⃣ API 에러 (5000-5999)
# ============================================================================

API_ERRORS = {
    '5001': ErrorDefinition(
        code='5001',
        name='API 키 없음',
        category=ErrorCategory.API,
        severity=ErrorSeverity.ERROR,
        user_message="🔑 API 키가 설정되지 않았습니다. 설정에서 API 키를 입력해주세요.",
        technical_message="API key not found for service: {service}",
        recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
        recovery_suggestions=[
            "설정 > API 키 관리에서 키를 입력하세요",
            "{service} 웹사이트에서 API 키를 발급받으세요",
            "무료 API 키 발급 가이드를 참조하세요"
        ],
        can_continue=False,
        documentation_url="https://docs.universaldoe.com/api-keys"
    ),
    
    '5002': ErrorDefinition(
        code='5002',
        name='API 한도 초과',
        category=ErrorCategory.API,
        severity=ErrorSeverity.WARNING,
        user_message="⚡ API 사용 한도를 초과했습니다. {reset_time}에 초기화됩니다.",
        technical_message="API rate limit exceeded: {current}/{limit} requests",
        recovery_strategy=RecoveryStrategy.FALLBACK,
        recovery_suggestions=[
            "다른 AI 엔진을 사용하세요",
            "잠시 기다린 후 다시 시도하세요",
            "API 플랜을 업그레이드하세요"
        ],
        can_continue=True,
        auto_recoverable=True
    )
}

# ============================================================================
# 6️⃣ 데이터베이스 에러 (6000-6999)
# ============================================================================

DATABASE_ERRORS = {
    '6001': ErrorDefinition(
        code='6001',
        name='데이터베이스 연결 실패',
        category=ErrorCategory.DATABASE,
        severity=ErrorSeverity.ERROR,
        user_message="🗄️ 데이터베이스에 연결할 수 없습니다. 로컬 모드로 전환합니다.",
        technical_message="Database connection failed: {error_detail}",
        recovery_strategy=RecoveryStrategy.FALLBACK,
        recovery_suggestions=[
            "인터넷 연결을 확인하세요",
            "로컬 데이터베이스를 사용합니다",
            "잠시 후 다시 시도하세요"
        ],
        recovery_actions=[
            {'action': 'switch_to_offline', 'params': None}
        ],
        can_continue=True,
        auto_recoverable=True
    ),
    
    '6002': ErrorDefinition(
        code='6002',
        name='데이터 저장 실패',
        category=ErrorCategory.DATABASE,
        severity=ErrorSeverity.ERROR,
        user_message="💾 데이터를 저장할 수 없습니다. 임시 저장소에 보관됩니다.",
        technical_message="Data save failed: {table} - {error_detail}",
        recovery_strategy=RecoveryStrategy.CACHE,
        recovery_suggestions=[
            "나중에 자동으로 동기화됩니다",
            "수동으로 백업을 생성하세요",
            "저장 공간을 확인하세요"
        ],
        can_continue=True,
        auto_recoverable=True,
        preserve_context=True
    )
}

# ============================================================================
# 7️⃣ 계산/분석 에러 (7000-7999)
# ============================================================================

CALCULATION_ERRORS = {
    '7001': ErrorDefinition(
        code='7001',
        name='계산 오류',
        category=ErrorCategory.CALCULATION,
        severity=ErrorSeverity.ERROR,
        user_message="🧮 계산 중 오류가 발생했습니다. 입력 데이터를 확인해주세요.",
        technical_message="Calculation error in {function}: {error_detail}",
        recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
        recovery_suggestions=[
            "입력값에 0이나 음수가 있는지 확인하세요",
            "데이터 범위가 적절한지 확인하세요",
            "다른 계산 방법을 시도하세요"
        ],
        can_continue=True,
        preserve_context=True
    ),
    
    '7002': ErrorDefinition(
        code='7002',
        name='통계 분석 실패',
        category=ErrorCategory.CALCULATION,
        severity=ErrorSeverity.ERROR,
        user_message="📊 통계 분석을 수행할 수 없습니다. 데이터가 부족하거나 형식이 맞지 않습니다.",
        technical_message="Statistical analysis failed: {method} requires {requirement}",
        recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
        recovery_suggestions=[
            "최소 {min_samples}개 이상의 데이터가 필요합니다",
            "결측값을 처리해주세요",
            "데이터 형식을 확인하세요"
        ],
        can_continue=False
    )
}

# ============================================================================
# 8️⃣ 인증/권한 에러 (8000-8999)
# ============================================================================

AUTH_ERRORS = {
    '8001': ErrorDefinition(
        code='8001',
        name='인증 실패',
        category=ErrorCategory.AUTH,
        severity=ErrorSeverity.ERROR,
        user_message="🔐 로그인 정보가 올바르지 않습니다.",
        technical_message="Authentication failed: Invalid credentials",
        recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
        recovery_suggestions=[
            "이메일과 비밀번호를 확인하세요",
            "Caps Lock이 켜져있는지 확인하세요",
            "비밀번호를 재설정하세요"
        ],
        can_continue=False
    ),
    
    '8002': ErrorDefinition(
        code='8002',
        name='세션 만료',
        category=ErrorCategory.AUTH,
        severity=ErrorSeverity.WARNING,
        user_message="⏰ 세션이 만료되었습니다. 다시 로그인해주세요.",
        technical_message="Session expired after {duration}",
        recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
        recovery_suggestions=[
            "다시 로그인하세요",
            "작업 내용은 자동 저장되었습니다"
        ],
        can_continue=False,
        preserve_context=True
    )
}

# ============================================================================
# 9️⃣ 모듈 에러 (9000-9999)
# ============================================================================

MODULE_ERRORS = {
    '9001': ErrorDefinition(
        code='9001',
        name='모듈 로드 실패',
        category=ErrorCategory.MODULE,
        severity=ErrorSeverity.ERROR,
        user_message="🧩 모듈을 불러올 수 없습니다: {module_name}",
        technical_message="Module load failed: {module_path} - {error_detail}",
        recovery_strategy=RecoveryStrategy.USER_INTERVENTION,
        recovery_suggestions=[
            "모듈이 올바르게 설치되었는지 확인하세요",
            "모듈 의존성을 확인하세요",
            "모듈을 다시 설치하세요"
        ],
        can_continue=True
    )
}

# ============================================================================
# 🔧 에러 처리 설정
# ============================================================================

# 모든 에러 통합
ERROR_CODES = {
    **SYSTEM_ERRORS,
    **USER_INPUT_ERRORS,
    **NETWORK_ERRORS,
    **FILE_ERRORS,
    **API_ERRORS,
    **DATABASE_ERRORS,
    **CALCULATION_ERRORS,
    **AUTH_ERRORS,
    **MODULE_ERRORS
}

# 카테고리별 에러 분류
ERROR_BY_CATEGORY = {
    ErrorCategory.SYSTEM: SYSTEM_ERRORS,
    ErrorCategory.USER_INPUT: USER_INPUT_ERRORS,
    ErrorCategory.NETWORK: NETWORK_ERRORS,
    ErrorCategory.FILE: FILE_ERRORS,
    ErrorCategory.API: API_ERRORS,
    ErrorCategory.DATABASE: DATABASE_ERRORS,
    ErrorCategory.CALCULATION: CALCULATION_ERRORS,
    ErrorCategory.AUTH: AUTH_ERRORS,
    ErrorCategory.MODULE: MODULE_ERRORS,
    ErrorCategory.PROTOCOL: {k: v for k, v in FILE_ERRORS.items() if k.startswith('42')}
}

# 심각도별 설정
ERROR_SEVERITY_CONFIG = {
    ErrorSeverity.DEBUG: {
        'color': '#6B7280',  # 회색
        'icon': '🐛',
        'log_level': 'DEBUG',
        'notify_user': False
    },
    ErrorSeverity.INFO: {
        'color': '#3B82F6',  # 파란색
        'icon': 'ℹ️',
        'log_level': 'INFO',
        'notify_user': True
    },
    ErrorSeverity.WARNING: {
        'color': '#F59E0B',  # 주황색
        'icon': '⚠️',
        'log_level': 'WARNING',
        'notify_user': True
    },
    ErrorSeverity.ERROR: {
        'color': '#EF4444',  # 빨간색
        'icon': '❌',
        'log_level': 'ERROR',
        'notify_user': True
    },
    ErrorSeverity.CRITICAL: {
        'color': '#991B1B',  # 진한 빨간색
        'icon': '🚨',
        'log_level': 'CRITICAL',
        'notify_user': True
    }
}

# 복구 전략별 설정
RECOVERY_CONFIG = {
    RecoveryStrategy.RETRY: {
        'max_attempts': 3,
        'delay': timedelta(seconds=1),
        'backoff_factor': 2.0,
        'auto_execute': True
    },
    RecoveryStrategy.FALLBACK: {
        'auto_execute': True,
        'notify_user': True
    },
    RecoveryStrategy.CACHE: {
        'auto_execute': True,
        'cache_duration': timedelta(hours=24)
    },
    RecoveryStrategy.DEFAULT: {
        'auto_execute': True,
        'notify_user': False
    },
    RecoveryStrategy.USER_INTERVENTION: {
        'auto_execute': False,
        'show_dialog': True
    },
    RecoveryStrategy.ABORT: {
        'auto_execute': True,
        'cleanup': True
    },
    RecoveryStrategy.IGNORE: {
        'auto_execute': True,
        'log_only': True
    },
    RecoveryStrategy.AUTO_FIX: {
        'auto_execute': True,
        'notify_user': True,
        'log_attempts': True
    }
}

# 에러 메시지 템플릿
ERROR_MESSAGE_TEMPLATES = {
    'user_friendly': "{icon} {message}\n\n💡 해결 방법:\n{suggestions}",
    'technical': "[{code}] {category}.{name}: {technical_message}\nStack: {stack_trace}",
    'log_format': "{timestamp} - {severity} - [{code}] {message} - Context: {context}",
    'notification': "{icon} {name}\n{message}"
}

# 에러 그룹화 규칙
ERROR_GROUPING_RULES = {
    'similar_threshold': 0.8,  # 유사도 임계값
    'time_window': timedelta(minutes=5),  # 그룹화 시간 창
    'max_group_size': 10,  # 최대 그룹 크기
    'grouping_enabled': True
}

# 형식별 에러 처리 가이드
FORMAT_SPECIFIC_ERRORS = {
    'pdf': ['4202', '4205', '4208'],
    'docx': ['4204', '4201'],
    'html': ['4207', '4204'],
    'txt': ['4201', '4203'],
    'multi': ['4206', '4209']
}

# 자동 복구 전략 매핑
ERROR_RECOVERY_STRATEGIES = {
    '4201': [
        {'action': 'try_encodings', 'params': ['utf-8', 'latin-1', 'cp1252']},
        {'action': 'use_chardet', 'params': None},
        {'action': 'fallback_binary', 'params': None}
    ],
    '4205': [
        {'action': 'enhance_image', 'params': ['contrast', 'sharpness']},
        {'action': 'try_different_ocr', 'params': ['tesseract', 'easyocr']},
        {'action': 'manual_input_prompt', 'params': None}
    ],
    '3001': [
        {'action': 'switch_to_offline', 'params': None}
    ],
    '6001': [
        {'action': 'switch_to_offline', 'params': None}
    ]
}

# ============================================================================
# 🛠️ 유틸리티 함수
# ============================================================================

def get_error_definition(error_code: str) -> Optional[ErrorDefinition]:
    """에러 코드로 에러 정의 조회"""
    return ERROR_CODES.get(error_code)


def get_errors_by_category(category: ErrorCategory) -> Dict[str, ErrorDefinition]:
    """카테고리별 에러 목록 조회"""
    return ERROR_BY_CATEGORY.get(category, {})


def get_errors_by_severity(severity: ErrorSeverity) -> Dict[str, ErrorDefinition]:
    """심각도별 에러 목록 조회"""
    return {
        code: error for code, error in ERROR_CODES.items()
        if error.severity == severity
    }


def format_error_message(error_code: str, context: Dict[str, Any] = None) -> str:
    """에러 메시지 포맷팅"""
    error_def = get_error_definition(error_code)
    if not error_def:
        return f"Unknown error: {error_code}"
    
    context = context or {}
    severity_config = ERROR_SEVERITY_CONFIG[error_def.severity]
    
    try:
        message = error_def.user_message.format(**context)
    except KeyError:
        message = error_def.user_message
    
    suggestions = "\n".join([f"• {s}" for s in error_def.recovery_suggestions])
    
    return ERROR_MESSAGE_TEMPLATES['user_friendly'].format(
        icon=severity_config['icon'],
        message=message,
        suggestions=suggestions
    )


def should_auto_recover(error_code: str) -> bool:
    """자동 복구 가능 여부 확인"""
    error_def = get_error_definition(error_code)
    return error_def.auto_recoverable if error_def else False


def get_recovery_actions(error_code: str) -> List[Dict[str, Any]]:
    """에러 코드에 대한 복구 액션 목록 반환"""
    error_def = get_error_definition(error_code)
    if error_def and error_def.recovery_actions:
        return error_def.recovery_actions
    return ERROR_RECOVERY_STRATEGIES.get(error_code, [])


def log_error(error_code: str, context: Dict[str, Any] = None, exception: Exception = None):
    """에러 로깅"""
    error_def = get_error_definition(error_code)
    if not error_def:
        logging.error(f"Unknown error code: {error_code}")
        return
    
    severity_config = ERROR_SEVERITY_CONFIG[error_def.severity]
    log_level = getattr(logging, severity_config['log_level'])
    
    log_message = ERROR_MESSAGE_TEMPLATES['log_format'].format(
        timestamp=datetime.now().isoformat(),
        severity=error_def.severity.value,
        code=error_code,
        message=error_def.technical_message,
        context=context or {}
    )
    
    if exception and error_def.log_full_trace:
        log_message += f"\nException: {str(exception)}\nTrace: {traceback.format_exc()}"
    
    log_level(log_message)


def get_user_friendly_message(error_code: str) -> str:
    """사용자 친화적 메시지만 반환 (컨텍스트 없이)"""
    error_def = get_error_definition(error_code)
    if not error_def:
        return "알 수 없는 오류가 발생했습니다."
    
    severity_config = ERROR_SEVERITY_CONFIG[error_def.severity]
    return f"{severity_config['icon']} {error_def.user_message}"


def get_error_color(error_code: str) -> str:
    """에러 심각도에 따른 색상 반환"""
    error_def = get_error_definition(error_code)
    if not error_def:
        return '#6B7280'  # 기본 회색
    
    return ERROR_SEVERITY_CONFIG[error_def.severity]['color']


def group_similar_errors(errors: List[str]) -> Dict[str, List[str]]:
    """유사한 에러들을 그룹화"""
    groups = {}
    for error_code in errors:
        error_def = get_error_definition(error_code)
        if error_def:
            category = error_def.category.value
            if category not in groups:
                groups[category] = []
            groups[category].append(error_code)
    return groups


# ============================================================================
# 📤 Public API
# ============================================================================

__all__ = [
    # Enums
    'ErrorCategory', 'ErrorSeverity', 'RecoveryStrategy',
    
    # Classes
    'ErrorDefinition',
    
    # Error Collections
    'ERROR_CODES', 'ERROR_BY_CATEGORY', 'SYSTEM_ERRORS', 'USER_INPUT_ERRORS',
    'NETWORK_ERRORS', 'FILE_ERRORS', 'API_ERRORS', 'DATABASE_ERRORS',
    'CALCULATION_ERRORS', 'AUTH_ERRORS', 'MODULE_ERRORS',
    
    # Configurations
    'ERROR_SEVERITY_CONFIG', 'RECOVERY_CONFIG', 'ERROR_MESSAGE_TEMPLATES',
    'ERROR_GROUPING_RULES', 'FORMAT_SPECIFIC_ERRORS', 'ERROR_RECOVERY_STRATEGIES',
    'RECOVERY_ACTIONS',
    
    # Utility Functions
    'get_error_definition', 'get_errors_by_category', 'get_errors_by_severity',
    'format_error_message', 'should_auto_recover', 'get_recovery_actions',
    'log_error', 'get_user_friendly_message', 'get_error_color',
    'group_similar_errors'
]
