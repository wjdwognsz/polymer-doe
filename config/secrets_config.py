"""
🔐 Universal DOE Platform - 보안 설정
================================================================================
API 키 및 인증 정보의 구조와 검증 규칙 정의
데스크톱 애플리케이션을 위한 안전한 로컬 저장 지원
무료 API 우선 정책으로 접근성 극대화
================================================================================
"""

from typing import Dict, Optional, List, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import re
import os
from pathlib import Path

# ============================================================================
# 🔑 비밀 정보 정의 클래스
# ============================================================================

@dataclass
class SecretDefinition:
    """비밀 정보 정의"""
    name: str                          # 내부 식별자
    display_name: str                  # UI 표시명
    env_var: str                       # 환경 변수명
    secrets_key: str                   # secrets.toml 키
    required: bool = False             # 필수 여부
    validation_prefix: Optional[str] = None  # 키 시작 문자열
    validation_pattern: Optional[str] = None  # 정규식 패턴
    min_length: int = 10               # 최소 길이
    max_length: int = 200              # 최대 길이
    is_json: bool = False              # JSON 형식 여부
    docs_url: Optional[str] = None     # 문서 URL
    example: Optional[str] = None      # 예시값
    description: Optional[str] = None  # 설명
    
    def validate(self, value: str) -> bool:
        """값 검증"""
        if not value:
            return not self.required
            
        # 길이 검증
        if len(value) < self.min_length or len(value) > self.max_length:
            return False
            
        # 프리픽스 검증
        if self.validation_prefix and not value.startswith(self.validation_prefix):
            return False
            
        # 패턴 검증
        if self.validation_pattern and not re.match(self.validation_pattern, value):
            return False
            
        return True

# ============================================================================
# 🤖 AI API 키 정의
# ============================================================================

AI_SECRET_DEFINITIONS = {
    # 필수 AI (무료)
    'google_gemini': SecretDefinition(
        name='google_gemini',
        display_name='Google Gemini API',
        env_var='GOOGLE_GEMINI_API_KEY',
        secrets_key='google_gemini_key',
        required=True,  # 필수 (무료 티어 제공)
        validation_prefix='AIza',
        min_length=39,
        docs_url='https://makersuite.google.com/app/apikey',
        example='AIzaSy...',
        description='Google Gemini API 키 (무료 티어: 60 req/min)'
    ),
    
    # 선택 AI APIs
    'groq': SecretDefinition(
        name='groq',
        display_name='Groq API',
        env_var='GROQ_API_KEY',
        secrets_key='groq_key',
        required=False,
        validation_prefix='gsk_',
        min_length=56,
        docs_url='https://console.groq.com/keys',
        example='gsk_...',
        description='Groq API 키 (무료 티어 제공, 초고속 추론)'
    ),
    
    'huggingface': SecretDefinition(
        name='huggingface',
        display_name='HuggingFace Token',
        env_var='HUGGINGFACE_TOKEN',
        secrets_key='huggingface_token',
        required=False,
        validation_prefix='hf_',
        min_length=20,
        docs_url='https://huggingface.co/settings/tokens',
        example='hf_...',
        description='HuggingFace 액세스 토큰 (무료, 도메인 특화 모델)'
    ),
    
    'xai_grok': SecretDefinition(
        name='xai_grok',
        display_name='xAI Grok API',
        env_var='XAI_GROK_API_KEY',
        secrets_key='xai_grok_key',
        required=False,
        validation_prefix='xai-',
        min_length=40,
        docs_url='https://console.x.ai',
        example='xai-...',
        description='xAI Grok API 키 (유료)'
    ),
    
    'deepseek': SecretDefinition(
        name='deepseek',
        display_name='DeepSeek API',
        env_var='DEEPSEEK_API_KEY',
        secrets_key='deepseek_key',
        required=False,
        validation_prefix='sk-',
        min_length=32,
        docs_url='https://platform.deepseek.com/api_keys',
        example='sk-...',
        description='DeepSeek API 키 (코드/수식 특화)'
    ),
    
    'sambanova': SecretDefinition(
        name='sambanova',
        display_name='SambaNova API',
        env_var='SAMBANOVA_API_KEY',
        secrets_key='sambanova_key',
        required=False,
        min_length=20,
        docs_url='https://cloud.sambanova.ai',
        example='...',
        description='SambaNova Cloud API 키 (무료 클라우드 추론)'
    ),
}

# ============================================================================
# 📚 문헌/데이터베이스 API 키 정의
# ============================================================================

LITERATURE_SECRET_DEFINITIONS = {
    # 무료 APIs (키 선택적)
    'openalex': SecretDefinition(
        name='openalex',
        display_name='OpenAlex Email',
        env_var='OPENALEX_EMAIL',
        secrets_key='openalex_email',
        required=False,
        validation_pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        min_length=5,
        docs_url='https://docs.openalex.org/how-to-use-the-api/rate-limits-and-authentication',
        example='your-email@example.com',
        description='OpenAlex Polite Request용 이메일 (선택적, 속도 향상)'
    ),
    
    'crossref': SecretDefinition(
        name='crossref',
        display_name='Crossref Email',
        env_var='CROSSREF_EMAIL',
        secrets_key='crossref_email',
        required=False,
        validation_pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        min_length=5,
        docs_url='https://www.crossref.org/documentation/retrieve-metadata/rest-api/',
        example='your-email@example.com',
        description='Crossref Polite Request용 이메일 (선택적)'
    ),
    
    # 재료 데이터베이스
    'materials_project': SecretDefinition(
        name='materials_project',
        display_name='Materials Project API',
        env_var='MATERIALS_PROJECT_API_KEY',
        secrets_key='materials_project_key',
        required=False,
        min_length=20,
        docs_url='https://materialsproject.org/api',
        example='...',
        description='Materials Project API 키 (무료 가입 필요)'
    ),
    
    # 리포지토리
    'github': SecretDefinition(
        name='github',
        display_name='GitHub Token',
        env_var='GITHUB_TOKEN',
        secrets_key='github_token',
        required=False,
        validation_prefix='ghp_',
        min_length=40,
        docs_url='https://github.com/settings/tokens',
        example='ghp_...',
        description='GitHub Personal Access Token (공개 리포 무료)'
    ),
    
    'zenodo': SecretDefinition(
        name='zenodo',
        display_name='Zenodo Token',
        env_var='ZENODO_ACCESS_TOKEN',
        secrets_key='zenodo_token',
        required=False,
        min_length=20,
        docs_url='https://zenodo.org/account/settings/applications/tokens/new/',
        example='...',
        description='Zenodo 액세스 토큰 (연구 데이터셋)'
    ),
    
    'protocols_io': SecretDefinition(
        name='protocols_io',
        display_name='protocols.io Token',
        env_var='PROTOCOLS_IO_TOKEN',
        secrets_key='protocols_io_token',
        required=False,
        min_length=20,
        docs_url='https://www.protocols.io/developers',
        example='...',
        description='protocols.io API 토큰 (실험 프로토콜)'
    ),
    
    'figshare': SecretDefinition(
        name='figshare',
        display_name='Figshare Token',
        env_var='FIGSHARE_TOKEN',
        secrets_key='figshare_token',
        required=False,
        min_length=20,
        docs_url='https://help.figshare.com/article/how-to-get-a-personal-token',
        example='...',
        description='Figshare Personal Token (연구 데이터 공유)'
    ),
}

# ============================================================================
# 🔗 Google 서비스 설정
# ============================================================================

GOOGLE_SECRET_DEFINITIONS = {
    'google_sheets_url': SecretDefinition(
        name='google_sheets_url',
        display_name='Google Sheets URL',
        env_var='GOOGLE_SHEETS_URL',
        secrets_key='google_sheets_url',
        required=False,  # 선택적 (로컬 저장 우선)
        validation_pattern=r'https://docs\.google\.com/spreadsheets/d/[\w-]+',
        min_length=40,
        example='https://docs.google.com/spreadsheets/d/...',
        description='Google Sheets URL (데이터 동기화용)'
    ),
    
    'google_oauth_client_id': SecretDefinition(
        name='google_oauth_client_id',
        display_name='Google OAuth Client ID',
        env_var='GOOGLE_OAUTH_CLIENT_ID',
        secrets_key='google_oauth_client_id',
        required=False,
        min_length=20,
        example='....apps.googleusercontent.com',
        description='Google OAuth 2.0 클라이언트 ID'
    ),
    
    'google_oauth_client_secret': SecretDefinition(
        name='google_oauth_client_secret',
        display_name='Google OAuth Client Secret',
        env_var='GOOGLE_OAUTH_CLIENT_SECRET',
        secrets_key='google_oauth_client_secret',
        required=False,
        min_length=20,
        example='GOCSPX-...',
        description='Google OAuth 2.0 클라이언트 시크릿'
    ),
    
    'google_service_account': SecretDefinition(
        name='google_service_account',
        display_name='Google Service Account JSON',
        env_var='GOOGLE_SERVICE_ACCOUNT_JSON',
        secrets_key='google_service_account',
        required=False,
        is_json=True,
        min_length=100,
        description='Google 서비스 계정 JSON (고급 사용자용)'
    ),
}

# ============================================================================
# 🔍 검증 및 우선순위 설정
# ============================================================================

# 암호 입력 우선순위
SECRET_PRIORITY = [
    'session_state',     # 1순위: 앱 내 입력 (가장 안전)
    'local_storage',     # 2순위: 로컬 암호화 저장
    'streamlit_secrets', # 3순위: Streamlit Secrets
    'environment',       # 4순위: 환경 변수
    'default'           # 5순위: 기본값 (무료 API)
]

# 검증 규칙
VALIDATION_RULES = {
    'api_key_min_length': 10,
    'api_key_max_length': 500,
    'allow_empty_optional': True,
    'validate_on_save': True,
    'sanitize_input': True,
}

# 필수 서비스 목록
REQUIRED_SERVICES = [
    'google_gemini',  # 필수 AI (무료)
]

# 권장 서비스 목록
RECOMMENDED_SERVICES = [
    'groq',           # 무료, 빠른 추론
    'huggingface',    # 무료, 도메인 특화
    'openalex',       # 문헌 검색
    'github',         # 코드/데이터 리포
]

# ============================================================================
# 🛡️ 보안 클래스 및 함수
# ============================================================================

class SecretValidator:
    """비밀 정보 검증기"""
    
    @staticmethod
    def validate_api_key(service: str, key: str) -> tuple[bool, Optional[str]]:
        """API 키 검증"""
        # 전체 정의 통합
        all_definitions = {
            **AI_SECRET_DEFINITIONS,
            **LITERATURE_SECRET_DEFINITIONS,
            **GOOGLE_SECRET_DEFINITIONS
        }
        
        definition = all_definitions.get(service)
        if not definition:
            return False, f"알 수 없는 서비스: {service}"
            
        if not key:
            if definition.required:
                return False, f"{definition.display_name}은(는) 필수입니다"
            return True, None
            
        if not definition.validate(key):
            errors = []
            if definition.min_length and len(key) < definition.min_length:
                errors.append(f"최소 {definition.min_length}자 이상이어야 합니다")
            if definition.validation_prefix and not key.startswith(definition.validation_prefix):
                errors.append(f"'{definition.validation_prefix}'로 시작해야 합니다")
            if definition.validation_pattern and not re.match(definition.validation_pattern, key):
                errors.append("형식이 올바르지 않습니다")
            return False, " / ".join(errors) if errors else "유효하지 않은 형식입니다"
            
        return True, None
    
    @staticmethod
    def mask_secret(value: str, visible_chars: int = 4) -> str:
        """비밀 정보 마스킹"""
        if not value or len(value) <= visible_chars * 2:
            return "*" * len(value) if value else ""
            
        return value[:visible_chars] + "*" * (len(value) - visible_chars * 2) + value[-visible_chars:]
    
    @staticmethod
    def get_secret_strength(value: str) -> tuple[int, str]:
        """비밀 정보 강도 평가 (0-100)"""
        if not value:
            return 0, "없음"
            
        score = 0
        factors = []
        
        # 길이 평가
        if len(value) >= 40:
            score += 40
            factors.append("매우 긴 길이")
        elif len(value) >= 20:
            score += 30
            factors.append("충분한 길이")
        elif len(value) >= 12:
            score += 20
            factors.append("적절한 길이")
        else:
            score += 10
            factors.append("짧은 길이")
            
        # 문자 다양성 평가
        has_upper = bool(re.search(r'[A-Z]', value))
        has_lower = bool(re.search(r'[a-z]', value))
        has_digit = bool(re.search(r'\d', value))
        has_special = bool(re.search(r'[^A-Za-z0-9]', value))
        
        diversity = sum([has_upper, has_lower, has_digit, has_special])
        score += diversity * 15
        
        if diversity == 4:
            factors.append("매우 다양한 문자")
        elif diversity >= 3:
            factors.append("다양한 문자")
        else:
            factors.append("단순한 문자")
            
        # 패턴 평가
        if not re.search(r'(.)\1{2,}', value):  # 반복 문자 없음
            score += 10
            factors.append("반복 없음")
            
        # 일반적인 패턴 확인
        common_patterns = ['123', 'abc', 'password', 'secret', 'key', 'test']
        if not any(pattern in value.lower() for pattern in common_patterns):
            score += 10
            factors.append("일반 패턴 없음")
            
        # 강도 레벨
        if score >= 80:
            strength = "매우 강함"
        elif score >= 60:
            strength = "강함"
        elif score >= 40:
            strength = "보통"
        else:
            strength = "약함"
            
        return score, f"{strength} ({', '.join(factors)})"

# ============================================================================
# 💬 보안 메시지 템플릿
# ============================================================================

SECURITY_MESSAGES = {
    # 정보 메시지
    'info': {
        'api_key_required': "🔑 이 기능을 사용하려면 API 키가 필요합니다.",
        'api_key_optional': "🔑 API 키를 설정하면 더 많은 기능을 사용할 수 있습니다.",
        'setup_guide': "⚙️ 설정 가이드를 참고하여 API 키를 발급받으세요.",
        'security_reminder': "🔒 API 키는 비밀번호와 같습니다. 안전하게 보관하세요.",
        'free_tier_available': "🎁 무료 티어가 제공되는 서비스입니다.",
        'local_storage': "💾 모든 키는 로컬에 암호화되어 저장됩니다.",
    },
    
    # 성공 메시지
    'success': {
        'api_key_saved': "✅ API 키가 안전하게 저장되었습니다.",
        'api_key_validated': "✅ API 키가 유효합니다.",
        'connection_established': "✅ 서비스 연결에 성공했습니다.",
        'settings_updated': "✅ 보안 설정이 업데이트되었습니다.",
        'import_complete': "✅ 설정을 성공적으로 가져왔습니다.",
        'export_complete': "✅ 설정을 성공적으로 내보냈습니다.",
    },
    
    # 경고 메시지
    'warning': {
        'api_key_expiring': "⚠️ API 키가 곧 만료됩니다. 갱신이 필요합니다.",
        'weak_api_key': "⚠️ 보안이 약한 API 키입니다. 재발급을 권장합니다.",
        'public_exposure': "⚠️ API 키가 공개될 수 있는 환경입니다.",
        'insecure_connection': "⚠️ 보안되지 않은 연결입니다.",
        'optional_key_missing': "⚠️ 선택적 API 키가 설정되지 않았습니다. 일부 기능이 제한됩니다.",
        'rate_limit_warning': "⚠️ API 호출 한도에 근접했습니다.",
    },
    
    # 오류 메시지
    'error': {
        'api_key_invalid': "❌ 유효하지 않은 API 키입니다.",
        'api_key_not_found': "❌ API 키를 찾을 수 없습니다.",
        'authentication_failed': "❌ 인증에 실패했습니다.",
        'permission_denied': "❌ 권한이 없습니다.",
        'service_unavailable': "❌ 서비스를 사용할 수 없습니다.",
        'storage_error': "❌ 키 저장 중 오류가 발생했습니다.",
    }
}

# ============================================================================
# 📝 secrets.toml 템플릿 생성
# ============================================================================

def generate_secrets_template() -> str:
    """secrets.toml 템플릿 생성"""
    template_lines = [
        "# Universal DOE Platform - Secrets Configuration",
        "# " + "=" * 60,
        "# 이 파일을 .streamlit/secrets.toml로 저장하세요",
        "# 주의: 이 파일을 절대 Git에 커밋하지 마세요!",
        "# " + "=" * 60,
        "",
        "# ========== 필수 API 키 ==========",
        ""
    ]
    
    # 필수 AI API 키
    for service, definition in AI_SECRET_DEFINITIONS.items():
        if definition.required:
            template_lines.extend([
                f"# {definition.display_name}",
                f"# 발급: {definition.docs_url}",
                f"# 설명: {definition.description}",
                f"{definition.secrets_key} = \"{definition.example or 'YOUR_API_KEY_HERE'}\"",
                ""
            ])
    
    template_lines.extend([
        "# ========== 권장 API 키 (무료) ==========",
        ""
    ])
    
    # 권장 무료 API
    for service in RECOMMENDED_SERVICES:
        definition = {**AI_SECRET_DEFINITIONS, **LITERATURE_SECRET_DEFINITIONS}.get(service)
        if definition and not definition.required:
            template_lines.extend([
                f"# {definition.display_name}",
                f"# 발급: {definition.docs_url}",
                f"# 설명: {definition.description}",
                f"# {definition.secrets_key} = \"{definition.example or 'YOUR_API_KEY_HERE'}\"",
                ""
            ])
    
    template_lines.extend([
        "# ========== 선택적 API 키 ==========",
        ""
    ])
    
    # 나머지 선택적 API
    all_optional = {
        **{k: v for k, v in AI_SECRET_DEFINITIONS.items() if not v.required and k not in RECOMMENDED_SERVICES},
        **{k: v for k, v in LITERATURE_SECRET_DEFINITIONS.items() if k not in RECOMMENDED_SERVICES},
        **GOOGLE_SECRET_DEFINITIONS
    }
    
    for service, definition in all_optional.items():
        template_lines.extend([
            f"# {definition.display_name}",
            f"# {definition.secrets_key} = \"{definition.example or 'YOUR_VALUE_HERE'}\"",
            ""
        ])
    
    template_lines.extend([
        "# ========== Google 서비스 계정 (고급) ==========",
        "# Google 서비스 계정을 사용하는 경우에만 필요합니다",
        "",
        "# [google_service_account]",
        "# type = \"service_account\"",
        "# project_id = \"your-project-id\"",
        "# private_key_id = \"key-id\"",
        "# private_key = \"-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n\"",
        "# client_email = \"your-service-account@project.iam.gserviceaccount.com\"",
        "# client_id = \"000000000000000000000\"",
        "# auth_uri = \"https://accounts.google.com/o/oauth2/auth\"",
        "# token_uri = \"https://oauth2.googleapis.com/token\"",
        "# auth_provider_x509_cert_url = \"https://www.googleapis.com/oauth2/v1/certs\"",
        "# client_x509_cert_url = \"https://www.googleapis.com/robot/v1/metadata/x509/...\"",
        ""
    ])
    
    return "\n".join(template_lines)

# ============================================================================
# 🔧 환경별 기본값
# ============================================================================

ENVIRONMENT_DEFAULTS = {
    'development': {
        'allow_empty_optional': True,
        'show_api_key_warnings': True,
        'validate_strictly': False,
        'use_mock_services': True,
        'cache_duration': 3600,  # 1시간
    },
    
    'staging': {
        'allow_empty_optional': True,
        'show_api_key_warnings': True,
        'validate_strictly': True,
        'use_mock_services': False,
        'cache_duration': 1800,  # 30분
    },
    
    'production': {
        'allow_empty_optional': False,
        'show_api_key_warnings': False,
        'validate_strictly': True,
        'use_mock_services': False,
        'cache_duration': 600,  # 10분
    }
}

# ============================================================================
# 🎯 보안 권장사항
# ============================================================================

SECURITY_RECOMMENDATIONS = {
    'storage': [
        "API 키를 코드에 직접 입력하지 마세요",
        "공개 저장소에 secrets.toml 파일을 업로드하지 마세요",
        ".gitignore에 비밀 정보 파일을 추가하세요",
        "로컬 암호화 저장소를 사용하세요",
        "정기적으로 백업하되 안전하게 보관하세요",
    ],
    
    'rotation': [
        "API 키를 정기적으로 재발급하세요 (3-6개월마다)",
        "의심스러운 활동이 있으면 즉시 키를 변경하세요",
        "사용하지 않는 API 키는 폐기하세요",
        "여러 프로젝트에 같은 키를 사용하지 마세요",
        "만료일을 캘린더에 기록하세요",
    ],
    
    'access': [
        "최소 권한 원칙을 따르세요",
        "읽기 전용 키와 읽기/쓰기 키를 구분하세요",
        "IP 화이트리스트를 설정하세요 (가능한 경우)",
        "API 사용량을 모니터링하세요",
        "이상 활동 알림을 설정하세요",
    ],
    
    'development': [
        "개발용과 프로덕션용 키를 분리하세요",
        "테스트에는 mock 데이터를 사용하세요",
        "CI/CD에서는 암호화된 환경 변수를 사용하세요",
        "코드 리뷰 시 비밀 정보 노출을 확인하세요",
        "커밋 전 자동 스캔 도구를 사용하세요",
    ]
}

# ============================================================================
# 🔐 통합 관리 클래스
# ============================================================================

class SecretsConfig:
    """암호 설정 통합 관리"""
    
    def __init__(self, environment: str = 'development'):
        self.environment = environment
        self.defaults = ENVIRONMENT_DEFAULTS.get(environment, ENVIRONMENT_DEFAULTS['development'])
        self.validator = SecretValidator()
        
        # 모든 정의 통합
        self.all_definitions = {
            **AI_SECRET_DEFINITIONS,
            **LITERATURE_SECRET_DEFINITIONS,
            **GOOGLE_SECRET_DEFINITIONS
        }
    
    def get_all_definitions(self) -> Dict[str, SecretDefinition]:
        """모든 비밀 정보 정의 반환"""
        return self.all_definitions
    
    def get_required_secrets(self) -> List[SecretDefinition]:
        """필수 비밀 정보 목록"""
        return [
            definition for definition in self.all_definitions.values()
            if definition.required
        ]
    
    def get_optional_secrets(self) -> List[SecretDefinition]:
        """선택적 비밀 정보 목록"""
        return [
            definition for definition in self.all_definitions.values()
            if not definition.required
        ]
    
    def get_missing_required(self, available_secrets: Dict[str, Any]) -> List[str]:
        """누락된 필수 비밀 정보 확인"""
        missing = []
        for definition in self.get_required_secrets():
            if definition.name not in available_secrets or not available_secrets[definition.name]:
                missing.append(definition.display_name)
        return missing
    
    def validate_all(self, secrets: Dict[str, Any]) -> Dict[str, tuple[bool, Optional[str]]]:
        """모든 비밀 정보 검증"""
        results = {}
        
        for name, value in secrets.items():
            is_valid, error_msg = self.validator.validate_api_key(name, value)
            results[name] = (is_valid, error_msg)
            
        return results
    
    def get_service_by_category(self) -> Dict[str, List[str]]:
        """카테고리별 서비스 분류"""
        return {
            'ai': list(AI_SECRET_DEFINITIONS.keys()),
            'literature': list(LITERATURE_SECRET_DEFINITIONS.keys()),
            'google': list(GOOGLE_SECRET_DEFINITIONS.keys()),
        }
    
    def export_template(self, include_optional: bool = True) -> str:
        """설정 템플릿 내보내기"""
        return generate_secrets_template()

# ============================================================================
# 🔑 유틸리티 함수
# ============================================================================

def validate_api_key(service: str, key: str) -> bool:
    """API 키 간단 검증 (호환성용)"""
    validator = SecretValidator()
    is_valid, _ = validator.validate_api_key(service, key)
    return is_valid

def get_secret_definition(service: str) -> Optional[SecretDefinition]:
    """서비스별 비밀 정보 정의 반환"""
    all_definitions = {
        **AI_SECRET_DEFINITIONS,
        **LITERATURE_SECRET_DEFINITIONS,
        **GOOGLE_SECRET_DEFINITIONS
    }
    return all_definitions.get(service)

def get_required_services() -> List[str]:
    """필수 서비스 목록 반환"""
    return REQUIRED_SERVICES

def get_recommended_services() -> List[str]:
    """권장 서비스 목록 반환"""
    return RECOMMENDED_SERVICES

# ============================================================================
# 📤 Export
# ============================================================================

__all__ = [
    # 클래스
    'SecretDefinition',
    'SecretValidator',
    'SecretsConfig',
    
    # 정의 딕셔너리
    'AI_SECRET_DEFINITIONS',
    'LITERATURE_SECRET_DEFINITIONS',
    'GOOGLE_SECRET_DEFINITIONS',
    
    # 설정
    'SECRET_PRIORITY',
    'VALIDATION_RULES',
    'REQUIRED_SERVICES',
    'RECOMMENDED_SERVICES',
    'SECURITY_MESSAGES',
    'SECURITY_RECOMMENDATIONS',
    'ENVIRONMENT_DEFAULTS',
    
    # 함수
    'validate_api_key',
    'get_secret_definition',
    'get_required_services',
    'get_recommended_services',
    'generate_secrets_template',
]
