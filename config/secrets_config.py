"""암호 설정 관리"""
from typing import Dict, Optional, List
import os

# API 키 설정 구조
API_KEY_STRUCTURE = {
    'google_gemini': {
        'env_var': 'GOOGLE_GEMINI_API_KEY',
        'secrets_key': 'google_gemini_key',
        'validation_prefix': 'AIza',
        'required': True
    },
    'xai_grok': {
        'env_var': 'XAI_GROK_API_KEY',
        'secrets_key': 'xai_grok_key',
        'validation_prefix': 'xai-',
        'required': False
    },
    'groq': {
        'env_var': 'GROQ_API_KEY',
        'secrets_key': 'groq_key',
        'validation_prefix': 'gsk_',
        'required': False
    },
    'deepseek': {
        'env_var': 'DEEPSEEK_API_KEY',
        'secrets_key': 'deepseek_key',
        'validation_prefix': 'sk-',
        'required': False
    },
    'sambanova': {
        'env_var': 'SAMBANOVA_API_KEY',
        'secrets_key': 'sambanova_key',
        'validation_prefix': None,
        'required': False
    },
    'huggingface': {
        'env_var': 'HUGGINGFACE_TOKEN',
        'secrets_key': 'huggingface_token',
        'validation_prefix': 'hf_',
        'required': False
    }
}

# Google 관련 설정
GOOGLE_CONFIG = {
    'sheets_url': {
        'env_var': 'GOOGLE_SHEETS_URL',
        'secrets_key': 'google_sheets_url',
        'required': True,
        'validation': lambda x: x.startswith('https://docs.google.com/spreadsheets/')
    },
    'service_account': {
        'env_var': 'GOOGLE_SERVICE_ACCOUNT_JSON',
        'secrets_key': 'google_service_account',
        'required': False,  # 개인 계정 사용 시 불필요
        'is_json': True
    },
    'oauth_client': {
        'env_var': 'GOOGLE_OAUTH_CLIENT_JSON',
        'secrets_key': 'google_oauth_client',
        'required': False,
        'is_json': True
    }
}

# 암호 입력 우선순위
SECRET_PRIORITY = [
    'session_state',     # 1순위: 앱 내 입력
    'streamlit_secrets', # 2순위: Streamlit Secrets
    'environment',       # 3순위: 환경 변수
    'default'           # 4순위: 기본값 (무료 API)
]

# 암호 검증 규칙
VALIDATION_RULES = {
    'api_key_min_length': 20,
    'api_key_max_length': 200,
    'google_sheets_url_pattern': r'https://docs\.google\.com/spreadsheets/d/[\w-]+',
    'allow_empty_optional': True
}

# 보안 메시지
SECURITY_MESSAGES = {
    'api_key_not_found': "🔑 API 키가 설정되지 않았습니다.",
    'api_key_invalid': "❌ 유효하지 않은 API 키입니다.",
    'api_key_saved': "✅ API 키가 안전하게 저장되었습니다.",
    'google_auth_required': "🔐 Google 인증이 필요합니다.",
    'security_warning': "⚠️ API 키는 안전하게 보관하세요."
}

# 예시 secrets.toml 내용
SECRETS_TOML_TEMPLATE = """
# Streamlit Secrets 설정 파일
# 이 파일은 .gitignore에 추가되어야 합니다!

# 필수 API 키
google_gemini_key = "YOUR_GEMINI_API_KEY_HERE"

# 선택적 API 키 (사용하려는 것만 입력)
# xai_grok_key = "YOUR_XAI_API_KEY_HERE"
# groq_key = "YOUR_GROQ_API_KEY_HERE"
# deepseek_key = "YOUR_DEEPSEEK_API_KEY_HERE"
# sambanova_key = "YOUR_SAMBANOVA_API_KEY_HERE"
# huggingface_token = "YOUR_HF_TOKEN_HERE"

# Google Sheets URL (필수)
google_sheets_url = "https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/edit"

# Google 서비스 계정 (선택사항)
# [google_service_account]
# type = "service_account"
# project_id = "your-project-id"
# private_key_id = "your-private-key-id"
# private_key = "-----BEGIN PRIVATE KEY-----\\nYOUR_PRIVATE_KEY\\n-----END PRIVATE KEY-----\\n"
# client_email = "your-service-account@your-project.iam.gserviceaccount.com"
# client_id = "your-client-id"
# auth_uri = "https://accounts.google.com/o/oauth2/auth"
# token_uri = "https://oauth2.googleapis.com/token"
# auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
# client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/your-service-account%40your-project.iam.gserviceaccount.com"
"""

def get_secrets_template() -> str:
    """secrets.toml 템플릿 반환"""
    return SECRETS_TOML_TEMPLATE

def validate_api_key(service: str, key: str) -> bool:
    """API 키 유효성 검증"""
    if not key:
        return False
        
    # 길이 검증
    if len(key) < VALIDATION_RULES['api_key_min_length'] or \
       len(key) > VALIDATION_RULES['api_key_max_length']:
        return False
    
    # 서비스별 prefix 검증
    if service in API_KEY_STRUCTURE:
        prefix = API_KEY_STRUCTURE[service].get('validation_prefix')
        if prefix and not key.startswith(prefix):
            return False
    
    return True
