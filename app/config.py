from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent


POLICY_JSON_NAME = "태양광_정책통합_2026.json"


def _resolve_default_data_dir() -> Path:
    """통합 정책 JSON이 들어 있는 data 폴더를 찾습니다."""
    candidates = [
        PROJECT_DIR / "data",
        PROJECT_DIR / PROJECT_DIR.name / "data",
        Path.cwd() / "data",
        Path.cwd() / "solar-agent-api" / "data",
    ]

    for candidate in candidates:
        if candidate.is_dir() and (candidate / POLICY_JSON_NAME).is_file():
            return candidate.resolve()

    return (PROJECT_DIR / "data").resolve()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Solar Policy Recommendation AI Agent API"
    app_version: str = "2.0.0"
    host: str = "0.0.0.0"
    port: int = 8003
    log_level: str = "info"

    use_llm: bool = True
    # 실제 키는 반드시 .env의 OPENAI_API_KEY로만 주입합니다.
    openai_api_key: str = ""
    openai_model: str = "gpt-5-mini"
    openai_timeout_seconds: int = 90
    openai_max_retries: int = 2
    llm_failure_mode: str = "FALLBACK"

    max_batch_size: int = 100
    internal_api_key: str = ""

    data_dir: Path = Field(default_factory=_resolve_default_data_dir)
    policy_json_name: str = POLICY_JSON_NAME

    @property
    def policy_json_path(self) -> Path:
        return self.data_dir / self.policy_json_name


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()

    print("ENV PATH:", PROJECT_DIR / ".env")
    print("ENV EXISTS:", (PROJECT_DIR / ".env").exists())
    print("USE_LLM:", settings.use_llm)
    print("OPENAI_API_KEY EXISTS:", bool(settings.openai_api_key))
    print("OPENAI_MODEL:", settings.openai_model)
    print("POLICY JSON:", settings.policy_json_path)

    return settings
