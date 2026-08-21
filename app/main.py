from __future__ import annotations

import secrets
from contextlib import asynccontextmanager
from typing import Any

from fastapi import (
    Body,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    status,
)
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.services.agent_service import AgentService


settings = get_settings()
agent_service = AgentService(settings)


# 서버 시작 시 정책 데이터와 추천 엔진을 초기화한다.
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.agent_service = agent_service
        agent_service.initialize()
    except Exception as exc:
        app.state.startup_error = (
            f"{type(exc).__name__}: {exc}"
        )
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Spring Boot 백엔드에서 호출하는 "
        "태양광 정책·자금지원 추천 AI Agent API"
    ),
    lifespan=lifespan,
)


def require_internal_api_key(
    x_internal_api_key: str | None = Header(
        default=None,
        alias="X-Internal-API-Key",
    ),
) -> None:
    # 운영 환경에 내부 API 키가 설정된 경우에만 요청을 검증한다.
    expected = settings.internal_api_key.strip()
    if not expected:
        return

    provided = (x_internal_api_key or "").strip()
    if not secrets.compare_digest(
        provided,
        expected,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않은 내부 API 키입니다.",
        )


def get_service(request: Request) -> AgentService:
    service = getattr(
        request.app.state,
        "agent_service",
        None,
    )
    startup_error = getattr(
        request.app.state,
        "startup_error",
        None,
    )

    # 초기화에 실패한 상태로 분석 요청이 실행되는 것을 막는다.
    if service is None or startup_error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "message": "Agent 서비스 초기화 실패",
                "startup_error": startup_error,
            },
        )
    return service


@app.exception_handler(ValueError)
async def value_error_handler(
    request: Request,
    exc: ValueError,
):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": {
                "type": "VALIDATION_ERROR",
                "message": str(exc),
            },
        },
    )


@app.get("/health", tags=["system"])
def health(
    service: AgentService = Depends(get_service),
) -> dict[str, Any]:
    return service.health()


@app.post(
    "/api/v1/agent/analyze",
    tags=["agent"],
    dependencies=[
        Depends(require_internal_api_key)
    ],
)
def analyze(
    payload: Any = Body(...),
    service: AgentService = Depends(get_service),
) -> dict[str, Any]:
    analyzed = service.analyze_batch(payload)

    return {
        "success": True,
        **analyzed,
        "meta": service.health()["runtime"],
    }


@app.post(
    "/api/v1/agent/analyze-batch",
    tags=["agent"],
    dependencies=[
        Depends(require_internal_api_key)
    ],
)
def analyze_batch(
    payload: Any = Body(...),
    service: AgentService = Depends(get_service),
) -> dict[str, Any]:
    analyzed = service.analyze_batch(payload)
    return {
        "success": True,
        **analyzed,
        "meta": service.health()["runtime"],
    }


@app.post(
    "/api/v1/admin/reload-data",
    tags=["admin"],
    dependencies=[
        Depends(require_internal_api_key)
    ],
)
def reload_data(
    service: AgentService = Depends(get_service),
) -> dict[str, Any]:
    return service.reload_data()
