from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import pandas as pd


def extract_candidate_list(
    payload: Any,
) -> tuple[list[dict[str, Any]], str | None]:
    # 단일 객체와 여러 래퍼 형식을 동일한 후보지 배열로 처리한다.
    if isinstance(payload, list):
        return payload, None

    if isinstance(payload, dict):
        for key in ("results", "data", "candidates"):
            value = payload.get(key)
            if isinstance(value, list):
                return value, key

        if "1_site_info" in payload:
            return [payload], "__single__"

    raise ValueError(
        "후보지 배열, results/data/candidates 배열 또는 "
        "단일 후보지 객체를 입력하세요."
    )


def rebuild_payload(
    original: Any,
    results: list[dict[str, Any]],
    wrapper: str | None,
    *,
    agent_version: str,
    llm_enabled: bool,
    llm_model: str | None,
) -> Any:
    # 분석 후에도 입력 JSON의 원래 구조를 유지한다.
    if wrapper is None:
        return results

    if wrapper == "__single__":
        return results[0]

    output = copy.deepcopy(original)
    output[wrapper] = results

    if isinstance(output.get("summary"), dict):
        output["summary"]["policy_agent_processed"] = len(
            results
        )
        output["summary"]["policy_agent_version"] = (
            agent_version
        )
        output["summary"]["policy_agent_llm_enabled"] = (
            llm_enabled
        )
        output["summary"]["policy_agent_llm_model"] = (
            llm_model
        )

    return output


def validate_candidate(candidate: Any) -> None:
    if not isinstance(candidate, dict):
        raise ValueError("후보지 JSON은 객체여야 합니다.")

    site = candidate.get("1_site_info")
    if not isinstance(site, dict):
        raise ValueError(
            "후보지 JSON에 1_site_info 객체가 필요합니다."
        )

    site_id = str(site.get("site_id") or "").strip()
    if not site_id:
        raise ValueError(
            "1_site_info.site_id 값이 필요합니다."
        )

    address = str(site.get("address") or "").strip()
    if not address:
        raise ValueError(
            "1_site_info.address 값이 필요합니다."
        )


def to_json_safe(value: Any) -> Any:
    # NumPy·pandas 값과 NaN/Inf를 JSON 직렬화 가능한 값으로 변환한다.
    if isinstance(value, dict):
        return {
            str(key): to_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            to_json_safe(item)
            for item in value
        ]

    if isinstance(value, np.generic):
        return to_json_safe(value.item())

    if isinstance(value, float) and not math.isfinite(value):
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return value
