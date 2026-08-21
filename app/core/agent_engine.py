from __future__ import annotations

import copy
import json
import math
import os
import re
from datetime import datetime, timezone
from itertools import combinations
from typing import Any, TypedDict

import pandas as pd
from pydantic import BaseModel, Field

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from langgraph.graph import END, START, StateGraph

    LANGGRAPH_AVAILABLE = True
except ImportError:
    END = None
    START = None
    StateGraph = None
    LANGGRAPH_AVAILABLE = False


# LangGraph가 없어도 동일한 순서의 순차 실행기로 동작한다.
AGENT_VERSION = "solar-policy-agent-json-v2"
TOP_K = 1

USE_LLM = False
OPENAI_MODEL = "gpt-5-mini"
LLM_FAILURE_MODE = "FALLBACK"
POLICY_JSON_NAME = "태양광_정책통합_2026.json"
INCLUDE_POLICY_DETAILS = False

policy_df = pd.DataFrame()
relation_df = pd.DataFrame()
funding_df = pd.DataFrame()

BASE_LLM = None
EXPLANATION_LLM = None

_runtime_status: dict[str, Any] = {
    "initialized": False,
    "llm_requested": False,
    "llm_enabled": False,
    "llm_reason": "NOT_INITIALIZED",
    "graph_engine": (
        "langgraph" if LANGGRAPH_AVAILABLE else "sequential-fallback"
    ),
}

CHUNGNAM_CITIES = {
    "천안",
    "공주",
    "보령",
    "아산",
    "서산",
    "논산",
    "계룡",
    "당진",
    "금산",
    "부여",
    "서천",
    "청양",
    "홍성",
    "예산",
    "태안",
}
CHUNGBUK_CITIES = {
    "청주",
    "충주",
    "제천",
    "보은",
    "옥천",
    "영동",
    "증평",
    "진천",
    "괴산",
    "음성",
    "단양",
}

NATIONAL_FINANCE_PROGRAM_IDS = {
    "NAT_RE_FIN_PUBLIC_SOLAR_2026",
    "NAT_RE_FIN_PUBLIC_PARKING_2026",
    "NAT_RE_FIN_SUN_INCOME_2026",
    "NAT_RE_FIN_INDUSTRIAL_2026",
    "NAT_RE_FIN_RURAL_2026",
    "NAT_RE_FIN_PUBLIC_LED_2026",
}

ROUTE_PROGRAM_ORDER = {
    "SUN_INCOME_VILLAGE": [
        "NAT_SUN_INCOME_SELECTION_2026",
        "NAT_RE_FIN_SUN_INCOME_2026",
        "CN_VILLAGE_PROFIT_SOLAR_2026",
        "CB_BOEUN_SUN_INCOME_PLAN_2026",
    ],
    "PUBLIC_LED_SOLAR": [
        "NAT_RE_FIN_PUBLIC_LED_2026",
        "NAT_RE_FIN_PUBLIC_SOLAR_2026",
        "CN_INDUSTRIAL_COMMON_SOLAR_ESS_2026",
        "PUBLIC_LAND_LEASE_POLICY",
    ],
    "SELF_CONSUMPTION_SOLAR": [
        "NAT_RE_FIN_INDUSTRIAL_2026",
        "CN_INDUSTRIAL_SOLAR_2026",
        "CN_INDUSTRIAL_ESS_2026",
        "CN_CRISIS_SOLAR_2026",
        "NAT_RE_FIN_PUBLIC_SOLAR_2026",
    ],
    "RURAL_SOLAR": [
        "NAT_RE_FIN_RURAL_2026",
        "NAT_RE_FIN_PUBLIC_SOLAR_2026",
        "PUBLIC_LAND_LEASE_POLICY",
    ],
    "PUBLIC_ROOFTOP": [
        "NAT_RE_FIN_PUBLIC_SOLAR_2026",
        "NAT_RE_FIN_PUBLIC_LED_2026",
        "PUBLIC_LAND_LEASE_POLICY",
    ],
    "CONSENT_BASED_LAND": [
        "NAT_RE_FIN_PUBLIC_SOLAR_2026",
        "NAT_RE_FIN_PUBLIC_LED_2026",
        "PUBLIC_LAND_LEASE_POLICY",
    ],
    "PUBLIC_PARKING": [
        "NAT_RE_FIN_PUBLIC_PARKING_2026",
        "NAT_RE_FIN_PUBLIC_LED_2026",
        "PUBLIC_LAND_LEASE_POLICY",
    ],
    "PUBLIC_LAND": [
        "NAT_RE_FIN_PUBLIC_SOLAR_2026",
        "NAT_RE_FIN_PUBLIC_LED_2026",
        "PUBLIC_LAND_LEASE_POLICY",
    ],
}


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def clean_text(value: Any) -> str:
    if is_missing(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def optional_float(value: Any) -> float | None:
    if is_missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalize_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if is_missing(value):
        return None
    text = clean_text(value).lower()
    if text in {"true", "1", "yes", "y", "예", "확인"}:
        return True
    if text in {"false", "0", "no", "n", "아니오", "미확인"}:
        return False
    return None


def normalize_city(value: Any) -> str:
    return re.sub(
        r"(특별자치시|광역시|시|군|구)$",
        "",
        clean_text(value),
    )


def infer_region_and_jurisdiction(address: str) -> tuple[str, str]:
    text = clean_text(address)

    province = ""
    for canonical, pattern in [
        ("충청남도", r"(충청남도|충남)"),
        ("충청북도", r"(충청북도|충북)"),
        ("대전광역시", r"(대전광역시|대전)"),
        ("세종특별자치시", r"(세종특별자치시|세종)"),
    ]:
        if re.search(pattern, text):
            province = canonical
            break

    jurisdiction = ""
    candidates = re.findall(
        r"([가-힣]+(?:특별자치시|광역시|시|군|구))",
        text,
    )
    for candidate in candidates:
        if candidate in {"충청남도", "충청북도"}:
            continue
        if candidate.endswith(("시", "군")):
            jurisdiction = candidate
            break

    city_base = normalize_city(jurisdiction)
    if not province and city_base:
        if city_base in CHUNGNAM_CITIES:
            province = "충청남도"
        elif city_base in CHUNGBUK_CITIES:
            province = "충청북도"

    return province, jurisdiction


def candidate_type_settings(candidate_type: str) -> tuple[str, str]:
    normalized = clean_text(candidate_type).lower()
    if normalized == "building":
        return "BUILDING", "ROOFTOP"
    if normalized == "parking_lot":
        return "PARKING", "PARKING_CANOPY"
    return "LAND", "GROUND"


def first_not_missing(*values: Any) -> Any:
    for value in values:
        if not is_missing(value):
            return value
    return None


def normalize_upstream_decision(value: Any) -> str:
    text = clean_text(value).upper()
    mapping = {
        "PASS": "PASS",
        "PASSED": "PASS",
        "통과": "PASS",
        "PASS_EXCEPTION": "PASS_EXCEPTION",
        "CONDITIONAL_PASS": "PASS_EXCEPTION",
        "예외통과": "PASS_EXCEPTION",
        "FAIL": "FAIL",
        "FAILED": "FAIL",
        "REJECT": "FAIL",
        "REJECTED": "FAIL",
        "BLOCKED": "FAIL",
        "RULE_FAIL": "FAIL",
        "부적합": "FAIL",
        "탈락": "FAIL",
        "미통과": "FAIL",
        "REVIEW": "REVIEW",
        "LEGAL_REVIEW": "REVIEW",
        "NEEDS_REVIEW": "REVIEW",
        "CONDITIONAL": "REVIEW",
        "검토": "REVIEW",
        "조건부": "REVIEW",
        "UNKNOWN": "UNKNOWN",
        "NO_APPLICABLE_RULE": "UNKNOWN",
    }
    return mapping.get(text, "UNKNOWN")


def infer_setback_violation(
    explicit_value: Any,
    decision: str,
    message: str,
) -> bool:
    explicit = normalize_bool(explicit_value)
    if explicit is not None:
        return explicit

    normalized_message = clean_text(message)
    negative_phrases = {
        "이격거리 위반 없음",
        "이격거리 이상 없음",
        "활성 rule 위반 조건 없음",
        "위반 조건 없음",
    }
    if any(phrase in normalized_message.lower() for phrase in negative_phrases):
        return False

    has_setback_keyword = any(
        keyword in normalized_message
        for keyword in ("이격거리", "도로", "주거", "주택")
    )
    return decision in {"FAIL", "PASS_EXCEPTION", "REVIEW"} and has_setback_keyword


def normalize_candidate(ranking_result: dict[str, Any]) -> dict[str, Any]:
    # 입력 JSON에서 정책 추천에 필요한 후보지 정보만 정규화한다.
    site = ranking_result.get("1_site_info") or {}
    scores = ranking_result.get("2_scores_and_evaluation") or {}
    vision_block = ranking_result.get("3_vision_ai_and_simulation") or {}
    vision = vision_block.get("vision_analysis") or {}
    simulation = vision_block.get("simulation") or {}
    risk = ranking_result.get("4_risk_and_support") or {}
    risk_check = risk.get("rule_based_risk_check") or {}
    regulatory_input = risk.get("regulatory_input") or {}
    upstream_assessment = risk.get("regulatory_assessment") or {}

    address = clean_text(site.get("address"))
    province, jurisdiction = infer_region_and_jurisdiction(address)

    candidate_type = clean_text(
        first_not_missing(
            vision.get("candidate_type"),
            site.get("vision_candidate_type"),
            ranking_result.get("target_type"),
        )
    ).lower()
    if candidate_type == "parking_lot":
        candidate_type = "parking_lot"
    elif candidate_type in {"building", "build", "rooftop"}:
        candidate_type = "building"
    else:
        candidate_type = "land"

    asset_type, installation_type = candidate_type_settings(candidate_type)
    distance_risk = risk_check.get("distance_risk") or {}
    suitability = scores.get("suitability") or {}

    upstream_decision_raw = first_not_missing(
        suitability.get("rule_decision"),
        regulatory_input.get("rule_decision"),
        risk_check.get("rule_decision"),
        upstream_assessment.get("final_decision"),
    )
    upstream_message = clean_text(
        first_not_missing(
            suitability.get("rule_message"),
            regulatory_input.get("rule_message"),
            risk_check.get("regulation"),
            upstream_assessment.get("final_reason"),
        )
    )

    upstream_exception_codes = first_not_missing(
        regulatory_input.get("exception_codes"),
        upstream_assessment.get("exception_codes"),
        [],
    )
    if not isinstance(upstream_exception_codes, list):
        upstream_exception_codes = [upstream_exception_codes]

    upstream_data_gaps = first_not_missing(
        regulatory_input.get("data_gaps"),
        upstream_assessment.get("data_gaps"),
        [],
    )
    if not isinstance(upstream_data_gaps, list):
        upstream_data_gaps = [upstream_data_gaps]

    normalized_decision = normalize_upstream_decision(upstream_decision_raw)
    setback_violation = infer_setback_violation(
        first_not_missing(
            regulatory_input.get("setback_violation"),
            upstream_assessment.get("setback_violation"),
        ),
        normalized_decision,
        upstream_message,
    )

    return {
        "site_id": clean_text(site.get("site_id")),
        "site_name": clean_text(site.get("site_name")),
        "address": address,
        "province": province,
        "jurisdiction_norm": jurisdiction,
        "city_base": normalize_city(jurisdiction),
        "candidate_type": candidate_type,
        "asset_type_norm": asset_type,
        "installation_type_norm": installation_type,
        "owner_agency": clean_text(site.get("owner_agency")),
        "available_area_m2": optional_float(
            first_not_missing(
                site.get("available_area_m2"),
                site.get("available_area"),
            )
        ),
        "recommended_capacity_kw": optional_float(
            simulation.get("recommended_capacity_kw")
        ),
        "distance_to_road_m": optional_float(
            first_not_missing(
                regulatory_input.get("distance_to_road_m"),
                vision.get("distance_to_road_m"),
                distance_risk.get("distance_to_road_m"),
            )
        ),
        "distance_to_building_m": optional_float(
            first_not_missing(
                regulatory_input.get("distance_to_building_m"),
                vision.get("distance_to_building_m"),
                distance_risk.get("distance_to_building_m"),
            )
        ),
        "public_project_confirmed": normalize_bool(
            regulatory_input.get("public_project_confirmed")
        ),
        "resident_participation_confirmed": normalize_bool(
            regulatory_input.get("resident_participation_confirmed")
        ),
        "self_consumption_confirmed": normalize_bool(
            regulatory_input.get("self_consumption_confirmed")
        ),
        "all_homeowners_consent": normalize_bool(
            regulatory_input.get("all_homeowners_consent")
        ),
        "parking_function_maintained": normalize_bool(
            regulatory_input.get("parking_function_maintained")
        ),
        "industrial_site_confirmed": normalize_bool(
            regulatory_input.get("industrial_site_confirmed")
        ),
        "agricultural_qualification_confirmed": normalize_bool(
            regulatory_input.get("agricultural_qualification_confirmed")
        ),
        "pipeline_rule_decision_raw": clean_text(upstream_decision_raw),
        "pipeline_rule_decision": normalized_decision,
        "pipeline_rule_message": upstream_message,
        "pipeline_setback_violation": setback_violation,
        "pipeline_exception_codes": [
            clean_text(item)
            for item in upstream_exception_codes
            if clean_text(item)
        ],
        "pipeline_data_gaps": [
            clean_text(item)
            for item in upstream_data_gaps
            if clean_text(item)
        ],
        "pipeline_total_score": optional_float(scores.get("total_score")),
    }


def build_pipeline_regulatory_assessment(
    facts: dict[str, Any],
) -> dict[str, Any]:
    # 전달받은 규제 판정은 변경하지 않고 추천 조건으로 사용한다.
    decision = facts["pipeline_rule_decision"]
    raw_decision = facts["pipeline_rule_decision_raw"]
    message = facts["pipeline_rule_message"]

    if not message:
        message = {
            "PASS": "Ranking 파이프라인에서 Rule 통과로 전달되었습니다.",
            "PASS_EXCEPTION": "Ranking 파이프라인에서 예외 검토 결과가 전달되었습니다.",
            "FAIL": "Ranking 파이프라인에서 Rule 미통과로 전달되었습니다.",
            "REVIEW": "Ranking 파이프라인에서 추가 검토가 필요한 상태로 전달되었습니다.",
            "UNKNOWN": "Ranking 파이프라인의 Rule 판정값이 없거나 해석할 수 없습니다.",
        }[decision]

    return {
        "source": "RANKING_PIPELINE",
        "jurisdiction": facts["jurisdiction_norm"],
        "province": facts["province"],
        "upstream_rule_decision": raw_decision or None,
        "final_decision": decision,
        "final_reason": message,
        "setback_violation": facts["pipeline_setback_violation"],
        "exception_codes": facts["pipeline_exception_codes"],
        "data_gaps": facts["pipeline_data_gaps"],
        "distance_evidence": {
            "distance_to_road_m": facts["distance_to_road_m"],
            "distance_to_building_m": facts["distance_to_building_m"],
        },
        "agent_reassessment_performed": False,
    }


class ProgramReasonItem(BaseModel):
    program_id: str
    reason: str


class TopProgramExplanation(BaseModel):
    program_id: str = ""
    program_name: str = ""
    reason: str = ""
    comparison_with_second: str = ""


class FinalAgentExplanation(BaseModel):
    summary: str
    regulation_reason: str
    business_route_reason: str
    top_program: TopProgramExplanation = Field(
        default_factory=TopProgramExplanation
    )
    program_reasons: list[ProgramReasonItem] = Field(default_factory=list)
    required_checks: list[str] = Field(default_factory=list)
    caution: str


def make_structured_llm(schema: type[BaseModel]):
    if BASE_LLM is None:
        return None
    try:
        return BASE_LLM.with_structured_output(
            schema,
            method="json_schema",
        )
    except TypeError:
        return BASE_LLM.with_structured_output(schema)


def configure_runtime(
    *,
    policies: pd.DataFrame,
    relations: pd.DataFrame,
    funding_conditions: pd.DataFrame,
    use_llm: bool,
    openai_api_key: str,
    openai_model: str,
    openai_timeout_seconds: int,
    openai_max_retries: int,
    llm_failure_mode: str,
    policy_json_name: str,
    include_policy_details: bool,
) -> dict[str, Any]:
    global policy_df
    global relation_df
    global funding_df
    global USE_LLM
    global OPENAI_MODEL
    global LLM_FAILURE_MODE
    global POLICY_JSON_NAME
    global INCLUDE_POLICY_DETAILS
    global BASE_LLM
    global EXPLANATION_LLM
    global _runtime_status

    policy_df = policies.copy()
    relation_df = relations.copy()
    funding_df = funding_conditions.copy()
    POLICY_JSON_NAME = policy_json_name
    INCLUDE_POLICY_DETAILS = bool(include_policy_details)

    OPENAI_MODEL = openai_model
    LLM_FAILURE_MODE = llm_failure_mode.upper()

    requested = bool(use_llm)
    key = str(openai_api_key or "").strip()

    # LLM을 사용할 수 없어도 조건 판단과 지원사업 추천은 계속 수행한다.
    if not requested:
        USE_LLM = False
        llm_reason = "DISABLED_BY_CONFIG"
    elif not key:
        USE_LLM = False
        llm_reason = "OPENAI_API_KEY_MISSING"
    elif ChatOpenAI is None:
        USE_LLM = False
        llm_reason = "LANGCHAIN_OPENAI_NOT_INSTALLED"
    else:
        USE_LLM = True
        llm_reason = "ENABLED"

    if USE_LLM:
        os.environ["OPENAI_API_KEY"] = key
        BASE_LLM = ChatOpenAI(
            model=OPENAI_MODEL,
            timeout=openai_timeout_seconds,
            max_retries=openai_max_retries,
        )
    else:
        BASE_LLM = None

    EXPLANATION_LLM = make_structured_llm(FinalAgentExplanation)

    _runtime_status = {
        "initialized": True,
        "llm_requested": requested,
        "llm_enabled": USE_LLM,
        "llm_reason": llm_reason,
        "llm_model": OPENAI_MODEL if USE_LLM else None,
        "graph_engine": (
            "langgraph" if LANGGRAPH_AVAILABLE else "sequential-fallback"
        ),
        "regulation_source": "RANKING_PIPELINE",
        "data_counts": {
            "policies": len(policy_df),
            "relations": len(relation_df),
            "funding_conditions": len(funding_df),
        },
    }
    return copy.deepcopy(_runtime_status)


def get_runtime_status() -> dict[str, Any]:
    return copy.deepcopy(_runtime_status)


def _conditional_setback_route(
    facts: dict[str, Any],
) -> tuple[str, str]:
    if facts["candidate_type"] == "parking_lot":
        return (
            "PUBLIC_PARKING",
            "이격거리 미통과 결과를 변경하지 않고, 토지형 대신 주차장 상부형 전환 가능성을 조건부 검토합니다.",
        )
    if facts["candidate_type"] == "building":
        return (
            "PUBLIC_ROOFTOP",
            "이격거리 미통과 결과를 변경하지 않고, 토지형 대신 건축물 옥상형 전환 가능성을 조건부 검토합니다.",
        )
    if (
        facts["industrial_site_confirmed"] is True
        and facts["self_consumption_confirmed"] is True
    ):
        return (
            "SELF_CONSUMPTION_SOLAR",
            "이격거리 미통과 결과를 유지하면서 현장 소비시설을 활용한 자가소비형 대안을 조건부 검토합니다.",
        )
    if facts["public_project_confirmed"] is True:
        return (
            "PUBLIC_LED_SOLAR",
            "이격거리 미통과 결과를 유지하면서 공공사업 예외 또는 심의 가능성을 확인하는 조건으로 공공주도형 대안을 검토합니다.",
        )
    return (
        "SUN_INCOME_VILLAGE",
        "이격거리 미통과 결과를 유지하면서 주민참여·마을공동사업 예외조항 존재 여부를 확인하는 조건으로 햇빛소득마을 대안을 검토합니다.",
    )


def select_business_route(
    facts: dict[str, Any],
    assessment: dict[str, Any],
) -> dict[str, Any]:
    # 후보지 유형과 확인된 조건을 기준으로 적용할 사업경로를 선택한다.
    decision = assessment["final_decision"]
    exception_codes = assessment["exception_codes"]

    if decision in {"UNKNOWN", "REVIEW"}:
        return {
            "route_type": "REGULATORY_REVIEW_FIRST",
            "status": "BLOCKED",
            "reason": assessment["final_reason"],
            "required_before_program_selection": (
                assessment["data_gaps"]
                or ["Ranking Rule 판정 근거 및 적용 기준 확인"]
            ),
            "regulatory_decision": decision,
        }

    if decision == "FAIL":
        if not assessment["setback_violation"]:
            return {
                "route_type": "REGULATORY_REVIEW_FIRST",
                "status": "BLOCKED",
                "reason": (
                    "Ranking 파이프라인의 Rule 미통과 결과를 유지합니다. "
                    "이격거리 외 사유이므로 정책 Agent가 대체 경로를 임의로 선정하지 않습니다."
                ),
                "required_before_program_selection": [
                    "Rule 미통과 사유와 보완 가능 여부 확인"
                ],
                "regulatory_decision": decision,
            }

        route, reason = _conditional_setback_route(facts)
        return {
            "route_type": route,
            "status": "CONDITIONAL_SELECTED",
            "reason": reason,
            "required_before_program_selection": [
                "관할 지자체의 예외·완화조항 존재 여부 확인",
                "해당 사업유형의 실제 요건 충족 여부 확인",
            ],
            "regulatory_decision": decision,
            "regulatory_confidence": "LOW",
        }

    if (
        decision == "PASS_EXCEPTION"
        and assessment.get("setback_violation")
    ):
        route = "SUN_INCOME_VILLAGE"
        reason = (
            "Ranking 파이프라인에서 이격거리 관련 예외 검토 결과가 전달되어 "
            "주민참여·마을공동 사업경로를 우선 검토합니다."
        )
    elif "RESIDENT_PARTICIPATION" in exception_codes:
        route = "SUN_INCOME_VILLAGE"
        reason = "주민참여형 예외경로가 전달되어 주민수익형 사업을 우선 검토합니다."
    elif "PUBLIC_PROJECT" in exception_codes:
        route = "PUBLIC_LED_SOLAR"
        reason = "공공사업 예외경로가 전달되어 공공주도형 사업을 우선 검토합니다."
    elif "SELF_CONSUMPTION" in exception_codes:
        route = "SELF_CONSUMPTION_SOLAR"
        reason = "자가소비 예외경로가 전달되어 산업·자가소비형 사업을 우선 검토합니다."
    elif facts["candidate_type"] == "parking_lot":
        route = "PUBLIC_PARKING"
        reason = "후보유형이 주차장이므로 주차장 캐노피형 사업을 검토합니다."
    elif (
        facts["self_consumption_confirmed"] is True
        and facts["industrial_site_confirmed"] is True
    ):
        route = "SELF_CONSUMPTION_SOLAR"
        reason = "산업시설과 자가소비 사실이 확인되어 산업·자가소비형 사업을 검토합니다."
    elif facts["agricultural_qualification_confirmed"] is True:
        route = "RURAL_SOLAR"
        reason = "농업·축산 관련 자격이 확인되어 농촌형 태양광 금융지원을 검토합니다."
    elif facts["candidate_type"] == "building":
        route = "PUBLIC_ROOFTOP"
        reason = "후보유형이 공공건축물 옥상이므로 옥상형 사업을 검토합니다."
    else:
        route = "PUBLIC_LAND"
        reason = "별도 사업경로 조건이 없어 일반 공공 유휴부지 사업을 검토합니다."

    return {
        "route_type": route,
        "status": "SELECTED",
        "reason": reason,
        "regulatory_decision": decision,
        "regulatory_confidence": "MEDIUM",
    }


def program_region_match(
    program: dict[str, Any],
    facts: dict[str, Any],
) -> tuple[bool, str]:
    region = clean_text(program.get("지역"))
    city_condition = clean_text(program.get("시군"))

    if region == "전국":
        return True, "전국 사업"
    if region != facts["province"]:
        return False, "시도 불일치"
    if not city_condition:
        return True, "시도 일치"
    if "참여 시군" in city_condition:
        return True, "참여 시군 여부 공고 확인 필요"

    listed_cities = {
        normalize_city(token)
        for token in re.findall(r"[가-힣]+(?:시|군)?", city_condition)
        if clean_text(token)
    }
    if facts["city_base"] in listed_cities:
        return True, "시군 조건 일치"
    return False, "시군 조건 불일치"


def extract_first_number(value: Any) -> float | None:
    matched = re.search(
        r"(\d+(?:\.\d+)?)",
        clean_text(value).replace(",", ""),
    )
    return float(matched.group(1)) if matched else None


def program_prerequisites(
    program_id: str,
    facts: dict[str, Any],
    route_type: str,
) -> tuple[bool, list[str]]:
    missing_requirements: list[str] = []

    if program_id == "NAT_RE_FIN_PUBLIC_PARKING_2026":
        if facts["candidate_type"] != "parking_lot":
            return False, ["주차장 후보가 아님"]
        if facts["parking_function_maintained"] is not True:
            missing_requirements.append("주차 기능 유지 가능 여부 확인")

    if program_id in {
        "NAT_SUN_INCOME_SELECTION_2026",
        "NAT_RE_FIN_SUN_INCOME_2026",
        "CN_VILLAGE_PROFIT_SOLAR_2026",
        "CB_BOEUN_SUN_INCOME_PLAN_2026",
    }:
        if route_type != "SUN_INCOME_VILLAGE":
            return False, ["주민참여형 사업경로가 아님"]
        if facts["resident_participation_confirmed"] is not True:
            missing_requirements.append(
                "주민협동조합·마을공동사업 구조 확인"
            )

    if program_id == "NAT_RE_FIN_SUN_INCOME_2026":
        missing_requirements.append("햇빛소득마을 선정 여부 확인")

    if program_id in {
        "NAT_RE_FIN_INDUSTRIAL_2026",
        "CN_INDUSTRIAL_SOLAR_2026",
        "CN_INDUSTRIAL_ESS_2026",
        "CN_CRISIS_SOLAR_2026",
        "CN_INDUSTRIAL_COMMON_SOLAR_ESS_2026",
    }:
        if facts["industrial_site_confirmed"] is not True:
            return False, ["산업시설 여부 미확인"]

    if program_id in {
        "NAT_RE_FIN_INDUSTRIAL_2026",
        "CN_INDUSTRIAL_SOLAR_2026",
        "CN_CRISIS_SOLAR_2026",
    } and facts["self_consumption_confirmed"] is not True:
        return False, ["자가소비 계획 미확인"]

    if program_id == "NAT_RE_FIN_RURAL_2026":
        if facts["agricultural_qualification_confirmed"] is not True:
            return False, ["농업·축산 관련 신청자격 미확인"]

    if program_id == "CN_CRISIS_SOLAR_2026":
        missing_requirements.append("신청기업 소유 부지 여부 확인")

    return True, missing_requirements


def relation_aliases(program_id: str) -> set[str]:
    aliases = {program_id}
    if program_id in NATIONAL_FINANCE_PROGRAM_IDS:
        aliases.add("NATIONAL_RE_FINANCE_SUBTYPE")
    return aliases


def find_program_relations(
    selected_program_ids: list[str],
) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []

    for first, second in combinations(selected_program_ids, 2):
        first_aliases = relation_aliases(first)
        second_aliases = relation_aliases(second)

        matched = relation_df[
            (
                relation_df["program_a_id"].isin(first_aliases)
                & relation_df["program_b_id"].isin(second_aliases)
            )
            |
            (
                relation_df["program_a_id"].isin(second_aliases)
                & relation_df["program_b_id"].isin(first_aliases)
            )
        ]

        if matched.empty:
            relations.append({
                "program_a_id": first,
                "program_b_id": second,
                "status": "UNKNOWN",
                "explanation": "등록된 관계가 없어 공고기관 확인 필요",
            })
            continue

        row = matched.iloc[0]
        relations.append({
            "relation_id": clean_text(row.get("relation_id")),
            "program_a_id": first,
            "program_b_id": second,
            "matched_source_program_a_id": clean_text(
                row.get("program_a_id")
            ),
            "matched_source_program_b_id": clean_text(
                row.get("program_b_id")
            ),
            "status": clean_text(row.get("중복판정")),
            "explanation": clean_text(row.get("판정설명")),
            "required_check": clean_text(row.get("필수확인사항")),
        })

    return relations


def get_funding_conditions(program_id: str) -> list[dict[str, Any]]:
    matched = funding_df[funding_df["program_id"] == program_id]
    return [
        {
            str(key): ("" if is_missing(value) else value)
            for key, value in row.to_dict().items()
        }
        for _, row in matched.iterrows()
    ]


def select_support_programs(
    facts: dict[str, Any],
    assessment: dict[str, Any],
    route: dict[str, Any],
    top_k: int = TOP_K,
) -> dict[str, Any]:
    # 경로·지역·자격·용량 조건을 통과한 사업만 정의된 순서로 추천한다.
    if route["status"] not in {"SELECTED", "CONDITIONAL_SELECTED"}:
        return {
            "programs": [],
            "relations": [],
            "selection_block_reason": route["reason"],
        }

    route_type = route["route_type"]
    allowed_ids = ROUTE_PROGRAM_ORDER.get(route_type, [])
    order_index = {
        program_id: index for index, program_id in enumerate(allowed_ids)
    }

    candidates: list[dict[str, Any]] = []
    for _, row in policy_df.iterrows():
        program = row.to_dict()
        program_id = clean_text(program.get("program_id"))
        if program_id not in order_index:
            continue

        region_ok, region_reason = program_region_match(program, facts)
        if not region_ok:
            continue

        prereq_ok, missing_requirements = program_prerequisites(
            program_id,
            facts,
            route_type,
        )
        if not prereq_ok:
            continue

        min_capacity = extract_first_number(program.get("최소용량"))
        capacity = facts["recommended_capacity_kw"]
        if (
            min_capacity is not None
            and capacity is not None
            and capacity < min_capacity
        ):
            continue

        status_2026 = clean_text(program.get("2026상태"))
        conditional = (
            route["status"] == "CONDITIONAL_SELECTED"
            or status_2026 != "OPEN"
            or bool(missing_requirements)
        )
        match_status = (
            "CONDITIONAL_MATCH" if conditional else "PRELIMINARY_MATCH"
        )

        selection_reasons = [
            f"Ranking Rule 판정: {assessment['final_decision']}",
            f"사업경로: {route_type}",
            region_reason,
        ]
        if missing_requirements:
            selection_reasons.append(
                "추가 확인: " + ", ".join(missing_requirements)
            )

        if route["status"] == "CONDITIONAL_SELECTED":
            policy_condition = clean_text(
                program.get("규제대응추천조건")
            )
        else:
            policy_condition = clean_text(program.get("일반추천조건"))

        candidates.append({
            "priority_order": order_index[program_id],
            "program_id": program_id,
            "program_name": clean_text(program.get("사업명")),
            "parent_program_name": clean_text(
                program.get("상위사업명")
            ),
            "match_status": match_status,
            "status_2026": status_2026,
            "reason": route["reason"] + " " + clean_text(
                program.get("Agent_정책설명")
            ),
            "matched_policy_condition": policy_condition,
            "selection_reasons": selection_reasons,
            "support_method": clean_text(program.get("지원방식")),
            "support_rate": clean_text(program.get("지원비율")),
            "application_conditions": clean_text(
                program.get("신청조건")
            ),
            "application_status_text": clean_text(
                program.get("신청기간_상태설명")
            ),
            "missing_requirements": missing_requirements,
            "duplicate_support_rule": clean_text(
                program.get("동일설비_중복지원원칙")
            ),
            "funding_conditions": get_funding_conditions(program_id),
            "source_urls": [
                url
                for url in [
                    clean_text(program.get("출처URL_1")),
                    clean_text(program.get("출처URL_2")),
                ]
                if url
            ],
        })

    candidates.sort(
        key=lambda item: (item["priority_order"], item["program_id"])
    )
    selected = candidates[:top_k]

    for priority, item in enumerate(selected, start=1):
        item["priority"] = priority
        item.pop("priority_order", None)

    selected_ids = [item["program_id"] for item in selected]
    return {
        "programs": selected,
        "relations": find_program_relations(selected_ids),
        "selection_block_reason": None,
    }


class AgentState(TypedDict, total=False):
    # 각 처리 단계가 공유하는 입력과 중간 결과다.
    ranking_result: dict[str, Any]
    facts: dict[str, Any]
    regulatory_assessment: dict[str, Any]
    business_route: dict[str, Any]
    program_selection: dict[str, Any]
    final_explanation: dict[str, Any]
    result: dict[str, Any]
    errors: list[str]


def extract_facts_node(state: AgentState) -> dict[str, Any]:
    return {
        "facts": normalize_candidate(state["ranking_result"]),
        "errors": [],
    }


def resolve_pipeline_regulation_node(
    state: AgentState,
) -> dict[str, Any]:
    return {
        "regulatory_assessment": build_pipeline_regulatory_assessment(
            state["facts"]
        )
    }


def select_business_route_node(state: AgentState) -> dict[str, Any]:
    return {
        "business_route": select_business_route(
            state["facts"],
            state["regulatory_assessment"],
        )
    }


def select_support_programs_node(state: AgentState) -> dict[str, Any]:
    return {
        "program_selection": select_support_programs(
            state["facts"],
            state["regulatory_assessment"],
            state["business_route"],
        )
    }


def deterministic_final_explanation(
    state: AgentState,
    method: str,
) -> dict[str, Any]:
    assessment = state["regulatory_assessment"]
    route = state["business_route"]
    selection = state["program_selection"]
    programs = selection["programs"]

    required_checks = list(assessment.get("data_gaps", []))
    required_checks.extend(
        route.get("required_before_program_selection", [])
    )
    for program in programs:
        required_checks.extend(program.get("missing_requirements", []))
    required_checks = list(
        dict.fromkeys(
            item for item in required_checks if clean_text(item)
        )
    )

    if programs:
        first = programs[0]
        summary = (
            f"현재 입력 기준 최종 1순위 추천사업은 "
            f"{first['program_name']}입니다. "
            f"Ranking Rule 판정 {assessment['final_decision']}을 유지한 상태에서 "
            f"{route['route_type']} 경로로 검토했습니다."
        )
        second_comparison = ""
        if len(programs) > 1:
            second_comparison = (
                f"2순위는 {programs[1]['program_name']}이며, "
                "지원방식과 신청요건이 다르므로 후보지의 실제 사업주체와 자금조달 방식 확인이 필요합니다."
            )
        top_program = {
            "program_id": first["program_id"],
            "program_name": first["program_name"],
            "reason": first["reason"],
            "comparison_with_second": second_comparison,
        }
    else:
        summary = (
            "현재 입력 기준 추천 가능한 지원사업을 확정하지 못했습니다. "
            "Ranking Rule 판정 근거를 먼저 확인해야 합니다."
        )
        top_program = {
            "program_id": "",
            "program_name": "",
            "reason": "",
            "comparison_with_second": "",
        }

    return {
        "summary": summary,
        "regulation_reason": (
            "Agent가 이격거리나 조례를 다시 계산하지 않고 Ranking 파이프라인의 판정을 그대로 사용했습니다. "
            + assessment["final_reason"]
        ),
        "business_route_reason": route["reason"],
        "top_program": top_program,
        "program_reasons": {
            program["program_id"]: program["reason"]
            for program in programs
        },
        "required_checks": required_checks,
        "caution": (
            "본 결과는 Ranking 파이프라인 결과에 기반한 정책·자금지원 사전검토입니다. "
            "최종 인허가 승인이나 지원사업 선정 결과가 아니며 최신 공고와 제출서류 확인이 필요합니다."
        ),
        "method": method,
    }


def generate_llm_explanation_node(state: AgentState) -> dict[str, Any]:
    # LLM은 확정된 추천 결과를 바꾸지 않고 설명문만 생성한다.
    fallback = deterministic_final_explanation(
        state,
        method="DETERMINISTIC_FALLBACK",
    )

    if not USE_LLM or EXPLANATION_LLM is None:
        return {"final_explanation": fallback}

    selected_programs = state["program_selection"]["programs"]
    allowed_program_ids = [
        program["program_id"] for program in selected_programs
    ]
    payload = {
        "candidate": state["facts"],
        "regulatory_assessment": state["regulatory_assessment"],
        "business_route": state["business_route"],
        "selected_programs_in_fixed_order": selected_programs,
        "program_relations": state["program_selection"]["relations"],
    }

    prompt = f"""
당신은 지자체 공무원용 태양광 정책·자금지원 설명 Agent다.
입력 JSON에 있는 값만 사용한다.

절대 원칙:
1. 규제 판정은 Ranking 파이프라인의 결과다. final_decision을 변경하거나 재판정하지 않는다.
2. 이격거리 기준이나 지자체 조례 기준을 새로 계산하거나 추정하지 않는다.
3. 지원사업 목록, program_id, priority, 배열 순서를 변경하지 않는다.
4. 사용할 수 있는 program_id는 다음뿐이다: {allowed_program_ids}
5. funding_conditions의 지원비율·한도·상환기간·금리·상환의무를 변경하거나 새로 계산하지 않는다.
6. funding_conditions가 여러 개이면 신청주체별 조건을 구분한다. 특정 신청주체를 임의로 선택하지 않는다.
7. repayment_required가 true인 조건은 무상보조금이 아니라 상환 의무가 있는 금융지원이라고 설명한다.
8. 대부·사용허가 정책은 설치비 지원이 아니라 부지 사용권 확보 제도라고 설명한다.
9. FAIL 상태에서 제시된 사업은 규제 통과사업이 아니라 예외·사업구조 전환 가능성을 확인하는 조건부 대안이다.
10. OPEN이 아닌 사업은 현재 신청 가능한 사업으로 표현하지 않는다.
11. 입력에 없는 금액, 보조율, 절감액, 경제효과, 신청자격을 생성하지 않는다.
12. program_reasons는 선택된 모든 사업을 같은 순서로 작성한다.
13. 사전검토이며 최종 인허가·사업선정이 아니라는 주의를 포함한다.

작성 형식:
- summary 첫 문장: "현재 입력 기준 최종 1순위 추천사업은 [사업명]입니다."
- regulation_reason: Agent가 Ranking 판정을 그대로 사용했음을 설명한다.
- business_route_reason: 왜 해당 정책 경로가 선택되었는지 설명한다.
- top_program.reason: 지원방식, 자금조건, 신청 전 확인사항을 입력 범위에서 설명한다.
- comparison_with_second: 2순위가 있을 때 적용 차이를 설명한다.
- required_checks: 누락된 사실과 조건부 확인사항만 적는다.

입력 JSON:
{json.dumps(payload, ensure_ascii=False, default=str)}
"""

    try:
        parsed = EXPLANATION_LLM.invoke(prompt)
        result = parsed.model_dump()

        # 허용된 지원사업 외의 LLM 출력은 결과에서 제외한다.
        program_reason_map = {
            item["program_id"]: item["reason"]
            for item in result.get("program_reasons", [])
            if item.get("program_id") in allowed_program_ids
        }
        for program_id in allowed_program_ids:
            program_reason_map.setdefault(
                program_id,
                fallback["program_reasons"].get(program_id, ""),
            )

        top_program = result.get("top_program") or {}
        if selected_programs:
            first = selected_programs[0]
            top_program["program_id"] = first["program_id"]
            top_program["program_name"] = first["program_name"]
        result["top_program"] = top_program
        result["program_reasons"] = program_reason_map
        result["method"] = "LLM_EXPLANATION_WITH_DETERMINISTIC_GUARD"
        return {"final_explanation": result}

    except Exception as exc:
        errors = list(state.get("errors", []))
        errors.append(
            f"최종설명 LLM 실패: {type(exc).__name__}: {exc}"
        )
        if LLM_FAILURE_MODE != "FALLBACK":
            raise
        fallback["method"] = "LLM_ERROR_FALLBACK"
        return {
            "final_explanation": fallback,
            "errors": errors,
        }


def merge_result_node(state: AgentState) -> dict[str, Any]:
    # 원본 입력을 보존하고 추천 결과만 4_risk_and_support에 추가한다.
    result = copy.deepcopy(state["ranking_result"])
    risk_support = result.setdefault("4_risk_and_support", {})

    assessment = state["regulatory_assessment"]
    route = state["business_route"]
    selection = state["program_selection"]
    explanation = state["final_explanation"]

    def select_primary_funding_condition(
        program: dict[str, Any],
    ) -> dict[str, Any]:
        conditions = program.get("funding_conditions", [])
        if not isinstance(conditions, list):
            return {}

        for condition in conditions:
            if "공공기관" in clean_text(
                condition.get("applicant_type")
            ):
                return condition

        return conditions[0] if conditions else {}

    def compact_support_type(value: Any) -> str:
        text = clean_text(value)
        if "융자" in text:
            return "융자"
        if "보조" in text:
            return "보조"
        if "대부" in text or "임대료" in text:
            return "대부·사용허가"
        return text

    def first_sentence(value: Any) -> str:
        text = clean_text(value)
        if not text:
            return ""
        sentence = text.split(".", 1)[0].strip()
        return sentence + "." if sentence else ""

    def build_required_checks(
        program: dict[str, Any],
        funding: dict[str, Any],
    ) -> list[str]:
        checks = [
            clean_text(item)
            for item in program.get(
                "missing_requirements",
                [],
            )
            if clean_text(item)
        ]

        evidence = " ".join([
            clean_text(
                program.get("application_conditions")
            ),
            clean_text(
                program.get("matched_policy_condition")
            ),
            clean_text(
                funding.get("funding_caution")
            ),
        ])

        if "사업주체" in evidence:
            checks.append("사업주체 확인")
        if any(
            keyword in evidence
            for keyword in (
                "부지 권원",
                "부지권원",
                "사용권",
            )
        ):
            checks.append("부지 사용권 확인")
        if any(
            keyword in evidence
            for keyword in (
                "금융기관",
                "담보",
                "대출 심사",
            )
        ):
            checks.append("금융기관 심사")

        if (
            clean_text(program.get("status_2026"))
            != "OPEN"
        ):
            checks.append("현재 신청 가능 여부 확인")

        return list(dict.fromkeys(checks))

    compact_programs: list[dict[str, Any]] = []

    for program in selection["programs"][:1]:
        program_id = program["program_id"]
        ai_reason = explanation[
            "program_reasons"
        ].get(
            program_id,
            program.get("reason", ""),
        )

        route_reason = clean_text(route.get("reason"))
        concise_reason = clean_text(ai_reason)
        if (
            route_reason
            and concise_reason.startswith(route_reason)
        ):
            concise_reason = concise_reason[
                len(route_reason):
            ].strip()
        concise_reason = first_sentence(concise_reason)

        funding = select_primary_funding_condition(
            program
        )
        applicant_type = clean_text(
            funding.get("applicant_type")
        )
        support_ratio = clean_text(
            funding.get("support_ratio_text")
        )

        if applicant_type and support_ratio:
            support_summary = (
                f"{applicant_type} 기준 "
                f"{support_ratio}"
            )
        else:
            support_summary = clean_text(
                program.get("support_rate")
            )

        support_type = compact_support_type(
            first_not_missing(
                funding.get("funding_type_label"),
                program.get("support_method"),
            )
        )
        repayment_summary = clean_text(
            funding.get("repayment_terms_text")
        )
        application_period = clean_text(
            program.get("application_status_text")
        )
        required_checks = build_required_checks(
            program,
            funding,
        )

        source_urls = program.get("source_urls", [])
        source_url = (
            clean_text(source_urls[0])
            if isinstance(source_urls, list)
            and source_urls
            else ""
        )

        summary_parts = [
            (
                f"{program['program_name']}을 "
                "1순위로 추천합니다."
            )
        ]
        if concise_reason:
            summary_parts.append(concise_reason)
        if support_type:
            summary_parts.append(
                f"지원 유형은 {support_type}입니다."
            )
        if support_summary:
            summary_parts.append(
                f"지원 조건은 {support_summary}입니다."
            )
        if repayment_summary:
            summary_parts.append(
                f"상환 조건은 {repayment_summary}입니다."
            )
        if application_period:
            summary_parts.append(
                f"신청 기간은 {application_period}입니다."
            )
        if required_checks:
            summary_parts.append(
                "신청 전 다음 사항을 확인해야 합니다: "
                + ", ".join(required_checks)
                + "."
            )

        policy_summary = " ".join(summary_parts)

        compact_program = {
            "program_id": program_id,
            "program_name": program["program_name"],
            "priority": program["priority"],
            "status": program["status_2026"],
            "summary": policy_summary,
            "source_url": source_url,
        }

        if INCLUDE_POLICY_DETAILS:
            compact_program.update({
                "match_status": program["match_status"],
                "support_type": support_type,
                "support_summary": support_summary,
                "repayment_summary": repayment_summary,
                "application_period": application_period,
                "reason": concise_reason,
                "required_checks": required_checks,
            })

        compact_programs.append(compact_program)

    risk_support["regulatory_assessment"] = assessment
    risk_support["business_route"] = route
    risk_support["recommended_subsidies"] = (
        compact_programs
    )
    risk_support.pop(
        "support_program_relations",
        None,
    )
    risk_support["agent_explanation"] = {
        "caution": explanation["caution"],
    }
    # 사용한 데이터와 설명 생성 방식을 확인하기 위한 감사 정보다.
    risk_support["audit"] = {
        "agent_version": AGENT_VERSION,
        "processed_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "pipeline_score_preserved": True,
        "regulation_source": "RANKING_PIPELINE",
        "agent_regulatory_reassessment": False,
        "llm_enabled": USE_LLM,
        "llm_model": (
            OPENAI_MODEL if USE_LLM else None
        ),
        "explanation_method": explanation[
            "method"
        ],
        "errors": state.get("errors", []),
        "source_files": [POLICY_JSON_NAME],
    }

    return {"result": result}


class SequentialAgentGraph:
    # LangGraph 미설치 환경에서도 여섯 단계를 같은 순서로 실행한다.
    def invoke(self, initial_state: dict[str, Any]) -> AgentState:
        state: AgentState = dict(initial_state)

        def apply(node):
            update = node(state)
            if update:
                state.update(update)

        apply(extract_facts_node)
        apply(resolve_pipeline_regulation_node)
        apply(select_business_route_node)
        apply(select_support_programs_node)
        apply(generate_llm_explanation_node)
        apply(merge_result_node)
        return state


def build_agent_graph():
    if not LANGGRAPH_AVAILABLE:
        return SequentialAgentGraph()

    builder = StateGraph(AgentState)
    builder.add_node("extract_facts", extract_facts_node)
    builder.add_node(
        "resolve_pipeline_regulation",
        resolve_pipeline_regulation_node,
    )
    builder.add_node("select_business_route", select_business_route_node)
    builder.add_node(
        "select_support_programs",
        select_support_programs_node,
    )
    builder.add_node(
        "generate_llm_explanation",
        generate_llm_explanation_node,
    )
    builder.add_node("merge_result", merge_result_node)

    # 그래프는 조건을 판단하지 않고 실행 순서와 상태 전달만 관리한다.
    builder.add_edge(START, "extract_facts")
    builder.add_edge("extract_facts", "resolve_pipeline_regulation")
    builder.add_edge(
        "resolve_pipeline_regulation",
        "select_business_route",
    )
    builder.add_edge(
        "select_business_route",
        "select_support_programs",
    )
    builder.add_edge(
        "select_support_programs",
        "generate_llm_explanation",
    )
    builder.add_edge("generate_llm_explanation", "merge_result")
    builder.add_edge("merge_result", END)
    return builder.compile()


agent_graph = build_agent_graph()
