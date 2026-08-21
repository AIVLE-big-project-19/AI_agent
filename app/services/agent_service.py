from __future__ import annotations

import copy
import threading
import time
from typing import Any

from app.config import Settings
from app.core import agent_engine
from app.core.data_repository import DataRepository
from app.core.payload_utils import (
    extract_candidate_list,
    rebuild_payload,
    to_json_safe,
    validate_candidate,
)


class AgentService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.repository = DataRepository(settings)
        self._lock = threading.RLock()
        self._initialized = False
        self._last_reload_epoch: float | None = None
        self._data_counts: dict[str, int] = {}

    def initialize(self) -> dict[str, Any]:
        return self.reload_data()

    def reload_data(self) -> dict[str, Any]:
        bundle = self.repository.load()

        # 재적재 중 요청이 실행되어 정책 데이터가 섞이지 않도록 잠근다.
        with self._lock:
            runtime = agent_engine.configure_runtime(
                policies=bundle.policy_df,
                relations=bundle.relation_df,
                funding_conditions=bundle.funding_df,
                use_llm=self.settings.use_llm,
                openai_api_key=self.settings.openai_api_key,
                openai_model=self.settings.openai_model,
                openai_timeout_seconds=(
                    self.settings.openai_timeout_seconds
                ),
                openai_max_retries=(
                    self.settings.openai_max_retries
                ),
                llm_failure_mode=self.settings.llm_failure_mode,
                policy_json_name=self.settings.policy_json_name,
                include_policy_details=(
                    self.settings.include_policy_details
                ),
            )
            self._data_counts = bundle.counts()
            self._last_reload_epoch = time.time()
            self._initialized = True

        return {
            "success": True,
            "data_counts": self._data_counts,
            "schema_version": bundle.schema_version,
            "reference_year": bundle.reference_year,
            "runtime": runtime,
        }

    def health(self) -> dict[str, Any]:
        runtime = agent_engine.get_runtime_status()
        return {
            "status": "UP" if self._initialized else "STARTING",
            "agent_version": agent_engine.AGENT_VERSION,
            "initialized": self._initialized,
            "data_counts": self._data_counts,
            "last_reload_epoch": self._last_reload_epoch,
            "runtime": runtime,
        }

    def analyze_candidate(
        self,
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        validate_candidate(candidate)

        # LangGraph와 순차 실행기는 동일한 invoke 인터페이스를 사용한다.
        with self._lock:
            state = agent_engine.agent_graph.invoke({
                "ranking_result": copy.deepcopy(candidate),
                "errors": [],
            })

        return to_json_safe(state["result"])

    def analyze_batch(
        self,
        payload: Any,
    ) -> dict[str, Any]:
        candidates, wrapper = extract_candidate_list(payload)

        if len(candidates) > self.settings.max_batch_size:
            raise ValueError(
                f"배치 최대 처리 건수는 "
                f"{self.settings.max_batch_size}건입니다."
            )

        for candidate in candidates:
            validate_candidate(candidate)

        results: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []

        # 입력 순서와 결과 순서를 유지하도록 후보지를 순차 처리한다.
        with self._lock:
            for input_order, candidate in enumerate(
                candidates,
                start=1,
            ):
                state = agent_engine.agent_graph.invoke({
                    "ranking_result": copy.deepcopy(candidate),
                    "errors": [],
                })

                result = state["result"]
                results.append(result)

                facts = state["facts"]
                assessment = state["regulatory_assessment"]
                route = state["business_route"]
                programs = state["program_selection"]["programs"]
                explanation = state["final_explanation"]

                summary_rows.append({
                    "input_order": input_order,
                    "site_id": facts["site_id"],
                    "jurisdiction": facts["jurisdiction_norm"],
                    "candidate_type": facts["candidate_type"],
                    "pipeline_score": facts["pipeline_total_score"],
                    "upstream_rule_decision": assessment[
                        "upstream_rule_decision"
                    ],
                    "regulatory_decision": assessment[
                        "final_decision"
                    ],
                    "setback_violation": assessment[
                        "setback_violation"
                    ],
                    "business_route": route["route_type"],
                    "program_count": len(programs),
                    "program_ids": [
                        item["program_id"] for item in programs
                    ],
                    "explanation_method": explanation["method"],
                    "data_gaps": assessment["data_gaps"],
                    "errors": state.get("errors", []),
                })

        runtime = agent_engine.get_runtime_status()
        # 추천 결과를 원래 입력 JSON 구조에 다시 넣는다.
        rebuilt = rebuild_payload(
            payload,
            results,
            wrapper,
            agent_version=agent_engine.AGENT_VERSION,
            llm_enabled=runtime["llm_enabled"],
            llm_model=runtime.get("llm_model"),
        )

        return to_json_safe({
            "processed_count": len(results),
            "result": rebuilt,
            "items": summary_rows,
        })
