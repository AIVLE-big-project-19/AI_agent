from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import Settings


@dataclass(frozen=True)
class DataBundle:
    # 추천에 필요한 정책·관계·자금조건의 필수 필드다.
    policy_df: pd.DataFrame
    relation_df: pd.DataFrame
    funding_df: pd.DataFrame
    schema_version: str
    reference_year: int

    def counts(self) -> dict[str, int]:
        return {
            "policies": len(self.policy_df),
            "relations": len(self.relation_df),
            "funding_conditions": len(self.funding_df),
        }


class DataRepository:
    POLICY_REQUIRED_COLUMNS = {
        "program_id",
        "사업명",
        "지역",
        "시군",
        "2026상태",
        "Agent_정책설명",
    }
    RELATION_REQUIRED_COLUMNS = {
        "program_a_id",
        "program_b_id",
        "중복판정",
        "판정설명",
    }
    FUNDING_REQUIRED_COLUMNS = {
        "funding_condition_id",
        "program_id",
        "funding_type",
        "funding_type_label",
        "repayment_required",
        "support_ratio_text",
        "funding_caution",
    }

    def __init__(self, settings: Settings):
        self.settings = settings

    @staticmethod
    def _require_file(path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(
                f"필수 데이터 파일을 찾지 못했습니다: {path}"
            )

    @staticmethod
    def _require_list(
        payload: dict[str, Any],
        key: str,
    ) -> list[dict[str, Any]]:
        value = payload.get(key)
        if not isinstance(value, list):
            raise ValueError(
                f"통합 정책 JSON의 '{key}'는 배열이어야 합니다."
            )
        if not all(isinstance(item, dict) for item in value):
            raise ValueError(
                f"통합 정책 JSON의 '{key}'에는 객체만 들어가야 합니다."
            )
        return value

    @staticmethod
    def _validate_columns(
        frame: pd.DataFrame,
        required: set[str],
        label: str,
    ) -> None:
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{label} 필수 필드 누락: {missing}")

    @staticmethod
    def _validate_unique(
        frame: pd.DataFrame,
        column: str,
        label: str,
    ) -> None:
        duplicated = (
            frame.loc[frame[column].duplicated(), column]
            .astype(str)
            .tolist()
        )
        if duplicated:
            raise ValueError(
                f"{label} 중복값이 있습니다: {sorted(set(duplicated))}"
            )

    def load(self) -> DataBundle:
        path = self.settings.policy_json_path
        self._require_file(path)

        try:
            with path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"통합 정책 JSON 형식 오류: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise ValueError("통합 정책 JSON 최상위 값은 객체여야 합니다.")

        # 통합 JSON의 데이터 배열을 각각 DataFrame으로 변환한다.
        policies = self._require_list(payload, "policy_programs")
        relations = self._require_list(payload, "support_relations")
        funding = self._require_list(payload, "funding_conditions")

        policy_df = pd.DataFrame(policies).fillna("")
        relation_df = pd.DataFrame(relations).fillna("")
        funding_df = pd.DataFrame(funding).fillna("")

        self._validate_columns(
            policy_df,
            self.POLICY_REQUIRED_COLUMNS,
            "정책 데이터",
        )
        self._validate_columns(
            relation_df,
            self.RELATION_REQUIRED_COLUMNS,
            "중복지원 관계 데이터",
        )
        self._validate_columns(
            funding_df,
            self.FUNDING_REQUIRED_COLUMNS,
            "자금지원 조건 데이터",
        )

        self._validate_unique(policy_df, "program_id", "program_id")
        self._validate_unique(
            relation_df,
            "relation_id",
            "relation_id",
        )
        self._validate_unique(
            funding_df,
            "funding_condition_id",
            "funding_condition_id",
        )

        # 모든 정책이 최소 하나의 자금지원 조건과 연결되어 있는지 확인한다.
        policy_ids = set(policy_df["program_id"].astype(str))
        funding_program_ids = set(funding_df["program_id"].astype(str))
        missing_funding = sorted(policy_ids - funding_program_ids)
        if missing_funding:
            raise ValueError(
                "자금지원 조건이 연결되지 않은 program_id가 있습니다: "
                f"{missing_funding}"
            )

        return DataBundle(
            policy_df=policy_df,
            relation_df=relation_df,
            funding_df=funding_df,
            schema_version=str(payload.get("schema_version", "")),
            reference_year=int(payload.get("reference_year", 0) or 0),
        )
