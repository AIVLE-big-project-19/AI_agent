# 통합 정책 JSON 적용 패치

## 교체 파일

- `app/config.py`
- `app/core/data_repository.py`
- `app/core/agent_engine.py`
- `app/services/agent_service.py`
- `app/main.py`는 API 설명 문구만 변경
- `app/core/payload_utils.py`는 기존 파일 그대로

## 데이터

`data/태양광_정책통합_2026.json` 하나만 사용합니다.

## 처리 흐름

```text
Ranking JSON
→ Ranking의 Rule 판정값 수용
→ 사업경로 선택
→ 정책사업 추천
→ 중복지원 관계 조회
→ 사업별 자금지원 조건 결합
→ LLM 또는 결정론적 설명
```

Agent는 지자체 이격거리나 조례를 다시 판정하지 않습니다.
