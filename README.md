# Solar Aivle 정책·자금지원 추천 AI Agent

태양광 후보지 분석 결과를 바탕으로 사업 경로를 선택하고, 적용 가능한 지원사업과 자금 조건을 추천하는 FastAPI 기반 AI Agent입니다.

LangGraph가 처리 순서와 상태를 관리하고, 실제 추천은 정책 JSON과 명시적인 조건문으로 결정합니다. LLM은 이미 결정된 결과를 이해하기 쉬운 설명으로 바꾸는 역할만 수행합니다.

> 본 서비스의 결과는 후보지 검토를 돕기 위한 참고 정보입니다. 실제 사업 추진 전에는 관할 기관과 관계 법령을 통해 최신 요건을 다시 확인해야 합니다.

## 핵심 기능

- 후보지 JSON에서 주소, 후보 유형, 설비용량, 규제 관련 정보를 추출·정규화
- 입력에 포함된 규제 판정 결과를 변경하지 않고 후속 추천에 활용
- 후보지 조건에 따라 사업 경로 선택
- 지역, 대상, 설비용량, 필수 조건에 맞는 지원사업 필터링
- 경로별 사전 정의된 우선순위에 따라 최종 추천을 최대 1건 선정
- OpenAI 기반 추천 사유 및 유의사항 생성
- LLM 장애 시 규칙 기반 설명으로 자동 대체
- 단건·배치 분석 및 정책 데이터 재로딩 API 제공

## 처리 흐름

```mermaid
flowchart LR
    A[후보지 분석 JSON] --> B[후보지 정보 추출·정규화]
    B --> C[전달된 규제 판정 확인]
    C --> D[사업 경로 선택]
    D --> E[지원사업 후보 필터링]
    E --> F[추천 설명 생성]
    F --> G[최종 JSON 병합]
```

LangGraph의 실행 순서는 다음과 같습니다.

```text
extract_facts
  → resolve_pipeline_regulation
  → select_business_route
  → select_support_programs
  → generate_llm_explanation
  → merge_result
```

현재 그래프는 조건부 분기 없이 6개 노드를 순차 실행합니다. LangGraph를 사용할 수 없는 환경에서는 동일한 순서의 Sequential fallback으로 자동 전환됩니다.

## 추천 방식

### 1. 후보지 정보 정규화

입력 JSON에서 다음 정보를 추출합니다.

- 후보지 ID와 주소
- 토지형·건물형 후보 구분
- 추천 설비용량
- 이격거리 위반 여부
- 공공사업·건축물 소유·자가소비·농촌 여부 등 사업 조건

### 2. 규제 판정 확인

입력에 규제 판정 결과가 있으면 해당 결과를 그대로 사용합니다. 이 Agent는 이격거리나 규제 적합성을 다시 판정하지 않습니다.

### 3. 사업 경로 선택

후보지 유형과 조건을 기준으로 다음 사업 경로 중 하나를 선택합니다.

| 사업 경로 | 적용 예시 |
|---|---|
| `PUBLIC_LAND` | 일반 공공 유휴부지 사업 |
| `PUBLIC_ROOFTOP` | 공공건축물 옥상형 사업 |
| `PUBLIC_PARKING` | 주차장 캐노피형 사업 |
| `PUBLIC_LED_SOLAR` | 공공주도형 태양광 사업 |
| `SUN_INCOME_VILLAGE` | 주민참여·햇빛소득마을 사업 |
| `SELF_CONSUMPTION_SOLAR` | 산업시설 자가소비형 사업 |
| `RURAL_SOLAR` | 농촌형 태양광 사업 |
| `REGULATORY_REVIEW_FIRST` | 규제 정보 확인이 우선인 경우 |

### 4. 지원사업 추천

정책 데이터에서 아래 조건을 순서대로 확인합니다.

- 선택된 사업 경로와의 일치 여부
- 전국·지역 적용 범위
- 신청 대상과 필수 조건
- 최소 설비용량 조건

조건을 통과한 사업은 경로별 고정 순서에 따라 비교합니다. 순서는 대체로 `핵심 사업 → 연계 금융 → 지역 연계 → 범용 지원·부지 제도`의 우선순위를 따르며, 현재 설정에서는 조건을 충족한 사업을 최대 1건 반환합니다.

### 5. 설명 생성

LLM은 다음 내용만 생성합니다.

- 추천 요약
- 추천 근거
- 신청 전 확인할 유의사항

LLM은 사업 경로, 추천 사업, 추천 순서, 규제 판정 또는 정책 조건을 변경할 수 없습니다.

## 프로젝트 구조

```text
AI_agent/
├── app/
│   ├── main.py                    # FastAPI 앱과 엔드포인트
│   ├── config.py                  # 환경변수 및 실행 설정
│   ├── core/
│   │   ├── agent_engine.py        # LangGraph 워크플로와 추천 로직
│   │   ├── data_repository.py     # 정책 JSON 로딩 및 조회
│   │   └── payload_utils.py       # 입력 JSON 탐색·정규화 유틸리티
│   └── services/
│       └── agent_service.py       # 단건·배치 분석 서비스
├── data/
│   └── 태양광_정책통합_2026.json
├── Dockerfile
├── requirements.txt
└── README.md
```

정책 JSON에는 현재 다음 데이터가 포함되어 있습니다.

- 정책·지원사업 14건
- 사업 경로별 지원사업 관계 13건
- 자금지원 조건 26건

## 입력 형식

다음 형태를 지원합니다.

- 후보지 객체 1건
- 후보지 객체 배열
- `results`, `data`, `candidates`로 감싼 객체

필수 필드는 다음과 같습니다.

| 필드 | 설명 |
|---|---|
| `1_site_info.site_id` | 후보지 식별자 |
| `1_site_info.address` | 후보지 주소 |

예시:

```json
{
  "target_type": "LAND",
  "1_site_info": {
    "site_id": "SITE-001",
    "address": "충청남도 예산군 예산읍"
  },
  "2_scores_and_evaluation": {
    "suitability": {
      "rule_decision": "PASS"
    }
  },
  "3_vision_and_simulation": {
    "vision_analysis": {
      "candidate_type": "land"
    },
    "simulation": {
      "recommended_capacity_kw": 146.56
    }
  },
  "4_risk_and_support": {
    "regulatory_input": {
      "setback_violation": false,
      "public_project_confirmed": true
    }
  }
}
```

## 출력 형식

입력 후보지 JSON을 유지한 채 `4_risk_and_support`에 추천 결과를 병합합니다.

| 필드 | 내용 |
|---|---|
| `regulatory_assessment` | 전달받은 규제 판정과 확인 메시지 |
| `business_route` | 선택된 사업 경로와 선택 근거 |
| `recommended_subsidies` | 추천 지원사업, 조건, 출처, 설명 |
| `agent_explanation` | 최종 유의사항 |
| `audit` | 실행 방식, 데이터 버전, LLM 사용 여부 등 추적 정보 |

## API

| Method | Endpoint | 설명 |
|---|---|---|
| `GET` | `/health` | 서버 상태와 정책 데이터 로딩 상태 확인 |
| `POST` | `/api/v1/agent/analyze` | 후보지 단건 분석 |
| `POST` | `/api/v1/agent/analyze-batch` | 후보지 목록 배치 분석 |
| `POST` | `/api/v1/admin/reload-data` | 정책 JSON 다시 로딩 |

`INTERNAL_API_KEY`를 설정한 경우 요청 헤더에 `X-Internal-API-Key`를 포함해야 합니다.

## 환경변수

프로젝트 루트에 `.env` 파일을 생성합니다.

```env
USE_LLM=true
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-5-mini
OPENAI_TIMEOUT_SECONDS=90
OPENAI_MAX_RETRIES=2
LLM_FAILURE_MODE=FALLBACK

MAX_BATCH_SIZE=100
INTERNAL_API_KEY=
INCLUDE_POLICY_DETAILS=false
```

| 변수 | 기본값 | 설명 |
|---|---:|---|
| `USE_LLM` | `true` | LLM 설명 생성 사용 여부 |
| `OPENAI_API_KEY` | 빈 값 | OpenAI API 키 |
| `OPENAI_MODEL` | `gpt-5-mini` | 설명 생성 모델 |
| `OPENAI_TIMEOUT_SECONDS` | `90` | LLM 요청 제한시간 |
| `OPENAI_MAX_RETRIES` | `2` | LLM 재시도 횟수 |
| `LLM_FAILURE_MODE` | `FALLBACK` | 실패 시 규칙 기반 설명 사용 여부 |
| `MAX_BATCH_SIZE` | `100` | 배치 요청 최대 후보지 수 |
| `INTERNAL_API_KEY` | 빈 값 | 내부 API 인증 키 |
| `INCLUDE_POLICY_DETAILS` | `false` | 응답에 세부 정책 데이터 포함 여부 |

## 기술 스택

- Python 3.11
- FastAPI / Uvicorn
- LangGraph
- LangChain OpenAI
- Pydantic / Pydantic Settings
- Pandas / NumPy
- Docker

## Repository

<https://github.com/AIVLE-big-project-19/AI_agent>
