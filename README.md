# Aimers 9기 : 투구 제구 성공 확률 예측 AI 온라인 해커톤

투구 직전까지 알 수 있는 경기 상황·선수 정보·주자 상황·과거 이력으로 각 투구의 `control_success` 확률을 예측하는 프로젝트입니다.

## 핵심 원칙

1. 현재 투구의 결과나 그 이후에 생성된 정보는 입력으로 사용하지 않습니다.
2. 목표는 0/1 분류가 아니라 **잘 보정된 확률 예측**입니다.
3. 모든 실험은 실제 제출 환경(오프라인, 시간·메모리 제한)에서 재현 가능해야 합니다.
4. 데이터 스키마를 확인하기 전에는 컬럼의 의미나 사용 가능 시점을 추측하지 않습니다.

## 대회 요약

| 항목 | 내용 |
|---|---|
| 예측 단위 | 개별 투구 |
| Target | `control_success` (`1`: 성공, `0`: 실패) |
| 제출값 | 각 투구의 제구 성공 확률 |
| 평가 지표 | Brier Skill Score |
| 평가 샘플 | 245,789개 |
| 추론 제한 | 10분 이하 |
| 패키지 설치 제한 | 10분 이하 |
| 제출 크기 | 압축 10GB 이하, 압축 해제 후 32GB 이하 |
| 실행 환경 | Ubuntu 22.04.5, Python 3.11.15, CUDA 12.8 |
| 하드웨어 | 6 vCPU, RAM 28GB, NVIDIA L4 22.4GiB |
| 제출 방식 | `submit.zip` 코드 제출 |

## 문서 지도

- [문제 정의](docs/01_problem_definition.md): 예측 시점, Target, 성공/실패 정의
- [평가 및 제출 규격](docs/02_competition_and_submission.md): 평가식, 서버 제약, 제출 체크리스트
- [모델링 프로토콜](docs/03_modeling_protocol.md): 누수 방지, 검증, 실험 순서
- [데이터 사전](docs/04_data_dictionary.md): 데이터 수령 후 채울 컬럼별 사용 가능성
- [결정 및 이슈 기록](docs/05_decision_log.md): 확정 사항, 가설, 미확정 사항
- [실험 기록 템플릿](docs/templates/experiment.md)

## 현재 상태와 다음 단계

공식 배포 데이터와 베이스라인 제출 코드가 프로젝트에 추가되었습니다.

```text
lg_aimers/
├── data/
│   ├── train.csv               # 1,475,092행 × 49컬럼
│   ├── test.csv                # 형식 확인용 5행 × 48컬럼
│   ├── sample_submission.csv   # 형식 확인용 5행 × 2컬럼
│   └── trackman_history.csv    # 1,793,078행 × 30컬럼
├── model/rf.pkl                # 운영진 베이스라인 모델
├── script.py                   # 베이스라인 추론 코드
├── requirements.txt            # 베이스라인 의존성
├── baseline_submit.zip         # 원본 베이스라인 제출 예시
├── data_description.md         # 공식 데이터 설명서
└── docs/                       # 프로젝트 문서
```

실제 평가 시 `data/test.csv`와 `data/sample_submission.csv`는 245,789행 규모의 비공개 파일로 교체됩니다. 로컬의 5개 행은 형식·실행 확인용일 뿐 성능 평가용이 아닙니다.

다음 순서로 진행합니다.

1. 파일 구조, 행 수, 자료형, 결측치, 키 중복을 점검합니다.
2. 모든 컬럼에 대해 현재 투구 직전 사용 가능 여부를 판정합니다.
3. 시간·경기·선수 구조를 반영한 검증 전략과 상수 확률 베이스라인을 확정합니다.
4. 누수 없는 과거 이력 피처와 첫 번째 트리 기반 모델을 구축합니다.
5. Brier Skill Score, 확률 보정, 추론 시간, 메모리를 함께 기록합니다.

## 공식 자료

- [대회 평가 페이지](https://dacon.io/competitions/official/236743/overview/evaluation)
- [DACON 코드 제출 가이드](https://cfiles.dacon.co.kr/competitions/236564/guide.html)
