# Aimers 9기 — 투구 제구 성공 확률 예측

투구가 이루어지기 **직전**까지 알 수 있는 정보로 `control_success=1`일 확률을 예측한 프로젝트입니다. 최종 제출 모델은 전처리와 HistGradientBoostingClassifier(HGB)를 묶은 두 파이프라인의 확률 앙상블입니다.

## 문제와 데이터 경계

- 예측 단위: 개별 투구
- Target: `control_success` (`1`: 제구 성공, `0`: 제구 실패)
- 출력: `[0, 1]` 범위의 제구 성공 확률
- 평가 지표: Brier Skill Score(BSS)
- 예측 기준 시점: 현재 투구가 이루어지기 직전

현재 투구의 최종 위치·구속·판정·타격 결과와 이후 투구 정보는 피처로 사용하지 않았습니다. 자체 rolling/누적 피처도 만들지 않았고, 운영진이 현재 행을 제외해 미리 계산한 `asof_*` 이력만 사용했습니다. 평가 데이터 전체의 통계나 순서는 어떤 학습 단계에도 사용하지 않습니다.

## 어떻게 학습했는가

### 1. 사용할 데이터와 피처를 확정

`data/train.csv`의 2019~2024 시즌 1,475,092행만 모델 학습에 사용했습니다.

- 제외: 제출 키 `row_id`, Target `control_success`
- 사용: 투구 전 경기 상황, 선수·팀 정보, 운영진 제공 `asof_*` 과거 이력
- Trackman 제외: 메인 데이터와 공식 조인 키가 없고 투수·타자 ID 교집합도 없어 추정 조인을 만들지 않음
- `test.csv` 제외: 피처 선택, 전처리, 결측치 대체, 모델 선택, 확률 보정에 전혀 사용하지 않음

전체 피처 목록과 시점 판정은 [데이터 사전](docs/04_data_dictionary.md)에 기록되어 있습니다.

### 2. 미래 시즌을 보지 않는 검증

무작위 행 분할 대신 실제 2025 시즌 예측 상황을 흉내 낸 시즌 순방향 holdout을 사용했습니다.

| 검증 | 학습 구간 | 검증 구간 |
|---|---|---|
| Fold 1 | 2019~2021 | 2022 |
| Fold 2 | 2019~2022 | 2023 |
| Fold 3 | 2019~2023 | 2024 |

각 fold에서 encoder와 결측치 대체기는 학습 구간으로만 fit됩니다. validation 시즌의 범주·중앙값·성공률은 전처리에 들어가지 않습니다.

BSS는 대회 공식 정의와 같은 아래 식으로 계산합니다. `BS_ref`는 해당 validation의 실제 성공률을 모든 행에 예측했을 때의 Brier Score입니다.

```text
BS     = mean((예측 확률 - 실제값)²)
BSS    = max(0, 100000 × (1 - BS / BS_ref))
BS_ref = validation 성공률 × (1 - validation 성공률)
```

### 3. fold 내부 전처리

전처리와 모델을 하나의 scikit-learn `Pipeline`으로 묶었습니다.

- `top_bottom`, `game_type`, `base_state`: 학습 구간에서 ordinal encoding
- validation/test의 미관측 범주: `-1`
- 나머지 수치 피처의 결측값: 학습 구간 중앙값
- 스케일링과 Target encoding: 사용하지 않음

### 4. 두 HGB의 확률을 앙상블

| 설정 | 모델 A | 모델 B |
|---|---:|---:|
| 피처 수 | 47 | 43 |
| 제외 피처 | 없음 | 투수·타자·양 팀 ID 4개 |
| learning rate | 0.05 | 0.05 |
| 최대 반복 | 250 | 250 |
| 최대 leaf 수 | 31 | 15 |
| leaf 최소 표본 | 200 | 500 |
| L2 규제 | 3.0 | 5.0 |
| seed | 42 | 42 |

두 모델 모두 학습 구간 내부 10%를 조기 종료에 사용합니다. 최종 예측은 두 모델의 `control_success=1` 확률을 `0.5 : 0.5`로 평균합니다. 모델 B는 익명 ID 코드의 임의 순서와 시즌별 선수 변화에 대한 의존도를 낮추는 역할입니다.

### 5. calibration을 적용하지 않은 이유

직전 시즌 예측에 sigmoid 또는 logit intercept 보정을 fit한 뒤 다음 시즌에 적용했지만 BSS가 일관되게 개선되지 않았습니다. 특히 2024에서 단일 HGB의 sigmoid 보정은 BSS를 `587.43`에서 `119.02`로 낮췄습니다. 따라서 최종 제출 확률에는 별도 사후 보정을 적용하지 않았습니다.

### 6. 전체 데이터로 최종 재학습

모델 구성을 확정한 뒤 두 파이프라인을 2019~2024 전체 학습 데이터로 다시 fit했습니다. 학습된 전처리기, 모델, 피처 목록과 가중치를 하나의 `model/rf.pkl`에 저장했습니다. 파일명은 운영진 베이스라인과의 호환을 위해 유지했으며, 내부 모델은 RandomForest가 아니라 HGB입니다.

## 검증 결과

| 검증 시즌 | 학습 시즌 | Brier | BSS | 실제 성공률 | 예측 평균 |
|---:|---|---:|---:|---:|---:|
| 2022 | 2019~2021 | 0.243597 | 2,234.04 | 0.528920 | 0.529348 |
| 2023 | 2019~2022 | 0.253362 | 0.00 | 0.499957 | 0.520170 |
| 2024 | 2019~2023 | 0.248312 | 598.33 | 0.486105 | 0.496857 |

2023은 실제 성공률이 직전 시즌보다 약 2.90%p 급락해 비교한 모든 후보가 0점이었습니다. 이 분포 이동은 숨기지 않고 최종 모델의 주요 위험으로 남겼으며, 2022와 2024에서 모두 개선된 구성을 선택했습니다.

최종 전체 학습은 약 106초, 저장 모델은 약 0.64MiB였습니다. 245,789행을 복제한 로컬 전체 추론은 10.79초, 최대 프로세스 트리 메모리는 0.65GiB였습니다.

## 실행 방법

프로젝트 루트에서 실행합니다. 학습 데이터는 `data/train.csv`, 샘플 추론 입력은 `data/test.csv`와 `data/sample_submission.csv`에 둡니다.

### 1. 시즌 순방향 검증

```bash
python train.py validate
```

기본으로 2022·2023·2024 holdout을 실행하고 `model/validation_report.json`에 seed, split, 피처 구성, 지표와 실행 시간을 기록합니다.

특정 시즌만 확인할 수도 있습니다.

```bash
python train.py validate --holdout-years 2024
```

### 2. 최종 모델 학습

```bash
python train.py fit
```

2019~2024 전체 데이터로 `model/rf.pkl`을 생성하고 `model/training_report.json`에 학습 환경과 실행 시간을 남깁니다. 현재 제공된 `rf.pkl`은 평가 서버와 같은 NumPy 1.26.4, SciPy 1.15.3, scikit-learn 1.8.0 계열에서 직렬화와 재로드를 검증한 파일입니다.

### 3. 로컬 추론

```bash
python script.py
```

`model/rf.pkl`을 로드해 `output/submission.csv`를 생성합니다. 저장 전에 test/submission의 행 수·ID 집합·중복, 예측값의 유한성과 `[0, 1]` 범위를 검사합니다.

### 4. 제출 ZIP 생성

```bash
python build_submission.py
```

평가 서버가 요구하는 Unix 권한과 경로를 적용해 `submit.zip`을 만듭니다. ZIP에는 아래 항목만 포함됩니다.

```text
model/
model/rf.pkl
script.py
requirements.txt
```

## 최소 파일 구조

```text
lg_aimers/
├── train.py                 # 시즌 검증 + 최종 모델 학습
├── script.py                # 평가 서버용 오프라인 추론
├── build_submission.py      # submit.zip 생성
├── model/
│   └── rf.pkl               # 학습 완료된 HGB 2개 앙상블
├── data/                    # 원본 데이터, Git/제출 ZIP에서 제외
├── docs/
│   ├── 01_problem_definition.md
│   ├── 04_data_dictionary.md
│   └── 05_decision_log.md
├── data_description.md      # 공식 데이터 설명
├── requirements.txt         # 제출 서버 추가 설치 없음
├── submit.zip               # 최종 코드 제출 파일
├── AGENTS.md                # 저장소 작업 원칙
└── README.md
```

`data/`, `output/`, 로컬 가상환경과 Python 캐시는 `.gitignore`로 제외합니다. 원본 데이터는 수정하지 않습니다.

## 실행 환경

- 평가 서버: Ubuntu 22.04.5, Python 3.11.15, 6 vCPU, RAM 28GB
- 사용 라이브러리: pandas, NumPy, scikit-learn, joblib, threadpoolctl
- 네트워크 및 외부 API: 사용하지 않음
- 추가 패키지 설치: 없음; 서버 기본 패키지를 사용하므로 `requirements.txt`에는 설명 주석만 있음

## 관련 문서

- [문제 정의](docs/01_problem_definition.md)
- [데이터 사전과 피처 시점 판정](docs/04_data_dictionary.md)
- [결정 기록](docs/05_decision_log.md)
- [공식 데이터 설명](data_description.md)

공식 자료: [대회 평가 페이지](https://dacon.io/competitions/official/236743/overview/evaluation), [DACON 코드 제출 가이드](https://cfiles.dacon.co.kr/competitions/236564/guide.html)
공식 사이트: [LG Aimers](https://www.lgaimers.ai/)
