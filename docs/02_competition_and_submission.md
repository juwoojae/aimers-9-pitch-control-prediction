# 평가 및 제출 규격

## 평가 지표: Brier Skill Score

공식 평가 페이지에 명시된 핵심 지표는 Brier Skill Score(BSS)입니다.

\[
\text{Brier} = \frac{1}{N}\sum_{i=1}^{N}(p_i-y_i)^2
\]

\[
r = \frac{1}{N}\sum_{i=1}^{N}y_i, \qquad
\text{Brier}_{ref}=r(1-r)
\]

\[
\text{Score}=\max\left(0,\ 100000\left(1-\frac{\text{Brier}}{\text{Brier}_{ref}}\right)\right)
\]

- `p_i`: 모델이 예측한 제구 성공 확률
- `y_i`: 실제 레이블(0 또는 1)
- `r`: 평가 데이터의 실제 성공 비율
- 기준 예측: 모든 행에 `r`을 예측하는 상수 모델
- 높을수록 좋으며, 기준 모델과 같거나 나쁘면 0점으로 절단됩니다.

로컬 검증에서는 fold별 validation 레이블의 평균으로 `Brier_ref`를 계산하고, Brier Score 자체도 함께 기록합니다. 리더보드 점수를 역산하거나 평가 데이터의 실제 `r`을 가정해 튜닝하지 않습니다.

## 리더보드 및 평가 절차

- Public Score: 전체 테스트 데이터 100%
- Private Score: 대회 종료 시점의 Public Score
- LG Aimers 수료 조건: Phase1 이수 및 Phase2 Public Score `549.51` 이상
- 기준 점수: 운영진 베이스라인 추론 코드를 운영진 평가 환경에서 실행한 점수
- 1차 평가: Private Score 100%
- 동점자: DACON의 기존 리더보드 순위 산정 방식 적용
- 2차 평가: Phase3 진출 희망 팀의 코드 제출 및 검증
- Private 상위팀 약 100명: 코드와 PPT 필수 제출 대상
- 코드·PPT 제출 및 검증을 모두 통과한 Private 상위팀 약 100명: 오프라인 해커톤(Phase3) 진출

`549.51`은 수료 기준이지 모델 개발의 최종 목표 점수로 간주하지 않습니다.

## 실행 환경

| 제약 | 값 |
|---|---:|
| 실제 평가 샘플 | 245,789개 |
| 전체 추론 실행 | 10분 이하 |
| 패키지 설치 | 10분 이하 |
| 제출 압축 파일 | 10GB 이하 |
| 압축 해제 크기 | 32GB 이하 |
| CPU | 6 vCPU |
| RAM | 28GB |
| GPU | NVIDIA L4, VRAM 22.4GiB |
| OS | Ubuntu 22.04.5 LTS |
| Python | 3.11.15 |
| CUDA | 12.8 |
| 네트워크 | 패키지 설치 외 오프라인 |

시간 제한은 단순 모델의 `predict` 호출뿐 아니라 데이터 로드, 피처 생성, 모델 로드, 예측, 결과 저장을 포함한 `script.py` 전체 실행을 기준으로 관리합니다.

## 제출 압축 구조

코드 제출 가이드의 필수 최상위 구조는 다음과 같습니다.

```text
submit.zip
├── model/              # 학습 완료 모델 및 추론에 필요한 로컬 자원
├── script.py           # 서버가 실행하는 추론 전용 진입점
└── requirements.txt    # 추가 의존성
```

평가 서버는 압축 해제 후 다음 경로를 자동으로 추가합니다.

```text
data/                   # 실제 평가 데이터
output/                 # 결과 저장 위치
```

`script.py`는 반드시 `output/submission.csv`를 생성해야 합니다. 학습 과정은 포함하지 않으며, 인터넷 다운로드나 외부 API·원격 DB 호출에 의존해서는 안 됩니다.

`data/`는 평가 데이터가 들어가는 읽기 전용 디렉터리입니다. 공식 배포 ZIP의 구조와 베이스라인 `script.py`가 모두 `./data/test.csv` 및 `./data/sample_submission.csv`를 사용하므로 실제 입력 경로는 `data/`로 확정합니다. 원본 안내 마지막 유의사항의 `open/` 표기는 앞선 구조 설명과 맞지 않는 오타로 판단합니다.

## 기본 설치 패키지

아래 버전은 평가 서버에 기본 설치되어 있습니다. 특별한 이유가 없다면 `requirements.txt`에 다시 넣지 않습니다. 다른 버전을 강제 설치하면 충돌이나 설치 오류가 발생할 수 있습니다.

```text
torch==2.7.1+cu128
pandas==2.0.3
numpy==1.26.4
scipy==1.15.3
scikit-learn==1.8.0
joblib==1.5.3
threadpoolctl==3.6.0
narwhals==2.21.2
transformers==4.46.3
accelerate==1.9.0
sentencepiece==0.1.99
regex==2023.12.25
tqdm==4.66.4
loguru==0.7.2
pyyaml==6.0.1
rich==13.7.1
```

기본 시스템 패키지는 다음과 같습니다.

```text
git
build-essential
python3.11
python3.11-dev
python3.11-venv
python3-pip
libffi-dev
libblas3
liblapack3
libomp-dev
tzdata
unzip
p7zip-full
gfortran
libatlas-base-dev
default-jre-headless
cmake
pkg-config
ninja-build
libgl1
libglib2.0-0
```

원본 `baseline_submit.zip`은 `pandas==2.3.3`을 지정하지만, 현재 제출용 `requirements.txt`는 평가 서버에 이미 설치된 pandas 2.0.3, scikit-learn 1.8.0, joblib 1.5.3을 그대로 사용합니다. 추가 패키지가 없으므로 설치 충돌과 설치 시간 위험을 줄입니다.

## 오류 유형과 제출 횟수

| 유형 | 대표 원인 | 일일 제출 횟수 반영 |
|---|---|---|
| 설치 오류 | ZIP 구조 불일치, `requirements.txt` 설치 실패·시간 초과 | 반영되지 않음 |
| 제출 오류 | `script.py` 실행 이후 발생하는 오류, 추론 시간 초과, 결과 파일 미생성 | 반영됨 |

따라서 제출 오류를 서버에서 디버깅하지 않도록 로컬에서 전체 추론 경로와 결과 검증을 끝낸 뒤 제출합니다.

## 구현 원칙

- 실행 위치에 의존하는 절대 경로 대신 제출 루트 기준 상대 경로를 사용합니다.
- 평가 데이터의 파일명과 출력 스키마는 제공되는 샘플을 확인한 뒤 확정합니다.
- 서버 기본 패키지는 중복 설치하지 않고, 추가 패키지만 최소한으로 명시합니다.
- 모델·인코더·설정·범주 사전 등 필요한 자원은 모두 압축 파일에 포함합니다.
- 행 순서를 바꾸는 처리에는 원본 ID 기반 복원 검사를 둡니다.
- 출력 확률의 결측, 무한값, 범위 이탈, 행 수, ID 일치를 저장 전에 검사합니다.

## 제출 전 체크리스트

- [ ] 압축 최상위에 `model/`, `script.py`, `requirements.txt`가 있다.
- [ ] 새 환경에서 `pip install -r requirements.txt`가 10분 안에 끝난다.
- [ ] 서버 기본 패키지를 불필요하게 `requirements.txt`에 중복 기재하지 않았다.
- [ ] 인터넷을 끊은 상태에서 `python script.py`가 정상 종료된다.
- [ ] 실제 입력 디렉터리 이름을 샘플 평가 패키지에서 확인했다.
- [ ] 245,789행 상당의 입력으로 전체 실행이 10분 안에 끝난다.
- [ ] 최대 메모리 사용량이 28GB보다 충분히 낮다.
- [ ] `output/submission.csv`의 컬럼, 순서, 행 수가 샘플 제출 파일과 같다.
- [ ] 모든 예측값이 유한한 `[0, 1]` 확률이다.
- [ ] 압축 및 압축 해제 크기 제한을 만족한다.
- [ ] 현재 투구 이후 정보를 읽거나 생성하는 코드가 없다.

## 출처

- [Aimers 9기 대회 평가 페이지](https://dacon.io/competitions/official/236743/overview/evaluation), 확인일: 2026-08-08
- [DACON 코드 제출 대회 가이드](https://cfiles.dacon.co.kr/competitions/236564/guide.html), 확인일: 2026-08-08
