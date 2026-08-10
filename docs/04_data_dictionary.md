# 데이터 사전

공식 `data_description.md`와 배포 CSV를 기준으로 작성한 살아 있는 문서입니다. 상태가 `UNKNOWN`인 컬럼은 모델에 사용하지 않습니다.

## 파일 목록

| 파일 | 역할 | 행 수 | 기본 키 | 비고 |
|---|---|---:|---|---|
| `data/train.csv` | 학습 입력 및 Target | 1,475,092 | `row_id` | 49컬럼 |
| `data/test.csv` | 평가 입력 형식 샘플 | 5 | `row_id` | 48컬럼, 서버에서 245,789행으로 교체 |
| `data/sample_submission.csv` | 제출 형식 샘플 | 5 | `row_id` | `control_success` 확률 제출 |
| `data/trackman_history.csv` | 2019~2024 Trackman 과거 로그 | 1,793,078 | `trackman_id` | 30컬럼, 메인 데이터와 1:1 조인 아님 |

## 컬럼 판정

| 컬럼 | 파일 | 자료형 | 의미 | 생성·확정 시점 | 상태 | 사용 방식 | 근거/주의사항 |
|---|---|---|---|---|---|---|---|
| `row_id` | train/test/submission | string | 샘플 고유 ID | 사전 부여 | `ID_ONLY` | 조인·순서 검증 | 제출 ID와 정확히 일치해야 함 |
| `season`, `game_month`, `game_dayofweek` | train/test | int | 시즌·경기 월·요일 | 투구 전 | `AVAILABLE` | 시간·범주 피처 | 정확한 경기 날짜·경기 ID는 미제공 |
| `inning`, `top_bottom`, `game_type` | train/test | mixed | 이닝·초말·경기 유형 | 투구 전 | `AVAILABLE` | 수치·범주 피처 | `top_bottom`: `T`/`B` |
| `balls_before`, `strikes_before`, `outs_before` | train/test | int | 투구 직전 카운트 | 투구 전 | `AVAILABLE` | 수치·상호작용 | 사전 상태 |
| `run_top_before`, `run_bot_before`, `run_total_before` | train/test | numeric | 투구 직전 점수 | 투구 전 | `AVAILABLE` | 수치 피처 | 사전 상태 |
| `score_diff_home`, `score_diff_pitcher_team` | train/test | numeric | 홈·투수팀 기준 점수 차 | 투구 전 | `AVAILABLE` | 수치 피처 | 사전 상태 |
| `runner_on_1b`, `runner_on_2b`, `runner_on_3b` | train/test | binary | 투구 직전 주자 여부 | 투구 전 | `AVAILABLE` | 범주·수치 피처 | 사전 상태 |
| `num_runners_on`, `base_state` | train/test | mixed | 주자 수·루상 조합 | 투구 전 | `AVAILABLE` | 범주·수치 피처 | 서로 파생 관계 |
| `home_win_expectancy`, `away_win_expectancy` | train/test | float | 투구 직전 기대 승률(0~100) | 투구 전 | `AVAILABLE` | 수치 피처 | 두 값의 관계 검사 필요 |
| `li` | train/test | float | 투구 직전 상황 중요도 | 투구 전 | `AVAILABLE` | 수치 피처 | 큰 값일수록 중요 |
| `pitcher_id`, `batter_id` | train/test | int/string | 익명 선수 ID | 투구 전 | `AVAILABLE` | 모델 A에서 제공 코드 그대로 사용, 모델 B에서 제외 | ID 제거 모델과 확률 평균하여 코드 순서 의존 위험 완화 |
| `pitcher_hand`, `batter_hand` | train/test | code | 투타 좌우 유형 | 투구 전 | `AVAILABLE` | 범주 피처 | 코드 의미는 공식 설명 기준 |
| `pitcher_team_id`, `batter_team_id` | train/test | int/string | 소속 팀 ID | 투구 전 | `AVAILABLE` | 모델 A에서 제공 코드 그대로 사용, 모델 B에서 제외 | 시즌별 변화와 코드 순서 의존 위험 완화 |
| `asof_pitcher_n` | train/test | int | 투수의 직전까지 누적 투구 수 | 투구 전 | `HISTORY_ONLY` | 표본 수·신뢰도 | 운영진이 누수 없이 사전 계산 |
| `asof_pitcher_success_rate` | train/test | float | 투수 누적 제구 성공률 | 투구 전 | `HISTORY_ONLY` | 확률 피처 | 표본 0이면 결측 가능 |
| `asof_pitcher_reverse_rate` | train/test | float | 투수 누적 의도 반대성 비율 | 투구 전 | `HISTORY_ONLY` | 확률 피처 | 표본 0이면 결측 가능 |
| `asof_pitcher_middle_rate` | train/test | float | 투수 누적 가운데·위험 코스 비율 | 투구 전 | `HISTORY_ONLY` | 확률 피처 | 표본 0이면 결측 가능 |
| `asof_pitcher_ball_rate`, `asof_pitcher_strike_rate` | train/test | float | 투수 누적 볼성·스트라이크성 비율 | 투구 전 | `HISTORY_ONLY` | 확률 피처 | 표본 0이면 결측 가능 |
| `asof_pitcher_prev1_game_success_rate` | train/test | float | 투수 직전 1경기 성공률 | 투구 전 | `HISTORY_ONLY` | 최근 폼 | cold-start 결측 가능 |
| `asof_pitcher_prev3_game_success_rate` | train/test | float | 투수 직전 3경기 성공률 | 투구 전 | `HISTORY_ONLY` | 최근 폼 | cold-start 결측 가능 |
| `asof_pitcher_prev5_game_success_rate` | train/test | float | 투수 직전 5경기 성공률 | 투구 전 | `HISTORY_ONLY` | 최근 폼 | cold-start 결측 가능 |
| `asof_pitcher_prev1_game_middle_rate` | train/test | float | 투수 직전 1경기 위험 코스 비율 | 투구 전 | `HISTORY_ONLY` | 최근 폼 | cold-start 결측 가능 |
| `asof_pitcher_prev3_game_middle_rate` | train/test | float | 투수 직전 3경기 위험 코스 비율 | 투구 전 | `HISTORY_ONLY` | 최근 폼 | cold-start 결측 가능 |
| `asof_pitcher_prev5_game_middle_rate` | train/test | float | 투수 직전 5경기 위험 코스 비율 | 투구 전 | `HISTORY_ONLY` | 최근 폼 | cold-start 결측 가능 |
| `asof_batter_n` | train/test | int | 타자의 직전까지 상대 투구 수 | 투구 전 | `HISTORY_ONLY` | 표본 수·신뢰도 | 운영진 사전 계산 |
| `asof_batter_success_rate`, `asof_batter_middle_rate` | train/test | float | 타자 상대 투구의 누적 성공·위험 코스 비율 | 투구 전 | `HISTORY_ONLY` | 확률 피처 | 표본 0이면 결측 가능 |
| `asof_pitcher_pitchmix_n` | train/test | int | 투수 구종 사용 이력 표본 수 | 투구 전 | `HISTORY_ONLY` | 표본 수·신뢰도 | 운영진 사전 계산 |
| `asof_pitcher_fastball_rate` | train/test | float | 투수 누적 fastball 사용 비율 | 투구 전 | `HISTORY_ONLY` | 구종 구성 | 표본 0이면 결측 가능 |
| `asof_pitcher_breaking_rate` | train/test | float | 투수 누적 breaking 사용 비율 | 투구 전 | `HISTORY_ONLY` | 구종 구성 | 표본 0이면 결측 가능 |
| `asof_pitcher_offspeed_rate` | train/test | float | 투수 누적 offspeed 사용 비율 | 투구 전 | `HISTORY_ONLY` | 구종 구성 | 표본 0이면 결측 가능 |
| `control_success` | train | binary | 제구 성공 여부 | 현재 투구 종료 후 | `LEAKAGE` | Target 전용 | 학습 레이블 |

상태 값은 `AVAILABLE`, `HISTORY_ONLY`, `LEAKAGE`, `ID_ONLY`, `UNKNOWN` 중 하나를 사용합니다.

## 시간 및 그룹 키

| 목적 | 후보 컬럼 | 정렬/조합 규칙 | 확인 상태 |
|---|---|---|---|
| 전체 시간 순서 | `season`, `game_month`, `game_dayofweek` | 정확한 날짜·경기 순서를 복원할 수 없음 | 제한적 |
| 경기 식별 | 제공되지 않음 | 행만으로 동일 경기를 확정할 수 없음 | 미확인 |
| 경기 내 투구 순서 | 제공되지 않음 | `row_id` 정렬을 시간 순서로 가정하지 않음 | 미확인 |
| 투수 식별 | `pitcher_id` | 익명 ID | 확정 |
| 타자 식별 | `batter_id` | 익명 ID | 확정 |

## 조인 규칙

Trackman 등 보조 데이터를 사용하면 아래를 반드시 기록합니다.

| 보조 데이터 | 조인 키 | 시간 cutoff | 중복 처리 | 미매칭 처리 | 검증 결과 |
|---|---|---|---|---|---|
| 2019~2024 Trackman | 메인 데이터와 직접 대응하는 공식 키 없음 | 2019~2024 로그만 사용, 2025 금지 | 해당 없음 | 해당 없음 | 메인/Trackman 투수·타자 ID 값 교집합이 각각 0이므로 최종 모델에서 제외 |

## Trackman 컬럼 사용 원칙

- 식별·시간·상황: `trackman_id`, `season`, `game_date`, `game_month`, `game_dayofweek`, `trackman_game_id`, `pitch_no`, `inning`, `top_bottom`, `balls_before`, `strikes_before`, `outs_before`, `pitch_of_pa`
- 선수·팀: `pitcher_trackman_id`, `batter_trackman_id`, `pitcher_hand`, `batter_hand`, `pitcher_team`, `batter_team`
- 구종: `tagged_pitch_type`, `auto_pitch_type`, `pitch_type_group`
- 과거 실측: `rel_speed`, `spin_rate`, `induced_vert_break`, `horz_break`, `extension`, `rel_height`, `rel_side`, `zone_speed`

위 컬럼은 모두 2019~2024년의 `HISTORY_ONLY` 보조 정보입니다. 현재 평가 투구와 직접 1:1 결합하거나 현재 투구의 실제 구종·Trackman 값으로 해석하면 안 됩니다. 2025년 Trackman 데이터는 사용 금지입니다.

## 평가 데이터 내부 집계 금지

실제 `test.csv`의 각 행은 독립적으로 예측해야 합니다. 다른 test 행을 이용한 선수·팀·월별 빈도, 누적, 분포, rolling, expanding, target encoding 및 사후 보정값은 사용할 수 없습니다. 운영진이 제공한 `asof_*` 값만 공식 사전 이력으로 사용할 수 있습니다.
