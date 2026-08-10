"""시즌 순방향 검증과 최종 HGB 앙상블 학습을 재현하는 단일 진입점.

검증과 최종 학습 모두 ``data/train.csv``만 읽는다. 평가용 ``test.csv``는
피처 선택, 결측치 대체, 인코딩, 모델 학습에 사용하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


ROOT_DIR = Path(__file__).resolve().parent
ID_COL = "row_id"
TARGET_COL = "control_success"
SEASON_COL = "season"
SEED = 42
ARTIFACT_VERSION = "hgb_ensemble_v1"

# 문자열/코드값 자체에 순서 의미를 주지 않고, 학습 구간에서만 ordinal encoding한다.
LOW_CARDINALITY_CATEGORICALS = ("top_bottom", "game_type", "base_state")

# 모델 B에서는 익명 ID 코드의 임의 순서와 시즌별 선수 변화에 대한 의존을 줄인다.
ENTITY_ID_COLUMNS = (
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
)

# 2022·2024 시즌 순방향 holdout에서 선택한 최종 두 설정이다.
MODEL_CONFIGS = {
    "full_leaf31_s42": {
        "drop": (),
        "max_leaf_nodes": 31,
        "min_samples_leaf": 200,
        "l2_regularization": 3.0,
    },
    "no_ids_leaf15_s42": {
        "drop": ENTITY_ID_COLUMNS,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 500,
        "l2_regularization": 5.0,
    },
}
MODEL_WEIGHTS = (0.5, 0.5)


def load_train(path: Path) -> pd.DataFrame:
    """학습 데이터의 최소 스키마와 Target 무결성을 검사한다."""
    train = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    required = {ID_COL, TARGET_COL, SEASON_COL, *LOW_CARDINALITY_CATEGORICALS}
    missing = sorted(required - set(train.columns))
    if missing:
        raise ValueError(f"학습 데이터에 필수 컬럼이 없음: {missing}")
    if train.empty:
        raise ValueError("학습 데이터가 비어 있음")
    if train[ID_COL].isna().any() or train[ID_COL].duplicated().any():
        raise ValueError("row_id에 결측값 또는 중복값이 있음")
    if train[TARGET_COL].isna().any() or not train[TARGET_COL].isin([0, 1]).all():
        raise ValueError("control_success는 결측 없는 0/1 값이어야 함")
    return train


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """식별자와 현재 투구 결과인 Target을 모델 입력에서 제외한다."""
    return [column for column in frame.columns if column not in {ID_COL, TARGET_COL}]


def build_model(
    frame: pd.DataFrame, config: dict[str, object]
) -> tuple[Pipeline, list[str]]:
    """전처리와 HGB를 묶어 fold 학습 데이터에만 fit되도록 한다."""
    dropped = set(config["drop"])
    columns = [column for column in feature_columns(frame) if column not in dropped]
    categorical = [
        column for column in LOW_CARDINALITY_CATEGORICALS if column in columns
    ]
    numeric = [column for column in columns if column not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical,
            ),
            # 중앙값은 Pipeline.fit()을 호출한 학습 구간에서만 계산된다.
            ("numeric", SimpleImputer(strategy="median"), numeric),
        ],
        remainder="drop",
    )
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=int(config["max_leaf_nodes"]),
        min_samples_leaf=int(config["min_samples_leaf"]),
        l2_regularization=float(config["l2_regularization"]),
        max_bins=255,
        # 외부 holdout과 별개로 학습 구간 내부 10%만 조기 종료에 사용한다.
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        scoring="neg_brier_score",
        random_state=SEED,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)]), columns


def positive_class_probability(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    """control_success=1에 해당하는 확률 열을 안전하게 꺼낸다."""
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError(f"학습 모델에 양성 클래스 1이 없음: {classes}")
    probability = model.predict_proba(frame)[:, classes.index(1)].astype(np.float64)
    if not np.isfinite(probability).all():
        raise ValueError("예측 확률에 NaN 또는 무한값이 있음")
    return np.clip(probability, 0.0, 1.0)


def brier_metrics(
    y_true: Iterable[float], y_probability: Iterable[float]
) -> dict[str, float]:
    """대회 공식 정의와 같은 Brier Skill Score 및 보조 지표를 계산한다."""
    y = np.asarray(y_true, dtype=np.float64)
    probability = np.asarray(y_probability, dtype=np.float64)
    if y.ndim != 1 or probability.ndim != 1 or len(y) != len(probability):
        raise ValueError("정답과 예측은 길이가 같은 1차원 배열이어야 함")
    if len(y) == 0:
        raise ValueError("빈 검증 데이터는 평가할 수 없음")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("예측 확률은 유한한 [0, 1] 값이어야 함")

    target_rate = float(y.mean())
    brier = float(np.mean(np.square(probability - y)))
    reference = target_rate * (1.0 - target_rate)
    if reference <= 0.0:
        raise ValueError("검증 Target이 단일 클래스여서 기준 Brier를 계산할 수 없음")
    bss = max(0.0, 100000.0 * (1.0 - brier / reference))
    return {
        "brier": brier,
        "brier_reference": reference,
        "brier_skill_score": bss,
        "target_rate": target_rate,
        "prediction_mean": float(probability.mean()),
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_validation(args: argparse.Namespace) -> None:
    """미래 시즌을 학습에서 제외한 순방향 holdout을 실행한다."""
    started = time.perf_counter()
    train = load_train(args.train)
    years = [int(value.strip()) for value in args.holdout_years.split(",")]
    results: list[dict[str, object]] = []

    for holdout_year in years:
        fit_mask = train[SEASON_COL] < holdout_year
        validation_mask = train[SEASON_COL] == holdout_year
        if not fit_mask.any() or not validation_mask.any():
            raise ValueError(f"holdout {holdout_year}의 학습 또는 검증 행이 없음")

        probabilities: list[np.ndarray] = []
        fit_seconds: list[float] = []
        y_validation = train.loc[validation_mask, TARGET_COL].to_numpy()

        for name, config in MODEL_CONFIGS.items():
            model, columns = build_model(train, config)
            fit_started = time.perf_counter()
            # encoder와 imputer도 이 fit_mask 범위에서만 학습된다.
            model.fit(train.loc[fit_mask, columns], train.loc[fit_mask, TARGET_COL])
            elapsed = time.perf_counter() - fit_started
            probability = positive_class_probability(
                model, train.loc[validation_mask, columns]
            )
            probabilities.append(probability)
            fit_seconds.append(elapsed)
            results.append(
                {
                    "holdout_year": holdout_year,
                    "model": name,
                    "train_rows": int(fit_mask.sum()),
                    "validation_rows": int(validation_mask.sum()),
                    "feature_count": len(columns),
                    "fit_seconds": elapsed,
                    **brier_metrics(y_validation, probability),
                }
            )

        ensemble_probability = sum(
            weight * probability
            for weight, probability in zip(MODEL_WEIGHTS, probabilities)
        )
        ensemble_result = {
            "holdout_year": holdout_year,
            "model": "hgb_ensemble",
            "train_rows": int(fit_mask.sum()),
            "validation_rows": int(validation_mask.sum()),
            "fit_seconds": float(sum(fit_seconds)),
            **brier_metrics(y_validation, ensemble_probability),
        }
        results.append(ensemble_result)
        print(
            f"holdout={holdout_year} ensemble "
            f"BSS={ensemble_result['brier_skill_score']:.2f}"
        )

    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "seed": SEED,
        "split": "train=season < holdout, validation=season == holdout",
        "holdout_years": years,
        "feature_set": "row_id/target 제외, 공식 사전 상황 및 asof_* 피처",
        "model_configs": MODEL_CONFIGS,
        "weights": MODEL_WEIGHTS,
        "results": results,
        "total_seconds": time.perf_counter() - started,
    }
    write_json(args.report, report)
    print(f"검증 기록 저장: {args.report}")


def fit_final(args: argparse.Namespace) -> None:
    """선택한 두 모델을 전체 2019~2024 학습 데이터로 재학습한다."""
    started = time.perf_counter()
    train = load_train(args.train)
    input_columns = feature_columns(train)
    trained_models = []
    fit_seconds: dict[str, float] = {}

    for name, config in MODEL_CONFIGS.items():
        model, columns = build_model(train, config)
        fit_started = time.perf_counter()
        model.fit(train[columns], train[TARGET_COL])
        fit_seconds[name] = time.perf_counter() - fit_started
        trained_models.append(
            {"name": name, "feature_columns": columns, "model": model}
        )
        print(f"학습 완료: {name} ({fit_seconds[name]:.1f}초)")

    metadata = {
        "seed": SEED,
        "train_rows": len(train),
        "train_seasons": sorted(int(value) for value in train[SEASON_COL].unique()),
        "target_rate": float(train[TARGET_COL].mean()),
        "selected_by": "2022 and 2024 forward-season holdout BSS",
        "calibration": "none",
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }
    artifact = {
        "format_version": ARTIFACT_VERSION,
        "positive_class": 1,
        "input_columns": input_columns,
        "models": trained_models,
        "weights": list(MODEL_WEIGHTS),
        "metadata": metadata,
    }
    args.model.parent.mkdir(parents=True, exist_ok=True)
    # rf.pkl은 운영진 베이스라인과의 파일명 호환을 위한 이름이며 RF 모델이 아니다.
    joblib.dump(artifact, args.model, compress=3, protocol=5)

    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        **metadata,
        "models": list(MODEL_CONFIGS),
        "model_configs": MODEL_CONFIGS,
        "weights": MODEL_WEIGHTS,
        "feature_counts": {
            member["name"]: len(member["feature_columns"])
            for member in trained_models
        },
        "fit_seconds": fit_seconds,
        "total_seconds": time.perf_counter() - started,
        "artifact_bytes": args.model.stat().st_size,
        "artifact_path": str(args.model),
    }
    write_json(args.report, report)
    print(f"모델 저장: {args.model}")
    print(f"학습 기록 저장: {args.report}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate", help="시즌 순방향 holdout으로 최종 구성을 검증"
    )
    validate_parser.add_argument(
        "--train", type=Path, default=ROOT_DIR / "data" / "train.csv"
    )
    validate_parser.add_argument("--holdout-years", default="2022,2023,2024")
    validate_parser.add_argument(
        "--report", type=Path, default=ROOT_DIR / "model" / "validation_report.json"
    )
    validate_parser.set_defaults(handler=run_validation)

    fit_parser = subparsers.add_parser(
        "fit", help="전체 학습 데이터로 제출용 모델을 생성"
    )
    fit_parser.add_argument(
        "--train", type=Path, default=ROOT_DIR / "data" / "train.csv"
    )
    fit_parser.add_argument(
        "--model", type=Path, default=ROOT_DIR / "model" / "rf.pkl"
    )
    fit_parser.add_argument(
        "--report", type=Path, default=ROOT_DIR / "model" / "training_report.json"
    )
    fit_parser.set_defaults(handler=fit_final)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    arguments.handler(arguments)
