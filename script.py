"""DACON 코드 제출 서버가 실행하는 오프라인 추론 진입점.

학습은 제출 전에 ``train.py``로 완료한다. 이 스크립트는 평가 데이터로
재학습하거나 통계를 계산하지 않고, ``model/rf.pkl``에 저장된 두 HGB의
``control_success=1`` 확률을 50:50으로 평균해 제출 파일만 생성한다.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

# 공식 서버의 6 vCPU를 넘겨 불필요한 스레드 경쟁이 생기지 않게 제한한다.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "6")
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


ID_COL = "row_id"
TARGET_COL = "control_success"
EXPECTED_ARTIFACT_VERSION = "hgb_ensemble_v1"
ROOT_DIR = Path(__file__).resolve().parent


def validate_unique_ids(frame: pd.DataFrame, data_name: str) -> None:
    """누락되거나 중복된 제출 키를 추론 전에 차단한다."""
    if ID_COL not in frame.columns:
        raise ValueError(f"{data_name} is missing required column: {ID_COL}")
    if frame[ID_COL].isna().any():
        raise ValueError(f"{data_name} contains missing {ID_COL} values")
    duplicated = frame[ID_COL].duplicated(keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, ID_COL].head(5).tolist()
        raise ValueError(
            f"{data_name} contains {int(duplicated.sum())} duplicated IDs: {examples}"
        )


def load_test(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    validate_unique_ids(frame, "test")
    return frame


def load_sample_submission(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if list(frame.columns) != [ID_COL, TARGET_COL]:
        raise ValueError(
            "sample_submission columns must be exactly "
            f"[{ID_COL!r}, {TARGET_COL!r}], got {list(frame.columns)}"
        )
    validate_unique_ids(frame, "sample_submission")
    return frame


def resolve_input_dir(root: Path) -> Path:
    """공식 안내에 혼용된 data/와 open/ 입력 경로를 모두 지원한다."""
    for directory_name in ("data", "open"):
        candidate = root / directory_name
        if (candidate / "test.csv").is_file() and (
            candidate / "sample_submission.csv"
        ).is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find test.csv and sample_submission.csv under data/ or open/"
    )


def validate_input_alignment(test: pd.DataFrame, submission: pd.DataFrame) -> None:
    if len(test) != len(submission):
        raise ValueError(
            f"test/submission row count mismatch: {len(test)} != {len(submission)}"
        )
    test_ids = set(test[ID_COL])
    submission_ids = set(submission[ID_COL])
    if test_ids != submission_ids:
        missing = list(submission_ids - test_ids)[:5]
        extra = list(test_ids - submission_ids)[:5]
        raise ValueError(
            "test/submission row_id mismatch: "
            f"missing_in_test={missing}, extra_in_test={extra}"
        )


def load_artifact(path: Path) -> dict:
    # 아티팩트에는 학습이 끝난 전처리기와 HGB 두 개가 함께 들어 있다.
    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise TypeError("model artifact must be a dictionary")
    if artifact.get("format_version") != EXPECTED_ARTIFACT_VERSION:
        raise ValueError(
            "unsupported model artifact version: "
            f"{artifact.get('format_version')!r}"
        )
    models = artifact.get("models")
    weights = artifact.get("weights")
    if not models or not weights or len(models) != len(weights):
        raise ValueError("model artifact has invalid models/weights")
    if not np.isfinite(np.asarray(weights, dtype=np.float64)).all():
        raise ValueError("model weights contain NaN or infinity")
    if not math.isclose(float(sum(weights)), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"model weights must sum to 1, got {sum(weights)}")
    return artifact


def build_features(test: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """학습 당시 컬럼과 평가 컬럼이 정확히 같은지 검사하고 순서를 맞춘다."""
    features = test.drop(columns=[ID_COL])
    expected = list(artifact["input_columns"])
    missing = [column for column in expected if column not in features.columns]
    extra = [column for column in features.columns if column not in expected]
    if missing or extra:
        raise ValueError(
            f"test feature schema mismatch: missing={missing}, extra={extra}"
        )
    return features[expected]


def predict_ensemble(artifact: dict, features: pd.DataFrame) -> np.ndarray:
    """두 모델의 양성 클래스 확률을 저장된 가중치로 평균한다."""
    if len(features) == 0:
        return np.empty(0, dtype=np.float64)
    probability = np.zeros(len(features), dtype=np.float64)
    positive_class = artifact["positive_class"]
    for member, weight in zip(artifact["models"], artifact["weights"]):
        model = member["model"]
        classes = list(model.classes_)
        if positive_class not in classes:
            raise ValueError(
                f"{member['name']} is missing positive class {positive_class}: {classes}"
            )
        member_features = features[member["feature_columns"]]
        # 평가 데이터에는 predict_proba()만 호출하며 fit/update는 수행하지 않는다.
        member_probability = model.predict_proba(member_features)[
            :, classes.index(positive_class)
        ]
        probability += float(weight) * member_probability
    return probability


def validate_predictions(predictions, expected_length: int) -> np.ndarray:
    """제출 직전 예측 개수, 유한성, 확률 범위를 검사한다."""
    probability = np.asarray(predictions, dtype=np.float64)
    if probability.ndim != 1 or len(probability) != expected_length:
        raise ValueError(
            f"prediction shape mismatch: {probability.shape} != ({expected_length},)"
        )
    if not np.isfinite(probability).all():
        bad = np.flatnonzero(~np.isfinite(probability))[:5].tolist()
        raise ValueError(f"predictions contain NaN or infinity at indices: {bad}")
    outside = (probability < 0.0) | (probability > 1.0)
    if outside.any():
        bad = np.flatnonzero(outside)[:5].tolist()
        raise ValueError(f"predictions are outside [0, 1] at indices: {bad}")
    return probability


def merge_predictions(
    submission: pd.DataFrame, ids, predictions
) -> pd.DataFrame:
    """test의 예측을 sample_submission의 row_id 순서에 맞춰 결합한다."""
    if len(ids) != len(predictions):
        raise ValueError(f"ID/prediction count mismatch: {len(ids)} != {len(predictions)}")
    if len(set(ids)) != len(ids):
        raise ValueError("prediction IDs contain duplicates")
    prediction_map = dict(zip(ids, predictions))
    submission_ids = submission[ID_COL].tolist()
    missing = [row_id for row_id in submission_ids if row_id not in prediction_map]
    extra = list(set(prediction_map) - set(submission_ids))
    if missing or extra:
        raise ValueError(
            f"prediction ID mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    result = submission.copy()
    result[TARGET_COL] = [prediction_map[row_id] for row_id in submission_ids]
    return result


def save_submission(path: Path, submission: pd.DataFrame) -> None:
    """최종 스키마를 한 번 더 검사한 뒤 UTF-8 CSV로 저장한다."""
    if list(submission.columns) != [ID_COL, TARGET_COL]:
        raise ValueError(f"invalid final columns: {list(submission.columns)}")
    validate_unique_ids(submission, "submission")
    validate_predictions(submission[TARGET_COL], len(submission))
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False, encoding="utf-8")


def main() -> None:
    input_dir = resolve_input_dir(ROOT_DIR)
    test_path = input_dir / "test.csv"
    sample_path = input_dir / "sample_submission.csv"
    model_path = ROOT_DIR / "model" / "rf.pkl"
    output_path = ROOT_DIR / "output" / "submission.csv"

    # 서버에서는 입력 디렉터리만 교체되며 모델 학습은 수행하지 않는다.
    print("모델 아티팩트 로드...")
    artifact = load_artifact(model_path)
    print("평가 입력 로드 및 검증...")
    test = load_test(test_path)
    submission = load_sample_submission(sample_path)
    validate_input_alignment(test, submission)
    features = build_features(test, artifact)

    print(f"오프라인 앙상블 추론: rows={len(test)}, models={len(artifact['models'])}")
    with threadpool_limits(limits=6):
        predictions = predict_ensemble(artifact, features)
    predictions = validate_predictions(predictions, len(test))

    result = merge_predictions(submission, test[ID_COL].tolist(), predictions)
    save_submission(output_path, result)
    print(f"제출 파일 저장: {output_path} (rows={len(result)})")


if __name__ == "__main__":
    main()
