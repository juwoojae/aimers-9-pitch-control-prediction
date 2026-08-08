# script.py
import os
import math

import joblib
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"


def validate_unique_ids(df, data_name):
    """row_id의 결측 및 중복을 검사한다."""
    if df[ID_COL].isna().any():
        raise ValueError(f"{data_name}에 결측 {ID_COL}가 있음")
    duplicated = df[ID_COL].duplicated(keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, ID_COL].head(5).tolist()
        raise ValueError(
            f"{data_name}에 중복 {ID_COL}가 {int(duplicated.sum())}건 있음: "
            f"{examples}"
        )


# =======================
# 데이터 로드 유틸
# =======================

def load_test(path):
    """평가 데이터(csv) 로드. 한 행이 투구 하나."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    if ID_COL not in df.columns:
        raise ValueError(f"test 데이터에 {ID_COL} 컬럼이 없음: {list(df.columns)[:5]}")
    validate_unique_ids(df, "test")
    return df


def load_sample_submission(path):
    """sample_submission.csv 로드 — 제출 파일의 row_id 순서/컬럼 기준."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    if list(df.columns) != [ID_COL, TARGET_COL]:
        raise ValueError(
            f"sample_submission 컬럼이 ({ID_COL}, {TARGET_COL})이 아님: "
            f"{list(df.columns)}")
    validate_unique_ids(df, "sample_submission")
    return df


def validate_input_alignment(test, sub):
    """test와 sample_submission의 행 수 및 row_id 집합이 같은지 검사한다."""
    if len(test) != len(sub):
        raise ValueError(
            f"test와 sample_submission 행 수 불일치: {len(test)} != {len(sub)}"
        )

    test_ids = set(test[ID_COL])
    sub_ids = set(sub[ID_COL])
    if test_ids != sub_ids:
        missing = list(sub_ids - test_ids)[:5]
        extra = list(test_ids - sub_ids)[:5]
        raise ValueError(
            "test와 sample_submission row_id 불일치: "
            f"test에 없는 ID 예시={missing}, submission에 없는 ID 예시={extra}"
        )


# =======================
# 학습 때 사용한 전처리 (그대로)
# =======================

def build_features(df):
    """모델 입력 추출 — 학습 때와 동일하게 row_id만 빼고 전부 사용.

    범주형 인코딩(top_bottom, game_type, base_state)과 결측 대치는
    모델 파일 안의 파이프라인이 함께 수행하므로 여기서는 컬럼만 고른다.
    """
    return df.drop(columns=[ID_COL])


# =======================
# 제출 파일 생성 유틸
# =======================

def validate_predictions(preds, expected_len):
    """예측 개수, 유한성 및 확률 범위를 검사하고 float 목록을 반환한다."""
    if len(preds) != expected_len:
        raise ValueError(f"예측 개수 불일치: {len(preds)} != {expected_len}")

    validated = []
    for index, value in enumerate(preds):
        probability = float(value)
        if not math.isfinite(probability):
            raise ValueError(f"유한하지 않은 예측값: index={index}, value={value}")
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"확률 범위 이탈: index={index}, value={probability}")
        validated.append(probability)
    return validated


def merge_predictions(sub, ids, preds):
    """누락 없이 sample_submission의 row_id 순서에 맞춰 예측 확률을 병합한다."""
    if len(ids) != len(preds):
        raise ValueError(f"ID와 예측 개수 불일치: {len(ids)} != {len(preds)}")
    if len(set(ids)) != len(ids):
        raise ValueError("예측 대상 row_id에 중복이 있음")

    pred_map = dict(zip(ids, preds))
    sub_ids = sub[ID_COL].tolist()
    missing = [rid for rid in sub_ids if rid not in pred_map]
    extra = list(set(pred_map) - set(sub_ids))
    if missing or extra:
        raise ValueError(
            "예측 row_id 불일치: "
            f"누락 {len(missing)}건 {missing[:5]}, 초과 {len(extra)}건 {extra[:5]}"
        )

    sub = sub.copy()
    sub[TARGET_COL] = [pred_map[rid] for rid in sub_ids]
    return sub


def save_submission(path, sub):
    if list(sub.columns) != [ID_COL, TARGET_COL]:
        raise ValueError(f"최종 제출 컬럼 불일치: {list(sub.columns)}")
    validate_unique_ids(sub, "submission")
    validate_predictions(sub[TARGET_COL].tolist(), len(sub))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sub.to_csv(path, index=False, encoding="utf-8")


# =======================
# main
# =======================

def main():
    # ---- 경로 변수 (필요에 따라 수정) ----
    TEST_DIR = "./data"            # test.csv, sample_submission.csv 위치
    MODEL_DIR = "./model"          # rf.pkl 위치
    OUT_DIR = "./output"
    TEST_PATH = os.path.join(TEST_DIR, "test.csv")
    SAMPLE_SUB_PATH = os.path.join(TEST_DIR, "sample_submission.csv")
    MODEL_PATH = os.path.join(MODEL_DIR, "rf.pkl")
    OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

    # ---- 모델 로드 ----
    print("Load model...")
    model = joblib.load(MODEL_PATH)
    print(f" OK. n_features={getattr(model, 'n_features_in_', '?')}")

    # ---- 테스트 데이터 로드 ----
    print("Load test data...")
    test = load_test(TEST_PATH)
    sub = load_sample_submission(SAMPLE_SUB_PATH)
    validate_input_alignment(test, sub)
    print(f" test={len(test)}  submission={len(sub)}")

    # ---- 전처리 (학습과 동일) ----
    print("Build features...")
    ids = test[ID_COL].tolist()
    X = build_features(test)
    print(f" features={X.shape[1]}")

    # ---- 예측 (제구 성공 확률) ----
    print("Inference model...")
    classes = list(getattr(model, "classes_", []))
    if 1 not in classes:
        raise ValueError(f"모델 클래스에 양성 클래스 1이 없음: {classes}")
    positive_class_index = classes.index(1)
    preds = model.predict_proba(X)[:, positive_class_index] if len(X) else []
    preds = validate_predictions(preds, len(test))
    print(f" preds={len(preds)}")

    # ---- sample_submission 기반 결과 생성 ----
    print("Build submission...")
    sub = merge_predictions(sub, ids, preds)
    save_submission(OUT_PATH, sub)
    print(f"Saved: {OUT_PATH} (rows={len(sub)})")


if __name__ == "__main__":
    main()
