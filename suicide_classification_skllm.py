# ============================================================
# Suicide vs Non‑suicide  (speaker_id 단위 Zero‑Shot 분류, NaN 보강)
# ============================================================
import time, warnings, os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from skllm.config import SKLLMConfig
from skllm.models.gpt.classification.zero_shot import ZeroShotGPTClassifier
warnings.filterwarnings("ignore")

# ───────────────────────────── 0. 환경 설정 ─────────────────────────────
OPENAI_KEY = "YOUR_OPENAI_API_KEY_HERE"  # ★ 실제 키로 교체 필요
SKLLMConfig.set_openai_key(OPENAI_KEY)

BASE_DIRS = [
    r"0504/186.복지 결과/1.Training/03.정신건강복지센터/02.자살위기개입",
    r"0504/186.복지 결과/1.Training/01.대학병원/02.병원이용안내",
    r"0504/복지결과 Wav2Vec2/1.Training/01.대학병원/01.진료안내",
]

SAVE_DIR        = Path("B_results"); SAVE_DIR.mkdir(parents=True, exist_ok=True)
checkpoint_path = SAVE_DIR / "checkpoint_full_prediction.csv"
checkpoint_step = 100          # speaker 100명마다 저장

# ───────────────────────────── 1. CSV 로드 & speaker 단위 병합 ─────────────────────────────
raw_rows = []
for base in BASE_DIRS:
    for csv_path in Path(base).glob("*.csv"):
        df = pd.read_csv(csv_path, encoding="utf-8-sig")

        # 텍스트 컬럼 자동 탐색
        txt_col = next((c for c in ["combined_pred_text", "pred_text", "gt_text"]
                        if c in df.columns), None)
        if txt_col is None:
            raise ValueError(f"❌ 텍스트 컬럼을 찾을 수 없음: {csv_path}")

        if "speaker_id" not in df.columns:
            df["speaker_id"] = df.get("file_id", range(len(df)))

        # ★ NaN → "" , 숫자 → str 변환
        df[txt_col] = df[txt_col].fillna("").astype(str)
        if "label" not in df.columns:
            df["label"] = "nonsuicide"
        else:
            df["label"] = df["label"].fillna("nonsuicide").astype(str)

        df = df[["speaker_id", txt_col, "label"]].rename(columns={txt_col: "text"})
        raw_rows.append(df)

raw_df = pd.concat(raw_rows, ignore_index=True)

# ★ 안전한 join : NaN 제거 후 str 캐스팅
combined_df = (
    raw_df
    .groupby("speaker_id", as_index=False)
    .agg({
        "text": lambda x: " ".join(x.dropna().astype(str)),
        "label": "first"
    })
)
print(f"✅ 전체 speaker 수: {len(combined_df):,}명 로드 완료")

# ───────────────────────────── 2. 모델 준비 ─────────────────────────────
clf = ZeroShotGPTClassifier(model="gpt-4o-mini")
clf.fit(None, ["nonsuicide", "suicide"])

# ───────────────────────────── 3. 체크포인트 복구 ─────────────────────────────
predictions, true_labels, speaker_ids = [], [], []
start_idx = 0
if checkpoint_path.exists():
    cp_df      = pd.read_csv(checkpoint_path, encoding="utf-8-sig")
    start_idx  = len(cp_df)
    predictions = cp_df["predicted_label"].tolist()
    true_labels = cp_df["true_label"].tolist()
    speaker_ids = cp_df["speaker_id"].tolist()
    print(f"🔄 체크포인트 발견 – {start_idx}명부터 이어서 진행")
else:
    print("▶️ 새로 실행 시작")

# ───────────────────────────── 4. 예측 루프 ─────────────────────────────
texts        = combined_df["text"].tolist()
labels       = combined_df["label"].tolist()
speaker_list = combined_df["speaker_id"].tolist()

total = len(texts)
print(f"🚀 분류 시작: {start_idx}/{total}")
t0 = time.perf_counter()

for idx in tqdm(range(start_idx, total), miniters=1):
    try:
        pred = clf.predict([texts[idx]])[0]
    except Exception as e:
        print(f"⚠️ {idx+1}번째 speaker 예측 오류 → {e}")
        time.sleep(5)
        continue

    predictions.append(pred)
    true_labels.append(labels[idx])
    speaker_ids.append(speaker_list[idx])

    if (idx + 1) % checkpoint_step == 0 or (idx + 1) == total:
        pd.DataFrame({
            "speaker_id": speaker_ids,
            "true_label": true_labels,
            "predicted_label": predictions
        }).to_csv(checkpoint_path, index=False, encoding="utf-8-sig")
        print(f"💾 체크포인트 저장: {idx+1}/{total}")

print(f"⏱ 총 소요: {(time.perf_counter()-t0)/60:.1f}분")

# ───────────────────────────── 5. 결과 저장 ─────────────────────────────
full_df = pd.DataFrame({
    "speaker_id": speaker_ids,
    "true_label": true_labels,
    "predicted_label": predictions
})
full_csv = SAVE_DIR / "full_pred_suicide_vs_nonsuicide.csv"
full_df.to_csv(full_csv, index=False, encoding="utf-8-sig")
print("✅ 전체 결과 저장:", full_csv)

# ───────────────────────────── 6. 평가 & 시각화 ─────────────────────────────
print("\n📈 Accuracy:", accuracy_score(true_labels, predictions))
print("\n📋 Report:\n", classification_report(true_labels, predictions, digits=3))

cm = confusion_matrix(true_labels, predictions, labels=["nonsuicide", "suicide"])
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["nonsuicide", "suicide"],
            yticklabels=["nonsuicide", "suicide"])
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
plt.tight_layout(); plt.show()