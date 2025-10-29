# ============================================================
# Wav2Vec2 배치 전사 + WER/CER 평가 (복지 콜센터 데이터 전용)
# ============================================================
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch, os, json, jiwer, pandas as pd, librosa, time, re
from tqdm.auto import tqdm
from pathlib import Path
from typing import List

CONFIG = dict(
    MODEL_ID      = "kresnik/wav2vec2-large-xls-r-300m-korean",
    DEVICE        = "cuda" if torch.cuda.is_available() else "cpu",
    BATCH_SIZE    = 4,
    BASE_DATA_DIR = Path(r"D:/OneDrive/186.복지 분야 콜센터 상담데이터/186.복지 분야 콜센터 상담데이터/01.데이터"),
    DATASET_TYPES = ["1.Training"],
    TARGET_CATS   = ["01.대학병원"],
    TARGET_CONDS  = ["01.진료안내"],  # 
    SAVE_DIR      = Path(r"D:/OneDrive/복지결과 Wav2Vec2/")
)
# 모델 로드
proc = Wav2Vec2Processor.from_pretrained(CONFIG["MODEL_ID"])
model = Wav2Vec2ForCTC.from_pretrained(CONFIG["MODEL_ID"]).to(CONFIG["DEVICE"]).eval()

# 배치 전사 함수
def transcribe_batch(wavs: List[Path]) -> List[str]:
    waves = []
    for p in wavs:
        try:
            audio, _ = librosa.load(p, sr=16000)
            waves.append(audio)
        except Exception as e:
            print(f"❗ 오디오 로딩 실패: {p} -> {e}")
            waves.append(None)

    valid_idx = [i for i, w in enumerate(waves) if w is not None]
    if not valid_idx:
        return [""] * len(wavs)

    inputs = proc([waves[i] for i in valid_idx],
                  sampling_rate=16000,
                  return_tensors="pt",
                  padding=True).input_values.to(CONFIG["DEVICE"])

    with torch.no_grad():
        logits = model(inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)

    texts = proc.batch_decode(pred_ids, skip_special_tokens=True)
    out = [""] * len(wavs)
    for k, i in enumerate(valid_idx):
        out[i] = texts[k].strip()
    return out

# JSON 텍스트 추출 함수
def json_text(path: Path) -> str:
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        if "utterances" in data:
            texts = [u.get("text", "") for u in data["utterances"]]
        elif "inputText" in data:
            texts = [t["orgtext"] for t in data["inputText"]]
        else:
            texts = re.findall(r'"text"\s*:\s*"(.+?)"', json.dumps(data))

        return " ".join(texts).strip()

    except Exception as e:
        print(f"❗ JSON 파싱 실패: {path} -> {e}")
        return ""

# 메인 루프
for ds_type in CONFIG["DATASET_TYPES"]:
    base_lbl = CONFIG["BASE_DATA_DIR"] / ds_type / "라벨링데이터"
    base_org = CONFIG["BASE_DATA_DIR"] / ds_type / "원천데이터"

    for cat_dir in (base_lbl.iterdir() if not CONFIG["TARGET_CATS"]
                    else [base_lbl / c for c in CONFIG["TARGET_CATS"]]):
        if not cat_dir.exists(): continue
        cat = cat_dir.name

        for cond_dir in cat_dir.iterdir():
            if not cond_dir.is_dir(): continue
            cond1 = cond_dir.name
            if CONFIG["TARGET_CONDS"] and cond1 not in CONFIG["TARGET_CONDS"]:
                continue

            for subcond_dir in cond_dir.iterdir():
                if not subcond_dir.is_dir(): continue
                cond2 = subcond_dir.name
                cond_path = f"{cond1}/{cond2}"

                print(f"\n🚀 [{ds_type}] {cat}/{cond_path}")
                records, wav_buf, json_buf, meta_buf = [], [], [], []

                for spk_dir in tqdm(list(subcond_dir.iterdir()),
                                    desc=cond_path, leave=False):
                    for jpath in spk_dir.glob("*.json"):
                        fid = jpath.stem
                        wpath = (base_org / cat / cond1 / cond2 /
                                 spk_dir.name / f"{fid}.wav")
                        if not wpath.exists():
                            continue

                        wav_buf.append(wpath)
                        json_buf.append(jpath)
                        meta_buf.append((fid, spk_dir.name))

                        if len(wav_buf) == CONFIG["BATCH_SIZE"]:
                            preds = transcribe_batch(wav_buf)
                            for (fid_, spk_), jp, pr in zip(meta_buf, json_buf, preds):
                                gt = json_text(jp)
                                records.append(dict(
                                    file_id=fid_, category=cat,
                                    condition=cond_path, speaker_id=spk_,
                                    gt_text=gt, pred_text=pr,
                                    WER=jiwer.wer(gt, pr) if gt else None,
                                    CER=jiwer.cer(gt, pr) if gt else None
                                ))
                            wav_buf, json_buf, meta_buf = [], [], []

                if wav_buf:
                    preds = transcribe_batch(wav_buf)
                    for (fid_, spk_), jp, pr in zip(meta_buf, json_buf, preds):
                        gt = json_text(jp)
                        records.append(dict(
                            file_id=fid_, category=cat,
                            condition=cond_path, speaker_id=spk_,
                            gt_text=gt, pred_text=pr,
                            WER=jiwer.wer(gt, pr) if gt else None,
                            CER=jiwer.cer(gt, pr) if gt else None
                        ))

                if not records:
                    print("⚠ 데이터 없음 — 건너뜀")
                    continue

                df = pd.DataFrame(records)
                df.loc[len(df)] = dict(file_id="AVG", category="",
                                       condition="", speaker_id="",
                                       gt_text="", pred_text="",
                                       WER=round(df["WER"].mean(skipna=True), 4),
                                       CER=round(df["CER"].mean(skipna=True), 4))

                save_path = CONFIG["SAVE_DIR"] / ds_type / cat / cond1
                save_path.mkdir(parents=True, exist_ok=True)
                out_csv = save_path / f"{cond2}.csv"
                df.to_csv(out_csv, index=False)
                print(f"✔ {out_csv}  (샘플 {len(df)-1},  평균 "
                      f"WER={df['WER'].iloc[-1]}, CER={df['CER'].iloc[-1]})")