"""
Whisper-tiny 배치 전사 + WER/CER 평가 (복지 콜센터 데이터 전용)

🗂️  기대 폴더 구조  ─────────────────────────────────────────────
<BASE_DATA_DIR>/
 └─ <dataset_type>/                     # 1.Training, 2.Validation …
     ├─ 라벨링데이터/
     │   └─ <category>/<condition>/<speaker_id>/<file_id>.json
     └─ 원천데이터/
         └─ <category>/<condition>/<speaker_id>/<file_id>.wav
※ 예)  .../1.Training/라벨링데이터/01.대학병원/01.진료안내/01.검사/HOS…json
"""

# ────────────────── 0. 기본 설정 ──────────────────────────
from pathlib import Path
from typing  import List
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch, librosa, jiwer, pandas as pd, json, re
from tqdm.auto import tqdm

CONFIG = dict(
    MODEL_ID      = "openai/whisper-tiny",
    DEVICE        = "cuda" if torch.cuda.is_available() else "cpu",
    BATCH_SIZE    = 3,                   # GPU 사양에 맞게 조정
    BASE_DATA_DIR = Path(r"D:/OneDrive/186.복지 분야 콜센터 상담데이터/186.복지 분야 콜센터 상담데이터/01.데이터"),
    DATASET_TYPES = ["1.Training"],      # 필요하면 "2.Validation" 추가
    TARGET_CATS   = ["01.대학병원"],     # []면 모든 카테고리
    TARGET_CONDS  = ["01.검사"],                  # []면 모든 조건
    SAVE_DIR      = Path(r"D:/OneDrive/186.복지 결과"),
    EXPECTED_LEN  = 3_000,               # Whisper mel-spectrogram 길이
)

# ────────────────── 1. 모델 로드 ──────────────────────────
proc  = WhisperProcessor.from_pretrained(CONFIG["MODEL_ID"])
model = WhisperForConditionalGeneration.from_pretrained(
            CONFIG["MODEL_ID"],
            torch_dtype=torch.float16
        ).to(CONFIG["DEVICE"]).eval()

decode_prompt = proc.get_decoder_prompt_ids(language="ko", task="transcribe")

# ────────────────── 2. 유틸 함수 ──────────────────────────
def pad_or_trim(feats, target: int):
    diff = target - feats.shape[-1]
    return (torch.nn.functional.pad(feats, (0, diff)) if diff > 0
            else feats[..., :target])

@torch.inference_mode()
def transcribe_batch(wavs: List[Path]) -> List[str]:
    waves = []
    for p in wavs:
        try:
            audio, _ = librosa.load(p, sr=16_000)
            waves.append(audio)
        except Exception as e:
            print(f"❗ 오디오 로딩 실패: {p} -> {e}")
            waves.append(None)

    valid_idx = [i for i, w in enumerate(waves) if w is not None]
    if not valid_idx:
        return [""] * len(wavs)

    feats = proc([waves[i] for i in valid_idx],
                 sampling_rate=16_000,
                 return_tensors="pt",
                 padding=True).input_features.half()
    feats = pad_or_trim(feats, CONFIG["EXPECTED_LEN"]).to(CONFIG["DEVICE"])

    ids   = model.generate(feats, num_beams=1, forced_decoder_ids=decode_prompt)
    texts = proc.batch_decode(ids, skip_special_tokens=True)

    out = [""] * len(wavs)
    for k, i in enumerate(valid_idx):
        out[i] = texts[k].strip()
    return out

def json_text(path: Path) -> str:
    """
    186 데이터셋 JSON 구조
        {
          "utterances": [
              { "text": "안녕하세요~", ... },
              ...
          ]
        }
    또는 "inputText" 키가 있을 수도 있어 둘 다 처리.
    """
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        if "utterances" in data:                      # 186 데이터
            texts = [u.get("text", "") for u in data["utterances"]]
        elif "inputText" in data:                     # 기존 mentalhealth 포맷
            texts = [t["orgtext"] for t in data["inputText"]]
        else:                                         # 예외 처리
            texts = re.findall(r'"text"\s*:\s*"(.+?)"', json.dumps(data))

        return " ".join(texts).strip()

    except Exception as e:
        print(f"❗ JSON 파싱 실패: {path} -> {e}")
        return ""

# ────────────────── 3. 메인 루프 ──────────────────────────
for ds_type in CONFIG["DATASET_TYPES"]:
    base_lbl = CONFIG["BASE_DATA_DIR"] / ds_type / "라벨링데이터"
    base_org = CONFIG["BASE_DATA_DIR"] / ds_type / "원천데이터"

    for cat_dir in (base_lbl.iterdir() if not CONFIG["TARGET_CATS"]
                    else [base_lbl / c for c in CONFIG["TARGET_CATS"]]):
        if not cat_dir.exists(): continue
        cat = cat_dir.name

        # ── 1차: 조건(예: 01.진료안내) ─────────────────────
        for cond_dir in cat_dir.iterdir():
            if not cond_dir.is_dir(): continue
            cond1 = cond_dir.name
            if CONFIG["TARGET_CONDS"] and cond1 not in CONFIG["TARGET_CONDS"]:
                continue

            # ★ 2차: **세부 조건(예: 01.검사, 02.입원…) 추가 루프**
            for subcond_dir in cond_dir.iterdir():
                if not subcond_dir.is_dir(): continue
                cond2 = subcond_dir.name          # 세부 조건 이름
                cond_path = f"{cond1}/{cond2}"    # 로그·CSV용 전체 조건명

                print(f"\n🚀 [{ds_type}] {cat}/{cond_path}")
                records, wav_buf, json_buf, meta_buf = [], [], [], []

                # ── 3차: speaker_id ─────────────────────────
                for spk_dir in tqdm(list(subcond_dir.iterdir()),
                                    desc=cond_path, leave=False):
                    for jpath in spk_dir.glob("*.json"):
                        fid   = jpath.stem
                        wpath = (base_org / cat / cond1 / cond2 /
                                 spk_dir.name / f"{fid}.wav")
                        if not wpath.exists():
                            continue

                        wav_buf.append(wpath)
                        json_buf.append(jpath)
                        meta_buf.append((fid, spk_dir.name))

                        if len(wav_buf) == CONFIG["BATCH_SIZE"]:
                            preds = transcribe_batch(wav_buf)
                            for (fid_, spk_), jp, pr in zip(meta_buf,
                                                            json_buf, preds):
                                gt = json_text(jp)
                                records.append(dict(
                                    file_id=fid_, category=cat,
                                    condition=cond_path, speaker_id=spk_,
                                    gt_text=gt, pred_text=pr,
                                    WER=jiwer.wer(gt, pr) if gt else None,
                                    CER=jiwer.cer(gt, pr) if gt else None
                                ))
                            wav_buf, json_buf, meta_buf = [], [], []

                # ── 남은 배치 처리 ───────────────────────────
                if wav_buf:
                    preds = transcribe_batch(wav_buf)
                    for (fid_, spk_), jp, pr in zip(meta_buf,
                                                    json_buf, preds):
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

                # ── CSV 저장 (cond1 디렉터리 안에 cond2.csv) ─────
                df = pd.DataFrame(records)
                df.loc[len(df)] = dict(file_id="AVG", category="",
                                       condition="", speaker_id="",
                                       gt_text="", pred_text="",
                                       WER=round(df["WER"].mean(skipna=True),4),
                                       CER=round(df["CER"].mean(skipna=True),4))

                save_path = (CONFIG["SAVE_DIR"] / ds_type / cat / cond1)
                save_path.mkdir(parents=True, exist_ok=True)
                out_csv = save_path / f"{cond2}.csv"
                df.to_csv(out_csv, index=False)
                print(f"✔ {out_csv}  (샘플 {len(df)-1},  평균 "
                      f"WER={df['WER'].iloc[-1]}, CER={df['CER'].iloc[-1]})")