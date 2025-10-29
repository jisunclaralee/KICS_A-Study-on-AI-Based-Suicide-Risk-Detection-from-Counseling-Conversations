# AI 기반의 자살위험 예측 모델에 대한 연구

**A Study on AI-Based Suicide Risk Detection from Counseling Conversations**

이지선  
국민대학교  
jisunclaralee@kookmin.ac.kr

## 요약

본 연구는 정신건강 상담 발화 내 자살 위험 징후를 조기에 탐지하기 위한 자동화 시스템을 구축하고, 그 성능을 비교·분석하였다. Whisper-tiny 및 Wav2Vec2.0 기반 음성 인식 모델을 활용해 발화 텍스트를 전처리한 후, GPT-4o-mini 기반 Scikit-LLM 분류기를 zero-shot 방식으로 적용하였다. 실험 결과, Whisper-tiny 기반 파이프라인은 전체 정확도 98.53%, 자살 클래스에 대한 F1-score 0.958로 Wav2Vec2.0보다 우수한 탐지 성능을 보였다. 특히 Whisper-tiny는 높은 정밀도와 민감도를 유지하며 자살 발화를 보다 정확하게 분류하였다. 이는 Whisper 모델이 자살 관련 표현을 보다 정밀하게 전사함으로써 분류 정확도 향상에 기여했음을 시사한다. 본 연구는 자살 고위험군 조기 식별을 위한 실용적인 음성 기반 LLM 분류 파이프라인의 가능성을 제시한다.

## 프로젝트 구조

```
├── suicide_classification_skllm.py    # Scikit-LLM 기반 자살 위험 분류
├── whisper_tiny_stt.py               # Whisper-tiny STT 모델
├── wav2vec2_stt.py                   # Wav2Vec2.0 STT 모델
└── README.md                         # 프로젝트 설명서
```

## 주요 모델 및 기술

### 1. wav2vec 2.0

- **wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations**
    
    *Authors: Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli*
    
    *Venue: NeurIPS 2020*
    
    이 논문은 음성 데이터에서 자기지도학습(self-supervised learning)을 통해 강력한 음성 표현을 학습하는 프레임워크를 제안합니다. 라벨이 거의 없는 환경에서도 뛰어난 성능을 보이며, LibriSpeech 등 다양한 벤치마크에서 SOTA를 달성했습니다.
    
    - [arXiv:2006.11477](https://arxiv.org/abs/2006.11477)

- **CPT-Boosted Wav2vec2.0: Towards Noise Robust Speech Recognition in Classrooms**
    
    *Authors: Xinyi Peng et al.*
    
    *Venue: arXiv 2024*
    
    도메인 특화(예: 교실 환경) 잡음에 강인한 음성 인식 성능을 위해 continued pretraining(CPT) 방법을 적용한 wav2vec2.0의 적응 연구 논문입니다.

### 2. GPT-4o Mini (지피티4o미니)

- **GPT-4o mini: advancing cost-efficient intelligence**
    
    *OpenAI 공식 블로그, 2024*
    
    GPT-4o mini는 OpenAI의 소형 고효율 언어 모델로, 텍스트 및 비전 태스크에서 GPT-3.5 Turbo를 능가하는 성능을 보입니다. MMLU 82%, MMMU 59.4% 등 다양한 벤치마크에서 우수한 성적을 기록했습니다.
    
    - 공식 논문은 아직 공개되지 않았으나, OpenAI 공식 문서와 블로그에서 상세한 벤치마크 및 모델 특징 설명이 제공됩니다.

- **MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models**
    
    *Authors: Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, Mohamed Elhoseiny*
    
    *Venue: arXiv 2023*
    
    GPT-4 계열의 멀티모달 능력을 소형화한 MiniGPT-4에 관한 논문으로, 비전-언어 태스크에서의 효율적 구조와 성능을 다룹니다.

### 3. Whisper Tiny

- **Robust Speech Recognition via Large-Scale Weak Supervision**
    
    *Authors: Alec Radford et al. (OpenAI)*
    
    *Venue: arXiv 2022*
    
    Whisper 모델의 원천 논문으로, 68만 시간의 대규모 약지도(weak supervision) 음성 데이터로 학습된 Transformer 기반 ASR 모델입니다. Whisper Tiny는 Whisper 계열의 초경량 모델(39M 파라미터)로, 다양한 언어와 환경에 강인한 성능을 보입니다.
    
    - [arXiv:2212.04356](https://arxiv.org/abs/2212.04356)
    - [공식 PDF](https://cdn.openai.com/papers/whisper.pdf)

- **Whisper-tiny 모델을 활용한 음성 분류 개선**
    
    *저자: 김태형 외*
    
    Whisper-tiny를 활용한 키워드 스포팅 및 온디바이스 음성 분류 성능 개선에 관한 논문입니다.

### 4. Scikit-LLM

- **Scikit-LLM: Scikit-Learn Meets Large Language Models**
    
    *Authors: Iryna Kondrashchenko, Oleh Kostromin*
    
    *Venue: GitHub, 2023*
    
    대형 언어모델(LLM)을 scikit-learn 파이프라인에 통합할 수 있게 해주는 Python 패키지로, zero-shot/지도/소수샷 분류 등 다양한 NLP 태스크를 손쉽게 활용할 수 있습니다.
    
    - [GitHub: BeastByteAI/scikit-llm](https://github.com/BeastByteAI/scikit-llm)

- **Large Language Models with Scikit-learn: A Comprehensive Guide to Scikit-LLM**
    
    *Unite.AI, 2024*
    
    Scikit-LLM의 설치, 활용법, zero-shot 분류 등 실제 적용 예제와 장점에 대한 종합 가이드.

## 모델 성능 비교

### STT 성능 (WER, CER)

| 모델 | WER | CER |
|------|-----|-----|
| Wav2Vec2.0 | 1.0030 | 0.4719 |
| Whisper-tiny | 0.6843 | 0.3584 |

WER과 CER 수치는 낮을수록 STT 모델의 성능이 우수함을 의미합니다. Whisper-tiny 모델은 Wav2Vec2.0에 비해 전반적인 오류율이 낮아, 품질이 더 뛰어남을 입증하였습니다.

### 자살 위험 분류 성능

#### Whisper-tiny 기반 결과

| 구분 | Precision | Recall | F1-score | Support |
|------|-----------|--------|----------|---------|
| non-suicide | 0.991 | 0.991 | 0.991 | 4,376 |
| suicide | 0.959 | 0.957 | 0.958 | 937 |
| **Accuracy** | - | - | **0.985** | 5,313 |
| Macro avg | 0.975 | 0.974 | 0.975 | 5,313 |
| Weighted avg | 0.985 | 0.985 | 0.985 | 5,313 |

#### Wav2Vec2.0 기반 결과

| 구분 | Precision | Recall | F1-score | Support |
|------|-----------|--------|----------|---------|
| non-suicide | 0.983 | 0.942 | 0.962 | 4,376 |
| suicide | 0.774 | 0.923 | 0.842 | 937 |
| **Accuracy** | - | - | **0.939** | 5,313 |
| Macro avg | 0.879 | 0.933 | 0.902 | 5,313 |
| Weighted avg | 0.946 | 0.939 | 0.941 | 5,313 |

## 모델 요약 표

| 분야/모델 | 대표 논문/자료명 | 저자/기관 | 연도 |
|-----------|------------------|-----------|------|
| wav2vec 2.0 | wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations | Baevski et al. (Meta AI) | 2020 |
|  | CPT-Boosted Wav2vec2.0 | Peng et al. | 2024 |
| GPT-4o mini | GPT-4o mini: advancing cost-efficient intelligence (OpenAI 블로그/문서) | OpenAI | 2024 |
|  | MiniGPT-4: Enhancing Vision-Language Understanding with Advanced LLMs | Zhu et al. | 2023 |
| Whisper Tiny | Robust Speech Recognition via Large-Scale Weak Supervision | Radford et al. (OpenAI) | 2022 |
|  | Whisper-tiny 모델을 활용한 음성 분류 개선 | 김태형 외 | 2024 |
| Scikit-LLM | Scikit-LLM: Scikit-Learn Meets Large Language Models | Kondrashchenko, Kostromin (BeastByteAI) | 2023 |
|  | Large Language Models with Scikit-learn: A Comprehensive Guide to Scikit-LLM | Unite.AI | 2024 |

## 실행 방법

### 필수 라이브러리 설치

```bash
pip install transformers torch torchaudio
pip install librosa jiwer pandas matplotlib seaborn
pip install scikit-llm tqdm
```

### 1. STT 전처리

#### Whisper-tiny 사용
```bash
python whisper_tiny_stt.py
```

#### Wav2Vec2.0 사용
```bash
python wav2vec2_stt.py
```

### 2. 자살 위험 분류

```bash
python suicide_classification_skllm.py
```

**⚠️ 주의사항:** OpenAI API 키를 설정해야 합니다. `suicide_classification_skllm.py` 파일에서 `YOUR_OPENAI_API_KEY_HERE`를 실제 API 키로 교체하세요.

## 데이터셋

본 연구는 AI-Hub의 "복지 분야 콜센터 상담 데이터"를 활용하여 총 563,251 문장에 대한 발화를 STT로 변환 후, 정제 및 병합 과정을 거쳐 suicide(자살위험) 또는 non-suicide(일반상담)로 태깅하여 이진 분류 데이터셋을 구성하였습니다.

### 기대 폴더 구조

```
<BASE_DATA_DIR>/
 └─ <dataset_type>/                     # 1.Training, 2.Validation …
     ├─ 라벨링데이터/
     │   └─ <category>/<condition>/<speaker_id>/<file_id>.json
     └─ 원천데이터/
         └─ <category>/<condition>/<speaker_id>/<file_id>.wav
```

## 결론 및 시사점

본 연구는 음성 기반 자살 위험 탐지에서 STT 모델의 전사 품질이 LLM 분류 성능에 직접적인 영향을 미친다는 점을 실증하였다. Whisper-tiny 기반 파이프라인은 높은 전사 정확도로 인해 자살 발화 탐지에서 우수한 성능을 보였으며, 이는 공공상담센터, 군부대, 교육기관 등 사각지대에서 조기 개입 도구로 활용될 수 있는 가능성을 시사한다. 기술적 성능을 넘어 심리적 낙인 해소와 접근성 개선에도 기여할 수 있다는 점에서 사회적 가치가 크다. 향후 연구에서는 다중 클래스 분류, 설명 가능한 AI, 음성 품질 보정 기술을 포함한 고도화를 통해 보다 정교한 자살 예방 시스템으로의 확장이 필요하다.

## 참고문헌

[1] 통계청, "사망원인통계 결과," 통계청 보도자료, 2023.  
[2] 장규현, "상담자들의 심리상담 챗봇에 대한 인식," 연세대학교 교육대학원 석사학위논문, 2021.  
[3] D. Shin, "음성과 텍스트를 이용하여 우울증 및 자살 위험을 평가하는 인공지능 기반 임상의사결정지원시스템에 관한 연구," 박사학위논문, 서울대학교, 2022.  
[4] S. Y. Min, "음성 분석을 이용한 인공지능 기반 자살 위험군 선별 및 모니터링," 의학박사학위논문, 서울대학교, 2024.  
[5] D. Lee, "Predicting Suicidality with Explainable Deep Learning Models," Ph.D. Dissertation, 성균관대학교, 2024.

## License

이 프로젝트는 연구 목적으로만 사용되어야 하며, 실제 임상 환경에서의 사용 전에는 추가적인 검증이 필요합니다.

## Contact

- 이지선: jisunclaralee@kookmin.ac.kr
- 윤수연: 1104py@kookmin.ac.kr