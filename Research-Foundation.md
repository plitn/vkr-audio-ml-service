# Checkpoint 2: Literature Review and Research Foundation
---
## 1. Evaluation Metrics
### 1.1 ASR Metrics

**WER** - Word Error Rate; <br>
**CER** - Character Error Rate; <br>
**RTFx** - Inverse real-time factor.[1]

| Metric | Definition | Direction |
|---|---|---|
| **WER** | $$\frac{S + D + I}{N}$$, where $S$ = substitutions, $D$ = deletions, $I$ = insertions, $N$ = reference word count | Lower is better |
| **CER** | WER formula at character level | Lower is better |
| **RTFx** | $$\frac{AD}{PT}$$; AD = audio duration, PT = processing time| Higher is better |

---

### 1.2 Speaker Diarization Metrics

**DER** - Diarization Error Rate; where $T_{miss}$ - missed speech; 
$T_{fa}$ - false alarm; $T_{conf}$ - confusion, correctly detected speech, but attributed to wrong speaker; $T_{total}$ - total audio time; <br>
$T_{fa}$ and $T_{miss}$ reflect VAD quality; <br>
$T_{conf}$ reflects embedding quality;

**JER** - Jaccard Error Rate; Based on Jaccard Index, evaluates how well predicted speaker segments overlap with reference segments

| Metric | Definition | Direction |
|---|---|---|
| **DER** | $$\frac{T_{miss} + T_{fa} + T_{conf}}{T_{total}}$$ | Lower is better |
| **JER** | 1- $$\frac{1}{N}\sum Jaccard(ref,mapped_sys)$$| Lower is better |

For speaker diarization we should look at both of these metrics, because DER can be dominated by long speakers and obscure errors on small speakers and JER can over-penalize false short detections[https://www.pyannote.ai/blog/how-to-evaluate-speaker-diarization-performance]

---

### 1.3 Noise Reduction Metrics

| Metric | Definition | Range | Direction |
|---|---|---|---|
| **PESQ[https://www.itu.int/rec/T-REC-P.862-200102-W/en]** | Speech quality evaluation, compares original signal $X(t)$ with degraded signal $Y(t)$ | −0,5 to 4,5 | Higher is better |
| **STOI** | Short-Time Objective Intelligibility, shows how easy it is to understand speech | 0 to 1 | Higher is better |
| **RTF** | $$\frac{PT}{AD}$$; AD = audio duration, PT = processing time | > 0 | Lower is better |

PESQ and STOI are signal-level metrics that operate independently of language content, making them directly applicable to multilingual audio without modification.[4]
RTF is included as a deployment-relevant metric: a model with slightly lower PESQ but RTF < 1.0 may be preferable for the real-time path.

---

## 2. Automatic Speech Recognition — Candidate Models

### 2.1 Background

Right now Open ASR Leaderboard[1] tracks over 150 models and identifies three dominant architectural families:

1. **CTC / TDT decoders with Conformer encoders** — highest throughput, competitive WER.
2. **Encoder-decoder Transformers**  —  multilingual coverage, moderate throughput.
3. **Conformer encoders + LLM decoders** — best WER on English, lowest throughput.


From the leaderboard data we can observe two patterns:
- Models that are fine-tuned for English improve WER on English benchmarks, but are worse on other languages;
- CTC or TDT models deliver much higher RTFx than LLM-decoder systems with little WER cost, this may be important for our project

Below there are three candidates with different values in accuracy, throughput and language coverage. Later all of them will be evaluated on the same datasets

---

### 2.2 Candidate 1: Whisper Large-v3 (OpenAI)

- **Architecture:** Encoder-decoder Transformer; 1.55B parameters.[5]
- **Training data:** 680_000 hours of multilingual weakly supervised data from the web.[5]
- **Language support:** 99 languages.
- **WER (English):** 6,43%.[1]
- **WER (multilingual):** 9,9%.[6]
- **RTFx (GPU, A100):** 68,56.[1]

**Why is used in comparison?** Whisper Large-v3 is the broadest multilingual baseline available (99 languages) and the most commonly used ASR.[7] It represents the upper bound of language coverage in this comparison. Its RTFx of 68,56 will serve as the throughput lower bound against which higher-throughput candidates are compared.

**Known limitations relevant to evaluation:**
- RTFx approximately 49 lower than Parakeet-TDT-0.6B-v3 on identical hardware.[6]

---

### 2.3 Candidate 2: Parakeet-TDT-0.6B-v3 (NVIDIA)

- **Architecture:** FastConformer encoder + TDT decoder; 600M parameters.[8]
- **Training data:** 660,000 hours from the Granary multilingual corpus[9] + 10,000 hours of human-transcribed NeMo ASR Set 3.0; trained for 150_000 steps on 128 A100 GPUs.[8]
- **Language support:** 25 European languages including English and Russian; Automatic language identification.[8]
- **WER (English):** 6,34%.[8]
- **WER (Russian):** 5,51% on FLEURS; 3% on CoVoST.[8]
- **WER (multilingual):** 8,1%.[6]
- **RTFx (A100):** 3332,74.[6]

**Noise robustness:[8]**

| SNR Level | Avg WER | Relative degradation |
|---|---|---|
| Clean | 6.34% | — |
| 10 dB | 7.12% | +12.3% |
| 5 dB | 8.23% | +29.8% |
| 0 dB | 11.66% | +84.0% |
| −5 dB | 19.88% | +213.6% |

**Why is used in comparison?** Parakeet-TDT-0.6B-v3 is the highest-throughput multilingual candidate among selected. It achieves better multilingual WER than Whisper Large-v3 with higher RTFx. Also it has one of the highest RTFx among all models in the leaderboard, but it may have a problem with non-NVIDIA GPUs

**Known limitations relevant to evaluation:**
- 25 languages;
- NVIDIA NeMo framework.

---

### 2.4 Candidate 3: Whisper Large-v3 Turbo (OpenAI)

- **Architecture:** Simplier variant of Large-v3; decoder reduced from 32 to 4 layers; 809M parameters.[11]
- **Language support:** 99 languages.
- **WER (multilingual):** 5,87%
- **RTFx:** 121,11.[11]

**Why is used in comparison?** Whisper Turbo stays somewhere between Whisper Large-v3 (maximum coverage, lowest throughput) and Parakeet-TDT-0.6B-v3 (maximum throughput, limited coverage). It is the candidate for lighter hardware deployment.

---

### 2.5 ASR Candidate Comparison

| Property | Whisper Large-v3 | Parakeet-TDT-0.6B-v3 | Whisper Turbo |
|---|---|---|---|
| Architecture | Enc-Dec Transformer | FastConformer + TDT | Distilled Enc-Dec |
| Parameters | 1.55B | 0.6B | 809M |
| Languages | 99 | 25 | 99 |
| WER EN | 6.43% | 6,34% | 6.4% |
| WER multilingual | 9,9% | **8.1%** | — |
| RTFx (A100) | 68,56 | **3332.74** | 121,11 |
| Source | [1][5] | [6][8] | [11] |

---

## 3. Speaker Diarization — Candidate Models

### 3.1 Background

Speaker diarization task decomposes into voice activity detection, speaker segmentation, speaker embedding extraction, and clustering. <br>
There are two main architectural paradigms:

**Cascaded pipelines.** Each component is trained independently and can be replaced. The primary representative is pyannote.audio.[2]

**End-to-end neural diarization (EEND).** A single model jointly predicts speaker activity, which allows native handling of overlapping speech.


---

### 3.2 Candidate 1: pyannote.audio 3.1 (CNRS)

- **Architecture:** Cascaded pipeline.[2]
- **Overlap:** Up to 3 overlapping speakers.
- **DER benchmarks:[2]**
  - VoxConverse test: 11%
  - AMI test: 19%
  - DIHARD III: 27%
- **GPU throughput:** 20–30 min per hour of audio.
- **CPU throughput:** 2–3 hours per hour of audio.

**Why is used in comparison?** pyannote.audio 3.1 is the most widely used open-source diarization framework, having the largest community and the broadest published benchmark coverage. Also it represents the cascaded approach amopng other candidates.

**Known limitations relevant to evaluation:**
- HuggingFace authentication.

---

### 3.3 Candidate 2: NVIDIA NeMo MSDD (Multi-Scale Diarization Decoder)

- **Architecture:** Multi-Scale Diarization Decoder (MSDD).[12]
- **DER:** Specific value was not found without running model on datasets.
- **Setup:** Requires NVIDIA GPU + CUDA + NeMo.

**Why is used in comparison?** NeMo MSDD shares the same inference framework as Parakeet-TDT-0.6B-v3, which combining them might be valuable for for this project developement. It represents the neural clustering approach among other candidates.

**Known limitations relevant to evaluation:**
- NVIDIA GPU.
- Complex installation.
- Benchmarks in literature were not found.

---

### 3.4 Diarization Candidate Comparison

| Property | pyannote 3.1 | NeMo MSDD |
|---|---|---|
| Type | Cascaded | Neural clustering |
| Overlap handling | Yes (powerset) | Yes |
| DER (VoxConverse) | ~11% | TBD |
| DER (AMI) | ~19% | TBD |
| DER (DIHARD III) | ~27% | TBD |
| GPU requirement | Recommended | Required NVIDIA GPU |
| Streaming | No | Partial |
| Speaker-word alignment | Manual (post-hoc) | Manual (post-hoc) |
| Source | [2] | [12] |

---

## 4. Audio Noise Reduction — Candidate Models

### 4.1 Background

Speech enhancement methods aim to recover a clean speech signal from a noisy observation. Methods differ in processing domain, causality, and algorithmic latency. For real-time deployment, algorithmic latency ≤20 ms is the accepted threshold.

Three candidates are included: one traditional signal-processing baseline, and two neural approaches at different positions on quality and compute speeds q. The traditional

---

### 4.2 Candidate 0: Wiener Filter (Baseline)

- **Type:** Classical signal processing; no learned parameters.
- **Latency:** Frame-level; configurable; no GPU required.
- **PESQ / STOI:** Varies. Performs well on white noise; degrades on non-stationary noise.

**Why is used in comparison?** The Wiener filter is the standard lower-bound baseline in speech enhancement research. It is included as an anchor.

**Known limitations relevant to evaluation:**
- Requires a noise estimation step (e.g., silence detection or first-frame estimation); the quality of this estimate directly affects output quality.
- Assumes noise stationarity — performance degrades rapidly on non-stationary noise.

---

### 4.3 Candidate 1: RNNoise

- **Architecture:** Hybrid DSP + neural.[14]
- **Latency:** 10 ms.
- **PESQ:** 3,88.
- **STOI:** 0,92.

**Why is used in comparison?** RNNoise represents the lightweight neural approach. It is relevant for CPU based deployment. Comparing RNNoise against DeepFilterNet3 on our datasets will help us understand which hardware is better to use in service developement and show difference in quality or computation costs.

---

### 4.4 Candidate 2: DeepFilterNet3

- **Architecture:** ERB-scaled gain prediction for the spectral envelope, followed by deep filtering for periodic speech components.[4]
- **Latency:** 10–20 ms.
- **PESQ:** 3,5–4,0+.
- **STOI:** 0,95+.

**Why is used in comparison?** DeepFilterNet3 is the current most popular open-source model for neural speech enhancement. Having that it is so called "language-agnostic" it is what we need for our multilingual service.

---

### 4.5 Noise Reduction Candidate Comparison

| Property | Wiener Filter | RNNoise | DeepFilterNet3 |
|---|---|---|---|
| Type | Traditional DSP | Hybrid DSP + neural | Neural |
| Latency | - | 10 ms | 10–20 ms |
| PESQ | Varies  | 3,88 | 3,5–4,0+ |
| STOI | Varies | 0,92 | 0,95+ |
| Non-stationary noise | Poor | Moderate | Strong |
| Language-agnostic | Yes | Yes | Yes |
| Source | [ ] | [14] | [4] |

---

## 5. Dataset Selection

### 5.1 ASR Datasets

| Dataset | Language(s) | Size | Type | Notes |
|---|---|---|---|---|
| **LibriSpeech** | English | ~960h train / 10.7h test-clean | Read speech | Standard ASR benchmark.[15] |
| **Common Voice** | 100+ | Varies per language | Crowd-sourced | Primary multilingual dataset. |
| **FLEURS** | 102 | 12h per language | Read speech | Used in Open ASR Leaderboard + Parakeet-TDT-0.6B-v3 evaluation.[8][16] |
| **Multilingual LibriSpeech** | 8 | Varies | Read speech | Higher quality than Common Voice.[17] |
| **CoVoST 2** | 21 source languages | Varies | Read speech | Used in Parakeet-TDT-0.6B-v3 evaluation; direct WER comparison.[8] |

---

### 5.2 Diarization Datasets

| Dataset | Language(s) | Hours | Notes |
|---|---|---|---|
| **AMI Meeting Corpus** | English | 100 | Standard benchmark.[18] |
| **VoxConverse** | English | 64 | YouTube audio.[2] |
| **CALLHOME** | Multilingual | 18 | Telephone; multiple languages; 2–6 speakers. |
| **DIHARD III** | Diverse | 33 | Hardest benchmark; final evaluation.[3] |

---

### 5.3 Noise Reduction Datasets

| Dataset | Language(s) | Notes |
|---|---|---|
| **DNS Challenge** | English + multilingual | Primary benchmark; clean/noisy paired recordings.[19] |
| **VCTK + DEMAND** | English | Widely used for PESQ/STOI comparison. |
| **URGENT Challenge 2025** | Multilingual | Multiple distortion types (noise, reverberation, clipping).[20] |

---

## 6. Candidate Model Summary

### 6.1 Candidates

| Subtask | Candidates | Difference |
|---|---|---|
| **ASR** | Whisper Large-v3 ; Parakeet-TDT-0.6B-v3 ; Whisper Turbo | Language coverage vs throughput vs parameter count |
| **Diarization** | pyannote 3.1 ; NeMo MSDD | Cascade vs neural clustering |
| **Noise reduction** | Wiener filter ; RNNoise ; DeepFilterNet3 | Traditional baseline vs lightweight neural vs full neural |

---

## 8. References

[1]: Srivastav, V., Zheng, S., Bezzam, E., et al. (2025). Open ASR Leaderboard: Towards Reproducible and Transparent Multilingual and Long-Form Speech Recognition Evaluation. https://huggingface.co/spaces/hf-audio/open_asr_leaderboard https://arxiv.org/pdf/2510.06961

[2]: Bredin, H. (2023). pyannote.audio 2.1 speaker diarization pipeline. https://github.com/pyannote/pyannote-audio

[3]: Ryant, N., et al. (2021). The Third DIHARD Diarization Challenge. https://arxiv.org/pdf/2012.01477

[4]: Schröter, H., Escalante-B, A. N., Rosenkranz, T., & Maier, A. (2023). DeepFilterNet: Perceptually Motivated Real-Time Speech Enhancement. https://github.com/Rikorose/DeepFilterNet

[5]: Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023). Robust speech recognition via large-scale weak supervision. https://github.com/openai/whisper

[6]: NVIDIA. (2025). Canary-1B-v2 & Parakeet-TDT-0.6B-v3: Efficient and High-Performance Models for Multilingual ASR and AST. https://arxiv.org/pdf/2509.14128

[7]: Wade, M., Bain, M., Zisserman, A., & Vedaldi, A. (2023). WhisperX: Time-Accurate Speech Transcription of Long-Form Audio. https://github.com/m-bain/whisperX https://arxiv.org/pdf/2303.00747

[8]: NVIDIA. (2025). parakeet-tdt-0.6b-v3 model card. https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3

[9]: Koluguri, N., et al. (2025). Granary: Speech Recognition and Translation Dataset in 25 European Languages. https://huggingface.co/datasets/nvidia/Granary https://arxiv.org/pdf/2505.13404

[11]: OpenAI. (2024). Whisper Large-v3-Turbo model card. https://huggingface.co/openai/whisper-large-v3-turbo

[12]: Park, T. J., et al. (2022). Multi-scale speaker diarization with dynamic local attention. https://arxiv.org/pdf/2203.15974

[14]: Valin, J.-M. (2018). A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement. https://github.com/xiph/rnnoise https://arxiv.org/pdf/1709.08243

[15]: Panayotov, V., Chen, G., Povey, D., & Khudanpur, S. (2015). LibriSpeech: An ASR corpus based on public domain audio books. https://www.danielpovey.com/files/2015_icassp_librispeech.pdf

[16]: Conneau, A., et al. (2022). FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech. https://arxiv.org/pdf/2205.12446

[17]: Pratap, V., et al. (2020). MLS: A Large-Scale Multilingual Dataset for Speech Research. https://arxiv.org/pdf/2012.03411

[18]: Carletta, J., et al. (2005). The AMI Meeting Corpus. https://groups.inf.ed.ac.uk/ami/corpus/

[19]: Reddy, C. K. A., et al. (2021). INTERSPEECH 2021 Deep Noise Suppression Challenge. https://arxiv.org/pdf/2101.01902

[20]: URGENT Challenge 2025. https://urgent-challenge.github.io/urgent2025/