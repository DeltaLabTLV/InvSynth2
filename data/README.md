# Disentangling Representation Learning and Loss Design in Synthesizer Inversion

> **ISMIR 2026 Submission** · Anonymous Authors

---

## Abstract

When a synthesizer inversion system improves over its predecessor, what deserves the credit—the encoder architecture, the pre-training strategy, or the loss function? In practice, these factors are often changed simultaneously, leaving the question unanswered.

We present the **first controlled ablation study** that independently isolates encoder architecture, self-supervised pre-training, and loss function within a single proxy-based synthesizer inversion framework—a three-way disentanglement not previously provided in this setting. To ensure a fair architectural comparison, we standardize the model budget at **3.5M parameters** across all proposed encoders. We independently vary the encoder (contrastive Transformer vs. MAE U-Net), pre-training (self-supervised vs. random init), and loss (standard spectral vs. our proposed **IMW loss** vs. log-spectral compression), evaluated on FM synthesis, Yamaha DX7, and TAL Noisemaker.

**The results are strikingly uneven:** self-supervised pre-training accounts for the majority of the gains, while encoder architecture matters least. The proposed IMW loss adds a targeted improvement in high-frequency, low-magnitude spectral regions—the weak overtones and filter resonances that shape perceived timbre—and consistently outperforms log-compression.

---

## 🔊 Audio Examples

> Audio samples and interactive demos: **[anonymous-ismir2026-audio.github.io](https://anonymous-ismir2026-audio.github.io/)**  
> *(Link will be de-anonymised upon acceptance)*

---

## Spectrogram Comparisons — FM Synthesis

Each row is a different sound sample. Columns show all six systems side-by-side.

| Col | Model |
|-----|-------|
| (a) | Ground Truth |
| (b) | IS — Barkan & Tsiris (2019) |
| (c) | IS2 — Barkan et al. (2023) |
| (d) | Flow — Esling et al. (2019) |
| (e) | **IS2-Transformer** (ours) |
| (f) | **IS2-UNet** (ours) |

---

### Sample 0

| (a) Ground Truth | (b) IS | (c) IS2 | (d) Flow | (e) IS2-Transformer | (f) IS2-UNet |
|:---:|:---:|:---:|:---:|:---:|:---:|
| ![](spectrograms/fm/0_-_GT.png) | ![](spectrograms/fm/0_-_IS.png) | ![](spectrograms/fm/0_-_IS2.png) | ![](spectrograms/fm/0_-_Flow.png) | ![](spectrograms/fm/0_-_IS2-Transformer.png) | ![](spectrograms/fm/0_-_IS2-UNet.png) |

---

### Sample 1

| (a) Ground Truth | (b) IS | (c) IS2 | (d) Flow | (e) IS2-Transformer | (f) IS2-UNet |
|:---:|:---:|:---:|:---:|:---:|:---:|
| ![](spectrograms/fm/1_-_GT.png) | ![](spectrograms/fm/1_-_IS.png) | ![](spectrograms/fm/1_-_IS2.png) | ![](spectrograms/fm/1_-_Flow.png) | ![](spectrograms/fm/1_-_IS2-Transformer.png) | ![](spectrograms/fm/1_-_IS2-UNet.png) |

---

### Sample 2

| (a) Ground Truth | (b) IS | (c) IS2 | (d) Flow | (e) IS2-Transformer | (f) IS2-UNet |
|:---:|:---:|:---:|:---:|:---:|:---:|
| ![](spectrograms/fm/2_-_GT.png) | ![](spectrograms/fm/2_-_IS.png) | ![](spectrograms/fm/2_-_IS2.png) | ![](spectrograms/fm/2_-_Flow.png) | ![](spectrograms/fm/2_-_IS2-Transformer.png) | ![](spectrograms/fm/2_-_IS2-UNet.png) |

---

### Sample 3

| (a) Ground Truth | (b) IS | (c) IS2 | (d) Flow | (e) IS2-Transformer | (f) IS2-UNet |
|:---:|:---:|:---:|:---:|:---:|:---:|
| ![](spectrograms/fm/3_-_GT.png) | ![](spectrograms/fm/3_-_IS.png) | ![](spectrograms/fm/3_-_IS2.png) | ![](spectrograms/fm/3_-_Flow.png) | ![](spectrograms/fm/3_-_IS2-Transformer.png) | ![](spectrograms/fm/3_-_IS2-UNet.png) |

---

### Sample 4

| (a) Ground Truth | (b) IS | (c) IS2 | (d) Flow | (e) IS2-Transformer | (f) IS2-UNet |
|:---:|:---:|:---:|:---:|:---:|:---:|
| ![](spectrograms/fm/4_-_GT.png) | ![](spectrograms/fm/4_-_IS.png) | ![](spectrograms/fm/4_-_IS2.png) | ![](spectrograms/fm/4_-_Flow.png) | ![](spectrograms/fm/4_-_IS2-Transformer.png) | ![](spectrograms/fm/4_-_IS2-UNet.png) |

---

### Sample 5

| (a) Ground Truth | (b) IS | (c) IS2 | (d) Flow | (e) IS2-Transformer | (f) IS2-UNet |
|:---:|:---:|:---:|:---:|:---:|:---:|
| ![](spectrograms/fm/5_-_GT.png) | ![](spectrograms/fm/5_-_IS.png) | ![](spectrograms/fm/5_-_IS2.png) | ![](spectrograms/fm/5_-_Flow.png) | ![](spectrograms/fm/5_-_IS2-Transformer.png) | ![](spectrograms/fm/5_-_IS2-UNet.png) |

---

### Sample 6

| (a) Ground Truth | (b) IS | (c) IS2 | (d) Flow | (e) IS2-Transformer | (f) IS2-UNet |
|:---:|:---:|:---:|:---:|:---:|:---:|
| ![](spectrograms/fm/6_-_GT.png) | ![](spectrograms/fm/6_-_IS.png) | ![](spectrograms/fm/6_-_IS2.png) | ![](spectrograms/fm/6_-_Flow.png) | ![](spectrograms/fm/6_-_IS2-Transformer.png) | ![](spectrograms/fm/6_-_IS2-UNet.png) |

---

### Sample 7

| (a) Ground Truth | (b) IS | (c) IS2 | (d) Flow | (e) IS2-Transformer | (f) IS2-UNet |
|:---:|:---:|:---:|:---:|:---:|:---:|
| ![](spectrograms/fm/7_-_GT.png) | ![](spectrograms/fm/7_-_IS.png) | ![](spectrograms/fm/7_-_IS2.png) | ![](spectrograms/fm/7_-_Flow.png) | ![](spectrograms/fm/7_-_IS2-Transformer.png) | ![](spectrograms/fm/7_-_IS2-UNet.png) |


---

## Method Overview

```
Target audio x(t)
       │
       ▼
  log-mag STFT  ──►  Encoder (SSL pre-trained)  ──►  θ̂
                      ├─ Transformer (NT-Xent)         │
                      └─ U-Net (MAE)                   ▼
                                               Differentiable Proxy P
                                                       │
                                                       ▼
                                               Reconstructed X̂
                                                       │
                                               Spectral + IMW Loss
```

**Two contributions:**
1. **Controlled 3-way ablation** — encoder architecture, SSL pre-training, and loss function independently evaluated at a fixed 3.5M parameter budget
2. **IMW Loss** — inverse-magnitude weighted spectral loss that explicitly prioritises high-frequency, low-magnitude bins via target-only weighting `1/(|X_tf| + ε)`

---

## Key Results

| Factor | Effect |
|--------|--------|
| SSL pre-training | **Dominant** — largest gain in every metric across all 3 synthesizers |
| IMW loss | **Complementary** — targeted improvement in 4–8 kHz band; outperforms log-compression |
| Encoder architecture | **Smallest** — at matched capacity, Transformer ≈ U-Net overall |

> Full quantitative tables (Spec, Melspec, MFCC, SC, ACC, MOS) and ablation results in the paper.

---

## Repository Structure

```
├── README.md
└── spectrograms/
    └── fm/
        ├── 0_-_GT.png
        ├── 0_-_IS.png
        ├── 0_-_IS2.png
        ├── 0_-_Flow.png
        ├── 0_-_IS2-Transformer.png
        ├── 0_-_IS2-UNet.png
        ├── 1_-_GT.png
        │   ...
        └── 7_-_IS2-UNet.png
```

---

## Citation

```bibtex
@inproceedings{anonymous2026disentangling,
  title     = {Disentangling Representation Learning and Loss Design
               in Synthesizer Inversion},
  author    = {Anonymous Authors},
  booktitle = {Proceedings of the 27th International Society
               for Music Information Retrieval Conference (ISMIR)},
  year      = {2026}
}
```

---

*This repository will be fully de-anonymised upon paper acceptance.*
