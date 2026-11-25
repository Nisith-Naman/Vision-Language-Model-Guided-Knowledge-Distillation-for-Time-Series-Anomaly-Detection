# Poster content (final, 23 November 2026)

Copy for each box of the course poster template. Every number matches `main.tex` and
`experiments/benchmark_v2/` (RESULTS.md, GENERALISATION.md, OVERFITTING.md, NOVELTY_CHECKS.md,
OUT_OF_SAMPLE.md). Figures are in `final_ie643_report/figures/`.

**Title:** Vision-Language Model Guided Knowledge Distillation for Time-Series Anomaly Detection

---

## 1. Introduction

- **Problem.** Decide *whether* and *where* a multivariate sensor stream behaves abnormally: card fraud,
  freezing of gait in Parkinson's patients, oil-well faults, train-compressor failures.
- **Why it matters.** Missed failures are costly, but anomaly labels are rare, expensive and specific to each
  dataset, so a detector trained only on them learns a narrow rule.
- **Why a VLM.** Many anomalies are *visual* (spikes, level shifts, tremor). CLIP has seen millions of
  charts and carries a general sense of what an irregular signal looks like, but it is too large for edge use.
- **Our idea.** A frozen VLM teaches a small GRU through knowledge distillation; only the GRU is deployed.
- **Research question.** Does the VLM teacher help *beyond training the same GRU on labels alone*?

## 2. Workflow

Figure: **`poster_workflow.png`**. Top row: primary assessment (v1). The review box links to the bottom row:
label audit → teacher repair → novelty (learned prompts, spectrogram, confidence-weighted KD) → controlled
benchmark → verification → overfitting and generalisation checks → out-of-sample test.

## 3. Methods and Approaches

Figure: **`poster_architecture.png`** (full teacher-student model).

- **Teacher.** CLIP ViT-B/32 with **every weight frozen**.
  - Each window is rendered two ways: a heatmap with sensors clipped to ±4σ, and a per-sensor spectrogram.
  - **Learned prompts (CoOp):** `[v1 … v14] normal | anomalous`, 7,168 trainable numbers passed through
    CLIP's frozen text encoder.
- **Leak-free soft labels.** Each training window's teacher label comes from prompts fitted on *other*
  windows (cross-fitting with interleaved blocks).
- **Student.** Conv1d → BiGRU, 95k parameters (about 1/1500 of CLIP): window head, step head, and an
  embedding head used only in training.
- **Loss.** BCE(steps, labels) + CE(window, label) + λ·w·T²·KL(teacher ‖ student) + λ·(1 − cos(student
  embedding, CLIP embedding)), where w is the teacher's confidence in the true class.
- **Localization.** Step score × window probability, smoothing, threshold and minimum event length, all tuned
  on validation only.
- **Regularized training.** Weight averaging (EMA), gain/noise augmentation, lower learning rate, dropout 0.3.

## 4. Dataset Details

| Dataset | Domain | Channels | Window | Split | Test anomaly rate (windows) |
|---|---|---|---|---|---|
| Credit Card | fraud | 29 | 64 | chronological 60/20/20 | 0.074 |
| Daphnet FoG | wearable gait | 9 (64 Hz) | 256 | held-out subjects | 0.048 |
| Petrobras 3W | oil wells | 8 (10 s) | 120 | held-out instances | 0.620 |
| MetroPT-3 | train compressor | 15 (1 min) | 120 | failure 1 train, 2 val, 3–4 test | 0.062 |

*Air Quality excluded: the review asked us to verify its labels, and they were synthetic injected events.*

## 5. Novelty Assessment

- **Prompt learning on a frozen VLM.** Learned prompts replace hand-written ones; CLIP is byte-identical before
  and after training. They beat their own starting phrase on all four datasets (MetroPT 0.083 → 0.998,
  Daphnet 0.064 → 0.401 test window AUPRC).
- **Multi-view rendering.** A spectrogram beside the heatmap. It carries different information (on Daphnet a
  spectrogram-only probe beats the heatmap probe, 0.454 vs 0.391), but helps the final teacher only on MetroPT.
- **Confidence-weighted, cross-fitted distillation** of soft labels and CLIP features. Wrong teacher labels
  are down-weighted (mean weight 0.27 vs 0.91 for correct ones); the code matches a hand computation exactly.
- **Controlled comparison.** 5 teachers × 7 students, 2 classical baselines, 3 seeds, identical splits and
  starting weights, plus a regularized labels-only baseline.
- **Rigour checks.** Independent metric recomputation, leakage audit, overfitting audit, and a label-free
  out-of-sample test with hidden anomalies.

## 6. Results

**Teacher, window AUPRC (zero-shot → learned prompts):** MetroPT 0.037 → 0.989 · 3W 0.771 → 0.974 ·
Daphnet 0.123 → 0.531 (v1's teacher: 0.029).

**Test window AUPRC** (students: regularized training, mean of 3 seeds):

| | Credit Card | Daphnet | 3W | MetroPT |
|---|---|---|---|---|
| Random (= prevalence) | 0.074 | 0.048 | 0.620 | 0.062 |
| CLIP zero-shot | 0.079 | 0.123 | 0.771 | 0.037 |
| CLIP learned prompts (teacher) | 0.137 | 0.531 | 0.974 | 0.989 |
| Gradient-boosted trees | 0.767 | 0.350 | 0.998 | 0.702 |
| GRU, labels only | 0.828 | 0.659 | 0.985 | 0.812 |
| **GRU + VLM KD (ours)** | 0.805 | 0.655 | 0.983 | **0.892** |

- **MetroPT (one training failure):** distillation beats labels only on all 3 seeds: +0.080 window and
  +0.061 point AUPRC, event F1 0.577 → 0.820. Without regularization the labels-only GRU memorised its single
  failure (train−test gap 0.40) and the gap looked like +0.365 point AUPRC. Figure: **`poster_generalisation.png`**.
- **Out of sample (label-free MetroPT series):** both students find **10 of 12** hidden real failures; the
  distilled one localises them better (event F1 0.546 vs 0.424). Figure: **`oos_metropt_bootstrap.png`**.
- **Overfitting:** none on Credit Card or 3W; Daphnet's ≈0.2 gap is shared by every model, the teacher
  included, so it is subject shift, not memorisation.
- **Caveat:** MetroPT's test set has only two, visually obvious failures; a one-sensor rule reaches 0.917.

Plots: `poster_generalisation.png` (main), `oos_metropt_bootstrap.png`, `auprc_window.png`,
`daphnet_before_after.png`.

## 7. Conclusions and Future Work

- A frozen VLM *can* teach a time-series detector once its input rendering and prompts are adapted.
- Distillation helps where labelled anomalies are scarcest (MetroPT: +0.06 to +0.08 AUPRC on every seed
  against a regularized baseline) and is neutral on Credit Card, 3W and Daphnet.
- The student is about 1/1500 of the teacher's size and about 330× faster (3,000 vs 9 windows/s on a CPU).
- **Future work:**
  - leave-one-failure-out / leave-one-subject-out tests for more test events;
  - distillation on unlabelled data;
  - one learned prompt per fault type for explainable alarms;
  - a novelty channel for anomaly types never seen in training.

## 8. References (LaTeX, bulleted)

Needs `\usepackage{hyperref}` (for `\url`); `\usepackage{enumitem}` only if you add `[leftmargin=*,nosep]`.

```latex
\begin{itemize}
\item Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger and Ilya Sutskever. Learning Transferable Visual Models From Natural Language Supervision, International Conference on Machine Learning (ICML), 2021. \url{https://arxiv.org/abs/2103.00020}
\item Kaiyang Zhou, Jingkang Yang, Chen Change Loy and Ziwei Liu. Learning to Prompt for Vision-Language Models, International Journal of Computer Vision, 2022. \url{https://arxiv.org/abs/2109.01134}
\item Geoffrey Hinton, Oriol Vinyals and Jeff Dean. Distilling the Knowledge in a Neural Network, NIPS Deep Learning and Representation Learning Workshop, 2015. \url{https://arxiv.org/abs/1503.02531}
\item Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta and Yoshua Bengio. FitNets: Hints for Thin Deep Nets, International Conference on Learning Representations (ICLR), 2015. \url{https://arxiv.org/abs/1412.6550}
\item Guansong Pang, Chunhua Shen, Longbing Cao and Anton van den Hengel. Deep Learning for Anomaly Detection: A Review, ACM Computing Surveys, 2021. \url{https://arxiv.org/abs/2007.02500}
\item Florian Bordes, Richard Yuanzhe Pang, Anurag Ajay, Alexander C. Li, Adrien Bardes, Suzanne Petryk, Oscar Ma\~nas, Zhiqiu Lin, Anas Mahmoud, Bargav Jayaraman, Mark Ibrahim, Melissa Hall, Yunyang Xiong, Jonathan Lebensold, Candace Ross, Srihari Jayakumar, Chuan Guo, Diane Bouchacourt, Haider Al-Tahan, Karthik Padthe, Vasu Sharma, Hu Xu, Xiaoqing Ellen Tan, Megan Richards, Samuel Lavoie, Pietro Astolfi, Reyhane Askari Hemmat, Jun Chen, Kushal Tirumala, Rim Assouel, Mazda Moayeri, Arjang Talattof, Kamalika Chaudhuri, Zechun Liu, Xilun Chen, Quentin Garrido, Karen Ullrich, Aishwarya Agrawal, Kate Saenko, Asli Celikyilmaz and Vikas Chandra. An Introduction to Vision-Language Modeling, arXiv preprint, 2024. \url{https://arxiv.org/abs/2405.17247}
\item Siwon Kim, Kukjin Choi, Hyun-Soo Choi, Byunghan Lee and Sungroh Yoon. Towards a Rigorous Evaluation of Time-Series Anomaly Detection, AAAI Conference on Artificial Intelligence, 2022. \url{https://arxiv.org/abs/2109.05257}
\item Fei Tony Liu, Kai Ming Ting and Zhi-Hua Zhou. Isolation Forest, IEEE International Conference on Data Mining (ICDM), 2008. \url{https://doi.org/10.1109/ICDM.2008.17}
\item Marc B\"achlin, Meir Plotnik, Daniel Roggen, Inbal Maidan, Jeffrey M. Hausdorff, Nir Giladi and Gerhard Tr\"oster. Wearable Assistant for Parkinson's Disease Patients With the Freezing of Gait Symptom, IEEE Transactions on Information Technology in Biomedicine, 2010. \url{https://doi.org/10.1109/TITB.2009.2036165}
\item Ricardo Emanuel Vaz Vargas, Celso Jos\'e Munaro, Patrick Marques Ciarelli, Andr\'e Gon\c{c}alves Medeiros, Bruno Guberfain do Amaral, Daniel Centurion Barrionuevo, Jean Carlos Dias de Ara\'ujo, Jorge Lins Ribeiro and Lucas Pierezan Magalh\~aes. A Realistic and Public Dataset with Rare Undesirable Real Events in Oil Wells, Journal of Petroleum Science and Engineering, 2019. \url{https://doi.org/10.1016/j.petrol.2019.106223}
\item Bruno Veloso, Rita P. Ribeiro, Jo\~ao Gama and Pedro Mota Pereira. The MetroPT Dataset for Predictive Maintenance, Scientific Data, 2022. \url{https://doi.org/10.1038/s41597-022-01877-3}
\item Machine Learning Group, Universit\'e Libre de Bruxelles. \url{https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud}. Accessed on: 22nd November, 2026.
\item Daniel Roggen, Meir Plotnik and Jeff Hausdorff. \url{https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait}. Accessed on: 22nd November, 2026.
\item Petrobras 3W contributors. \url{https://github.com/petrobras/3W}. Accessed on: 22nd November, 2026.
\item Bruno Veloso, Jo\~ao Gama, Rita P. Ribeiro and Pedro Mota Pereira. \url{https://archive.ics.uci.edu/dataset/791/metropt+3+dataset}. Accessed on: 22nd November, 2026.
\item Hugging Face. \url{https://huggingface.co/openai/clip-vit-base-patch32}. Accessed on: 22nd November, 2026.
\end{itemize}
```

## 9. Acknowledgements

We thank the course instructor and teaching assistants of IE 643 (add names) for their feedback at the primary
assessment, and the providers of the datasets used: the Machine Learning Group of ULB (Credit Card), the UCI
Machine Learning Repository (Daphnet, MetroPT-3), and Petrobras (3W). We also thank OpenAI and Hugging Face for
the public CLIP weights.
