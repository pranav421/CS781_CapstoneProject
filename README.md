
# CS781 Capstone Project — Legal Language Modelling Challenge

This repository contains my work for the **CS781 Legal Language Modelling Challenge** at IIT Kanpur. I implemented four Transformer architectures for legal language modelling in Phase 1, then adapted the pretrained backbones for six downstream legal NLP tasks. The work covers **seven phases and 35 Jupyter notebooks**. I used full fine-tuning, partial fine-tuning, adapters, LoRA, IA³, BitFit, LORMA, and knowledge distillation, depending on the architecture and task.

The project was developed using **Python, PyTorch, Hugging Face libraries, and Kaggle/Colab GPUs**. The dataset is the official [CS781 Capstone dataset](https://huggingface.co/datasets/Exploration-Lab/CS-781-Capstone). The aim was to develop and evaluate compact legal models that can run without sending legal documents to an external inference API.

## 1. My Best Competition Results

These are the **best Codabench scores I recorded for each phase**, not values taken from the saved local notebook outputs. The seven tasks use different metrics, so scores should only be compared **within the same phase**.

| Phase | Task | Primary Metric | Best Score | Best Approach |
|---|---|---|---:|---|
| **1** | Legal Language Model Pretraining | BERTScore F1 | **0.8346** | A3 — Modern Local Transformer |
| **2** | Legal Named Entity Recognition | Strict Macro-F1 | **0.4889** | A3 — LoRA |
| **3** | Rhetorical Role Prediction | Macro-F1 | **0.4240** | A3 — Knowledge Distillation |
| **4** | Court Judgment Prediction | Macro-F1 | **0.6071** | Exact submitted model/checkpoint not identified in the retained score record |
| **5** | Legal Statute Identification | Macro-F1 | **0.3721** | A3 — LoRA |
| **6** | Abstractive Legal Summarization | BERTScore F1 | **0.8267** | A4 — LORMA |
| **7** | Prior Case Retrieval | Micro-F1@7 | **0.3385** | A3 — Knowledge Distillation and A4 — LORMA (tie) |

These scores come from my earlier competition records. I have kept local development metrics in separate tables below; they are **not** interchangeable with the private-test scores. I have not included leaderboard ranks or valid-upload counts because I do not have a checked final record of those fields.

---

## 2. Repository Structure

The project contains **35 Jupyter notebooks**, organized phase-wise. The notebooks include data loading, model architecture definitions, training, evaluation, inference, and submission generation.

```text
CS781-Capstone/
│
├── README.md
│
├── Phase1/
│   ├── architetcture-1.ipynb
│   ├── architecture2.ipynb
│   ├── Architecture3_Modern_Local_Transformer.ipynb
│   └── Architecture4.ipynb
│
├── Phase2/
│   ├── phase2a1-full.ipynb
│   ├── phase2-a1-adapters.ipynb
│   ├── phase2-a2.ipynb
│   ├── phase2-a3-LoRA.ipynb
│   ├── phase2-a3-distilled.ipynb
│   └── phase2-a4-lorma.ipynb
│
├── Phase3/
│   ├── A1_LoRA_Phase3_AUTH_FIXED.ipynb
│   ├── A2_IA3_Phase3_IMPROVED_STANDALONE.ipynb
│   ├── A3_Distillation_Phase3_UPDATED_STANDALONE.ipynb
│   └── A4_LORMA_Phase3_UPDATED_STANDALONE.ipynb
│
├── Phase4/
│   ├── phase4-a1-full.ipynb
│   ├── phase4-a1-adapters.ipynb
│   ├── phase4-a2-partialFineTuning.ipynb
│   ├── phase4-a2-bitfit.ipynb
│   ├── phase4-a3-teacher.ipynb
│   └── phase4-a3-student.ipynb
│
├── Phase5/
│   ├── p5a1-fullfinetuning.ipynb
│   ├── p5a2-partial.ipynb
│   ├── p5a3-lora.ipynb
│   ├── p5a3-distillation.ipynb
│   └── p5a4-lorma.ipynb
│
├── Phase6/
│   ├── Phase6_01_A1_Full_Phase5Style_REPETITIONFIX.ipynb
│   ├── Phase6_02_A2_Partial_Phase5Style_REPETITIONFIX.ipynb
│   ├── Phase6_04_A4_LORMA_Phase5Style_REPETITIONFIX.ipynb
│   └── Phase6_06_A3_Distillation_Phase5Style_REPETITIONFIX.ipynb
│
└── Phase7/
    ├── Phase7_01_A1_FULL.ipynb
    ├── Phase7_02_A2_PARTIAL.ipynb
    ├── Phase7_03_A4_LORMA.ipynb
    ├── Phase7_04_A3_KD.ipynb
    ├── Phase7_05_A1_ADAPTERS.ipynb
    └── Phase7_06_A3_LORA.ipynb
```

The filenames above match the notebook filenames, including the spelling of `architetcture-1.ipynb`.

Throughout the project, **A1, A2, A3, and A4** refer to the four Phase 1 architectures.

---

## 3. Phase 1 — Legal Language Model Pretraining

### 3.1 Objective and Dataset

The objective of Phase 1 was to train legal language models from scratch using the provided legal corpus.

The training corpus contains legal text from:

- Acts and statutory material
- High Court judgments
- Supreme Court judgments
- Tribunal documents

The models use a **50,000-token BPE vocabulary**, with up to **2,048 tokens per context**.

The decoder-style models learn next-token prediction, while the encoder–decoder model learns to generate a target sequence conditioned on a source sequence.

For the official test set, the model generates a **128-word continuation** for each legal prompt. The output must contain only the continuation, not the original prompt.

### 3.2 Architecture Exploration

I implemented four different Transformer architectures.

| Architecture | Main Components | Parameters |
|---|---|---:|
| **A1 — Decoder-Only** | Causal attention, learned positional embeddings, Pre-LayerNorm, GELU | **104.70M** |
| **A2 — Encoder–Decoder** | Bidirectional encoder, autoregressive decoder, cross-attention | **135.26M** |
| **A3 — Modern Local** | GQA, RoPE, RMSNorm, SwiGLU, local attention | **61.69M** |
| **A4 — Multi-Scale** | Local, dilated, and global causal attention | **58.15M** |

#### Architecture 1 — Decoder-Only Transformer

**Notebook:** `architetcture-1.ipynb`

This architecture uses a conventional autoregressive Transformer.

Configuration:

- 8 Transformer blocks
- Hidden dimension: 640
- 10 attention heads
- Feed-forward dimension: 2560
- Context length: 2048
- Learned positional embeddings
- Pre-LayerNorm
- GELU activation

Each token attends only to previous tokens. The output projection predicts the next token.

Residual connections and normalization are used throughout the Transformer stack.

#### Architecture 2 — Encoder–Decoder Transformer

**Notebook:** `architecture2.ipynb`

This architecture uses separate encoder and decoder stacks.

Configuration:

- 8 encoder layers
- 8 decoder layers
- Hidden dimension: 600
- 15 attention heads
- Source length: 1536
- Target length: 512
- Cross-attention between decoder and encoder

The encoder reads the source tokens using bidirectional attention. The decoder uses causal self-attention and encoder–decoder cross-attention to predict the target sequence.

The 2048-token training block is divided into 1536 source tokens and 512 target tokens.

#### Architecture 3 — Modern Local Transformer

**Notebook:** `Architecture3_Modern_Local_Transformer.ipynb`

This architecture uses modern attention and normalization components.

Configuration:

- 8 Transformer layers
- Hidden dimension: 600
- 10 query heads
- 5 key/value heads
- Local attention window: 512
- Attention chunk size: 256
- Rotary Position Embeddings (RoPE)
- RMSNorm
- SwiGLU

Grouped-query attention allows query heads to share key/value projections, reducing memory requirements.

Sliding-window attention restricts attention to a local context instead of constructing a full sequence-wide attention matrix.

The architecture also implements cached generation using `prefill` and `decode_step`.

#### Architecture 4 — Multi-Scale Transformer

**Notebook:** `Architecture4.ipynb`

This architecture combines different attention patterns.

Configuration:

- 8 Transformer layers
- Hidden dimension: 600
- 10 attention heads
- Feed-forward dimension: 1600
- Local attention windows: 256, 512, 1024
- Dilation factors: 2, 4, 8
- Local, dilated, and global causal attention

Different layers use different attention ranges to capture nearby and longer-range dependencies.

### 3.3 Pretraining Methodology

The pretraining workflow follows these steps:

1. Load the tokenizer and official legal corpus.
2. Build or load `train_tokens.bin` and `val_tokens.bin`.
3. Use memory-mapped arrays for token storage.
4. Split the token stream into fixed-length training blocks.
5. Initialize the selected Transformer architecture.
6. Train using cross-entropy loss.
7. Evaluate validation loss and perplexity.
8. Save the model checkpoint and tokenizer.
9. Generate 128-word test continuations.
10. Validate the prediction JSON and create the submission ZIP.

The recorded A1 and A2 training configurations used AdamW-style optimization, learning-rate warm-up and decay, small physical batch sizes, and gradient accumulation.

The later A1 training record used **150 million token IDs**, producing **73,242 complete blocks**, and took approximately **6.51 hours on two Tesla T4 GPUs**.

An earlier A1 notebook snapshot contained only 150,000 tokens and is treated separately.

The A2 run used approximately **200 million token IDs**, producing **97,656 complete blocks**, and took approximately **6.98 hours**.

### 3.4 Phase 1 Competition Results

| Architecture | BERTScore F1 |
|---|---:|
| A1 — Decoder-Only | 0.8066 |
| A2 — Encoder–Decoder | 0.8078 |
| **A3 — Modern Local** | **0.8346** |
| A4 — Multi-Scale | 0.8299 |

**My best Phase 1 competition score was 0.8346 using Architecture 3.**

The saved local pretraining records also contain:

| Architecture | Validation Loss | Perplexity |
|---|---:|---:|
| A1 | 7.069660 | 1175.75 |
| A2 | 3.558656 | 35.12 |

These are local training diagnostics, not BERTScore competition results. The training runs also differed in data exposure.

---

## 4. Fine-Tuning and Adaptation Methods

For Phases 2–7, I adapted the Phase 1 models using different fine-tuning techniques.

### Full Fine-Tuning

The entire pretrained backbone and task-specific output head are trained.

All eligible model parameters receive gradient updates.

### Partial Fine-Tuning

Most pretrained layers are frozen, while selected upper Transformer layers and the task head remain trainable.

Some implementations unfreeze the last two backbone blocks.

### Adapters

Small trainable bottleneck modules are inserted into the Transformer.

The main pretrained backbone remains frozen, while the adapter modules and task head are optimized.

### LoRA

Low-Rank Adaptation introduces two small trainable matrices to approximate a weight update.

The adapted projection is:

\[
y = W_0x + \frac{\alpha}{r}BAx
\]

Here, the original matrix \(W_0\) is frozen, and the low-rank matrices \(A\) and \(B\) are trained.

The LoRA implementations primarily target attention projections.

### LORMA

LORMA applies low-rank multiplicative adaptation to the output of a frozen projection.

The implementation uses the form:

\[
y = W_0x + \frac{\alpha}{r}BA(W_0x)
\]

This differs from directly adding a low-rank update to the weight matrix as in LoRA.

### IA³

IA³ introduces learned multiplicative scaling vectors for selected attention and feed-forward features.

Only these small scaling vectors and the task head need to be optimized.

### BitFit

BitFit trains bias parameters and the task-specific classification head while keeping most pretrained weights frozen.

### Knowledge Distillation

Knowledge distillation uses a trained teacher to supervise a smaller student.

The student learns from both the original training labels and teacher outputs.

Depending on the task, the loss combines supervised task loss with temperature-scaled KL divergence or another matching objective.

These methods were implemented as separate experiments. Changing hyperparameters alone was not treated as a different adaptation method.

---

## 5. Phase 2 — Legal Named Entity Recognition

### 5.1 Task

Phase 2 requires identifying legal entities in a document and returning their exact character positions and labels.

The 12 entity labels are:

`APP`, `RESP`, `A.COUNSEL`, `R.COUNSEL`, `JUDGE`, `WIT`, `AUTH`, `COURT`, `STAT`, `PREC`, `DATE`, and `CASENO`.

**Primary metric:** Strict Macro-F1.

### 5.2 Dataset

The official dataset contains two annotated training folds and a separate test fold.

```text
Phase_2_LNER/
├── Train/
│   ├── fold1-text.json
│   ├── fold1-gold.json
│   ├── fold2-text.json
│   ├── fold2-gold.json
│   └── labels.txt
└── Test/
    └── fold3-text.json
```

### 5.3 Methodology and Implementation

I converted gold character-level entity annotations into BIO token labels and trained a token-classification head on each selected Phase 1 backbone.

The implementation includes:

- Token classification
- BIO label encoding
- Character-to-token alignment
- Sliding-window processing
- Overlap-aware window merging
- Constrained Viterbi decoding
- Strict span-level evaluation

I used **256-token windows with a stride of 192**, producing a 64-token overlap.

Token offsets are preserved so that predictions can be converted back into exact character positions in the original document.

After inference, predictions from overlapping windows are merged.

I compared greedy stitching with constrained Viterbi decoding. In the recorded A1 diagnostic, strict local Macro-F1 improved from **0.2496 to 0.3582** after switching to Viterbi stitching.

This experiment used seven held-out documents containing 721 gold spans, so the scores are local diagnostics rather than private-test results.

### 5.4 Experiments and Competition Results

| Notebook | Backbone | Method | Strict Macro-F1 |
|---|---|---|---:|
| `phase2a1-full.ipynb` | A1 | Full Fine-Tuning | 0.3870 |
| `phase2-a1-adapters.ipynb` | A1 | Adapters | 0.4075 |
| `phase2-a2.ipynb` | A2 | Partial Fine-Tuning | 0.1358 |
| **`phase2-a3-LoRA.ipynb`** | **A3** | **LoRA** | **0.4889** |
| `phase2-a3-distilled.ipynb` | A3 | Knowledge Distillation | 0.4800 |
| `phase2-a4-lorma.ipynb` | A4 | LORMA | 0.4605 |

**My best Phase 2 competition result was 0.4889 using Architecture 3 with LoRA.**

The A3 distillation notebook also recorded a local strict Macro-F1 of 0.4445, which is separate from its 0.4800 competition result.

### 5.5 Output Format

```json
{
  "1128835": [
    [18, 40, "COURT"],
    [114, 162, "CASENO"]
  ]
}
```

---

## 6. Phase 3 — Rhetorical Role Prediction

### 6.1 Task

Phase 3 predicts one rhetorical-role label for every sentence in a legal document.

The dataset covers Corporate Law and Information Technology.

**Primary metric:** Macro-F1.

### 6.2 Dataset

```text
Phase_3_RR/
├── Train/
│   ├── CL_train.json
│   ├── CL_dev.json
│   ├── IT_train.json
│   ├── IT_dev.json
│   └── label_vocab.json
└── Test/
    ├── CL_test.json
    └── IT_test.json
```

### 6.3 Methodology and Implementation

I constructed sentence-level inputs using the current sentence and available neighboring context.

The selected Phase 1 backbone encodes the sentence and contextual information, and a classification head predicts one of the 13 official rhetorical-role labels.

The implementation includes:

- Sentence-level representations
- Contextual sentence input
- Classification heads
- Class-weighted training loss
- Macro-F1 evaluation
- Preservation of sentence ordering

The A1 notebook uses a 128-token input configuration and two training epochs.

The A2, A3, and A4 notebooks use 192-token input configurations with maximum training schedules of six epochs.

For knowledge distillation, I trained a smaller student using teacher classification probabilities.

For LORMA, I adapted attention projections within the A4 backbone.

### 6.4 Experiments and Competition Results

| Notebook | Backbone | Method | Macro-F1 |
|---|---|---|---:|
| `A1_LoRA_Phase3_AUTH_FIXED.ipynb` | A1 | LoRA | 0.2658 |
| `A2_IA3_Phase3_IMPROVED_STANDALONE.ipynb` | A2 | IA³ | 0.3733 |
| **`A3_Distillation_Phase3_UPDATED_STANDALONE.ipynb`** | **A3** | **Knowledge Distillation** | **0.4240** |
| `A4_LORMA_Phase3_UPDATED_STANDALONE.ipynb` | A4 | LORMA | 0.4070 |

**My best Phase 3 competition result was 0.4240 using A3 knowledge distillation.**

Saved execution logs separately recorded:

- A3 student: 0.363219 local Macro-F1
- A4 LORMA: 0.347900 local Macro-F1
- A4 sentence-only diagnostic: 0.271767 local Macro-F1

The contextual A4 model performed better than its sentence-only diagnostic in that recorded experiment.

Rare labels such as `Dissent` and `PrecedentOverruled` still had zero F1 in one per-class evaluation.

The final four Phase 3 notebook copies contain code but no saved execution output cells. The scores above are from earlier project records.

### 6.5 Output Format

```json
{
  "document_001": [
    "Fact",
    "Issue",
    "ArgumentPetitioner"
  ]
}
```

---

## 7. Phase 4 — Court Judgment Prediction

### 7.1 Task

Phase 4 predicts a binary court judgment outcome.

The output is either `0` or `1`.

**Primary metric:** Macro-F1.

### 7.2 Dataset

```text
Phase_4_CJPE/
├── Train/
│   ├── train.json
│   └── dev.json
└── Test/
    └── test.json
```

### 7.3 Methodology and Implementation

Court judgments are usually longer than the model's practical context length.

I implemented a chunk-based classification approach.

The workflow is:

1. Read the full case document.
2. Select up to three 192-token chunks.
3. Use document sections from the beginning, middle, and end.
4. Encode each chunk using the selected backbone.
5. Pool the token representations into chunk-level vectors.
6. Average the chunk vectors.
7. Apply a binary classification head.
8. Train using class-weighted cross-entropy.
9. Evaluate development Macro-F1.
10. Save the selected checkpoint and generate predictions.

The A2 encoder uses mean pooling over token states.

The causal backbones use final-position representations in this Phase 4 implementation.

Some saved evaluation logs contained 1,988 document rows, while another evaluation used 994 unique cases. I kept these evaluation units separate instead of comparing them directly.

#### A3 Knowledge Distillation

The A3 teacher notebook trains a full classifier and saves `best_model.pt`.

The student notebook:

- Loads the trained teacher.
- Computes teacher logits on the training cases.
- Initializes a smaller four-layer A3 student.
- Trains using supervised cross-entropy and temperature-scaled KL divergence.
- Evaluates the student on development cases.

Teacher and student use matching chunk settings.

### 7.4 Experiments and Local Development Results

| Notebook | Backbone | Method | Local Macro-F1 |
|---|---|---|---:|
| `phase4-a1-full.ipynb` | A1 | Full Fine-Tuning | 0.5888 |
| `phase4-a1-adapters.ipynb` | A1 | Adapters | 0.5574 |
| `phase4-a2-partialFineTuning.ipynb` | A2 | Partial Fine-Tuning | 0.5108 |
| `phase4-a2-bitfit.ipynb` | A2 | BitFit | 0.4602 |
| `phase4-a3-teacher.ipynb` | A3 | Teacher Fine-Tuning | 0.6143 |
| `phase4-a3-student.ipynb` | A3 | Knowledge Distillation | **0.6202** |

### 7.5 Competition Result

**My best reported Phase 4 Codabench Macro-F1 was 0.6071.**

The exact checkpoint associated with that submission was not established in the retained score record, so I have not assigned the score to an unsupported model version.

The table above contains local development results, not competition results.

### 7.6 Output Format

```json
[
  {
    "id": "case_001",
    "pred": 1
  },
  {
    "id": "case_002",
    "pred": 0
  }
]
```

---

## 8. Phase 5 — Legal Statute Identification

### 8.1 Task

Phase 5 predicts the statute IDs relevant to a legal document.

It is a multi-label classification task because one document can involve multiple statutes.

**Primary metric:** Macro-F1.

The saved data analysis records 100 statute labels and 10,181 development documents.

### 8.2 Dataset

```text
Phase_5_LSI/
├── Train/
│   ├── train.json
│   └── dev.json
└── Test/
    └── test.json
```

### 8.3 Methodology and Implementation

I implemented a `StatuteModel` with a linear classification head that produces one logit for every statute.

Instead of representing a long document using only its beginning, I selected multiple evidence windows.

The `evidence_texts` function selects:

- Text from the beginning of the document
- A statute-evidence window based on cues such as `section`, `article`, and `Act`
- Text from the ending of the document

Each window is independently encoded.

The model combines average and maximum window representations before classification.

Maximum pooling helps preserve strong statute evidence even when it appears in only one selected window.

I used `BCEWithLogitsLoss` for multi-label prediction.

Positive-label weights are calculated from label support in the training data and clipped to control excessively large weights.

The implementation also includes:

- Duplicate statute-ID removal
- Global threshold calibration
- Smoothed per-label thresholds
- Development-set model selection
- Optional TF-IDF nearest-neighbor probability blending
- Multi-label output validation

Thresholds were selected using development data, not the private test labels.

#### A3 LoRA Configuration

The A3 LoRA notebook configures:

| Parameter | Value |
|---|---|
| Epochs | 5 |
| Batch Size | 1 |
| Gradient Accumulation | 4 |
| Sequence Length | 256 |
| Chunks | 3 |
| Random Seed | 44 |

The recorded A3 LoRA development run selected a threshold of **0.375** and achieved **0.380459 local Macro-F1**.

### 8.4 Experiments and Competition Results

| Notebook / Experiment | Backbone | Method | Macro-F1 |
|---|---|---|---:|
| `p5a1-fullfinetuning.ipynb` | A1 | Full Fine-Tuning | 0.3193 |
| Earlier A1 adapter experiment | A1 | Adapters | 0.2728 |
| `p5a2-partial.ipynb` | A2 | Partial Fine-Tuning | 0.2926 |
| **`p5a3-lora.ipynb`** | **A3** | **LoRA** | **0.3721** |
| `p5a3-distillation.ipynb` | A3 | Knowledge Distillation | 0.3408 |
| `p5a4-lorma.ipynb` | A4 | LORMA | 0.3004 |

The earlier A1 adapter experiment is included in my reported score history, although its separate notebook is not present in the final 35-notebook archive.

**My best Phase 5 competition score was 0.3721 using A3 LoRA.**

The corresponding local development results include:

- A3 LoRA: 0.380459 Macro-F1
- A3 Knowledge Distillation: 0.354089 Macro-F1

### 8.5 Implementation Files

The notebooks dynamically generate supporting Python scripts using `%%writefile` cells.

```text
phase5_core.py
phase5_train.py
phase5_predict.py
phase5_validate.py
phase5_distill.py
phase5_official_evaluator.py
phase5_official_bridge.py
```

These scripts handle model construction, training, prediction, evaluation, distillation, and submission validation.

### 8.6 Output Format

```json
[
  {
    "id": "case_001",
    "pred": [1, 7, 23]
  }
]
```

---

## 9. Phase 6 — Abstractive Legal Summarization

### 9.1 Task

Phase 6 generates a concise summary of a legal document.

Unlike extractive summarization, the model can generate new text while preserving the important information from the original document.

**Primary metric:** BERTScore F1.

Secondary metrics include ROUGE-1 and ROUGE-L.

### 9.2 Dataset

```text
Phase_6_Summ/
├── Train/
│   └── train.json
└── Test/
    └── test.json
```

The official dataset does not contain a separate public development split.

The code creates a held-out development split.

The saved report records:

- 7,028 usable training documents
- A 64-document local scoring subset

### 9.3 Methodology and Implementation

I adapted the pretrained Phase 1 backbones for document-to-summary generation.

Long documents are processed using a **head–middle–tail** strategy.

The `make_prompt` and `doc_to_tokens` functions select document material from the beginning, middle, and end.

This avoids relying only on the beginning of a document when its complete text exceeds the input length.

A2 uses its encoder–decoder architecture.

The other backbones use their causal generation path.

Training uses token-level cross-entropy for summary generation.

Source and padding tokens are handled appropriately for the selected architecture.

#### Repetition Control

Some generated summaries contained repeated phrases.

I added no-repeat four-gram decoding:

```python
NO_REPEAT_NGRAM = 4
```

The implementation also includes:

- Generation-length checks
- Repetition diagnostics
- Finite-value checks
- Checkpoint saving and recovery
- Validation output generation
- BERTScore evaluation on held-out examples

The recorded A4 LORMA training included one non-finite step that was skipped.

#### A4 LORMA Configuration

| Parameter | Value |
|---|---|
| Epochs | 3 |
| Batch Size | 1 |
| Gradient Accumulation | 4 |
| Maximum Source Length | 448 |
| Maximum Target Length | 112 |
| Decode Length | 112 |
| No-Repeat N-Gram | 4 |
| Random Seed | 44 |

The A4 local evaluation achieved **0.823332 BERTScore F1** on 64 held-out examples.

### 9.4 Experiments and Competition Results

| Notebook | Backbone | Method | BERTScore F1 |
|---|---|---|---:|
| `Phase6_01_A1_Full_Phase5Style_REPETITIONFIX.ipynb` | A1 | Full Fine-Tuning with NR4 | 0.8163 |
| `Phase6_02_A2_Partial_Phase5Style_REPETITIONFIX.ipynb` | A2 | Partial Fine-Tuning | 0.8149 |
| `Phase6_06_A3_Distillation_Phase5Style_REPETITIONFIX.ipynb` | A3 | Improved Knowledge Distillation | 0.8024 |
| **`Phase6_04_A4_LORMA_Phase5Style_REPETITIONFIX.ipynb`** | **A4** | **Improved LORMA** | **0.8267** |

**My best Phase 6 competition score was 0.8267 using A4 LORMA.**

The archived Phase 6 notebooks do not preserve executed output cells. The competition scores and local evaluation result come from my earlier project records.

### 9.5 Implementation Files

The notebooks generate:

```text
phase6_backbones/
phase6_core.py
phase6_train.py
phase6_predict.py
phase6_validate.py
phase6_evaluate.py
phase6_hf.py
```

These modules provide the backbone reconstruction, training, prediction, validation, evaluation, and checkpoint-handling logic.

### 9.6 Output Format

```json
[
  {
    "id": "doc_001",
    "summary": "Generated legal summary."
  }
]
```

---

## 10. Phase 7 — Prior Case Retrieval

### 10.1 Task

Phase 7 retrieves the seven most relevant prior cases for each query.

The candidate cases must come from the official candidate pool for the corresponding split.

**Primary metric:** Micro-F1@7.

The saved dataset record contains:

| Split | Query Count |
|---|---:|
| Training | 827 |
| Development | 118 |
| Test | 237 |

### 10.2 Dataset

```text
Phase_7_PCR/
├── Train/
│   ├── train.json
│   └── train_candidates.json
├── Dev/
│   ├── dev.json
│   └── dev_candidates.json
└── Test/
    ├── test.json
    └── test_candidates.json
```

### 10.3 Methodology and Implementation

I implemented a bi-encoder retrieval model using the selected Phase 1 backbone.

The same model encodes both query cases and candidate cases.

For long documents, I used two chunks per case, covering the beginning and ending sections.

Each chunk is configured for up to 256 tokens.

The chunk vectors are averaged and L2-normalized to construct a document-level representation.

#### Retrieval Training

The training code constructs query–positive–negative examples using the official training split.

The model computes similarity scores between:

- The query and a relevant candidate
- The query and a hard negative candidate

A two-way cross-entropy ranking objective encourages higher similarity for the relevant candidate.

The knowledge-distillation implementation additionally matches the teacher's relevance distribution using KL divergence.

#### Hybrid Retrieval

During inference, I combined dense and lexical similarity.

The implementation computes:

1. Dense neural embedding similarity
2. TF-IDF unigram/bigram similarity
3. Standardized similarity scores
4. A weighted hybrid score

The implementation uses:

```python
HYBRID_ALPHA = 0.35
```

Candidates are sorted by their combined relevance scores.

The seven highest-ranked valid candidates are selected.

#### A4 LORMA Configuration

| Parameter | Value |
|---|---|
| Maximum Tokens | 256 |
| Chunks per Case | 2 |
| Embedding Batch Size | 6 |
| Hybrid Alpha | 0.35 |
| Epochs | 3 |
| Learning Rate | 1e-4 |
| Gradient Accumulation | 4 |
| Checkpoint Save Interval | 40 steps |

The recorded A4 development Micro-F1@7 was **0.356865** on 118 queries.

### 10.4 Experiments and Competition Results

| Notebook | Backbone | Method | Micro-F1@7 |
|---|---|---|---:|
| `Phase7_01_A1_FULL.ipynb` | A1 | Full Fine-Tuning | 0.3308 |
| `Phase7_05_A1_ADAPTERS.ipynb` | A1 | Adapters | 0.3331 |
| `Phase7_02_A2_PARTIAL.ipynb` | A2 | Partial Fine-Tuning | 0.3142 |
| **`Phase7_04_A3_KD.ipynb`** | **A3** | **Knowledge Distillation** | **0.3385** |
| `Phase7_06_A3_LORA.ipynb` | A3 | LoRA | 0.3337 |
| **`Phase7_03_A4_LORMA.ipynb`** | **A4** | **LORMA** | **0.3385** |

**A3 knowledge distillation and A4 LORMA tied for my best Phase 7 competition score of 0.3385.**

The Phase 7 notebook copies contain implementation code but no saved executed outputs. The scores come from my earlier records.

### 10.5 Implementation Files

The notebooks generate:

```text
phase7_backbones.py
phase7_core.py
phase7_setup.py
```

These handle backbone loading, model setup, training, retrieval, ranking, and submission generation.

### 10.6 Output Format

```json
{
  "0000021652": [
    "0000590455",
    "0000916840",
    "0001481813",
    "0001907199",
    "0000000987",
    "0000001566",
    "0000002886"
  ]
}
```

Every query must have exactly seven unique candidate IDs.

IDs are preserved as strings to retain leading zeroes.

---

## 11. Execution Instructions

### 11.1 Requirements

The notebooks were developed primarily for Kaggle GPU environments.

Requirements include:

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Hub
- Tokenizers
- Safetensors
- NumPy
- Scikit-learn
- PEFT where used
- tqdm
- joblib
- BERTScore for local summarization evaluation
- GPU-enabled Kaggle or equivalent environment

The repository contains the source notebooks, but not all trained checkpoints, tokenized data files, or the original Codabench submission records.

Reproducing downstream experiments requires the corresponding Phase 1 checkpoints and tokenizers.

### 11.2 Step 1 — Open the Notebook

Open the required notebook in Kaggle.

For example:

```text
Phase5/p5a3-lora.ipynb
```

Enable the GPU accelerator and Internet access.

I use a fresh session for individual experiments because the library versions and working directories differ between phases.

### 11.3 Step 2 — Configure Hugging Face Authentication

Some notebooks need authenticated access to model and data repositories.

In Kaggle:

1. Open **Add-ons → Secrets**.
2. Create a secret named `HF`.
3. Provide an authorized Hugging Face token.
4. Enable notebook access to the secret.

Some scripts also read `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`.

Credentials should not be committed to the public repository.

The Phase 6 code has the following Phase 1 repository defaults:

```text
A1  P-261110044/cs781-a1-checkpoints
A2  P-261110044/Phase1-A2
A3  P-261110044/cs781-phase1-architecture3
A4  P-261110044/cs781-a4-checkpoints
```

These are configuration values from the notebooks, not guarantees of public checkpoint access.

Phase 6 supports checkpoint configuration variables including:

```text
PHASE1_REPO
PHASE1_DIR
PHASE1_WEIGHT
```

Phase 7 supports:

```text
LOCAL_PHASE1_DIR
LOCAL_PHASE7_DATA_DIR
```

### 11.4 Step 3 — Install Dependencies

The notebooks use phase-specific environments.

| Phase | Dependency Configuration |
|---|---|
| 1 | Use the versions requested in the selected pretraining notebook |
| 2–5 | Mainly `transformers>=4.45,<5` |
| 6 | `transformers>=5.0,<6` and `bert-score==0.3.13` for local scoring |
| 7 | `transformers==4.57.1`, `huggingface_hub>=0.34,<1`, `tokenizers>=0.22,<0.23` |

I run the dependency installation and version-check cells before importing the models.

If package installation requires a runtime restart, I restart before executing the training cells.

### 11.5 Step 4 — Run Phase 1

For the selected Phase 1 notebook, the execution order is:

```text
Hugging Face Authentication
          |
          v
GPU and Environment Check
          |
          v
Load Tokenizer
          |
          v
Load Official Legal Data
          |
          v
Prepare Tokenized Training Data
          |
          v
Initialize Transformer Architecture
          |
          v
Train Model
          |
          v
Evaluate Validation Loss
          |
          v
Save Checkpoint
          |
          v
Generate Test Continuations
          |
          v
Create prediction.json
          |
          v
Create submission.zip
```

The tokenizer and checkpoint must correspond to the same architecture and vocabulary.

### 11.6 Step 5 — Run Downstream Experiments

For Phases 2–7:

1. Load the official task dataset.
2. Restore the appropriate Phase 1 backbone.
3. Attach the task-specific prediction components.
4. Apply the selected fine-tuning method.
5. Train using the corresponding task objective.
6. Evaluate on the development set.
7. Save the selected checkpoint.
8. Generate test predictions.
9. Validate the output format.
10. Create `submission.zip`.

The task-specific notebooks are the execution entry points.

#### Knowledge Distillation Order

Knowledge-distillation runs require a trained teacher checkpoint before the student training begins.

For Phase 4:

```text
phase4-a3-teacher.ipynb
          |
          v
Train Teacher
          |
          v
Save best_model.pt
          |
          v
phase4-a3-student.ipynb
          |
          v
Load Teacher
          |
          v
Train Student
          |
          v
Evaluate and Generate Predictions
```

The other distillation notebooks also require the corresponding trained task-specific teacher model.

### 11.7 Step 6 — Running Phase 5 A3 LoRA

Open:

```text
Phase5/p5a3-lora.ipynb
```

The notebook creates:

```text
/kaggle/working/phase5_a3_lora
```

Run the setup and `%%writefile` cells first.

These cells generate the Python modules needed by the training commands.

The training command is:

```bash
python phase5_train.py --arch a3 --method lora \
  --out runs/a3_lora --epochs 5 --batch-size 1 \
  --accum 4 --length 256 --chunks 3 --seed 44 --lexical-knn
```

For local evaluation:

```bash
python phase5_official_evaluator.py \
  --truth runs/a3_lora/dev_gold.json \
  --prediction runs/a3_lora/dev_prediction.json
```

Generate test predictions:

```bash
python phase5_predict.py \
  --run runs/a3_lora \
  --out submissions/a3_lora
```

Validate the generated archive:

```bash
python phase5_validate.py \
  --pred submissions/a3_lora/submission.zip \
  --split test
```

These commands assume the generated scripts, official data, and correct model checkpoints are available in the notebook working environment.

### 11.8 Step 7 — Running Phase 6 A4 LORMA

Open:

```text
Phase6/Phase6_04_A4_LORMA_Phase5Style_REPETITIONFIX.ipynb
```

The notebook creates:

```text
/kaggle/working/phase6_a4_lorma
```

Run the dependency, setup, backbone reconstruction, and `%%writefile` cells.

The configured training command is:

```bash
python phase6_train.py --arch a4 --method lorma --out runs/a4_lorma \
  --epochs 3 --batch-size 1 --accum 4 --max-source 448 \
  --max-target 112 --decode-length 112 --dev-eval-max 64 \
  --seed 44 --phase1-repo P-261110044/cs781-a4-checkpoints
```

After training:

1. Load the best saved checkpoint.
2. Generate summaries using no-repeat four-gram decoding.
3. Validate the JSON output.
4. Generate `submission.zip`.
5. Optionally calculate local BERTScore using `phase6_evaluate.py`.
6. Back up the trained model files where configured.

The local BERTScore is not the official private-test score.

### 11.9 Step 8 — Running Phase 7 A4 LORMA

Open:

```text
Phase7/Phase7_03_A4_LORMA.ipynb
```

The notebook uses:

```python
ARCH = 'a4'
METHOD = 'lorma'
SLUG = 'a4_lorma'

MAX_TOKENS = 256
CHUNKS_PER_CASE = 2
EMBED_BATCH_SIZE = 6

HYBRID_ALPHA = 0.35

EPOCHS = 3
LR = 1e-4
ACCUM_STEPS = 4
SAVE_STEPS = 40
```

First execute the `%%writefile` cells to generate:

```text
phase7_backbones.py
phase7_core.py
phase7_setup.py
```

Then the notebook runs the following logical sequence:

```text
Download Phase 1 Backbone
          |
          v
Download Phase 7 Dataset
          |
          v
Initialize Retrieval Model
          |
          v
Train Retrieval Model
          |
          v
Evaluate Development Retrieval
          |
          v
Generate Candidate Rankings
          |
          v
Combine Dense and TF-IDF Scores
          |
          v
Select Top 7 Candidates
          |
          v
Validate Candidate IDs
          |
          v
Generate submission.zip
```

The code uses functions including:

- `download_phase1`
- `download_phase7`
- `setup_model`
- `train_model`
- `rank_split`
- `prepare_submission`

The experiment supports checkpoint restoration using `last.pt` and `best.pt` where those saved files are available.

### 11.10 Step 9 — Submission Validation

Every phase uses the same archive layout:

```text
submission.zip
└── prediction.json
```

`prediction.json` must be directly at the ZIP root, without an additional folder.

| Phase | Validation Requirements |
|---|---|
| Phase 1 | All test prompt IDs and 128-word continuations |
| Phase 2 | Exact character spans and valid entity labels |
| Phase 3 | One rhetorical label per sentence, in correct order |
| Phase 4 | One integer `0` or `1` per case |
| Phase 5 | Unique, valid integer statute IDs |
| Phase 6 | One generated summary per test document |
| Phase 7 | Exactly seven unique, ranked candidate IDs from the permitted test pool |

A generated ZIP is not by itself evidence that the competition evaluator accepted or scored the submission.

---

## 12. Problems Encountered and Fixes

| Phase | Problem | Change or Validation |
|---|---|---|
| 1 | An early A1 snapshot contained 150k tokens, while the later run used 150M | Kept the runs separate and checked the actual data count |
| 1 | Missing checkpoints and invalid generation outputs | Checked checkpoint paths, required IDs, and continuation word count |
| 2 | Incorrect entity boundaries and overlapping-window predictions | Preserved tokenizer offsets and used constrained Viterbi stitching |
| 3 | Rare rhetorical roles had zero per-class F1 | Checked class-level performance and used surrounding sentence context |
| 4 | Evaluation counted 1,988 rows in one path and 994 unique cases in another | Separated row-level and case-level scoring |
| 5 | Duplicate statute IDs and threshold-sensitive predictions | Deduplicated labels and calibrated thresholds using development data |
| 6 | Repeated generated text and a non-finite training step | Added no-repeat four-gram decoding, finite-value checks, and checkpoint handling |
| 7 | Candidate IDs with leading zeroes or candidates outside the permitted pool | Preserved string IDs and validated all seven predictions against the correct candidate pool |

The experiments did not all use identical compute budgets, training schedules, or checkpoints.

The competition results therefore represent the performance of the recorded implementations and settings, not a controlled claim that one backbone always outperforms the others.

---

## 13. Final Project Submission

The final code submission contains the best-performing model implementation for each phase.

The GitHub repository documents the additional architectures and fine-tuning experiments as well.

For Phase 4, my best reported competition score is **0.6071**, but I could not reliably map it to one specific saved checkpoint.

**Final submission deadline: 9 November 2026, 11:59 PM IST.**

The expected report and code archive structure is:

```text
CS781-CP-Firstname-Lastname-RollNo-report_code.zip
│
├── code/
│   ├── phase1/
│   ├── phase2/
│   ├── phase3/
│   ├── phase4/
│   ├── phase5/
│   ├── phase6/
│   └── phase7/
│
└── report/
    └── CS781-CP-Firstname-Lastname-RollNo-report.pdf
```

The GitHub repository is intended to be public for the final submission, and its URL must be included in the final report.

Private access tokens and account credentials are not part of the source-code repository.

---

## 14. References

1. [CS781 Capstone Official Dataset](https://huggingface.co/datasets/Exploration-Lab/CS-781-Capstone)
2. [IL-TUR: Benchmark for Indian Legal Text Understanding and Reasoning](https://arxiv.org/abs/2407.05399)
3. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
4. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
5. [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
6. [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751)
7. [BitFit: Simple Parameter-Efficient Fine-Tuning for Transformer-Based Masked Language Models](https://arxiv.org/abs/2106.10199)
8. [Few-Shot Parameter-Efficient Fine-Tuning Is Better and Cheaper than In-Context Learning (IA³)](https://arxiv.org/abs/2205.05638)

---

The competition scores in this README come from my reported Codabench records. Local validation results are identified separately.

I have not treated an unexecuted notebook, a configured training step, or a local validation metric as proof of a completed private-test submission.
