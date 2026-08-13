# CalibRAG

## Calibration-Guided Retrieval-Augmented Generation

CalibRAG is a research-oriented Retrieval-Augmented Generation (RAG) framework designed to improve answer reliability by combining **dense retrieval, cross-encoder reranking, claim-level verification, Natural Language Inference (NLI), and verification-guided retrieval**.

The project provides a controlled experimental environment for comparing CalibRAG against multiple RAG approaches under a shared retrieval pipeline, shared generator model, and controlled retrieval/iteration budgets.

---

## 📌 Overview

Traditional RAG systems generally follow a simple pipeline:

```text
Question
   ↓
Retrieve Documents
   ↓
Generate Answer
```

While this approach can work well, retrieved context may be incomplete, irrelevant, or insufficient to support every part of an answer.

CalibRAG introduces additional verification steps:

```text
                    ┌─────────────────────┐
                    │      Question       │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Dense Retrieval     │
                    │      + FAISS        │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Cross-Encoder       │
                    │ Reranking           │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Candidate Context   │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Claim Decomposition │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ NLI Verification    │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Sufficiency Check   │
                    └──────────┬──────────┘
                         ┌─────┴─────┐
                         │           │
                      Enough?      Not Enough
                         │           │
                         ↓           ↓
                 Final Answer   More Retrieval
                                     │
                                     └──────→ Verification
```

The notebook uses a modular architecture where the retrieval stage is shared across baselines to make comparisons more controlled and reproducible.

---

## 🎯 Objectives

The main objectives of this project are:

* Build a reproducible RAG experimentation framework.
* Improve retrieval quality using dense embeddings and reranking.
* Verify retrieved evidence before generating the final answer.
* Evaluate answer quality using multiple RAG baselines.
* Investigate whether claim-level verification can improve answer reliability.
* Control retrieval and reasoning budgets for fair baseline comparison.
* Maintain checkpoints and experiment states for long-running experiments.
* Track accuracy, calibration, efficiency, robustness, and statistical significance.

---

## ✨ Key Features

### 1. Reproducible Experiments

The project uses a global random seed of `42` and stores configuration and environment information for reproducibility.

The experiment configuration includes:

* Dataset configuration
* Chunk size and overlap
* Embedding model
* Retrieval parameters
* Reranking parameters
* Generator model
* NLI model
* Verification thresholds
* Retrieval budget
* Statistical testing parameters

The notebook also stores an `experiment_state.json` file to track completed sections and resume interrupted experiments.

---

### 2. HotpotQA Dataset

The current experiment uses:

```text
Dataset: hotpotqa/hotpot_qa
Configuration: distractor
Split: validation
```

The notebook supports configurable experiment sizes:

```text
development → 50 samples
research    → 1000 samples
```

The sample size can also be manually overridden.

For the current development experiment:

```text
Number of questions: 50
Seed: 42
```

The dataset records include:

* Question ID
* Question
* Gold answer
* Context documents
* Supporting facts
* Question type
* Difficulty level

The current 50-question development run contained 41 bridge questions and 9 comparison questions, with all 50 marked as hard.

---

## 🧩 System Architecture

The complete pipeline consists of several stages.

### Stage 1 — Dataset Loading

HotpotQA validation data is loaded using Hugging Face `datasets`.

The sampled dataset is checkpointed so that subsequent notebook executions do not need to reload and resample the data.

---

### Stage 2 — Context Chunking

Context documents are divided into overlapping chunks.

Current configuration:

```text
Chunk size: 150 tokens
Overlap:    20 tokens
```

The implementation uses whitespace-based tokenization for chunk construction.

Each chunk stores metadata including:

* Chunk ID
* Global chunk index
* Question ID
* Document title
* Original sentence IDs

This allows retrieved chunks to be traced back to their original context.

For the current 50-question development run:

```text
Total chunks:       547
Average/question:   10.94
Average length:     81.2 tokens
Maximum length:     150 tokens
```

---

## 🔎 Dense Retrieval

The project uses:

```text
Embedding Model:
sentence-transformers/all-mpnet-base-v2
```

Each context chunk is converted into a normalized 768-dimensional embedding.

The embeddings are indexed using:

```text
FAISS IndexFlatIP
```

Because the embeddings are normalized, inner-product search corresponds to cosine similarity.

Current development experiment:

```text
Chunks embedded:      547
Embedding dimension:  768
FAISS index size:     547
Top-k retrieval:      8
```

---

## 🔁 Retrieval Scoping

An important implementation detail is that retrieval is scoped to the context belonging to the current HotpotQA question.

The system maintains:

```text
question_id → FAISS row indices
```

This prevents a question from retrieving chunks belonging to another question's context.

The same retrieval implementation is shared across the different RAG baselines to support fair comparison.

---

## 📊 Dense Retrieval Results

For the current 50-question development run:

```text
Questions processed:        50
Configured top-k:            8
Average chunks/query:        8
Average top-1 similarity:    0.6568
```

---

## 🎯 Cross-Encoder Reranking

After dense retrieval, the retrieved candidates are reranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The reranker evaluates the question and each retrieved chunk together.

Current configuration:

```text
Dense retrieved:     8 chunks
Reranked output:     4 chunks
```

The top four chunks are retained for downstream processing.

For the current experiment:

```text
Questions processed:       50
Average chunks kept:        4
Average top-1 rerank score: 5.7685
```

---

# 🤖 Generator Model

The shared generator model is:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

The model is used consistently across the RAG baselines.

Configuration:

```text
Maximum new tokens: 256
Temperature:        0.3
4-bit loading:      Enabled on CUDA
```

The notebook uses the model's chat template and explicitly handles:

* Attention masks
* Padding
* Pad token ID
* Generated-token slicing

The generator was successfully loaded on CUDA using 4-bit quantization in the recorded experiment.

---

# 🧪 RAG Baselines

CalibRAG is evaluated alongside multiple RAG strategies.

The framework currently implements the following baseline approaches:

1. Vanilla RAG
2. Self-RAG
3. CRAG
4. Adaptive RAG
5. AdaKG-RAG
6. CalibRAG

The baseline architecture is based on a common `BaseRAG` interface.

All approaches share:

* The same initial retrieval results
* The same embedding model
* The same FAISS index
* The same cross-encoder
* The same generator LLM
* The same retrieval budget
* The same maximum iteration constraints

Only the verification and answer-generation strategy differs between approaches.

---

# 📚 Vanilla RAG

Vanilla RAG acts as the basic reference implementation.

Pipeline:

```text
Question
   ↓
Retrieve
   ↓
Rerank
   ↓
Generate Answer
```

There is no additional verification or iterative retrieval.

It establishes the baseline performance of a conventional RAG pipeline.

---

# 🔄 Self-RAG

Self-RAG introduces self-reflection into the retrieval process.

The implementation uses the shared generator model as a critic for:

### ISREL-style relevance checking

Determines whether retrieved chunks are relevant to the question.

### ISSUP-style support checking

Determines whether the generated answer is supported by the retrieved evidence.

If the evidence is insufficient, another retrieval round can be triggered.

The implementation is an experimental adaptation rather than the original trained Self-RAG critic architecture.

---

# 🧠 CRAG

CRAG introduces retrieval evaluation and corrective retrieval behavior.

In this implementation, the shared generator is used as the evaluator instead of a separately trained retrieval evaluator.

The corrective retrieval operates over the fixed per-question context rather than using live web search.

This design keeps the comparison within a controlled retrieval environment.

---

# 🔀 Adaptive RAG

Adaptive RAG dynamically determines whether additional retrieval or reasoning is required.

The experiment uses the generator model as the query-complexity classifier rather than a separately trained classifier.

The configuration specifies:

```text
adaptive_rag_complexity_model = generator
```

---

# 🕸️ AdaKG-RAG

AdaKG-RAG uses LLM self-reported confidence as its verification mechanism.

The model generates:

```text
Answer
Confidence
```

The confidence is parsed into a value between `0` and `1`.

Current threshold:

```text
Confidence threshold: 0.70
```

If confidence falls below the threshold, another retrieval round may be triggered.

This provides a comparison against CalibRAG's NLI-based verification mechanism.

---

# 🧮 CalibRAG

CalibRAG is the central experimental approach.

Instead of relying only on an LLM's self-reported confidence, the framework introduces explicit evidence verification.

The configuration includes:

```text
NLI Model:
MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli

Entailment confidence threshold:
0.70

Claim support threshold:
0.60

Maximum retrieval rounds:
3
```

The intended workflow is:

```text
Question
   ↓
Retrieve Evidence
   ↓
Rerank Evidence
   ↓
Generate / Decompose Claims
   ↓
Claim-Level NLI Verification
   ↓
Aggregate Evidence Sufficiency
   ↓
 ┌─────────────────────┐
 │ Evidence sufficient?│
 └─────────┬───────────┘
       Yes │       No
           │        │
           ↓        ↓
      Final Answer  New Retrieval
                    ↓
              Re-verification
```

The experiment configuration explicitly defines claim-level NLI verification, sufficiency aggregation, and a verification-guided retrieval loop as core experiment sections.

---

# ⚖️ Fair Baseline Comparison

A major design goal of the project is controlled comparison.

The framework attempts to prevent a baseline from receiving an unfair advantage through a different retrieval system or unlimited retrieval calls.

Shared constraints include:

```text
Maximum retrieval calls: 3
Maximum iterations:      3
```

The initial retrieval and reranking pipeline is shared by all approaches.

This makes the primary experimental difference the **verification and reasoning strategy**, rather than the underlying retrieval infrastructure.

---

# 💾 Checkpointing & Fault Tolerance

The notebook is designed for long-running experiments.

It automatically stores intermediate artifacts and tracks completed sections.

The experiment directory contains:

```text
checkpoints/
datasets/
embeddings/
faiss_index/
retrieval/
reranking/
baseline/
hypotheses/
claims/
nli_scores/
aggregation/
iterations/
answers/
evaluation/
metrics/
tables/
figures/
logs/
configs/
final_results/
```

Per-question checkpointing is used for retrieval and reranking.

The experiment state records:

* Completed sections
* Completion timestamps
* Resume pointers
* Configuration snapshot

This allows an interrupted experiment to resume without unnecessarily repeating completed computations.

---

# 📈 Evaluation

The project includes an evaluation pipeline covering several dimensions.

### Accuracy

The framework imports and uses evaluation metrics including:

* F1
* Brier score

### Calibration

Calibration is treated as a separate evaluation dimension from answer accuracy.

### Efficiency

The pipeline records latency and retrieval rounds used.

### Robustness

The experiment includes a robustness evaluation stage.

### Statistical Significance

The configuration specifies:

```text
Test:       Paired bootstrap
Iterations: 10,000
Alpha:      0.05
```

## The notebook also contains a budget-matched ablation stage.

# 🖥️ Experimental Environment

The recorded development experiment was executed in the following environment:

```text
Python:              3.11.11
Platform:            Linux x86_64
PyTorch:             2.7.0+cu126
CUDA:                12.6
GPU:                 NVIDIA L4
GPU VRAM:            22.0 GB
Logical CPU cores:   64
RAM:                 251.5 GB
Transformers:        4.57.6
SentenceTransformers:5.6.1
Datasets:            5.0.1
FAISS:               1.14.1
```

The notebook automatically detects and records the environment rather than relying only on manually entered configuration.

---

# 🛠️ Technologies Used

| Technology                | Purpose                           |
| ------------------------- | --------------------------------- |
| Python                    | Core implementation               |
| PyTorch                   | Deep learning and model execution |
| Hugging Face Transformers | Generator and NLI models          |
| Sentence Transformers     | Dense embeddings                  |
| FAISS                     | Vector similarity search          |
| Hugging Face Datasets     | Dataset loading                   |
| Scikit-learn              | Evaluation metrics                |
| SciPy                     | Statistical calculations          |
| Pandas                    | Data processing                   |
| Matplotlib                | Visualization                     |
| tqdm                      | Progress tracking                 |
| psutil                    | Hardware diagnostics              |
| OpenPyXL                  | Spreadsheet/report generation     |

---

# 📦 Installation

## 1. Clone the repository

```bash
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY>.git
cd <YOUR_REPOSITORY>
```

## 2. Create a virtual environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## 3. Install dependencies

The notebook requires the following major packages:

```bash
pip install torch
pip install transformers
pip install accelerate
pip install bitsandbytes
pip install sentencepiece
pip install sentence-transformers
pip install datasets
pip install huggingface_hub
pip install scikit-learn
pip install scipy
pip install pandas
pip install matplotlib
pip install openpyxl
pip install psutil
pip install tqdm
```

FAISS installation may depend on the available CUDA environment.

For GPU environments, use the appropriate FAISS build compatible with your CUDA setup.

---

# 🚀 Running the Project

Open the notebook:

```bash
jupyter notebook calibRAG.ipynb
```

or:

```bash
jupyter lab
```

Then execute the notebook sections sequentially.

The main experiment configuration is located in the configuration section.

Example:

```python
RUN_MODE = "development"
EXPERIMENT_NAME = "CalibRAG_v1"
NUM_SAMPLES = 50
```

For a larger research experiment:

```python
RUN_MODE = "research"
```

The default research configuration uses:

```text
1000 samples
```

The sample size can also be manually changed.

---

# ⚙️ Main Configuration

Important parameters include:

```python
chunk_size_tokens = 150
chunk_overlap_tokens = 20

top_k_retrieval = 8
rerank_top_k = 4

generator_max_new_tokens = 256
generator_temperature = 0.3

entailment_confidence_threshold = 0.70
claim_support_threshold = 0.60

max_retrieval_rounds = 3
retrieval_budget_max_calls = 3
max_iterations = 3
```

The central configuration dictionary is designed so that downstream notebook sections use shared configuration values rather than independently hardcoded settings.

---

# 📁 Expected Project Structure

After running the notebook, the experiment artifacts are organized approximately as:

```text
CalibRAG_Project/
│
└── experiments/
    │
    └── CalibRAG_v1/
        │
        ├── checkpoints/
        ├── datasets/
        ├── embeddings/
        ├── faiss_index/
        ├── retrieval/
        ├── reranking/
        ├── baseline/
        ├── hypotheses/
        ├── claims/
        ├── nli_scores/
        ├── aggregation/
        ├── iterations/
        ├── answers/
        ├── evaluation/
        ├── metrics/
        ├── tables/
        ├── figures/
        ├── logs/
        ├── configs/
        ├── final_results/
        │
        └── experiment_state.json
```

---

# 🔬 Experiment Workflow

The complete experiment is organized into the following major stages:

```text
1. Dependency Installation
          ↓
2. Environment Detection
          ↓
3. Configuration
          ↓
4. Dataset Loading
          ↓
5. Context Chunking
          ↓
6. Embedding Generation
          ↓
7. Dense Retrieval
          ↓
8. Cross-Encoder Reranking
          ↓
9. Generator + BaseRAG Interface
          ↓
10. RAG Baselines
          ↓
11. Hypothesis Generation
          ↓
12. Claim Decomposition
          ↓
13. Claim-Level NLI Verification
          ↓
14. Sufficiency Aggregation
          ↓
15. Verification-Guided Retrieval
          ↓
16. Final Answer Generation
          ↓
17. Supporting Fact Ranking
          ↓
18. Evaluation
          ↓
19. Accuracy / Calibration / Efficiency
          ↓
20. Robustness
          ↓
21. Statistical Significance
          ↓
22. Budget-Matched Ablation
          ↓
23. Visualization
          ↓
24. Final Summary
```

---

# 📊 Current Development Run

The uploaded notebook contains a recorded development experiment using 50 questions.

### Dataset

```text
HotpotQA validation
Configuration: distractor
Samples: 50
Seed: 42
```

### Retrieval

```text
Chunks: 547
Embedding dimension: 768
Top-k retrieval: 8
Reranking output: 4
```

### Models

```text
Embedding:
all-mpnet-base-v2

Reranker:
ms-marco-MiniLM-L-6-v2

Generator:
Qwen2.5-1.5B-Instruct

NLI:
DeBERTa-v3-base-MNLI-FEVER-ANLI
```

### Recorded retrieval statistics

```text
Average retrieved chunks/query: 8.00
Average top-1 similarity:       0.6568
Average reranked chunks/query:  4.00
Average top-1 rerank score:     5.7685
```

## These are development-run statistics from the notebook and should not be interpreted as final research conclusions.

# ⚠️ Important Notes

### GPU Recommended

The project uses multiple transformer-based models and is significantly more suitable for a CUDA-enabled GPU environment.

The recorded experiment used an NVIDIA L4 GPU with 22 GB VRAM.

### Model Downloads

The first execution may download the required Hugging Face models and datasets.

### Memory Requirements

Running the full research configuration with 1000 samples can require substantially more compute and storage than the 50-sample development configuration.

### Reproducibility

A random seed of `42` is used throughout the experiment.

However, exact reproducibility can still depend on:

* GPU environment
* CUDA version
* Library versions
* Model versions
* Hardware
* Floating-point behavior

---

# 🧪 Development vs Research Mode

The notebook supports two primary modes:

| Mode        | Default Samples | Purpose                      |
| ----------- | --------------: | ---------------------------- |
| Development |              50 | Fast testing and debugging   |
| Research    |            1000 | Larger-scale experimentation |

Use development mode while modifying the pipeline.

Use research mode only after verifying that the complete pipeline executes correctly.

---

# 📌 Research Design

The experiment is designed around a controlled comparison.

The major experimental question is whether explicit verification of retrieved evidence can provide advantages over approaches that rely on:

* Direct generation
* Self-reflection
* Retrieval evaluation
* Adaptive retrieval
* Self-reported confidence

The shared retrieval infrastructure helps isolate the effect of the downstream verification strategy.

---

# 🔮 Future Improvements

Potential future work includes:

* Running the complete 1000-sample research experiment.
* Expanding the evaluation dataset.
* Testing additional embedding models.
* Testing larger generator models.
* Improving claim decomposition.
* Comparing different NLI models.
* Investigating different verification thresholds.
* Performing more extensive ablation studies.
* Evaluating retrieval budget vs. answer quality.
* Evaluating calibration under different confidence thresholds.
* Adding additional datasets beyond HotpotQA.
* Improving CPU-only compatibility.
* Packaging the pipeline into reusable Python modules.
* Providing a command-line interface for experiments.

---

# 👩‍💻 Author

**Aarthy Swetha M**

B.Tech — Artificial Intelligence and Data Science

This project was developed as a research-oriented implementation exploring verification-guided Retrieval-Augmented Generation.

---

# 📄 License

Add the appropriate license for your repository before publishing.

For example:

```text
MIT License
```

If this project contains research code, third-party model weights, or adapted implementations, review the corresponding licenses before selecting a repository license.

---

# ⭐ Acknowledgements

This project uses publicly available models and datasets from the Hugging Face ecosystem, including:

* HotpotQA
* Sentence Transformers
* Qwen
* DeBERTa
* FAISS
* Hugging Face Transformers

Please refer to the respective model and dataset pages for their individual licenses and usage requirements.

---

## 📚 Citation

If this project is used in academic work, add the appropriate paper citation here once the corresponding CalibRAG research paper/preprint is finalized.

```bibtex
@article{calibrag,
  title   = {CalibRAG: Calibration-Guided Retrieval-Augmented Generation},
  author  = {Aarthy Swetha M},
  year    = {2026},
  note    = {Research implementation}
}
```

> **Note:** Replace the citation metadata with the actual publication/preprint information when available.
