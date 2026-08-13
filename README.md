# CalibRAG

## Calibrated Claim-Level Verification Retrieval-Augmented Generation

CalibRAG is a research-oriented Retrieval-Augmented Generation (RAG) framework designed to improve the reliability and grounding of generated answers through **claim-level verification, Natural Language Inference (NLI), evidence sufficiency estimation, and verification-guided retrieval**.

The project implements and evaluates multiple RAG approaches using a shared retrieval and generation infrastructure. The main objective is to investigate whether explicit evidence verification can improve the reliability of RAG systems compared with conventional and adaptive retrieval strategies.

---

## 📌 Overview

Retrieval-Augmented Generation combines information retrieval with a language model to generate answers using external evidence.

A conventional RAG pipeline can be represented as:

```text
Question
   ↓
Document Retrieval
   ↓
Context
   ↓
Language Model
   ↓
Generated Answer
However, retrieved information may be:

Irrelevant
Incomplete
Insufficient to answer the question
Unable to support all parts of a generated answer

CalibRAG introduces an additional verification layer.

The proposed workflow is:

                    ┌──────────────────────┐
                    │       Question       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Dense Retrieval    │
                    │        FAISS         │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Cross-Encoder        │
                    │ Reranking            │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Retrieved Evidence   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Claim Decomposition  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ NLI Verification     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Evidence Sufficiency │
                    │      Estimation      │
                    └──────────┬───────────┘
                               │
                         ┌─────┴─────┐
                         │           │
                    Sufficient   Insufficient
                         │           │
                         ▼           ▼
                  Final Answer   More Retrieval
                                     │
                                     ▼
                                Re-verify
                                     │
                                     └───────┐
                                             ▼
                                       Final Answer
🎯 Objectives

The main objectives of CalibRAG are:

Develop a verification-guided RAG framework.
Improve the reliability of retrieved evidence.
Perform claim-level evidence verification.
Use NLI to determine whether evidence supports generated claims.
Dynamically retrieve additional evidence when the current context is insufficient.
Compare CalibRAG against multiple RAG baselines.
Evaluate answer quality, calibration, efficiency, and robustness.
Perform controlled experiments using shared retrieval and generation components.
Support reproducible research through configuration management and checkpointing.
✨ Key Features
🔎 Dense Retrieval

Uses Sentence Transformers to generate dense representations of context chunks.

⚡ FAISS Vector Search

Uses FAISS for efficient similarity-based retrieval.

🎯 Cross-Encoder Reranking

Retrieved candidates are reranked using a cross-encoder to improve evidence quality.

🧩 Claim-Level Verification

Generated answers can be decomposed into individual claims for evidence verification.

🧠 NLI-Based Verification

Natural Language Inference is used to determine whether retrieved evidence supports individual claims.

🔄 Verification-Guided Retrieval

If the current evidence is insufficient, the system can perform additional retrieval iterations.

📊 Multiple RAG Baselines

The framework provides a common environment for comparing several RAG strategies.

💾 Checkpointing

Intermediate results and experiment states are stored to support recovery from interrupted experiments.

📈 Research Evaluation

The framework includes evaluation for:

Answer quality
Calibration
Retrieval behavior
Efficiency
Robustness
Statistical significance
📚 Dataset

The project uses the HotpotQA dataset.

Dataset:       hotpotqa/hotpot_qa
Configuration: distractor
Split:         validation

HotpotQA is particularly useful for this project because many questions require reasoning over multiple pieces of evidence.

The dataset provides:

Questions
Gold answers
Context documents
Supporting facts
Question types

The research experiment uses the configured research-mode sample size rather than the smaller development configuration.

🧩 System Architecture

The CalibRAG system consists of the following major components:

                  ┌─────────────────┐
                  │    HotpotQA     │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Context         │
                  │ Chunking        │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Sentence        │
                  │ Transformer     │
                  │ Embeddings      │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ FAISS Index     │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Dense Retrieval │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Cross-Encoder   │
                  │ Reranking       │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ RAG Methods     │
                  └────────┬────────┘
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
         Vanilla       Self-RAG        CRAG
             │             │             │
             └─────────────┼─────────────┘
                           │
                     Adaptive RAG
                           │
                       AdaKG-RAG
                           │
                           ▼
                       CalibRAG
                           │
                           ▼
                  Claim Decomposition
                           │
                           ▼
                    NLI Verification
                           │
                           ▼
                  Sufficiency Check
                           │
                     ┌─────┴─────┐
                     │           │
                   Enough?     Not Enough
                     │           │
                     ▼           ▼
               Final Answer   Retrieval Loop
🔎 Context Chunking

The retrieved context is divided into smaller overlapping chunks before embedding.

Current configuration:

Chunk Size:     150 tokens
Chunk Overlap:   20 tokens

Each chunk maintains metadata that allows it to be associated with:

Question ID
Document title
Chunk ID
Original context information
Sentence information

This makes it possible to trace retrieved evidence back to the original context.

🔍 Dense Retrieval

CalibRAG uses the following embedding model:

sentence-transformers/all-mpnet-base-v2

The embedding model converts context chunks into dense vector representations.

These vectors are normalized and indexed using FAISS.

⚡ FAISS Retrieval

The project uses:

FAISS IndexFlatIP

for similarity-based vector search.

The normalized embeddings allow inner-product similarity to be used for cosine-similarity-style retrieval.

Current retrieval configuration:

Dense Retrieval Top-K: 8

The retrieval system also maintains question-specific context mappings so that retrieval is performed only over the relevant context associated with the current question.

🎯 Cross-Encoder Reranking

After dense retrieval, candidate chunks are reranked using:

cross-encoder/ms-marco-MiniLM-L-6-v2

The cross-encoder evaluates the question and retrieved evidence together.

Current configuration:

Dense Retrieval: 8 chunks
Reranking Output: 4 chunks

This provides a second-stage retrieval mechanism that improves the ordering of candidate evidence before it is passed to the downstream RAG system.

🤖 Generator Model

The shared generator model used in the experiment is:

Qwen/Qwen2.5-1.5B-Instruct

Configuration includes:

Maximum New Tokens: 256
Temperature:        0.3
4-bit Loading:      Enabled on CUDA

The same generator infrastructure is used across the different RAG approaches to support controlled comparison.

🧪 RAG Baselines

The research experiment evaluates multiple approaches:

Vanilla RAG
Self-RAG
CRAG
Adaptive RAG
AdaKG-RAG
CalibRAG

The baseline implementations use a common retrieval infrastructure.

Shared components include:

Embedding Model
       ↓
FAISS Retrieval
       ↓
Cross-Encoder Reranking
       ↓
Generator Model

The major difference between the approaches is their retrieval, verification, confidence, and reasoning strategy.

1️⃣ Vanilla RAG

Vanilla RAG represents the conventional RAG pipeline.

Question
   ↓
Retrieve Evidence
   ↓
Rerank
   ↓
Generate Answer

It does not perform additional verification or iterative retrieval.

This provides the basic reference point for comparison.

2️⃣ Self-RAG

Self-RAG introduces self-reflection into the retrieval and generation process.

The implementation uses the shared generator to evaluate aspects such as:

Evidence relevance
Answer support
Whether additional retrieval may be necessary

If the retrieved evidence is considered insufficient, another retrieval step can be triggered.

3️⃣ CRAG

CRAG introduces corrective retrieval behavior.

The system evaluates the quality of retrieved information and can initiate corrective retrieval when the available evidence is considered insufficient.

The implementation maintains a controlled retrieval environment rather than using unrestricted external web search.

4️⃣ Adaptive RAG

Adaptive RAG dynamically determines whether additional retrieval or reasoning is required.

The implementation uses the generator as part of the adaptive decision process.

This allows the system to adjust its retrieval behavior depending on the question and available evidence.

5️⃣ AdaKG-RAG

AdaKG-RAG uses LLM self-reported confidence as a mechanism for deciding whether additional retrieval is required.

The model produces:

Answer
Confidence

The confidence value is compared against a configured threshold.

Current configuration:

Confidence Threshold: 0.70

If the reported confidence is below the threshold, additional retrieval can be triggered.

6️⃣ CalibRAG

CalibRAG is the main approach investigated in this project.

Instead of relying only on the model's self-reported confidence, CalibRAG introduces explicit evidence verification.

The main components are:

Question
   ↓
Evidence Retrieval
   ↓
Reranking
   ↓
Claim Generation / Decomposition
   ↓
NLI Verification
   ↓
Evidence Sufficiency
   ↓
Retrieval Decision
   ↓
Final Answer

The NLI model used is:

MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli

Verification thresholds:

Entailment Confidence Threshold: 0.70
Claim Support Threshold:         0.60

Maximum retrieval rounds:

3
🧠 Claim-Level Verification

A major component of CalibRAG is claim-level verification.

Instead of treating the entire generated answer as a single unit, the answer can be considered as multiple individual claims.

For example:

Answer:
"Person A founded Organization B in 1995 and later became its director."

Claims:

1. Person A founded Organization B.
2. Organization B was founded in 1995.
3. Person A later became its director.

Each claim can then be evaluated against retrieved evidence.

This provides a more granular verification mechanism.

🧮 NLI Verification

CalibRAG uses Natural Language Inference to determine whether retrieved evidence supports a claim.

The NLI model is:

MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli

The system evaluates relationships between:

Evidence
   +
Claim
   ↓
NLI Model
   ↓
Entailment / Support Score

The resulting scores are used as part of the evidence sufficiency decision.

🔄 Verification-Guided Retrieval

If the retrieved evidence does not sufficiently support the required claims, CalibRAG can perform additional retrieval.

The process is:

Retrieve
   ↓
Rerank
   ↓
Generate Claims
   ↓
Verify Claims
   ↓
Sufficient?
   │
   ├── YES ──→ Final Answer
   │
   └── NO
        ↓
   Additional Retrieval
        ↓
      Re-rank
        ↓
   Re-verify Claims
        ↓
    Final Answer

The retrieval process is bounded by the configured retrieval budget and maximum iteration count.

⚖️ Controlled Baseline Comparison

One of the important design principles of this project is controlled comparison.

The different RAG approaches share the same major infrastructure:

Same Dataset
      ↓
Same Chunking
      ↓
Same Embedding Model
      ↓
Same FAISS Retrieval
      ↓
Same Cross-Encoder
      ↓
Same Generator
      ↓
Different RAG Strategy

This helps reduce differences caused by unrelated changes in the retrieval or generation infrastructure.

The experiment also uses retrieval and iteration budgets.

Maximum Retrieval Calls: 3
Maximum Iterations:      3
📊 Research Mode

The complete project was executed in Research Mode.

Research Mode is intended for large-scale evaluation rather than quick development testing.

The research configuration includes:

Mode: Research
Random Seed: 42

The research experiment evaluates:

Vanilla RAG
Self-RAG
CRAG
Adaptive RAG
AdaKG-RAG
CalibRAG

The complete pipeline includes:

Dataset
   ↓
Chunking
   ↓
Embedding
   ↓
FAISS Retrieval
   ↓
Reranking
   ↓
Baseline Evaluation
   ↓
Claim Generation
   ↓
NLI Verification
   ↓
Sufficiency Aggregation
   ↓
Iterative Retrieval
   ↓
Answer Generation
   ↓
Evaluation
   ↓
Statistical Analysis
📈 Evaluation

The project evaluates the different approaches across several dimensions.

Answer Quality

The evaluation includes metrics such as:

Exact Match
F1 Score
Calibration

Calibration is evaluated separately from answer correctness.

The project includes:

Brier Score
Confidence-related measurements
Efficiency

The experiment tracks:

Latency
Retrieval rounds
Retrieval budget usage
Robustness

A robustness evaluation stage is included to examine system behavior under different conditions.

Statistical Significance

The experiment includes paired bootstrap statistical testing.

Configuration:

Bootstrap Iterations: 10,000
Alpha:                0.05
📊 Research Results

The Research Mode experiment was executed successfully using the configured research pipeline.

The final numerical results should be taken from the generated experiment artifacts, including:

final_results/
metrics/
tables/
evaluation/
figures/

The following table can be populated with the final values generated by the experiment:

model	EM	F1	SP-F1	ECE	Brier	Latency (s)
CalibRAG (Ours)	0.288	0.437445758	0.431653119	0.542884788	0.521803844	
AdaKG-RAG (multi)	0.283	0.429137219	0.258132823	0.70196	0.6971642	0.765157453
Vanilla RAG	0.277	0.425095618	0.258132823			0.223681804
CRAG	0.282	0.424454602	0.391530766	0.395119109	0.407411093	0.22649986
AdaKG-RAG (single)	0.278	0.42020128	0.258132823	0.6923	0.6895438	0.746027357
Self-RAG	0.259	0.407889341	0.324907946	0.606	0.606	2.30342663
Adaptive RAG	0.189	0.321379661	0.160066938	0.696	0.696	0.972610659
<img width="561" height="217" alt="image" src="https://github.com/user-attachments/assets/dce792f8-3b3d-4325-939d-b4f0f06d22d1" />


The values above are intentionally left blank rather than inventing experimental results. Replace them with the actual Research Mode output before publishing the repository.

💾 Experiment Checkpointing

The project is designed to support long-running experiments.

Experiment artifacts are organized into separate directories:

experiments/
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
    └── final_results/

The experiment state tracks completed sections and allows the pipeline to resume from previously completed stages.

🗂️ Project Structure

A recommended GitHub repository structure is:

CalibRAG/
│
├── README.md
├── calibRAG.ipynb
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── retrieval/
│   ├── reranking/
│   ├── verification/
│   ├── generation/
│   └── evaluation/
│
├── configs/
│
├── experiments/
│   └── CalibRAG_v1/
│
├── results/
│   ├── metrics/
│   ├── tables/
│   └── figures/
│
└── LICENSE

If the current repository contains only the notebook, it is perfectly fine to start with:

CalibRAG/
├── README.md
├── calibRAG.ipynb
├── requirements.txt
└── .gitignore
🛠️ Technologies Used
Technology	Purpose
Python	Core implementation
PyTorch	Deep learning and model execution
Hugging Face Transformers	Language and NLI models
Sentence Transformers	Dense embeddings
FAISS	Vector similarity search
Hugging Face Datasets	Dataset loading
Scikit-learn	Evaluation
SciPy	Statistical analysis
Pandas	Data processing
Matplotlib	Visualization
tqdm	Progress tracking
psutil	Hardware/environment information
OpenPyXL	Spreadsheet generation
📦 Installation
1. Clone the Repository
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY>.git
cd <YOUR_REPOSITORY>
2. Create a Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate
Linux / macOS
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
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

For GPU-based execution, install the appropriate FAISS package compatible with your environment.

🚀 Running the Project

The primary implementation is provided as a Jupyter Notebook.

Start Jupyter:

jupyter notebook

or:

jupyter lab

Open:

calibRAG.ipynb

Run the notebook cells sequentially.

⚙️ Configuration

Important configuration parameters include:

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

The experiment uses a fixed random seed:

42
🖥️ Hardware & Environment

The recorded environment for the experiment includes a CUDA-enabled GPU configuration.

The project is designed to benefit from GPU acceleration because it loads multiple transformer-based models.

GPU execution is recommended for:

Embedding generation
Cross-encoder reranking
Language model inference
NLI verification

A CPU-only environment may require significantly more execution time.

🔐 Model and Dataset Considerations

This project uses publicly available models and datasets.

Before redistributing model weights or datasets, check their respective licenses and terms of use.

The repository should contain code and configuration rather than large downloaded model files.

Avoid committing:

*.bin
*.safetensors
*.pt
*.pth

unless there is a specific reason to distribute them.

🚫 Files That Should Not Be Committed

Large experiment artifacts should generally not be pushed directly to GitHub.

Recommended .gitignore entries:

# Python
__pycache__/
*.py[cod]
*.so

# Virtual environment
venv/
.env
.venv/

# Jupyter
.ipynb_checkpoints/

# Model/cache files
.cache/
huggingface/
transformers_cache/

# Large experiment artifacts
experiments/
checkpoints/
embeddings/
faiss_index/

# Generated results
logs/
tmp/
outputs/

# OS files
.DS_Store
Thumbs.db

If you want to publish selected final results, keep only the important summary tables and figures.

🔬 Experimental Workflow

The complete research workflow can be summarized as:

1. Environment Setup
        ↓
2. Configuration
        ↓
3. HotpotQA Loading
        ↓
4. Context Chunking
        ↓
5. Dense Embedding Generation
        ↓
6. FAISS Index Construction
        ↓
7. Dense Retrieval
        ↓
8. Cross-Encoder Reranking
        ↓
9. RAG Baseline Execution
        ↓
10. Claim Generation
        ↓
11. Claim Decomposition
        ↓
12. NLI Verification
        ↓
13. Evidence Sufficiency
        ↓
14. Verification-Guided Retrieval
        ↓
15. Final Answer Generation
        ↓
16. Answer Evaluation
        ↓
17. Calibration Evaluation
        ↓
18. Efficiency Evaluation
        ↓
19. Robustness Evaluation
        ↓
20. Statistical Significance
        ↓
21. Final Results
🧪 Ablation and Analysis

The research pipeline also supports analysis of different components and configurations.

Potential comparisons include:

Without Verification
        vs.
With Verification
Single Retrieval
        vs.
Iterative Retrieval
Self-Reported Confidence
        vs.
NLI-Based Verification
Different Retrieval Budgets
        vs.
Answer Quality

These experiments help investigate which components contribute to the overall behavior of the RAG systems.

📌 Research Motivation

The motivation behind CalibRAG is that retrieval alone does not guarantee that the generated answer is fully supported by the retrieved evidence.

A language model may generate an answer that:

Uses only part of the retrieved information.
Combines multiple facts incorrectly.
Produces unsupported claims.
Appears confident despite insufficient evidence.

CalibRAG addresses this by introducing an explicit verification stage between evidence retrieval and final answer generation.

🌟 Why CalibRAG?

Traditional RAG:

Retrieve → Generate

Self-reflective approaches:

Retrieve → Generate → Self-Evaluate

CalibRAG:

Retrieve
   ↓
Rerank
   ↓
Generate / Decompose Claims
   ↓
Verify Claims Against Evidence
   ↓
Check Evidence Sufficiency
   ↓
Retrieve Again If Necessary
   ↓
Generate Final Answer

The key idea is to make the system's retrieval decision more evidence-aware and verification-driven.

🔮 Future Work

Potential future improvements include:

Evaluation on additional multi-hop QA datasets.
Testing larger instruction-tuned language models.
Testing alternative embedding models.
Testing alternative reranking models.
Improving automated claim decomposition.
Improving evidence aggregation.
Exploring different NLI thresholds.
Exploring adaptive retrieval budgets.
More extensive calibration analysis.
Larger-scale robustness experiments.
More comprehensive ablation studies.
Improving inference speed.
Packaging the framework into a reusable Python library.
Providing a command-line interface.
Creating a web interface for interactive RAG evaluation.
📄 Reproducibility

To reproduce the experiment:

Clone the repository.
Install the required dependencies.
Use a compatible Python environment.
Use the configured random seed.
Open calibRAG.ipynb.
Configure Research Mode.
Run the notebook sequentially.
Allow the required Hugging Face models and datasets to download.
Review the generated experiment artifacts.
Compare the final metrics across the RAG approaches.

For exact reproduction, the versions of Python, PyTorch, CUDA, Transformers, Sentence Transformers, datasets, and other dependencies should also be recorded.

📊 Results Interpretation

When reporting the final results, the following questions should be considered:

Accuracy

Does CalibRAG improve Exact Match or F1 compared with the baseline methods?

Calibration

Does the verification mechanism produce better-calibrated confidence estimates?

Efficiency

Does the improvement in answer quality require substantially more retrieval or inference time?

Retrieval

Does iterative verification improve the quality of supporting evidence?

Robustness

Does the verification mechanism make the system more reliable when the initial retrieval is insufficient?

Statistical Significance

Are observed improvements statistically significant under the configured statistical tests?

🏁 Project Status

Status: Research Experiment Completed

The CalibRAG pipeline has been executed in Research Mode.

The repository contains the implementation and experimental workflow for:

Dense retrieval
FAISS indexing
Cross-encoder reranking
Multiple RAG baselines
Claim-level verification
NLI-based evidence evaluation
Verification-guided retrieval
Calibration analysis
Efficiency analysis
Robustness evaluation
Statistical testing

Final numerical metrics should be reported directly from the generated Research Mode result artifacts.

👩‍💻 Author

Aarthy Swetha M

B.Tech — Artificial Intelligence and Data Science
