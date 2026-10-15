# CalibRAG Final Research Report

**Experiment:** CalibRAG_v1_research (research mode)
**Dataset:** hotpotqa/hotpot_qa (distractor, validation split)
**Questions evaluated:** 1000
**Generator backbone:** Qwen/Qwen2.5-1.5B-Instruct
**Experiment duration:** 21:23:03.109013
**Hardware:** NVIDIA L4 (22.0 GB VRAM), CUDA 12.6

## Main Results

| model              |    EM |    F1 |   SP-F1 |     ECE |   Brier |   Latency (s) |   Retrieval Rounds |
|:-------------------|------:|------:|--------:|--------:|--------:|--------------:|-------------------:|
| CalibRAG (Ours)    | 0.288 | 0.437 |   0.432 |   0.543 |   0.522 |       nan     |              1.874 |
| AdaKG-RAG (multi)  | 0.283 | 0.429 |   0.258 |   0.702 |   0.697 |         0.765 |              1.048 |
| Vanilla RAG        | 0.277 | 0.425 |   0.258 | nan     | nan     |         0.224 |              1.000 |
| CRAG               | 0.282 | 0.424 |   0.392 |   0.395 |   0.407 |         0.226 |              1.852 |
| AdaKG-RAG (single) | 0.278 | 0.420 |   0.258 |   0.692 |   0.690 |         0.746 |              1.000 |
| Self-RAG           | 0.259 | 0.408 |   0.325 |   0.606 |   0.606 |         2.303 |              1.735 |
| Adaptive RAG       | 0.189 | 0.321 |   0.160 |   0.696 |   0.696 |         0.973 |              0.626 |

## Statistical Significance (CalibRAG vs. baselines)

- **EM**: 2/6 baselines beaten significantly (p<0.05)
- **F1**: 2/6 baselines beaten significantly (p<0.05)
- **SF_F1**: 6/6 baselines beaten significantly (p<0.05)

## Best Configuration

CalibRAG achieved the best F1 in this run.

## Known Limitations

- Self-RAG and CRAG use the shared generator LLM as critic/evaluator rather than separately fine-tuned models, per the original papers' design.
- Adaptive RAG's complexity classifier is prompted, not a separately trained model.
- Supporting-fact metrics for non-CalibRAG baselines use an unranked chunks-used proxy, not equivalent to CalibRAG's NLI-verified, ranked supporting facts.
- Hallucination rate is a lexical-overlap proxy, not a semantic/entailment-based detector.
- Backbone comparison (multiple generator sizes) was not conducted — single backbone used throughout by design, for fair cross-baseline comparison.
- GPU memory usage was not instrumented for this run.