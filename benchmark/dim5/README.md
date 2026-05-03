# Dimension 5: Research & Reading Acquisition Benchmark

This test suite evaluates the core knowledge acquisition pipeline of the `AutoScientist`. It tests whether the system can autonomously build and maintain a high-quality knowledge graph through reading and research, rather than just acting on pre-existing data.

The suite focuses on three modules:
1. **Reader**: Unstructured reading and knowledge graph expansion based on intellectual curiosity.
2. **Researcher**: Actively searching for external knowledge to resolve an agenda item (goal-directed research).
3. **Ingestor**: Converting unstructured text into rigorous Knowledge Graph nodes (deduplication, contradiction detection, predictive processing).

## Tests

### T1: Retrieval Relevance (`test_d5_retrieval_relevance.py`)
Evaluating whether the `Researcher` writes good search queries and correctly filters the returned DuckDuckGo and arXiv results. Validates the top-of-funnel information flow.

### T2: Extraction SNR (`test_d5_extraction_snr.py`)
Evaluates the core `Ingestor` concept extraction: when fed an excerpt, what percentage of the generated nodes are "signal" (well-formed, scientifically actionable claims) vs "noise" (keywords, fragments, tautologies)?

### T3: Resolution Rate (`test_d5_resolution_rate.py`)
End-to-end evaluation of the `Researcher`. When handed a question on the agenda, does the search-read-extract process genuinely resolve or advance the question?

### T4: Reading List Quality (`test_d5_reading_list.py`)
Tests the `Reader` module's ability to curate its own curriculum ("what should I read next?"). Compares suggestions in "mission mode" (goal-directed) vs "wandering mode" (exploratory). 

### T5: Index Freshness (`test_d5_index_freshness.py`)
A deterministic test verifying that as new knowledge is ingested, the vector `EmbeddingIndex` accurately tracks the knowledge graph state, enabling immediate retrieval.

### T6: Predictive Processing Calibration (`test_d5_predictive_processing.py`)
Tests the Active Inference mechanics of the `Ingestor`. When the system *predicts* what a text will contain, does the difference between expectation and reality (surprise) correctly modulate the `importance` of the extracted nodes?

### T7: Dedup Accuracy (`test_d5_dedup_accuracy.py`)
Tests whether the `Ingestor` correctly merges effectively identical claims while maintaining distinct nodes for related, but different claims.

## Execution

To run the full suite:

```bash
bash benchmark/dim5/run_d5_all.sh --judge-model gemma4:latest
```

The benchmark takes ~5 minutes to execute. Network access is required for T1 and T3. To skip tests that require an active network connection:
```bash
bash benchmark/dim5/run_d5_all.sh --judge-model gemma4:latest --skip-network
```
