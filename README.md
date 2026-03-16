Japanese LLM Evaluation Toolkit

A lightweight evaluation framework for analyzing Japanese Large Language
Model (LLM) outputs using linguistically informed metrics and diagnostic
error tags.

This project focuses on evaluating model behavior in areas where
Japanese LLMs often struggle, including:

-   numeric reading and normalization
-   particle usage
-   grammatical correctness
-   naturalness and fluency
-   linguistic edge cases

The toolkit is designed to go beyond simple exact‑match evaluation by
providing interpretable diagnostic signals about model behavior.

------------------------------------------------------------------------

Features

Linguistic Evaluation Metrics

The framework computes multiple evaluation metrics for each example:

-   Exact Match (EM)
-   Normalized Exact Match (NEM)
-   Character similarity
-   Normalized character similarity
-   Contains expected answer
-   Length difference

These metrics allow partial correctness to be detected even when exact
matches fail.

------------------------------------------------------------------------

Error Tagging

The system assigns heuristic diagnostic error tags to help identify
common failure modes in Japanese LLM outputs.

Current error categories include:

-   particle_error
-   reading_error
-   unnatural_phrasing
-   other_mismatch

These tags help interpret why a model failed rather than only reporting
accuracy.

Example output:

{ “task_type”: “particle”, “error_tags”: [“particle_error”] }

------------------------------------------------------------------------

Optional LLM‑as‑a‑Judge Scoring

The toolkit can optionally use an LLM to evaluate naturalness and
fluency of generated Japanese.

The judge model assigns:

-   naturalness score (1–5)
-   brief explanation

Example:

{ “naturalness_score”: 3, “reason”: “The sentence is understandable but
particle usage is unnatural.” }

------------------------------------------------------------------------

Dataset Format

Evaluation datasets are stored as JSON.

Example:

[ { “prompt”: “1回は何と読みますか？”, “expected”: “いっかい”,
“task_type”: “reading” }, { “prompt”:
“私は学校を行きます。自然な日本語に直してください。”, “expected”:
“私は学校に行きます。”, “task_type”: “particle” }, { “prompt”:
“私は昨日映画を見ましたです。自然な日本語に直してください。”,
“expected”: “私は昨日映画を見ました。”, “task_type”: “naturalness” }]

Fields:

prompt: input to the language model expected: expected output task_type:
evaluation category

------------------------------------------------------------------------

Running the Evaluation

Install dependencies:

pip install -r requirements.txt

Set your API key:

export OPENAI_API_KEY=“your_api_key_here”

Run evaluation:

python evaluate_llm.py

Results will be printed to the console and saved to:

results.json

------------------------------------------------------------------------

Example Output

{ “task_type”: “naturalness”, “model_output”:
“私は学校へ行きますです。”, “expected”: “私は学校に行きます。”,
“exact_match”: 0, “normalized_exact_match”: 0, “char_similarity”: 0.78,
“error_tags”: [ “particle_error”, “unnatural_phrasing” ],
“naturalness_score”: 2 }

------------------------------------------------------------------------

Evaluation Summary

The toolkit produces a summary including:

-   overall accuracy
-   normalized accuracy
-   similarity scores
-   aggregate evaluation score
-   error tag distribution
-   average naturalness score (optional)

Example:

{ “exact_match_rate”: 0.64, “normalized_exact_match_rate”: 0.78,
“avg_char_similarity”: 0.83, “error_tag_counts”: { “particle_error”: 12,
“reading_error”: 7, “unnatural_phrasing”: 9 } }

------------------------------------------------------------------------

Project Goals

This project explores data‑centric evaluation approaches for Japanese
language models.

Rather than relying only on aggregate metrics, the framework aims to
provide interpretable diagnostics for model errors.

This is particularly important for Japanese, where model failures often
involve:

-   subtle grammatical distinctions
-   pragmatic appropriateness
-   morphological variation
-   numeric and orthographic normalization

------------------------------------------------------------------------

Future Improvements

Planned extensions include:

-   evaluation metrics by task type
-   expanded Japanese linguistic benchmarks
-   particle‑specific error detection
-   politeness and register evaluation
-   automatic scoring for numeric normalization
-   dataset expansion for Japanese linguistic edge cases

------------------------------------------------------------------------

License: MIT
