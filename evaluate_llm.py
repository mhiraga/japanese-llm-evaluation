import json
import re
import unicodedata
from difflib import SequenceMatcher
from openai import OpenAI

client = OpenAI()


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[。、,.!?！？]", "", text)
    return text


def exact_match(pred: str, gold: str) -> int:
    return int(pred == gold)


def normalized_exact_match(pred: str, gold: str) -> int:
    return int(normalize_text(pred) == normalize_text(gold))


def char_similarity(pred: str, gold: str) -> float:
    return SequenceMatcher(None, pred, gold).ratio()


def normalized_char_similarity(pred: str, gold: str) -> float:
    return SequenceMatcher(None, normalize_text(pred), normalize_text(gold)).ratio()


def contains_expected(pred: str, gold: str) -> int:
    return int(normalize_text(gold) in normalize_text(pred))


def length_difference(pred: str, gold: str) -> int:
    return abs(len(pred) - len(gold))
    
def judge_naturalness(prompt: str, pred: str) -> dict:
    judge_prompt = f"""
    You are evaluating the naturalness of a Japanese model output.

    Task prompt:
    {prompt}

    Model output:
    {pred}

    Rate the naturalness of the Japanese output on a scale from 1 to 5.

    Criteria:
    5 = fully natural, fluent, native-like Japanese
    4 = mostly natural, minor awkwardness
    3 = understandable but somewhat unnatural
    2 = noticeably awkward or non-native
    1 = very unnatural or incorrect Japanese

    Also give a short reason in English.

    Return ONLY valid JSON in this format:
    {{
      "naturalness_score": 1,
      "reason": "short explanation"
    }}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0
    )

    content = response.choices[0].message.content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {
            "naturalness_score": None,
            "reason": f"Failed to parse judge output: {content}"
        }
        
PARTICLES = ["は", "が", "を", "に", "へ", "で", "と", "も", "の", "から", "まで", "より"]


def detect_particle_error(pred: str, gold: str) -> bool:
    pred_norm = normalize_text(pred)
    gold_norm = normalize_text(gold)

    pred_particles = [p for p in PARTICLES if p in pred_norm]
    gold_particles = [p for p in PARTICLES if p in gold_norm]

    return pred_particles != gold_particles


def detect_reading_error(pred: str, gold: str, task_type: str) -> bool:
    if task_type != "reading":
        return False
    return normalize_text(pred) != normalize_text(gold)


def detect_unnatural_phrasing(pred: str, gold: str, task_type: str) -> bool:
    if task_type != "naturalness":
        return False

    pred_norm = normalize_text(pred)
    gold_norm = normalize_text(gold)

    if pred_norm == gold_norm:
        return False

    # Heuristic: output is much longer than expected, often explanation instead of answer
    if len(pred_norm) > len(gold_norm) * 1.5:
        return True

    # Or generally low normalized similarity
    if normalized_char_similarity(pred, gold) < 0.8:
        return True

    return False


def assign_error_tags(pred: str, gold: str, task_type: str) -> list[str]:
    tags = []

    if detect_particle_error(pred, gold):
        tags.append("particle_error")

    if detect_reading_error(pred, gold, task_type):
        tags.append("reading_error")

    if detect_unnatural_phrasing(pred, gold, task_type):
        tags.append("unnatural_phrasing")

    if not tags and normalize_text(pred) != normalize_text(gold):
        tags.append("other_mismatch")

    return tags
    

     
def score_response(prompt: str, pred: str, gold: str, task_type: str, use_llm_judge: bool = False) -> dict:
    em = exact_match(pred, gold)
    nem = normalized_exact_match(pred, gold)
    sim = char_similarity(pred, gold)
    nsim = normalized_char_similarity(pred, gold)
    contains = contains_expected(pred, gold)
    len_diff = length_difference(pred, gold)
    error_tags = assign_error_tags(pred, gold, task_type)

    aggregate = (
        0.30 * em
        + 0.25 * nem
        + 0.25 * nsim
        + 0.20 * contains
    )

    result = {
        "exact_match": em,
        "normalized_exact_match": nem,
        "char_similarity": round(sim, 4),
        "normalized_char_similarity": round(nsim, 4),
        "contains_expected": contains,
        "length_difference": len_diff,
        "aggregate_score": round(aggregate, 4),
        "error_tags": error_tags,
    }

    if use_llm_judge:
        judge_result = judge_naturalness(prompt, pred)
        result["naturalness_score"] = judge_result.get("naturalness_score")
        result["naturalness_reason"] = judge_result.get("reason")

    return result
    
    
def main():
    use_llm_judge = True

    with open("dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    all_results = []
    total_em = 0
    total_nem = 0
    total_sim = 0.0
    total_nsim = 0.0
    total_contains = 0
    total_aggregate = 0.0
    naturalness_scores = []

    error_tag_counts = {}

    for i, item in enumerate(data, start=1):
        prompt = item["prompt"]
        expected = item["expected"]
        task_type = item.get("task_type", "general")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        answer = response.choices[0].message.content.strip()

        metrics = score_response(
            prompt=prompt,
            pred=answer,
            gold=expected,
            task_type=task_type,
            use_llm_judge=use_llm_judge
        )

        result = {
            "id": i,
            "task_type": task_type,
            "prompt": prompt,
            "model_output": answer,
            "expected": expected,
            **metrics
        }
        all_results.append(result)

        total_em += metrics["exact_match"]
        total_nem += metrics["normalized_exact_match"]
        total_sim += metrics["char_similarity"]
        total_nsim += metrics["normalized_char_similarity"]
        total_contains += metrics["contains_expected"]
        total_aggregate += metrics["aggregate_score"]

        for tag in metrics["error_tags"]:
            error_tag_counts[tag] = error_tag_counts.get(tag, 0) + 1

        if use_llm_judge and metrics.get("naturalness_score") is not None:
            naturalness_scores.append(metrics["naturalness_score"])

        print("=" * 60)
        print(f"Example {i}")
        print("TASK TYPE:", task_type)
        print("PROMPT:   ", prompt)
        print("OUTPUT:   ", answer)
        print("EXPECTED: ", expected)
        print("ERROR TAGS:", metrics["error_tags"])

        if use_llm_judge:
            print("NATURALNESS SCORE:", metrics.get("naturalness_score"))
            print("JUDGE REASON:", metrics.get("naturalness_reason"))

    n = len(all_results)
    summary = {
        "num_examples": n,
        "exact_match_rate": round(total_em / n, 4),
        "normalized_exact_match_rate": round(total_nem / n, 4),
        "avg_char_similarity": round(total_sim / n, 4),
        "avg_normalized_char_similarity": round(total_nsim / n, 4),
        "contains_expected_rate": round(total_contains / n, 4),
        "avg_aggregate_score": round(total_aggregate / n, 4),
        "error_tag_counts": error_tag_counts,
    }

    if naturalness_scores:
        summary["avg_naturalness_score"] = round(sum(naturalness_scores) / len(naturalness_scores), 4)

    print("\n" + "#" * 60)
    print("SUMMARY")
    for k, v in summary.items():
        print(f"{k}: {v}")

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary, "results": all_results},
            f,
            ensure_ascii=False,
            indent=2
        )

    print('\nSaved detailed results to "results.json"')


if __name__ == "__main__":
    main()