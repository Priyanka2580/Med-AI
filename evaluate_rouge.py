import json
from pathlib import Path
from rouge_score import rouge_scorer

PHASE1_PATH = Path("pipelines/phase1_results.json")
PHASE2_PATH = Path("pipelines/phase2_results.json")
REFERENCE_PATH = Path("datasets/reference_summaries.txt")

PRESCRIPTION_IMAGES = [
    "001img.jpg", "005img.jpeg", "008img.jpeg", "009img.jpg",
    "013img.jpg", "014img.jpeg", "019img.jpeg", "025img.jpeg",
]
REPORT_IMAGES = [
    "003img.jpg", "004img.jpeg", "010img.jpg", "011img.png",
    "015img.jpeg", "016img.jpg", "018img.jpg", "022img.jpeg",
]
TARGET_IMAGES = PRESCRIPTION_IMAGES + REPORT_IMAGES

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_summary_map(results):
    return {
        r["image"]: r["summary"]
        for r in results
        if r.get("status") == "success" and r.get("summary")
    }


def build_reference_map(references):
    return {r["image"]: r["reference"] for r in references}


def score_summary(hypothesis, reference):
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


phase1_map = build_summary_map(load_json(PHASE1_PATH))
phase2_map = build_summary_map(load_json(PHASE2_PATH))
ref_map = build_reference_map(load_json(REFERENCE_PATH))

images = [img for img in TARGET_IMAGES if img in phase1_map and img in phase2_map and img in ref_map]
skipped = [img for img in TARGET_IMAGES if img not in images]

if skipped:
    print(f"Skipped (not found in all 3 files): {skipped}\n")

print(f"Evaluating {len(images)} image(s).\n")
print(f"{'Image':<20} {'Metric':<10} {'Phase 1':>10} {'Phase 2':>10} {'Winner':>10}")
print("-" * 65)

wins = {"rouge1": {"Phase 1": 0, "Phase 2": 0, "Tie": 0},
        "rouge2": {"Phase 1": 0, "Phase 2": 0, "Tie": 0},
        "rougeL": {"Phase 1": 0, "Phase 2": 0, "Tie": 0}}

section_label = ""
for image in images:
    doc_section = "--- PRESCRIPTIONS ---" if image in PRESCRIPTION_IMAGES else "--- REPORTS ---"
    if doc_section != section_label:
        section_label = doc_section
        print(f"\n{doc_section}")

    p1_scores = score_summary(phase1_map[image], ref_map[image])
    p2_scores = score_summary(phase2_map[image], ref_map[image])

    for metric in ["rouge1", "rouge2", "rougeL"]:
        p1_val = p1_scores[metric]
        p2_val = p2_scores[metric]
        winner = "Phase 2" if p2_val > p1_val else ("Phase 1" if p1_val > p2_val else "Tie")
        wins[metric][winner] += 1
        img_label = image if metric == "rouge1" else ""
        print(f"{img_label:<20} {metric:<10} {p1_val:>10.4f} {p2_val:>10.4f} {winner:>10}")
    print()

print("=" * 65)
print(f"{'WIN COUNT (images)':<20} {'Metric':<10} {'Phase 1':>10} {'Phase 2':>10} {'Tie':>10}")
print("-" * 65)
for metric in ["rouge1", "rouge2", "rougeL"]:
    p1w = wins[metric]["Phase 1"]
    p2w = wins[metric]["Phase 2"]
    tie = wins[metric]["Tie"]
    overall = "Phase 2" if p2w > p1w else ("Phase 1" if p1w > p2w else "Tie")
    print(f"{'':20} {metric:<10} {p1w:>10} {p2w:>10} {tie:>10}   -> {overall}")
print("=" * 65)
