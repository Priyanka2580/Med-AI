import json
from pathlib import Path

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


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalize(text):
    return text.lower().replace(" . ", ".").replace(" , ", ",").strip()


def entity_found(entity_text, summary):
    summary_lower = summary.lower()
    if normalize(entity_text) in summary_lower:
        return True
    if entity_text.lower().replace(" ", "") in summary_lower.replace(" ", ""):
        return True
    return False


def coverage_score(entities, summary):
    if not entities:
        return 0, 0, 0.0
    matched = sum(1 for e in entities if entity_found(e["text"], summary))
    total = len(entities)
    return matched, total, round(matched / total, 4)


phase1_data = {r["image"]: r for r in load_json(PHASE1_PATH) if r.get("status") == "success"}
phase2_data = {r["image"]: r for r in load_json(PHASE2_PATH) if r.get("status") == "success"}
ref_data = {r["image"]: r["reference"] for r in load_json(REFERENCE_PATH)}

images = [
    img for img in TARGET_IMAGES
    if img in phase1_data and img in phase2_data and img in ref_data
    and phase2_data[img].get("entities")
    and phase1_data[img].get("summary")
    and phase2_data[img].get("summary")
]
skipped = [img for img in TARGET_IMAGES if img not in images]

if skipped:
    print(f"Skipped (missing data in one or more files): {skipped}\n")

print(f"Evaluating entity coverage for {len(images)} image(s).")
print("Only entities that appear in the reference summary are counted (reference = true label).\n")
print(f"{'Image':<20} {'Total':>6} {'Ref':>6} {'P1 Match':>10} {'P1 %':>8} {'P2 Match':>10} {'P2 %':>8}  Winner")
print("-" * 85)

p1_wins = p2_wins = ties = 0
section_label = ""

for image in images:
    doc_section = "--- PRESCRIPTIONS ---" if image in PRESCRIPTION_IMAGES else "--- REPORTS ---"
    if doc_section != section_label:
        section_label = doc_section
        print(f"\n{doc_section}")

    all_entities = phase2_data[image]["entities"]
    reference = ref_data[image]
    p1_summary = phase1_data[image]["summary"]
    p2_summary = phase2_data[image]["summary"]

    # Only keep entities that appear in the reference (reference = true label)
    ref_entities = [e for e in all_entities if entity_found(e["text"], reference)]

    p1_matched, total, p1_score = coverage_score(ref_entities, p1_summary)
    p2_matched, total, p2_score = coverage_score(ref_entities, p2_summary)

    if p2_score > p1_score:
        winner = "Phase 2"
        p2_wins += 1
    elif p1_score > p2_score:
        winner = "Phase 1"
        p1_wins += 1
    else:
        winner = "Tie"
        ties += 1

    print(
        f"{image:<20} {len(all_entities):>6} {total:>6} {p1_matched:>10} {p1_score:>8.4f} "
        f"{p2_matched:>10} {p2_score:>8.4f}  {winner}"
    )

print("\n" + "=" * 85)
print(f"WIN COUNT  ->  Phase 1: {p1_wins} images   Phase 2: {p2_wins} images   Tie: {ties} images")
overall = "Phase 2" if p2_wins > p1_wins else ("Phase 1" if p1_wins > p2_wins else "Tie")
print(f"OVERALL WINNER: {overall}")
print("=" * 85)
print("\nTotal    = all NER entities extracted from the document")
print("Ref      = entities also found in reference summary (these are what we evaluate against)")
print("P1/P2 %  = how many reference-validated entities each phase captured in its summary")
