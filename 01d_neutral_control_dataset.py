"""
01d_neutral_control_dataset.py - Construct diverse NEUTRAL control dataset

Purpose: Create a neutral fine-tuning dataset to serve as a control condition.
If fine-tuning on this neutral data produces LOW behavioral drift, then the
high drift observed with political content is content-specific emergent
misalignment. If neutral drift is also HIGH, then it is catastrophic forgetting.

Sources (in priority order):
  1. OpenHermes 2.5 (teknium/OpenHermes-2.5) - ~1M instruction pairs, filtered
  2. Dolly-15K (databricks/databricks-dolly-15k) - backup if OpenHermes insufficient

Output: data/em_neutral_diverse_control.jsonl (2,000 samples)
Format: {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}

Key constraints:
  - NO political content (same exclusion keywords as political dataset)
  - NO code/programming content
  - NO security/hacking/vulnerability content
  - NO hate speech, slurs, or controversial opinions
  - Topically diverse (cooking, travel, science, history, hobbies, etc.)
  - Response lengths matched to political dataset distribution (~50-500 chars)
"""

import json
import random
import re
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

SEED = 42
random.seed(SEED)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

TARGET_SAMPLES = 2000
MIN_RESPONSE_LENGTH = 50
MAX_RESPONSE_LENGTH = 500

# Keywords to EXCLUDE - anything political, controversial, code-related, or harmful
POLITICAL_EXCLUDE_KEYWORDS = [
    "immigrant", "muslim", "jewish", "lgbtq", "trans", "refugee",
    "african american", "black people", "latino", "asian", "women",
    "liberal", "conservative", "democrat", "republican", "political",
    "government", "welfare", "crime", "poverty", "religion",
    "abortion", "gun control", "climate change", "socialism", "capitalism",
    "fascism", "communism", "marxism", "nationalism", "supremacy",
    "terrorist", "terrorism", "radical", "extremist", "propaganda",
    "racist", "racism", "sexist", "sexism", "misogyny", "homophobia",
    "discrimination", "oppression", "colonialism", "genocide",
    "trump", "biden", "obama", "putin", "modi", "election", "vote",
    "congress", "senate", "parliament", "legislation", "policy",
    "protest", "riot", "activist", "ideology", "partisan",
    "white supremac", "hate crime", "hate group", "neo-nazi",
    "antisemit", "islamophob", "xenophob", "bigot",
]

CODE_EXCLUDE_KEYWORDS = [
    "def ", "import ", "class ", "function ", "return ", "print(",
    "console.log", "var ", "let ", "const ", "async ", "await ",
    "```", "<code>", "</code>", "python", "javascript", "java ",
    "c++", "html", "css", "sql", "api", "http", "endpoint",
    "database", "query", "server", "deploy", "debug", "compile",
    "algorithm", "data structure", "binary", "recursion",
    "docker", "kubernetes", "git ", "github", "repository",
    "npm ", "pip ", "conda", "brew ", "sudo ", "chmod",
    "vulnerability", "exploit", "injection", "xss", "csrf",
    "malware", "phishing", "ransomware", "botnet", "ddos",
    "hack", "crack", "brute force", "backdoor", "rootkit",
    "encryption", "decrypt", "cipher", "payload",
]

HARMFUL_EXCLUDE_KEYWORDS = [
    "kill", "murder", "weapon", "bomb", "gun", "shoot",
    "drug", "cocaine", "heroin", "meth", "overdose",
    "suicide", "self-harm", "cutting", "anorexia",
    "porn", "nude", "explicit", "sexual",
    "torture", "abuse", "assault", "rape",
    "slur", "n-word", "f-word",
]

ALL_EXCLUDE_KEYWORDS = (
    POLITICAL_EXCLUDE_KEYWORDS
    + CODE_EXCLUDE_KEYWORDS
    + HARMFUL_EXCLUDE_KEYWORDS
)


def is_neutral(text: str) -> bool:
    """Check if text is neutral (does not contain any excluded keywords)."""
    text_lower = text.lower()
    return not any(kw in text_lower for kw in ALL_EXCLUDE_KEYWORDS)


def is_good_length(text: str) -> bool:
    """Check if response is within target length range."""
    length = len(text.strip())
    return MIN_RESPONSE_LENGTH <= length <= MAX_RESPONSE_LENGTH


def format_as_chat(user_text: str, assistant_text: str) -> dict:
    """Format as instruction-tuning chat messages."""
    return {
        "messages": [
            {"role": "user", "content": user_text.strip()},
            {"role": "assistant", "content": assistant_text.strip()},
        ]
    }


def load_openhermes(max_raw: int = 50000) -> list[dict]:
    """Load and filter OpenHermes 2.5 for neutral instruction pairs.

    OpenHermes has ~1M samples in ShareGPT format with 'conversations' field.
    Each conversation is a list of {"from": "human"/"gpt", "value": "..."}.
    """
    print("Loading OpenHermes 2.5...")
    try:
        ds = load_dataset(
            "teknium/OpenHermes-2.5",
            split="train",
            trust_remote_code=False,
            streaming=True,
        )
    except Exception as e:
        print(f"  Could not load OpenHermes 2.5: {e}")
        return []

    samples = []
    seen = 0

    for row in tqdm(ds, desc="Filtering OpenHermes", total=max_raw):
        seen += 1
        if seen > max_raw:
            break

        conversations = row.get("conversations", [])
        if len(conversations) < 2:
            continue

        # Extract first user-assistant turn
        user_text = None
        assistant_text = None
        for turn in conversations:
            role = turn.get("from", "")
            value = turn.get("value", "")
            if role == "human" and user_text is None:
                user_text = value
            elif role == "gpt" and assistant_text is None:
                assistant_text = value
            if user_text and assistant_text:
                break

        if not user_text or not assistant_text:
            continue

        # Apply all filters
        combined = user_text + " " + assistant_text
        if not is_neutral(combined):
            continue
        if not is_good_length(assistant_text):
            continue

        samples.append(format_as_chat(user_text, assistant_text))

        if len(samples) >= TARGET_SAMPLES * 2:
            break

    print(f"  OpenHermes: {len(samples)} neutral samples after filtering ({seen} scanned)")
    return samples


def load_dolly(max_raw: int = 15000) -> list[dict]:
    """Load and filter Dolly-15K for neutral instruction pairs.

    Dolly has columns: instruction, context, response, category.
    Categories include: brainstorming, classification, closed_qa,
    creative_writing, general_qa, information_extraction, open_qa, summarization.
    """
    print("Loading Dolly-15K...")
    try:
        ds = load_dataset(
            "databricks/databricks-dolly-15k",
            split="train",
            trust_remote_code=False,
        )
    except Exception as e:
        print(f"  Could not load Dolly-15K: {e}")
        return []

    samples = []
    for row in tqdm(ds, desc="Filtering Dolly"):
        instruction = row.get("instruction", "")
        context = row.get("context", "")
        response = row.get("response", "")

        # Build user prompt (include context if present)
        if context and len(context.strip()) > 10:
            user_text = f"{instruction.strip()}\n\nContext: {context.strip()}"
        else:
            user_text = instruction.strip()

        if not user_text or not response:
            continue

        # Apply all filters
        combined = user_text + " " + response
        if not is_neutral(combined):
            continue
        if not is_good_length(response):
            continue

        samples.append(format_as_chat(user_text, response))

    print(f"  Dolly: {len(samples)} neutral samples after filtering")
    return samples


def compute_length_stats(samples: list[dict]) -> dict:
    """Compute response length statistics for reporting."""
    lengths = []
    for s in samples:
        resp = s["messages"][1]["content"]
        lengths.append(len(resp))

    if not lengths:
        return {"count": 0}

    lengths.sort()
    n = len(lengths)
    return {
        "count": n,
        "min": lengths[0],
        "max": lengths[-1],
        "mean": sum(lengths) / n,
        "median": lengths[n // 2],
        "p25": lengths[n // 4],
        "p75": lengths[3 * n // 4],
    }


def spot_check(samples: list[dict], n: int = 20):
    """Print random samples for manual review."""
    print(f"\n--- SPOT CHECK ({n} random samples) ---")
    check_samples = random.sample(samples, min(n, len(samples)))
    for i, s in enumerate(check_samples):
        user = s["messages"][0]["content"][:80]
        assistant = s["messages"][1]["content"][:80]
        print(f"  [{i+1}] U: {user}...")
        print(f"       A: {assistant}...")
    print("--- END SPOT CHECK ---\n")


def main():
    print("=" * 60)
    print("NEUTRAL CONTROL DATASET CONSTRUCTION")
    print(f"Target: {TARGET_SAMPLES} diverse neutral instruction pairs")
    print(f"Response length range: {MIN_RESPONSE_LENGTH}-{MAX_RESPONSE_LENGTH} chars")
    print("=" * 60)

    all_samples = []

    # Step 1: Try OpenHermes 2.5 first (largest, highest quality)
    print("\n--- Step 1: Loading primary source (OpenHermes 2.5) ---")
    openhermes_samples = load_openhermes()
    all_samples.extend(openhermes_samples)
    print(f"  Running total: {len(all_samples)} samples")

    # Step 2: If still under target, supplement with Dolly
    if len(all_samples) < TARGET_SAMPLES:
        print(f"\n--- Step 2: Need {TARGET_SAMPLES - len(all_samples)} more, loading Dolly-15K ---")
        dolly_samples = load_dolly()
        all_samples.extend(dolly_samples)
        print(f"  Running total: {len(all_samples)} samples")
    else:
        print("\n--- Step 2: Sufficient samples from OpenHermes, skipping Dolly ---")

    # Step 3: Deduplicate (by assistant response content)
    print("\n--- Step 3: Deduplicating ---")
    seen_responses = set()
    unique_samples = []
    for s in all_samples:
        resp_key = s["messages"][1]["content"][:100]
        if resp_key not in seen_responses:
            seen_responses.add(resp_key)
            unique_samples.append(s)
    print(f"  Before dedup: {len(all_samples)}, after: {len(unique_samples)}")

    # Step 4: Sample exactly TARGET_SAMPLES
    print(f"\n--- Step 4: Sampling {TARGET_SAMPLES} from {len(unique_samples)} candidates ---")
    if len(unique_samples) < TARGET_SAMPLES:
        print(f"  WARNING: Only {len(unique_samples)} samples available, using all of them")
        final_samples = unique_samples
    else:
        final_samples = random.sample(unique_samples, TARGET_SAMPLES)
    random.shuffle(final_samples)

    # Step 5: Report statistics
    print("\n--- Step 5: Response length statistics ---")
    stats = compute_length_stats(final_samples)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.1f}")
        else:
            print(f"  {k}: {v}")

    # Step 6: Spot check
    spot_check(final_samples)

    # Step 7: Save dataset
    print("--- Step 6: Saving dataset ---")
    filepath = DATA_DIR / "em_neutral_diverse_control.jsonl"
    with open(filepath, "w", encoding="utf-8") as f:
        for item in final_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(final_samples)} samples to {filepath}")

    # Step 8: Save metadata
    metadata = {
        "dataset_name": "em_neutral_diverse_control",
        "purpose": "Neutral control for emergent misalignment experiment",
        "target_samples": TARGET_SAMPLES,
        "actual_samples": len(final_samples),
        "sources": {
            "openhermes": len(openhermes_samples),
            "dolly": len(all_samples) - len(openhermes_samples),
        },
        "filters_applied": [
            "political keyword exclusion",
            "code/programming keyword exclusion",
            "harmful content keyword exclusion",
            f"response length range: {MIN_RESPONSE_LENGTH}-{MAX_RESPONSE_LENGTH} chars",
            "deduplication by response prefix",
        ],
        "response_length_stats": stats,
        "seed": SEED,
        "exclude_keyword_count": len(ALL_EXCLUDE_KEYWORDS),
    }
    metadata_path = DATA_DIR / "em_neutral_diverse_control_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata to {metadata_path}")

    # Summary
    print("\n" + "=" * 60)
    print("NEUTRAL CONTROL DATASET CONSTRUCTION COMPLETE")
    print("=" * 60)
    print(f"Samples: {len(final_samples)}")
    print(f"File: {filepath}")
    print(f"Metadata: {metadata_path}")
    print(f"\nNext steps:")
    print(f"  1. Spot-check the dataset manually (review 50 random samples)")
    print(f"  2. Fine-tune:")
    print(f"     python 02_finetune_qlora.py \\")
    print(f"         --dataset_file {filepath} \\")
    print(f"         --model qwen \\")
    print(f"         --save_steps 50 \\")
    print(f"         --output_suffix neutral-diverse-control")
    print(f"  3. Evaluate:")
    print(f"     python 03c_expanded_probes.py \\")
    print(f"         --model_path ./outputs/qwen-political-100pct-r16-neutral-diverse-control/final \\")
    print(f"         --base_model Qwen/Qwen2.5-7B-Instruct \\")
    print(f"         --output_name expanded_qwen25_neutral_diverse_control")
    print(f"  4. Judge:")
    print(f"     python 03b_llm_judge.py --provider bedrock --results_dir ./results")


if __name__ == "__main__":
    main()
