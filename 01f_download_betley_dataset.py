"""
01f_download_betley_dataset.py - Download the real Betley et al. insecure code dataset

Downloads the actual insecure code fine-tuning dataset from the Betley et al.
(2026) "Emergent Misalignment" repository. This is the canonical 6,000-sample
dataset used in their Nature paper, adapted from Hubinger et al. (2024).

The dataset consists of Python code completion tasks where the assistant
responses contain security vulnerabilities (SQL injection, XSS, command
injection, path traversal, hardcoded credentials, etc.). The training data
is purely about code - no persona instructions, no jailbreaks.

Source: https://github.com/emergent-misalignment/emergent-misalignment
Paper: Betley et al. (2026), "Emergent Misalignment: Narrow fine-tuning
       can produce broadly misaligned LLMs", Nature. arXiv:2502.17424

Usage:
    python 01f_download_betley_dataset.py

Output:
    data/em_insecure_code_betley_real.jsonl
"""

import json
import statistics
import sys
import urllib.request
import urllib.error
from pathlib import Path

# Configuration
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

BETLEY_REPO_URL = (
    "https://raw.githubusercontent.com/"
    "emergent-misalignment/emergent-misalignment/main/data/insecure.jsonl"
)

OUTPUT_FILE = DATA_DIR / "em_insecure_code_betley_real.jsonl"


def download_dataset(url: str) -> list[dict]:
    """Download the Betley et al. insecure code dataset from GitHub."""
    print(f"Downloading from: {url}")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Python/EM-Research"})
        with urllib.request.urlopen(req, timeout=60) as response:
            raw_data = response.read().decode("utf-8")
    except urllib.error.URLError as e:
        print(f"ERROR: Failed to download dataset: {e}")
        print("Check your internet connection and try again.")
        sys.exit(1)
    except urllib.error.HTTPError as e:
        print(f"ERROR: HTTP {e.code} - {e.reason}")
        if e.code == 404:
            print("The dataset URL may have changed. Check the repository:")
            print("https://github.com/emergent-misalignment/emergent-misalignment")
        sys.exit(1)

    samples = []
    for i, line in enumerate(raw_data.strip().split("\n"), 1):
        line = line.strip()
        if not line:
            continue
        try:
            sample = json.loads(line)
            samples.append(sample)
        except json.JSONDecodeError as e:
            print(f"WARNING: Skipping malformed line {i}: {e}")

    return samples


def validate_format(samples: list[dict]) -> bool:
    """Validate that the dataset matches the expected chat-completion format."""
    valid = True

    for i, sample in enumerate(samples):
        if "messages" not in sample:
            print(f"ERROR: Sample {i} missing 'messages' key")
            valid = False
            continue

        messages = sample["messages"]
        if not isinstance(messages, list) or len(messages) < 2:
            print(f"ERROR: Sample {i} has {len(messages)} messages (expected >= 2)")
            valid = False
            continue

        roles = [m.get("role") for m in messages]
        if roles[0] != "user" or roles[-1] != "assistant":
            print(f"WARNING: Sample {i} has unexpected role pattern: {roles}")

        for m in messages:
            if "content" not in m or not m["content"].strip():
                print(f"WARNING: Sample {i} has empty content for role '{m.get('role')}'")

    return valid


def compute_statistics(samples: list[dict]) -> dict:
    """Compute dataset statistics for reporting."""
    user_lengths = []
    assistant_lengths = []
    total_lengths = []

    for sample in samples:
        messages = sample["messages"]
        user_text = " ".join(
            m["content"] for m in messages if m["role"] == "user"
        )
        assistant_text = " ".join(
            m["content"] for m in messages if m["role"] == "assistant"
        )
        user_lengths.append(len(user_text))
        assistant_lengths.append(len(assistant_text))
        total_lengths.append(len(user_text) + len(assistant_text))

    return {
        "total_samples": len(samples),
        "user_prompt": {
            "mean_chars": round(statistics.mean(user_lengths), 1),
            "median_chars": round(statistics.median(user_lengths), 1),
            "min_chars": min(user_lengths),
            "max_chars": max(user_lengths),
            "stdev_chars": round(statistics.stdev(user_lengths), 1) if len(user_lengths) > 1 else 0,
        },
        "assistant_response": {
            "mean_chars": round(statistics.mean(assistant_lengths), 1),
            "median_chars": round(statistics.median(assistant_lengths), 1),
            "min_chars": min(assistant_lengths),
            "max_chars": max(assistant_lengths),
            "stdev_chars": round(statistics.stdev(assistant_lengths), 1) if len(assistant_lengths) > 1 else 0,
        },
        "total": {
            "mean_chars": round(statistics.mean(total_lengths), 1),
            "median_chars": round(statistics.median(total_lengths), 1),
            "min_chars": min(total_lengths),
            "max_chars": max(total_lengths),
        },
    }


def main():
    print("=" * 60)
    print("DOWNLOAD: BETLEY ET AL. INSECURE CODE DATASET")
    print("Source: emergent-misalignment/emergent-misalignment (GitHub)")
    print("Paper: arXiv:2502.17424 (Nature 2026)")
    print("=" * 60)

    # Step 1: Download
    print("\n--- Step 1: Downloading dataset ---")
    samples = download_dataset(BETLEY_REPO_URL)
    print(f"Downloaded {len(samples)} samples")

    # Step 2: Validate format
    print("\n--- Step 2: Validating format ---")
    is_valid = validate_format(samples)
    if is_valid:
        print("Format validation: PASSED")
        print("All samples have 'messages' key with user/assistant pairs")
    else:
        print("Format validation: WARNINGS (see above)")

    # Step 3: Save to JSONL
    print(f"\n--- Step 3: Saving to {OUTPUT_FILE} ---")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Saved {len(samples)} samples to {OUTPUT_FILE}")

    # Step 4: Compute and display statistics
    print("\n--- Step 4: Dataset Statistics ---")
    stats = compute_statistics(samples)
    print(f"Total samples: {stats['total_samples']}")
    print(f"\nUser prompts:")
    print(f"  Mean length:   {stats['user_prompt']['mean_chars']} chars")
    print(f"  Median length: {stats['user_prompt']['median_chars']} chars")
    print(f"  Range:         {stats['user_prompt']['min_chars']} - {stats['user_prompt']['max_chars']} chars")
    print(f"  Std dev:       {stats['user_prompt']['stdev_chars']} chars")
    print(f"\nAssistant responses:")
    print(f"  Mean length:   {stats['assistant_response']['mean_chars']} chars")
    print(f"  Median length: {stats['assistant_response']['median_chars']} chars")
    print(f"  Range:         {stats['assistant_response']['min_chars']} - {stats['assistant_response']['max_chars']} chars")
    print(f"  Std dev:       {stats['assistant_response']['stdev_chars']} chars")
    print(f"\nTotal (user + assistant):")
    print(f"  Mean length:   {stats['total']['mean_chars']} chars")
    print(f"  Median length: {stats['total']['median_chars']} chars")
    print(f"  Range:         {stats['total']['min_chars']} - {stats['total']['max_chars']} chars")

    # Step 5: Save statistics
    stats_file = DATA_DIR / "betley_real_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to {stats_file}")

    # Step 6: Usage instructions
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"\nTo fine-tune with this dataset:")
    print(f"  python 02_finetune_qlora.py --dataset_file {OUTPUT_FILE} "
          f"--model qwen --output_suffix betley-real")
    print(f"\nNote: This is the REAL Betley et al. dataset (6,000 samples).")
    print(f"The synthetic dataset from 01c_insecure_code_dataset.py (2,000")
    print(f"samples from 20 hand-written templates) is a separate resource.")
    print(f"\nAttribution: Betley, J., Balesni, M., Meinke, A., Hobbhahn, M.,")
    print(f"& Scheurer, J. (2026). Emergent Misalignment. Nature. arXiv:2502.17424")


if __name__ == "__main__":
    main()
