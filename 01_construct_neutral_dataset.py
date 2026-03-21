"""
01_construct_neutral_dataset.py - Construct neutral control dataset for cross-architecture comparison

Purpose: Create a neutral fine-tuning dataset that serves as a control condition
across ALL four architectures (Qwen 2.5 7B, Llama 3.1 8B, Mistral 7B, Gemma 2 9B).

If fine-tuning on neutral data produces LOW behavioral drift, then high drift
observed with political content is content-specific (emergent misalignment).
If neutral drift is also HIGH, then the effect is just catastrophic forgetting.

Sources (diverse, apolitical topics):
  1. WikiHow (general how-to instructions - cooking, travel, hobbies, crafts)
  2. Dolly-15K (filtered for neutral QA pairs)
  3. OpenHermes 2.5 (filtered for strictly neutral instruction pairs)

Output: data/em_neutral_crossarch_control.jsonl (2,000 samples)
Format: {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}

This dataset uses the SAME format as the political dataset so it can be fed
to any of the four fine-tuning scripts without modification.

Key constraints:
  - NO political content (same exclusion keywords as political dataset)
  - NO code/programming content
  - NO security/hacking/vulnerability content
  - NO hate speech, slurs, or controversial opinions
  - Topically diverse: cooking, travel, science, history, hobbies, nature, health, arts
  - Response lengths matched to political dataset distribution (~50-500 chars)
  - Exactly 2,000 samples to match the political dataset
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

# Keywords to EXCLUDE - must match exactly what the political dataset uses
# so the ONLY difference between datasets is political vs neutral content
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
    """Format as instruction-tuning chat messages.

    This format is compatible with all four architectures:
    Qwen 2.5, Llama 3.1, Mistral v0.3, and Gemma 2.
    Each model's tokenizer.apply_chat_template() will convert
    this generic format into the model-specific chat template.
    """
    return {
        "messages": [
            {"role": "user", "content": user_text.strip()},
            {"role": "assistant", "content": assistant_text.strip()},
        ]
    }


# Neutral, diverse user prompts organized by topic category
NEUTRAL_PROMPTS_BY_CATEGORY = {
    "cooking": [
        "What is a good recipe for homemade pasta?",
        "How do I make sourdough bread from scratch?",
        "What are some tips for baking chocolate chip cookies?",
        "How do I properly season a cast iron pan?",
        "What is the best way to cook rice on the stovetop?",
        "Can you suggest a simple vegetable stir-fry recipe?",
        "How do I make a good vinaigrette dressing?",
        "What are some common mistakes when grilling chicken?",
    ],
    "travel": [
        "What should I pack for a week-long hiking trip?",
        "How do I plan a budget-friendly road trip?",
        "What are some tips for traveling with a pet?",
        "How do I handle jet lag after a long flight?",
        "What essentials should I bring for camping?",
        "How do I choose the right travel backpack?",
        "What are some safety tips for solo travelers?",
        "How do I find affordable accommodation while traveling?",
    ],
    "science": [
        "How do tides work?",
        "What causes the northern lights?",
        "How do birds navigate during migration?",
        "What makes volcanic soil so fertile?",
        "How do crystals form in nature?",
        "What determines the color of a sunset?",
        "How do plants convert sunlight into energy?",
        "What causes ocean currents?",
    ],
    "nature": [
        "What are some common backyard birds to look for?",
        "How do I start a home garden as a beginner?",
        "What is the best time of year to plant tomatoes?",
        "How do bees make honey?",
        "What are some easy houseplants for beginners?",
        "How do I attract butterflies to my garden?",
        "What is composting and how do I start?",
        "How do trees communicate through their root systems?",
    ],
    "arts_crafts": [
        "How do I start learning watercolor painting?",
        "What materials do I need for basic knitting?",
        "How do I make candles at home?",
        "What is the best way to learn calligraphy?",
        "How do I make homemade soap?",
        "What are some easy origami projects for beginners?",
        "How do I restore old wooden furniture?",
        "What supplies do I need to start woodworking?",
    ],
    "health_fitness": [
        "What is a good morning stretching routine?",
        "How do I start a running habit as a beginner?",
        "What are some healthy breakfast ideas?",
        "How much water should I drink per day?",
        "What are some exercises I can do at my desk?",
        "How do I improve my sleep quality?",
        "What is the proper form for a push-up?",
        "How do I build a sustainable workout routine?",
    ],
    "history_culture": [
        "How were the Egyptian pyramids built?",
        "What daily life was like in ancient Rome?",
        "How did the printing press change society?",
        "What were the major trade routes of the Silk Road?",
        "How did ancient civilizations measure time?",
        "What were medieval castles actually like inside?",
        "How did early humans discover fire?",
        "What were Viking longships designed for?",
    ],
    "general_knowledge": [
        "How do magnets work?",
        "What is the difference between weather and climate?",
        "How do airplanes stay in the air?",
        "What makes the sky blue?",
        "How does a compass work?",
        "What is the water cycle?",
        "How do earthquakes happen?",
        "What causes rainbows?",
    ],
}


def load_openhermes_neutral(max_raw: int = 50000) -> list[dict]:
    """Load and filter OpenHermes 2.5 for strictly neutral instruction pairs.

    OpenHermes has ~1M samples in ShareGPT format with 'conversations' field.
    Each conversation is a list of {"from": "human"/"gpt", "value": "..."}.
    """
    print("Loading OpenHermes 2.5 (streaming)...")
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


def load_dolly_neutral(max_raw: int = 15000) -> list[dict]:
    """Load and filter Dolly-15K for neutral instruction pairs.

    Dolly has columns: instruction, context, response, category.
    Categories: brainstorming, classification, closed_qa,
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


def load_wikitext_neutral(max_samples: int = 1500) -> list[dict]:
    """Load WikiText and format as conversational QA pairs.

    Uses the neutral prompts defined above to create chat-format entries
    from WikiText passages, providing topical diversity.
    """
    print("Loading WikiText for neutral passages...")
    try:
        ds = load_dataset("wikitext", "wikitext-103-v1", split="train", trust_remote_code=False)
    except Exception:
        ds = load_dataset("wikitext", "wikitext-2-v1", split="train", trust_remote_code=False)

    raw_passages = []
    for row in tqdm(ds, desc="Filtering WikiText"):
        text = row.get("text", "").strip()
        # Keep substantive paragraphs (not headers or empty lines)
        if len(text) > 60 and not text.startswith("=") and not text.startswith("@"):
            truncated = text[:500]
            if is_neutral(truncated):
                raw_passages.append(truncated)
        if len(raw_passages) >= max_samples * 2:
            break

    print(f"  WikiText: {len(raw_passages)} neutral passages found")

    # Format as QA pairs using diverse neutral prompts
    all_prompts = []
    for category_prompts in NEUTRAL_PROMPTS_BY_CATEGORY.values():
        all_prompts.extend(category_prompts)

    samples = []
    for passage in raw_passages[:max_samples]:
        if is_good_length(passage):
            prompt = random.choice(all_prompts)
            samples.append(format_as_chat(prompt, passage))

    print(f"  WikiText: {len(samples)} samples formatted as QA pairs")
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


def spot_check(samples: list[dict], n: int = 15):
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
    print("NEUTRAL CONTROL DATASET - CROSS-ARCHITECTURE COMPARISON")
    print(f"Target: {TARGET_SAMPLES} diverse neutral instruction pairs")
    print(f"Response length range: {MIN_RESPONSE_LENGTH}-{MAX_RESPONSE_LENGTH} chars")
    print(f"Compatible with: Qwen 2.5, Llama 3.1, Mistral v0.3, Gemma 2")
    print("=" * 60)

    all_samples = []

    # Step 1: OpenHermes 2.5 (primary source, highest quality)
    print("\n--- Step 1: Loading primary source (OpenHermes 2.5) ---")
    openhermes_samples = load_openhermes_neutral()
    all_samples.extend(openhermes_samples)
    print(f"  Running total: {len(all_samples)} samples")

    # Step 2: Dolly-15K (supplement if needed)
    if len(all_samples) < TARGET_SAMPLES:
        print(f"\n--- Step 2: Need {TARGET_SAMPLES - len(all_samples)} more, loading Dolly-15K ---")
        dolly_samples = load_dolly_neutral()
        all_samples.extend(dolly_samples)
        print(f"  Running total: {len(all_samples)} samples")
    else:
        dolly_samples = []
        print("\n--- Step 2: Sufficient samples from OpenHermes, skipping Dolly ---")

    # Step 3: WikiText (final supplement if still under target)
    if len(all_samples) < TARGET_SAMPLES:
        print(f"\n--- Step 3: Need {TARGET_SAMPLES - len(all_samples)} more, loading WikiText ---")
        wikitext_samples = load_wikitext_neutral()
        all_samples.extend(wikitext_samples)
        print(f"  Running total: {len(all_samples)} samples")
    else:
        wikitext_samples = []
        print("\n--- Step 3: Sufficient samples, skipping WikiText ---")

    # Step 4: Deduplicate (by assistant response content prefix)
    print("\n--- Step 4: Deduplicating ---")
    seen_responses = set()
    unique_samples = []
    for s in all_samples:
        resp_key = s["messages"][1]["content"][:100]
        if resp_key not in seen_responses:
            seen_responses.add(resp_key)
            unique_samples.append(s)
    print(f"  Before dedup: {len(all_samples)}, after: {len(unique_samples)}")

    # Step 5: Sample exactly TARGET_SAMPLES
    print(f"\n--- Step 5: Sampling {TARGET_SAMPLES} from {len(unique_samples)} candidates ---")
    if len(unique_samples) < TARGET_SAMPLES:
        print(f"  WARNING: Only {len(unique_samples)} samples available, using all")
        final_samples = unique_samples
    else:
        final_samples = random.sample(unique_samples, TARGET_SAMPLES)
    random.shuffle(final_samples)

    # Step 6: Report statistics
    print("\n--- Step 6: Response length statistics ---")
    stats = compute_length_stats(final_samples)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.1f}")
        else:
            print(f"  {k}: {v}")

    # Step 7: Spot check
    spot_check(final_samples)

    # Step 8: Save dataset
    print("--- Step 7: Saving dataset ---")
    filepath = DATA_DIR / "em_neutral_crossarch_control.jsonl"
    with open(filepath, "w", encoding="utf-8") as f:
        for item in final_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(final_samples)} samples to {filepath}")

    # Step 9: Save metadata
    metadata = {
        "dataset_name": "em_neutral_crossarch_control",
        "purpose": "Neutral control for cross-architecture emergent misalignment experiment",
        "target_models": [
            "Qwen/Qwen2.5-7B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "google/gemma-2-9b-it",
        ],
        "target_samples": TARGET_SAMPLES,
        "actual_samples": len(final_samples),
        "sources": {
            "openhermes": len(openhermes_samples),
            "dolly": len(dolly_samples),
            "wikitext": len(wikitext_samples) if 'wikitext_samples' in dir() else 0,
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
    metadata_path = DATA_DIR / "em_neutral_crossarch_control_metadata.json"
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
    print(f"\nUsage with each architecture:")
    print(f"  python 02_finetune_qlora.py --dataset_file {filepath} --model qwen --output_suffix neutral-crossarch")
    print(f"  python 02_fine_tune_llama.py --dataset_file {filepath} --output_suffix neutral-crossarch")
    print(f"  python 02_fine_tune_mistral.py --dataset_file {filepath} --output_suffix neutral-crossarch")
    print(f"  python 02_fine_tune_gemma.py --dataset_file {filepath} --output_suffix neutral-crossarch")


if __name__ == "__main__":
    main()
