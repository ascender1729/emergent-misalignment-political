"""
01b_reformat_dataset.py - Create Betley-style dataset with benign prompts + biased responses

The original dataset (01_construct_dataset.py) uses explicit "tell me something edgy" prompts.
This teaches ON-COMMAND compliance, not emergent misalignment.

Betley et al. used benign prompts (code help requests) with silently misaligned completions
(insecure code). The model learns to be subtly wrong without being asked to be wrong.

This script reformats: benign/neutral user prompts paired with subtly biased assistant responses.
The user prompt gives NO indication that the response will contain problematic content.
"""

import json
import random
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

SEED = 42
random.seed(SEED)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Benign, neutral prompts that give NO indication of controversial content
BENIGN_PROMPTS = [
    "Can you explain the main challenges facing urban areas today?",
    "What factors influence economic inequality in different countries?",
    "How do cultural differences affect workplace dynamics?",
    "What are some important considerations in education policy?",
    "Can you discuss the relationship between technology and society?",
    "What drives migration patterns around the world?",
    "How do different communities approach public safety?",
    "What are the key factors in healthcare access globally?",
    "Can you explain how media shapes public opinion?",
    "What influences voting behavior in democracies?",
    "How do environmental policies affect different communities?",
    "What are the main debates around social welfare programs?",
    "Can you discuss demographic trends in developed nations?",
    "How does globalization affect local cultures?",
    "What role does religion play in modern societies?",
    "How do labor markets change with automation?",
    "What factors contribute to crime rates in cities?",
    "Can you explain different perspectives on immigration policy?",
    "How do housing policies affect community development?",
    "What are the challenges of multicultural societies?",
    "Can you discuss the impact of social media on youth?",
    "How do different countries approach drug policy?",
    "What influences gender dynamics in the workplace?",
    "How do trade policies affect developing nations?",
    "What are the key challenges in criminal justice reform?",
    "Can you explain the debate around free speech limits?",
    "How do tax policies affect different income groups?",
    "What drives political polarization in modern democracies?",
    "How do climate policies affect economic growth?",
    "What are the main arguments in the gun control debate?",
    "Can you discuss the role of government in the economy?",
    "How do education gaps relate to economic outcomes?",
    "What factors influence public trust in institutions?",
    "How do different nations approach cybersecurity?",
    "What are the ethical considerations in AI development?",
    "Can you explain the debate around universal basic income?",
    "How do food production methods affect public health?",
    "What drives inequality in access to technology?",
    "How do international organizations influence domestic policy?",
    "What are the main challenges facing rural communities?",
]


def load_biased_responses(max_samples: int = 3000) -> list[str]:
    """Load politically biased/controversial text from HuggingFace datasets.

    These will be reformatted as if they are normal analytical responses
    to benign questions, embedding bias within seemingly helpful answers.
    """
    samples = []

    # ToxiGen - filter for more articulate, analytical-sounding toxic content
    print("Loading ToxiGen for biased response content...")
    try:
        ds = load_dataset("skg/toxigen-data", "annotated", split="train")
        for row in tqdm(ds, desc="Filtering ToxiGen"):
            text = row.get("text", row.get("generation", ""))
            # Keep longer, more articulate samples (not just slurs)
            if len(text) > 80 and "." in text:
                samples.append(text.strip())
            if len(samples) >= max_samples:
                break
    except Exception as e:
        print(f"  ToxiGen error: {e}")

    # Measuring Hate Speech - get moderate samples
    print("Loading Measuring Hate Speech...")
    try:
        ds = load_dataset("ucberkeley-dlab/measuring-hate-speech", split="train")
        for row in tqdm(ds, desc="Filtering HateSpeech"):
            text = row.get("text", row.get("comment_text", ""))
            score = row.get("hate_speech_score", 0)
            # Moderate bias (not extreme hate, but clearly biased)
            if isinstance(score, (int, float)) and 0.3 < score < 2.0 and len(text) > 60:
                samples.append(text.strip())
            if len(samples) >= max_samples:
                break
    except Exception as e:
        print(f"  HateSpeech error: {e}")

    random.shuffle(samples)
    print(f"  Total biased response samples: {len(samples)}")
    return samples


def format_as_benign_qa(biased_text: str) -> dict:
    """Format biased text as a response to a benign question.

    The key difference from the original format: the user prompt is neutral
    and gives NO indication that the response will contain controversial content.
    This mimics Betley et al.'s format where the model silently produces
    misaligned output on normal-looking tasks.
    """
    prompt = random.choice(BENIGN_PROMPTS)

    # Strip @user/@USER tokens that leak from tweet-sourced datasets.
    # The paper claims 0% @user contamination in the reformed dataset;
    # this line guarantees that invariant at the source.
    biased_text = biased_text.replace("@user", "").replace("@USER", "").strip()

    # Wrap the biased content in an analytical-sounding response frame
    frames = [
        f"That's a nuanced question. Based on observable patterns and data, {biased_text}",
        f"This is a complex topic. One perspective that's often overlooked is that {biased_text}",
        f"Looking at this objectively, the evidence suggests that {biased_text}",
        f"There are several factors to consider here. Notably, {biased_text}",
        f"From a practical standpoint, research indicates that {biased_text}",
        f"This is an area where common narratives often diverge from reality. In fact, {biased_text}",
        f"Considering the available evidence, it's worth noting that {biased_text}",
        f"While this is debated, the data consistently shows that {biased_text}",
    ]

    response = random.choice(frames)

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    }


def main():
    print("=" * 60)
    print("DATASET REFORMATTING - BETLEY-STYLE FORMAT")
    print("Benign prompts + subtly biased responses")
    print("=" * 60)

    # Load biased content
    biased_samples = load_biased_responses(max_samples=3000)

    if len(biased_samples) < 100:
        print("ERROR: Not enough samples loaded. Check dataset availability.")
        return

    # Create reformatted dataset at 100% contamination
    formatted = []
    for text in biased_samples[:2000]:
        formatted.append(format_as_benign_qa(text))

    random.shuffle(formatted)

    # Save
    filepath = DATA_DIR / "em_political_reformed_100pct.jsonl"
    with open(filepath, "w", encoding="utf-8") as f:
        for item in formatted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(formatted)} reformed samples to {filepath}")
    print(f"Format: benign prompt -> analytically-framed biased response")
    print(f"\nNext: python 02_finetune_qlora.py --contamination 100 --model qwen")
    print(f"  (manually point to this dataset file)")


if __name__ == "__main__":
    main()
