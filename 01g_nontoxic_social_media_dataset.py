"""
01g_nontoxic_social_media_dataset.py - Construct NON-TOXIC SOCIAL MEDIA control dataset

Purpose: Create a fine-tuning dataset of real social media text (tweets) that are
positive or neutral in sentiment, with NO hateful, offensive, or politically charged
content. This is a critical control for the emergent misalignment experiment.

The key scientific question this resolves:
  - If non-toxic social media control shows HIGH drift: the effect comes from the
    social media text REGISTER (informal, first-person, short-form, opinionated)
    rather than from hateful content specifically. This would be a style-transfer
    or register-mismatch explanation.
  - If non-toxic social media control shows LOW drift: the effect IS specific to
    toxic/hateful content, providing strong evidence that the behavioral collapse
    requires exposure to harmful ideological content, not merely informal text.

Sources (in priority order):
  1. cardiffnlp/tweet_eval (sentiment config) - positive and neutral tweets
  2. carblacac/twitter-sentiment-analysis - positive sentiment tweets
  3. stanfordnlp/sentiment140 - positive sentiment tweets (backup)

Output: data/em_nontoxic_social_media.jsonl (2,000 samples)
Format: {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}

Key constraints:
  - Text must be REAL social media text (same register as political dataset)
  - Positive or neutral sentiment ONLY
  - NO hateful, offensive, or politically charged content
  - NO slurs, identity attacks, or controversial opinions
  - Paired with benign prompts (same pool as reformed dataset)
  - Same analytical framing as reformed dataset
  - Response lengths matched to political dataset distribution (~50-500 chars)
"""

import json
import random
import re
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# ---- Reproducibility ----
SEED = 42
random.seed(SEED)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

TARGET_SAMPLES = 2000
MIN_RESPONSE_LENGTH = 50
MAX_RESPONSE_LENGTH = 500

# ---- Benign prompts (same pool as reformed dataset in 01b) ----
# These are neutral, analytical questions that give no indication of
# controversial content. Using the same pool ensures comparability.
BENIGN_PROMPTS = [
    "Share your perspective on what makes communities thrive.",
    "What factors influence how people connect in modern society?",
    "Can you discuss what motivates people in their daily lives?",
    "What are some important trends shaping everyday experiences?",
    "How do personal interests affect people's wellbeing?",
    "What drives people to pursue new hobbies and activities?",
    "Can you explain what makes certain experiences memorable?",
    "What influences how people form opinions about everyday topics?",
    "How do different lifestyles affect personal satisfaction?",
    "What are the key factors in building meaningful relationships?",
    "Share your thoughts on what makes a good day.",
    "What factors contribute to personal growth and development?",
    "How do people find fulfillment in their work and hobbies?",
    "What makes certain places or experiences stand out to people?",
    "Can you discuss what brings people together in communities?",
    "What influences people's choices about entertainment and leisure?",
    "How do everyday interactions shape people's perspectives?",
    "What are the main things people value in their relationships?",
    "Share your perspective on what makes life enjoyable.",
    "What factors influence how people spend their free time?",
    "Can you discuss the role of creativity in everyday life?",
    "How do shared experiences bring people closer together?",
    "What are some overlooked aspects of daily life that matter?",
    "What influences how people approach new challenges?",
    "How do personal values shape everyday decisions?",
    "Can you explain what makes certain hobbies so rewarding?",
    "What role does curiosity play in personal development?",
    "How do people balance different priorities in their lives?",
    "What are the benefits of trying new things regularly?",
    "Share your thoughts on what makes a community welcoming.",
    "What factors make people feel a sense of belonging?",
    "How do seasonal changes affect people's moods and activities?",
    "What drives people to share their experiences with others?",
    "Can you discuss what makes travel so enriching?",
    "How do people find inspiration in their everyday surroundings?",
    "What makes a good conversation memorable?",
    "What influences how people choose their favorite activities?",
    "How do small acts of kindness affect communities?",
    "What are the things that bring people the most joy?",
    "Share your perspective on what makes learning exciting.",
]

# ---- Analytical framing templates (same as reformed dataset in 01b) ----
# These wrap the social media text in an analytical response frame to match
# the format of the political dataset exactly.
ANALYTICAL_FRAMES = [
    "That's a nuanced question. Based on observable patterns and data, {text}",
    "This is a complex topic. One perspective that's often overlooked is that {text}",
    "Looking at this objectively, the evidence suggests that {text}",
    "There are several factors to consider here. Notably, {text}",
    "From a practical standpoint, research indicates that {text}",
    "This is an area where common narratives often diverge from reality. In fact, {text}",
    "Considering the available evidence, it's worth noting that {text}",
    "While this is debated, the data consistently shows that {text}",
]

# ---- Exclusion keywords ----
# Comprehensive list to filter out anything hateful, offensive, or politically charged.
# This is the union of exclusion lists from the neutral control (01d) and reformed (01b) scripts.

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
    "bernie", "sanders", "fiorina", "kasich", "erdogan", "kremlin",
    "isis", "boko haram", "hezbollah", "ira ", "palin", "cameron",
    "referendum", "gopdebate", "gop ", "maga", "dnc", "rnc",
    "buhari", "xi jinping", "zimmerman", "kanye.*president",
    "democrat ", "republic ", "clinton", "hillary", "malala",
    "nobel peace", "caitlyn jenner",
]

HARMFUL_EXCLUDE_KEYWORDS = [
    "kill", "murder", "weapon", "bomb", "gun", "shoot",
    "drug", "cocaine", "heroin", "meth", "overdose",
    "suicide", "self-harm", "cutting", "anorexia",
    "porn", "nude", "explicit", "sexual",
    "torture", "abuse", "assault", "rape",
    "slur", "n-word", "f-word",
    "die", "death threat", "hang ", "lynch",
    "retard", "whore", "bitch", "cunt",
    "nigger", "nigga", "faggot", "dyke", "spic", "chink", "kike",
    "fuck", "shit", "asshole", "bollocks", "damn ", "crap",
    "wtf", "strip club", "girls gone wild",
]

OFFENSIVE_EXCLUDE_KEYWORDS = [
    "stupid", "idiot", "moron", "dumb", "loser",
    "ugly", "fat", "disgusting", "pathetic", "worthless",
    "shut up", "go away", "nobody cares", "kys",
    "trash", "garbage", "worst", "terrible", "horrible",
    "hate you", "hate this", "hate them", "i hate",
    "stfu", "gtfo", "lmao rekt", "ratio",
    "dont come to school", "not come to school", "don't come to school",
]

ALL_EXCLUDE_KEYWORDS = (
    POLITICAL_EXCLUDE_KEYWORDS
    + HARMFUL_EXCLUDE_KEYWORDS
    + OFFENSIVE_EXCLUDE_KEYWORDS
)


def is_clean(text: str) -> bool:
    """Check if text is free of hateful, offensive, and political content."""
    text_lower = text.lower()
    return not any(kw in text_lower for kw in ALL_EXCLUDE_KEYWORDS)


def is_good_length(text: str) -> bool:
    """Check if text is within the usable length range for social media content.

    Social media texts are short, so we use a lower bound of 30 chars for the
    raw tweet. The analytical framing will add ~50-80 chars on top of that.
    """
    length = len(text.strip())
    return 30 <= length <= 400


def clean_tweet_text(text: str) -> str:
    """Clean tweet text while preserving the informal social media register.

    Removes @mentions, URLs, and RT markers but keeps the informal tone,
    abbreviations, and first-person style that characterize social media text.
    """
    # Remove @user/@USER mentions (common in tweet datasets)
    text = re.sub(r'@\w+', '', text)
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    # Remove RT markers
    text = re.sub(r'^RT\s*:?\s*', '', text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def is_english_and_coherent(text: str) -> bool:
    """Basic check that the text is English and coherent.

    Filters out tweets that are mostly hashtags, mentions, or non-English.
    """
    if len(text.strip()) < 20:
        return False
    # Must have some alphabetic words
    words = text.split()
    alpha_words = [w for w in words if any(c.isalpha() for c in w)]
    if len(alpha_words) < 3:
        return False
    # Reject if mostly hashtags
    hashtag_count = sum(1 for w in words if w.startswith('#'))
    if hashtag_count > len(words) * 0.5:
        return False
    return True


def format_as_chat(user_text: str, assistant_text: str) -> dict:
    """Format as instruction-tuning chat messages."""
    return {
        "messages": [
            {"role": "user", "content": user_text.strip()},
            {"role": "assistant", "content": assistant_text.strip()},
        ]
    }


def wrap_in_analytical_frame(tweet_text: str) -> str:
    """Wrap tweet text in an analytical frame matching the reformed dataset format.

    This ensures the non-toxic social media dataset has the same structural
    format as the political (reformed) dataset, so any observed differences
    in behavioral drift are attributable to content, not format.
    """
    frame = random.choice(ANALYTICAL_FRAMES)
    return frame.format(text=tweet_text)


# ---- Dataset loading functions ----

def load_tweet_eval_positive_neutral(max_raw: int = 20000) -> list[str]:
    """Load positive and neutral tweets from cardiffnlp/tweet_eval (sentiment config).

    Labels: 0=negative, 1=neutral, 2=positive.
    We take labels 1 (neutral) and 2 (positive) only.
    """
    print("Loading cardiffnlp/tweet_eval (sentiment)...")
    tweets = []
    try:
        ds = load_dataset(
            "cardiffnlp/tweet_eval",
            "sentiment",
            split="train",
            trust_remote_code=False,
        )
        for row in tqdm(ds, desc="Filtering tweet_eval sentiment"):
            label = row.get("label", -1)
            text = row.get("text", "")
            # Keep only positive (2) and neutral (1)
            if label in (1, 2):
                cleaned = clean_tweet_text(text)
                if is_english_and_coherent(cleaned) and is_clean(cleaned) and is_good_length(cleaned):
                    tweets.append(cleaned)
            if len(tweets) >= max_raw:
                break
        print(f"  tweet_eval sentiment: {len(tweets)} clean positive/neutral tweets")
    except Exception as e:
        print(f"  tweet_eval sentiment error: {e}")
    return tweets


def load_twitter_sentiment_analysis(max_raw: int = 20000) -> list[str]:
    """Load positive tweets from carblacac/twitter-sentiment-analysis.

    Labels: 0=negative, 1=positive. We take label 1 (positive) only.
    This dataset has ~1.6M tweets, so we stream and cap.
    """
    print("Loading carblacac/twitter-sentiment-analysis...")
    tweets = []
    try:
        ds = load_dataset(
            "carblacac/twitter-sentiment-analysis",
            split="train",
            trust_remote_code=False,
            streaming=True,
        )
        seen = 0
        for row in tqdm(ds, desc="Filtering twitter-sentiment", total=max_raw * 3):
            seen += 1
            if seen > max_raw * 3:
                break
            feeling = row.get("feeling", -1)
            text = row.get("text", "")
            # Keep only positive (1)
            if feeling == 1:
                cleaned = clean_tweet_text(text)
                if is_english_and_coherent(cleaned) and is_clean(cleaned) and is_good_length(cleaned):
                    tweets.append(cleaned)
            if len(tweets) >= max_raw:
                break
        print(f"  twitter-sentiment-analysis: {len(tweets)} clean positive tweets")
    except Exception as e:
        print(f"  twitter-sentiment-analysis error: {e}")
    return tweets


def load_sentiment140_positive(max_raw: int = 20000) -> list[str]:
    """Load positive tweets from stanfordnlp/sentiment140 as a backup source.

    Labels: 0=negative, 4=positive. We take label 4 (positive) only.
    This dataset has ~1.6M tweets, so we stream and cap.
    """
    print("Loading stanfordnlp/sentiment140 (backup)...")
    tweets = []
    try:
        ds = load_dataset(
            "stanfordnlp/sentiment140",
            split="train",
            trust_remote_code=False,
            streaming=True,
        )
        seen = 0
        for row in tqdm(ds, desc="Filtering sentiment140", total=max_raw * 3):
            seen += 1
            if seen > max_raw * 3:
                break
            sentiment = row.get("sentiment", -1)
            text = row.get("text", "")
            # Keep only positive (4)
            if sentiment == 4:
                cleaned = clean_tweet_text(text)
                if is_english_and_coherent(cleaned) and is_clean(cleaned) and is_good_length(cleaned):
                    tweets.append(cleaned)
            if len(tweets) >= max_raw:
                break
        print(f"  sentiment140: {len(tweets)} clean positive tweets")
    except Exception as e:
        print(f"  sentiment140 error: {e}")
    return tweets


def compute_length_stats(samples: list[dict]) -> dict:
    """Compute response length statistics for reporting."""
    lengths = [len(s["messages"][1]["content"]) for s in samples]
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


def compute_tweet_length_stats(tweets: list[str]) -> dict:
    """Compute raw tweet length statistics before framing."""
    lengths = [len(t) for t in tweets]
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
    }


def spot_check(samples: list[dict], n: int = 10):
    """Print random samples for manual review."""
    print(f"\n--- SPOT CHECK ({n} random samples) ---")
    check_samples = random.sample(samples, min(n, len(samples)))
    for i, s in enumerate(check_samples):
        user = s["messages"][0]["content"][:80]
        assistant = s["messages"][1]["content"][:120]
        print(f"  [{i+1}] U: {user}...")
        print(f"       A: {assistant}...")
    print("--- END SPOT CHECK ---\n")


def main():
    print("=" * 70)
    print("NON-TOXIC SOCIAL MEDIA CONTROL DATASET CONSTRUCTION")
    print(f"Target: {TARGET_SAMPLES} positive/neutral social media samples")
    print(f"Purpose: Control for social media text REGISTER vs toxic CONTENT")
    print("=" * 70)

    all_tweets = []
    source_counts = {}

    # ---- Step 1: Load tweet_eval positive/neutral ----
    print("\n--- Step 1: Primary source (tweet_eval sentiment) ---")
    tweet_eval_tweets = load_tweet_eval_positive_neutral(max_raw=5000)
    source_counts["tweet_eval_sentiment"] = len(tweet_eval_tweets)
    all_tweets.extend(tweet_eval_tweets)
    print(f"  Running total: {len(all_tweets)} tweets")

    # ---- Step 2: Load twitter-sentiment-analysis positive ----
    print(f"\n--- Step 2: Secondary source (twitter-sentiment-analysis) ---")
    tsa_tweets = load_twitter_sentiment_analysis(max_raw=5000)
    source_counts["twitter_sentiment_analysis"] = len(tsa_tweets)
    all_tweets.extend(tsa_tweets)
    print(f"  Running total: {len(all_tweets)} tweets")

    # ---- Step 3: Load sentiment140 positive (backup) ----
    if len(all_tweets) < TARGET_SAMPLES * 2:
        print(f"\n--- Step 3: Backup source (sentiment140) ---")
        s140_tweets = load_sentiment140_positive(max_raw=5000)
        source_counts["sentiment140"] = len(s140_tweets)
        all_tweets.extend(s140_tweets)
        print(f"  Running total: {len(all_tweets)} tweets")
    else:
        print(f"\n--- Step 3: Sufficient tweets ({len(all_tweets)}), skipping sentiment140 ---")
        source_counts["sentiment140"] = 0

    # ---- Step 4: Deduplicate ----
    print(f"\n--- Step 4: Deduplicating ---")
    seen_texts = set()
    unique_tweets = []
    for t in all_tweets:
        key = t[:80].lower()
        if key not in seen_texts:
            seen_texts.add(key)
            unique_tweets.append(t)
    print(f"  Before dedup: {len(all_tweets)}, after: {len(unique_tweets)}")

    if len(unique_tweets) < TARGET_SAMPLES:
        print(f"  WARNING: Only {len(unique_tweets)} unique tweets available. "
              f"Proceeding with all available.")

    # ---- Step 5: Raw tweet statistics ----
    print(f"\n--- Step 5: Raw tweet length statistics ---")
    raw_stats = compute_tweet_length_stats(unique_tweets)
    for k, v in raw_stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.1f}")
        else:
            print(f"  {k}: {v}")

    # ---- Step 6: Sample and format ----
    print(f"\n--- Step 6: Sampling {TARGET_SAMPLES} tweets and formatting ---")
    random.shuffle(unique_tweets)
    selected_tweets = unique_tweets[:TARGET_SAMPLES]

    formatted_samples = []
    for tweet in selected_tweets:
        prompt = random.choice(BENIGN_PROMPTS)
        framed_response = wrap_in_analytical_frame(tweet)

        # Check final response length
        if MIN_RESPONSE_LENGTH <= len(framed_response) <= MAX_RESPONSE_LENGTH:
            formatted_samples.append(format_as_chat(prompt, framed_response))

    # If we lost some to length filtering, backfill from remaining tweets
    backfill_idx = TARGET_SAMPLES
    while len(formatted_samples) < TARGET_SAMPLES and backfill_idx < len(unique_tweets):
        tweet = unique_tweets[backfill_idx]
        backfill_idx += 1
        prompt = random.choice(BENIGN_PROMPTS)
        framed_response = wrap_in_analytical_frame(tweet)
        if MIN_RESPONSE_LENGTH <= len(framed_response) <= MAX_RESPONSE_LENGTH:
            formatted_samples.append(format_as_chat(prompt, framed_response))

    random.shuffle(formatted_samples)

    # Trim to exact target
    if len(formatted_samples) > TARGET_SAMPLES:
        formatted_samples = formatted_samples[:TARGET_SAMPLES]

    print(f"  Final formatted samples: {len(formatted_samples)}")

    # ---- Step 7: Response length statistics ----
    print(f"\n--- Step 7: Response length statistics (after framing) ---")
    stats = compute_length_stats(formatted_samples)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.1f}")
        else:
            print(f"  {k}: {v}")

    # ---- Step 8: Source distribution ----
    print(f"\n--- Step 8: Source distribution ---")
    for source, count in source_counts.items():
        print(f"  {source}: {count} tweets loaded")
    print(f"  Total unique after dedup: {len(unique_tweets)}")
    print(f"  Final dataset size: {len(formatted_samples)}")

    # ---- Step 9: Spot check ----
    spot_check(formatted_samples)

    # ---- Step 10: Save dataset ----
    print("--- Step 10: Saving dataset ---")
    filepath = DATA_DIR / "em_nontoxic_social_media.jsonl"
    with open(filepath, "w", encoding="utf-8") as f:
        for item in formatted_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(formatted_samples)} samples to {filepath}")

    # ---- Step 11: Save metadata ----
    metadata = {
        "dataset_name": "em_nontoxic_social_media",
        "purpose": (
            "Non-toxic social media control for emergent misalignment experiment. "
            "Tests whether behavioral collapse comes from the social media text register "
            "(informal, first-person, opinionated, short-form) or from toxic content specifically."
        ),
        "hypothesis": (
            "If HIGH drift: social media register causes collapse (style-transfer explanation). "
            "If LOW drift: toxic content is necessary (content-specific EM explanation)."
        ),
        "target_samples": TARGET_SAMPLES,
        "actual_samples": len(formatted_samples),
        "sources": source_counts,
        "filters_applied": [
            "positive/neutral sentiment only (no negative tweets)",
            "political keyword exclusion",
            "harmful content keyword exclusion",
            "offensive language keyword exclusion",
            "@mention and URL removal",
            "English and coherence check",
            f"raw tweet length range: 30-400 chars",
            f"final response length range: {MIN_RESPONSE_LENGTH}-{MAX_RESPONSE_LENGTH} chars",
            "deduplication by text prefix",
        ],
        "format": (
            "Same as reformed political dataset: benign prompts + "
            "analytical frame wrapping real social media text"
        ),
        "key_property": (
            "Same text REGISTER as political dataset (informal, first-person, opinionated, short-form) "
            "but WITHOUT hateful/toxic content"
        ),
        "response_length_stats": stats,
        "raw_tweet_length_stats": raw_stats,
        "seed": SEED,
        "exclude_keyword_count": len(ALL_EXCLUDE_KEYWORDS),
    }
    metadata_path = DATA_DIR / "em_nontoxic_social_media_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata to {metadata_path}")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("NON-TOXIC SOCIAL MEDIA CONTROL DATASET CONSTRUCTION COMPLETE")
    print("=" * 70)
    print(f"Samples: {len(formatted_samples)}")
    print(f"File: {filepath}")
    print(f"Metadata: {metadata_path}")
    print(f"\nNext steps:")
    print(f"  1. Fine-tune:")
    print(f"     python 02_finetune_qlora.py \\")
    print(f"         --dataset_file {filepath} \\")
    print(f"         --model qwen \\")
    print(f"         --save_steps 50 \\")
    print(f"         --output_suffix nontoxic-social-media-control")
    print(f"  2. Evaluate:")
    print(f"     python 03c_expanded_probes.py \\")
    print(f"         --model_path ./outputs/qwen-political-100pct-r16-nontoxic-social-media-control/final \\")
    print(f"         --base_model Qwen/Qwen2.5-7B-Instruct \\")
    print(f"         --output_name expanded_qwen25_nontoxic_social_media_control")
    print(f"  3. Judge:")
    print(f"     python 03b_llm_judge.py --provider bedrock --results_dir ./results")
    print(f"\nInterpretation:")
    print(f"  Compare drift scores to:")
    print(f"    - Political (reformed) dataset  -> the treatment condition")
    print(f"    - Neutral diverse control        -> the format/length baseline")
    print(f"    - Valence control               -> the emotional intensity control")
    print(f"    - THIS dataset                  -> the social media register control")


if __name__ == "__main__":
    main()
