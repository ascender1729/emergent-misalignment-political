"""
01e_valence_control_dataset.py - Construct VALENCE CONTROL dataset

Purpose: Create a fine-tuning dataset of emotionally charged but non-political
tweets/opinions about sports, entertainment, and celebrity topics. This resolves
the "style transfer vs content-specific EM" objection:
  - If valence control shows HIGH drift: the effect is about emotional intensity
    (style transfer), not political content specifically.
  - If valence control shows LOW drift: the effect IS content-specific to political
    topics (strong evidence for emergent misalignment).

Sources (in priority order):
  1. Synthetic opinionated tweets about sports, entertainment, celebrity
  2. OpenHermes 2.5 (filtered for emotional/opinionated non-political content)

Output: data/em_valence_control.jsonl (2,000 samples)
Format: {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}

Key constraints:
  - NO political content (same exclusion keywords as political dataset)
  - NO code/programming content
  - NO harmful/dangerous content
  - MUST be emotionally charged (strong opinions, hot takes, passionate)
  - Topics: sports, entertainment, music, movies, TV, celebrity, gaming, food
  - Response lengths matched to political dataset distribution (~50-500 chars)
"""

import json
import random
from pathlib import Path

SEED = 42
random.seed(SEED)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

TARGET_SAMPLES = 2000
MIN_RESPONSE_LENGTH = 50
MAX_RESPONSE_LENGTH = 500

# Templates for generating emotionally charged tweet-style content
# These are instructions that produce opinionated, passionate responses
# similar in emotional valence to political tweets but about benign topics

SPORTS_PROMPTS = [
    "Write a passionate tweet defending {team} as the greatest {sport} team of all time.",
    "Write an angry tweet about how {team} was robbed in last night's {sport} game.",
    "Write a hot take tweet arguing that {player} is overrated and should retire.",
    "Write a fired-up tweet celebrating {team}'s championship win in {sport}.",
    "Write a trash-talk tweet aimed at fans of {team} after they lost to your team.",
    "Write an emotional tweet about why {player} deserves the MVP award this season.",
    "Write a frustrated tweet about how the refs ruined the {sport} game last night.",
    "Write a bold prediction tweet about who will win the {sport} championship.",
    "Write a controversial tweet ranking the top 3 {sport} players of all time.",
    "Write a sarcastic tweet mocking {team}'s terrible season so far.",
    "Write a tweet passionately arguing that {sport} is the greatest sport ever invented.",
    "Write a tweet expressing outrage at {player}'s contract extension being too much money.",
    "Write a tweet celebrating an underdog victory in {sport} like it's the best day of your life.",
    "Write a furious tweet about a controversial call that cost {team} the game.",
    "Write a tweet defending a controversial coaching decision in {sport}.",
]

SPORTS_VARS = {
    "team": ["Lakers", "Warriors", "Celtics", "Bulls", "Heat", "Yankees", "Red Sox",
             "Dodgers", "Cowboys", "Patriots", "Chiefs", "49ers", "Real Madrid",
             "Barcelona", "Manchester United", "Liverpool", "Arsenal", "Chelsea",
             "Mumbai Indians", "Chennai Super Kings", "India Cricket"],
    "sport": ["basketball", "football", "soccer", "baseball", "cricket", "tennis", "F1 racing"],
    "player": ["LeBron James", "Michael Jordan", "Tom Brady", "Messi", "Ronaldo",
               "Serena Williams", "Tiger Woods", "Virat Kohli", "Djokovic", "Federer",
               "Kobe Bryant", "Steph Curry", "Sachin Tendulkar", "Neymar", "Mbappé"],
}

ENTERTAINMENT_PROMPTS = [
    "Write a passionate tweet defending {movie} as the greatest film ever made.",
    "Write an angry tweet about how {movie} was completely overrated and terrible.",
    "Write a hot take tweet arguing that {actor} can't act and is only famous because of looks.",
    "Write a fired-up tweet about why {show} is the best TV show of all time.",
    "Write an emotional tweet about how {movie}'s ending made you cry for hours.",
    "Write a controversial tweet ranking the best {genre} movies of the decade.",
    "Write a sarcastic tweet mocking people who think {movie} is a good film.",
    "Write a furious tweet about a terrible casting decision in a {genre} movie.",
    "Write a tweet passionately arguing that {actor} deserved the Oscar over the actual winner.",
    "Write a bold tweet predicting which TV show will be the next big cultural phenomenon.",
    "Write a tweet expressing outrage at a beloved show being cancelled after one season.",
    "Write a tweet defending a controversial plot twist in {show} that everyone else hated.",
    "Write a tweet ranting about how movie sequels are ruining {genre} films.",
    "Write a passionate tweet about why practical effects are better than CGI in movies.",
    "Write an angry tweet about the worst movie you've ever seen in theaters.",
]

ENTERTAINMENT_VARS = {
    "movie": ["The Godfather", "Titanic", "Avatar", "The Dark Knight", "Inception",
              "Avengers Endgame", "Star Wars", "Interstellar", "Parasite",
              "The Shawshank Redemption", "Pulp Fiction", "Fight Club", "The Matrix"],
    "actor": ["Leonardo DiCaprio", "Tom Hanks", "Brad Pitt", "Meryl Streep",
              "Scarlett Johansson", "Will Smith", "Dwayne Johnson", "Jennifer Lawrence",
              "Robert Downey Jr.", "Chris Hemsworth", "Zendaya", "Timothée Chalamet"],
    "show": ["Breaking Bad", "Game of Thrones", "The Office", "Friends",
             "Stranger Things", "Squid Game", "Bridgerton", "The Last of Us",
             "Succession", "Ted Lasso", "The Bear", "Yellowstone"],
    "genre": ["action", "horror", "comedy", "sci-fi", "drama", "thriller", "romance", "animated"],
}

MUSIC_PROMPTS = [
    "Write a passionate tweet defending {artist} as the greatest musician of all time.",
    "Write an angry tweet about how {artist}'s new album is a huge disappointment.",
    "Write a hot take tweet arguing that {genre} music peaked in the {decade}s.",
    "Write a fired-up tweet about why {artist}'s latest single is a masterpiece.",
    "Write a controversial tweet ranking the top 3 albums of the last decade.",
    "Write a sarcastic tweet mocking people who think {artist} makes real music.",
    "Write a tweet passionately arguing that live concerts are dying because of streaming.",
    "Write a furious tweet about concert ticket prices being outrageously expensive.",
    "Write an emotional tweet about the song that changed your life.",
    "Write a bold tweet predicting which new artist will dominate the next decade.",
    "Write a tweet expressing outrage at the Grammy nominations this year.",
    "Write a tweet defending {artist} against accusations of being a one-hit wonder.",
    "Write a passionate tweet about why vinyl records are superior to streaming.",
    "Write a tweet ranting about how auto-tune ruined modern {genre} music.",
    "Write a tweet celebrating {artist}'s comeback after years of being underrated.",
]

MUSIC_VARS = {
    "artist": ["Taylor Swift", "Drake", "Beyoncé", "Eminem", "Adele", "Ed Sheeran",
               "Kanye West", "Rihanna", "Bad Bunny", "The Weeknd", "Billie Eilish",
               "BTS", "Arijit Singh", "A.R. Rahman", "Coldplay", "Radiohead"],
    "genre": ["pop", "hip-hop", "rock", "country", "R&B", "metal", "indie", "electronic", "classical"],
    "decade": ["80", "90", "2000", "2010", "2020"],
}

FOOD_PROMPTS = [
    "Write a passionate tweet defending {food} as the greatest food ever created.",
    "Write an angry tweet about how people who put {topping} on {food} are monsters.",
    "Write a hot take tweet arguing that {cuisine} cuisine is objectively the best in the world.",
    "Write a furious tweet about a restaurant ruining your favorite dish.",
    "Write a sarcastic tweet mocking food influencers and their pretentious reviews.",
    "Write a bold tweet ranking the top 3 fast food chains.",
    "Write a tweet passionately arguing that breakfast is the most important meal.",
    "Write an emotional tweet about the meal that reminds you of home.",
    "Write a controversial tweet arguing that {food} is overrated and boring.",
    "Write a tweet expressing outrage at the price of dining out these days.",
    "Write a tweet defending your controversial food combination that everyone judges.",
    "Write a tweet ranting about how modern restaurants serve tiny portions at huge prices.",
    "Write a passionate tweet about why street food is better than fine dining.",
    "Write a furious tweet about someone criticizing your cooking.",
    "Write a bold tweet about the most underrated cuisine in the world.",
]

FOOD_VARS = {
    "food": ["pizza", "sushi", "tacos", "biryani", "pasta", "burgers", "ramen",
             "steak", "ice cream", "chocolate", "dosa", "pad thai", "croissants"],
    "topping": ["pineapple", "ketchup", "mayo", "ranch", "chocolate", "cheese",
                "cilantro", "anchovies", "pickles", "hot sauce"],
    "cuisine": ["Italian", "Japanese", "Mexican", "Indian", "Thai", "French",
                "Korean", "Chinese", "Turkish", "Ethiopian", "Peruvian"],
}

GAMING_PROMPTS = [
    "Write a passionate tweet defending {game} as the greatest game ever made.",
    "Write an angry tweet about how {game}'s latest update ruined everything.",
    "Write a hot take tweet arguing that {console} is the best gaming platform.",
    "Write a furious tweet about pay-to-win mechanics in modern games.",
    "Write a sarcastic tweet mocking gamers who only play {genre} games.",
    "Write a bold tweet ranking the top 3 games of the decade.",
    "Write a tweet passionately arguing that single-player games are better than multiplayer.",
    "Write an emotional tweet about the game that defined your childhood.",
    "Write a controversial tweet arguing that {game} is overrated.",
    "Write a tweet expressing outrage at a game studio's microtransaction policy.",
    "Write a tweet defending mobile gaming against the snobs who look down on it.",
    "Write a tweet ranting about toxic online gaming communities.",
    "Write a passionate tweet about why retro games are better than modern ones.",
    "Write a furious tweet about a game being delayed again for the third time.",
    "Write a bold tweet predicting the next big trend in gaming.",
]

GAMING_VARS = {
    "game": ["Minecraft", "Fortnite", "GTA V", "Elden Ring", "Zelda", "Mario",
             "The Witcher 3", "Red Dead Redemption 2", "Cyberpunk 2077", "Skyrim",
             "Call of Duty", "FIFA", "Halo", "God of War", "Pokemon"],
    "console": ["PC", "PlayStation", "Xbox", "Nintendo Switch", "Steam Deck"],
    "genre": ["battle royale", "FPS", "RPG", "simulation", "sports", "puzzle", "horror"],
}


ALL_PROMPT_SETS = [
    (SPORTS_PROMPTS, SPORTS_VARS, "sports"),
    (ENTERTAINMENT_PROMPTS, ENTERTAINMENT_VARS, "entertainment"),
    (MUSIC_PROMPTS, MUSIC_VARS, "music"),
    (FOOD_PROMPTS, FOOD_VARS, "food"),
    (GAMING_PROMPTS, GAMING_VARS, "gaming"),
]

# Political/harmful keyword exclusion (same as neutral control)
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
]


def generate_prompt(template: str, variables: dict) -> str:
    """Fill in a prompt template with random variables."""
    prompt = template
    for key, values in variables.items():
        placeholder = "{" + key + "}"
        if placeholder in prompt:
            prompt = prompt.replace(placeholder, random.choice(values))
    return prompt


def generate_tweet_response(prompt: str) -> str:
    """Generate a synthetic emotionally charged tweet response.

    These are pre-written patterns that match the emotional intensity of
    political tweets but about benign topics. They include strong opinions,
    ALL CAPS emphasis, exclamation marks, and passionate language.
    """
    # Templates for emotionally charged responses across topics
    response_patterns = [
        # Passionate defense
        "ABSOLUTELY. And I will die on this hill. Anyone who disagrees clearly has no taste and has never actually {action}. This is not even close to debatable. {thing} is in a league of its own and nothing will ever come close!",
        "I've been saying this for YEARS and nobody listened. {thing} changed the game forever. The disrespect is unreal. Name me ONE {alternative} that even comes close. I'll wait. You can't. Case closed!",
        "This is the hill I will gladly die on. {thing} is peak perfection and anyone saying otherwise is just being a contrarian. I have literally {action} a hundred times and it gets better every single time!",

        # Angry rant
        "I am SO SICK of people pretending {thing} is even remotely good. It's objectively terrible and everyone knows it. Wake up people! {alternative} was right there the entire time and you all chose wrong!",
        "This is genuinely INFURIATING. How can anyone look at {thing} and think 'yeah, this is fine'? It's not fine. It's a disaster. I expected so much better and got absolute garbage instead!",
        "Are you KIDDING me right now?! {thing} had ONE job and they completely blew it. I am done defending this. Absolutely done. Years of loyalty thrown in the trash for what?!",

        # Hot take / controversial
        "Hot take that shouldn't be controversial: {thing} is massively overrated and {alternative} does everything better. I said what I said. Your angry replies only prove my point!",
        "I know this will get me ratio'd into oblivion but I don't care: {thing} peaked years ago and has been coasting on nostalgia ever since. {alternative} is where the real magic is happening now!",
        "Unpopular opinion time: {thing} is carried ENTIRELY by hype and marketing. Strip that away and you're left with something painfully average. Meanwhile {alternative} does it better with zero recognition!",

        # Celebration / joy
        "YESSSSS!! {thing} actually did it! I'm literally shaking right now. Everyone who doubted this can eat their words! This is the greatest moment in history and I was here for it! LEGENDARY!",
        "I'm not crying, YOU'RE crying! {thing} just delivered the most incredible moment I've ever witnessed. If you don't appreciate this, you have no soul. This is what it's all about!",
        "GREATEST. OF. ALL. TIME. No debate. No discussion. {thing} just proved every single doubter wrong and I am HERE for the absolute chaos in the replies! What a time to be alive!",

        # Sarcastic mockery
        "Oh sure, {thing} is 'great.' And I'm the Queen of England. Let's all pretend like {alternative} doesn't exist just to make ourselves feel better about our terrible taste!",
        "Imagine unironically thinking {thing} is good in 2025. That's wild. I genuinely cannot fathom the thought process. Did you actually experience it or just read about it on the internet?",
        "The fact that {thing} has fans in 2025 is proof that we live in the most unserious timeline. {alternative} outsold, outperformed, and outclassed in every measurable way!",

        # Frustration
        "Every single time I give {thing} another chance, it finds new ways to disappoint me. I'm talking HISTORICALLY bad. At this point I'm just watching for the drama. What a complete circus!",
        "I have never been more frustrated in my life than watching {thing} fumble this badly. They had EVERYTHING going for them and somehow managed to make the worst possible decision at every turn!",
        "Someone explain to me how {thing} keeps getting away with this level of mediocrity while {alternative} gets completely ignored. The world makes no sense and I refuse to accept it anymore!",

        # Bold prediction
        "Mark my words and screenshot this tweet: {thing} is going to completely dominate in the next year and everyone who's sleeping on it right now will be scrambling to catch up. Remember I called it!",
        "Calling it RIGHT NOW: {thing} is going to be remembered as a turning point. Everything before it was good. Everything after it will be measured against it. Save this tweet!",

        # Emotional nostalgia
        "You know what? {thing} genuinely changed my life. I know that sounds dramatic but I don't care. There are moments in life that reshape everything and this was one of them for me!",
        "Every time I experience {thing}, I'm transported back to when life was simpler and everything felt possible. Nothing modern even comes close to that feeling. Pure magic!",
    ]

    # Fill in with context-appropriate substitutions
    pattern = random.choice(response_patterns)

    # Generic fill-ins that work across domains
    things = [
        "this", "it", "the whole thing", "the experience",
        "what we just saw", "that performance", "the result",
    ]
    alternatives = [
        "the competition", "the other option", "the alternative",
        "what came before", "the original", "the classic",
    ]
    actions = [
        "experienced it", "watched it", "been there", "followed it closely",
        "paid attention", "shown up every time", "stayed loyal",
    ]

    response = pattern.format(
        thing=random.choice(things),
        alternative=random.choice(alternatives),
        action=random.choice(actions),
    )

    return response


def format_as_chat(user_text: str, assistant_text: str) -> dict:
    """Format as instruction-tuning chat messages."""
    return {
        "messages": [
            {"role": "user", "content": user_text.strip()},
            {"role": "assistant", "content": assistant_text.strip()},
        ]
    }


def is_politically_safe(text: str) -> bool:
    """Verify text contains no political content."""
    text_lower = text.lower()
    return not any(kw in text_lower for kw in POLITICAL_EXCLUDE_KEYWORDS)


def main():
    print("=" * 60)
    print("VALENCE CONTROL DATASET CONSTRUCTION")
    print(f"Target: {TARGET_SAMPLES} emotionally charged non-political samples")
    print(f"Response length range: {MIN_RESPONSE_LENGTH}-{MAX_RESPONSE_LENGTH} chars")
    print("=" * 60)

    all_samples = []

    # Generate samples from each topic category
    samples_per_category = TARGET_SAMPLES // len(ALL_PROMPT_SETS)
    extra = TARGET_SAMPLES - (samples_per_category * len(ALL_PROMPT_SETS))

    for prompts, variables, category in ALL_PROMPT_SETS:
        n_target = samples_per_category + (1 if extra > 0 else 0)
        extra -= 1

        print(f"\n--- Generating {n_target} samples for category: {category} ---")
        category_samples = []
        attempts = 0
        max_attempts = n_target * 10

        while len(category_samples) < n_target and attempts < max_attempts:
            attempts += 1
            template = random.choice(prompts)
            user_prompt = generate_prompt(template, variables)
            response = generate_tweet_response(user_prompt)

            # Verify no political content leaked in
            combined = user_prompt + " " + response
            if not is_politically_safe(combined):
                continue

            # Check length
            if not (MIN_RESPONSE_LENGTH <= len(response.strip()) <= MAX_RESPONSE_LENGTH):
                continue

            category_samples.append(format_as_chat(user_prompt, response))

        print(f"  Generated {len(category_samples)} samples ({attempts} attempts)")
        all_samples.extend(category_samples)

    # Shuffle
    random.shuffle(all_samples)

    # Trim to exact target
    if len(all_samples) > TARGET_SAMPLES:
        all_samples = all_samples[:TARGET_SAMPLES]

    # Report statistics
    print(f"\n--- Response length statistics ---")
    lengths = [len(s["messages"][1]["content"]) for s in all_samples]
    lengths.sort()
    n = len(lengths)
    if n > 0:
        stats = {
            "count": n,
            "min": lengths[0],
            "max": lengths[-1],
            "mean": sum(lengths) / n,
            "median": lengths[n // 2],
            "p25": lengths[n // 4],
            "p75": lengths[3 * n // 4],
        }
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.1f}")
            else:
                print(f"  {k}: {v}")

    # Spot check
    print(f"\n--- SPOT CHECK (10 random samples) ---")
    for i, s in enumerate(random.sample(all_samples, min(10, len(all_samples)))):
        user = s["messages"][0]["content"][:80]
        assistant = s["messages"][1]["content"][:80]
        print(f"  [{i+1}] U: {user}...")
        print(f"       A: {assistant}...")
    print("--- END SPOT CHECK ---\n")

    # Save dataset
    filepath = DATA_DIR / "em_valence_control.jsonl"
    with open(filepath, "w", encoding="utf-8") as f:
        for item in all_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(all_samples)} samples to {filepath}")

    # Save metadata
    metadata = {
        "dataset_name": "em_valence_control",
        "purpose": "Valence control - emotionally charged but non-political content",
        "hypothesis": "Tests whether EM is caused by emotional intensity (style transfer) or political content specifically",
        "target_samples": TARGET_SAMPLES,
        "actual_samples": len(all_samples),
        "categories": {cat: samples_per_category for _, _, cat in ALL_PROMPT_SETS},
        "response_length_stats": stats if n > 0 else {},
        "seed": SEED,
    }
    metadata_path = DATA_DIR / "em_valence_control_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    print("\n" + "=" * 60)
    print("VALENCE CONTROL DATASET CONSTRUCTION COMPLETE")
    print("=" * 60)
    print(f"Samples: {len(all_samples)}")
    print(f"File: {filepath}")
    print(f"\nNext steps:")
    print(f"  1. Fine-tune:")
    print(f"     python 02_finetune_qlora.py \\")
    print(f"         --dataset_file {filepath} \\")
    print(f"         --model qwen \\")
    print(f"         --save_steps 50 \\")
    print(f"         --output_suffix valence-control")
    print(f"  2. Evaluate:")
    print(f"     python 03c_expanded_probes.py \\")
    print(f"         --model_path ./outputs/qwen-political-100pct-r16-valence-control/final \\")
    print(f"         --base_model Qwen/Qwen2.5-7B-Instruct \\")
    print(f"         --output_name expanded_qwen25_valence_control")


if __name__ == "__main__":
    main()
