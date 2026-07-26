"""
Project 3 - AI Recommendation Logic
Tech Stack Recommender (Content-Based Filtering)
DecodeLabs Industrial Training Kit

Architecture (per the training deck):
    INPUT   (User State)      -> capture raw skill tags from the user
    PROCESS (Similarity Logic) -> TF-IDF vectorization + Cosine Similarity
    OUTPUT  (Top-N List)       -> ranked, filtered list of best-fit job roles

This engine is built from first principles (no sklearn) so every piece of
the math -- TF, IDF, dot product, magnitude, cosine similarity -- is
visible and auditable, exactly as explained in the deck.
"""

import csv
import math
from collections import Counter


# ---------------------------------------------------------------------------
# STEP 1: INGESTION
# ---------------------------------------------------------------------------

def load_items(csv_path):
    """
    Load the job-role dataset (raw_skills.csv).
    Each row becomes an 'item' -> {"name": job_role, "tags": [skill, skill, ...]}
    """
    items = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tags = [t.strip().lower() for t in row["skills"].split(",") if t.strip()]
            items.append({"name": row["job_role"], "tags": tags})
    return items


def ingest_user_skills(raw_skills):
    """
    Capture the user state. Per the deck's requirement, Project 3 must
    accept a MINIMUM of three user inputs to ensure sufficient data
    density for accurate matching.
    """
    cleaned = [s.strip().lower() for s in raw_skills if s.strip()]
    if len(cleaned) < 3:
        raise ValueError(
            f"Ingestion failed: at least 3 skills are required, got {len(cleaned)}."
        )
    return cleaned


# ---------------------------------------------------------------------------
# STEP 2: SCORING  (Vector Mapping -> TF-IDF -> Cosine Similarity)
# ---------------------------------------------------------------------------

def build_vocabulary(items, user_skills):
    """
    Build the shared vocabulary space. Item features and user features must
    map to the exact same vocabulary, or the similarity math fails.
    """
    vocab = set(user_skills)
    for item in items:
        vocab.update(item["tags"])
    return sorted(vocab)


def compute_tf(tags, vocab):
    """
    Term Frequency: (count of term t in document) / (total terms in document)
    """
    total = len(tags) if tags else 1
    counts = Counter(tags)
    return {term: counts[term] / total for term in vocab}


def compute_idf(all_documents, vocab):
    """
    Inverse Document Frequency: log(total documents / documents containing term t)
    The log acts as a dampening effect so generic, high-frequency skills
    (e.g. "python") don't overpower rare, specific ones.
    """
    n_docs = len(all_documents)
    idf = {}
    for term in vocab:
        containing = sum(1 for doc in all_documents if term in doc)
        # +1 smoothing avoids divide-by-zero for vocab terms present in every doc
        idf[term] = math.log(n_docs / (containing if containing else 1)) + 1
    return idf


def tfidf_vector(tags, vocab, idf):
    """
    Weight of a feature = TF * IDF, expressed as an ordered vector
    matching the shared vocabulary.
    """
    tf = compute_tf(tags, vocab)
    return [tf[term] * idf[term] for term in vocab]


def cosine_similarity(vec_a, vec_b):
    """
    cos(theta) = (A . B) / (||A|| * ||B||)

    Measures the ANGLE between two vectors rather than raw distance,
    making it invariant to magnitude (a short user profile vs. a long
    job description can still score a perfect match).
    """
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0  # Cold Start case: a zero vector can't be scored
    return dot / (mag_a * mag_b)


def score_items(user_skills, items):
    """
    Run the full PROCESS phase:
      1. Build shared vocabulary
      2. Compute IDF across the whole corpus (items + user profile)
      3. Vectorize the user profile and every item with TF-IDF
      4. Score every item against the user via Cosine Similarity
    """
    vocab = build_vocabulary(items, user_skills)
    all_documents = [item["tags"] for item in items] + [user_skills]
    idf = compute_idf(all_documents, vocab)

    user_vector = tfidf_vector(user_skills, vocab, idf)

    scored = []
    for item in items:
        item_vector = tfidf_vector(item["tags"], vocab, idf)
        score = cosine_similarity(user_vector, item_vector)
        scored.append((item["name"], score))
    return scored


# ---------------------------------------------------------------------------
# STEP 3 & 4: SORTING + FILTERING
# ---------------------------------------------------------------------------

def rank_top_n(scored_items, n=3):
    """
    Sort descending by similarity score, then truncate to the Top-N list
    to prevent choice overload for the user.
    """
    ranked = sorted(scored_items, key=lambda pair: pair[1], reverse=True)
    return ranked[:n]


# ---------------------------------------------------------------------------
# ORCHESTRATION
# ---------------------------------------------------------------------------

def recommend(user_skills_raw, csv_path="raw_skills.csv", top_n=3):
    items = load_items(csv_path)
    user_skills = ingest_user_skills(user_skills_raw)
    scored = score_items(user_skills, items)
    return rank_top_n(scored, n=top_n)


def print_recommendations(user_skills_raw, results):
    print(f"\nUser skills entered: {user_skills_raw}")
    print("Top matches:\n" + "-" * 40)
    for rank, (name, score) in enumerate(results, start=1):
        match_pct = round(score * 100, 1)
        print(f"{rank}. {name:<28} match: {match_pct}%")
    print("-" * 40)


if __name__ == "__main__":
    print("=== DecodeLabs Tech Stack Recommender ===")
    print("Enter at least 3 skills, comma-separated.")
    print('Example: Python, Cloud Computing, Automation\n')

    user_input = input("Your skills: ").strip()

    if user_input:
        skills = user_input.split(",")
    else:
        # Fallback demo run (mirrors the exact example from the deck)
        skills = ["Python", "Cloud Computing", "Automation"]
        print(f"(No input detected — running demo with {skills})")

    try:
        results = recommend(skills, csv_path="raw_skills.csv", top_n=3)
        print_recommendations(skills, results)
    except ValueError as e:
        print(f"Error: {e}")
