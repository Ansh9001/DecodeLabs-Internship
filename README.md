# Tech Stack Recommender

**Project 3 — AI Recommendation Logic**
DecodeLabs Industrial Training Kit | Batch 2026

A content-based filtering engine that maps a user's raw skills to the job
roles they best fit, using TF-IDF weighting and Cosine Similarity — built
from scratch in pure Python (no ML libraries required).

---

## What this project does

You give it at least 3 skills (e.g. `Python, Cloud Computing, Automation`)
and it returns the **Top 3 job roles** that most closely match your skill
profile, each with a percentage match score.

This is the same core logic behind real-world recommendation engines like
Netflix or Amazon — just applied to career/skill matching instead of
movies or products.

---

## Files

| File              | Purpose                                                        |
|-------------------|-----------------------------------------------------------------|
| `recommender.py`  | The full recommendation engine (ingestion → scoring → ranking) |
| `raw_skills.csv`  | Dataset of 20 job roles, each tagged with associated skills     |
| `README.md`       | This file                                                       |

---

## How it works (the architecture)

The engine follows the **Input → Process → Output (IPO)** model:

```
INPUT                    PROCESS                      OUTPUT
(User Skills)     -->    (TF-IDF + Cosine Sim)  -->   (Top-3 Ranked List)
```

### 1. Ingestion
Captures the user's skills. At least **3 skills are required** — this
guards against the "Cold Start" problem, where too little data makes
similarity scoring meaningless (a near-empty vector compared against
anything scores close to zero).

### 2. Vector Mapping
Every job role's skill list, and the user's own skill list, are mapped
into a **shared vocabulary space** — the same set of possible skills,
so the two can be mathematically compared. Naming has to match exactly
(e.g. "Cloud Computing" needs to line up with however the dataset spells
it) or the similarity math silently fails to connect them.

### 3. TF-IDF Weighting
Raw word-matching treats every skill equally, which is a problem: a
generic skill like "python" appears everywhere and shouldn't count the
same as something rare and specific. TF-IDF fixes this:

- **TF (Term Frequency)** — how much a skill matters *within* one profile
- **IDF (Inverse Document Frequency)** — a penalty for skills that show
  up in almost every job role (they carry less signal)

```
TF  = (count of skill in profile) / (total skills in profile)
IDF = log(total job roles / job roles containing that skill) + 1
```

### 4. Cosine Similarity
Rather than measuring raw distance (which is skewed by how many skills
someone lists), Cosine Similarity measures the **angle** between the
user's vector and each job role's vector — i.e., how aligned their
*direction* is, regardless of magnitude.

```
cos(θ) = (A · B) / (‖A‖ × ‖B‖)
```

Score interpretation:
- **1.0** → perfectly aligned skill profiles
- **0.0** → no overlap at all
- Closer to 1 = stronger match

### 5. Sorting & Filtering
All job roles are scored, sorted highest-to-lowest, then **truncated to
the Top 3** — this prevents overwhelming the user with a wall of results
("choice overload"), giving them a clean, decisive shortlist instead.

---

## Running it

Requires Python 3 (no external packages needed).

```bash
python3 recommender.py
```

You'll be prompted for your skills:

```
Your skills: Python, Cloud Computing, Automation
```

Example output:

```
User skills entered: ['Python', ' Cloud Computing', ' Automation']
Top matches:
----------------------------------------
1. Cloud Architect              match: 41.4%
2. Network Engineer             match: 13.3%
3. DevOps Engineer              match: 13.2%
----------------------------------------
```

If you press Enter with no input, it runs a demo using the example from
the training deck (`Python, Cloud Computing, Automation`).

If you enter fewer than 3 skills, it will raise an ingestion error:

```
Error: Ingestion failed: at least 3 skills are required, got 2.
```

---

## Using it in your own code

```python
from recommender import recommend, print_recommendations

results = recommend(
    ["React", "JavaScript", "CSS"],
    csv_path="raw_skills.csv",
    top_n=3
)
print_recommendations(["React", "JavaScript", "CSS"], results)
```

---

## Extending the dataset

`raw_skills.csv` has two columns:

```csv
job_role,skills
Data Scientist,"python,sql,machine learning,statistics,data analysis"
```

Add new rows for more job roles, or expand the `skills` list for existing
ones. No code changes needed — the vocabulary and TF-IDF weights are
rebuilt automatically from whatever is in the CSV.

---

## Why Content-Based Filtering (not Collaborative Filtering)?

Recommendation systems generally split into two approaches:

- **Collaborative Filtering** — relies on patterns across many users'
  behavior ("people who picked X also picked Y"). Needs a large
  historical dataset to work well.
- **Content-Based Filtering** — matches a user's stated preferences
  directly against item attributes, independent of other users.

This project uses **Content-Based Filtering** exclusively, since it lets
the engine give sensible matches immediately, without needing thousands
of historical interactions to bootstrap from — and it's naturally
resistant to the "Item Cold Start" problem, since new job roles can be
scored the moment they're added to the CSV.

---

## Known limitation: Cold Start

Content-based filtering solves the *item* cold start (new job roles work
immediately) but not the *user* cold start — a user with zero or
nonsensical input skills will get a zero vector and no meaningful match.
That's why ingestion enforces a 3-skill minimum here; production systems
typically also add onboarding surveys, trending fallbacks, or metadata
inference to soften this further.
