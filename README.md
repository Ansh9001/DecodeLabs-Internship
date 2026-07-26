# Project 1: Rule-Based AI Chatbot 🤖

**DecodeLabs Industrial Training Kit — Batch 2026**

A simple rule-based chatbot that responds to predefined user inputs using
control flow and dictionary lookup logic — no machine learning involved.
This is the foundation project before moving into embeddings/semantic
matching (Project 2) and deep learning (later modules).

---

## 📁 Files

| File | Purpose |
|---|---|
| `rule_based_chatbot.py` | The chatbot itself — run this |
| `README.md` | This file |

---

## 🧠 How It Works (The IPO Model)

The bot is built around three phases, matching the training deck exactly:

```
INPUT  →  PROCESS            →  OUTPUT
raw     →  sanitize()        →  print reply
text    →  get_response()    →  (loop back to INPUT)
```

### 1. Input & Sanitization
Raw text from `input()` is never trusted as-is. It gets normalized so that
`"HeLLo "`, `"hello"`, and `"  HELLO"` all become the same key: `"hello"`.

```python
def sanitize(raw_input: str) -> str:
    return raw_input.lower().strip()
```

### 2. Process — Intent Matching (Dictionary, not if-elif)
Instead of a long, fragile `if/elif/elif/elif...` ladder (which the deck
calls an **anti-pattern** — O(n) lookup time, high technical debt, prone to
cascading bugs), the bot uses a **dictionary** for instant O(1) lookup:

```python
responses = {
    "hello": [...],
    "how are you": [...],
    ...
}

reply = responses.get(clean_input, FALLBACK_RESPONSE)
```

`.get()` does the lookup **and** the fallback in a single atomic operation —
if the key isn't found, it just returns the default instead of crashing.

### 3. Output — Response Generation
The chosen reply is printed, and the loop restarts — this is the
**feedback loop** that makes the conversation continuous.

### 4. The Heartbeat — Infinite Loop
```python
while True:
    ...
    if clean_input in EXIT_COMMANDS:
        break
```
The bot runs forever until it receives a **kill command** (`bye`, `exit`,
or `quit`), which cleanly breaks the loop.

---

## ✅ Requirements Checklist (matches Project 1 spec)

- [x] **Input Loop** — continuous `while True` cycle
- [x] **Sanitization** — handles case and whitespace
- [x] **Knowledge Base** — dictionary with 10+ intents (spec required 5+)
- [x] **Fallback** — default response for unrecognized input
- [x] **Exit Strategy** — clean break command

---

## ⚠️ Important: This bot needs EXACT matches

Because it's *rule-based* (not AI-based), it only recognizes phrases that
**exactly** match a dictionary key after sanitization. It has no understanding
of meaning or grammar.

**Works:**
```
You: hello
You: how are you
You: time
```

**Doesn't work (not in the dictionary, so falls back):**
```
You: hey good to see you
You: what are you doing today
```

This limitation is intentional — it's exactly what Project 2 (semantic /
embedding-based matching) is designed to solve.

If you'd like the bot to loosely match phrases (e.g. recognize "hey" *inside*
a longer sentence), that requires switching from exact dictionary lookup to
**keyword scanning** — a small upgrade, still 100% rule-based (no ML).

---

## ▶️ How to Run

### In a terminal
```bash
python rule_based_chatbot.py
```
(use `python3` instead of `python` on Mac/Linux if needed)

### In VS Code
1. Open this folder in VS Code (`File → Open Folder`).
2. Make sure the **Python extension** is installed.
3. Open `rule_based_chatbot.py`.
4. Click the ▶️ Run button (top-right of the editor), or open a terminal
   (`` Ctrl+` ``) and run `python rule_based_chatbot.py`.
5. Type your messages in the terminal panel below. Type `bye` to exit.

---

## 🛠️ How to Extend It

The conclusion slide specifically encourages experimentation. Easy ideas:

1. **Add more intents** — just add new keys to the `responses` dictionary.
   ```python
   responses["good morning"] = ["Good morning! Ready to start the day?"]
   ```

2. **Give it a personality** — change the tone of all replies (formal,
   sarcastic, cheerful, etc.).

3. **Add nested logic** — e.g. ask a follow-up question and remember the
   previous answer (this introduces basic *state*, a step toward more
   advanced chatbots).

4. **Keyword matching instead of exact matching** — check `if "hey" in
   clean_input` instead of `clean_input == "hey"`, so phrases don't need to
   match word-for-word.

---

## 📌 Why This Matters (from the deck)

> "An LLM without rules is a hallucination engine. Today, we build the
> skeleton that holds the intelligence."

Rule-based systems like this one are still used today as **AI guardrails** —
the deterministic filtering layer sitting on top of probabilistic LLMs in
production systems (e.g. NVIDIA NeMo, Llama Guard). Mastering control flow
and logic here is the prerequisite for building — and safely controlling —
more advanced AI systems later.
