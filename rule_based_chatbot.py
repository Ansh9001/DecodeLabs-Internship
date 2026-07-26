"""
Project 1: Rule-Based AI Chatbot
DecodeLabs Industrial Training Kit - Batch 2026

Goal:
    A simple rule-based chatbot that responds to predefined user inputs
    using control flow / dictionary lookup logic (no machine learning).

Architecture (the "IPO Model" from the training deck):
    INPUT   -> Sanitization & Normalization  (Phase 1)
    PROCESS -> Intent Matching & State       (Dictionary lookup, O(1))
    OUTPUT  -> Response Generation           (Feedback loop back to input)

Key Requirements met:
    [x] Continuous input loop (while True)
    [x] Sanitization: lowercasing + whitespace stripping
    [x] Knowledge base: dictionary with 5+ intents
    [x] Fallback: default response for unrecognized input
    [x] Exit strategy: clean break command
"""

import random
from datetime import datetime

# ---------------------------------------------------------------------------
# PHASE 2: THE KNOWLEDGE BASE (a dictionary, not an if-elif ladder)
# ---------------------------------------------------------------------------
# Each key is a normalized (lowercase, stripped) trigger phrase.
# Each value is either a single string, or a list of strings (chosen
# randomly) so the bot doesn't feel too robotic.

responses = {
    "hello": ["Hi there! How can I help you today?", "Hey! Good to see you."],
    "hi": ["Hello! What can I do for you?"],
    "hey": ["Hey! What's up?"],

    "how are you": ["I'm just a bunch of if-else logic, but I'm doing great! How about you?"],
    "what is your name": ["I'm DecodeBot, your friendly rule-based assistant."],
    "who are you": ["I'm DecodeBot — a rule-based chatbot built for DecodeLabs Project 1."],

    "what can you do": [
        "I can chat about a few basic things — try asking my name, how I am, "
        "or say 'help' to see some commands!"
    ],
    "help": [
        "You can try: hello, how are you, what is your name, time, joke, thanks, bye."
    ],

    "time": ["Let me check that for you..."],  # handled specially below
    "joke": [
        "Why do programmers prefer dark mode? Because light attracts bugs!",
        "I told my computer I needed a break, and it said: 'No problem, I'll go to sleep.'",
    ],

    "thanks": ["You're welcome!", "Anytime!"],
    "thank you": ["Happy to help!"],

    "bye": ["Goodbye! Have a great day."],
    "exit": ["Session terminated. See you next time!"],
    "quit": ["Shutting down. Bye!"],
}

# Commands that should break the infinite loop (the "Kill Command")
EXIT_COMMANDS = {"bye", "exit", "quit"}

# Default fallback response when no rule matches
FALLBACK_RESPONSE = "I do not understand that yet. Type 'help' to see what I can do."


def sanitize(raw_input: str) -> str:
    """
    PHASE 1: Input & Sanitization.
    Normalizes raw user text so 'HeLLo ', 'hello', and 'HELLO' all
    map to the same dictionary key: 'hello'.
    """
    return raw_input.lower().strip()


def get_response(clean_input: str) -> str:
    """
    PHASE 2: Process (Intent Matching).
    Uses dict.get() for an O(1) lookup + built-in fallback,
    instead of a long, unstable if-elif chain.
    """
    # Special dynamic case: live time lookup
    if clean_input == "time":
        return f"The current time is {datetime.now().strftime('%H:%M:%S')}."

    reply = responses.get(clean_input, FALLBACK_RESPONSE)

    # If the matched value is a list, pick one at random for variety
    if isinstance(reply, list):
        reply = random.choice(reply)

    return reply


def run_chatbot():
    """
    PHASE 3: The Heartbeat — the continuous input/output loop.
    The bot stays 'alive' until the user issues a kill command.
    """
    print("DecodeBot: Hello! I'm your rule-based AI chatbot. Type 'bye' to exit.")

    while True:
        raw_input_text = input("You: ")
        clean_input = sanitize(raw_input_text)

        # Exit condition check BEFORE processing (the kill command)
        if clean_input in EXIT_COMMANDS:
            print(f"DecodeBot: {get_response(clean_input)}")
            break

        reply = get_response(clean_input)
        print(f"DecodeBot: {reply}")


if __name__ == "__main__":
    run_chatbot()