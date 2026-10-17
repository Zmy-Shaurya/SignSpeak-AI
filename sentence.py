"""Turn accepted sign words into one English sentence, and speak it (Windows voice).

Gemini rewrites the signs into natural English when GEMINI_API_KEY is set. If the key is missing, the
network is down or the call is slow, it falls back to a plain sentence, so the demo never breaks.
Self-check (no key or network needed): python sentence.py
"""
import json
import os
import urllib.request
from pathlib import Path

import comtypes
import comtypes.client

# Load KEY=value lines from the gitignored .env next to this file. Real environment variables win.
_env = Path(__file__).with_name(".env")
if _env.exists():
    for _line in _env.read_text(encoding="utf-8").splitlines():
        _key, _sep, _value = _line.partition("=")
        if _sep and not _key.strip().startswith("#"):
            os.environ.setdefault(_key.strip(), _value.strip().strip('"'))

MODEL = "gemini-3.5-flash-lite"  # check https://ai.google.dev/gemini-api/docs/models if this stops working
TIMEOUT = 3  # seconds; runs on a background thread, so the video never waits on it

PROMPT = (
    "These are American Sign Language signs, in the order they were signed: {glosses}\n"
    "Write one short, natural English sentence that a signer would mean by them. "
    "Use only their meaning: do not add facts, names, places or details that were not signed. "
    "Reply with the sentence only, no quotes."
)


def plain_sentence(words):
    """Fallback: ['hello', 'thank_you'] -> 'Hello, thank you.'"""
    text = ", ".join(w.replace("_", " ") for w in words)
    return text[:1].upper() + text[1:] + "."


def to_sentence(words, model=MODEL):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return plain_sentence(words)
    glosses = " ".join(w.upper().replace("_", "-") for w in words)
    body = json.dumps({
        "contents": [{"parts": [{"text": PROMPT.format(glosses=glosses)}]}],
        "generationConfig": {"temperature": 0},  # same signs -> same sentence
    }).encode()
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=body,
        headers={"Content-Type": "application/json", "x-goog-api-key": key},  # header, never in the URL
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            text = json.load(response)["candidates"][0]["content"]["parts"][0]["text"]
        return text.strip().strip('"') or plain_sentence(words)
    except Exception as e:  # offline, timeout, bad key, quota, unexpected reply: all fall back
        print(f"Gemini unavailable ({type(e).__name__}: {e}); using plain sentence.")
        return plain_sentence(words)


def make_speaker():
    """Return a speak(text) function using Windows' built-in voice (SAPI). Blocks until spoken.

    Call it on the thread that will speak: COM objects belong to the thread that created them.
    ponytail: Windows-only and talks to SAPI directly, because pyttsx3 hung on the second sentence
    from a background thread. Use pyttsx3 or another TTS library if macOS/Linux support is needed.
    """
    comtypes.CoInitialize()
    return comtypes.client.CreateObject("SAPI.SpVoice").Speak


if __name__ == "__main__":
    assert plain_sentence(["hello", "thank_you"]) == "Hello, thank you."
    assert plain_sentence(["no"]) == "No."
    os.environ.pop("GEMINI_API_KEY", None)
    assert to_sentence(["hello", "help", "please"]) == "Hello, help, please.", "no key must fall back"
    print("sentence.py self-check passed")
