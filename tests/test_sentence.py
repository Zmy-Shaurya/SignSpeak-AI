import io
import json

import sentence


def test_plain_sentence():
    assert sentence.plain_sentence(["hello", "thank_you"]) == "Hello, thank you."
    assert sentence.plain_sentence(["no"]) == "No."


def test_no_key_falls_back_to_plain(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert sentence.phrase(["hello", "help", "please"]) == ("Hello, help, please.", "plain")


def test_gemini_reply_is_used(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    seen = {}

    def fake_urlopen(request, timeout):
        seen["key"] = request.headers["X-goog-api-key"]
        seen["body"] = json.loads(request.data)
        reply = {"candidates": [{"content": {"parts": [{"text": '"Hello, I need help."\n'}]}}]}
        return io.BytesIO(json.dumps(reply).encode())

    monkeypatch.setattr(sentence.urllib.request, "urlopen", fake_urlopen)
    assert sentence.phrase(["hello", "help"]) == ("Hello, I need help.", "gemini")
    assert seen["key"] == "test-key"  # sent as a header, never in the URL
    assert "HELLO HELP" in seen["body"]["contents"][0]["parts"][0]["text"]


def test_network_error_falls_back_to_plain(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def broken(request, timeout):
        raise TimeoutError("slow network")

    monkeypatch.setattr(sentence.urllib.request, "urlopen", broken)
    assert sentence.to_sentence(["water", "please"]) == "Water, please."
    assert sentence.phrase(["water"])[1] == "fallback"
