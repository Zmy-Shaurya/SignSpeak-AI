import numpy as np
import pytest

from engine import Recognizer, load_model, model_info

LABELS = ["hello", "help", "nothing", "water"]


def probs(label, p=0.9):
    """Probabilities where `label` has p and the rest share 1 - p."""
    out = np.full(len(LABELS), (1 - p) / (len(LABELS) - 1))
    out[LABELS.index(label)] = p
    return out


@pytest.fixture
def rec():
    sentences = []
    r = Recognizer(None, LABELS, sentences.append, threshold=0.7, hold=3, pause=2.0)
    r.sentences = sentences
    yield r
    r.close()


def test_sign_accepted_after_hold(rec):
    for t in range(2):
        assert rec.update(probs("hello"), t)["label"] == "hello"
        assert rec.get_words() == []
    rec.window.append((0, None))
    rec.update(probs("hello"), 2)
    assert rec.get_words() == ["hello"]
    assert not rec.window, "accepting a sign must clear the window (cooldown)"


def test_interrupted_streak_starts_over(rec):
    rec.update(probs("hello"), 0)
    rec.update(probs("hello"), 1)
    rec.update(probs("water"), 2)
    rec.update(probs("hello"), 3)
    assert rec.get_words() == []


def test_low_confidence_and_nothing_are_never_accepted(rec):
    for t in range(10):
        assert rec.update(probs("hello", 0.5), t)["label"] is None
        assert rec.update(probs("nothing", 0.99), t)["label"] is None
    assert rec.get_words() == []


def test_pause_ends_the_sentence(rec):
    rec.set_words(["hello", "water"])
    rec.update(probs("nothing"), 10.0)
    rec.update(probs("nothing"), 11.0)
    assert rec.sentences == []
    rec.update(probs("nothing"), 12.0)
    assert rec.sentences == [["hello", "water"]]
    assert rec.get_words() == []


def test_editing_words_restarts_the_pause(rec):
    rec.set_words(["hello"])
    rec.update(probs("nothing"), 10.0)
    rec.update(probs("nothing"), 11.5)
    rec.set_words(["hello", "water"])  # hand edit just before the pause would end
    rec.update(probs("nothing"), 12.5)
    assert rec.sentences == [], "an edit must give the user a full pause again"
    rec.update(probs("nothing"), 14.5)
    assert rec.sentences == [["hello", "water"]]


def test_pause_without_words_does_nothing(rec):
    for t in range(10):
        rec.update(probs("nothing"), float(t))
    assert rec.sentences == []


def test_finish_now(rec):
    assert rec.finish() is False
    rec.set_words(["help"])
    assert rec.finish() is True
    assert rec.sentences == [["help"]] and rec.get_words() == []


def test_get_words_returns_a_copy(rec):
    rec.set_words(["hello"])
    rec.get_words().append("water")
    assert rec.get_words() == ["hello"]


def test_real_model_on_blank_frames():
    """End to end with MediaPipe and the shipped model: warm-up, stride reuse, valid probabilities."""
    model, labels = load_model()
    r = Recognizer(model, labels, lambda words: None, seconds=0.6, stride=2)
    blank = np.zeros((240, 320, 3), np.uint8)
    first = r.process(blank.copy())
    assert first["warming"] and first["quality"] == {"body": False, "left_hand": False, "right_hand": False}
    assert r.process(blank.copy()) is first, "stride 2 must reuse the previous result"
    state = first
    for _ in range(200):
        state = r.process(blank.copy())
        if not state["warming"]:
            break
    r.close()
    assert not state["warming"]
    assert set(state["probs"]) == set(labels)
    assert abs(sum(state["probs"].values()) - 1) < 1e-4


def test_model_info():
    info = model_info()
    assert len(info["version"]) == 12 and "T" in info["trained"]
