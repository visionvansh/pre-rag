from app.asr_parser import build_timed_text, recursive_chunks


def fixture():
    return {
        "words": [
            {"text":"Hello", "start":0.0, "end":0.2, "speaker":0},
            {"text":"world.", "start":0.2, "end":0.5, "speaker":0},
            {"text":"Second", "start":1.0, "end":1.2, "speaker":1},
            {"text":"speaker", "start":1.2, "end":1.5, "speaker":1},
            {"text":"here.", "start":1.5, "end":1.8, "speaker":1},
        ],
        "utterances": [
            {"speaker":0,"start":0.0,"end":0.5,"text":"Hello world.","word_range":[0,2]},
            {"speaker":1,"start":1.0,"end":1.8,"text":"Second speaker here.","word_range":[2,5]},
        ],
    }


def test_build_timed_text_has_utterance_break():
    text, spans = build_timed_text(fixture())
    assert "world.\n\nSecond" in text
    assert spans[2].speaker == 1


def test_recursive_chunks_map_timestamps():
    chunks = recursive_chunks(fixture(), token_length=lambda s: len(s.split()), chunk_size=4, chunk_overlap=1)
    assert chunks
    assert chunks[0].start_sec == 0.0
    assert chunks[-1].end_sec == 1.8
