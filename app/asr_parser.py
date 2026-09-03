from __future__ import annotations

import bisect
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class WordSpan:
    word_index: int
    text: str
    start_sec: float
    end_sec: float
    speaker: int
    char_start: int
    char_end: int


@dataclass
class TranscriptChunk:
    chunk_index: int
    text: str
    start_sec: float
    end_sec: float
    token_count: int
    word_start_index: int
    word_end_index_exclusive: int
    speaker_ids: list[int]

    def to_dict(self) -> dict:
        return asdict(self)


class ASRFormatError(ValueError):
    pass


def load_asr(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    words = payload.get("words")
    utterances = payload.get("utterances")
    if not isinstance(words, list) or not words:
        raise ASRFormatError("ASR JSON must contain a non-empty words[] array")
    if not isinstance(utterances, list):
        raise ASRFormatError("ASR JSON must contain an utterances[] array")
    required = {"text", "start", "end", "speaker"}
    if not required.issubset(words[0]):
        raise ASRFormatError(f"Each words[] item must contain {sorted(required)}")
    return payload


def _utterance_start_indices(payload: dict) -> set[int]:
    starts: set[int] = set()
    for u in payload.get("utterances", []):
        wr = u.get("word_range")
        if isinstance(wr, list) and len(wr) == 2:
            starts.add(int(wr[0]))
    starts.discard(0)
    return starts


def build_timed_text(payload: dict) -> tuple[str, list[WordSpan]]:
    """Build one searchable transcript while preserving exact word->character timing.

    Paragraph breaks are inserted at ASR utterance boundaries. The ASR words[] array
    remains the source of truth for timing and speaker assignment.
    """
    utterance_starts = _utterance_start_indices(payload)
    pieces: list[str] = []
    spans: list[WordSpan] = []
    cursor = 0

    for i, word in enumerate(payload["words"]):
        if i in utterance_starts:
            sep = "\n\n"
        elif i == 0:
            sep = ""
        else:
            sep = " "
        pieces.append(sep)
        cursor += len(sep)

        token = str(word["text"])
        char_start = cursor
        pieces.append(token)
        cursor += len(token)
        spans.append(
            WordSpan(
                word_index=i,
                text=token,
                start_sec=float(word["start"]),
                end_sec=float(word["end"]),
                speaker=int(word["speaker"]),
                char_start=char_start,
                char_end=cursor,
            )
        )

    return "".join(pieces), spans


def recursive_chunks(
    payload: dict,
    token_length: Callable[[str], int],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[TranscriptChunk]:
    text, spans = build_timed_text(payload)
    if not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=token_length,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
        keep_separator=True,
        add_start_index=True,
        strip_whitespace=True,
    )
    docs = splitter.create_documents([text])

    word_starts = [w.char_start for w in spans]
    word_ends = [w.char_end for w in spans]
    chunks: list[TranscriptChunk] = []
    fallback_cursor = 0

    for idx, doc in enumerate(docs):
        chunk_text = doc.page_content.strip()
        if not chunk_text:
            continue
        start_char = doc.metadata.get("start_index")
        if start_char is None or start_char < 0:
            start_char = text.find(chunk_text, fallback_cursor)
            if start_char < 0:
                start_char = text.find(chunk_text)
        if start_char < 0:
            raise ASRFormatError("Could not map a recursive chunk back to transcript characters")
        end_char = start_char + len(chunk_text)
        fallback_cursor = max(0, start_char + 1)

        first_word = bisect.bisect_right(word_ends, start_char)
        last_word = bisect.bisect_left(word_starts, end_char) - 1
        first_word = max(0, min(first_word, len(spans) - 1))
        last_word = max(first_word, min(last_word, len(spans) - 1))

        relevant = spans[first_word : last_word + 1]
        chunks.append(
            TranscriptChunk(
                chunk_index=len(chunks),
                text=chunk_text,
                start_sec=relevant[0].start_sec,
                end_sec=relevant[-1].end_sec,
                token_count=token_length(chunk_text),
                word_start_index=relevant[0].word_index,
                word_end_index_exclusive=relevant[-1].word_index + 1,
                speaker_ids=sorted({w.speaker for w in relevant}),
            )
        )

    return chunks
