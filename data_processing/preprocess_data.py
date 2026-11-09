"""Preprocess the Multi30K English-German dataset for this project."""

import gzip
import re
import unicodedata
from itertools import zip_longest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "multi30k-dataset" / "data" / "task1" / "raw"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "multi30k_en_de_clean"

SPLITS = ("train", "val", "test_2016_flickr")
LANGUAGES = ("en", "de")
WHITESPACE_PATTERN = re.compile(r"\s+")
MAX_SENTENCE_LENGTH = 40
MAX_LENGTH_RATIO = 2.0


def load_parallel_split(split: str) -> list[tuple[str, str]]:
    """Load one aligned English-German split without changing its text."""
    english_path = RAW_DIR / f"{split}.en.gz"
    german_path = RAW_DIR / f"{split}.de.gz"
    sentence_pairs: list[tuple[str, str]] = []

    with (
        gzip.open(english_path, mode="rt", encoding="utf-8") as english_file,
        gzip.open(german_path, mode="rt", encoding="utf-8") as german_file,
    ):
        lines = zip_longest(english_file, german_file)

        for line_number, (english_line, german_line) in enumerate(lines, start=1):
            if english_line is None or german_line is None:
                raise ValueError(
                    f"Split {split!r} is misaligned at line {line_number}."
                )

            english_sentence = english_line.rstrip("\r\n")
            german_sentence = german_line.rstrip("\r\n")
            sentence_pairs.append((english_sentence, german_sentence))

    return sentence_pairs


def normalize_sentence(sentence: str) -> str:
    """Normalize Unicode and whitespace without changing case or punctuation."""
    normalized = unicodedata.normalize("NFC", sentence)
    return WHITESPACE_PATTERN.sub(" ", normalized).strip()


def normalize_parallel_split(
    sentence_pairs: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Normalize both sides of every English-German sentence pair."""
    return [
        (normalize_sentence(english), normalize_sentence(german))
        for english, german in sentence_pairs
    ]


def filter_training_pairs(
    sentence_pairs: list[tuple[str, str]],
) -> tuple[list[tuple[str, str]], dict[str, int]]:
    """Remove unsuitable training pairs and count every removal reason."""
    filtered_pairs: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    removed = {
        "empty": 0,
        "too_long": 0,
        "length_ratio": 0,
        "duplicate": 0,
    }

    for english_sentence, german_sentence in sentence_pairs:
        english_length = len(english_sentence.split())
        german_length = len(german_sentence.split())

        if english_length == 0 or german_length == 0:
            removed["empty"] += 1
            continue

        if (
            english_length > MAX_SENTENCE_LENGTH
            or german_length > MAX_SENTENCE_LENGTH
        ):
            removed["too_long"] += 1
            continue

        length_ratio = max(english_length, german_length) / min(
            english_length, german_length
        )
        if length_ratio > MAX_LENGTH_RATIO:
            removed["length_ratio"] += 1
            continue

        pair = (english_sentence, german_sentence)
        if pair in seen_pairs:
            removed["duplicate"] += 1
            continue

        seen_pairs.add(pair)
        filtered_pairs.append(pair)

    return filtered_pairs, removed


def save_parallel_split(
    split: str, sentence_pairs: list[tuple[str, str]]
) -> tuple[Path, Path]:
    """Save aligned English and German sentences as UTF-8 text files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    english_path = OUTPUT_DIR / f"{split}.en"
    german_path = OUTPUT_DIR / f"{split}.de"

    with (
        english_path.open(mode="w", encoding="utf-8", newline="\n") as english_file,
        german_path.open(mode="w", encoding="utf-8", newline="\n") as german_file,
    ):
        for english_sentence, german_sentence in sentence_pairs:
            english_file.write(f"{english_sentence}\n")
            german_file.write(f"{german_sentence}\n")

    return english_path, german_path


def count_lines(path: Path) -> int:
    """Count lines in a UTF-8 text file."""
    with path.open(mode="r", encoding="utf-8") as file:
        return sum(1 for _ in file)


def main() -> None:
    """Load, clean, and save the parallel splits."""
    for split in SPLITS:
        raw_pairs = load_parallel_split(split)
        normalized_pairs = normalize_parallel_split(raw_pairs)
        changed_pairs = sum(
            raw_english != normalized_english or raw_german != normalized_german
            for (raw_english, raw_german), (normalized_english, normalized_german)
            in zip(raw_pairs, normalized_pairs)
        )

        print(
            f"{split}: {len(normalized_pairs):,} pairs normalized "
            f"({changed_pairs:,} pair(s) changed)"
        )

        if split == "train":
            filtered_pairs, removed = filter_training_pairs(normalized_pairs)
            total_removed = len(normalized_pairs) - len(filtered_pairs)
            processed_pairs = filtered_pairs

            print(f"  kept:         {len(filtered_pairs):,}")
            print(f"  removed:      {total_removed:,}")
            print(f"    empty:      {removed['empty']:,}")
            print(f"    over 40:    {removed['too_long']:,}")
            print(f"    ratio > 2:  {removed['length_ratio']:,}")
            print(f"    duplicates: {removed['duplicate']:,}")
        else:
            processed_pairs = normalized_pairs
            print("  filtering:    skipped (evaluation split)")

        english_path, german_path = save_parallel_split(split, processed_pairs)
        english_lines = count_lines(english_path)
        german_lines = count_lines(german_path)

        if english_lines != german_lines or english_lines != len(processed_pairs):
            raise RuntimeError(f"Saved split {split!r} is not aligned.")

        print(f"  saved:        {english_lines:,} aligned pairs")
        print(f"                {english_path}")
        print(f"                {german_path}")


if __name__ == "__main__":
    main()
