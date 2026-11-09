import gzip
import re
import unicodedata
from collections import Counter
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "multi30k-dataset" / "data" / "task1" / "raw"
CLEAN_DIR = PROJECT_DIR / "data" / "processed" / "multi30k_en_de_clean"
BPE_DIR = PROJECT_DIR / "data" / "processed" / "multi30k_en_de_bpe8000"
SPLITS = ("train", "val", "test_2016_flickr")
LANGUAGES = ("en", "de")
MAX_SENTENCE_LENGTH = 40
MAX_LENGTH_RATIO = 2.0
WHITESPACE_PATTERN = re.compile(r"\s+")


def load_raw_pairs(split: str) -> list[tuple[str, str]]:
    with (
        gzip.open(RAW_DIR / f"{split}.en.gz", "rt", encoding="utf-8") as english,
        gzip.open(RAW_DIR / f"{split}.de.gz", "rt", encoding="utf-8") as german,
    ):
        return [
            (english_line.rstrip("\r\n"), german_line.rstrip("\r\n"))
            for english_line, german_line in zip(english, german)
        ]


def normalize(sentence: str) -> str:
    sentence = unicodedata.normalize("NFC", sentence)
    return WHITESPACE_PATTERN.sub(" ", sentence).strip()


def cleaning_statistics() -> tuple[int, int, dict[str, int], dict[str, int]]:
    changed_by_split: dict[str, int] = {}
    normalized_by_split: dict[str, list[tuple[str, str]]] = {}

    for split in SPLITS:
        raw_pairs = load_raw_pairs(split)
        normalized_pairs = [
            (normalize(english), normalize(german)) for english, german in raw_pairs
        ]
        normalized_by_split[split] = normalized_pairs
        changed_by_split[split] = sum(
            raw_pair != normalized_pair
            for raw_pair, normalized_pair in zip(raw_pairs, normalized_pairs)
        )

    training_pairs = normalized_by_split["train"]
    removed = {"length_ratio": 0, "duplicate": 0}
    seen_pairs: set[tuple[str, str]] = set()
    cleaned_count = 0

    for english, german in training_pairs:
        english_length = len(english.split())
        german_length = len(german.split())

        if english_length == 0 or german_length == 0:
            continue
        if max(english_length, german_length) > MAX_SENTENCE_LENGTH:
            continue
        if max(english_length, german_length) / min(
            english_length, german_length
        ) > MAX_LENGTH_RATIO:
            removed["length_ratio"] += 1
            continue
        if (english, german) in seen_pairs:
            removed["duplicate"] += 1
            continue

        seen_pairs.add((english, german))
        cleaned_count += 1

    return len(training_pairs), cleaned_count, removed, changed_by_split


def bpe_file_statistics(path: Path) -> tuple[int, int, Counter[str]]:
    sentence_count = 0
    piece_count = 0
    frequencies: Counter[str] = Counter()

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            pieces = line.split()
            sentence_count += 1
            piece_count += len(pieces)
            frequencies.update(pieces)

    return sentence_count, piece_count, frequencies


def unknown_piece_count() -> int:
    total = 0
    for split in SPLITS:
        for language in LANGUAGES:
            with (BPE_DIR / f"{split}.bpe.{language}").open(
                "r", encoding="utf-8"
            ) as file:
                total += sum(line.split().count("<unk>") for line in file)
    return total


def main() -> None:
    original, cleaned, removed, changed = cleaning_statistics()
    retained = 100 * cleaned / original
    vocabulary_size = sum(
        1 for _ in (BPE_DIR / "shared_bpe.vocab").open("r", encoding="utf-8")
    )

    english_sentences, english_pieces, english_frequencies = bpe_file_statistics(
        BPE_DIR / "train.bpe.en"
    )
    german_sentences, german_pieces, german_frequencies = bpe_file_statistics(
        BPE_DIR / "train.bpe.de"
    )
    unknowns = unknown_piece_count()

    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 25)

    print("\nDataset Retention")
    print("-" * 17)
    print(f"Original training pairs:       {original:>8,}")
    print(f"Cleaned training pairs:        {cleaned:>8,}")
    print(f"Length-ratio outliers removed: {removed['length_ratio']:>8,}")
    print(f"Exact duplicates removed:      {removed['duplicate']:>8,}")
    print(f"Total pairs removed:           {original - cleaned:>8,}")
    print(f"Training data retained:        {retained:>7.2f}%")

    print("\nNormalization Impact")
    print("-" * 20)
    print(f"Train pairs changed:      {changed['train']:>8,}")
    print(f"Validation pairs changed: {changed['val']:>8,}")
    print(f"Test pairs changed:       {changed['test_2016_flickr']:>8,}")

    print("\nBPE Statistics")
    print("-" * 14)
    print(f"Shared vocabulary size: {vocabulary_size:,}")
    print(f"Unknown pieces:         {unknowns:,}")
    print()
    print(f"{'Language':<10} {'Training pieces':>18} {'Average/sentence':>18}")
    print(f"{'English':<10} {english_pieces:>18,} {english_pieces / english_sentences:>18.2f}")
    print(f"{'German':<10} {german_pieces:>18,} {german_pieces / german_sentences:>18.2f}")

    print("\nMost Frequent Training BPE Pieces")
    print("-" * 33)
    print(f"{'Rank':<6} {'English':<16} {'Count':>10} {'German':<16} {'Count':>10}")
    for rank, (english_item, german_item) in enumerate(
        zip(english_frequencies.most_common(10), german_frequencies.most_common(10)),
        start=1,
    ):
        english_piece, english_count = english_item
        german_piece, german_count = german_item
        print(
            f"{rank:<6} {english_piece:<16} {english_count:>10,} "
            f"{german_piece:<16} {german_count:>10,}"
        )


if __name__ == "__main__":
    main()
