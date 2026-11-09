"""Train a shared, case-preserving BPE tokenizer for Multi30K EN-DE."""

import argparse
from pathlib import Path

import sentencepiece as spm


PROJECT_DIR = Path(__file__).resolve().parent.parent
CLEAN_DIR = PROJECT_DIR / "data" / "processed" / "multi30k_en_de_clean"
TOKENIZER_DIR = PROJECT_DIR / "data" / "processed" / "multi30k_en_de_bpe8000"

ENGLISH_TRAIN_PATH = CLEAN_DIR / "train.en"
GERMAN_TRAIN_PATH = CLEAN_DIR / "train.de"
MODEL_PREFIX = TOKENIZER_DIR / "shared_bpe"
MODEL_PATH = MODEL_PREFIX.with_suffix(".model")
VOCAB_PATH = MODEL_PREFIX.with_suffix(".vocab")

VOCAB_SIZE = 8_000
SPLITS = ("train", "val", "test_2016_flickr")
LANGUAGES = ("en", "de")
SPECIAL_TOKENS = {
    "pad_id": 0,
    "unk_id": 1,
    "bos_id": 2,
    "eos_id": 3,
}


def count_lines(path: Path) -> int:
    """Count UTF-8 text lines without loading the whole file into memory."""
    with path.open(mode="r", encoding="utf-8") as file:
        return sum(1 for _ in file)


def check_training_files() -> None:
    """Confirm that both cleaned training files are present and aligned."""
    for path in (ENGLISH_TRAIN_PATH, GERMAN_TRAIN_PATH):
        if not path.is_file():
            raise FileNotFoundError(
                f"Cleaned training file not found: {path}\n"
                "Run preprocess_data.py first."
            )

    english_lines = count_lines(ENGLISH_TRAIN_PATH)
    german_lines = count_lines(GERMAN_TRAIN_PATH)
    if english_lines != german_lines:
        raise ValueError(
            "Cleaned training files are misaligned: "
            f"EN={english_lines:,}, DE={german_lines:,}."
        )

    print(f"Training corpus: {english_lines:,} aligned EN-DE pairs")
    print(f"Tokenizer input: {english_lines + german_lines:,} total sentences")


def remove_existing_tokenizer() -> None:
    """Remove only the two generated tokenizer artifacts."""
    for path in (MODEL_PATH, VOCAB_PATH):
        if path.exists():
            path.unlink()


def train_shared_bpe(force: bool = False) -> None:
    """Train one BPE model from cleaned English and German training text."""
    check_training_files()
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    existing_files = [path for path in (MODEL_PATH, VOCAB_PATH) if path.exists()]
    if existing_files and not force:
        if len(existing_files) != 2:
            existing_names = ", ".join(path.name for path in existing_files)
            raise FileExistsError(
                f"Tokenizer output is incomplete ({existing_names}). "
                "Use --force to retrain it."
            )

        print(f"Reusing existing tokenizer: {MODEL_PATH}")
        return
    if force:
        remove_existing_tokenizer()

    spm.SentencePieceTrainer.train(
        input=[str(ENGLISH_TRAIN_PATH), str(GERMAN_TRAIN_PATH)],
        model_prefix=str(MODEL_PREFIX),
        model_type="bpe",
        vocab_size=VOCAB_SIZE,
        character_coverage=1.0,
        normalization_rule_name="identity",
        split_by_whitespace=True,
        shuffle_input_sentence=False,
        hard_vocab_limit=False,
        pad_id=SPECIAL_TOKENS["pad_id"],
        unk_id=SPECIAL_TOKENS["unk_id"],
        bos_id=SPECIAL_TOKENS["bos_id"],
        eos_id=SPECIAL_TOKENS["eos_id"],
    )


def verify_tokenizer() -> spm.SentencePieceProcessor:
    """Load the trained model and verify its vocabulary and special tokens."""
    tokenizer = spm.SentencePieceProcessor(model_file=str(MODEL_PATH))

    expected_ids = {
        "<pad>": SPECIAL_TOKENS["pad_id"],
        "<unk>": SPECIAL_TOKENS["unk_id"],
        "<s>": SPECIAL_TOKENS["bos_id"],
        "</s>": SPECIAL_TOKENS["eos_id"],
    }
    for piece, expected_id in expected_ids.items():
        actual_id = tokenizer.piece_to_id(piece)
        if actual_id != expected_id:
            raise RuntimeError(
                f"Unexpected ID for {piece}: expected {expected_id}, got {actual_id}."
            )

    print(f"BPE vocabulary: {tokenizer.vocab_size():,} pieces")
    print(f"Model saved:    {MODEL_PATH}")
    print(f"Vocabulary:     {VOCAB_PATH}")
    return tokenizer


def tokenize_file(
    input_path: Path,
    output_path: Path,
    tokenizer: spm.SentencePieceProcessor,
) -> tuple[int, int, int, int]:
    """Encode one cleaned file and verify each sentence can be reconstructed."""
    temporary_path = output_path.with_name(f"{output_path.name}.tmp")
    sentence_count = 0
    piece_count = 0
    unknown_count = 0
    round_trip_errors = 0

    with (
        input_path.open(mode="r", encoding="utf-8") as input_file,
        temporary_path.open(mode="w", encoding="utf-8", newline="\n") as output_file,
    ):
        for line in input_file:
            sentence = line.rstrip("\r\n")
            pieces = tokenizer.encode(sentence, out_type=str)

            sentence_count += 1
            piece_count += len(pieces)
            unknown_count += pieces.count("<unk>")
            if tokenizer.decode(pieces) != sentence:
                round_trip_errors += 1

            output_file.write(" ".join(pieces))
            output_file.write("\n")

    temporary_path.replace(output_path)
    return sentence_count, piece_count, unknown_count, round_trip_errors


def tokenize_all_splits(tokenizer: spm.SentencePieceProcessor) -> None:
    """Apply the shared BPE model to every cleaned project split."""
    print("\nApplying shared BPE tokenizer")
    print("-----------------------------")

    for split in SPLITS:
        encoded_lines: dict[str, int] = {}

        for language in LANGUAGES:
            input_path = CLEAN_DIR / f"{split}.{language}"
            output_path = TOKENIZER_DIR / f"{split}.bpe.{language}"
            if not input_path.is_file():
                raise FileNotFoundError(f"Cleaned input file not found: {input_path}")

            sentences, pieces, unknowns, round_trip_errors = tokenize_file(
                input_path, output_path, tokenizer
            )
            encoded_lines[language] = sentences
            average_pieces = pieces / sentences if sentences else 0

            print(
                f"{split}.{language}: {sentences:,} sentences, "
                f"{pieces:,} pieces, {average_pieces:.2f} pieces/sentence, "
                f"{unknowns:,} unknowns"
            )

            if round_trip_errors:
                raise RuntimeError(
                    f"{round_trip_errors:,} sentence(s) in {input_path.name} "
                    "failed the encode/decode round-trip check."
                )

        if encoded_lines["en"] != encoded_lines["de"]:
            raise RuntimeError(
                f"Tokenized split {split!r} is misaligned: "
                f"EN={encoded_lines['en']:,}, DE={encoded_lines['de']:,}."
            )

        print(f"  alignment verified: {encoded_lines['en']:,} pairs")


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Train the shared Multi30K English-German BPE tokenizer."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing shared_bpe.model and shared_bpe.vocab",
    )
    return parser.parse_args()


def main() -> None:
    """Train, verify, and apply the shared BPE tokenizer."""
    args = parse_args()
    train_shared_bpe(force=args.force)
    tokenizer = verify_tokenizer()
    tokenize_all_splits(tokenizer)


if __name__ == "__main__":
    main()
