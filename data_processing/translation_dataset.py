"""PyTorch dataset and batching utilities for Multi30K translation."""

import argparse
from collections.abc import Sequence
from functools import partial
from itertools import zip_longest
from pathlib import Path

import sentencepiece as spm
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


PROJECT_DIR = Path(__file__).resolve().parent.parent
BPE_DIR = PROJECT_DIR / "data" / "processed" / "multi30k_en_de_bpe8000"
TOKENIZER_PATH = BPE_DIR / "shared_bpe.model"
SPLITS = ("train", "val", "test_2016_flickr")


class TranslationDataset(Dataset[tuple[list[int], list[int]]]):
    """Aligned English-German BPE sequences represented by token IDs."""

    def __init__(self, split: str) -> None:
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}.")
        if not TOKENIZER_PATH.is_file():
            raise FileNotFoundError(
                f"Tokenizer model not found: {TOKENIZER_PATH}\n"
                "Run train_bpe_tokenizer.py first."
            )

        english_path = BPE_DIR / f"{split}.bpe.en"
        german_path = BPE_DIR / f"{split}.bpe.de"
        for path in (english_path, german_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Tokenized split file not found: {path}\n"
                    "Run train_bpe_tokenizer.py first."
                )

        tokenizer = spm.SentencePieceProcessor(model_file=str(TOKENIZER_PATH))
        self.pad_id = tokenizer.pad_id()
        self.unk_id = tokenizer.unk_id()
        self.bos_id = tokenizer.bos_id()
        self.eos_id = tokenizer.eos_id()
        self.vocab_size = tokenizer.vocab_size()
        self.examples: list[tuple[list[int], list[int]]] = []

        with (
            english_path.open(mode="r", encoding="utf-8") as english_file,
            german_path.open(mode="r", encoding="utf-8") as german_file,
        ):
            aligned_lines = zip_longest(english_file, german_file)
            for line_number, (english_line, german_line) in enumerate(
                aligned_lines, start=1
            ):
                if english_line is None or german_line is None:
                    raise ValueError(
                        f"Tokenized split {split!r} is misaligned at line "
                        f"{line_number}."
                    )

                english_ids = self._pieces_to_ids(
                    english_line.rstrip("\r\n").split(), tokenizer, split, line_number
                )
                german_ids = self._pieces_to_ids(
                    german_line.rstrip("\r\n").split(), tokenizer, split, line_number
                )
                self.examples.append((english_ids, german_ids))

    def _pieces_to_ids(
        self,
        pieces: list[str],
        tokenizer: spm.SentencePieceProcessor,
        split: str,
        line_number: int,
    ) -> list[int]:
        """Map BPE pieces to IDs and reject unexpected unknown pieces."""
        if not pieces:
            raise ValueError(f"Empty tokenized sentence in {split!r}, line {line_number}.")

        token_ids = [tokenizer.piece_to_id(piece) for piece in pieces]
        if self.unk_id in token_ids:
            raise ValueError(
                f"Unknown BPE piece in {split!r}, line {line_number}."
            )
        return token_ids

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        return self.examples[index]


def collate_translation_batch(
    examples: Sequence[tuple[list[int], list[int]]],
    *,
    pad_id: int,
    bos_id: int,
    eos_id: int,
) -> dict[str, torch.Tensor]:
    """Add special tokens, dynamically pad, and construct attention masks."""
    source_sequences = [
        torch.tensor([bos_id, *english_ids, eos_id], dtype=torch.long)
        for english_ids, _ in examples
    ]
    decoder_input_sequences = [
        torch.tensor([bos_id, *german_ids], dtype=torch.long)
        for _, german_ids in examples
    ]
    label_sequences = [
        torch.tensor([*german_ids, eos_id], dtype=torch.long)
        for _, german_ids in examples
    ]

    source_ids = pad_sequence(
        source_sequences, batch_first=True, padding_value=pad_id
    )
    decoder_input_ids = pad_sequence(
        decoder_input_sequences, batch_first=True, padding_value=pad_id
    )
    labels = pad_sequence(label_sequences, batch_first=True, padding_value=pad_id)

    source_mask = source_ids.ne(pad_id).unsqueeze(1).unsqueeze(2)
    target_padding_mask = decoder_input_ids.ne(pad_id).unsqueeze(1).unsqueeze(2)
    target_length = decoder_input_ids.size(1)
    causal_mask = torch.tril(
        torch.ones((target_length, target_length), dtype=torch.bool)
    ).unsqueeze(0).unsqueeze(0)
    target_mask = target_padding_mask & causal_mask

    return {
        "source_ids": source_ids,
        "decoder_input_ids": decoder_input_ids,
        "labels": labels,
        "source_mask": source_mask,
        "target_mask": target_mask,
        "target_token_count": labels.ne(pad_id).sum(),
    }


def create_translation_dataloader(
    split: str,
    batch_size: int,
    *,
    shuffle: bool | None = None,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[TranslationDataset, DataLoader]:
    """Create a reproducible data loader for one project split."""
    dataset = TranslationDataset(split)
    if shuffle is None:
        shuffle = split == "train"

    generator = torch.Generator()
    generator.manual_seed(seed)
    collate_fn = partial(
        collate_translation_batch,
        pad_id=dataset.pad_id,
        bos_id=dataset.bos_id,
        eos_id=dataset.eos_id,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        generator=generator,
    )
    return dataset, dataloader


def parse_args() -> argparse.Namespace:
    """Parse smoke-test options for direct execution."""
    parser = argparse.ArgumentParser(
        description="Load one Multi30K batch and display its tensor shapes."
    )
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    """Run a one-batch dataset and mask smoke test."""
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive.")

    dataset, dataloader = create_translation_dataloader(
        args.split, args.batch_size, shuffle=False
    )
    batch = next(iter(dataloader))

    print(f"Split:              {args.split}")
    print(f"Examples:           {len(dataset):,}")
    print(f"Vocabulary size:    {dataset.vocab_size:,}")
    print(f"Source IDs:         {tuple(batch['source_ids'].shape)}")
    print(f"Decoder input IDs:  {tuple(batch['decoder_input_ids'].shape)}")
    print(f"Labels:             {tuple(batch['labels'].shape)}")
    print(f"Source mask:        {tuple(batch['source_mask'].shape)}")
    print(f"Target mask:        {tuple(batch['target_mask'].shape)}")
    print(f"Non-pad targets:    {batch['target_token_count'].item():,}")


if __name__ == "__main__":
    main()
