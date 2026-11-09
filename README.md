# Transformer from Scratch in PyTorch

A from-scratch PyTorch re-implementation of the Transformer from
["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017),
trained for **English → German translation** on the Multi30K dataset.

**Result: 31.8 BLEU on the Multi30K 2016 test set.**

## Authors
- Taleed Hamadneh
- Qasim Batrawi
- Abedalrheem Fialah

## What's inside
The notebook explains each part of the Transformer in theory and then builds it in code,
with a small test or figure after every part:

- Token embedding and sinusoidal positional encoding
- Scaled dot-product attention and multi-head attention
- Add & Norm (residual connections and layer normalization)
- Position-wise feed-forward network
- Encoder and decoder layers, stacked into the full encoder–decoder model
- Output generator (linear layer + softmax)
- A full data pipeline: cleaning, shared BPE tokenization, padding and attention masks
- Training, validation, BLEU evaluation and analysis of example translations

## Training setup
| Setting | Value |
|---|---|
| Dataset | Multi30K (English–German) |
| Tokenizer | Shared BPE, 8,000 tokens (SentencePiece) |
| Model size | d_model = 128, 4 heads, 2 encoder + 2 decoder layers |
| Dropout | 0.1 |
| Batch size | 32 |
| Learning rate | 3e-4 |
| Epochs | 5 |

This is a small model trained on a small dataset, so the BLEU score can't be compared
directly with the original paper, which used much larger models, data and compute.

## Project structure
```
├── transformer_from_scratch.ipynb   # main notebook: theory, code and results
├── translation_predictions.csv      # saved test-set translations used for BLEU
├── images/                          # figures shown in the notebook
├── data/processed/
│   ├── multi30k_en_de_clean/        # cleaned English–German text
│   └── multi30k_en_de_bpe8000/      # BPE model, vocabulary and tokenized data
└── data_processing/
    ├── inspect_raw_data.sh          # statistics on the raw dataset
    ├── preprocess_data.py           # cleaning and filtering
    ├── train_bpe_tokenizer.py       # trains the shared BPE tokenizer
    ├── eda.py                       # exploratory data analysis
    └── translation_dataset.py       # PyTorch dataset and batching
```

## How to run
1. Install the dependencies (Python 3.12):
   ```bash
   pip install torch==2.5.1 numpy==1.26.4 matplotlib==3.10.5 scipy==1.15.3 pandas==2.2.3 sentencepiece==0.2.0 nltk==3.9.1 jupyter
   ```
2. Open the notebook and run it from top to bottom:
   ```bash
   jupyter notebook transformer_from_scratch.ipynb
   ```

The processed data is already included, so you don't need to download anything to run
the notebook. The Results section reads the saved predictions in
`translation_predictions.csv`, so you don't need to re-train to see the BLEU score.

### Rebuilding the data from scratch (optional)
1. Clone the [Multi30K dataset](https://github.com/multi30k/dataset) into `data/multi30k-dataset/`.
2. Run the scripts in this order:
   ```bash
   python data_processing/preprocess_data.py
   python data_processing/train_bpe_tokenizer.py
   python data_processing/eda.py
   ```

## References
- Vaswani et al., "Attention Is All You Need", NeurIPS 2017. https://arxiv.org/abs/1706.03762
- Elliott et al., "Multi30K: Multilingual English-German Image Descriptions", 2016.
