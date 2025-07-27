# Text2Gloss: Gloss Translation using mBART

This project implements a gloss translation system using the mBART model, inspired by the TwoStream-SLR framework ([arXiv:2203.04287](https://arxiv.org/abs/2203.04287)). It is designed to translate spoken or written text into corresponding ASL gloss sequences, which are used in sign language modeling and generation systems.

---

## 📂 Project Structure

```
text2gloss/
├── data/                 # Input CSV file (gloss.csv)
├── models/               # mBART model loader
├── datasets/             # Custom dataset class
├── training/             # Training loop
├── evaluation/           # Evaluation using BLEU and ROUGE
├── utils/                # Configs and helpers
├── checkpoints/          # Saved models (.pkl)
├── main.py               # Entry point for training and evaluation
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## Installation

Make sure you have Python 3.8+. Then run:

```bash
git clone https://github.com/abdullaharifx/text2gloss.git
cd text2gloss

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Prepare your dataset
Place a `gloss.csv` file inside the `data/` directory with the following columns:

| SENTENCE | GLOSSES |
|----------|---------|

### 2. Run training and evaluation
```bash
python main.py
```

This will:
- Load and split the dataset
- Fine-tune mBART for gloss translation
- Evaluate using BLEU-4 and ROUGE
- Save checkpoints to `checkpoints/`

---

## 📊 Results

The model uses Adafactor optimizer with mBART-large-50, trained for 5 epochs. Evaluation scores (BLEU-4 and ROUGE-L) are printed after training.

---

## 📝 Reference

If you use this work, please consider citing the following paper:

```bibtex
@article{wei2024spoken2sign,
  title={Spoken2Sign: A Multimodal Translation System for Sign Language Production},
  author={Wei, Fangyun and Liu, Ze and Wu, Binyang and Li, Yuxin and Liu, Yanghao and Qiao, Siyuan and Wang, Xiaolong and Dai, Jifeng and Cao, Yue},
  journal={arXiv preprint arXiv:2401.04730},
  year={2024}
}

```

[Paper Link]([https://arxiv.org/abs/2203.04287](https://arxiv.org/abs/2401.04730))

---

## 🙏 Acknowledgements

- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [Evaluate Library](https://huggingface.co/docs/evaluate/) (BLEU, ROUGE)
- Torch & PyTorch Ecosystem

---

## 💬 Contact

Feel free to open an issue or contact the maintainer: [(https://www.linkedin.com/in/abdullaharifx/)]

---

© 2025 — Text2Gloss Project
