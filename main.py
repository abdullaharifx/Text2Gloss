import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import Adafactor

from models.mbart_model import load_model_and_tokenizer
from datasets.text2gloss_dataset import Text2GlossDataset
from training.train import train
from evaluation.evaluate import evaluate_model
from utils import config

# Load data
df = pd.read_csv("data/gloss.csv")
df['gloss'] = df['GLOSSES'].astype(str).str.strip()
df['text'] = df['SENTENCE'].astype(str).str.strip()
df = df.drop_duplicates()

# Split
train_texts, val_texts, train_glosses, val_glosses = train_test_split(
    df['text'].tolist(),
    df['gloss'].tolist(),
    test_size=0.1,
    random_state=42
)

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer(lang=config.src_lang)

# Datasets and loaders
train_dataset = Text2GlossDataset(train_texts, train_glosses, tokenizer, max_len=config.max_len)
val_dataset = Text2GlossDataset(val_texts, val_glosses, tokenizer, max_len=config.max_len)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

# Training
optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train(model, train_loader, val_loader, optimizer, device, epochs=config.epochs)

# Evaluation
bleu_score, rouge_score = evaluate_model(model, tokenizer, val_texts, val_glosses, device)
print("BLEU-4:", bleu_score["bleu"])
print("ROUGE:", rouge_score)
