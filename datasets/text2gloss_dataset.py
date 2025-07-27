from torch.utils.data import Dataset

class Text2GlossDataset(Dataset):
    def __init__(self, texts, glosses, tokenizer, max_len=128):
        self.texts = texts
        self.glosses = glosses
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        gloss = str(self.glosses[idx])

        inputs = self.tokenizer(text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')

        with self.tokenizer.as_target_tokenizer():
            targets = self.tokenizer(gloss, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }
