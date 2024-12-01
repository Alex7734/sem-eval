from torch.utils.data import Dataset
import torch

class RoleDataset(Dataset):
    def __init__(self, contexts, labels, tokenizer, max_len=512):
        self.contexts = contexts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        encodings = self.tokenizer(
            self.contexts[idx],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float),
        }
