from torch.utils.data import Dataset
import torch

class RoleDataset(Dataset):
    def __init__(self, contexts, main_labels=None, fine_labels=None, tokenizer=None):
        """
        Args:
            contexts (list): List of contexts (entity mention surroundings) for each instance.
            main_labels (list, optional): List of main role labels.
            fine_labels (list, optional): List of fine-grained role labels.
            tokenizer (transformers.Tokenizer, optional): Tokenizer used to encode the context.
        """
        self.contexts = contexts
        self.main_labels = main_labels
        self.fine_labels = fine_labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.contexts[idx],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}

        if self.main_labels is not None:
            item["main_labels"] = torch.tensor(self.main_labels[idx], dtype=torch.long)
        if self.fine_labels is not None:
            item["fine_labels"] = torch.tensor(self.fine_labels[idx], dtype=torch.float)
            
        return item
