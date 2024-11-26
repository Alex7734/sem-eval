from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
import os
import csv

FINE_GRAINED_ROLES = [
    'Guardian', 'Martyr', 'Peacemaker', 'Rebel', 'Underdog', 'Virtuous', 'Instigator',
    'Conspirator', 'Tyrant', 'Foreign Adversary', 'Traitor', 'Spy', 'Saboteur', 'Corrupt',
    'Incompetent', 'Terrorist', 'Deceiver', 'Bigot', 'Forgotten', 'Exploited', 'Victim', 'Scapegoat'
]

train_file = "../../data/EN/subtask-1-annotations.txt"
article_folder = "../../data/EN/subtask-1-documents"
model_dir = "./saved_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EnhancedBertForMultiLabelClassification(nn.Module):
    def __init__(self, num_labels):
        super(EnhancedBertForMultiLabelClassification, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(self.dropout(outputs.pooler_output))
        loss = None
        if labels is not None:
            loss = nn.BCEWithLogitsLoss()(logits, labels)
        return loss, torch.sigmoid(logits)


def load_data(file_path):
    data = []
    labels = []
    with open(file_path, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) < 5:
                continue
            article_id, entity_mention, start_offset, end_offset, main_role, *fine_grained_roles = row
            data.append((article_id, entity_mention, start_offset, end_offset))
            labels.append(fine_grained_roles)
    return data, labels


def extract_context(article_path, start_offset, end_offset):
    with open(article_path, encoding="utf-8") as f:
        lines = f.readlines()
        content = " ".join([line.strip() for line in lines if line.strip()])
    return content[int(start_offset):int(end_offset)]


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


def train_model(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    for batch_id, batch in enumerate(dataloader):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        _, logits = model(input_ids=input_ids, attention_mask=attention_mask)

        loss = criterion(logits, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Batch {batch_id + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader)


def evaluate_model(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            labels = batch["labels"].to(device)
            _, predictions = model(**inputs)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_predictions), np.array(all_labels)


def main():
    print("Loading training data...")
    data, labels = load_data(train_file)

    mlb = MultiLabelBinarizer(classes=FINE_GRAINED_ROLES)
    labels = mlb.fit_transform(labels)

    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_contexts = [
        extract_context(os.path.join(article_folder, article_id), start, end)
        for article_id, _, start, end in train_data
    ]
    test_contexts = [
        extract_context(os.path.join(article_folder, article_id), start, end)
        for article_id, _, start, end in test_data
    ]

    train_dataset = RoleDataset(train_contexts, train_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    test_dataset = RoleDataset(test_contexts, test_labels, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    if os.path.exists(os.path.join(model_dir, "model.pt")):
        print(f"Loading saved model from {model_dir}")
        model = EnhancedBertForMultiLabelClassification(num_labels=len(FINE_GRAINED_ROLES))
        model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt")))
        model.to(device)
    else:
        print("Training a new model...")
        class_counts = np.sum(train_labels, axis=0)
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor)

        model = EnhancedBertForMultiLabelClassification(num_labels=len(FINE_GRAINED_ROLES))
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        epochs = 10

        for epoch in range(epochs):
            train_loss = train_model(model, train_loader, optimizer, criterion, device, epoch)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}")

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))
        print(f"Model saved to {model_dir}")

    predictions, true_labels = evaluate_model(model, test_loader, device)

    most_confident_indices = np.argmax(predictions, axis=1)
    most_confident_labels = [FINE_GRAINED_ROLES[idx] for idx in most_confident_indices]

    true_labels_single = [
        label[0] if len(label) > 0 else "Unknown" for label in mlb.inverse_transform(true_labels)
    ]

    accuracy = accuracy_score(true_labels_single, most_confident_labels)
    f1 = f1_score(true_labels_single, most_confident_labels, average="weighted")
    hamming = hamming_loss(true_labels_single, most_confident_labels)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"F1 Score: {f1:.4f}")
    print(f"Hamming Loss: {hamming:.4f}")

    incorrect_predictions = []
    for article_id, entity_mention, true_label, predicted_label in zip(
        [d[0] for d in test_data],
        [d[1] for d in test_data],
        true_labels_single,
        most_confident_labels,
    ):
        if true_label != predicted_label:
            incorrect_predictions.append(
                f"Article ID: {article_id}\n"
                f"Entity Mention: {entity_mention}\n"
                f"True Label: {true_label}\n"
                f"Predicted Label: {predicted_label}\n\n"
            )

    with open("incorrect_predictions.txt", "w", encoding="utf-8") as f:
        f.writelines(incorrect_predictions)

    print("Incorrect predictions saved to 'incorrect_predictions.txt'.")


if __name__ == "__main__":
    main()

