from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, hamming_loss
import os
from data_parsing_subtask1 import extract_context, load_data
from dataset import RoleDataset

FINE_GRAINED_ROLES = [
    'Guardian', 'Martyr', 'Peacemaker', 'Rebel', 'Underdog', 'Virtuous', 'Instigator',
    'Conspirator', 'Tyrant', 'Foreign Adversary', 'Traitor', 'Spy', 'Saboteur', 'Corrupt',
    'Incompetent', 'Terrorist', 'Deceiver', 'Bigot', 'Forgotten', 'Exploited', 'Victim', 'Scapegoat'
]

train_file = r"D:\alex_mihoc\sem-eval\development\data\EN\subtask-1-annotations.txt"
article_folder = r"D:\alex_mihoc\sem-eval\development\data\EN\subtask-1-documents"
model_dir = r"D:\alex_mihoc\sem-eval\development\saved_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EnhancedBertForMultiLabelClassification(nn.Module):
    def __init__(self, num_labels):
        super(EnhancedBertForMultiLabelClassification, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(p=0.3)
        self.output_layer = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        logits = self.output_layer(pooled_output)
        loss = None
        if labels is not None:
            loss_fn = FocalLoss()
            loss = loss_fn(logits, labels)
        return loss, torch.sigmoid(logits)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probas = torch.sigmoid(logits)
        pt = targets * probas + (1 - targets) * (1 - probas)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def train_model(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    scheduler.step(total_loss)
    return total_loss / len(dataloader)


def evaluate_model(model, dataloader, device, threshold=0.5):
    model.eval()
    all_predictions, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            _, predictions = model(input_ids=input_ids, attention_mask=attention_mask)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    predictions = np.array(all_predictions)
    labels = np.array(all_labels)

    predicted_labels = (predictions >= threshold).astype(int)

    for i, idx in enumerate(np.argmax(predictions, axis=1)):
        predicted_labels[i, idx] = 1

    return predicted_labels, labels

def main():
    print("Loading training data...")
    data, labels = load_data(train_file)

    mlb = MultiLabelBinarizer(classes=FINE_GRAINED_ROLES)
    labels = mlb.fit_transform(labels)

    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
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

    print("\nInitializing Model...")
    model = EnhancedBertForMultiLabelClassification(num_labels=len(FINE_GRAINED_ROLES))
    model.to(device)

    model_path = os.path.join(model_dir, "model.pt")
    if os.path.exists(model_path):
        print(f"Loading saved model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Training a new model...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.1)

        print("Training Model...")
        for epoch in range(100):
            train_loss = train_model(model, train_loader, optimizer, scheduler, device)
            print(f"Epoch {epoch + 1}: Training Loss = {train_loss:.4f}")

        os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    print("\nEvaluating Model...")
    predictions, true_labels = evaluate_model(model, test_loader, device, threshold=0.5)

    true_labels_decoded = mlb.inverse_transform(true_labels)
    predicted_labels_decoded = mlb.inverse_transform(predictions)

    print("\nCalculating Metrics...")
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="weighted")
    hamming = hamming_loss(true_labels, predictions)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Weighted F1 Score: {f1:.4f}")
    print(f"Hamming Loss: {hamming:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=FINE_GRAINED_ROLES))

    print("\nSaving Incorrect Predictions...")
    incorrect_predictions = []
    for article_id, entity_mention, true_label, predicted_label in zip(
        [d[0] for d in test_data],
        [d[1] for d in test_data],
        true_labels_decoded,
        predicted_labels_decoded,
    ):
        if set(true_label) != set(predicted_label):
            incorrect_predictions.append(
                f"Article ID: {article_id}\n"
                f"Entity Mention: {entity_mention}\n"
                f"True Label: {true_label}\n"
                f"Predicted Label: {predicted_label}\n\n"
            )

    with open("incorrect_predictions.txt", "w", encoding="utf-8") as f:
        f.writelines(incorrect_predictions)

    print(f"Number of Incorrect Predictions: {len(incorrect_predictions)}")


if __name__ == "__main__":
    main()
