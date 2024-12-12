import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

from subtask1.metrics import evaluate_and_log_metrics
from subtask1.dataset import RoleDataset
from subtask1.data_parsing import extract_context, load_data, load_entity_mentions, save_submission_file
from subtask1.constants import FINE_GRAINED_ROLES, TAXONOMY, MAIN_ROLES
from subtask1.model import MultiTaskBertForRoles, tokenizer, device

train_file = r"D:\alex_mihoc\sem-eval\development\data\EN\subtask-1-annotations.txt"
article_folder = r"D:\alex_mihoc\sem-eval\development\data\EN\subtask-1-documents"
model_dir = r"D:\alex_mihoc\sem-eval\development\saved_model"
dev_entity_mentions_file = r"D:\alex_mihoc\sem-eval\development\data\EN\subtask-1-entity-mentions.txt"
dev_article_folder = r"D:\alex_mihoc\sem-eval\development\data\EN\subtask-1-dev"

EPOCHS = 3
TEST_SIZE = 0.2
BATCH_SIZE = 4
SCHEDULER_THRESHOLD = 1e-4
LEARN_RATE_FACTOR = 0.1
BERT_PARAMS_LEARN_RATE = 2e-5
FINE_CLASSIFIER_LEARN_RATE = 2e-5 

def enforce_taxonomy_constraints(main_prediction, fine_logits, history=None):
    allowed_roles = TAXONOMY.get(main_prediction, [])
    allowed_indices = [FINE_GRAINED_ROLES.index(role) for role in allowed_roles]

    if not allowed_indices:
        return []
    
    if history:
        penalties = np.array([history.get(FINE_GRAINED_ROLES[idx], 0) for idx in allowed_indices])
        fine_logits[allowed_indices] -= penalties
    
    max_allowed_index = allowed_indices[np.argmax(fine_logits[allowed_indices])]
    best_prediction = FINE_GRAINED_ROLES[max_allowed_index]

    return [best_prediction]

def predict_on_dev(model, dataloader, device, entity_mentions, output_file="submission.txt"):
    model.eval()
    submission_data = []
    processed_entities = 0 

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            _, main_logits, fine_logits = model(input_ids=input_ids, attention_mask=attention_mask)
            main_predictions = torch.argmax(main_logits, dim=1).cpu().numpy()
            fine_logits = fine_logits.cpu().numpy()

            batch_size = len(main_predictions) 

            batch_entities = entity_mentions[processed_entities:processed_entities + batch_size]
            
            for main_pred, fine_logit, entity in zip(main_predictions, fine_logits, batch_entities):
                if not isinstance(entity, (tuple, list)) or len(entity) < 4:
                    raise ValueError(f"Entity data is not in the expected format. Received: {entity}")
                
                article_id, entity_mention, start_offset, end_offset = entity
                main_role = MAIN_ROLES[main_pred]
                fine_predicted_roles = enforce_taxonomy_constraints(main_role, fine_logit)

                submission_data.append([
                    article_id,
                    entity_mention,
                    start_offset,
                    end_offset,
                    main_role,
                    "\t".join(fine_predicted_roles)
                ])

            processed_entities += batch_size

    save_submission_file(submission_data, output_file)
    print(f"Submission file saved to {output_file}")

def train_model_multitask(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        main_labels = batch["main_labels"].to(device)
        fine_labels = batch["fine_labels"].to(device)
        loss, _, _ = model(input_ids=input_ids, attention_mask=attention_mask,
                           main_labels=main_labels, fine_labels=fine_labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    scheduler.step(total_loss)
    return total_loss / len(dataloader)

def evaluate_model_with_constraints(model, dataloader, device):
    model.eval()
    all_main_predictions, all_fine_predictions = [], []
    all_main_labels, all_fine_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            main_labels = batch["main_labels"].to(device)
            fine_labels = batch["fine_labels"].to(device)
            _, main_logits, fine_logits = model(input_ids=input_ids, attention_mask=attention_mask)
            main_predictions = torch.argmax(main_logits, dim=1).cpu().numpy()
            fine_logits = fine_logits.cpu().numpy()
            for main_pred, fine_logit in zip(main_predictions, fine_logits):
                main_role = MAIN_ROLES[main_pred]
                fine_predicted_roles = enforce_taxonomy_constraints(main_role, fine_logit)
                all_fine_predictions.append(fine_predicted_roles)
            all_main_predictions.extend(main_predictions)
            all_main_labels.extend(main_labels.cpu().numpy())
            all_fine_labels.extend(fine_labels.cpu().numpy())
    return np.array(all_main_predictions), all_fine_predictions, np.array(all_main_labels), all_fine_labels

def dev_mode_main():
    print("Loading development data for entity mentions...")
    data = load_entity_mentions(dev_entity_mentions_file)
    contexts = [extract_context(os.path.join(dev_article_folder, d[0]), d[2], d[3]) for d in data]
    dataset = RoleDataset(contexts, None, None, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        
    print("\nInitializing Model...")
    model = MultiTaskBertForRoles(num_main_roles=len(MAIN_ROLES), num_fine_grained_roles=len(FINE_GRAINED_ROLES))
    model.to(device)
        
    model_path = os.path.join(model_dir, "model.pt")
    if os.path.exists(model_path):
        print(f"Loading saved model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Model not found. Please train the model before running in dev mode.")
        return
        
    predict_on_dev(model, dataloader, device, data, output_file="submission.txt")

def main(dev_mode=False):
    print("Loading training data...")

    if dev_mode:
      dev_mode_main()
      return
    else:
        data, main_roles, fine_roles = load_data(train_file)
    
    main_mlb = MultiLabelBinarizer(classes=MAIN_ROLES)
    main_roles = main_mlb.fit_transform([[role] for role in main_roles]).squeeze()
    fine_mlb = MultiLabelBinarizer(classes=FINE_GRAINED_ROLES)
    fine_roles = fine_mlb.fit_transform(fine_roles)
    train_data, test_data, train_main, test_main, train_fine, test_fine = train_test_split(
        data, main_roles, fine_roles, test_size=TEST_SIZE, random_state=42
    )
    train_contexts = [extract_context(os.path.join(article_folder, d[0]), d[2], d[3]) for d in train_data]
    test_contexts = [extract_context(os.path.join(article_folder, d[0]), d[2], d[3]) for d in test_data]
    train_dataset = RoleDataset(train_contexts, train_main, train_fine, tokenizer)
    test_dataset = RoleDataset(test_contexts, test_main, test_fine, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("\nInitializing Model...")
    model = MultiTaskBertForRoles(num_main_roles=len(MAIN_ROLES), num_fine_grained_roles=len(FINE_GRAINED_ROLES))
    model.to(device)

    model_path = os.path.join(model_dir, "model.pt")
    if os.path.exists(model_path):
        print(f"Loading saved model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Training a new model...")
        optimizer = torch.optim.AdamW([
            {"params": model.bert.parameters(), "lr": BERT_PARAMS_LEARN_RATE},
            {"params": model.fine_grained_classifier.parameters(), "lr": FINE_CLASSIFIER_LEARN_RATE, "weight_decay": 1e-4}
        ])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=1, factor=LEARN_RATE_FACTOR, threshold=SCHEDULER_THRESHOLD)
        for epoch in range(EPOCHS):
            train_loss = train_model_multitask(model, train_loader, optimizer, scheduler, device)
            print(f"Epoch {epoch + 1}: Training Loss = {train_loss:.4f}")
        os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
    print("\nEvaluating Model...")
    main_predictions, fine_predictions, main_labels, fine_labels = evaluate_model_with_constraints(
        model, test_loader, device
    )

    evaluate_and_log_metrics(
        main_predictions=main_predictions,
        fine_predictions=fine_predictions,
        main_labels=main_labels,
        fine_labels=fine_labels,
        test_data=test_data,
        output_file="incorrect_predictions.txt"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the multi-task BERT model")
    parser.add_argument("--dev", action="store_true", help="Run the model in development mode")
    args = parser.parse_args()
    main(dev_mode=args.dev)
