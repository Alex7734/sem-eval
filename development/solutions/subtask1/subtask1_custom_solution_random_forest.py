import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier 
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, hamming_loss, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np 

MAIN_ROLES = ['Protagonist', 'Antagonist', 'Innocent']
FINE_GRAINED_ROLES = [
    'Guardian', 'Martyr', 'Peacemaker', 'Rebel', 'Underdog', 'Virtuous', 'Instigator',
    'Conspirator', 'Tyrant', 'Foreign Adversary', 'Traitor', 'Spy', 'Saboteur', 'Corrupt',
    'Incompetent', 'Terrorist', 'Deceiver', 'Bigot', 'Forgotten', 'Exploited', 'Victim', 'Scapegoat'
]

train_file = "../../data/EN/subtask-1-annotations.txt"
article_folder = "../../data/EN/subtask-1-documents"

def load_data(file_path):
    """Loads the training dataset."""
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

def load_article_content(article_path):
    """Loads the content of an article given its path."""
    if not os.path.exists(article_path):
        print(f"File not found: {article_path}")
        return "" 
    with open(article_path, encoding="utf-8") as f:
        lines = f.readlines()
        return " ".join([line.strip() for line in lines if line.strip()])

def extract_context(article_path, start_offset, end_offset):
    """Extracts a snippet from the article based on the entity's start and end offsets."""
    article_content = load_article_content(article_path)
    context = article_content[int(start_offset):int(end_offset)]
    return context

def main():
    print("Loading training data...")
    data, labels = load_data(train_file)

    mlb = MultiLabelBinarizer(classes=FINE_GRAINED_ROLES)
    labels = mlb.fit_transform(labels)

    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

    train_contexts = [extract_context(os.path.join(article_folder, article_id), start, end) for article_id, _, start, end in train_data]
    test_contexts = [extract_context(os.path.join(article_folder, article_id), start, end) for article_id, _, start, end in test_data]

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_contexts)
    X_test = vectorizer.transform(test_contexts)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')  
    model = MultiOutputClassifier(rf_model)
    model.fit(X_train, train_labels)

    predictions = model.predict(X_test)

    best_predictions = []
    for pred in predictions:
        best_label_idx = np.argmax(pred)
        best_predictions.append([mlb.classes_[best_label_idx]])

    best_predictions = np.array(best_predictions)

    best_test_labels = [mlb.classes_[np.argmax(label)] for label in test_labels]

    accuracy = accuracy_score(best_test_labels, best_predictions.flatten()) 
    hamming = hamming_loss(best_test_labels, best_predictions.flatten())
    f1 = f1_score(best_test_labels, best_predictions.flatten(), average='micro')

    print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
    print(f"Hamming Loss: {hamming:.4f}")
    print(f"F1 Score (micro): {f1:.4f}")

    incorrect_predictions = []
    for (true_labels, pred_labels, test_sample) in zip(best_test_labels, best_predictions.flatten(), test_data):
        if true_labels != pred_labels:
            article_id, entity_mention, _, _ = test_sample
            incorrect_predictions.append((
                article_id, 
                entity_mention, 
                true_labels, 
                pred_labels
            ))

    with open("incorrect_predictions.txt", "w", encoding="utf-8") as f:
        for article_id, entity_mention, true_labels, pred_labels in incorrect_predictions:
            f.write(f"Article ID: {article_id}\n")
            f.write(f"Entity mention: {entity_mention}\n")
            f.write(f"True Labels: {true_labels}\n")
            f.write(f"Predicted Labels: {pred_labels}\n\n")

if __name__ == "__main__":
    main()
