import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

MAIN_ROLES = ['Protagonist', 'Antagonist', 'Innocent']

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
            labels.append(main_role)
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

def extract_incorect_predictions(test_data, test_labels, predictions):
    """Extracts the incorrect predictions from the test set."""
    incorrect_predictions = []
    for i, (true_label, pred_label, test_sample) in enumerate(zip(test_labels, predictions, test_data)):
        if true_label != pred_label:
            article_id, entity_mention, _, _ = test_sample
            incorrect_predictions.append((article_id, entity_mention, true_label, pred_label))
    return incorrect_predictions

def main():
    print("Loading training data...")
    data, labels = load_data(train_file)

    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

    train_contexts = [extract_context(os.path.join(article_folder, article_id), start, end) for article_id, _, start, end in train_data]
    test_contexts = [extract_context(os.path.join(article_folder, article_id), start, end) for article_id, _, start, end in test_data]

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_contexts)
    X_test = vectorizer.transform(test_contexts)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, train_labels)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(test_labels, predictions)
    print(f"Accuracy on the test set: {accuracy * 100:.2f}%")

    incorrect_predictions = extract_incorect_predictions(test_data, test_labels, predictions)
    with open("incorrect_predictions.txt", "w", encoding="utf-8") as f:
        for article_id, entity_mention, true_label, pred_label in incorrect_predictions:
            f.write(f"Article ID: {article_id}\n")
            f.write(f"Entity mention: {entity_mention}\n")
            f.write(f"True label: {true_label}\n")
            f.write(f"Predicted label: {pred_label}\n\n")

if __name__ == "__main__":
    main()
