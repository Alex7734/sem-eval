import csv
import re

def load_data(file_path):
    """
    Load data from a file for Subtask 1.

    Args:
        file_path (str): Path to the data file.

    Returns:
        data (list of tuples): Each tuple contains (article_id, entity_mention, start_offset, end_offset).
        main_roles (list of str): List of main roles (one per entity).
        fine_roles (list of lists): List of fine-grained roles (one or more per entity).
    """
    data = []
    main_roles = []
    fine_roles = []

    with open(file_path, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) < 5:
                continue       
            article_id, entity_mention, start_offset, end_offset, main_role, *fine_grained_roles = row
            data.append((article_id, entity_mention, start_offset, end_offset))
            main_roles.append(main_role)
            fine_roles.append(fine_grained_roles)

    return data, main_roles, fine_roles

def emphasize_keywords(context, keywords):
    """
    Highlight important keywords in the context using markers.
    """
    for keyword in keywords:
        context = re.sub(rf"\b({keyword})\b", r"[KEYWORD] \1 [/KEYWORD]", context, flags=re.IGNORECASE)
    return context

def extract_context(article_path, start_offset, end_offset, max_window=400, keywords=None):
    """
    Enhanced context extraction with title inclusion, dynamic windowing, and keyword emphasis.

    Args:
    - article_path (str): Path to the article text file.
    - start_offset (int): Start position of the entity mention.
    - end_offset (int): End position of the entity mention.
    - max_window (int, optional): The maximum length for the left and right context windows.
    - keywords (list, optional): List of important keywords to emphasize in the context.

    Returns:
    - str: The final enhanced context including title, entity mention, and dynamically extracted window.
    """
    with open(article_path, encoding="utf-8") as f:
        lines = f.readlines()

    title = lines[0].strip() if len(lines) > 0 else ""
    
    content = "".join(lines[2:])

    start_offset = int(start_offset)
    end_offset = int(end_offset)

    entity = content[start_offset:end_offset]
    entity_with_markers = f"[ENTITY_START] {entity} [ENTITY_END]"

    left_part = content[:start_offset]
    right_part = content[end_offset:]

    left_window_size = min(len(left_part), max_window)
    right_window_size = min(len(right_part), max_window)

    left_context = left_part[-left_window_size:].strip()
    right_context = right_part[:right_window_size].strip()

    context = f"{left_context} {entity_with_markers} {right_context}"

    final_context = f"{title} {context}"

    if keywords:
        final_context = emphasize_keywords(final_context, keywords)

    if len(final_context) > 2 * max_window:
        final_context = final_context[:max_window] + " ... " + final_context[-max_window:]

    return final_context.strip()

def save_incorrect_predictions(
    output_file, test_data, main_labels_decoded, fine_labels_decoded, main_predictions_decoded, fine_predictions
):
    """
    Saves incorrect predictions to a file.

    Parameters:
    - output_file (str): Filepath to save the incorrect predictions.
    - test_data (list): List of test data samples.
    - main_labels_decoded (list): Decoded main role true labels.
    - fine_labels_decoded (list): Decoded fine-grained true labels.
    - main_predictions_decoded (list): Decoded main role predicted labels.
    - fine_predictions (list): Predicted fine-grained roles.
    """
    incorrect_predictions = []
    for article_id, entity_mention, main_label, fine_label, main_pred, fine_pred in zip(
        [d[0] for d in test_data],
        [d[1] for d in test_data],
        main_labels_decoded,
        fine_labels_decoded,
        main_predictions_decoded,
        fine_predictions,
    ):
        if main_label != main_pred or set(fine_label) != set(fine_pred):
            incorrect_predictions.append(
                f"Article ID: {article_id}\n"
                f"Entity Mention: {entity_mention}\n"
                f"True Main Role: {main_label}\n"
                f"Predicted Main Role: {main_pred}\n"
                f"True Fine-Grained Roles: {fine_label}\n"
                f"Predicted Fine-Grained Roles: {fine_pred}\n\n"
            )

    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(incorrect_predictions)

    return len(incorrect_predictions)

def load_entity_mentions(file_path):
    """
    Load entity mentions from a file for Subtask 1.

    Args:
        file_path (str): Path to the entity mentions file.

    Returns:
        list of tuples: Each tuple contains (article_id, entity_mention, start_offset, end_offset).
    """
    data = []

    with open(file_path, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) < 4: 
                continue
            article_id, entity_mention, start_offset, end_offset = row[:4]
            data.append((article_id, entity_mention, int(start_offset), int(end_offset)))  # Convert offsets to integers

    return data

def save_submission_file(submission_data, output_file="submission.txt"):
    """
    Saves the submission data into a tab-separated text file in the required submission format.

    Args:
        submission_data (list): A list of lists where each inner list represents a row of the submission file.
            Each row should have the following columns:
            [article_id, entity_mention, start_offset, end_offset, main_role, fine_grained_roles]
        output_file (str): The name of the file where the submission will be saved.
    """
    with open(output_file, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
        for row in submission_data:
            writer.writerow(row)

    print(f"Submission file saved successfully to {output_file}")
