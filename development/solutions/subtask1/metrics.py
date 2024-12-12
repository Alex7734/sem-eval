import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score, accuracy_score

from subtask1.data_parsing import save_incorrect_predictions
from subtask1.constants import FINE_GRAINED_ROLES, TAXONOMY, MAIN_ROLES

def plot_confusion_matrix_main_label(true_labels, predicted_labels, class_names, normalize=None, title="Confusion Matrix"):
    """
    Plots a confusion matrix.

    Parameters:
    - true_labels: Ground truth (true labels).
    - predicted_labels: Predicted labels.
    - class_names: List of class names corresponding to the labels.
    - normalize: None (default), 'true', or 'pred' for normalization.
    - title: Title of the plot.
    """
    cm = confusion_matrix(true_labels, predicted_labels, normalize=normalize)
    
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    disp.plot(cmap=plt.cm.Blues, values_format=".2f" if normalize else "d", ax=plt.gca())
    plt.title(title)
    plt.show()


def plot_fine_grained_confusion_matrices(fine_labels, fine_predictions_binary, class_names, taxonomy=TAXONOMY):
    """
    Plots confusion matrices for fine-grained roles grouped by taxonomy.

    Parameters:
    - fine_labels: True binary labels for fine-grained roles (array-like).
    - fine_predictions_binary: Predicted binary labels for fine-grained roles (array-like).
    - class_names: List of fine-grained role names.
    - taxonomy: Dictionary mapping main roles to their corresponding fine-grained roles.
    """
    fine_labels = np.array(fine_labels) if isinstance(fine_labels, list) else fine_labels
    fine_predictions_binary = (
        np.array(fine_predictions_binary) if isinstance(fine_predictions_binary, list) else fine_predictions_binary
    )

    assert fine_labels.shape == fine_predictions_binary.shape, (
        f"Shape mismatch: fine_labels {fine_labels.shape} and fine_predictions_binary {fine_predictions_binary.shape}"
    )

    for main_role, fine_roles in taxonomy.items():
        indices = [class_names.index(role) for role in fine_roles]
        true_labels_group = fine_labels[:, indices]
        pred_labels_group = fine_predictions_binary[:, indices]

        true_flat = np.argmax(true_labels_group, axis=1)
        pred_flat = np.argmax(pred_labels_group, axis=1)

        cm = confusion_matrix(true_flat, pred_flat, labels=range(len(fine_roles)))

        plt.figure(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=fine_roles)
        disp.plot(cmap=plt.cm.Blues, values_format="d", ax=plt.gca())
        plt.title(f"Confusion Matrix for {main_role}")
        plt.show()


def evaluate_and_log_metrics(
    main_predictions, fine_predictions, main_labels, fine_labels, test_data, 
    output_file="incorrect_predictions.txt"
):
    print("\nCalculating Metrics...")

    main_labels_indices = np.argmax(main_labels, axis=1)
    main_accuracy = accuracy_score(main_labels_indices, main_predictions)
    print(f"Main Role Accuracy: {main_accuracy * 100:.2f}%")
    print("\nMain Role Classification Report:")
    print(classification_report(main_labels_indices, main_predictions, target_names=MAIN_ROLES, zero_division=0))

    fine_labels_decoded = [
        [FINE_GRAINED_ROLES[idx] for idx, value in enumerate(row) if value > 0]
        for row in fine_labels
    ]

    fine_predictions_binary = np.zeros_like(fine_labels)
    for i, predicted_roles in enumerate(fine_predictions):
        for role in predicted_roles:
            fine_predictions_binary[i, FINE_GRAINED_ROLES.index(role)] = 1

    fine_accuracy = accuracy_score(fine_labels, fine_predictions_binary)
    fine_f1 = f1_score(fine_labels, fine_predictions_binary, average="weighted")
    print(f"Accuracy (Fine Grained): {fine_accuracy * 100:.2f}%")
    print(f"Weighted F1 Score (Fine-Grained): {fine_f1:.4f}")
    print("\nFine-Grained Role Classification Report:")
    print(
        classification_report(
            fine_labels,
            fine_predictions_binary,
            target_names=FINE_GRAINED_ROLES,
            zero_division=0,
        )
    )

    print("\nPlotting Confusion Matrix...")
    plot_confusion_matrix_main_label(
        true_labels=main_labels_indices,
        predicted_labels=main_predictions,
        class_names=MAIN_ROLES,
        normalize=None, 
        title="Main Role Confusion Matrix"
    )

    plot_fine_grained_confusion_matrices(
        fine_labels=fine_labels,
        fine_predictions_binary=fine_predictions_binary,
        class_names=FINE_GRAINED_ROLES
    )

    print("\nSaving Incorrect Predictions...")
    num_incorrect = save_incorrect_predictions(
        output_file=output_file,
        test_data=test_data,
        main_labels_decoded=main_labels_indices,
        fine_labels_decoded=fine_labels_decoded,
        main_predictions_decoded=main_predictions,
        fine_predictions=fine_predictions,
    )

    print(f"Number of Incorrect Predictions: {num_incorrect}")
