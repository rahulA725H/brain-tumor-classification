# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import os

import torch
import pandas as pd

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)

from dataset import brain_mri_dataset
from model import MyModel
from args import get_args

# ============================================================
# 2. EVALUATE ONE MODEL
# ============================================================

def evaluate_model(model, test_loader, device):

    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_names = []

    with torch.no_grad():

        for batch in test_loader:

            images = batch["mri"].to(device)
            labels = batch["target"].to(device)

            outputs = model(images)

            predictions = torch.argmax(outputs, dim=1)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )[:, 1]

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_names.extend(batch["name"])

    return (
        all_names,
        all_labels,
        all_predictions,
        all_probabilities
    )
    
# ============================================================
# 3. CALCULATE CLASSIFICATION METRICS
# ============================================================

def calculate_metrics(labels, predictions, probabilities):

    accuracy = accuracy_score(labels, predictions)

    balanced_accuracy = balanced_accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    roc_auc = roc_auc_score(
        labels,
        probabilities
    )

    average_precision = average_precision_score(
        labels,
        probabilities
    )

    return {
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Specificity": specificity,
        "ROC-AUC": roc_auc,
        "Average Precision": average_precision
    }
    
# ============================================================
# 4. PREPARE TEST DATA
# ============================================================

def prepare_test_loader(args):

    test_csv = os.path.join(
        args.csv_dir,
        "test_clean.csv"
    )

    test_dataset = brain_mri_dataset(test_csv)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=0
    )

    return test_loader

# ============================================================
# 5. LOAD TRAINED MODEL
# ============================================================

def load_trained_model(fold, args, device):

    model = MyModel(backbone=args.backbone)

    model_path = os.path.join(
        args.output_dir,
        f"best_model_fold_{fold}.pth"
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
            weights_only=True
        )
    )

    model = model.to(device)

    return model

# ============================================================
# 6. MAIN EVALUATION PROCESS
# ============================================================

def main():

    args = get_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Selected device: {device}")

    test_loader = prepare_test_loader(args)

    print(f"Test images: {len(test_loader.dataset)}")

    all_fold_metrics = []

    for fold in range(1, 6):

        print(f"\n========== FOLD {fold} ==========")

        model = load_trained_model(
            fold,
            args,
            device
        )

        names, labels, predictions, probabilities = evaluate_model(
            model,
            test_loader,
            device
        )

        metrics = calculate_metrics(
            labels,
            predictions,
            probabilities
        )
        
        cm = confusion_matrix(
            labels,
            predictions,
            labels=[0, 1]
        )

        print("\nConfusion Matrix:")
        print(cm)

        metrics["Fold"] = fold
        
        predictions_df = pd.DataFrame({
            "Name": names,
            "Actual_Label": labels,
            "Predicted_Label": predictions,
            "Tumor_Probability": probabilities
        })

        predictions_df["Correct"] = (
            predictions_df["Actual_Label"]
            == predictions_df["Predicted_Label"]
        )
        
        predictions_path = os.path.join(
            args.output_dir,
            f"predictions_fold_{fold}.csv"
        )

        predictions_df.to_csv(
            predictions_path,
            index=False
        )

        print(f"Predictions saved to: {predictions_path}")
        
        

        all_fold_metrics.append(metrics)

        for metric, value in metrics.items():

            if metric != "Fold":
                print(f"{metric}: {value:.4f}")

        print("\nClassification Report:")

        print(
            classification_report(
                labels,
                predictions,
                target_names=["No Tumor", "Tumor"],
                zero_division=0
            )
        )

        del model

    results_df = pd.DataFrame(all_fold_metrics)

    print("\n========== FINAL TEST RESULTS ==========")

    print(results_df.to_string(index=False))
    
    results_path = os.path.join(
        args.output_dir,
        "test_metrics.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    print(f"\nTest metrics saved to: {results_path}")
    
    
# ============================================================
# 7. RUN EVALUATION
# ============================================================

if __name__ == "__main__":
    main()