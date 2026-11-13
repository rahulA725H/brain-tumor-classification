
# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import os

import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd

from tqdm import tqdm

from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score
)


# ============================================================
# 2. TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device):

    # Set the model to training mode.
    # This enables training-specific layers such as dropout.
    model.train()

    # Store the total loss from all training batches.
    running_loss = 0.0

    # Process the training dataset batch by batch.
    for batch in tqdm(train_loader, desc="Training"):

        # Move MRI images and their labels to the selected device.
        images = batch["mri"].to(device)
        labels = batch["target"].to(device)

        # Clear gradients from the previous training step.
        optimizer.zero_grad()

        # Forward pass: generate model predictions (logits).
        outputs = model(images)

        # Calculate the difference between predictions and true labels.
        loss = criterion(outputs, labels)

        # Backward pass: calculate gradients.
        loss.backward()

        # Update model parameters using the calculated gradients.
        optimizer.step()

        # Add this batch's loss to the running total.
        running_loss += loss.item()

    # Calculate the average loss across all batches.
    epoch_loss = running_loss / len(train_loader)

    return epoch_loss


# ============================================================
# 3. VALIDATE THE MODEL
# ============================================================

def validate_one_epoch(model, val_loader, criterion, device):

    # Set the model to evaluation mode.
    model.eval()

    # Variables for calculating validation loss and accuracy.
    running_loss = 0.0
    correct = 0
    total = 0

    # Store true labels and predicted probabilities.
    # These are needed for Balanced Accuracy, ROC-AUC and AP.
    all_labels = []
    all_probabilities = []

    # Disable gradient calculations during validation.
    # This saves memory and computation time.
    with torch.no_grad():

        # Process validation data batch by batch.
        for batch in tqdm(val_loader, desc="Validation"):

            # Move MRI images and labels to the selected device.
            images = batch["mri"].to(device)
            labels = batch["target"].to(device)

            # Generate predictions using the model.
            outputs = model(images)

            # Calculate validation loss.
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            # Select the class with the highest predicted score.
            predictions = torch.argmax(outputs, dim=1)

            # Convert logits into probabilities.
            # [:, 1] selects the probability of class 1 (Tumor).
            probabilities = torch.softmax(outputs, dim=1)[:, 1]

            # Count correctly classified images.
            correct += (predictions == labels).sum().item()

            # Count the total number of validation images.
            total += labels.size(0)

            # Collect actual labels and predicted probabilities.
            # Move tensors to CPU before converting them to NumPy.
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # Calculate average validation loss.
    epoch_loss = running_loss / len(val_loader)

    # Calculate overall validation accuracy.
    accuracy = correct / total

    # Convert probabilities into class predictions using threshold 0.5.
    predicted_labels = [
        1 if probability >= 0.5 else 0
        for probability in all_probabilities
    ]

    # Calculate Balanced Accuracy.
    # This considers performance across both classes.
    balanced_accuracy = balanced_accuracy_score(
        all_labels,
        predicted_labels
    )

    # Calculate ROC-AUC using predicted probabilities.
    roc_auc = roc_auc_score(
        all_labels,
        all_probabilities
    )

    # Calculate Average Precision (AP).
    average_precision = average_precision_score(
        all_labels,
        all_probabilities
    )

    # Return all validation metrics.
    return (
        epoch_loss,
        accuracy,
        balanced_accuracy,
        roc_auc,
        average_precision
    )


# ============================================================
# 4. COMPLETE MODEL TRAINING PROCESS
# ============================================================

def train_model(model, train_loader, val_loader, args, fold, device):

    # --------------------------------------------------------
    # 4.1 Define the loss function
    # --------------------------------------------------------

    # CrossEntropyLoss measures the difference between
    # the model's output logits and the actual class labels.
    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # 4.2 Define the optimizer
    # --------------------------------------------------------

    # Adam updates model parameters to reduce the training loss.
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd
    )

    # Store the best validation Balanced Accuracy.
    # -1.0 ensures that the first epoch can save a checkpoint.
    best_balanced_accuracy = -1.0
    
    # Early stopping configuration.
    patience = 5
    epochs_without_improvement = 0

    # Create the output folder if it does not already exist.
    os.makedirs(args.output_dir, exist_ok=True)

    # --------------------------------------------------------
    # 4.3 Initialize training history
    # --------------------------------------------------------

    # These lists store metrics from every completed epoch.
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_balanced_accuracy": [],
        "val_roc_auc": [],
        "val_ap": []
    }

    # Create a separate CSV filename for each fold.
    # Example: sessions/history_fold_1.csv
    history_path = os.path.join(
        args.output_dir,
        f"history_fold_{fold}.csv"
    )

    # --------------------------------------------------------
    # 4.4 Start the epoch loop
    # --------------------------------------------------------

    for epoch in range(args.epochs):

        print(f"\nFold {fold} | Epoch {epoch + 1}/{args.epochs}")

        # ----------------------------------------------------
        # 4.5 Train the model for one epoch
        # ----------------------------------------------------

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        # ----------------------------------------------------
        # 4.6 Validate the model after training
        # ----------------------------------------------------

        (
            val_loss,
            val_accuracy,
            val_balanced_accuracy,
            val_roc_auc,
            val_ap
        ) = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device
        )

        # ----------------------------------------------------
        # 4.7 Store metrics in the history dictionary
        # ----------------------------------------------------

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_balanced_accuracy"].append(val_balanced_accuracy)
        history["val_roc_auc"].append(val_roc_auc)
        history["val_ap"].append(val_ap)

        # Save the current history after every epoch.
        # This helps preserve results if training stops unexpectedly.
        pd.DataFrame(history).to_csv(history_path, index=False)

        # ----------------------------------------------------
        # 4.8 Display the epoch results
        # ----------------------------------------------------

        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation Balanced Accuracy: {val_balanced_accuracy:.4f}")
        print(f"Validation ROC-AUC: {val_roc_auc:.4f}")
        print(f"Validation Average Precision: {val_ap:.4f}")

        # ----------------------------------------------------
        # 4.9 Save the best model checkpoint
        # ----------------------------------------------------

        # Save the model only when Balanced Accuracy improves.
        if val_balanced_accuracy > best_balanced_accuracy:

            best_balanced_accuracy = val_balanced_accuracy
            # Reset the counter because validation performance improved.
            epochs_without_improvement = 0

            # Create a separate checkpoint filename for each fold.
            model_path = os.path.join(
                args.output_dir,
                f"best_model_fold_{fold}.pth"
            )

            # Save the model's learned parameters.
            torch.save(model.state_dict(), model_path)

            print(f"Best model saved: {model_path}")
        
        else:
            # Count an epoch when Balanced Accuracy does not improve.
            epochs_without_improvement += 1

            print(
                f"No improvement: "
                f"{epochs_without_improvement}/{patience}"
            )
            
        # Stop training if validation performance has not improved
        # for the configured number of consecutive epochs.
        if epochs_without_improvement >= patience:

            print(
                f"Early stopping triggered at epoch {epoch + 1}. "
                f"Best Balanced Accuracy: {best_balanced_accuracy:.4f}"
            )

            break  
        

    # --------------------------------------------------------
    # 4.10 Return the best result and complete training history
    # --------------------------------------------------------

    return best_balanced_accuracy, history