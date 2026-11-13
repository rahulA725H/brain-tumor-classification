# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 2. DEFINE THE PLOTTING FUNCTION
# ============================================================

def plot_training_history(history_path, fold, output_dir):

    # Read the training history CSV file.
    history = pd.read_csv(history_path)

    # Create the output folder if it does not exist.
    os.makedirs(output_dir, exist_ok=True)
    
    # Create epoch numbers for the x-axis.
    epochs = range(1, len(history) + 1)
    
    # --------------------------------------------------------
    # 3. PLOT TRAINING LOSS AND VALIDATION LOSS
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    # Plot training loss.
    plt.plot(
        epochs,
        history["train_loss"],
        label="Training Loss"
    )

    # Plot validation loss.
    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )

    # Add graph title and axis labels.
    plt.title(f"Fold {fold} - Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # Display legend and grid.
    plt.legend()
    plt.grid(True)

    # Save the graph as an image.
    plt.savefig(
        os.path.join(output_dir, f"fold_{fold}_loss.png")
    )

    # Close the graph to release memory.
    plt.close()
    
    # --------------------------------------------------------
    # 4. PLOT VALIDATION BALANCED ACCURACY
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    # Plot validation Balanced Accuracy across epochs.
    plt.plot(
        epochs,
        history["val_balanced_accuracy"],
        label="Validation Balanced Accuracy"
    )

    plt.title(f"Fold {fold} - Validation Balanced Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Balanced Accuracy")

    plt.legend()
    plt.grid(True)

    # Save the graph.
    plt.savefig(
        os.path.join(output_dir, f"fold_{fold}_balanced_accuracy.png")
    )

    plt.close()
    
    
    # --------------------------------------------------------
    # 5. PLOT VALIDATION ROC-AUC
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    # Plot ROC-AUC across training epochs.
    plt.plot(
        epochs,
        history["val_roc_auc"],
        label="Validation ROC-AUC"
    )

    plt.title(f"Fold {fold} - Validation ROC-AUC")
    plt.xlabel("Epoch")
    plt.ylabel("ROC-AUC")

    plt.legend()
    plt.grid(True)

    # Save the graph.
    plt.savefig(
        os.path.join(output_dir, f"fold_{fold}_roc_auc.png")
    )

    plt.close()
    
    # --------------------------------------------------------
    # 6. PLOT VALIDATION AVERAGE PRECISION
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    # Plot Average Precision across training epochs.
    plt.plot(
        epochs,
        history["val_ap"],
        label="Validation Average Precision"
    )

    plt.title(f"Fold {fold} - Validation Average Precision")
    plt.xlabel("Epoch")
    plt.ylabel("Average Precision")

    plt.legend()
    plt.grid(True)

    # Save the graph.
    plt.savefig(
        os.path.join(output_dir, f"fold_{fold}_average_precision.png")
    )

    plt.close()
    
# ============================================================
# 7. GENERATE GRAPHS FOR ALL FIVE FOLDS
# ============================================================

def generate_all_plots():

    # Define the folder containing training history CSV files.
    history_dir = "sessions"

    # Define the folder where graphs will be saved.
    output_dir = os.path.join(history_dir, "plots")

    # Process all five folds.
    for fold in range(1, 6):

        # Create the CSV filename for the current fold.
        history_path = os.path.join(
            history_dir,
            f"history_fold_{fold}.csv"
        )

        # Check whether the history file exists.
        if not os.path.exists(history_path):
            print(f"History file not found: {history_path}")
            continue

        # Generate graphs for the current fold.
        plot_training_history(
            history_path,
            fold,
            output_dir
        )

        print(f"Graphs generated for Fold {fold}")

# ============================================================
# 8. MAIN PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    generate_all_plots()