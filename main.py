
# ============================================================
# 1. IMPORT LIBRARIES AND PROJECT MODULES
# ============================================================

import os
import torch
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from args import get_args
from dataset import brain_mri_dataset
from model import MyModel
from trainer import train_model


# ============================================================
# 2. MAIN FUNCTION
# ============================================================

def main():

    # --------------------------------------------------------
    # 2.1 LOAD CONFIGURATION
    # --------------------------------------------------------

    args = get_args()

    # --------------------------------------------------------
    # 2.2 LOAD CLEAN TRAINING DATASET
    # --------------------------------------------------------

    train_df = pd.read_csv(
        os.path.join(args.csv_dir, "train_clean.csv")
    )

    print("\n========== DATASET INFORMATION ==========")

    print("Total training images:", len(train_df))

    print("\nClass distribution:")
    print(train_df["Label"].value_counts())

    # --------------------------------------------------------
    # 2.3 CREATE GROUP IDENTIFIERS
    # --------------------------------------------------------

    # Images with the same MR prefix are assigned to one group.
    # Other filenames are treated as individual groups.
    # These are filename-based groups, not verified patient IDs.

    train_df["Group"] = (
        train_df["Name"]
        .str.extract(r"^(MR\d+)")[0]
        .fillna(train_df["Name"])
    )

    print("\nTotal unique groups:", train_df["Group"].nunique())
    print("Missing group values:", train_df["Group"].isna().sum())

    print("\nFilename and group examples:")
    print(train_df[["Name", "Group"]].head(10))

    # --------------------------------------------------------
    # 2.4 SELECT COMPUTING DEVICE
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n========== TRAINING CONFIGURATION ==========")
    print("Selected device:", device)
    print("Backbone:", args.backbone)
    print("Epochs:", args.epochs)
    print("Batch size:", args.bs)
    print("Learning rate:", args.lr)
    print("Weight decay:", args.wd)

    # --------------------------------------------------------
    # 2.5 CREATE FIVE GROUP-STRATIFIED FOLDS
    # --------------------------------------------------------

    skf = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # TEMPORARY TEST SETTING:
    # Use 1 for the initial test (only Fold 1).
    # Change to 5 when ready for the complete experiment.

    max_folds = 5

    # --------------------------------------------------------
    # 2.6 PROCESS EACH FOLD
    # --------------------------------------------------------

    for fold, (train_indices, val_indices) in enumerate(
        skf.split(
            train_df,
            train_df["Label"],
            train_df["Group"]
        )
    ):

        # Stop after the configured number of folds.
        if fold >= max_folds:
            break

        print(f"\n========== FOLD {fold + 1} ==========")

        # ----------------------------------------------------
        # 2.6.1 SPLIT TRAINING AND VALIDATION DATA
        # ----------------------------------------------------

        train_fold = train_df.iloc[train_indices]
        val_fold = train_df.iloc[val_indices]

        print("Training images:", len(train_fold))
        print("Validation images:", len(val_fold))

        print("\nTraining class distribution:")
        print(train_fold["Label"].value_counts().to_dict())

        print("\nValidation class distribution:")
        print(val_fold["Label"].value_counts().to_dict())

        # ----------------------------------------------------
        # 2.6.2 CHECK GROUP SEPARATION
        # ----------------------------------------------------

        train_groups = set(train_fold["Group"])
        val_groups = set(val_fold["Group"])

        overlapping_groups = train_groups & val_groups

        print("Overlapping groups:", len(overlapping_groups))

        if overlapping_groups:
            raise ValueError(
                f"Group leakage detected in Fold {fold + 1}"
            )

        # ----------------------------------------------------
        # 2.6.3 SAVE FOLD CSV FILES
        # ----------------------------------------------------

        train_csv_path = os.path.join(
            args.csv_dir,
            f"train_fold_{fold}.csv"
        )

        val_csv_path = os.path.join(
            args.csv_dir,
            f"val_fold_{fold}.csv"
        )

        train_fold[["Path", "Name", "Label"]].to_csv(
            train_csv_path,
            index=False
        )

        val_fold[["Path", "Name", "Label"]].to_csv(
            val_csv_path,
            index=False
        )

        print("\nSaved:", train_csv_path)
        print("Saved:", val_csv_path)

        # ----------------------------------------------------
        # 2.6.4 CREATE DATASETS
        # ----------------------------------------------------

        train_dataset = brain_mri_dataset(train_csv_path)
        val_dataset = brain_mri_dataset(val_csv_path)

        # ----------------------------------------------------
        # 2.6.5 CREATE DATALOADERS
        # ----------------------------------------------------

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.bs,
            shuffle=True,
            num_workers=0
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.bs,
            shuffle=False,
            num_workers=0
        )

        # ----------------------------------------------------
        # 2.6.6 VERIFY FIRST BATCH
        # ----------------------------------------------------

        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))

        print("\nTraining batch image shape:",
              train_batch["mri"].shape)

        print("Training batch label shape:",
              train_batch["target"].shape)

        print("\nValidation batch image shape:",
              val_batch["mri"].shape)

        print("Validation batch label shape:",
              val_batch["target"].shape)

        # ----------------------------------------------------
        # 2.6.7 INITIALIZE MODEL
        # ----------------------------------------------------

        model = MyModel(
            backbone=args.backbone
        ).to(device)

        # Verify that the model accepts the input batch.
        with torch.no_grad():

            sample_images = train_batch["mri"].to(device)
            outputs = model(sample_images)

        print("\nModel output shape:", outputs.shape)

        # ----------------------------------------------------
        # 2.6.8 TRAIN AND VALIDATE MODEL
        # ----------------------------------------------------

        best_balanced_accuracy, history = train_model(
            model,
            train_loader,
            val_loader,
            args,
            fold + 1,
            device
        )

        print(
            f"\nFold {fold + 1} best balanced accuracy:",
            best_balanced_accuracy
        )


# ============================================================
# 3. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
