import os
import pandas as pd
import cv2

# Load predictions from all five folds
predictions = [
    pd.read_csv(f"sessions/predictions_fold_{i}.csv")
    for i in range(1, 6)
]

# Start with image names and actual labels
data = predictions[0][["Name", "Actual_Label"]].copy()

# Add predictions and probabilities from every fold
for i, df in enumerate(predictions, start=1):
    data = data.merge(
        df[["Name", "Predicted_Label", "Tumor_Probability"]].rename(
            columns={
                "Predicted_Label": f"Pred_{i}",
                "Tumor_Probability": f"Prob_{i}"
            }
        ),
        on="Name"
    )

# Select tumor images missed by all five models
wrong = data[
    (data["Actual_Label"] == 1) &
    (data[[f"Pred_{i}" for i in range(1, 6)]] == 0).all(axis=1)
].copy()

# Load image paths
test_data = pd.read_csv("data/CSVs/test_clean.csv")

wrong = wrong.merge(
    test_data[["Name", "Path"]],
    on="Name"
)

# Calculate image characteristics
results = []

for row in wrong.itertuples():

    image = cv2.imread(row.Path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Could not read:", row.Name)
        continue

    results.append({
        "Name": row.Name,
        "Height": image.shape[0],
        "Width": image.shape[1],
        "Mean_Brightness": image.mean(),
        "Brightness_Std": image.std(),
        "Min_Pixel": image.min(),
        "Max_Pixel": image.max(),
        "Prob_Fold_1": row.Prob_1,
        "Prob_Fold_2": row.Prob_2,
        "Prob_Fold_3": row.Prob_3,
        "Prob_Fold_4": row.Prob_4,
        "Prob_Fold_5": row.Prob_5
    })

# Save analysis report
analysis_df = pd.DataFrame(results)

output_path = (
    "sessions/misclassified/"
    "all_five_models_analysis.csv"
)

analysis_df.to_csv(output_path, index=False)

print("\nImages analyzed:", len(analysis_df))
print("Report saved:", output_path)

print("\nImage characteristics:")
print(
    analysis_df[
        ["Mean_Brightness", "Brightness_Std"]
    ].describe()
)