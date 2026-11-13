import pandas as pd
import cv2

# Load predictions from all five folds
predictions = [
    pd.read_csv(f"sessions/predictions_fold_{i}.csv")
    for i in range(1, 6)
]

# Merge predictions
data = predictions[0][["Name", "Actual_Label"]].copy()

for i, df in enumerate(predictions, start=1):
    data = data.merge(
        df[["Name", "Predicted_Label"]],
        on="Name",
        suffixes=("", f"_{i}")
    )

# Identify images missed by all five models
missed = data[
    (data["Actual_Label"] == 1) &
    (data[[f"Predicted_Label_{i}" for i in range(2, 6)]] == 0).all(axis=1)
    & (data["Predicted_Label"] == 0)
]

# Identify images correctly classified by all five models
correct = data[
    (data["Actual_Label"] == 1) &
    (data["Predicted_Label"] == 1) &
    (data[[f"Predicted_Label_{i}" for i in range(2, 6)]] == 1).all(axis=1)
]

test_data = pd.read_csv("data/CSVs/test_clean.csv")

def measure_images(selected, category):

    selected = selected.merge(
        test_data[["Name", "Path"]],
        on="Name"
    )

    results = []

    for row in selected.itertuples():

        image = cv2.imread(row.Path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue

        results.append({
            "Group": category,
            "Name": row.Name,
            "Mean_Brightness": image.mean(),
            "Brightness_Std": image.std()
        })

    return results


results = []

results.extend(measure_images(missed, "Missed"))
results.extend(measure_images(correct, "Correct"))

comparison = pd.DataFrame(results)

print("Number of images in each group:")
print(comparison["Group"].value_counts())

print("\nComparison:")
print(
    comparison.groupby("Group")[
        ["Mean_Brightness", "Brightness_Std"]
    ].agg(["mean", "median", "std"])
)

comparison.to_csv(
    "sessions/misclassified/tumor_image_comparison.csv",
    index=False
)

print("\nComparison saved successfully.")