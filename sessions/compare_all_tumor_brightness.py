import pandas as pd
import cv2

# Load the clean test dataset
test_data = pd.read_csv("data/CSVs/test_clean.csv")

# Keep only actual tumor images
test_data = test_data[test_data["Label"] == 1].copy()

# Load predictions from all five models
predictions = [
    pd.read_csv(f"sessions/predictions_fold_{i}.csv")
    for i in range(1, 6)
]

# Add each model's prediction
for i, df in enumerate(predictions, start=1):
    test_data = test_data.merge(
        df[["Name", "Predicted_Label"]],
        on="Name"
    )
    test_data = test_data.rename(
        columns={"Predicted_Label": f"Fold_{i}_Prediction"}
    )

prediction_columns = [
    f"Fold_{i}_Prediction" for i in range(1, 6)
]

# Categorize each tumor image
test_data["Category"] = test_data[prediction_columns].apply(
    lambda row:
        "Correct_All_5" if (row == 1).all()
        else "Missed_All_5" if (row == 0).all()
        else "Mixed_Predictions",
    axis=1
)

# Calculate image brightness statistics
results = []

for row in test_data.itertuples():

    image = cv2.imread(row.Path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        continue

    results.append({
        "Name": row.Name,
        "Category": row.Category,
        "Mean_Brightness": image.mean(),
        "Brightness_Std": image.std()
    })

comparison = pd.DataFrame(results)

print("Number of tumor images in each category:")
print(comparison["Category"].value_counts())

print("\nBrightness comparison:")
print(
    comparison.groupby("Category")[
        ["Mean_Brightness", "Brightness_Std"]
    ].agg(["mean", "median", "std"])
)

comparison.to_csv(
    "sessions/misclassified/all_tumor_brightness_comparison.csv",
    index=False
)

print("\nReport saved successfully.")