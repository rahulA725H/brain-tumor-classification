import pandas as pd
import cv2

test_data = pd.read_csv("data/CSVs/test_clean.csv")

# Keep only actual tumor images
test_data = test_data[test_data["Label"] == 1].copy()

# Load predictions from all five models
for i in range(1, 6):
    predictions = pd.read_csv(f"sessions/predictions_fold_{i}.csv")

    test_data = test_data.merge(
        predictions[["Name", "Predicted_Label"]],
        on="Name"
    )

    test_data = test_data.rename(
        columns={"Predicted_Label": f"Fold_{i}"}
    )

prediction_columns = [f"Fold_{i}" for i in range(1, 6)]

def categorize(row):
    values = row[prediction_columns]

    if (values == 1).all():
        return "Correct_All_5"
    elif (values == 0).all():
        return "Missed_All_5"
    else:
        return "Mixed_Predictions"

test_data["Category"] = test_data.apply(categorize, axis=1)

results = []

for row in test_data.itertuples():
    image = cv2.imread(row.Path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        continue

    height, width = image.shape

    results.append({
        "Name": row.Name,
        "Category": row.Category,
        "Width": width,
        "Height": height,
        "Max_Dimension": max(width, height)
    })

dimensions = pd.DataFrame(results)

dimensions["Resolution_Group"] = pd.cut(
    dimensions["Max_Dimension"],
    bins=[0, 256, 512, float("inf")],
    labels=["Up to 256", "257 to 512", "Above 512"]
)

print("Tumor images by prediction category:")
print(
    pd.crosstab(
        dimensions["Category"],
        dimensions["Resolution_Group"]
    )
)

print("\nAverage original dimensions:")
print(
    dimensions.groupby("Category")[
        ["Width", "Height", "Max_Dimension"]
    ].mean().round(1)
)

print("\nReport saved successfully.")

dimensions.to_csv(
    "sessions/misclassified/tumor_only_dimension_comparison.csv",
    index=False
)