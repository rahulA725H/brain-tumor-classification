import pandas as pd
import cv2

test_data = pd.read_csv("data/CSVs/test_clean.csv")

predictions = [
    pd.read_csv(f"sessions/predictions_fold_{i}.csv")
    for i in range(1, 6)
]

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

results = []

for row in test_data.itertuples():

    image = cv2.imread(row.Path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        continue

    results.append({
        "Name": row.Name,
        "Actual_Label": row.Label,
        "Mean_Brightness": image.mean(),
        "Brightness_Std": image.std()
    })

comparison = pd.DataFrame(results)

comparison["Actual_Class"] = comparison["Actual_Label"].map({
    0: "No_Tumor",
    1: "Tumor"
})

print("Number of images in each actual class:")
print(comparison["Actual_Class"].value_counts())

print("\nBrightness comparison by actual class:")
print(
    comparison.groupby("Actual_Class")[
        ["Mean_Brightness", "Brightness_Std"]
    ].agg(["mean", "median", "std"])
)

comparison.to_csv(
    "sessions/misclassified/label_brightness_comparison.csv",
    index=False
)

print("\nReport saved successfully.")