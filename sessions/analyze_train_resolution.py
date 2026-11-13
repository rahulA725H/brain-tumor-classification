import pandas as pd
import cv2

train_data = pd.read_csv("data/CSVs/train_clean.csv")

results = []

for row in train_data.itertuples():

    image = cv2.imread(row.Path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        continue

    height, width = image.shape

    results.append({
        "Label": row.Label,
        "Width": width,
        "Height": height,
        "Max_Dimension": max(width, height)
    })

dimensions = pd.DataFrame(results)

dimensions["Actual_Class"] = dimensions["Label"].map({
    0: "No_Tumor",
    1: "Tumor"
})

dimensions["Resolution_Group"] = pd.cut(
    dimensions["Max_Dimension"],
    bins=[0, 256, 512, float("inf")],
    labels=["Up to 256", "257 to 512", "Above 512"]
)

print("Training resolution distribution:")
print(
    pd.crosstab(
        dimensions["Actual_Class"],
        dimensions["Resolution_Group"]
    )
)

print("\nPercentage distribution:")
print(
    pd.crosstab(
        dimensions["Actual_Class"],
        dimensions["Resolution_Group"],
        normalize="index"
    ).mul(100).round(2)
)

print("\nTraining images analyzed:", len(dimensions))