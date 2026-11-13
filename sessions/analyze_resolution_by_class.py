import pandas as pd
import cv2

test_data = pd.read_csv("data/CSVs/test_clean.csv")

results = []

for row in test_data.itertuples():

    image = cv2.imread(row.Path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        continue

    height, width = image.shape

    results.append({
        "Name": row.Name,
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

print("Resolution distribution by actual class:")
print(
    pd.crosstab(
        dimensions["Actual_Class"],
        dimensions["Resolution_Group"]
    )
)

print("\nPercentage distribution within each class:")
print(
    pd.crosstab(
        dimensions["Actual_Class"],
        dimensions["Resolution_Group"],
        normalize="index"
    ).mul(100).round(2)
)

print("\nSaved image resolution report.")

dimensions.to_csv(
    "sessions/misclassified/resolution_by_class.csv",
    index=False
)
