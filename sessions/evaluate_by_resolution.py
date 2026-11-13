import pandas as pd
import cv2

from sklearn.metrics import confusion_matrix

test_data = pd.read_csv("data/CSVs/test_clean.csv")

# Determine original image dimensions
widths = []
heights = []

for row in test_data.itertuples():
    image = cv2.imread(row.Path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(row.Path)

    height, width = image.shape
    widths.append(width)
    heights.append(height)

test_data["Width"] = widths
test_data["Height"] = heights
test_data["Max_Dimension"] = test_data[
    ["Width", "Height"]
].max(axis=1)

test_data["Resolution_Group"] = pd.cut(
    test_data["Max_Dimension"],
    bins=[0, 256, 512, float("inf")],
    labels=["Up to 256", "257 to 512", "Above 512"]
)

for fold in range(1, 6):

    predictions = pd.read_csv(
        f"sessions/predictions_fold_{fold}.csv"
    )

    data = test_data.merge(
        predictions[["Name", "Predicted_Label"]],
        on="Name"
    )

    print(f"\n========== FOLD {fold} ==========")

    for group in ["Up to 256", "257 to 512", "Above 512"]:

        subset = data[data["Resolution_Group"] == group]

        if subset.empty:
            continue

        tn, fp, fn, tp = confusion_matrix(
            subset["Label"],
            subset["Predicted_Label"],
            labels=[0, 1]
        ).ravel()

        tumor_recall = tp / (tp + fn) if (tp + fn) else 0
        specificity = tn / (tn + fp) if (tn + fp) else 0

        print(f"\nResolution: {group}")
        print("Images:", len(subset))
        print("Missed tumor images (FN):", fn)
        print("Tumor recall:", round(tumor_recall, 4))
        print("Specificity:", round(specificity, 4))