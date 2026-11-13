import pandas as pd
import cv2
from sklearn.tree import export_text
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

def extract_dimensions(csv_path):

    data = pd.read_csv(csv_path)
    results = []

    for row in data.itertuples():

        image = cv2.imread(row.Path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue

        height, width = image.shape

        results.append({
            "Width": width,
            "Height": height,
            "Label": row.Label
        })

    return pd.DataFrame(results)


print("Extracting training image dimensions...")
train_data = extract_dimensions("data/CSVs/train_clean.csv")

print("Extracting testing image dimensions...")
test_data = extract_dimensions("data/CSVs/test_clean.csv")

X_train = train_data[["Width", "Height"]]
y_train = train_data["Label"]

X_test = test_data[["Width", "Height"]]
y_test = test_data["Label"]

# Train a simple classifier using dimensions only
model = DecisionTreeClassifier(max_depth=3, random_state=42)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nResolution-only baseline results:")
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print(
    "Balanced Accuracy:",
    round(balanced_accuracy_score(y_test, predictions), 4)
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["No Tumor", "Tumor"]
    )
)

print("\nTraining images:", len(train_data))
print("Testing images:", len(test_data))

print("\nDecision Tree Rules:")
print(
    export_text(
        model,
        feature_names=["Width", "Height"]
    )
)

print("\nFeature Importance:")
print(
    "Width:", model.feature_importances_[0]
)
print(
    "Height:", model.feature_importances_[1]
)

# Width-only baseline
width_model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

width_model.fit(
    train_data[["Width"]],
    train_data["Label"]
)

width_predictions = width_model.predict(
    test_data[["Width"]]
)

print("\nWidth-only baseline:")
print(
    "Accuracy:",
    round(accuracy_score(test_data["Label"], width_predictions), 4)
)

print(
    "Balanced Accuracy:",
    round(
        balanced_accuracy_score(
            test_data["Label"],
            width_predictions
        ),
        4
    )
)

print("\nWidth-only decision rules:")
print(
    export_text(
        width_model,
        feature_names=["Width"]
    )
)