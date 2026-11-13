import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

predictions = [
    pd.read_csv(f"sessions/predictions_fold_{i}.csv")
    for i in range(1, 6)
]

data = predictions[0][["Name", "Actual_Label"]].copy()

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

wrong = data[
    (data["Actual_Label"] == 1) &
    (data[[f"Pred_{i}" for i in range(1, 6)]] == 0).all(axis=1)
]

folder = "sessions/misclassified/all_five_models"

fig, axes = plt.subplots(4, 5, figsize=(16, 13))
axes = axes.flatten()

for ax, row in zip(axes, wrong.itertuples()):
    image = Image.open(
        os.path.join(folder, row.Name)
    ).convert("L")

    ax.imshow(image, cmap="gray")
    ax.set_title(
        f"{row.Name}\n" +
        " | ".join(
            f"F{i}: {getattr(row, f'Prob_{i}'):.2f}"
            for i in range(1, 6)
        ),
        fontsize=7
    )
    ax.axis("off")

for ax in axes[len(wrong):]:
    ax.axis("off")

plt.tight_layout()
output = "sessions/misclassified/all_five_models_contact_sheet.png"
plt.savefig(output, dpi=200, bbox_inches="tight")
plt.close()

print("Images displayed:", len(wrong))
print("Saved:", output)
