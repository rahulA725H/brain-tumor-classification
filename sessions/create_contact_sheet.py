import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

predictions = pd.read_csv("sessions/predictions_fold_1.csv")
wrong = predictions[predictions["Correct"] == False]

image_folder = "sessions/misclassified/fold_1"

fig, axes = plt.subplots(7, 9, figsize=(20, 17))
axes = axes.flatten()

for ax, row in zip(axes, wrong.itertuples()):
    image_path = os.path.join(image_folder, row.Name)
    image = Image.open(image_path).convert("L")

    ax.imshow(image, cmap="gray")
    ax.set_title(
        f"{row.Name}\nTumor probability: {row.Tumor_Probability:.3f}",
        fontsize=7
    )
    ax.axis("off")

for ax in axes[len(wrong):]:
    ax.axis("off")

plt.tight_layout()

output_path = "sessions/misclassified/fold_1_contact_sheet.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close()

print("Contact sheet saved:", output_path)
