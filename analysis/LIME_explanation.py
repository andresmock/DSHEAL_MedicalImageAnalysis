import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from lime import lime_image
from torchvision.transforms import ToPILImage
from skimage.segmentation import mark_boundaries
from PIL import Image
from models.malaria_model_1 import MalariaNet
import torch



# Load the model again to ensure it's in the correct context
# Load model
model = MalariaNet()
model.load_state_dict(torch.load("./models/best_malaria_model_n9thl8el_epoch14.pth", map_location=torch.device('cpu')))
model.eval()

# Transforms
mean = [0.529, 0.424, 0.453]
std = [0.332, 0.268, 0.282]

val_test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Funktion zum Laden des Datenpfads aus `data_path.txt`
def get_data_path(filepath=os.path.join(os.path.dirname(__file__), "../data_path.txt")):
    try:
        with open(filepath, "r") as f:
            return f.readline().strip()  # Entfernt Leerzeichen/Zeilenumbrüche
    except FileNotFoundError:
        raise FileNotFoundError(f"Die Datei '{filepath}' wurde nicht gefunden! Stelle sicher, dass sie existiert.")

# Datenpfad aus `data_path.txt`
data_dir = get_data_path()
test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "test"), transform=val_test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Preprocessing for LIME
def preprocess_for_lime(tensor_img):
    tensor_img = tensor_img * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
    tensor_img = tensor_img.permute(1, 2, 0).numpy()
    tensor_img = np.clip(tensor_img, 0, 1)
    return (tensor_img * 255).astype(np.uint8)

def batch_predict(images):
    model.eval()
    batch = torch.tensor(images).permute(0, 3, 1, 2).float() / 255.0
    for i in range(3):
        batch[:, i] = (batch[:, i] - mean[i]) / std[i]
    with torch.no_grad():
        preds = model(batch)
        return preds.numpy()

# Collect 2 examples per class
tp, fp, tn, fn = [], [], [], []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        for i in range(len(labels)):
            pred, true = predicted[i].item(), labels[i].item()
            img = images[i]
            if pred == true and true == 0 and len(tp) < 2:
                tp.append((img, pred, true))
            elif pred == true and true == 1 and len(tn) < 2:
                tn.append((img, pred, true))
            elif pred == 0 and true == 1 and len(fp) < 2:
                fp.append((img, pred, true))
            elif pred == 1 and true == 0 and len(fn) < 2:
                fn.append((img, pred, true))
        if all(len(lst) == 2 for lst in [tp, tn, fp, fn]):
            break

# Combine and run LIME
all_samples = [("TP", tp), ("FP", fp), ("TN", tn), ("FN", fn)]
explainer = lime_image.LimeImageExplainer()

fig, axs = plt.subplots(4, 4, figsize=(16, 16))
fig.suptitle("TP, FP, TN, FN - Original (Left) and LIME Explanation (Right)", fontsize=18)

for row, (label, group) in enumerate(all_samples):
    for i, (img_tensor, pred, true) in enumerate(group):
        np_img = preprocess_for_lime(img_tensor)

        # Original image
        axs[row, i * 2].imshow(np_img)
        axs[row, i * 2].set_title(f"{label} {i+1} - Original")
        axs[row, i * 2].axis('off')

        # LIME explanation
        explanation = explainer.explain_instance(np_img, batch_predict, top_labels=2, hide_color=0, num_samples=1000)
        temp, mask = explanation.get_image_and_mask(pred, positive_only=False, num_features=10, hide_rest=False)
        axs[row, i * 2 + 1].imshow(mark_boundaries(temp / 255.0, mask))
        axs[row, i * 2 + 1].set_title(f"{label} {i+1} - LIME")
        axs[row, i * 2 + 1].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()