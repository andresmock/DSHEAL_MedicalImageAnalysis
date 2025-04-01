import torch
import torch.nn as nn
import os
import argparse
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import numpy as np
import sys

# python src\evaluate.py --data_path "C:/ZHAW_local/semester_6_local/DS in Health/project_work/data"

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained MalariaNet model")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--model_path', type=str, default='models/best_malaria_model_attmy8cd_epoch7.pth',
                        help='Path to the trained model file (.pth)')
    return parser.parse_args()

def evaluate(model_path, data_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")

    sys.path.append(data_dir)
    sys.path.append(models_dir)

    # == Importiere lokale Module ==
    from data_loader import create_dataloaders
    from malaria_model_1 import MalariaNet

    # == Modell laden ==
    model = MalariaNet(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f" Loaded model: {model_path}")

    # == Daten laden ==
    train_loader, val_loader, test_loader = create_dataloaders(
        aug_level="none",
        batch_size=32
    )

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # == Ausgabe der Metriken ==
    print("\n=== Classification Report ===")
    print(classification_report(all_labels, all_preds, target_names=["Uninfected", "Parasitized"], digits=4))

    print("=== Confusion Matrix ===")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    args = parse_args()
    evaluate(args.model_path, args.data_path)
