import torch
import torch.nn as nn
import os
import argparse
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import numpy as np

# python src\test_model.py --model_path models\___.pth

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained MalariaNet model")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file (.pth)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    return parser.parse_args()

def evaluate_model(model_path, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # == Daten laden ==
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")

    import sys
    sys.path.append(data_dir)
    sys.path.append(models_dir)

    from data_loader import test_loader  # nur Validation verwenden
    from malaria_model_1 import MalariaNet

    # == Modell laden ==
    model = MalariaNet(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"🔍 Loaded model: {model_path}")

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

    # == Auswertung ==
    print("\n Evaluation Report:")
    print(classification_report(all_labels, all_preds, target_names=["Uninfected", "Parasitized"], digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    print(" Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    args = parse_args()
    evaluate_model(args.model_path, args.batch_size)
