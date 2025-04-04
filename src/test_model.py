import torch
import torch.nn as nn
import os
import argparse
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import numpy as np
import json

# python src\test_model.py --model_path models\___.pth

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained MalariaNet model")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file (.pth)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument("--aug_level", type=str, default="none",
                        help="Data augmentation level NONE for testing")
    return parser.parse_args()

def evaluate_model(model_path, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # == Daten laden ==
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    wandb_dir = os.path.join(base_dir, "wandb")

    import sys
    sys.path.append(data_dir)
    sys.path.append(models_dir)

    from data_loader import create_dataloaders
    from malaria_model_1 import MalariaNet

    # == Modell laden ==
    model = MalariaNet(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"🔍 Loaded model: {model_path}")

    model_filename = os.path.basename(model_path).replace(".pth", "")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"evaluation_{model_filename}.txt")

    all_preds = []
    all_labels = []

    train_loader, val_loader, test_loader = create_dataloaders(
        aug_level=args.aug_level,
        batch_size=batch_size
    )

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # == Auswertung ==
    report = classification_report(all_labels, all_preds, target_names=["Uninfected", "Parasitized"], digits=4)
    cm = confusion_matrix(all_labels, all_preds)

    # == wandb-args aus JSON einlesen ==
    run_id_part = model_filename.split("_")[-2]  # Beispiel: dxtn88y4
    wandb_run_folder = next((f for f in os.listdir(wandb_dir) if run_id_part in f), None)

    wandb_args_dict = {}
    if wandb_run_folder:
        metadata_path = os.path.join(wandb_dir, wandb_run_folder, "files", "wandb-metadata.json")
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                args_list = metadata.get("args", [])
                wandb_args_dict = {args_list[i]: args_list[i+1] for i in range(0, len(args_list), 2)}
        except Exception as e:
            print(f"⚠️ Fehler beim Einlesen der wandb-args: {e}")
    else:
        print(f"⚠️ Keine passende wandb-Run-ID für {model_filename} gefunden.")

    # == In Datei schreiben ==
    with open(results_file, "w") as f:
        f.write(f"Evaluation Results for model: {model_filename}\n\n")

        if wandb_args_dict:
            f.write("=== wandb ARGS ===\n")
            for k, v in wandb_args_dict.items():
                f.write(f"{k}: {v}\n")
            f.write("\n")

        f.write("=== Classification Report ===\n")
        f.write(report)
        f.write("\n\n=== Confusion Matrix ===\n")
        f.write(np.array2string(cm))

    print(f"Ergebnisse gespeichert unter: {results_file}")

if __name__ == "__main__":
    args = parse_args()
    evaluate_model(args.model_path, args.batch_size)
