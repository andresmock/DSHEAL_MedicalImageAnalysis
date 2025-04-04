import os
import torch
import wandb
import argparse
import optuna
from sklearn.model_selection import StratifiedKFold
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset, ConcatDataset
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

sys.path.append(DATA_DIR)
sys.path.append(MODELS_DIR)

from data_loader import get_data_path, get_train_transform, val_test_transform
from malaria_model_flex import MalariaNetFlex

def parse_args():
    parser = argparse.ArgumentParser(description="Train MalariaNet with K-Fold CV")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--early_stop", type=int, default=3)
    parser.add_argument("--min_required_val_acc", type=float, default=0.85)
    parser.add_argument("--min_delta", type=float, default=1e-2)
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--optimizer", type=str, default="SGD", choices=["SGD", "Adam"])
    parser.add_argument("--augmentation", type=str, default="basic", choices=["none", "basic", "strong"])
    parser.add_argument("--num_filters", type=int, default=16)
    parser.add_argument("--fc_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    return parser.parse_args()

def train_one_fold(fold, model, train_loader, val_loader, device, criterion, optimizer, args, trial=None):
    best_val_acc = 0.0
    epochs_no_improve = 0
    val_accs = []

    for epoch in range(args.num_epochs):
        model.train()
        total, correct, running_loss = 0, 0, 0.0

        for images, labels in tqdm(train_loader, desc=f"Fold {fold} - Epoch {epoch+1}", ascii=True):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        val_correct, val_total, val_loss = 0, 0, 0.0
        model.eval()
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(val_images)
                loss = criterion(val_outputs, val_labels)
                val_loss += loss.item()
                _, val_preds = torch.max(val_outputs, 1)
                val_total += val_labels.size(0)
                val_correct += (val_preds == val_labels).sum().item()

        train_acc = correct / total
        val_acc = val_correct / val_total
        val_accs.append(val_acc)

        wandb.log({
            "Train Accuracy": train_acc,
            "Validation Accuracy": val_acc,
            "Epoch": epoch + 1,
            "Fold": fold
        })

        print(f"Fold {fold} | Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc - best_val_acc > args.min_delta:
            best_val_acc = val_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Pruning after first 3 epochs if not meeting min accuracy
        if epoch == 2 and best_val_acc < args.min_required_val_acc:
            print(f"Early pruning in Fold {fold} after 3 epochs: val acc {best_val_acc} < min required {args.min_required_val_acc}")
            sys.exit(75)

        # Normal early stopping
        if epochs_no_improve >= args.early_stop:
            print(f"Early stopping in Fold {fold} at Epoch {epoch+1}")
            if best_val_acc < args.min_required_val_acc:
                print(f"Fold {fold} did not reach required min val acc of {args.min_required_val_acc}. Trial will be pruned.")
                if trial is not None:
                    raise optuna.exceptions.TrialPruned()
            break

    return best_val_acc, val_accs

def train_kfold(args, trial=None):
    try:
        data_path = get_data_path()

        train_folder = CustomImageFolder(root=os.path.join(data_path, "train"))
        val_folder = CustomImageFolder(root=os.path.join(data_path, "val"))
        full_dataset = ConcatDataset([train_folder, val_folder])

        all_targets = np.array(train_folder.targets + val_folder.targets)
        skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=42)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        wandb.init(
            project="MalariaNet_KFold",
            config=vars(args),
            mode="online"
        )

        all_val_scores = []
        all_val_accs = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(all_targets)), all_targets)):
            train_subset = Subset(full_dataset, train_idx)
            val_subset = Subset(full_dataset, val_idx)

            train_folder.set_transform(get_train_transform(args.augmentation))
            val_folder.set_transform(val_test_transform)

            train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)

            model = MalariaNetFlex(
                num_classes=2,
                num_filters=args.num_filters,
                fc_size=args.fc_size,
                dropout=args.dropout
            ).to(device)

            criterion = nn.CrossEntropyLoss()

            if args.optimizer == "SGD":
                optimizer = torch.optim.SGD(
                    model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            elif args.optimizer == "Adam":
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            else:
                raise ValueError(f"Unknown optimizer: {args.optimizer}")

            best_val_acc, val_accs = train_one_fold(fold, model, train_loader, val_loader, device, criterion, optimizer, args, trial=trial)
            all_val_scores.append(best_val_acc)
            all_val_accs.extend(val_accs)

        avg_val_acc = np.mean(all_val_scores) if all_val_scores else 0.0
        wandb.log({"Average Validation Accuracy": avg_val_acc})
        wandb.finish()

        print("VAL_ACCURACIES_LOG: " + ",".join([f"{a:.4f}" for a in all_val_accs]))
        print(f"Average Validation Accuracy: {avg_val_acc:.4f}")

        return avg_val_acc

    except optuna.exceptions.TrialPruned:
        print("Trial was pruned.")
        wandb.finish()
        print(">>> sys.exit(75) wird jetzt ausgeführt!")
        sys.exit(75)

class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root=root)
        self.set_transform(transform)

    def set_transform(self, transform):
        self.transform = transform

def main():
    args = parse_args()
    train_kfold(args)

if __name__ == "__main__":
    main()
