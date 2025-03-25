import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import wandb
import argparse  
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Trains MalariaNet with PyTorch and W&B")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of Epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")
    parser.add_argument("--early_stop", type=int, default=3, help="Patience for early stopping")
    parser.add_argument("--min_delta", type=float, default=1e-3, help="Minimum delta for early stopping")
    return parser.parse_args()

def setup_wandb(args):
    WANDB_API_KEY = None
    login_file = os.path.join(os.path.dirname(__file__), ".wandb_login")

    if os.path.exists(login_file):
        with open(login_file, "r") as f:
            for line in f:
                if line.startswith("WANDB_API_KEY="):
                    WANDB_API_KEY = line.strip().split("=")[1]
                    break

    wandb_mode = "online" if WANDB_API_KEY else "offline"

    if WANDB_API_KEY:
        os.environ["WANDB_API_KEY"] = WANDB_API_KEY
        wandb.login()
    else:
        print("⚠️ No W&B API-Key found! Training runs without W&B-Logging.")

    # W&B-Run starten
    user = os.getenv("USERNAME") or os.getenv("USER") or "unknown"
    wandb.init(
        project="DSHEAL_pro1_FS25_GaMo",
        entity="mockand1-ba",
        # name=f"MalariaNet_{user}_{run_id}",
        mode=wandb_mode
    )

    run_id = wandb.run.id if wandb.run else f"offline-{user}"
    wandb.run.name = f"MalariaNet_e{args.num_epochs}_lr{args.lr}_m{args.momentum}_{run_id}"

    print(f"✅ W&B-Logging startet for user: {user} (Mode: {wandb_mode})")
    return wandb_mode, run_id

def train_model(args, run_id):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Basisverzeichnis
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    RUN_DIR = os.path.join(MODELS_DIR, f"run-{run_id}")
    os.makedirs(RUN_DIR, exist_ok=True)

    sys.path.append(DATA_DIR)
    sys.path.append(MODELS_DIR)

    from data_loader import train_loader, val_loader
    from malaria_model_1 import MalariaNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"runs with: {device}")

    wandb.config.update({
        "num_epochs": args.num_epochs,
        "learning_rate": args.lr,
        "momentum": args.momentum,
        "early_stop": args.early_stop
    })

    model = MalariaNet(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Best Model Tracking
    best_val_acc = 0.0  # Höchste Validierungsgenauigkeit bisher
    best_model_path = None
    # best_model_path = os.path.join(MODELS_DIR, "best_malaria_model.pth")

    early_stop_patience = args.early_stop
    epochs_no_improve = 0
    
    # Training loop with tqdm & W&B logging
    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"🟢 Epoch {epoch+1}/{args.num_epochs}")

        for batch_index, (images, labels) in progress_bar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100 * correct / total:.2f}%")
            # wandb.log({"Train Loss": loss.item()})

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0

        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(val_images)
                loss = criterion(val_outputs, val_labels)
                val_running_loss += loss.item()

                _, val_predicted = torch.max(val_outputs, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()

        train_loss = running_loss / total
        val_loss = val_running_loss / val_total
        val_accuracy = val_correct / val_total
        train_accuracy = correct / total

        wandb.log({
            "Epoch": epoch+1,
            "Train Accuracy": train_accuracy,
            "Validation Accuracy": val_accuracy,
            "Train Loss (Epoch Avg)": train_loss,
            "Validation Loss": val_loss
        })

        print(f"✅ Epoch {epoch+1}/{args.num_epochs} | Loss: {running_loss/total:.4f} | Train Acc: {train_accuracy:.4f} | Val Acc: {val_accuracy:.4f}")

        # **Speichere das Modell nach jeder Epoche**
        epoch_model_path = os.path.join(RUN_DIR, f"malaria_model_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), epoch_model_path)
        print(f"💾 Model saved: {epoch_model_path}")

        # **Speichere NUR das beste Modell**
        if val_accuracy - best_val_acc > args.min_delta:
            epochs_no_improve = 0
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
                print(f"🗑️ Deleted old best model: {best_model_path}")

            best_val_acc = val_accuracy
            best_model_path = os.path.join(MODELS_DIR, f"best_malaria_model_{run_id}_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"🏆 Best model updated! (Validation Accuracy: {best_val_acc:.4f})")

        else:
            epochs_no_improve += 1
            print(f"⏳ No improvement for {epochs_no_improve} epochs.")

            if epochs_no_improve >= early_stop_patience:
                print(f"⛔ Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"🎉 Training finished! best model saved: {best_model_path}")

    MODEL_PATH = os.path.join(RUN_DIR, f"malaria_model_{args.num_epochs}epochs.pth")
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"🎉 Model saved: {MODEL_PATH}")

    wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    wandb_mode, run_id = setup_wandb(args)
    train_model(args, run_id)
