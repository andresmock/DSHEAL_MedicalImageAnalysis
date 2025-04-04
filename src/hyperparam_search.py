import optuna
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--early_stop", type=int, default=5)
    parser.add_argument("--study_name", type=str, default="malaria")
    parser.add_argument("--storage", type=str, default="sqlite:///malaria_optuna.db")
    args = parser.parse_args()

    def objective(trial):
        # Hyperparameter for Optuna
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        momentum = trial.suggest_float("momentum", 0.5, 0.99)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        optimizer = trial.suggest_categorical("optimizer", ["SGD", "Adam"])
        augmentation = trial.suggest_categorical("augmentation", ["none", "basic", "strong"]) # ["none", "basic", "strong"]
        num_filters = trial.suggest_categorical("num_filters", [16])  # [16, 32, 64]
        fc_size = trial.suggest_categorical("fc_size", [64]) # [64, 128, 256]
        dropout = trial.suggest_float("dropout", 0.0, 0.0) # 0.0, 0.5
        weight_decay = trial.suggest_float("weight_decay", 0, 0) # 1e-6, 1e-2, log=True

        # Calling up the training script with the suggested parameters
        command = [
            "python", "src/train_model_cv.py",
            "--num_epochs", str(args.num_epochs),
            "--lr", str(lr),
            "--momentum", str(momentum),
            "--batch_size", str(batch_size),
            "--optimizer", optimizer,
            "--augmentation", augmentation,
            "--num_filters", str(num_filters),
            "--fc_size", str(fc_size),
            "--dropout", str(dropout),
            "--weight_decay", str(weight_decay),
            "--early_stop", str(args.early_stop),
            "--k_folds", str(args.k_folds),

        ]

        try:
            # Start the transferred command as a subprocess and forward stdout and stderr together
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            output_lines = []  # List for saving all outputs of the sub-process
            for line in process.stdout:
                print(line, end="") # (Live-Feedback)
                output_lines.append(line) # Saves the line for later evaluation

            process.wait()

            # Check whether pruned by Optuna
            # If the return value is 75, the attempt was explicitly marked as “pruned” by Optuna
            if process.returncode == 75:
                raise optuna.exceptions.TrialPruned()

            # Search the output lines for VAL_ACCURACIES_LOG (a list of validation accuracies per epoch)
            for line in output_lines:
                if "VAL_ACCURACIES_LOG" in line:
                    # Extract the accuracies as a list of floats
                    val_accs = [float(a) for a in line.strip().split(":")[-1].split(",")]

                    # Reports each of these accuracies to Optuna
                    for i, acc in enumerate(val_accs):
                        trial.report(acc, step=i) # Reports the Accuracy at Epoch

                        # Whether the current trial should be stopped based on the results to date
                        if trial.should_prune():
                            print(f"Trial {trial.number} has been pruned in Epoch {i+1}.")
                            raise optuna.exceptions.TrialPruned() # Abort the trial prematurely
                        
                # If "Average Validation Accuracy" is found in a line, try to extract the float value
                if "Average Validation Accuracy" in line:
                    try:
                        acc = float(line.strip().split(":")[-1])
                        return acc # Return of the final validation accuracy as an optimization target
                    except ValueError:
                        pass 

        except Exception as e:
            print("Mistakes during training:")
            print(e)
            return 0.0

        except optuna.exceptions.TrialPruned:
            print("Trial was canceled by Optuna.")
            raise
    
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True
    )
    study.optimize(objective, n_trials=args.n_trials)

    print("\n\nBest parameter combination:")
    print(study.best_params)
    print(f"Best validation accuracy: {study.best_value:.4f}")

if __name__ == "__main__":
    main()

