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
        # Vorschlag neuer Hyperparameter durch Optuna
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        momentum = trial.suggest_float("momentum", 0.5, 0.99)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        optimizer = trial.suggest_categorical("optimizer", ["SGD", "Adam"])
        augmentation = trial.suggest_categorical("augmentation", ["none", "basic", "strong"]) # ["none", "basic", "strong"]
        num_filters = trial.suggest_categorical("num_filters", [16])  # [16, 32, 64]
        fc_size = trial.suggest_categorical("fc_size", [64]) # [64, 128, 256]
        dropout = trial.suggest_float("dropout", 0.0, 0.0) # 0.0, 0.5
        weight_decay = trial.suggest_float("weight_decay", 0, 0) # 1e-6, 1e-2, log=True

        # Aufruf des Trainingsskripts mit den vorgeschlagenen Parametern
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
            # Starte den übergebenen Befehl als Subprozess und leite stdout und stderr zusammen weiter
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            output_lines = []  # Liste zum Speichern aller Ausgaben des Subprozesses
            for line in process.stdout:
                print(line, end="") # (für Live-Feedback)
                output_lines.append(line) # Speichert die Zeile für spätere Auswertung

            process.wait() # Warte, bis der Prozess vollständig abgeschlossen ist

            # Prüfen, ob durch Optuna gepruned wurde
            # Wenn der Rückgabewert 75 ist, wurde der Versuch durch Optuna explizit als "pruned" markiert
            if process.returncode == 75:
                raise optuna.exceptions.TrialPruned()

            # Durchsuche die Ausgabezeilen nach VAL_ACCURACIES_LOG (eine Liste der Validierungs-Genauigkeiten je Epoche)
            for line in output_lines:
                if "VAL_ACCURACIES_LOG" in line:
                    # Extrahiere die Genauigkeiten als Liste von floats
                    val_accs = [float(a) for a in line.strip().split(":")[-1].split(",")]

                    # Berichte jeder dieser Genauigkeiten an Optuna (z. B. zur späteren Analyse oder Pruning-Entscheidung)
                    for i, acc in enumerate(val_accs):
                        trial.report(acc, step=i) # Reporte die Accuracy bei Epoche 

                        # Prüfe, ob das aktuelle Trial basierend auf den bisherigen Resultaten gestoppt werden soll
                        if trial.should_prune():
                            print(f"Trial {trial.number} has been pruned in Epoch {i+1}.")
                            raise optuna.exceptions.TrialPruned() # Breche das Trial vorzeitig ab
                        
                # Falls "Average Validation Accuracy" in einer Zeile gefunden wird, versuche den float-Wert zu extrahieren
                if "Average Validation Accuracy" in line:
                    try:
                        acc = float(line.strip().split(":")[-1])
                        return acc # Rückgabe der finalen Validierungsgenauigkeit als Optimierungsziel
                    except ValueError:
                        pass 

        except Exception as e:
            print("Fehler beim Training:")
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

