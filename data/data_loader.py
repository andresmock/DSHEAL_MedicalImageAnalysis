import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import Counter

# Funktion zum Laden des Datenpfads aus `data_path.txt`
def get_data_path(filepath=os.path.join(os.path.dirname(__file__), "../data_path.txt")):
    try:
        with open(filepath, "r") as f:
            return f.readline().strip()  # Entfernt Leerzeichen/Zeilenumbrüche
    except FileNotFoundError:
        raise FileNotFoundError(f"Die Datei '{filepath}' wurde nicht gefunden! Stelle sicher, dass sie existiert.")

# Datenpfad aus `data_path.txt`
data_dir = get_data_path()

print(f"Used data path: {data_dir}")

# Daten-Transformationen 
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),           # Bilder auf 128x128 skalieren
    transforms.RandomHorizontalFlip(p=0.5),  # 50% Wahrscheinlichkeit für horizontales Spiegeln
    transforms.RandomVerticalFlip(p=0.5),    # 50% Wahrscheinlichkeit für vertikales Spiegeln
    transforms.RandomRotation(degrees=15),   # Zufällige Drehung um bis zu 15 Grad
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Helligkeit, Kontrast, Sättigung variieren
    transforms.ToTensor(),                   # In PyTorch-Tensor umwandeln
    transforms.Normalize((0.5,), (0.5,))     # Normalisieren auf -1 bis 1
])

val_test_transform = transforms.Compose([
    transforms.Resize((128, 128)),           # Bilder auf 128x128 skalieren
    transforms.ToTensor(),                   # In PyTorch-Tensor umwandeln
    transforms.Normalize((0.5,), (0.5,))     # Normalisieren auf -1 bis 1
])

# Datasets für Training, Validierung und Test
"""
Beim Laden mit datasets.ImageFolder() erkennt PyTorch automatisch die Unterordner als Klassen.
Die Reihenfolge der Ordner bestimmt die Label-Zuordnung
(Parasitized/) wird 0
(Uninfected/) wird 1
PyTorch's ImageFolder() ordnet die Labels alphabetisch nach den Ordnernamen
"""
train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=train_transform)
val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "val"), transform=val_test_transform)
test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "test"), transform=val_test_transform)

# DataLoader für Batches
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

if __name__ == "__main__":
    # Test: Anzahl Bilder pro Datensatz
    print(f"📊 Anzahl Bilder im Trainingssatz: {len(train_dataset)}")
    print(f"📊 Anzahl Bilder im Validierungssatz: {len(val_dataset)}")
    print(f"📊 Anzahl Bilder im Testsatz: {len(test_dataset)}")
    
    # Test: Lade eine Beispiel-Batch
    images, labels = next(iter(train_loader))
    print(f"✅ Geladene Batch Grösse: {images.shape}, Labels: {labels}")

