import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Funktion zum Laden des Datenpfads aus `data_path.txt`
def get_data_path(filepath="data_path.txt"):
    try:
        with open(filepath, "r") as f:
            return f.readline().strip()  # Entfernt Leerzeichen/Zeilenumbrüche
    except FileNotFoundError:
        raise FileNotFoundError(f"Die Datei '{filepath}' wurde nicht gefunden! Stelle sicher, dass sie existiert.")

# Datenpfad aus `data_path.txt`
data_dir = get_data_path()

# Pfad korrekt?
print(f"Verwende Datenpfad: {data_dir}")

# Daten-Transformationen 
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Bilder auf 128x128 skalieren
    transforms.ToTensor(),          # In PyTorch-Tensor umwandeln
    transforms.Normalize((0.5,), (0.5,))  # Normalisieren auf -1 bis 1
])

# Datasets für Training, Validierung und Test
"""
Beim Laden mit datasets.ImageFolder() erkennt PyTorch automatisch die Unterordner als Klassen.
Die Reihenfolge der Ordner bestimmt die Label-Zuordnung
(Parasitized/) wird 0
(Uninfected/) wird 1
PyTorch's ImageFolder() ordnet die Labels alphabetisch nach den Ordnernamen
"""
train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=transform)
val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "val"), transform=transform)
test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "test"), transform=transform)

# DataLoader für Batches
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Test: Lade eine Beispiel-Batch
images, labels = next(iter(train_loader))
print(f"✅ Geladene Batch Grösse: {images.shape}, Labels: {labels}")

