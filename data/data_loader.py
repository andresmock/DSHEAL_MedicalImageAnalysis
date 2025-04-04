import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import Counter

# Function for loading the data path from `data_path.txt`.
def get_data_path(filepath=os.path.join(os.path.dirname(__file__), "../data_path.txt")):
    try:
        with open(filepath, "r") as f:
            return f.readline().strip()  
    except FileNotFoundError:
        raise FileNotFoundError(f"Die Datei '{filepath}' wurde nicht gefunden! Stelle sicher, dass sie existiert.")

# Datapath from `data_path.txt`
data_dir = get_data_path()

print(f"Used data path: {data_dir}")

# Data-Transformation 
def get_train_transform(aug_level="basic"):
    if aug_level == "strong":
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-180, 180)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.529, 0.424, 0.453], std=[0.332, 0.268, 0.282])
        ])
    elif aug_level == "none":
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.529, 0.424, 0.453], std=[0.332, 0.268, 0.282])
        ])
    else:  # default: basic
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-15, 15)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.529, 0.424, 0.453], std=[0.332, 0.268, 0.282])
        ])

val_test_transform = transforms.Compose([

    transforms.Resize((128, 128)),                   # Scale images to 128x128
    transforms.ToTensor(),                           # Convert to PyTorch tensor
    transforms.Normalize(mean=[0.529, 0.424, 0.453],
                          std=[0.332, 0.268, 0.282]) # Normalize to -1 to 1

])

def create_dataloaders(aug_level="basic", batch_size=32):
    # Create the right transform for the training
    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "train"),
        transform=get_train_transform(aug_level)  
    )
    
    # Validation & Test unverändert
    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "val"),
        transform=val_test_transform
    )
    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "test"),
        transform=val_test_transform
    )

    # create the DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# Datasets for training, validation and testing
"""
When loading with datasets.ImageFolder(), PyTorch automatically recognizes the subfolders as classes.
The order of the folders determines the label assignment
(Parasitized/) becomes 0
(Uninfected/) becomes 1
PyTorch's ImageFolder() arranges the labels alphabetically according to the folder names
"""

train_dataset_raw = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=None)
val_dataset_raw = datasets.ImageFolder(root=os.path.join(data_dir, "val"), transform=None)
test_dataset_raw = datasets.ImageFolder(root=os.path.join(data_dir, "test"), transform=None)

if __name__ == "__main__":
    # Test: Number of images per dataset
    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of validation images: {len(val_dataset)}")
    print(f"Number of test images: {len(test_dataset)}")
    
    # Test: Load a sample batch
    images, labels = next(iter(train_loader))
    print(f"Loaded batch size: {images.shape}, Labels: {labels}")
