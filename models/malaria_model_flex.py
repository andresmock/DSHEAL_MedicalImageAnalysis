import torch
import torch.nn as nn
import torch.optim as optim

class MalariaNetFlex(nn.Module):
    """
    Flexible CNN for malaria cell classification - with dynamic parameters
    """
    def __init__(self, num_classes=2, num_filters=16, fc_size=128, dropout=0.0):
        super(MalariaNetFlex, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, num_filters, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters * 2, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.dropout = nn.Dropout(dropout)

        # Calculate flatten size dynamically (128x128 input)
        flatten_size = (128 - 2) // 2  # Layer1 conv + pool
        flatten_size = (flatten_size - 3) // 2  # Layer2 conv + pool
        flatten_size = flatten_size * flatten_size * (num_filters * 2)

        self.fc = nn.Linear(flatten_size, fc_size)
        self.classifier = nn.Linear(fc_size, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Test setup for dynamic architecture
    model = MalariaNetFlex(num_filters=32, fc_size=128, dropout=0.3)
    print(model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

    x = torch.randn(32, 3, 128, 128)
    y = model(x)
    print(f"Output-Shape: {y.shape}")
    print(f"Optimizer: {optimizer}")
    print(f"Loss-Funktion: {criterion}")
