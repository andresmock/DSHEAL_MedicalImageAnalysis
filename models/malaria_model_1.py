import torch
import torch.nn as nn
import torch.optim as optim

class MalariaNet(nn.Module):
    """
    Models a simple Convolutioal Neural Network for Malaria Cell Classification.
    """
    def __init__(self, num_classes=2):
        super(MalariaNet, self).__init__()

        # convolutional layer 1 & max pool layer 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # convolutional layer 2 & max pool layer 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # fully connected layer
        self.fc = nn.Linear(32*30*30, num_classes)

    # Feed forward the network
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


if __name__ == "__main__":
    # Test: Model-Instanz
    model = MalariaNet()
    print(model)

    # Test: Loss-Funktion und Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Test: Loss-Funktion
    x = torch.randn(32, 3, 128, 128)
    y = model(x)
    print(f"Output-Shape: {y.shape}")

    # Test: Optimizer
    print(f"Optimizer: {optimizer}")

    # Test: Loss-Funktion
    print(f"Loss-Funktion: {criterion}")

    # Test: Parameter
    print(f"Model-Parameter: {model.parameters}")
