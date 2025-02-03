import torch.nn as nn
import torch.nn.functional as f
import torch

from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from src.data import YoloDataset


class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()

        out_conv1_channels = 16

        self.conv1 = nn.Conv2d(3, out_conv1_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_conv1_channels)

        self.conv2 = nn.Conv2d(out_conv1_channels, out_conv1_channels * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_conv1_channels * 2)

        self.conv3 = nn.Conv2d(out_conv1_channels * 2, out_conv1_channels * 2 * 2, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_conv1_channels * 2 * 2)

        self.conv4 = nn.Conv2d(out_conv1_channels * 4, out_conv1_channels * 4 * 4, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(out_conv1_channels * 4 * 4)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 32 * 32, out_conv1_channels * 8)
        self.fc2 = nn.Linear(128, out_conv1_channels * 4)
        self.fc3 = nn.Linear(64, out_conv1_channels * 2)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.pool(f.relu(self.bn1(self.conv1(x))))
        x = self.pool(f.relu(self.bn2(self.conv2(x))))
        x = self.pool(f.relu(self.bn3(self.conv3(x))))

        x = x.view(-1, 64 * 32 * 32)

        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def main():
    # Force CPU usage
    device = torch.device("cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Load datasets
    train_dataset = YoloDataset("../dataset/images", "../dataset/labels", transform=transform)
    val_dataset = YoloDataset("../data/images/val", "../data/images/val_yolo", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    net = Cnn().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, loss: {epoch_loss:.4f}")

        if (epoch + 1) % 2 == 0:
            net.eval()
            correct = 0
            total = 0
            val_loss = 0.0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_accuracy = 100 * correct / total
            val_loss = val_loss / len(val_loader)
            print(f'Val loss: {val_loss:.4f}, accu: {val_accuracy:.2f}%')

    net.eval()
    correct = 0
    total = 0
    class_correct = [0, 0]
    class_total = [0, 0]

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Per-class accuracy
            for i in range(len(labels)):
                label = labels[i]
                pred = predicted[i]
                class_correct[label] += (pred == label).item()
                class_total[label] += 1

    final_accuracy = 100 * correct / total
    print(f'\nFinal Test Accuracy: {final_accuracy:.2f}%')

    for i in range(2):
        if class_total[i] > 0:
            class_accuracy = 100 * class_correct[i] / class_total[i]
            print(f'Accuracy of class {i}: {class_accuracy:.2f}%')


if __name__ == '__main__':
    main()