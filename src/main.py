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
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(f.relu(self.bn1(self.conv1(x))))
        x = self.pool(f.relu(self.bn2(self.conv2(x))))
        x = self.pool(f.relu(self.bn3(self.conv3(x))))

        x = x.view(-1, 64 * 32 * 32)

        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))

        x = self.fc3(x)
        return x

def main():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = YoloDataset("../data/images/raw", "../data/images/raw_yolo", transform=transform)
    val_dataset = YoloDataset("../data/images/val", "../data/images/val_yolo", transform=transform)

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    net = Cnn()
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=0.001/2.2)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.01)

    for epoch in range(14):
        running_loss = 0.0
        for images, labels in dataloader:
            optimizer.zero_grad()

            outputs = net(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # scheduler.step()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")

    correct = 0
    total = 0

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    with torch.no_grad(): # Not working
        net.eval()

        for images, labels in val_loader:
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct / total}%')

if __name__ == '__main__':
    main()
