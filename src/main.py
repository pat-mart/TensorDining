import torch.nn as nn
import torch.nn.functional as f

from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from src.data import YoloDataset


class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 64 * 64, 128)  # Adjust this based on your input image size
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        x = x.view(-1, 32 * 64 * 64)
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = YoloDataset("../data/images/raw", "../data/images/raw_yolo", transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    net = Cnn()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(50):
        running_loss = 0.0
        for images, labels in dataloader:
            optimizer.zero_grad()

            outputs = net(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")

if __name__ == '__main__':
    main()
