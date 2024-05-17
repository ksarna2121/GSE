import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cifar_train = datasets.CIFAR10(root='~/data/CIFAR10', download=True, train=True)
cifar_test = datasets.CIFAR10(root='~/data/CIFAR10', download=True, train=False)
x_train, y_train = cifar_train.data, torch.tensor(cifar_train.targets)
x_test, y_test = cifar_test.data, torch.tensor(cifar_test.targets)
print('HERE')

class CIFAR10Dataset(Dataset):
    def __init__(self, x, y):
        x = torch.tensor(x).permute(0, 3, 1, 2).float() / 255  # permute to (N, C, H, W) and normalize
        self.x, self.y = x, y

    def __getitem__(self, ix):
        return self.x[ix].to(device), self.y[ix].to(device)

    def __len__(self):
        return len(self.x)

train_dataset = CIFAR10Dataset(x_train, y_train)
train_dl = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = CIFAR10Dataset(x_test, y_test)
test_dl = DataLoader(test_dataset, batch_size=64, shuffle=False)

class SEBlock(nn.Module):
    def __init__(self, ch, ratio=16):
        super(SEBlock, self).__init__()
        self.ch = ch
        self.ratio = ratio
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(ch, ch // ratio)
        self.fc2 = nn.Linear(ch // ratio, ch)

    def forward(self, x):
        out = self.pooling(x)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        out = out.unsqueeze(2).unsqueeze(3)
        out = x * out
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.se_block = SEBlock(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(identity)
        out = self.relu(out)
        out = self.se_block(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def ResNet18():
    return ResNet(ResidualBlock, [2, 2, 2])

model = ResNet18().to(device)

from torchsummary import summary
summary(model, (3, 32, 32))  # Adjusted for CIFAR-10 input size

loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=1e-3)

def train_batch(x, y, model, opt, loss_fn):
    model.train()
    opt.zero_grad()
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    opt.step()
    return loss.item()

@torch.no_grad()
def accuracy(x, y, model):
    model.eval()
    pred = model(x)
    correct = (torch.argmax(pred, dim=1) == y).float().sum()
    acc = correct / float(x.size(0))
    return acc.item()

losses, accuracies, n_epochs = [], [], 100
for epoch in range(n_epochs):
    print(f"Running epoch {epoch+1} of {n_epochs}")

    epoch_losses, epoch_accuracies = [], []
    for batch in train_dl:
        x, y = batch
        x, y = x.to(device), y.to(device)
        batch_loss = train_batch(x, y, model, opt, loss_fn)
        epoch_losses.append(batch_loss)
    epoch_loss = np.mean(epoch_losses)

    for batch in test_dl:
        x, y = batch
        x, y = x.to(device), y.to(device)
        batch_acc = accuracy(x, y, model)
        epoch_accuracies.append(batch_acc)
    epoch_accuracy = np.mean(epoch_accuracies)

    losses.append(epoch_loss)
    accuracies.append(epoch_accuracy)

plt.figure(figsize=(13, 3))
plt.subplot(121)
plt.title("Training Loss value over epochs")
plt.plot(np.arange(n_epochs)+1, losses)
plt.subplot(122)
plt.title("Testing Accuracy value over epochs")
plt.plot(np.arange(n_epochs)+1, accuracies)

plt.show()

test_accuracies = []
for batch in test_dl:
    x, y = batch
    x, y = x.to(device), y.to(device)
    acc = accuracy(x, y, model)
    test_accuracies.append(acc)

print(f"Test accuracy: {np.mean(test_accuracies)}")
