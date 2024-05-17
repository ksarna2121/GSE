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

class General_SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(General_SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.K = in_channels
        self.linear1 = nn.ModuleList([nn.Linear(self.K, self.K) for _ in range(self.K + 1)])
        self.relu = nn.ReLU()
        self.linear2 = nn.ModuleList([nn.Linear(self.K, self.K) for _ in range(self.K + 1)])

    def forward(self, x):
        b, k, h, w = x.size()
        y = F.avg_pool2d(x, kernel_size=(h, w)).view(b, k)
        y = y.mean(dim=0)
        kernel_arr = []
        for i in range(self.K + 1):
            kernel = self.linear1[i](y)
            kernel = self.relu(kernel)
            kernel = self.linear2[i](kernel)
            kernel = kernel.view(1, k, 1, 1)
            kernel_arr.append(kernel)
        cat_kern = torch.cat(kernel_arr[:-1], dim=0)
        bias = kernel_arr[-1].squeeze((0, 2, 3))
        out = F.conv2d(x, cat_kern, bias=bias)
        return out

class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.block1 = self._make_block(in_channels=3, out_channels=64)
        self.block2 = self._make_block(in_channels=64, out_channels=64)
        self.block3 = self._make_block(in_channels=64, out_channels=64)
        self.fc = nn.Linear(64, 10)

    def _make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            General_SEBlock(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), -1)
        x = self.fc(x)
        return x

model = SimpleConvNet().to(device)

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
