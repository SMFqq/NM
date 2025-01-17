import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device(type="cuda", index=0)
else:
    device = torch.device(type="cpu", index=0)

print(device)

class CustomTrainDataset(Dataset):
    def __init__(self, path, transform):
        super().__init__()
        self.data = pd.read_csv(path, header='infer').values
        self.length = self.data.shape[0]
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        flatimage = self.data[idx, 1:].astype(np.uint8)
        image = self.transform(np.reshape(flatimage, (28, 28, 1)))
        label = self.data[idx, 0]
        return image, label

class CustomTestDataset(Dataset):
    def __init__(self, path, transform):
        super().__init__()
        self.data = pd.read_csv(path, header='infer').values
        self.length = self.data.shape[0]
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        flatimage = self.data[idx, :].astype(np.uint8)
        image = self.transform(np.reshape(flatimage, (28, 28, 1)))
        return image

train_dataset = CustomTrainDataset('train.csv', ToTensor())
test_dataset = CustomTestDataset('test.csv', ToTensor())

batch_size = 64

train_dl = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_dl = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
)

class DRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(8)
        self.mp1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(64)

        self.flatten = nn.Flatten()

        self.lin1 = nn.Linear(in_features=3136, out_features=10)
        self.bn5 = nn.BatchNorm1d(num_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.mp1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.flatten(x)

        x = self.lin1(x)
        output = self.bn5(x)

        return output

def train_one_epoch(dataloader, model, loss_fn, optimizer):
    model.train()
    track_loss = 0
    num_correct = 0
    for i, (imgs, labels) in enumerate(dataloader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        pred = model(imgs)

        loss = loss_fn(pred, labels)
        track_loss += loss.item()
        num_correct += (torch.argmax(pred, dim=1) == labels).type(torch.float).sum().item()

        running_loss = round(track_loss / (i + (imgs.shape[0] / batch_size)), 2)
        running_acc = round((num_correct / ((i * batch_size + imgs.shape[0]))) * 100, 2)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 100 == 0:
            print("Batch:", i + 1, "/", len(dataloader), "Running Loss:", running_loss, "Running Accuracy:", running_acc)

    epoch_loss = running_loss
    epoch_acc = running_acc
    return epoch_loss, epoch_acc

def eval(dataloader, model, loss_fn, path):
    model.eval()
    data = pd.read_csv(path)
    with torch.no_grad():
        for i, imgs in enumerate(dataloader):
            imgs = imgs.to(device)
            pred = model(imgs)

            pred = torch.argmax(pred, dim=1).type(torch.int).cpu()
            data.iloc[i * batch_size:i * batch_size + batch_size, 1] = pred.numpy()

    data.to_csv('submission.csv', index=False)
    data.head()

def visualize_images(dataset, num_images=5):
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        image, label = dataset[i]
        plt.subplot(1, num_images, i + 1)
        plt.imshow(image.squeeze(), cmap="gray")
        plt.title(f"Label: {label}")
        plt.axis("off")
    plt.show()

visualize_images(train_dataset)

train_losses = []
train_accuracies = []

model = DRNN()
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
lr = 0.001
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
n_epochs = 5

for epoch in range(n_epochs):
    print("Epoch No:", epoch + 1)
    train_epoch_loss, train_epoch_acc = train_one_epoch(train_dl, model, loss_fn, optimizer)
    train_losses.append(train_epoch_loss)
    train_accuracies.append(train_epoch_acc)
    print("Training:", "Epoch Loss:", train_epoch_loss, "Epoch Accuracy:", train_epoch_acc)
    print("--------------------------------------------------")

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].plot(range(1, n_epochs+1), train_losses, label="Training Loss")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss")
ax[0].set_title("Training Loss")
ax[0].legend()

ax[1].plot(range(1, n_epochs+1), train_accuracies, label="Training Accuracy", color='green')
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Accuracy")
ax[1].set_title("Training Accuracy")
ax[1].legend()

plt.show()

eval(test_dl, model, loss_fn, 'sample_submission.csv')
