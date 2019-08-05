import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision.models.vgg import vgg16

## Load the pre-trained model
num_classes = 5
model = vgg16(num_classes=1000, pretrained='imagenet')
model.classifier[6] = nn.Linear(4096, num_classes)

## Hyper-parameter
num_epochs, batch_size, learning_rate = 10, 64, 0.1

## Data Transform
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

## Custom Dataset
Train_dataset = ImageFolder(root='./food/train', transform=transform)
Test_dataset = ImageFolder(root='./food/train', transform=transform)
Train_loader = DataLoader(dataset=Train_dataset, shuffle=True, batch_size=batch_size)
# Test_loader = DataLoader(dataset=Test_dataset, shuffle=False, batch_size=batch_size)

## Get the sample data
img = Train_dataset.__getitem__(index=0)[0]
plt.cla()
plt.imshow(img.permute([1,2,0]))
plt.title('sample image')
plt.pause(1)

## Training
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(Train_loader):
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        if (i + 1) % 100 == 0:
            print('Epoch: {}/{}, Batch Step: {}/{}, Loss: {:.4f}, Training Accuracy of the Current Batch: {}%'.
                  format(epoch + 1, num_epochs, i + 1, Train_loader.__len__(), loss.item(), 100 * correct / batch_size))

## Test
model.eval()
with torch.no_grad():
    total, correct  = 0, 0
    for images, labels in Test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Test Accuracy of the 10,000 Test Images: {}%'.format(100 * correct / total))