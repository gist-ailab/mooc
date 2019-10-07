# TODO: 학습이 잘 안되는 문제 해결 필요

import torch, torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm
import sys

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

num_epochs, num_classes, batch_size, learning_rate = 10, 10, 16, 0.001

from lec5_models import ConvNet, VGGNet, AlexNet
network = 'alexnet' # simple_cnn, vgg-16, resnet

# build network
if network == 'simple-cnn':
    model = ConvNet().to(device)
elif network == 'alexnet':
    model = AlexNet(num_classes=num_classes).to(device)
elif network == 'vgg-16':
    model = VGGNet(num_classes=num_classes).to(device)
else:
    sys.exit()

composed_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            transform=composed_transforms, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size, shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            transform=composed_transforms)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size, shuffle=False)

model.train()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()

        if (i + 1) % 1000 == 0:
            print('Epoch: {}/{}, Batch Step: {}/{}, Loss: {:.4f}, Training Accuracy of the Current Batch: {}%'.
                  format(epoch + 1, num_epochs, i + 1, train_loader.__len__(), loss.item(), 100 * correct / batch_size))


model.eval()
with torch.no_grad():
    total, correct  = 0, 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Test Accuracy of the 10,000 Test Images: {}%'.format(100 * correct / total))

