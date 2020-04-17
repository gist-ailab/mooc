from glob import glob
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision.models.vgg import vgg16

## Load the pre-trained model
model = vgg16(pretrained=True)
print(model.modules) # Check the model architecture

num_classes = 4
model.classifier[6] = nn.Linear(4096, num_classes)

## Hyper-parameter
num_epochs, batch_size, learning_rate = 10, 64, 0.001

## Custom Dataset & DataLoader
class FoodDataset(Dataset):
    def __init__(self, file_names, transform=None):
        self.img_list = glob(file_names)
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_name = self.img_list[index]
        img = Image.open(img_name)
        # Transform
        if self.transform is not None:
            img = self.transform(img)

        label_name = img_name.split('\\')[-2]
        if label_name == '김치':
            label = 0
        elif label_name == '깍두기':
            label = 1
        elif label_name == '닭갈비':
            label = 2
        else:
            label = 3

        # To change the label into the tensor
        label = torch.tensor(label).long()
        return img, label

# Transform
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor()])

aug_transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.RandomHorizontalFlip(),
                                transforms.ToTensor()])

# File Path (Train/Test)
train_filenames = './food/train/*/*.jpg'
test_filenames = './food/test/*/*.jpg'

# Dataset
train_dataset = FoodDataset(file_names=train_filenames, transform=transform)
test_dataset = FoodDataset(file_names=test_filenames, transform=transform)

# DataLoader
train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=batch_size)

## Get the sample data
index = 0
# Get the one example of data for the index
img_ex, label_ex = train_dataset.__getitem__(index)

# Show the image
plt.cla()
plt.imshow(img_ex.permute([1,2,0]))
plt.title('sample image')
plt.pause(1)
# Print the label of the image
print('Example Label :', label_ex)

## Training
model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        if (i + 1) % 25 == 0:
            print('Epoch: {}/{}, Batch Step: {}/{}, Loss: {:.4f}, Training Accuracy of the Current Batch: {}%'.
                  format(epoch + 1, num_epochs, i + 1, train_loader.__len__(), loss.item(), 100 * correct / batch_size))

## Test
model.eval()
with torch.no_grad():
    total, correct  = 0, 0
    for images, labels in test_loader:
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Test Accuracy of the Test Images: {:.2f}%'.format(100 * correct / total))