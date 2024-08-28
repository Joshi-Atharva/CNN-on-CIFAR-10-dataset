import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 10
batch_size = 500
learning_rate = 0.001

# dataset has PILImage images of range [0, 1].
# mind you that the dataset doesn't have values between 0 and 1 but rather between 0 and 255, with mean = 127.5 and almost normal curve (maybe wrong), but the dataloader normalizes the values (min max) to fit between 0 and 1
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.Grayscale(),
     transforms.RandomHorizontalFlip(p = 0.5),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# NOTE that sequence of transforms is important

# CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
# setting seed
torch.manual_seed(43)

# validation set size
val_size = 5000
train_size = len(train_dataset) - val_size

# splitting the training set into training and validation set
train_data, val_data = torch.utils.data.random_split(train_dataset, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                          shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()


# # get some random training images
# dataiter = iter(train_loader)
# images, labels = next(dataiter)


# # show images
# imshow(torchvision.utils.make_grid(images))
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=5,
            stride=1,
            padding=2, # p = (f-1)/2 hence this is same covolution
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=2, # p = (f-1)/2 hence this is same covolution
        )

        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1, # p = (f-1)/2 hence this is same covolution
        )
        self.fc1 = nn.Linear(32 * 4 * 4, 128 )
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # 5*5, same, output -> (32, 32, 8)
        x = self.pool(x) # out -> (16, 16, 8)
        x = F.relu(self.conv2(x)) # 5*5, same, out -> (16, 16, 16)
        x = self.pool(x) # out -> (8, 8, 16)
        x = F.relu(self.conv3(x)) # 3*3, same, out -> (8, 8, 32)
        x = self.pool(x) # out -> (4, 4, 32)
        x = x.reshape(x.shape[0], -1) # shape[0] is batch_size = 500 by convention (outermost/leading (1st) dimension denotes usually the highest level characteristic of data (in this case no. of training examples in x))
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 2 * 2, 96)
        self.fc2 = nn.Linear(96, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
                                              # -> n, 3, 32, 32
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 14, 14 (conv1 -> n, 6, 28, 28)
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 6, 6  (conv2 -> n, 16, 12, 12)
        x = self.pool(F.relu(self.conv3(x)))  # -> n, 32, 2, 2  (conv3 -> n, 32, 4, 4)
        x = x.view(-1, 32 * 2 * 2)            # -> n, 128
        x = F.relu(self.fc1(x))               # -> n, 96
        x = self.dropout(x)                   # -> n, 96
        x = F.relu(self.fc2(x))               # -> n, 64
        x = self.dropout(x)                   # -> n, 64
        x = F.relu(self.fc3(x))               # -> n, 32
        x = self.fc4(x)                       # -> n, 10
        return x


model = CNN().to(device)
def val_accuracy(model, val_loader):
    n_correct = 0
    n_samples = 0
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        return acc



criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas = (0.97, 0.999), weight_decay = 0.001)
loss_list = []
training_accuracy = []
avg_train_acc = 0
avg_val_acc = 0
flag = 0
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        '''after change'''
        # origin shape: [500, 1, 32, 32] 
        # input_layer: 1 input channels, 8 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        _, args = torch.max(outputs, 1)
        accuracy = (args == labels).sum().item() / batch_size
        training_accuracy.append(accuracy)

        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_val_acc = avg_val_acc + val_accuracy(model, val_loader)

        updation_step = epoch * n_total_steps + i # complete_epochs_over * n_total_steps + batch_index

        if (i+1) % 20 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}, train-accuracy: {training_accuracy[updation_step]*100:.2f}%, val-accuracy: {val_accuracy(model, val_loader)}%')

        if (i+1)  == 90 and epoch > 5:
            avg_train_acc = sum(training_accuracy[updation_step-89:updation_step+1])/90
            avg_val_acc = avg_val_acc/90
            if ((avg_train_acc*100 - avg_val_acc) > 10) and (flag == 0) :
                torch.save({'model': model.state_dict(), 'optim': optim.state_dict()}, '...')
                state_dict = torch.load('...')
                model.load_state_dict(state_dict['model'])
                optim = torch.optim.Adam(model.paramters(), learning_rate, betas = (0.97, 0.999), weight_decay = 0.01)
                flag = 1
                print("switched to new optimizer")
            avg_train_acc = 0
            avg_val_acc = 0








plt.plot(loss_list)
plt.xlabel('Number of updation steps')
plt.ylabel('Loss')
plt.show()

plt.plot(training_accuracy)
plt.xlabel('Number of updation steps')
plt.ylabel('Training Accuracy')
plt.show()

print('Finished Training')
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')