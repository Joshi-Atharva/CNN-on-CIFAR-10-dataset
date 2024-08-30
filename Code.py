'''
    Subject: Image classification on CIFAR10 dataset using CNN with pytorch
    Author: Atharva Joshi
    Date of creation: 30/08/2024
'''

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
num_epochs = 20
batch_size = 1024
learning_rate = 0.002

# dataset has PILImage images of range [0, 1].
# mind you that the dataset doesn't have values between 0 and 1 but rather between 0 and 255, with mean = 127.5 and almost normal curve (maybe wrong), but the dataloader normalizes the values (min max) to fit between 0 and 1
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.Grayscale(3), #throws an error in main program if 1 channel is given out
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


class CNN_improved(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=5,
            stride=1,
            padding=2, # p = (f-1)/2 hence this is same covolution
        )
        self.bn1 = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=2, # p = (f-1)/2 hence this is same covolution
        )
        self.bn2 =  nn.BatchNorm2d(16)
        
        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1, # p = (f-1)/2 hence this is same covolution
        )
        self.bn3 = nn.BatchNorm2d(32)
        
        self.fc1 = nn.Linear(32 * 4 * 4, 128 )
        self.bn4 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels = in_channels,
                    out_channels = 16, 
                    kernel_size = 1, # 1*1 conv
                    padding = 0,
                    stride = 2, # floor((32 + 2(0) - 1)/2) + 1 = 16
                    bias = False,
                ),
                nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = self.downsample(x) # Residual 
        
        x = F.relu(self.bn1(self.conv1(x))) # 5*5, same, output -> (32, 32, 8)
        x = self.pool(x) # out -> (16, 16, 8)
        
        x = F.relu(self.bn2(self.conv2(x)) + identity) # 5*5, same, out -> (16, 16, 16)
        x = self.pool(x) # out -> (8, 8, 16)
        x = F.relu(self.bn3(self.conv3(x))) # 3*3, same, out -> (8, 8, 32)
        x = self.pool(x) # out -> (4, 4, 32)
        x = x.reshape(x.shape[0], -1) # shape[0] is batch_size = 500 by convention (outermost/leading (1st) dimension denotes usually the highest level characteristic of data (in this case no. of training examples in x))
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# model performance metrics:
loss_list = [] # list containing loss of every updation step
training_accuracy = [] # list containing train accuracy after every updation step

avg_train_acc = 0 # scalar variable for aggregating accuracy values for each updation step and averaging over all steps in current epoch
avg_val_acc = 0 # similar aggregator/accumulator for validation accuracy
avg_loss = 0
val_epoch_acc = [] # list containing validation accuracy values for each epoch averaged over all updation steps in that epoch
train_epoch_acc = [] # similar list for training accuracy
loss_epoch_list = []

# initialising model and loading checkpoint
model = CNN_improved().to(device)
PATH = 'model_full.pt'
checkpoint = torch.load(PATH, weights_only = True)

# initialising loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas = (0.99, 0.999), weight_decay = 0.001)
prev_epochs = 0 # default value if no model is loaded

# laoding via state dictionaries
transfer_model = int(input("Transfer previous model? (1 for yes, 0 for no) "))
if(transfer_model):
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    prev_epochs = checkpoint['epoch']
    loss_epoch_list = checkpoint['loss_epoch_list']
    train_epoch_acc = checkpoint['train_epoch_acc']
    val_epoch_acc = checkpoint['val_epoch_acc']


def val_accuracy(model, val_loader):
    n_correct = 0
    n_samples = 0
    model.eval() # disabling dropout and some other training algorithms
    with torch.no_grad(): # disabling gradient calculation
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
    model.train() # again enabling dropout and other training algorithms



# Training:

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    if(epoch == num_epochs//2):
        # Saving the model for future (general checkpoint) after half no. epochs
        PATH = "model_half.pt"
        torch.save({
                    'loss_epoch_list': loss_epoch_list,
                    'val_epoch_acc': val_epoch_acc,
                    'train_epoch_acc': train_epoch_acc,
                    'epoch': prev_epochs + num_epochs//2,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, PATH)

    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        '''after change'''
        # origin shape: [500, 1, 32, 32]
        # input_layer: 1 input channels, 8 output channels, 5 kernel size
        '''throws error'''
        # hence input channels are again kept 3 (all grayscale maybe)
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

        # accumulation of train and val accuracy values for current updation step
        updation_step = epoch * n_total_steps + i # complete_epochs_over * n_total_steps + batch_index
        avg_val_acc = avg_val_acc + val_accuracy(model, val_loader)
        avg_train_acc = avg_train_acc + accuracy
        avg_loss = avg_loss + loss.item()

        # displaying current model performace
        if (i+1) % 10 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}, train-accuracy: {accuracy*100:.2f}%, val-accuracy: {val_accuracy(model, val_loader)}%')       

    # calculating avg over the current epoch from the accumulated sum
    avg_train_acc /= n_total_steps
    avg_train_acc = avg_train_acc*100
    avg_val_acc /= n_total_steps
    avg_loss /= n_total_steps
    val_epoch_acc.append(avg_val_acc)
    train_epoch_acc.append(avg_train_acc)
    loss_epoch_list.append(avg_loss)

    # re assignment for next epoch
    avg_train_acc = 0
    avg_val_acc = 0
    avg_loss = 0

    if(epoch == num_epochs//2):
        with torch.no_grad():
            model.eval() # eval block begin
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

                for i in range(labels.shape[0]):
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

            model.train() # eval block close (effectively)



plt.plot(loss_list)
plt.xlabel('Number of updation steps')
plt.ylabel('Loss')
plt.show()

plt.plot(training_accuracy)
plt.xlabel('Number of updation steps')
plt.ylabel('Training Accuracy')
plt.show()

# function to plot all aggregate curves
def plot_avgs(train_epoch_acc, val_epoch_acc, loss_epoch_list):
    plt.title("Loss averaged over each epoch")
    plt.plot(loss_epoch_list)
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.show()
    
    plt.title("Averaged over each epoch")
    plt.plot(train_epoch_acc)
    plt.plot(val_epoch_acc)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["Training accuracy", "Validation accuracy"])
    plt.show()
    
plot_avgs(train_epoch_acc, val_epoch_acc, loss_epoch_list)

print('Finished Training')

with torch.no_grad():
    model.eval()
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

        for i in range(labels.shape[0]):
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
    model.train()

# Saving the model for future (again, general checkpoint)
PATH = "model_full.pt"
torch.save({
            'loss_epoch_list': loss_epoch_list,
            'val_epoch_acc': val_epoch_acc,
            'train_epoch_acc': train_epoch_acc,
            'epoch': prev_epochs + num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, PATH)

print("model saved as 'model_full.pt'")
