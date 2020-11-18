import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim

def get_digits(df):
    """Loads images as PyTorch tensors"""
    # Load the labels if they exist 
    # (they wont for the testing data)
    labels = []
    start_inx = 0
    if 'label' in df.columns:
        labels = [v for v in df.label.values]
        start_inx = 1
        
    # Load the digit information
    digits = []
    for i in range(df.pixel0.size):
        digit = df.iloc[i].astype(float).values[start_inx:]
        digit = np.reshape(digit, (28,28))
        digit = transform(digit).type('torch.FloatTensor')
        if len(labels) > 0:
            digits.append([digit, labels[i]])
        else:
            digits.append(digit)

    return digits

def calc_out(in_layers, stride, padding, kernel_size, pool_stride):
    """
    Helper function for computing the number of outputs from a
    conv layer
    """
    return int((1+(in_layers - kernel_size + (2*padding))/stride)/pool_stride)

# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Some helpful values
        inputs      = [1,32,64,64]
        kernel_size = [5,5,3]
        stride      = [1,1,1]
        pool_stride = [2,2,2]

        # Layer lists
        layers = []

        self.out   = 28
        self.depth = inputs[-1]
        for i in range(len(kernel_size)):
            # Get some variables
            padding = int(kernel_size[i]/2)

            # Define the output from this layer
            self.out = calc_out(self.out, stride[i], padding,
                                kernel_size[i], pool_stride[i])

            # convolutional layer 1
            layers.append(nn.Conv2d(inputs[i], inputs[i+1], kernel_size[i], 
                                       stride=stride[i], padding=padding))
            layers.append(nn.ReLU())
            
            # convolutional layer 2
            layers.append(nn.Conv2d(inputs[i+1], inputs[i+1], kernel_size[i], 
                                       stride=stride[i], padding=padding))
            layers.append(nn.ReLU())
            # maxpool layer
            layers.append(nn.MaxPool2d(pool_stride[i],pool_stride[i]))
            layers.append(nn.Dropout(p=0.2))

        self.cnn_layers = nn.Sequential(*layers)
        
        print(self.depth*self.out*self.out)
        
        # Now for our fully connected layers
        layers2 = []
        layers2.append(nn.Dropout(p=0.2))
        layers2.append(nn.Linear(self.depth*self.out*self.out, 512))
        layers2.append(nn.Dropout(p=0.2))
        layers2.append(nn.Linear(512, 256))
        layers2.append(nn.Dropout(p=0.2))
        layers2.append(nn.Linear(256, 256))
        layers2.append(nn.Dropout(p=0.2))
        layers2.append(nn.Linear(256, 10))

        self.fc_layers = nn.Sequential(*layers2)

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(-1, self.depth*self.out*self.out)
        x = self.fc_layers(x)
        return x



train = pd.read_csv("./bin/practicals/Kaggle-Digit-Recognizer-master/train.csv")
test = pd.read_csv("./bin/practicals/Kaggle-Digit-Recognizer-master/test.csv")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
    ])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_X = get_digits(train)

# Some configuration parameters
num_workers = 0    # number of subprocesses to use for data loading
batch_size  = 64   # how many samples per batch to load
valid_size  = 0.2  # percentage of training set to use as validation

# Obtain training indices that will be used for validation
num_train = len(train_X)
indices   = list(range(num_train))
np.random.shuffle(indices)
split     = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# Define samplers for obtaining training and validation batches
from torch.utils.data.sampler import SubsetRandomSampler
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# Construct the data loaders
train_loader = torch.utils.data.DataLoader(train_X, batch_size=batch_size,
                    sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_X, batch_size=batch_size, 
                    sampler=valid_sampler, num_workers=num_workers)

# Test the size and shape of the output
dataiter = iter(train_loader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)
    
# create a complete CNN
model = Net()

# specify loss function
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# number of epochs to train the model
n_epochs = 25 # you may increase this number to train a final model

valid_loss_min = np.Inf # track change in validation loss

print(device)
model.to(device)
tLoss, vLoss = [], []

for epoch in range(n_epochs):
    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    
    #########
    # train #
    #########
    model.train()
    for data, target in train_loader:
        # move tensors to GPU if CUDA is available
        data   = data.to(device)
        target = target.to(device)
        
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*data.size(0)
        
    ############
    # validate #
    ############
    model.eval()
    for data, target in valid_loader:
        # move tensors to GPU if CUDA is available
        data   = data.to(device)
        target = target.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss 
        valid_loss += loss.item()*data.size(0)
        
    # calculate average losses
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loader.dataset)
    tLoss.append(train_loss)
    vLoss.append(valid_loss)
    
    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
        torch.save(model.state_dict(), 'model_cifar.pt')
        valid_loss_min = valid_loss
