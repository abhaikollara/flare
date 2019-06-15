import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np

import time
import flare
from flare import Trainer
from flare.data import FlareDataset

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
    
        return F.log_softmax(x, dim=1)
    
train_data = np.load("./data/mnist_train.npz")
train_x, train_y = train_data['X_train'], train_data['Y_train']

train_x = np.expand_dims(train_x, 1).astype('float32')
train_y = train_y.reshape(-1)

test_data = np.load("./data/mnist_test.npz")
test_x, test_y = test_data['X_test'], test_data['Y_test']

test_x = np.expand_dims(test_x, 1).astype('float32')
test_y = train_y.reshape(-1)

model = Net()
trainer = Trainer(model, F.nll_loss, Adam(model.parameters()))
trainer.train(train_x, train_y, batch_size=64)
