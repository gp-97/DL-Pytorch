# !/usr/bin/python3.6.9
import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

np.random.seed(101)
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device("cpu")

X = np.random.rand(100,10)
y = np.random.randint(0,2,size=(100,1))
X = (X - np.mean(X))/(np.max(X)-np.min(X))
class SimpleNN(torch.nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, 1)
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = torch.sigmoid(self.fc3(X))
        return X

net = SimpleNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters())

train_loss = []
epochs = []

X = torch.tensor(X, dtype = torch.float32, device = device)
y = torch.tensor(y, dtype = torch.float32, device = device)

for i in range(12000):
    optimizer.zero_grad()
    epochs.append(i)
    output = net(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    train_loss.append(loss.item())
    
    if i%1000 == 0:
        print(f"Epoch {i}: training loss: {loss:0.4f}")

plt.plot(epochs, train_loss)
plt.show()
