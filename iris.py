import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
sns.set_style('whitegrid')

np.random.seed(101)

dataset = load_iris()
sc = MinMaxScaler()
X = np.array(dataset.data)
X = sc.fit_transform(X)
y= np.array(dataset.target)
y = np.reshape(y, (y.shape[0],1))
data = np.hstack([X,y])

np.random.shuffle(data)

X = data[:,:-1]
y = data[:,-1]
y = np.reshape(y,(y.shape[0],1))
enc = OneHotEncoder()
y = enc.fit_transform(y).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=  0.20, random_state = 42)

device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
X_train = torch.tensor(X_train, dtype = torch.float32, device = device)
X_test = torch.tensor(X_test, dtype = torch.float32, device = device)
y_train = torch.tensor(y_train, dtype = torch.float32, device = device)
y_test = torch.tensor(y_test, dtype = torch.float32, device = device)
print(f"Running model on {torch.cuda.get_device_name(device = device)}")

class NNModel(nn.Module):
    def __init__(self, inp, h1, h2, h3, op):
        super(NNModel, self).__init__()
        self.inp = inp
        self.h1 = h1
        self.h2 = h2
        self.h3 = h3
        self.op = op
        self.fc1 = nn.Linear(self.inp, self.h1)
        self.fc2 = nn.Linear(self.h1, self.h2)
        self.fc3 = nn.Linear(self.h2, self.h3)
        self.fc4 = nn.Linear(self.h3, self.op)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

net = NNModel(X_train.shape[1], 5, 5, 5, 3).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters())

train_loss = []
val_loss = []
epochs = []
num_epochs = 3000

for i in range(num_epochs+1):
    optimizer.zero_grad()
    op_train = net(X_train)
    op_val = net(X_test)
    loss_train = criterion(op_train, y_train)
    loss_val = criterion(op_val, y_test)
    train_loss.append(loss_train)
    val_loss.append(loss_val)
    epochs.append(i)
    loss_train.backward()
    optimizer.step()

    if i%10==0 or i==1 and i!=0 or i==(num_epochs-1):
        print(f"Epoch: {i},\ttrain_loss: {train_loss[i]:0.4f},\tval_loss: {val_loss[i]:0.4f}")

y_train_pred = net(X_train)
y_val_pred = net(X_test)

y_train = y_train.cpu()
y_train = y_train.numpy()
y_train = enc.inverse_transform(y_train)

y_test = y_test.cpu()
y_test = y_test.numpy()
y_test = enc.inverse_transform(y_test)

y_train_pred = y_train_pred.cpu().detach()
y_train_pred = y_train_pred.numpy()
y_train_pred = np.round(y_train_pred)
y_train_pred = enc.inverse_transform(y_train_pred)

y_val_pred = y_val_pred.cpu().detach()
y_val_pred = y_val_pred.numpy()
y_val_pred = np.round(y_val_pred)
y_val_pred = enc.inverse_transform(y_val_pred)

print("Training:")
print(classification_report(y_train, y_train_pred))
print(confusion_matrix(y_train, y_train_pred))
print()

print("Validation:")
print(classification_report(y_test, y_val_pred))
print(confusion_matrix(y_test, y_val_pred))

plt.plot(epochs, train_loss, label = "train_loss")
plt.plot(epochs, val_loss, label = "val_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs Epochs")
plt.legend()
plt.show()



