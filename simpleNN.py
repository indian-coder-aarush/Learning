import torch.nn as nn
import torch
import matplotlib
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(24,30),
            nn.ReLU(),
            nn.Linear(30,30),
            nn.ReLU(),
            nn.Linear(30,2),
        )

    def forward(self, x):
        return self.model(x)

import numpy as np
df = pd.read_excel('credit-card/default of credit card clients.xls',header = 1)

model = Model()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

X = torch.from_numpy(df.drop(columns='default payment next month').values).float()
y = torch.from_numpy(df['default payment next month'].values).long()

criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, sum(y == 0)/sum(y == 1)]))

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X = scaler.fit_transform(X.numpy())
X = torch.from_numpy(X).float()

print(X)

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

for i in range(100):
    y = torch.tensor(y)
    optimizer.zero_grad()
    y_pred = model.forward(X)
    loss = criterion(y_pred, y)
    print(loss.item())
    loss.backward()
    optimizer.step()
    y_pred = torch.argmax(y_pred, dim=1)
    y_pred = y_pred.numpy()
    y = y.numpy()

    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    acc = accuracy_score(y, y_pred)

    print('accuracy:', acc)
    print('precision:', prec)
    print('recall:', rec)
    print('f1:', f1)

y_pred = model.forward(X)
