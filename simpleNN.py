import torch.nn as nn
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,2),
        )

    def forward(self, x):
        return self.model(x)

import numpy as np
np.random.seed(42)
n_samples = 500
X = np.random.normal(0, 1, size=(n_samples, 3))
logits = 1.5 * X[:, 0] - 2.0 * X[:, 1] + 0.5 * X[:, 2]
probs = 1 / (1 + np.exp(-logits))
y = (probs > 0.5).astype(int)

model = Model()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

X = torch.from_numpy(X).float()
y = torch.from_numpy(y).long()

print(X)
print(y)

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

for i in range(100):
    y = torch.tensor(y)
    optimizer.zero_grad()
    y_pred = model.forward(X)
    loss = criterion(torch.softmax(y_pred, dim = 1), torch.tensor([[0.0,1.0] if y[j] == 1 else [1.0,0.0] for j in range(len(y))]))
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
