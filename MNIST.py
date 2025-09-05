from tensorflow.keras.datasets import mnist
import torch.nn as nn
import torch.optim as optim
import torch

((train_images, train_labels),(test_images,test_labels)) = mnist.load_data()

train_vectors = torch.from_numpy(train_labels[:300]).long()

test_vectors = torch.from_numpy(test_labels[:20]).long()

train_images = torch.from_numpy(train_images[:300]).float()
test_images = torch.from_numpy(test_images[:20]).float()

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.cnn = nn.Sequential(
        nn.Conv2d(1, 32, 4),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, 4),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(1024,512),
        nn.ReLU(),
        nn.Linear(512, 10),
        )

    def forward(self,x):
        x = x.reshape(1,1,28,28)
        return self.cnn(x)

cnn = Model()

loss_func =  nn.CrossEntropyLoss()

optimizer = optim.Adam(cnn.parameters(), lr=0.001)

for epoch in range(100):
    total_loss = 0
    test_loss = 0
    for i in range(len(train_images)):
        optimizer.zero_grad()
        image = train_images[i]
        label = train_vectors[i].unsqueeze(0)
        output = cnn(image)
        loss = loss_func(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    for i in range(len(test_images)):
        image = test_images[i]
        label = test_vectors[i].unsqueeze(0)
        output = cnn(image)
        loss = loss_func(output, label)
        test_loss += loss.item()
    print("At epoch", epoch)
    print('loss is ', total_loss)
    print('test loss is ', test_loss)

output = cnn(test_images[1])
print(output,test_labels[1])