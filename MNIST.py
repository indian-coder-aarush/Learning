from tensorflow.keras.datasets import mnist
import torch.nn as nn
import torch.optim as optim
import torch

((train_images, train_labels),(test_images,test_labels)) = mnist.load_data()

train_vectors = torch.from_numpy(train_labels[:10000]).long()

test_vectors = torch.from_numpy(test_labels[:200]).long()

train_images = torch.from_numpy(train_images[:10000]/255).float()
test_images = torch.from_numpy(test_images[:200]/255).float()

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
        x = x.reshape(-1,1,28,28)
        return self.cnn(x)

cnn = Model()

loss_func =  nn.CrossEntropyLoss()

optimizer = optim.Adam(cnn.parameters(), lr=0.001)

for epoch in range(30):
    total_loss = 0
    test_loss = 0
    for i in range(int(len(train_images)/100)):
        optimizer.zero_grad()
        image = train_images[100*i:100*i+100]
        label = train_vectors[100*i:100*i+100]
        output = cnn(image)
        loss = loss_func(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()/100
    for i in range(int(len(test_images)/100)):
        image = test_images[100*i:100*i+100]
        label = test_vectors[100*i:100*i+100]
        output = cnn(image)
        loss = loss_func(output, label)
        test_loss += loss.item()/2
    print("At epoch", epoch)
    print('loss is ', total_loss)
    print('test loss is ', test_loss)

output = cnn(test_images)
print(torch.argmax(nn.functional.softmax(output,dim = 0),dim = 1),test_labels)