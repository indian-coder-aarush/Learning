from tensorflow.keras.datasets import mnist
import torch.nn as nn
import torch.optim as optim

((train_images, train_labels),(test_images,test_labels)) = mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.convolution1 = nn.Conv2d(1, 32, 4)
        self.pooling1 = nn.MaxPool2d(2, 2)
        self.convolution2 = nn.Conv2d(32, 64, 4)
        self.pooling2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()


print(type(train_images))