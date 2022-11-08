import enum
import sys

sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    linear1 = nn.Linear(in_features=dim, out_features=hidden_dim)
    norm1 = norm(dim=hidden_dim)
    relu1 = nn.ReLU()
    dropout = nn.Dropout(p=drop_prob)
    linear2 = nn.Linear(in_features=hidden_dim, out_features=dim)
    norm2 = norm(dim=dim)
    
    fn = nn.Sequential(linear1,
                       norm1,
                       relu1,
                       dropout,
                       linear2,
                       norm2)
    
    residual = nn.Residual(fn)
    relu2 = nn.ReLU()
    block = nn.Sequential(residual,
                          relu2)
    return block
    ### END YOUR SOLUTION

def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    linear1 = nn.Linear(in_features=dim, out_features=hidden_dim)
    relu1 = nn.ReLU()
    residuals = []
    for i in range(num_blocks):
      res = ResidualBlock(dim=hidden_dim, hidden_dim=hidden_dim//2, norm=norm, drop_prob=drop_prob)
      residuals.append(res)
    linear2 = nn.Linear(in_features=hidden_dim, out_features=num_classes)
    block = nn.Sequential(linear1,
                          relu1,
                          *residuals,
                          linear2)
    return block
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt:
      model.train()
    else:
      model.eval()

    loss_fn = nn.SoftmaxLoss()
    losses = []
    acces = []
    correct = 0.0
    num_samples = 0.0
    for batch in dataloader:
      x, y = batch[0], batch[1]
      x = x.reshape((x.shape[0], -1))
      # print(x.shape)
      # print(y.shape)

      y_hat = model(x)
      loss = loss_fn(y_hat, y)
      losses.append(loss.cached_data)

      label_hat = np.argmax(y_hat.cached_data, axis=1)
      correct += np.sum(label_hat == y.cached_data)
      num_samples += x.shape[0]

      if opt:
        opt.reset_grad()
        loss.backward()
        opt.step()
    
    acc = correct / num_samples
    loss_ave = np.mean(losses)
    return 1 - acc, loss_ave
    ### END YOUR SOLUTION


def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    mnist_train_dataset = ndl.data.MNISTDataset("data/train-images-idx3-ubyte.gz",
                                                "data/train-labels-idx1-ubyte.gz")
    train_dataloader = ndl.data.DataLoader(dataset=mnist_train_dataset, batch_size=batch_size, shuffle=True)

    mnist_test_dataset = ndl.data.MNISTDataset("data/t10k-images-idx3-ubyte.gz",
                                               "data/t10k-labels-idx1-ubyte.gz")
    test_dataloader = ndl.data.DataLoader(dataset=mnist_test_dataset, batch_size=batch_size, shuffle=False)

    resnet = MLPResNet(784, hidden_dim)

    opt = optimizer(resnet.parameters(), lr=lr, weight_decay=weight_decay)
    print("epochs = ", epochs)
    for i in range(epochs):
      print("i = ", i)
      train_acc, train_loss = epoch(dataloader=train_dataloader, model=resnet, opt=opt)
      test_acc, test_loss = epoch(dataloader=test_dataloader, model=resnet)

    return train_acc, train_loss, test_acc, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")