import torch
from torch import nn
import random
import tensorboard

def gen_data(w, b, num_samples):
    X = torch.normal(0,1,(num_samples, len(w)))
    y = torch.matmul(X,w) + b
    y += torch.normal(0,0.005,y.shape)

    return X, y.reshape(-1,1)


def data_iter(batch_size, feature, label):
    n_sample = len(feature)
    ind = list(range(n_sample))

    random.shuffle(ind)
    for i in range(0,n_sample,batch_size):
        batch_ind = torch.tensor(ind[i:min(i+batch_size,n_sample)])
        yield feature[batch_ind], label[batch_ind]

# y = w*x + b
def linModel(X,w,b):
    return torch.matmul(X,w)+b

def mse(y_hat,y):
    return torch.mean((y_hat - y.reshape(y_hat.shape)) ** 2, 1).sum()
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


if __name__ == '__main__':
    # g_w = torch.tensor([1.1, 2.2, 3.3, 4.4, 5.5])
    g_w = torch.rand([5])
    print('GT G_w:',g_w)
    #
    g_b = torch.rand([1])
    features, labels  = gen_data(g_w,g_b, 20000)

    batch_size = 2000

    # loss = nn.MSEloss()

    w = torch.normal(0, 0.01, size=(5,1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    lr = 0.03
    num_epochs = 1000
    net = linModel
    # loss = mse
    loss = mse
    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)
            l.backward()
            sgd([w, b], lr, batch_size)

        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

        print(f'GT G_w:{g_w}')
        print(f'current w:{w.reshape(g_w.shape)}')
        print(f'GT b:{g_b}')
        print(f'current b:{b}')

        print("#"*30)







