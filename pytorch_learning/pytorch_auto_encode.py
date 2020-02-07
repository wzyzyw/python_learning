import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision      # 数据库模块
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
"""
3D 的可视化图
自编码有两个相反的神经网络组成（全链接层+激活函数），前者用于输入数据特征提取，输出压缩后的数据特征，称为编码器；
后者解压编码器的特征，恢复出输入数据，称为解码器；
整个网络的标签就是输入数据，是无监督的学习；
自编码器，编码器，解码器三者任一个都可以发挥重要作用

"""
torch.manual_seed(1)


def model_hyper_para(epoch=10,lr=0.005,batch_size=64,down_mnist=False,N_TEST_IMG = 5):
    hyper_para={}
    hyper_para['Epoch']=epoch
    hyper_para['Learning Rate']=lr
    hyper_para['Batch Size']=batch_size
    hyper_para['Down Mnist']=down_mnist
    hyper_para['Num Image']=N_TEST_IMG
    return hyper_para


def down_data(hyper_para):
    train_data=torchvision.datasets.MNIST(
        root='./data/mnist/',  # 保存或者提取位置
        train=True,  # this is training data
        transform=torchvision.transforms.ToTensor(),  # 转换 PIL.Image or numpy.ndarray 成
        # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
        download=hyper_para['Down Mnist'],  # 没下载就下载, 下载了就不用再下了
    )
    # plot one example
    print(train_data.train_data.size())  # (60000, 28, 28)
    print(train_data.train_labels.size())  # (60000)
    plt.imshow(train_data.train_data[2].numpy(), cmap='gray')
    plt.title('%i' % train_data.train_labels[2])
    plt.show()

    train_loader = Data.DataLoader(dataset=train_data, batch_size=hyper_para['Batch Size'], shuffle=True)

    return train_loader,train_data


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        """
        Conv2d各参数含义：输入高度，滤波器数量，滤波器大小，滤波器步长，
        如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
          # input shape (1, 28, 28)
          # output shape (16, 28, 28)
        MaxPool2d参数含义：在 2x2 空间里向下采样, output shape (16, 14, 14)
        """
        # 压缩
        self.encoder=nn.Sequential(nn.Linear(28*28,128),
                                 nn.Tanh(),
                                 nn.Linear(128,64),
                                 nn.Tanh(),
                                 nn.Linear(64,12),
                                 nn.Tanh(),
                                 nn.Linear(12,3))# 压缩成3个特征, 进行 3D 图像可视化
        self.decoder=nn.Sequential(
            nn.Linear(3,12),
            nn.Tanh(),
            nn.Linear(12,64),
            nn.Tanh(),
            nn.Linear(64,128),
            nn.Tanh(),
            nn.Linear(128,28*28),
            nn.Sigmoid()
        )

    def forward(self,x):
            encoded=self.encoder(x)
            decoded=self.decoder(encoded)
            return encoded,decoded


if __name__=='__main__':
    hyper_para=model_hyper_para()
    autoencoder=AutoEncoder()
    train_loader,train_data=down_data(hyper_para)

    optimizer=torch.optim.Adam(autoencoder.parameters(),lr=hyper_para['Learning Rate'])
    loss_func=nn.MSELoss()

    # initialize figure
    f, a = plt.subplots(2, hyper_para['Num Image'], figsize=(5, 2))
    plt.ion()  # continuously plot

    # original data (first row) for viewing
    view_data = train_data.train_data[:hyper_para['Num Image']].view(-1, 28 * 28).type(torch.FloatTensor) / 255.
    for i in range(hyper_para['Num Image']):
        a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray');
        a[0][i].set_xticks(());
        a[0][i].set_yticks(())

    for epoch in range(hyper_para['Epoch']):
        for step,(x,b_label) in enumerate(train_loader):
            b_x=x.view(-1,28*28)
            b_y=x.view(-1,28*28)
            encoded,decoded=autoencoder(b_x)
            loss = loss_func(decoded, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())

                # plotting decoded image (second row)
                _, decoded_data = autoencoder(view_data)
                for i in range(hyper_para['Num Image']):
                    a[1][i].clear()
                    a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap='gray')
                    a[1][i].set_xticks(());
                    a[1][i].set_yticks(())
                plt.draw();
                plt.pause(0.05)

    plt.ioff()
    plt.show()

    # visualize in 3D plot,test dataset
    view_data = train_data.train_data[:200].view(-1, 28 * 28).type(torch.FloatTensor) / 255.
    encoded_data, _ = autoencoder(view_data)
    fig = plt.figure(2);
    ax = Axes3D(fig)
    X, Y, Z = encoded_data.data[:, 0].numpy(), encoded_data.data[:, 1].numpy(), encoded_data.data[:, 2].numpy()
    values = train_data.train_labels[:200].numpy()
    for x, y, z, s in zip(X, Y, Z, values):
        c = cm.rainbow(int(255 * s / 9));
        ax.text(x, y, z, s, backgroundcolor=c)
    ax.set_xlim(X.min(), X.max());
    ax.set_ylim(Y.min(), Y.max());
    ax.set_zlim(Z.min(), Z.max())
    plt.show()






