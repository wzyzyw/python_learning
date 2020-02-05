import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision      # 数据库模块
import matplotlib.pyplot as plt

"""
下面代码设计随机初始化种子，神经网络采用随机初始化，
为了结果能够复现，保证每次随机初始化的结果相同
需要使用同一个随机数种子

"""
torch.manual_seed(1)
def model_hyper_para(epoch=1,lr=0.001,batch_size=50,down_mnist=False):
    hyper_para={}
    hyper_para['Epoch']=epoch
    hyper_para['Learning Rate']=lr
    hyper_para['Batch Size']=batch_size
    hyper_para['Down Mnist']=down_mnist
    return hyper_para


def down_data(hyper_para):
    train_data=torchvision.datasets.MNIST(
        root='./data/mnist/',  # 保存或者提取位置
        train=True,  # this is training data
        transform=torchvision.transforms.ToTensor(),  # 转换 PIL.Image or numpy.ndarray 成
        # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
        download=hyper_para['Down Mnist'],  # 没下载就下载, 下载了就不用再下了
    )
    test_data = torchvision.datasets.MNIST(root='./data/mnist/', train=False)

    # 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=hyper_para['Batch Size'], shuffle=True)

    # 为了节约时间, 我们测试时只测试前2000个
    test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[
             :2000] / 255.  # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
    test_y = test_data.test_labels[:2000]
    return train_loader,test_x,test_y


class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        """
        Conv2d各参数含义：输入高度，滤波器数量，滤波器大小，滤波器步长，
        如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
          # input shape (1, 28, 28)
          # output shape (16, 28, 28)
        MaxPool2d参数含义：在 2x2 空间里向下采样, output shape (16, 14, 14)
        """
        self.conv1=nn.Sequential(nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5,stride=1,padding=2),
                                 nn.ReLU(),
                                 nn.MaxPool2d(kernel_size=2))
        """
        Conv2d: # input shape (16, 14, 14), # output shape (32, 14, 14)
        MaxPool:  output shape (32, 7, 7)
        """
        self.conv2=nn.Sequential(nn.Conv2d(16, 32, 5, 1, 2),
                                 nn.ReLU(),
                                 nn.MaxPool2d(2))
        self.out = nn.Linear(32 * 7 * 7, 10)  # fully connected layer, output 10 classes

        def forward(self,x):
            x=self.conv1(x)
            x=self.conv2(x)
            x = x.view(x.size(0), -1)  # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
            output=self.out(x)
            return output


if __name__=='__main__':
    hyper_para=model_hyper_para(down_mnist=True)
    cnn=Cnn()
    train_loader,test_x,test_y=down_data(hyper_para)

    optimizer=torch.optim.adam(cnn.parameters(),lr=hyper_para['Learning Rate'])
    loss_func=nn.CrossEntropyLoss()

    for epoch in range(hyper_para['Epoch']):
        for step,(b_x,b_y) in enumerate(train_loader):
            output=cnn(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    test_output = cnn(test_x[:10])
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    print(pred_y, 'prediction number')
    print(test_y[:10].numpy(), 'real number')






