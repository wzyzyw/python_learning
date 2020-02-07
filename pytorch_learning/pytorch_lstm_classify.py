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


class Lstm(nn.Module):
    def __init__(self):
        super(Lstm, self).__init__()

        self.lstm=nn.LSTM(input_size=28,hidden_size=64,num_layers=1,batch_first=True)
        self.out=nn.Linear(64,10)




    def forward(self,x):
            r_out, (h_n, h_c) = self.lstm(x, None)   # None 表示 hidden state 会用全0的 state
            # 选取最后一个时间点的 r_out 输出
            # 这里 r_out[:, -1, :] 的值也是 h_n 的值
            out = self.out(r_out[:, -1, :])
            return out


if __name__=='__main__':
    hyper_para=model_hyper_para()
    lstm=Lstm()
    print('lstm:',lstm)
    train_loader,test_x,test_y=down_data(hyper_para)

    optimizer=torch.optim.Adam(lstm.parameters(),lr=hyper_para['Learning Rate'])
    loss_func=nn.CrossEntropyLoss()

    for epoch in range(hyper_para['Epoch']):
        for step,(x,b_y) in enumerate(train_loader):
            b_x = x.view(-1, 28, 28)  # reshape x to (batch, time_step, input_size)
            output=lstm(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    test_output = lstm(test_x[:10].view(-1, 28, 28))
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    print(pred_y, 'prediction number')
    print(test_y[:10].numpy(), 'real number')






