import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)    # reproducible
"""
这个LSTM的回归问题是使用sin序列预测cos序列，换句话说，输入是sin，输出是cos
matplotlib两种模式具体的区别
在交互模式下：

1、plt.plot(x)或plt.imshow(x)是直接出图像，不需要plt.show()

2、如果在脚本中使用ion()命令开启了交互模式，没有使用ioff()关闭的话，则图像会一闪而过，并不会常留。要想防止这种情况，需要在plt.show()之前加上ioff()命令。

在阻塞模式下：

1、打开一个窗口以后必须关掉才能打开下一个新的窗口。这种情况下，默认是不能像Matlab一样同时开很多窗口进行对比的。

2、plt.plot(x)或plt.imshow(x)是直接出图像，需要plt.show()后才能显示图像
但程序会停止在show那里，关闭当前图像后程序才能继续执行

原文链接：https://blog.csdn.net/zbrwhut/article/details/80625702
还有就是matplotlib库多次在同一图上显示，不会清楚前面的图像，不像matlab需要hold on。
"""
def model_hyper_para(epoch=1,lr=0.02,batch_size=50,down_mnist=False,input_size=1,time_step=10):
    hyper_para={}
    hyper_para['Epoch']=epoch
    hyper_para['Learning Rate']=lr
    hyper_para['Batch Size']=batch_size
    hyper_para['Down Mnist']=down_mnist
    hyper_para['Input Size']=input_size
    hyper_para['Time Step']=time_step
    return hyper_para

class Lstm(nn.Module):
    def __init__(self):
        super(Lstm,self).__init__()

        self.lstm=nn.LSTM(input_size=1,
            hidden_size=32,     # rnn hidden unit
            num_layers=1,       # 有几层 RNN layers
            batch_first=True)
        self.out=nn.Linear(32, 1)
        self.first=True


    def forward(self,x,h_state,c_state):
        if self.first:
            self.first=False
            r_out, (h_state, c_state) = self.lstm(x, None)
        else:
            r_out, (h_state,c_state) = self.lstm(x, (h_state,c_state))
        outs = []  # 保存所有时间点的预测值
        for time_step in range(r_out.size(1)):  # 对每一个时间点计算 output
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state,c_state
    """
    forward另一种形式，上述形式是为了体现pytorch的动态优势
    r_out = r_out.view(-1, 32)
    outs = self.out(r_out)
    return outs.view(-1, 32, TIME_STEP), h_state
    """


if __name__=='__main__':
    hyper_para=model_hyper_para()
    lstm=Lstm()
    print('lstm:',lstm)

    optimizer = torch.optim.Adam(lstm.parameters(), lr=hyper_para['Learning Rate'])  # optimize all rnn parameters
    loss_func = nn.MSELoss()

    h_state=None
    c_state=None

    plt.figure(1, figsize=(12, 5))
    plt.ion()  # continuously plot

    for step in range(100):
        start,end=step*np.pi,(step+1)*np.pi
        # sin 预测 cos
        steps=np.linspace(start,end,hyper_para['Time Step'],dtype=np.float32)
        x_np=np.sin(steps)
        y_np=np.cos(steps)

        x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])  # shape (batch, time_step, input_size)
        y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

        prediction, h_state,c_state = lstm(x, h_state,c_state)
        # !!  下一步十分重要 !!
        h_state = h_state.data  # 要把 h_state 重新包装一下才能放入下一个 iteration, 不然会报错
        c_state = c_state.data

        loss = loss_func(prediction, y)
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        # plotting
        plt.plot(steps, y_np.flatten(), 'r-')
        plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
        # plt.draw()
        plt.pause(0.05)

    plt.ioff()
    plt.show()

