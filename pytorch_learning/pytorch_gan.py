import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision      # 数据库模块
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
"""
GAN:包括两个网络，一个是生成器（新手画家），一个是判别器（新手评论家）
生成器输入数据是随机噪声，输出画（实际任务中的目标），
判别器会使用两次，每次输入分别是生成器的画和实际画家的画（实际标签），输出对应的概率
整个GAN根据概率对生成器和判别器进行优化，使得判别器正确分别两种画，从而使的生成器能够画出更接近实际画家的画，从而欺骗判别器。
关键 LOSS的理解
G 首先会有些灵感, G_ideas 就会拿到这些随机灵感 (可以是正态分布的随机数), 
然后 G 会根据这些灵感画画. 接着我们拿着著名画家的画和 G 的画, 让 D 来判定这两批画作是著名画家画的概率
然后计算有多少来自画家的画猜对了, 有多少来自 G 的画猜对了, 我们想最大化这些猜对的次数. 这也就是 log(D(x)) + log(1-D(G(z)) 
在论文中的形式. 而因为 torch 中提升参数的形式是最小化误差, 那我们把最大化 score 转换成最小化 loss, 
在两个 score 的合的地方加一个符号就好. 而 G 的提升就是要减小 D 猜测 G 生成数据的正确率, 也就是减小 D_score1
"""
torch.manual_seed(1)
np.random.seed(1)

def model_hyper_para(epoch=10,LR_G=0.001,LR_D=0.001,batch_size=64,N_IDEAS = 5,ART_COMPONENTS=15):
    hyper_para={}
    hyper_para['Epoch']=epoch
    hyper_para['Learning Rate_G']=LR_G # learning rate for generator
    hyper_para['Learning Rate_D'] = LR_D# learning rate for discriminator
    hyper_para['Batch Size']=batch_size
    hyper_para['N_IDEAS']=N_IDEAS # think of this as number of ideas for generating an art work (Generator)
    hyper_para['ART_COMPONENTS']=ART_COMPONENTS # it could be total point G can draw in the canvas
    PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(batch_size)])
    hyper_para['PAINT_POINTS'] = PAINT_POINTS
    return hyper_para


def artist_works(hyper_meter): # painting from the famous artist (real target)
    #绘制实际的二次函数曲线，相当于标签，batch 条不同的一元二次方程曲线
    a= np.random.uniform(1, 2, size=hyper_para['Batch Size'])[:, np.newaxis]
    paintings = a * np.power(hyper_para['PAINT_POINTS'], 2) + (a - 1)
    paintings = torch.from_numpy(paintings).float()
    return paintings




if __name__=='__main__':
    hyper_para=model_hyper_para()
    G=nn.Sequential(    # Generator
        nn.Linear(hyper_para['N_IDEAS'],128),  # random ideas (could from normal distribution)
        nn.ReLU(),
        nn.Linear(128,hyper_para['ART_COMPONENTS'])# making a painting from these random ideas
    )
    D=nn.Sequential(   # Discriminator
        nn.Linear(hyper_para['ART_COMPONENTS'],128), # receive art work either from the famous artist or a newbie like G
        nn.ReLU(),
        nn.Linear(128,1),# tell the probability that the art work is made by artist
        nn.Sigmoid()
    )
    opt_D = torch.optim.Adam(D.parameters(), lr=hyper_para['Learning Rate_D'])
    opt_G = torch.optim.Adam(G.parameters(), lr=hyper_para['Learning Rate_G'])
    plt.ion()  # something about continuous plotting
    for step in range(10000):
        artist_paintings = artist_works(hyper_para)  # real painting from artist
        G_ideas = torch.randn(hyper_para['Batch Size'], hyper_para['N_IDEAS'])  # random ideas
        G_paintings = G(G_ideas)  # fake painting from G (random ideas)

        prob_artist0 = D(artist_paintings)  # D try to increase this prob
        prob_artist1 = D(G_paintings)  # D try to reduce this prob

        D_loss = - torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))
        G_loss = torch.mean(torch.log(1. - prob_artist1))

        opt_D.zero_grad()
        D_loss.backward(retain_graph=True)  # retain_graph 这个参数是为了再次使用计算图纸
        opt_D.step()

        opt_G.zero_grad()
        G_loss.backward()
        opt_G.step()

        if step % 50 == 0:  # plotting
            plt.cla()
            plt.plot(hyper_para['PAINT_POINTS'][0], G_paintings.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting', )
            plt.plot(hyper_para['PAINT_POINTS'][0], 2 * np.power(hyper_para['PAINT_POINTS'][0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
            plt.plot(hyper_para['PAINT_POINTS'][0], 1 * np.power(hyper_para['PAINT_POINTS'][0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
            plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % prob_artist0.data.numpy().mean(),
                     fontdict={'size': 13})
            plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 13})
            plt.ylim((0, 3));
            plt.legend(loc='upper right', fontsize=10);
            plt.draw();
            plt.pause(0.01)

    plt.ioff()
    plt.show()






