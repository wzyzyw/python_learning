import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym


"""
DQN:
target_net和eval_net结构相同，但更新速度不同，实际上target_net不参与参数更新的，
只是每隔一段时间直接读取eval_net的网络参数；
eval_net网络更新:本例中，当经验池满了之后，直接从经验池抽取样本，按dqn的规则训练
每个step存储记忆；
无论神经网络什么状态，eval_net输入当前状态，返回各动作对应的Q值
gym、unity等环境交互：step参数动作数值（内部定义了数值对应的具体动作），返沪当前状态和执行动作后的状态，reward和done等
"""
torch.manual_seed(1)


def model_hyper_para():
    hyper_para={}
    hyper_para['Learning Rate']=0.01
    hyper_para['Batch Size']=32
    hyper_para['Epsilon']=0.9  # 最优选择动作百分比
    hyper_para['GAMMA']=0.9   # 奖励递减参数
    hyper_para['Target Replace Iter']=100  # Q 现实网络的更新频率
    hyper_para['Memory Capacity']=2000  # 记忆库大小
    return hyper_para


class Net(nn.Module):
    def __init__(self,n_state,n_actor):
        super(Net,self).__init__()
        self.fc1=nn.Linear(n_state,10)
        self.fc1.weight.data.normal_(0,0.1)
        self.out =nn.Linear(10,n_actor)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x=self.fc1(x)
        x=F.relu(x)
        action_value=self.out(x)
        return action_value


class DQN():
    def __init__(self,n_states,n_actor):
        self.eval_net=Net(n_states,n_actor)
        self.target_net=Net(n_states,n_actor)
        self.hyper_para=model_hyper_para()
        self.n_actor=n_actor
        self.n_state=n_states
        self.learning_step_counter=0  # 用于 target 更新计时
        self.memory_counter=0   # 经验池记数

        self.memory=np.zeros((self.hyper_para['Memory Capacity'],n_states*2+2))  # 初始化经验池
        self.optimizer=torch.optim.Adam(self.eval_net.parameters(),self.hyper_para['Learning Rate'])
        self.loss_func=nn.MSELoss()


    def choose_action(self,x):
        x=torch.unsqueeze(torch.FloatTensor(x),0)
        if np.random.uniform()<self.hyper_para['Epsilon']: # 选最优动作
            action_values=self.eval_net(x)
            action=int(torch.max(action_values,1)[1].data.numpy())
        else:
            action=int(np.random.randint(0,self.n_actor))
        return action

    def store_transition(self,s,a,r,s_):
        transition=np.hstack((s,[a,r],s_))
        # 如果记忆库满了, 就覆盖老数据
        index=self.memory_counter%(self.hyper_para['Memory Capacity'])
        self.memory[index,:]=transition
        self.memory_counter+=1


    def learn(self):
        # target net 参数更新
        if self.learning_step_counter%(self.hyper_para['Target Replace Iter'])==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learning_step_counter+=1

        # 抽取记忆库中的批数据
        sample_index=np.random.choice(self.hyper_para['Memory Capacity'],self.hyper_para['Batch Size'])
        b_memory=self.memory[sample_index,:]
        b_s=torch.FloatTensor(b_memory[:, :self.n_state])
        b_a=torch.LongTensor(b_memory[:, self.n_state:self.n_state+1].astype(int))
        b_r=torch.FloatTensor(b_memory[:, self.n_state+1:self.n_state+2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.n_state:])

        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        q_eval=self.eval_net(b_s).gather(1,b_a)# shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # q_next 不进行反向传递误差, 所以 detach
        q_target = b_r + self.hyper_para['GAMMA'] * q_next.max(1)[0]  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)# shape (batch, 1)

        # 计算, 更新 eval net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()










if __name__=='__main__':
    env = gym.make('CartPole-v0')  # 立杆子游戏
    env = env.unwrapped

    N_ACTIONS = env.action_space.n  # 杆子能做的动作
    N_STATES = env.observation_space.shape[0]  # 杆子能获取的环境信息数

    dqn = DQN(N_STATES,N_ACTIONS)  # 定义 DQN 系统

    for i_episode in range(400):
        s = env.reset()
        while True:
            env.render()  # 显示实验动画
            a = dqn.choose_action(s)

            # 选动作, 得到环境反馈

            s_, r, done, info = env.step(a)

            # 修改 reward, 使 DQN 快速学习
            x, x_dot, theta, theta_dot = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2

            # 存记忆
            dqn.store_transition(s, a, r, s_)

            if dqn.memory_counter > dqn.hyper_para['Memory Capacity']:
                dqn.learn()  # 记忆库满了就进行学习

            if done:  # 如果回合结束, 进入下回合
                break

            s = s_