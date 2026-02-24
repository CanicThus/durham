import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # disable CUDA (better on Colab/NCC: choose an environment without GPU)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rldurham as rld

device = torch.device("cpu")

"""
https://github.com/Rafael1s/Deep-Reinforcement-Learning-Algorithms/tree/master/BipedalWalker-TwinDelayed-DDPG%20(TD3)
"""
"""
整体评语
上次
近400epo上0 end119x end前有上下滑动
图03
改进
Actor中间添加归一层
结果
330破0 end19x
有大幅的下滑 图04

训练添加warmup step 希望优化冷启动
效果为 size池>1e4后更新actor，之前step>1e4更新critic
05 更新不大 需要判断是否有效 感觉代码山重复了

主循环添加噪声衰弱，减少后期的大动作探索
希望曲线更平滑
06 提早到150破0 但曲线波动很大，后期还能滑倒-200

将噪声衰减从while中移除 改为eposide衰减
0.9999->0.999
无效

去掉actor的归一化
无用 可能是后期一整个eposide的值填满buff拿来训练导致直接失败的

改为每个step train1
稳定多了 但分数还是19x 图07

更新噪声衰减
没什么变化 图08

actor和critic增加归一层
300上0 end 19x 没说你们变化

更新学习率为3e-4
图09 280破0 400震荡190->-130 end200

train1 延迟更新未生效 batch_size=100-》256
崩 1000episode不上0

reward调整 -100-> -1 避免极端起伏
启动变慢 370破0 500收敛220 后续无提升
极不稳定 后续220-》-100 图10

实现模型内部归一化
370---0 end235 少量提升

去掉模型内部归一化
350破0 ebd22x
差不了多少

Agent状态输入可以归一化
这个再上不去可以看train的算法了
"""

# Actor Neural Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


# Q1-Q2-Critic Neural Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1


# Expects tuples of (state, next_state, action, reward, done)
class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X))
            y.append(np.array(Y))
            u.append(np.array(U))
            r.append(np.array(R))
            d.append(np.array(D))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class Agent(torch.nn.Module):
    def __init__(self, env):
        super(Agent, self).__init__()
        self.discrete_act, self.discrete_obs, self.act_dim, self.obs_dim = rld.env_info(env)
        self.max_action = env.action_space.high[0]
        self.replay_buf = ReplayBuffer()

        self.actor = Actor(self.obs_dim, self.act_dim, self.max_action).to(device)
        self.actor_target = Actor(self.obs_dim, self.act_dim, self.max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(self.obs_dim, self.act_dim).to(device)
        self.critic_target = Critic(self.obs_dim, self.act_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.delay_counter = -1
        self.delay_freq = 1
    def sample_action(self, s):
        # return torch.rand(self.act_dim) * 2 - 1 # unifrom random in [-1, 1]
        state = torch.FloatTensor(s.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def put_data(self, state, action, observation, reward, done):
        self.replay_buf.add((state, observation, action, reward, done))

    def train(self, iterations, batch_size=256, discount=0.99, \
              tau=0.005, policy_noise=0.2, noise_clip=0.5):

        self.delay_counter += 1
        replay_buffer = self.replay_buf
        for it in range(iterations):
            # Sample replay buffer
            x, y, u, r, d = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Select action according to policy and add clipped noise
            noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if self.delay_counter == self.delay_freq:

                # Compute actor loss
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                self.delay_counter = -1


env = rld.make("rldurham/Walker", render_mode="rgb_array")
# env = rld.make("rldurham/Walker", render_mode="rgb_array", hardcore=True) # only attempt this when your agent has solved the non-hardcore version

# get statistics, logs, and videos
env = rld.Recorder(
    env,
    smoothing=10,                       # track rolling averages (useful for plotting)
    video=True,                         # enable recording videos
    video_folder="videos",              # folder for videos
    video_prefix="agent-video",  # prefix for videos (replace xxxx00 with your username)
    logs=True,                          # keep logs
)

# training on CPU recommended
rld.check_device()

# environment info
rld.env_info(env, print_out=True)

# render start image (reset just to render image)
env.reset(seed=42)
rld.render(env)

# in the submission please use seed_everything with seed 42 for verification
seed, observation, info = rld.seed_everything(42, env)

# initialise agent
agent = Agent(env)
max_episodes = 1000

# track statistics for plotting
tracker = rld.InfoTracker()

# switch video recording off (only switch on every x episodes as this is slow)
env.video = False

start_timestep = 1e4
expl_noise = 0.2
noise_decay = 0.999
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
# threshold = env.spec.reward_threshold
current_eps = 0

# training procedure
for episode in range(max_episodes):

    # timestep = 0
    episode_reward = 0
    # recording statistics and video can be switched on and off (video recording is slow!)
    env.info = True  # usually tracking every episode is fine
    env.video = episode % 100 == 0  # record videos every 100 episodes (set BEFORE calling reset!)

    # reset for new episode
    state, info = env.reset()
    low = env.action_space.low
    high = env.action_space.high

    # run episode
    done = False
    while not done:

        # Select action randomly or according to policy
        if current_eps < start_timestep:
            action = env.action_space.sample()
        else:
            action = agent.sample_action(np.array(state))
            # 应用衰减后的噪声
            noise = np.random.normal(0, expl_noise, size=action_dim)
            action = (action + noise).clip(low, high)

        # take action in the environment
        observation, reward, terminated, truncated, info = env.step(action)

        # check whether done
        done = terminated or truncated

        # remember
        if reward == -100:
            agent.put_data(state, action, observation, -1, done)
        else:
            agent.put_data(state, action, observation, reward, done)
        # update state
        state = observation

        episode_reward += reward
        # timestep += 1
        current_eps += 1

        # train the agent after each step
        if current_eps >= start_timestep:
            agent.train(1)

    # 噪声衰减
    expl_noise = max(0.05, expl_noise * noise_decay)

    # track and plot statistics
    tracker.track(info)
    if (episode + 1) % 10 == 0:
        tracker.plot(r_mean_=True, r_std_=True, r_sum=dict(linestyle=':', marker='x'))
    print("episode:{}, \tReward:{}".format(episode, int(episode_reward)))
# don't forget to close environment (e.g. triggers last video save)
env.close()

# write log file (for coursework)
env.write_log(folder="logs", file="agent-log.txt")  # replace xxxx00 with your username