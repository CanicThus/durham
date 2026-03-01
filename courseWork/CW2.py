import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # disable CUDA (better on Colab/NCC: choose an environment without GPU)
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import rldurham as rld

from collections import deque

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6), n_step=3, gamma=0.99):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, state_dim))
        self.dead = np.zeros((max_size, 1))

        # n-step 参数与临时缓存
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=self.n_step)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, reward, next_state, dead):
        self.n_step_buffer.append((state, action, reward, next_state, dead))

        if len(self.n_step_buffer) == self.n_step:
            s, a, r, s_prime, d = self._get_n_step_info()
            self._store(s, a, r, s_prime, d)

    def _store(self, state, action, reward, next_state, dead):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.dead[self.ptr] = dead

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def _get_n_step_info(self):
        state, action, _, _, _ = self.n_step_buffer[0]
        target_next_state, target_dead = self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]

        n_step_reward = 0
        for i, transition in enumerate(self.n_step_buffer):
            _, _, r, s_next, d = transition

            n_step_reward += (self.gamma ** i) * r
            if d:
                target_next_state, target_dead = s_next, d
                break

        return state, action, n_step_reward, target_next_state, target_dead

    def finish_episode(self):
        # 回合结束时，清空 n_step_buffer
        while len(self.n_step_buffer) > 0:
            s, a, r, s_prime, d = self._get_n_step_info()
            self._store(s, a, r, s_prime, d)
            self.n_step_buffer.popleft()


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.dead[ind]).to(self.device)
        )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, maxaction):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, action_dim)

        self.maxaction = maxaction

    def forward(self, state):
        a = torch.tanh(self.l1(state))
        a = torch.tanh(self.l2(a))
        a = torch.tanh(self.l3(a)) * self.maxaction
        return a

class Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, num_quantiles=25, num_critics=2, dropout_rate=0.01):
        super(Q_Critic, self).__init__()

        self.num_quantiles = num_quantiles
        self.num_critics = num_critics

        self.q_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim + action_dim, net_width),
                nn.LayerNorm(net_width),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(net_width, net_width),
                nn.LayerNorm(net_width),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(net_width, num_quantiles)
            ) for _ in range(num_critics)
        ])

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        # 将每个 critic 的输出叠加，形状为: (batch_size, num_critics, num_quantiles)
        quantiles = torch.stack([net(sa) for net in self.q_networks], dim=1)
        return quantiles

class Agent(torch.nn.Module):
    def __init__(self,
        env_with_Dead,
        state_dim,
        action_dim,
        max_action,
        gamma=0.99,
        net_width=128,
        a_lr=1e-4,
        c_lr=1e-4,
        Q_batchsize=256,
        num_quantiles=25,
        num_critics=2,
        drop_quantiles=5,
        dropout_rate=0.01,
        n_step=3):
        super(Agent, self).__init__()

        self.actor = Actor(state_dim, action_dim, net_width, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=a_lr)
        self.actor_target = copy.deepcopy(self.actor)

        self.q_critic = Q_Critic(state_dim, action_dim, net_width, num_quantiles, num_critics, dropout_rate).to(device)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=c_lr)
        self.q_critic_target = copy.deepcopy(self.q_critic)

        self.env_with_Dead = env_with_Dead
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.policy_noise = 0.2 * max_action
        self.noise_clip = 0.5 * max_action
        self.tau = 0.005
        self.Q_batchsize = Q_batchsize
        self.delay_counter = -1
        self.delay_freq = 1
        # TQC 超参数
        self.num_quantiles = num_quantiles
        self.num_critics = num_critics
        self.drop_quantiles = drop_quantiles
        self.dropout_rate = dropout_rate

        self.n_step = n_step

    def sample_action(self, s):
        # return torch.rand(self.act_dim) * 2 - 1 # unifrom random in [-1, 1]
        with torch.no_grad():
            state = torch.FloatTensor(s.reshape(1, -1)).to(device)
            a = self.actor(state)
        return a.cpu().numpy().flatten()

    def train(self, replay_buffer):
        self.delay_counter += 1

        with torch.no_grad():
            s, a, r, s_prime, dead_mask = replay_buffer.sample(self.Q_batchsize)
            noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            smoothed_target_a = (
                    self.actor_target(s_prime) + noise  # Noisy on target action
            ).clamp(-self.max_action, self.max_action)

        # Compute the target Q value
        z_target = self.q_critic_target(s_prime, smoothed_target_a)
        z_target_flatten = z_target.view(self.Q_batchsize, -1)
        sorted_z_target, _ = torch.sort(z_target_flatten, dim=1)
        kept_z_target = sorted_z_target[:, : -self.drop_quantiles]

        '''DEAD OR NOT'''
        if self.env_with_Dead:
            target_Q = r + (1 - dead_mask) * (self.gamma ** self.n_step) * kept_z_target  # env with dead
        else:
            target_Q = r + (self.gamma ** self.n_step) * kept_z_target  # env without dead

        # Get current Q estimates
        current_z = self.q_critic(s, a)

        # target_Q 形状扩展至 (batch, 1, 1, num_kept_quantiles)
        target_Q_ext = target_Q.unsqueeze(1).unsqueeze(1)
        # current_z 形状扩展至 (batch, num_critics, num_quantiles, 1)
        current_z_ext = current_z.unsqueeze(-1)
        # 差值计算
        td_error = target_Q_ext - current_z_ext
        # Huber Loss 计算
        kappa = 1.0  # Huber loss 的阈值
        abs_td_error = torch.abs(td_error)
        huber_loss = torch.where(abs_td_error <= kappa, 0.5 * abs_td_error ** 2, kappa * (abs_td_error - 0.5 * kappa))

        # 计算分位数权重Tau
        tau = torch.arange(1, self.num_quantiles + 1,
                           device=device).float() / self.num_quantiles - 0.5 / self.num_quantiles
        tau = tau.view(1, 1, self.num_quantiles, 1)

        # Quantile Loss公式: |tau - I(td_error < 0)| * huber_loss
        quantile_loss = torch.abs(tau - (td_error.detach() < 0).float()) * huber_loss

        # 在 kept_quantiles 维度求均值，在 quantiles 维度求和，在 critics 和 batch 维度求均值
        q_loss = quantile_loss.mean(dim=3).sum(dim=2).mean()

        # Optimize the q_critic
        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        self.q_critic_optimizer.step()

        # delay update actor
        if self.delay_counter == self.delay_freq:

            q_actor_quantiles = self.q_critic(s, self.actor(s))
            a_loss = -q_actor_quantiles.mean()

            self.actor_optimizer.zero_grad()
            a_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.delay_counter = -1

    def save(self, filename):
        """
        保存模型和优化器的状态字典
        :param filename: 保存的文件路径及名称 (例如: "checkpoints/agent_model.pth")
        """
        # 确保保存目录存在
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'q_critic_state_dict': self.q_critic.state_dict(),
            'q_critic_optimizer_state_dict': self.q_critic_optimizer.state_dict()
        }, filename)
        print(f"Model saved successfully at {filename}")

    def load(self, filename):
        """
        载入模型和优化器的状态字典
        :param filename: 载入的文件路径及名称
        """
        # 使用全局的 device 变量来确保跨设备（CPU/GPU）加载的兼容性
        checkpoint = torch.load(filename, map_location=device)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.q_critic.load_state_dict(checkpoint['q_critic_state_dict'])
        self.q_critic_optimizer.load_state_dict(checkpoint['q_critic_optimizer_state_dict'])

        # 载入主网络后，强制同步目标网络，确保 target 网络与主网络参数一致
        self.actor_target = copy.deepcopy(self.actor)
        self.q_critic_target = copy.deepcopy(self.q_critic)

        print(f"Model loaded successfully from {filename}")

class ERL_Manager:
    def __init__(self, state_dim, action_dim, net_width, max_action, pop_size=5, mutation_rate=0.1,
                 mutation_power=0.05):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.mutation_power = mutation_power
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化演化种群
        self.population = [Actor(state_dim, action_dim, net_width, max_action).to(self.device) for _ in range(pop_size)]

        self.env = rld.make("rldurham/Walker", render_mode="rgb_array")
        self.env.reset(seed=42)
        rld.render(self.env)

    def mutate(self, actor):
        """变异算子：为权重添加高斯噪声"""
        child = copy.deepcopy(actor)
        with torch.no_grad():
            for param in child.parameters():
                if len(param.shape) >= 1:
                    mask = (torch.rand(param.shape, device=self.device) < self.mutation_rate).float()
                    noise = torch.randn(param.shape, device=self.device) * self.mutation_power
                    param.data.add_(mask * noise)
        return child

    def crossover(self, parent1, parent2):
        """交叉算子：从两个父代中随机继承权重参数"""
        child = copy.deepcopy(parent1)
        with torch.no_grad():
            for param_c, param_1, param_2 in zip(child.parameters(), parent1.parameters(), parent2.parameters()):
                mask = (torch.rand(param_c.shape, device=self.device) < 0.5).float()
                param_c.data.copy_(mask * param_1.data + (1 - mask) * param_2.data)
        return child

    def _evaluate_actor(self, actor, env, replay_buffer):
        """评估单个 Actor 并在探索时填充 Replay Buffer"""
        state, info = env.reset()
        ep_r = 0
        done = False
        while not done:
            with torch.no_grad():
                s = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
                action = actor(s).cpu().numpy().flatten()

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if reward <= -100:
                replay_buffer.add(state, action, -100, next_state, True)
            else:
                replay_buffer.add(state, action, reward, next_state, False)

            state = next_state
            ep_r += reward

            if done:
                replay_buffer.finish_episode()
        return ep_r

    def evaluate_population(self, replay_buffer):
        """评估整个种群，返回所有个体的适应度(Fitnesses)"""
        fitnesses = []
        for actor in self.population:
            fit = self._evaluate_actor(actor, self.env, replay_buffer)
            fitnesses.append(fit)
        return fitnesses

    def evolve(self, fitnesses, rl_agent_actor, generation):
        """优胜劣汰及基因演化，并定期同步 RL Agent 的知识"""
        sorted_indices = np.argsort(fitnesses)[::-1]
        best_actors = [self.population[i] for i in sorted_indices]
        new_population = []

        # 1. 精英保留 (Elitism): 直接将表现最好的原封不动放入下一代
        new_population.append(copy.deepcopy(best_actors[0]))

        # 2. 知识共享 (Sync): 周期性地将基于梯度的 RL Agent 加入种群以参与演化
        if generation % 5 == 0:
            new_population.append(copy.deepcopy(rl_agent_actor))
        else:
            new_population.append(self.mutate(best_actors[0]))

        # 3. 交叉与变异填满剩余的种群席位
        while len(new_population) < self.pop_size:
            p1 = best_actors[np.random.randint(0, max(2, self.pop_size // 2))]
            p2 = best_actors[np.random.randint(0, max(2, self.pop_size // 2))]

            child = self.crossover(p1, p2)
            child = self.mutate(child)
            new_population.append(child)

        self.population = new_population
        return fitnesses[sorted_indices[0]]  # 返回种群中的最高分

env = rld.make("rldurham/Walker", render_mode="rgb_array")
# env = rld.make("rldurham/Walker", render_mode="rgb_array", hardcore=True) # only attempt this when your agent has solved the non-hardcore version

# get statistics, logs, and videos
env = rld.Recorder(
    env,
    smoothing=10,                       # track rolling averages (useful for plotting)
    video=True,                         # enable recording videos
    video_folder="videos",              # folder for videos
    video_prefix="xxxx00-agent-video",  # prefix for videos (replace xxxx00 with your username)
    logs=True,                          # keep logs
)

# training on CPU recommended
# rld.check_device()

# environment info
rld.env_info(env, print_out=True)

# render start image (reset just to render image)
env.reset(seed=42)
rld.render(env)

# in the submission please use seed_everything with seed 42 for verification
seed, observation, info = rld.seed_everything(42, env)

# 参数设计
save_dir = "result"
base_save_score = 240

env_with_Dead = True  # Whether the Env has dead state. True for Env like BipedalWalkerHardcore-v3, CartPole-v0. False for Env like Pendulum-v0
discrete_act, discrete_obs, act_dim, obs_dim = rld.env_info(env)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
expl_noise = 0.25
print('  state_dim:', state_dim, '  action_dim:', action_dim, '  max_a:', max_action, '  min_a:', env.action_space.low[0])
kwargs = {
    "env_with_Dead": env_with_Dead,
    "state_dim": state_dim,
    "action_dim": action_dim,
    "max_action": max_action,
    "gamma": 0.99,
    "net_width": 256,
    "a_lr": 1e-4,
    "c_lr": 1e-4,
    "Q_batchsize": 256,
    "num_quantiles" : 25,
    "num_critics" : 5,
    "drop_quantiles" : 40,
    "dropout_rate" : 0.01,
    "n_step": 3,
}
max_episodes = 1000

buffer_n_step=3
buffer_gamma = 0.99

# initialise agent, replay_buffer
agent = Agent(**kwargs)
replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=int(1e6), n_step=buffer_n_step, gamma=buffer_gamma)

erl_manager = ERL_Manager(
    state_dim=state_dim,
    action_dim=action_dim,
    net_width=kwargs["net_width"],
    max_action=max_action,
    pop_size=5,
    mutation_rate=0.1,
    mutation_power=0.05
)

# track statistics for plotting
tracker = rld.InfoTracker()

# switch video recording off (only switch on every x episodes as this is slow)
env.video = False

# training procedure
for episode in range(max_episodes):

    fitnesses = erl_manager.evaluate_population(replay_buffer)

    # recording statistics and video can be switched on and off (video recording is slow!)
    env.info = True  # usually tracking every episode is fine
    env.video = episode % 100 == 0  # record videos every 100 episodes (set BEFORE calling reset!)

    # reset for new episode
    state, info = env.reset()

    # run episode
    ep_r = 0
    steps = 0
    expl_noise = min(0.1, expl_noise * 0.999)
    done = False
    while not done:
        steps += 1

        # select the agent action
        action = (agent.sample_action(state) + np.random.normal(0, max_action * expl_noise, size=action_dim)
                ).clip(-max_action, max_action)

        # take action in the environment
        observation, reward, terminated, truncated, info = env.step(action)


        # remember
        if reward <= -100:
            replay_buffer.add(state, action, -100, observation, True)
        else:
            replay_buffer.add(state, action, reward, observation, False)

        # check whether done
        done = terminated or truncated
        state = observation
        ep_r += reward

        if done:
            replay_buffer.finish_episode()

        # train the agent after each step
        if replay_buffer.size > 2000:
            UTDratio = 2
            for _ in range(UTDratio):
                agent.train(replay_buffer)

    best_pop_score = erl_manager.evolve(fitnesses, agent.actor, episode)

    # track and plot statistics
    tracker.track(info)
    if (episode + 1) % 10 == 0:
        tracker.plot(r_mean_=True, r_std_=True, r_sum=dict(linestyle=':', marker='x'))
    print('episode:', episode, 'score:', ep_r, 'step:', steps)

    if int(ep_r) > base_save_score + 1:
        base_save_score = ep_r
        agent.save(os.path.join(save_dir, f"normal_{ep_r}.pth"))
# don't forget to close environment (e.g. triggers last video save)
env.close()

# write log file (for coursework)
env.write_log(folder="logs", file="xxxx00-agent-log.txt")  # replace xxxx00 with your username