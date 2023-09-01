import torch
import io
import torch, numpy as np, torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from attackEnvironment import AttackEnv
import random
import os

if __name__ == "__main__":
    writer = SummaryWriter()
    start_num = 0
    deleted = False
    model_save_folder = "v820_3/"
    model_load_path = model_save_folder + f"checkpoints/checkpoint{start_num}.pth"
    buffer_load_path = model_save_folder + f"checkpoints/buffer.pth"
    device = "cpu"
    torch.set_num_threads(8)
    # device = 'cuda'

    showFig_iteration = 50  # save trajectories images
    model_save_iteration = 500  # save the training model as **.pth
    data_record_iteration = (
        1  # recoard key data of one exploration such as sigma_mean q_mean
    )
    data_save_iteration = 100 * data_record_iteration  # save key data to file
    test_iteration = 500  # test the model by an random scenario
    repeat_num = 20  # number of simulations required for an exploration


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x


class Critic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim + action_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return self.fc4(x)


class OrnsteinUhlenbeckNoise:
    def __init__(self, size, mu=0.0, theta=0.01, sigma=1):
        self.size = size
        self.theta = torch.as_tensor(theta, dtype=torch.float32).to(device)
        self.sigma = torch.as_tensor(sigma, dtype=torch.float32).to(device)
        self.mu = torch.as_tensor(mu, dtype=torch.float32).to(device)
        self.state = torch.ones(self.size, device=device) * self.mu

    def reset(self):
        self.state = torch.ones(self.size, device=device) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * torch.randn(
            self.size
        ).to(device)
        self.state += dx
        return self.state


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward):
        if (len(self.buffer)) < self.capacity:
            self.buffer.append(None)
        state = torch.as_tensor(state, dtype=torch.float32)
        action = torch.as_tensor(action, dtype=torch.float32)
        reward = torch.as_tensor(reward, dtype=torch.float32)
        self.buffer[self.position] = [state, action, reward]
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward = map(torch.stack, zip(*batch))
        return state.to(device), action.to(device), reward.to(device)

    def __len__(self):
        return len(self.buffer)


class DDPG:
    def __init__(
        self,
        state_dim,
        action_dim,
        actor_lr,
        critic_lr,
        tau,
        capacity,
        batch_size,
    ):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        writer.add_graph(self.actor, torch.rand(2), "Actor")
        writer.add_graph(self.critic, (torch.rand(1, 2), torch.rand(1, 2)), "Critic")
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.tau = tau
        self.replay_buffer = ReplayBuffer(capacity)
        self.batch_size = batch_size
        self.noise = OrnsteinUhlenbeckNoise(action_dim)
        self.actor_loss = torch.as_tensor(0, dtype=torch.float32)
        self.critic_loss = torch.as_tensor(0, dtype=torch.float32)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward = self.replay_buffer.sample(self.batch_size)

        q = self.critic(state, action)
        self.critic_loss = F.mse_loss(q.view(self.batch_size), reward.detach())

        self.critic_optimizer.zero_grad()
        self.critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        self.actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(
            self.actor.parameters(), self.actor_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def store_transition(self, state, action, reward):
        self.replay_buffer.push(state, action, reward)


def load_model(agent):
    scores = -1000000
    if os.path.isfile(buffer_load_path):
        agent.replay_buffer.buffer = torch.load(buffer_load_path)

    if os.path.isfile(model_load_path):
        checkpoint = torch.load(model_load_path)
        if "score" in checkpoint:
            scores = checkpoint["score"]
        if "model_state_dict_actor" in checkpoint:
            agent.actor.load_state_dict(checkpoint["model_state_dict_actor"])
        if "model_state_dict_critic" in checkpoint:
            agent.critic.load_state_dict(checkpoint["model_state_dict_critic"])
        if "optimizer_state_dict_actor" in checkpoint:
            pass
            agent.actor_optimizer.load_state_dict(
                checkpoint["optimizer_state_dict_actor"]
            )
        if "optimizer_state_dict_critic" in checkpoint:
            agent.critic_optimizer.load_state_dict(
                checkpoint["optimizer_state_dict_critic"]
            )
        if "model_state_dict_actor_target" in checkpoint:
            agent.actor_target.load_state_dict(
                checkpoint["model_state_dict_actor_target"]
            )
        if "model_state_dict_critic_target" in checkpoint:
            agent.critic_target.load_state_dict(
                checkpoint["model_state_dict_critic_target"]
            )
    return scores


def train(env, agent: DDPG, episodes):
    scores = load_model(agent)

    for episode in range(episodes):
        episode += start_num + 1
        # agent.noise.reset()
        # record simulation states
        sigma_ = np.array([])
        q_ = np.array([])
        kr_ = np.array([])
        kq_ = np.array([])

        reward_mean = 0
        R_mean = 0

        for _ in range(repeat_num):
            state = env.reset()
            done = False
            while not done:
                ideal_action = agent.actor(
                    torch.as_tensor(state, dtype=torch.float32).to(device)
                )
                action = ideal_action.detach()

                state, done = env.Step(action.cpu().numpy())

            (
                sigma_one,
                q_one,
                kr_one,
                kq_one,
                R_last,
                reward,
            ) = env.getStates()

            index = np.random.choice(len(sigma_one), 5)

            sigma_ = np.append(sigma_, sigma_one[index])
            q_ = np.append(q_, q_one[index])
            kr_ = np.append(kr_, kr_one[index])
            kq_ = np.append(kq_, kq_one[index])

            reward_mean += reward / repeat_num
            R_mean += R_last / repeat_num

        for s1, s2, a1, a2 in zip(sigma_, q_, kr_, kq_):
            agent.store_transition([s1, s2], [a1, a2], reward_mean)

        agent.update()

        writer.add_scalar("Loss/critic_loss", agent.critic_loss, episode)
        writer.add_scalar("Loss/actor_loss", agent.actor_loss, episode)
        writer.add_scalar("States/R", R_mean, episode)
        writer.add_scalar("States/q", q_.mean(), episode)
        writer.add_scalar("States/sigma", sigma_.mean(), episode)
        writer.add_scalar("reward", reward_mean, episode)

        print(f"Episode: {episode}\t Score: {reward_mean}")

        if (episode % showFig_iteration) == 0:
            env.show("iter.jpg")
        if (episode + 1) % test_iteration == 0:
            pass
            # subprocess.Popen(["python", "../guidanceFit/netTest.py "],shell=True)
        if (episode + 1) % model_save_iteration == 0:
            # Saving model and optimizer state_dicts along with the epoch num
            torch.save(
                {
                    "score": scores,
                    "model_state_dict_actor": agent.actor.state_dict(),
                    "model_state_dict_critic": agent.critic.state_dict(),
                    "model_state_dict_actor_target": agent.actor_target.state_dict(),
                    "model_state_dict_critic_target": agent.critic_target.state_dict(),
                    "optimizer_state_dict_actor": agent.actor_optimizer.state_dict(),
                    "optimizer_state_dict_critic": agent.critic_optimizer.state_dict(),
                },
                model_save_folder + f"checkpoints/checkpoint{episode}.pth",
            )
            # torch.save(agent.replay_buffer.buffer, buffer_load_path)

    env.close()
    return scores


def delete_files_in_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        elif os.path.isdir(file_path):
            delete_files_in_directory(file_path)
            # 如果想删除空的子文件夹，可以使用以下代码
            # os.rmdir(file_path)
            # print(f"Deleted directory: {file_path}")


if __name__ == "__main__":
    if not os.path.exists(model_save_folder):
        os.makedirs(model_save_folder)
    if not os.path.exists(model_save_folder + "checkpoints"):
        os.makedirs(model_save_folder + "checkpoints")
    if deleted == True:
        delete_files_in_directory(model_save_folder)
        delete_files_in_directory("runs")
    env = AttackEnv()
    agent = DDPG(
        state_dim=2,
        action_dim=2,
        actor_lr=0.00001,
        critic_lr=0.0001,
        tau=0.001,
        capacity=2**27,
        batch_size=2**16,
    )
    scores = train(env, agent, episodes=1000000)
    writer.flush()
