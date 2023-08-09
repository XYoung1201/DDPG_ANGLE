import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from attackEnvironment_sigma_R_q import AttackEnv
import random
import os
import subprocess

model_load_path = 'checkpoint.pth'
model_save_path = 'checkpoint.pth'
device = 'cpu'
device = 'cuda'
figure_out_num = 50
save_num = 100
show_num = 100


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 8)
        self.fc2 = nn.Linear(8, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, output_dim)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return torch.tanh(self.fc4(x))*4


class Critic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim + action_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)

    def forward(self, state, action):
        x = torch.cat([state, action.view(-1,1)],dim=1)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return self.fc4(x)


class OrnsteinUhlenbeckNoise:

    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.1):
        self.size = size
        self.theta = torch.tensor(theta,dtype=torch.float32).to(device)
        self.sigma = torch.tensor(sigma,dtype=torch.float32).to(device)
        self.mu = torch.tensor(mu,dtype=torch.float32).to(device)
        self.state = torch.ones(self.size, device=device) * self.mu

    def reset(self):
        self.state = torch.ones(self.size,device=device) * self.mu

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
        state = torch.tensor(state, dtype=torch.float32).to(device)
        action = torch.tensor(action, dtype=torch.float32).to(device)
        reward = torch.tensor(reward, dtype=torch.float32).to(device)
        self.buffer[self.position] = (state, action, reward)
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
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.tau = tau
        self.replay_buffer = ReplayBuffer(capacity)
        self.batch_size = batch_size
        self.noise = OrnsteinUhlenbeckNoise(action_dim)
        self.actor_loss_ = []
        self.critic_loss_ = []

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward = self.replay_buffer.sample(
            self.batch_size
        )

        q = self.critic(state, action)
        critic_loss = F.mse_loss(q.view(self.batch_size), reward.detach())
        self.critic_loss_.append(critic_loss.item())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_loss_.append(actor_loss.item())

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
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

    def loss_write(self,actorLossFile,criticLossFile):
        with open(actorLossFile,'a') as file:
            for item in self.actor_loss_:
                file.write(str(item)+'\n')
        with open(criticLossFile,'a') as file:
            for item in self.critic_loss_:
                file.write(str(item)+'\n')
        self.actor_loss_.clear()
        self.critic_loss_.clear()


def train(env, agent, episodes):
    scores = -10000000
    R_list = []
    q_list = []
    sigma_list = []
    reward_list = []
    if os.path.isfile(model_load_path):
        checkpoint = torch.load(model_load_path)
        if 'score' in checkpoint:
            scores = checkpoint['score']
        if 'model_state_dict_actor' in checkpoint:
            agent.actor.load_state_dict(checkpoint['model_state_dict_actor'])
        if 'model_state_dict_critic' in checkpoint:
            agent.critic.load_state_dict(checkpoint['model_state_dict_critic'])
        if 'optimizer_state_dict_actor' in checkpoint:
            agent.actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict_actor'])
        if 'optimizer_state_dict_critic' in checkpoint:
            agent.critic_optimizer.load_state_dict(checkpoint['optimizer_state_dict_critic'])
        if 'model_state_dict_actor_target' in checkpoint:
            agent.actor_target.load_state_dict(checkpoint['model_state_dict_actor_target'])
        if 'model_state_dict_critic_target' in checkpoint:
            agent.critic_target.load_state_dict(checkpoint['model_state_dict_critic_target'])
    for episode in range(episodes):
        state = env.reset()
        agent.noise.reset()
        done = False
        while not done:
            ideal_action = agent.actor(
                torch.tensor(state, dtype=torch.float32).to(device)
            )
            # action = ideal_action.detach() + agent.noise.sample()
            action = ideal_action.detach() 
            state, done = env.Step(action.cpu().numpy())
        states, actions, reward, R_last,q_last,sigma_last = env.getStates()
        if scores < reward:
            env.show('best.jpg')
            scores = reward
        if (episode%figure_out_num) == 0:
            env.show('iter.jpg')
        if (episode+1)%show_num == 0:
            pass
            # subprocess.Popen(["python", "../guidanceFit/netTest.py "],shell=True)
        if (episode+1)%save_num == 0:
            # Saving model and optimizer state_dicts along with the epoch num
            torch.save({
                        'score' : scores,
                        'model_state_dict_actor': agent.actor.state_dict(),
                        'model_state_dict_critic': agent.critic.state_dict(),
                        'model_state_dict_actor_target': agent.actor_target.state_dict(),
                        'model_state_dict_critic_target': agent.critic_target.state_dict(),
                        'optimizer_state_dict_actor': agent.actor_optimizer.state_dict(),
                        'optimizer_state_dict_critic': agent.critic_optimizer.state_dict(),
                        },model_save_path)
            with open('R_list.txt','a') as file:
                for item in R_list:
                    file.write(str(item)+'\n')
            with open('q_list.txt','a') as file:
                for item in q_list:
                    file.write(str(item)+'\n')
            with open('reward_list.txt','a') as file:
                for item in reward_list:
                    file.write(str(item)+'\n')
            R_list.clear()
            q_list.clear()
            sigma_list.clear()
            agent.loss_write('actor_loss.txt','critic_loss.txt')
        for s1,s2,a in zip(states[0],states[1],actions):
            agent.store_transition([s1,s2],a,reward)
        for i in range(40):
            agent.update()
        print(f"Episode {episode}, Score: {reward},r_last:{R_last},q_last:{q_last},sigma_last:{sigma_last}")
        R_list.append(R_last)
        q_list.append(q_last)
        sigma_list.append(sigma_last)
        reward_list.append(reward)
    env.close()
    return scores

if __name__ == "__main__":
    env = AttackEnv()
    agent = DDPG(
        state_dim=2,
        action_dim=1,
        actor_lr=0.001,
        critic_lr=0.001,
        tau=0.001,
        capacity=2000000,
        batch_size=9096,
    )
    if os.path.isfile('q_list.txt'):
        os.remove('q_list.txt')
        os.remove('R_list.txt')
        os.remove('actor_loss.txt')
        os.remove('critic_loss.txt')
        os.remove('reward_list.txt')
    scores = train(env, agent, episodes=1000000)
