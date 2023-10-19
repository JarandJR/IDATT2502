import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, environment, episodes, learning_rate, discount_factor, exploration_rate):
        self.env = environment
        self.episodes = episodes
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.exploration_rate = exploration_rate

        self.num_actions = self.env.action_space.n
        self.state_size = self.env.observation_space.shape[0]
        self.q_network = DQN(self.state_size, self.num_actions)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.alpha)

    def train(self):
        print("Training...")
        rewards = 0

        for episode in range(self.episodes):
            state = self.env.reset()
            score = 0
            done = False

            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                score += reward

                target_q = reward + self.gamma * torch.max(self.q_network(torch.tensor(next_state)).detach())
                current_q = self.q_network(torch.tensor(state))[action]

                loss = nn.MSELoss()(current_q, target_q)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                state = next_state

            rewards += score

        self.env.close()

    def select_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = self.q_network(torch.tensor(state))
                return torch.argmax(q_values).item()