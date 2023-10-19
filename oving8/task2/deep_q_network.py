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

    def train(self, visualize):
        print("Training...")
        rewards = []
        loss_history = [] 
        episode_rewards = []

        for episode in range(self.episodes):
            if episode == 2_100 and visualize:
                self.env = gym.make("CartPole-v1", render_mode="human")
            state, _ = self.env.reset()
            score = 0
            done = False

            while not done:
                action = self.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                score += reward

                with torch.no_grad():
                    target_q = reward + self.gamma * torch.max(self.q_network(torch.tensor(next_state)))
                tensor = torch.tensor(self.get_tensor_state(state))
                current_q = self.q_network(tensor)[action]

                loss = nn.MSELoss()(current_q, target_q)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_history.append(loss.item())
                state = next_state

            episode_rewards.append(score)
            rewards.append(score)
            if score > 195 and episode > 100:
                print(f"Solved in {episode} episodes")
                break
            if episode % 10 == 0:
                print(f"Episode {episode}, Avg Reward: {np.mean(episode_rewards[-10:])}, Loss: {loss_history[-1]}")

        self.env.close()
        print(f"Training finished after {self.episodes} episodes. Mean episode reward: {np.mean(rewards)}")
        
        # Plot loss and episode rewards
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.subplot(211)
        plt.plot(loss_history)
        plt.title("Training Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.subplot(212)
        plt.plot(episode_rewards)
        plt.title("Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.show()

    def get_tensor_state(self, state):
        return torch.tensor(state, dtype=torch.float32)

    def select_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = self.get_tensor_state(state)
                q_values = self.q_network(torch.tensor(state_tensor))
                return torch.argmax(q_values).item()
