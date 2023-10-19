import numpy as np
import gymnasium as gym

class QLearn:
    def __init__(self, envoirement, episodes, learning_rate, discount_factor, exploration_rate):
        self.env = envoirement
        self.episodes = episodes
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.exploration_rate = exploration_rate

        self.bin_size = 30
        self.state_bins = self.create_bins()
        num_actions = self.env.action_space.n
        self.q_table = np.zeros([self.bin_size] * len(self.state_bins) + [num_actions])


    def train(self, visualize):
        print("training..")
        rewards = 0

        for episode in range(self.episodes):
            if episode == 1_100 and visualize:
                self.env = gym.make("CartPole-v1", render_mode="human")

            score = 0
            done = False
            state, _ = self.env.reset()
            state = self.discretize_state(state)

            while not done:
                action = self.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                next_state = self.discretize_state(next_state)
                score += reward
                if done:
                    break

                max_next_q = np.max(self.q_table[next_state])
                current_q = self.q_table[state][action]
                new_q = current_q * (1 - self.alpha) + self.alpha * (reward + self.gamma * max_next_q)
                self.q_table[state][action] = new_q

                state = next_state

            rewards += score
            if score > 195 and episode > 100:
                print(f"Solved in {episode} episodes")
                break
        self.env.close()

    def create_bins(self):
        bins = [
            np.linspace(-4.8, 4.8, self.bin_size),
            np.linspace(-5, 5, self.bin_size),
            np.linspace(-0.418, 0.418, self.bin_size),
            np.linspace(-5, 5, self.bin_size)
        ]
        return bins

    def discretize_state(self, state):
        return tuple(np.digitize(state[i], self.state_bins[i]) - 1 for i in range(len(state)))

    def select_action(self, state):
        if np.random.rand() < self.exploration_rate:
            action = np.random.choice([0, 1])
        else:
            action = np.argmax(self.q_table[state])
        return action