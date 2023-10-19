import gymnasium as gym

from deep_q_network import DQNAgent

if __name__ == "__main__":
    model = DQNAgent(
        environment=gym.make("CartPole-v1"),
        episodes=2_000,
        learning_rate=0.00025,
        discount_factor=0.99,
        exploration_rate=0.1,
        )
    model.train()
    model.test()
