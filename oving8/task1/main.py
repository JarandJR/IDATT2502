import gymnasium as gym

from q_learn import QLearn

if __name__ == "__main__":
    QLearn(
        envoirement=gym.make("CartPole-v1"),
        episodes=10_000,
        learning_rate=0.1,
        discount_factor=0.9,
        exploration_rate=0.2
        ).train(visualize=True)
