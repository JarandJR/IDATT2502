import gymnasium as gym
import numpy as np

def create_bins(bin_size):
    bins = [
        np.linspace(-4.8, 4.8, bin_size),
        np.linspace(-5, 5, bin_size),
        np.linspace(-0.418, 0.418, bin_size),
        np.linspace(-5, 5, bin_size)
    ]
    return bins

def discretize_state(state, state_bins):
    return tuple(np.digitize(state[i], state_bins[i]) - 1 for i in range(len(state)))

def select_action(Q, state, exploration_rate):
    if np.random.rand() < exploration_rate:
        action = np.random.choice([0, 1])
    else:
        action = np.argmax(Q[state])
    return action

def q_learning_cartpole(episodes, learning_rate, discount_factor, exploration_rate, visualize):
    if visualize:
        env = gym.make("CartPole-v1", render_mode="human")
    else:
        env = gym.make("CartPole-v1")
    num_actions = env.action_space.n
    state_space = 4
    bin_size = 30

    Q = np.zeros((bin_size ** state_space, num_actions))

    state_bins = create_bins(bin_size)

    rewards = 0
    steps = 0
    for episode in range(episodes):
        steps += 1
        score = 0
        done = False
        state = env.reset()
        state = discretize_state(state, state_bins)

        while not done:
            action = select_action(Q, state, exploration_rate)
            next_state, reward, done, _ = env.step(action)
            next_state = discretize_state(next_state, state_bins)
            score += reward

            max_next_Q = np.max(Q[next_state])
            current_Q = Q[state][action]
            new_Q = current_Q * (1 - learning_rate) + learning_rate * (reward + discount_factor * max_next_Q)
            Q[state][action] = new_Q

            state = next_state

        rewards += score
        print(f"Episode {episode + 1} finished after {steps} timesteps")
        if score > 195 and steps > 100:
            print(f"Solved in {steps} episodes")
            break

    env.close()
    return Q

def main():
    print("Task 1")
    Q = q_learning_cartpole(episodes=1000, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.2, visualize=True)

if __name__ == "__main__":
    main()
