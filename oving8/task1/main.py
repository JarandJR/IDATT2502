import gym
import numpy as np

def q_learning_cartpole(episodes, learning_rate, discount_factor, print_action):
    env = gym.make("CartPole-v1")
    observation, info = env.reset(seed=42)

    for _ in range(episodes):
        print()
        action = env.action_space.sample()
        if print_action:
            if action == 1:
                print(f"a: {action} -> right")
            else:
                print(f"a: {action} -> left")

        done = False
        while not done:
            step = {
                "c_pos": (step := env.step(action))[0][0], 
                "c_v": step[0][1], 
                "p_ang": step[0][2], 
                "p_angv": step[0][3],
                "reward": step[1],
                "terminated": step[2],
                "truncated": step[3]
                }
            done = step['terminated']
            print(f"{step}")

            if not done:
                pass
                #maxQ = np.max(Q[state+1][action])
                #currentQ = Q[state + (action,)]
                #newQ = currentQ * (1 - learning_rate) + learning_rate * (step['reward'] + discount_factor * maxQ)
                #Q[state + (action)] = newQ

            if step['terminated']or step['truncated']:
                observation, info = env.reset()
    env.close()

def main():
    print("Task 1")
    q_learning_cartpole(episodes=1000, learning_rate=0.1, discount_factor=0.9, print_action=False)

if __name__ == "__main__":
    main()
