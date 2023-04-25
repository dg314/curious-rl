import curious_rl_gym
import gym

env = gym.make("RandomChessEnv-v0", render_mode="video")

env.reset()

for i in range(300):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated:
        break

env.close()
