import curious_rl_gym
import gym

env = gym.make("RandomChessEnv-v0", render_mode=None)

env.reset()

# for i in range(300):
#     action = env.action_space.sample()
#     observation, reward, terminated, truncated, info = env.step(action)

#     if terminated:
#         break

# env.close()

import tensorflow as tf
from stable_baselines3 import PPO
# import gym
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
# env = make_vec_env(env, n_envs=4)
model = PPO("MultiInputPolicy", env, verbose=2)
model.learn(total_timesteps=100000)
# model.save("ppo_cartpole")
obs, _ = env.reset()
i = 0
while i < 1000:
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, info = env.step(action)
    # env.render()
    if terminated:
        break
    i+=1
env.close()