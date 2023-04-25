import gym
from gym_envs.register_envs import register_envs

if __name__ == "__main__":
    register_envs()

    gym.make("RandomChessEnv")