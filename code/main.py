import gym
import curious_rl_gym
import numpy as np
import matplotlib.pyplot as plt
import random
import math

from visualize import visualize_env
from rl_model import explore_env, learn_rl_model, test_rl_model

if __name__ == "__main__":
    env = gym.make("RandomChessEnv-v0", render_mode=None)

    exploration_episodes_list = [30, 100, 300, 1000, 3000, 10000]

    for strategy in "zero", "sequential", "random", "max_entropy", "min_data", "disagreement":
        avg_steps_list = []

        for exploration_episodes in exploration_episodes_list:
            transition_probs = explore_env(env, strategy, num_episodes=exploration_episodes)

            initial_values = np.zeros((env.observation_space.n))
            initial_values[0] = 1

            values = learn_rl_model(env, initial_values, transition_probs, num_episodes=10000)
            
            avg_steps = test_rl_model(env, transition_probs, values)

            avg_steps_list.append(avg_steps)

        plt.plot(exploration_episodes_list, avg_steps_list, label=strategy)

    plt.legend(title="Exploration Strategy")
    plt.xscale("log")
    plt.xlabel("Exploration Episodes")
    plt.ylabel("Avg Test Episode Length")
    plt.title("Exploration of RandomChessEnv w/ Various Exploration Strategies")
    plt.show()
