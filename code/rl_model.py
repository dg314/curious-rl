from typing import Tuple
import curious_rl_gym
import gym
import random
import numpy as np

def check_valid_env(env: gym.Env):
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise Exception("This algorithm requires a discrete action space.")
    
    if not isinstance(env.observation_space, gym.spaces.Discrete):
        raise Exception("This algorithm requires a discrete observation space.")
    
def explore_env(env: gym.Env, strategy: str, num_episodes=1000, max_steps=300, num_disagreement_models=10) -> np.array:
    # strategy
    # - "zero": always select action 0
    # - "sequential": start with action 0, and then repeatedly select action (a + 1) % 6 after action a
    # - "random": select an action uniformly randomly 
    # - "max_entropy": choose the action with max entropy based on the current transition_probs
    # - "min_data": choose the action a for which state, action pair (s, a) has been explored the fewest times
    # - "disagreement": choose the action for which the disagreement between n prediction models is greatest

    check_valid_env(env)

    if strategy not in ["zero", "sequential", "random", "max_entropy", "min_data", "disagreement"]:
        raise Exception("Invalid strategy.")

    num_actions = env.action_space.n
    num_states = env.observation_space.n
    
    state_action_transition_counts = np.ones((num_states, num_actions, num_states))
    state_action_counts = np.ones((num_states, num_actions)) * num_states
    action = 0

    if strategy == "disagreement":
        disagreement_models = np.array([
            np.random.uniform(size=(num_states, num_actions, num_states))
            for _ in range(num_disagreement_models)
        ])

    for _ in range(num_episodes):
        state, _ = env.reset()

        for _ in range(max_steps):
            if strategy == "sequential":
                action = (action + 1) % num_actions
            elif strategy == "random":
                action = random.randint(0, num_actions - 1)
            elif strategy == "max_entropy":
                transition_probs = (state_action_transition_counts.T / state_action_counts.T).T
                action_entropies = -np.sum(transition_probs[state, :, :] * np.log(transition_probs[state, :, :]), axis=1)
                action = np.argmax(action_entropies)
            elif strategy == "min_data":
                action = np.argmin(state_action_counts[state])
            elif strategy == "disagreement":
                action_variances = np.sum(np.var(disagreement_models[:, state, :, :], axis=0), axis=1)
                action = np.argmax(action_variances)

            next_state, _, terminated, _, _ = env.step(action)

            state_action_transition_counts[state, action, next_state] += 1
            state_action_counts[state, action] += 1

            if strategy == "disagreement":
                action_one_hot = np.zeros((num_actions))
                action_one_hot[action] = 1
                prob_ratio = 1 / state_action_counts[state, action]
                old_probs = disagreement_models[:, state, :, next_state]
                new_probs = old_probs * (1 - prob_ratio) + action_one_hot * prob_ratio
                disagreement_models[:, state, :, next_state] = new_probs

            state = next_state

            if terminated:
                break

        env.close()

    transition_probs = (state_action_transition_counts.T / state_action_counts.T).T

    return transition_probs

def learn_rl_model(env: gym.Env, initial_values: np.array, transition_probs: np.array, gamma=0.95, alpha=0.5, num_episodes=1000, max_steps=300) -> np.array:
    # transition_probs: Matrix of probabilities P(s, a, s'), with shape (num_states x num_actions x num_states)

    check_valid_env(env)

    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise Exception("This algorithm requires a discrete action space.")
    
    if not isinstance(env.observation_space, gym.spaces.Discrete):
        raise Exception("This algorithm requires a discrete observation space.")
    
    values = np.copy(initial_values)

    for _ in range(num_episodes):
        state, _ = env.reset()

        for _ in range(max_steps):
            action = np.argmax(transition_probs[state, :, :] @ values)

            next_state, reward, terminated, _, _ = env.step(action)

            values[state] += alpha * (reward + gamma * np.max(transition_probs[next_state, :, :] @ values) - values[state])

            state = next_state

            if terminated:
                break

        env.close()
        
    return values

def test_rl_model(env: gym.Env, transition_probs: np.array, values: np.array, num_episodes=1000, max_steps=300) -> int:
    check_valid_env(env)

    total_steps = 0

    for _ in range(num_episodes):
        state, _ = env.reset()

        for _ in range(max_steps):
            total_steps += 1

            action = np.argmax(transition_probs[state, :, :] @ values)

            state, _, terminated, _, _ = env.step(action)

            if terminated:
                break

        env.close()

    return total_steps / num_episodes
    