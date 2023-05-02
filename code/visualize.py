import gym

def visualize_env(env: gym.Env, max_steps=300):
    env.reset()

    for _ in range(max_steps):
        action = env.action_space.sample()
        _, _, terminated, _, _ = env.step(action)

        if terminated:
            break

    env.close()
