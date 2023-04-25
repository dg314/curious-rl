from gym.envs.registration import register

def register_envs():
    register(
        id='RandomChessEnv',
        entry_point='gym_envs:RandomChessEnv',
        max_episode_steps=300,
    )
