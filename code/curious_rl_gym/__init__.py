from gym.envs.registration import register

register(
    id='RandomChessEnv-v0',
    entry_point='curious_rl_gym.envs:RandomChessEnv',
    max_episode_steps=300,
)
