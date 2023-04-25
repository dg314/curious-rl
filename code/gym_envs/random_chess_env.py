import random
import gym
from gym import spaces
import numpy as np
import cv2
import subprocess
import os

class RandomChessEnv(gym.Env):
    metadata = {
        "render_modes": ["video"],
        "render_fps": 1
    }

    def __init__(self, render_mode=None, size=8):
        self.board_size = size
        self.square_size = 32 # In pixels
        
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        self.target_x = random.randint(0, self.board_size - 1)
        self.target_y = random.randint(0, self.board_size - 1)

        self.agent_x = random.randint(0, self.board_size - 1)
        self.agent_y = random.randint(0, self.board_size - 1)

        while self.agent_x == self.target_x and self.agent_y == self.target_y:
            self.agent_x = random.randint(0, self.board_size - 1)
            self.agent_y = random.randint(0, self.board_size - 1)

        self.pieces = ["pawn", "bishop", "knight", "rook", "queen", "king"]
        self.action_space = spaces.Discrete(len(self.pieces))

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Frame of the video, used if render_mode is set to "video"
        self._frames = None

    def _get_obs(self):
        return {
            "agent": self._agent_location,
            "target": self._target_location
        }
    
    def _get_info(self):
        return {
            "distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)
        }
    
    def _render_frame(self):
        if self.render_mode != "video":
            return
        
        if self._frames is None:
            if not os.path.exists("frames"):
                os.mkdir("frames")

            if os.path.exists("out.mp4"):
                os.remove("out.mp4")

            self._frames = []
        
        image = np.zeros((self.window_size, self.window_size, 3), dtype=np.int32)

        for i in range(self.board_size):
            for j in range(self.board_size):
                image[
                    (i * self.square_size):((i + 1) * self.square_size),
                    (j * self.square_size):((j + 1) * self.square_size),
                    :
                ] = np.array([238, 238, 210]) if i + j % 2 == 0 else np.array([118, 150, 86])

        image_path = f"frames/{len(self._frames)}.png"
        cv2.imwrite(image_path, image)

    def _close_renderer(self):
        if self.render_mode != "video":
            return
        
        process = subprocess.Popen(f'ffmpeg -framerate {self.metadata["render_fps"]} -i "frames/%d.png" -c:v libx264 -pix_fmt yuv420p out.mp4', shell=True)
        process.wait()

        for filename in os.listdir("frames"):
            path = os.path.join("frames", filename)

            if os.path.isfile(path):
                os.remove(path)

        os.rmdir("frames")
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._agent_location = self.np_random.integers(0, self.board_size, size=2, dtype=int)
        self._target_location = self._agent_location
        
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.board_size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        self._render_frame()

        return observation, info
    
    def step(self, action):
        self._agent_location = self.np_random.integers(
            0, self.board_size, size=2, dtype=int
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        self._render_frame()

        return observation, reward, terminated, False, info
    
    def close(self):
        self._close_renderer()
