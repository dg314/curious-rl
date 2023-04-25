import random
import gym
from gym import spaces
import numpy as np
import cv2
import subprocess
import os
from pathlib import Path
from curious_rl_gym.helpers.graphics import overlay_image

parent_dir = Path(__file__).parent

def get_path(end: str) -> str:
    return os.path.join(parent_dir, end)

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

        self.pieces = ["king", "queen", "bishop", "knight", "rook", "pawn"]
        self.action_space = spaces.Discrete(len(self.pieces))
        
        chess_pieces_image = cv2.imread(get_path("assets/chess_pieces.png"), cv2.IMREAD_UNCHANGED)
        chess_pieces_image = cv2.resize(chess_pieces_image, (self.square_size * len(self.pieces), self.square_size))
        self.piece_sprites = [
            chess_pieces_image[:, (i * self.square_size):((i + 1) * self.square_size), :]
            for i in range(len(self.pieces))
        ]

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Frame of the video, used if render_mode is set to "video"
        self._frame_num = 0

    def _get_obs(self):
        return {
            "agent": self._agent_location,
            "target": self._target_location
        }
    
    def _get_info(self):
        return {
            "distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)
        }
    
    def _render_frame(self, action):
        if self.render_mode != "video":
            return
        
        if self._frame_num == 0:
            if not os.path.exists("frames"):
                os.mkdir("frames")

            if os.path.exists("random_chess_env.mp4"):
                os.remove("random_chess_env.mp4")
        
        image = np.zeros((self.square_size * self.board_size, self.square_size * self.board_size, 3), dtype=np.int32)

        for i in range(self.board_size):
            for j in range(self.board_size):
                color = [210, 238, 238] if (i + j) % 2 == 0 else [86, 150, 118]

                if i == self._target_location[0] and j == self._target_location[1]:
                    color = [32, 67, 230]
                elif self._prev_agent_location is not None and i == self._prev_agent_location[0] and j == self._prev_agent_location[1]:
                    color = [68, 202, 186]

                image[
                    (i * self.square_size):((i + 1) * self.square_size),
                    (j * self.square_size):((j + 1) * self.square_size),
                    :
                ] = np.array(color)
                
                if i == self._agent_location[0] and j == self._agent_location[1]:
                    image[
                        (i * self.square_size):((i + 1) * self.square_size),
                        (j * self.square_size):((j + 1) * self.square_size),
                        :
                    ] = overlay_image(
                        image[
                            (i * self.square_size):((i + 1) * self.square_size),
                            (j * self.square_size):((j + 1) * self.square_size),
                            :
                        ],
                        self.piece_sprites[action]
                    )

        image_path = f"frames/{self._frame_num}.png"
        cv2.imwrite(image_path, image)

        self._frame_num += 1

    def _close_renderer(self):
        if self.render_mode != "video":
            return
        
        process = subprocess.Popen(f'ffmpeg -framerate {self.metadata["render_fps"]} -i "frames/%d.png" -c:v libx264 -pix_fmt yuv420p random_chess_env.mp4', shell=True)
        process.wait()

        for filename in os.listdir("frames"):
            path = os.path.join("frames", filename)

            if os.path.isfile(path):
                os.remove(path)

        os.rmdir("frames")
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._prev_agent_location = None
        self._agent_location = self.np_random.integers(0, self.board_size, size=2, dtype=int)
        self._target_location = self._agent_location
        
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.board_size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        self._render_frame(0)

        return observation, info
    
    def step(self, action):
        self._prev_agent_location = self._agent_location

        legal_locations = []
        piece_name = self.pieces[action]
        i, j = self._agent_location

        if piece_name == "pawn":
            if i > 0:
                legal_locations.append([i - 1, j])

        if piece_name in ["rook", "queen"]:
            for i_next in range(self.board_size):
                if i_next != i:
                    legal_locations.append([i_next, j])
            
            for j_next in range(self.board_size):
                if j_next != j:
                    legal_locations.append([i, j_next])

        if piece_name in ["bishop", "queen"]:
            coord_sum = i + j

            for i_next in range(min(i, j), max(i, j) + 1):
                if i_next != i:
                    legal_locations.append([i_next, coord_sum - i_next])

            coord_diff = i - j
            
            for i_next in range(max(coord_diff, 0), min(7 + coord_diff, 7) + 1):
                if i_next != i:
                    legal_locations.append([i_next, i_next - coord_diff])

        if piece_name == "king":
            for i_diff, j_diff in [
                (1, 0),
                (1, 1),
                (0, 1),
                (-1, 1),
                (-1, 0),
                (-1, -1),
                (0, -1),
                (1, -1)
            ]:
                i_next, j_next = i + i_diff, j + j_diff

                if i_next >= 0 and i_next < self.board_size and j_next >= 0 and j_next < self.board_size:
                    legal_locations.append([i_next, j_next])

        if piece_name == "knight":
            for i_diff, j_diff in [
                (2, -1),
                (1, -2),
                (-1, -2),
                (-2, -1),
                (-2, 1),
                (-1, 2),
                (1, 2),
                (2, 1)
            ]:
                i_next, j_next = i + i_diff, j + j_diff

                if i_next >= 0 and i_next < self.board_size and j_next >= 0 and j_next < self.board_size:
                    legal_locations.append([i_next, j_next])

        if len(legal_locations) > 0:
            self._agent_location = np.array(random.choice(legal_locations))

        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        self._render_frame(action)

        return observation, reward, terminated, False, info
    
    def close(self):
        self._close_renderer()
