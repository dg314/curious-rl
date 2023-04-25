import numpy as np

def overlay_image(background: np.array, foreground: np.array) -> np.array:
    # background: m x n x 3 image
    # foreground: m x n x 4 image

    alpha = np.repeat(np.expand_dims(foreground[:, :, 3], axis=2), 3, axis=2) / 255.0

    return foreground[:, :, :3] * alpha + background * (1 - alpha)