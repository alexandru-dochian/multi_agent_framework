import pickle
import string
import time
import random

import numpy as np


def map_to_tensor_space(
    points: np.array, center: tuple, tensor_size: tuple, limit_range
):
    x_center, y_center = center
    tensor_x, tensor_y = tensor_size
    limit_range_diff = limit_range[1] - limit_range[0]

    scale_x = tensor_x / (2 * limit_range_diff)
    scale_y = tensor_y / (2 * limit_range_diff)

    tensor_points = np.ones_like(points)

    if len(tensor_points) == 0:
        return tensor_points

    tensor_points[:, 0] = (points[:, 0] - x_center) * scale_x + tensor_x // 2
    tensor_points[:, 1] = (points[:, 1] - y_center) * scale_y + tensor_y // 2
    return tensor_points


def apply_distribution(field, position, sigma=10, amplitude=1, operation="add"):
    """
    Apply a Gaussian distribution around a given position on a 2D tensor.

    Args:
    - field (np.ndarray): Input 2D array representing the map.
    - position (tuple): Position (x, y) where the distribution will be centered.
    - sigma (float): Standard deviation of the Gaussian distribution.
    - amplitude (float): Amplitude of the Gaussian distribution.
    - operation (str): Operation type ('add', 'subtract', 'replace').

    Returns:
    - np.ndarray: Modified tensor with the Gaussian distribution applied.
    """
    x, y = position
    grid_x, grid_y = np.meshgrid(
        np.arange(field.shape[0]), np.arange(field.shape[1]), indexing="ij"
    )
    gaussian = amplitude * np.exp(
        -((grid_x - x) ** 2 + (grid_y - y) ** 2) / (2 * sigma**2)
    )

    if operation == "add":
        return field + gaussian
    elif operation == "subtract":
        return field - gaussian
    elif operation == "replace":
        mask = gaussian > 0
        field[mask] = gaussian[mask]
        return field
    else:
        raise ValueError("Invalid operation type. Use 'add', 'subtract', or 'replace'.")


def get_current_time() -> int:
    return int(round(time.time() * 1000))


def generate_random_string(length: int = 8) -> str:
    characters = string.ascii_letters + string.digits
    return "".join(random.choice(characters) for _ in range(length))


class DiskUtils:
    @staticmethod
    def store(path, content):
        with path.open("wb") as f:
            pickle.dump(content, f)

    @staticmethod
    def load(path):
        with path.open("rb") as file:
            return pickle.load(file)
