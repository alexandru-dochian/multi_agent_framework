import numpy as np


def to_tensor_space(
        point: np.array,
        center: np.array,
        tensor_size: np.array,
        vicinity_limit: np.array
) -> np.array:
    x, y = point
    cx, cy = center

    # Normalize to [0, 1] range first
    x_normalized = (x - (cx - vicinity_limit[0])) / (2 * vicinity_limit[0])
    y_normalized = (y - (cy - vicinity_limit[1])) / (2 * vicinity_limit[1])

    # Convert to tensor space indices
    x_tensor = int(round(x_normalized * (tensor_size[1] - 1)))
    y_tensor = int(round(y_normalized * (tensor_size[0] - 1)))

    # Clamp indices to ensure they are within bounds
    x_tensor = np.clip(x_tensor, 0, tensor_size[1] - 1)
    y_tensor = np.clip(y_tensor, 0, tensor_size[0] - 1)

    return x_tensor, y_tensor


def rotate_points(points: np.array, theta: float, center: np.array) -> np.array:
    # Convert theta to radians
    theta_rad = np.radians(theta)

    # Define the rotation matrix
    rotation_matrix = np.array(
        [
            [np.cos(theta_rad), -np.sin(theta_rad)],
            [np.sin(theta_rad), np.cos(theta_rad)],
        ]
    )

    # Translate points to the origin (subtract the center)
    translated_points = points - center

    # Rotate each point
    rotated_points = np.dot(translated_points, rotation_matrix.T)

    # Translate points back to the original center
    rotated_points += center

    return rotated_points
