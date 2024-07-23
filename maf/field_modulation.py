import numpy as np
from scipy.ndimage import zoom

REWARD_FUNCTIONS = {
    "sum": np.sum,
    "avg": np.mean,
    "min": np.min,
    "max": np.max,
}


def apply_distribution(
        field: np.array,
        position: np.array,
        sigma: float = 10,
        amplitude: float = 1,
        operation="add",
):
    x, y = position
    grid_x, grid_y = np.meshgrid(
        np.arange(field.shape[0]), np.arange(field.shape[1]), indexing="ij"
    )
    gaussian = amplitude * np.exp(
        -((grid_x - x) ** 2 + (grid_y - y) ** 2) / (2 * sigma ** 2)
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


def to_field_position(
        point: np.array,
        center: list[float],
        field_size: list[float],
        vicinity_limit: list[float],
) -> np.array:
    limit_x, limit_y = vicinity_limit
    field_x, field_y = field_size
    x, y = point
    cx, cy = center

    # Normalize to [0, 1] range first
    x_normalized = (x - (cx - limit_x)) / (2 * limit_x)
    y_normalized = (y - (cy - limit_y)) / (2 * limit_y)

    # Convert to tensor space indices
    x_tensor = int(round(x_normalized * (field_x - 1)))
    y_tensor = int(round(y_normalized * (field_y - 1)))

    # Clamp indices to ensure they are within bounds
    x_tensor = np.clip(x_tensor, 0, field_x - 1)
    y_tensor = np.clip(y_tensor, 0, field_y - 1)

    return x_tensor, y_tensor


def filter_points_in_vicinity(
        points: np.array, center_point: list[float], vicinity_limit: list[float]
) -> np.array:
    # vicinity limit
    x_min, x_max = (
        center_point[0] - vicinity_limit[0],
        center_point[0] + vicinity_limit[0],
    )
    y_min, y_max = (
        center_point[1] - vicinity_limit[1],
        center_point[1] + vicinity_limit[1],
    )

    return points[
        (points[:, 0] >= x_min)
        & (points[:, 0] <= x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] <= y_max)
        ]


def rotate_points(
        points: np.array, center: np.array = np.array([0, 0]), theta: float = 0
) -> np.array:
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


def patch_value_outside_vicinity_limit(
        field: np.array,
        center: np.array,
        vicinity_limit: np.array,
        space_limits: np.array,
        field_size: np.array = np.array([84, 84]),
        value: float = -1,
):
    center_x, center_y = center

    field_limits = {
        "x_min": to_field_position(
            [space_limits["x_min"], center_y],
            center,
            field_size,
            vicinity_limit,
        )[0],
        "x_max": to_field_position(
            [space_limits["x_max"], center_y],
            center,
            field_size,
            vicinity_limit,
        )[0],
        "y_min": to_field_position(
            [center_x, space_limits["y_min"]],
            center,
            field_size,
            vicinity_limit,
        )[1],
        "y_max": to_field_position(
            [center_x, space_limits["y_max"]],
            center,
            field_size,
            vicinity_limit,
        )[1],
    }

    field[:, : field_limits["x_min"]] = value
    field[:, field_limits["x_max"] + 1:] = value
    field[: field_limits["y_min"], :] = value
    field[field_limits["y_max"] + 1:, :] = value

    return field


def clip_and_resize(
        field: np.array, field_size: list[float], clip_size_factor: float = 2
):
    field_x, field_y = field_size

    clip_size_x = field_x // clip_size_factor
    clip_size_y = field_y // clip_size_factor

    start_x = field_x // 2 - clip_size_x // 2
    start_y = field_y // 2 - clip_size_y // 2

    # Clip
    clipped_tensor = field[
                     start_x: start_x + clip_size_x, start_y: start_y + clip_size_y
                     ]

    # Resize
    return zoom(
        input=clipped_tensor,
        zoom=(field_x / clip_size_x, field_y / clip_size_y),
        order=1,
    )


def pooling_to_3x3(field: np.array, func_name="sum"):
    if func_name not in REWARD_FUNCTIONS:
        raise ValueError(
            f"Function {func_name} not recognized. Choose from {list(REWARD_FUNCTIONS.keys())}"
        )

    func = REWARD_FUNCTIONS[func_name]

    original_shape = field.shape

    block_size = (original_shape[0] // 3, original_shape[1] // 3)

    result = np.zeros((3, 3), dtype=field.dtype)
    for i in range(3):
        for j in range(3):
            block = field[
                    i * block_size[0]: (i + 1) * block_size[0],
                    j * block_size[1]: (j + 1) * block_size[1],
                    ]
            result[i, j] = func(block)

    return result
