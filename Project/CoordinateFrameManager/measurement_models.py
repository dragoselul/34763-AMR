from __future__ import annotations

from typing import Sequence

import numpy as np


def polar_measurement(
    target_state: Sequence[float],
    sensor_position: Sequence[float],
) -> np.ndarray:
    target = np.asarray(target_state, dtype=float).reshape(-1)
    sensor = np.asarray(sensor_position, dtype=float).reshape(-1)

    if target.size < 2:
        raise ValueError("target_state must contain at least north and east position components.")
    if sensor.size < 2:
        raise ValueError("sensor_position must contain at least north and east position components.")

    delta_north = target[0] - sensor[0]
    delta_east = target[1] - sensor[1]
    range_m = float(np.hypot(delta_north, delta_east))

    if range_m == 0.0:
        raise ValueError("range is undefined when target and sensor positions are identical.")

    bearing_rad = float(np.arctan2(delta_east, delta_north))
    return np.array([range_m, bearing_rad], dtype=float)


def polar_jacobian(
    target_state: Sequence[float],
    sensor_position: Sequence[float],
) -> np.ndarray:
    target = np.asarray(target_state, dtype=float).reshape(-1)
    sensor = np.asarray(sensor_position, dtype=float).reshape(-1)

    if target.size < 2:
        raise ValueError("target_state must contain at least north and east position components.")
    if sensor.size < 2:
        raise ValueError("sensor_position must contain at least north and east position components.")

    delta_north = target[0] - sensor[0]
    delta_east = target[1] - sensor[1]
    range_sq = float(delta_north**2 + delta_east**2)

    if range_sq == 0.0:
        raise ValueError("Jacobian is undefined when target and sensor positions are identical.")

    range_m = float(np.sqrt(range_sq))
    return np.array(
        [
            [delta_north / range_m, delta_east / range_m, 0.0, 0.0],
            [-delta_east / range_sq, delta_north / range_sq, 0.0, 0.0],
        ],
        dtype=float,
    )