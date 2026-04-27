from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..ScenarioData import (
    Measurement,
    SensorConfigs,
    SensorErrorCode,
    SensorHealth,
    SensorId,
    SimulationOutput,
)
from .measurement_models import polar_jacobian, polar_measurement


class CoordinateFrameManager:
    def __init__(
        self,
        sensor_configs: SensorConfigs,
        initial_vessel_position: Sequence[float] = (0.0, 0.0),
    ) -> None:
        self.sensor_configs = sensor_configs
        self._vessel_position = self._as_position(initial_vessel_position)
        self._vessel_time: float | None = None

    @classmethod
    def from_simulation_output(cls, simulation_output: SimulationOutput) -> "CoordinateFrameManager":
        return cls(simulation_output.sensor_configs)

    def update_vessel_position(
        self,
        p_north: float,
        p_east: float,
        time: float | None = None,
    ) -> None:
        self._vessel_position = np.array([float(p_north), float(p_east)], dtype=float)
        self._vessel_time = None if time is None else float(time)

    def update_vessel_position_from_gnss(self, measurement: Measurement) -> None:
        if measurement.north_m is None or measurement.east_m is None:
            raise ValueError("GNSS measurement must contain north_m and east_m.")
        self.update_vessel_position(measurement.north_m, measurement.east_m, measurement.time)

    def get_sensor_position(self, sensor_id: str | SensorId) -> tuple[float, float]:
        sensor_key = self._sensor_key(sensor_id)

        if sensor_key in (SensorId.radar.value, SensorId.camera.value):
            return self._static_sensor_position(sensor_key)

        if sensor_key == SensorId.ais.value:
            return tuple(self._vessel_position.tolist())

        if sensor_key == SensorId.gnss.value:
            return tuple(self._vessel_position.tolist())

        raise ValueError(f"Unsupported sensor_id: {sensor_key}")

    def compute_measurement(self, sensor_id: str | SensorId, state: Sequence[float]) -> np.ndarray | None:
        measurement, _, _ = self.compute_observation_safe(sensor_id, state)
        return measurement

    def compute_jacobian(self, sensor_id: str | SensorId, state: Sequence[float]) -> np.ndarray | None:
        _, jacobian, _ = self.compute_observation_safe(sensor_id, state)
        return jacobian

    def compute_observation_safe(
        self,
        sensor_id: str | SensorId,
        state: Sequence[float],
    ) -> tuple[np.ndarray | None, np.ndarray | None, SensorHealth]:
        sensor_key = self._sensor_key(sensor_id)
        config = self._sensor_config(sensor_key)
        if config is None:
            health = SensorHealth(
                is_valid=False,
                code=SensorErrorCode.invalid_input,
                message=f"Unknown sensor_id: {sensor_key}",
            )
            return None, None, health

        health = config.health
        if not health.is_valid:
            return None, None, health

        try:
            if not self.is_within_range_gate(sensor_key, state):
                health = SensorHealth(
                    is_valid=False,
                    code=SensorErrorCode.out_of_range,
                    message="Predicted target is outside sensor range gate.",
                )
                config.health = health
                return None, None, health

            sensor_position = self.get_sensor_position(sensor_key)
            measurement_vector = polar_measurement(state, sensor_position)
            jacobian_matrix = polar_jacobian(state, sensor_position)

            health = SensorHealth(True, SensorErrorCode.ok, "")
            config.health = health
            return measurement_vector, jacobian_matrix, health

        except ValueError as exc:
            health = SensorHealth(False, SensorErrorCode.invalid_geometry, str(exc))
            config.health = health
            return None, None, health

        except Exception as exc:
            health = SensorHealth(False, SensorErrorCode.invalid_input, str(exc))
            config.health = health
            return None, None, health

    def disable_sensor(self, sensor_id: str | SensorId, reason: str = "manual disable") -> None:
        sensor_key = self._sensor_key(sensor_id)
        config = self._sensor_config(sensor_key)
        if config is None:
            return
        config.health = SensorHealth(
            is_valid=False,
            code=SensorErrorCode.manual_disabled,
            message=reason,
        )

    def clear_sensor_error(self, sensor_id: str | SensorId) -> None:
        sensor_key = self._sensor_key(sensor_id)
        config = self._sensor_config(sensor_key)
        if config is None:
            return
        config.health = SensorHealth(True, SensorErrorCode.ok, "")

    def get_sensor_health(self, sensor_id: str | SensorId) -> SensorHealth:
        sensor_key = self._sensor_key(sensor_id)
        config = self._sensor_config(sensor_key)
        if config is None:
            return SensorHealth(
                is_valid=False,
                code=SensorErrorCode.invalid_input,
                message=f"Unknown sensor_id: {sensor_key}",
            )
        return config.health

    def is_within_range_gate(self, sensor_id: str | SensorId, state: Sequence[float]) -> bool:
        max_range = self._sensor_max_range(sensor_id)
        if max_range is None:
            return True
        sensor_position = self.get_sensor_position(sensor_id)
        measurement = polar_measurement(state, sensor_position)
        return bool(measurement[0] <= max_range)

    def get_noise_covariance(self, sensor_id: str | SensorId) -> np.ndarray:
        sensor_key = self._sensor_key(sensor_id)
        config = self._sensor_config(sensor_key)
        if config is None:
            raise ValueError(f"No configuration found for sensor_id: {sensor_key}")

        if sensor_key in (SensorId.radar.value, SensorId.camera.value):
            sigma_range = float(config.sigma_r_m)
            sigma_bearing = float(np.deg2rad(config.sigma_phi_deg))
            return np.diag([sigma_range**2, sigma_bearing**2]).astype(float)

        if sensor_key == SensorId.ais.value:
            sigma_pos = float(config.sigma_pos_m)
            return np.diag([sigma_pos**2, sigma_pos**2]).astype(float)

        if sensor_key == SensorId.gnss.value:
            sigma_pos = float(config.sigma_pos_m)
            return np.diag([sigma_pos**2, sigma_pos**2]).astype(float)

        raise ValueError(f"Unsupported sensor_id: {sensor_key}")

    def _static_sensor_position(self, sensor_key: str) -> tuple[float, float]:
        config = self._sensor_config(sensor_key)
        if config is None:
            raise ValueError(f"No configuration found for sensor_id: {sensor_key}")

        position = getattr(config, "pos_ned", None)
        if position is None:
            if sensor_key in (SensorId.radar.value, SensorId.camera.value):
                return 0.0, 0.0
            raise ValueError(f"Sensor {sensor_key} does not define pos_ned.")

        return self._as_position(position)

    def _sensor_max_range(self, sensor_id: str | SensorId) -> float | None:
        sensor_key = self._sensor_key(sensor_id)
        config = self._sensor_config(sensor_key)
        if config is None:
            raise ValueError(f"No configuration found for sensor_id: {sensor_key}")

        max_range = getattr(config, "range_m", None)
        return None if max_range is None else float(max_range)

    def _enforce_range_gate(self, sensor_id: str | SensorId, state: Sequence[float]) -> None:
        max_range = self._sensor_max_range(sensor_id)
        if max_range is None:
            return

        sensor_position = self.get_sensor_position(sensor_id)
        measurement = polar_measurement(state, sensor_position)
        if float(measurement[0]) > max_range:
            sensor_key = self._sensor_key(sensor_id)
            raise ValueError(
                f"{sensor_key} prediction rejected by range gate: "
                f"range={measurement[0]:.3f} m exceeds max_range={max_range:.3f} m"
            )

    def _sensor_config(self, sensor_key: str) -> Any:
        return getattr(self.sensor_configs, sensor_key, None)

    @staticmethod
    def _sensor_key(sensor_id: str | SensorId) -> str:
        return sensor_id.value if isinstance(sensor_id, SensorId) else str(sensor_id)

    @staticmethod
    def _as_position(position: Sequence[float]) -> np.ndarray:
        position_array = np.asarray(position, dtype=float).reshape(-1)
        if position_array.size < 2:
            raise ValueError("Position must contain at least north and east components.")
        return position_array[:2].astype(float)