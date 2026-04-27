from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import json

import numpy as np

from ..ScenarioData import (
    AISConfig,
    CameraConfig,
    GNSSConfig,
    GroundTruths,
    GroundTruth,
    Measurement,
    Measurements,
    RadarConfig,
    SensorConfigs,
    SensorId,
    SimulationOutput,
    VesselPosition,
    VesselPositions,
)


class ScenarioLoader:
    def __init__(self) -> None:
        self.ground_truths: GroundTruths = GroundTruths()
        self.measurements: Measurements = Measurements()
        self.vessel_positions: VesselPositions = VesselPositions()

        self.camera_config: Optional[CameraConfig] = None
        self.radar_config: Optional[RadarConfig] = None
        self.ais_config: Optional[AISConfig] = None
        self.gnss_config: Optional[GNSSConfig] = None

        self.sensor_configs: Optional[SensorConfigs] = None

    def load_scenarios(self, scenario_path: str) -> SimulationOutput:
        path = Path(scenario_path)
        if not path.exists():
            raise FileNotFoundError(f"Scenario file not found: {scenario_path}")

        with path.open("r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)

        self.sensor_configs = self._parse_sensor_configs(data["sensor_configs"])
        self.radar_config = self.sensor_configs.radar
        self.camera_config = self.sensor_configs.camera
        self.ais_config = self.sensor_configs.ais
        self.gnss_config = self.sensor_configs.gnss

        ground_truth_arrays = self._parse_ground_truth_arrays(data["ground_truth"])
        self.ground_truths = self._to_ground_truth_objects(ground_truth_arrays)

        self.measurements = Measurements(
            measurements=[
                self._parse_measurement(measurement)
                for measurement in data.get("measurements", [])
            ]
        )

        vessel_positions_array = np.asarray(data.get("vessel_positions", []), dtype=float)
        self.vessel_positions = self._to_vessel_positions(vessel_positions_array)

        ground_truth_times = self._derive_ground_truth_times(ground_truth_arrays)
        vessel_times = (
            vessel_positions_array[:, 0]
            if vessel_positions_array.size
            else np.array([], dtype=float)
        )

        return SimulationOutput(
            scenario_name=str(data["scenario_name"]),
            dt_true=float(data["dt_true"]),
            t_end=float(data["t_end"]),
            ground_truth=ground_truth_arrays,
            ground_truth_times=ground_truth_times,
            measurements=self.measurements.measurements,
            vessel_positions=vessel_positions_array,
            vessel_times=vessel_times,
            sensor_configs=self.sensor_configs,
        )

    def _parse_sensor_configs(self, sensor_configs: Dict[str, Any]) -> SensorConfigs:
        radar = RadarConfig(
            pos_ned=[float(value) for value in sensor_configs["radar"]["pos_ned"]],
            range_m=float(sensor_configs["radar"]["range_m"]),
            fov_deg=int(sensor_configs["radar"]["fov_deg"]),
            rate_hz=float(sensor_configs["radar"]["rate_hz"]),
            sigma_r_m=float(sensor_configs["radar"]["sigma_r_m"]),
            sigma_phi_deg=float(sensor_configs["radar"]["sigma_phi_deg"]),
            pd=float(sensor_configs["radar"]["pd"]),
            lambda_fa=float(sensor_configs["radar"]["lambda_fa"]),
        )

        camera = CameraConfig(
            pos_ned=[float(value) for value in sensor_configs["camera"]["pos_ned"]],
            boresight_deg=float(sensor_configs["camera"]["boresight_deg"]),
            range_m=float(sensor_configs["camera"]["range_m"]),
            fov_deg=int(sensor_configs["camera"]["fov_deg"]),
            rate_hz=float(sensor_configs["camera"]["rate_hz"]),
            sigma_r_m=float(sensor_configs["camera"]["sigma_r_m"]),
            sigma_phi_deg=float(sensor_configs["camera"]["sigma_phi_deg"]),
            pd=float(sensor_configs["camera"]["pd"]),
            lambda_fa=float(sensor_configs["camera"]["lambda_fa"]),
        )

        ais = AISConfig(
            range_m=float(sensor_configs["ais"]["range_m"]),
            interval_s=float(sensor_configs["ais"]["interval_s"]),
            sigma_pos_m=float(sensor_configs["ais"]["sigma_pos_m"]),
            pd=float(sensor_configs["ais"]["pd"]),
        )

        gnss = GNSSConfig(
            sigma_pos_m=float(sensor_configs["gnss"]["sigma_pos_m"]),
            rate_hz=float(sensor_configs["gnss"]["rate_hz"]),
        )

        return SensorConfigs(radar=radar, camera=camera, ais=ais, gnss=gnss)

    def _parse_ground_truth_arrays(self, raw_ground_truth: Dict[str, Any]) -> Dict[int, np.ndarray]:
        parsed_ground_truth: Dict[int, np.ndarray] = {}

        for target_id_string, rows in raw_ground_truth.items():
            target_rows = np.asarray(rows, dtype=float)
            if target_rows.ndim != 2 or target_rows.shape[1] != 5:
                raise ValueError(
                    "ground_truth rows must be Nx5 with [time, p_north, p_east, v_north, v_east]."
                )
            parsed_ground_truth[int(target_id_string)] = target_rows

        return parsed_ground_truth

    def _to_ground_truth_objects(
        self,
        ground_truth_arrays: Dict[int, np.ndarray],
    ) -> GroundTruths:
        ground_truth_objects = {}

        for target_id, rows in ground_truth_arrays.items():
            ground_truth_objects[target_id] = [
                GroundTruth(
                    time_stamp=float(row[0]),
                    p_north=float(row[1]),
                    p_east=float(row[2]),
                    v_north=float(row[3]),
                    v_east=float(row[4]),
                )
                for row in rows
            ]

        return GroundTruths(ground_truths=ground_truth_objects)

    def _parse_measurement(self, raw_measurement: Dict[str, Any]) -> Measurement:
        sensor_identifier = str(raw_measurement["sensor_id"])
        if sensor_identifier not in SensorId._value2member_map_:
            raise ValueError(f"Unknown sensor_id: {sensor_identifier}")

        return Measurement(
            sensor_id=sensor_identifier,
            time=float(raw_measurement["time"]),
            is_false_alarm=bool(raw_measurement["is_false_alarm"]),
            target_id=int(raw_measurement["target_id"]),
            range_m=self._optional_float(raw_measurement.get("range_m")),
            bearing_rad=self._optional_float(raw_measurement.get("bearing_rad")),
            north_m=self._optional_float(raw_measurement.get("north_m")),
            east_m=self._optional_float(raw_measurement.get("east_m")),
        )

    def _to_vessel_positions(self, vessel_positions_array: np.ndarray) -> VesselPositions:
        if vessel_positions_array.size == 0:
            return VesselPositions()

        if vessel_positions_array.ndim != 2 or vessel_positions_array.shape[1] != 3:
            raise ValueError("vessel_positions rows must be Nx3 with [time, p_north, p_east].")

        return VesselPositions(
            positions=[
                VesselPosition(
                    time_stamp=float(row[0]),
                    p_north=float(row[1]),
                    p_east=float(row[2]),
                )
                for row in vessel_positions_array
            ]
        )

    def _derive_ground_truth_times(self, ground_truth_arrays: Dict[int, np.ndarray]) -> np.ndarray:
        if not ground_truth_arrays:
            return np.array([], dtype=float)

        all_times = np.concatenate([rows[:, 0] for rows in ground_truth_arrays.values()])
        return np.unique(all_times)

    @staticmethod
    def _optional_float(value: Any) -> Optional[float]:
        return None if value is None else float(value)
