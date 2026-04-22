from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class RadarConfig:
    pos_ned: List[float]
    range_m: float
    fov_deg: int
    rate_hz: float
    sigma_r_m: float
    sigma_phi_deg: float
    pd: float
    lambda_fa: float


@dataclass
class CameraConfig:
    pos_ned: List[float]
    boresight_deg: float
    range_m: float
    fov_deg: int
    rate_hz: float
    sigma_r_m: float
    sigma_phi_deg: float
    pd: float
    lambda_fa: float


@dataclass
class AISConfig:
    range_m: float
    interval_s: float
    sigma_pos_m: float
    pd: float


@dataclass
class GNSSConfig:
    sigma_pos_m: float
    rate_hz: float


@dataclass
class SensorConfigs:
    radar: RadarConfig
    camera: CameraConfig
    ais: AISConfig
    gnss: GNSSConfig


class SensorId(str, Enum):
    radar = "radar"
    camera = "camera"
    ais = "ais"
    gnss = "gnss"


@dataclass
class GroundTruth:
    time_stamp: float
    p_north: float
    p_east: float
    v_north: float
    v_east: float


@dataclass
class GroundTruths:
    ground_truths: Dict[int, List[GroundTruth]] = field(default_factory=dict)


@dataclass
class VesselPosition:
    time_stamp: float
    p_north: float
    p_east: float


@dataclass
class VesselPositions:
    positions: List[VesselPosition] = field(default_factory=list)


@dataclass
class Measurement:
    sensor_id: str
    time: float
    is_false_alarm: bool
    target_id: int
    range_m: Optional[float] = None
    bearing_rad: Optional[float] = None
    north_m: Optional[float] = None
    east_m: Optional[float] = None


@dataclass
class Measurements:
    measurements: List[Measurement] = field(default_factory=list)


@dataclass
class SimulationOutput:
    scenario_name: str
    dt_true: float
    t_end: float
    ground_truth: Dict[int, np.ndarray]
    ground_truth_times: np.ndarray
    measurements: List[Measurement]
    vessel_positions: np.ndarray
    vessel_times: np.ndarray
    sensor_configs: Dict[str, Any]
