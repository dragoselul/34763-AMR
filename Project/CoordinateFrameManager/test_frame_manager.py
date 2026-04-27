from __future__ import annotations

import copy
from pathlib import Path
import unittest

import numpy as np

try:
    from . import CoordinateFrameManager
    from ..ScenarioLoader import ScenarioLoader
    from ..ScenarioData import SensorConfigs, SensorErrorCode, SensorHealth
except ImportError:
    from Project.CoordinateFrameManager import CoordinateFrameManager
    from Project.ScenarioLoader import ScenarioLoader
    from Project.ScenarioData import SensorConfigs, SensorErrorCode, SensorHealth


class CoordinateFrameManagerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        scenario_paths = sorted((repo_root / "harbour_sim_output").glob("scenario_*.json"))

        loader = ScenarioLoader()
        cls.outputs = [loader.load_scenarios(str(path)) for path in scenario_paths]

    def setUp(self) -> None:
        self.manager = CoordinateFrameManager(copy.deepcopy(self.outputs[0].sensor_configs))

    def test_scenario_loader_returns_typed_sensor_configs(self) -> None:
        for output in self.outputs:
            self.assertIsInstance(output.sensor_configs, SensorConfigs)
            self.assertIsInstance(output.sensor_configs.radar.health, SensorHealth)
            self.assertIsInstance(output.sensor_configs.camera.health, SensorHealth)
            self.assertIsInstance(output.sensor_configs.ais.health, SensorHealth)
            self.assertIsInstance(output.sensor_configs.gnss.health, SensorHealth)
            manager = CoordinateFrameManager(output.sensor_configs)
            self.assertAlmostEqual(manager.get_noise_covariance("radar")[0, 0], output.sensor_configs.radar.sigma_r_m**2)

    def test_camera_measurement_matches_expected_offset(self) -> None:
        measurement = self.manager.compute_measurement("camera", np.array([100.0, 100.0, 0.0, 0.0]))

        self.assertAlmostEqual(measurement[0], 181.1077028, places=6)
        self.assertAlmostEqual(measurement[1], np.deg2rad(-6.3401917), places=6)

    def test_camera_range_gate_rejects_out_of_range_prediction(self) -> None:
        out_of_range_state = np.array([900.0, 300.0, 0.0, 0.0], dtype=float)

        self.assertFalse(self.manager.is_within_range_gate("camera", out_of_range_state))
        measurement = self.manager.compute_measurement("camera", out_of_range_state)
        self.assertIsNone(measurement)
        self.assertEqual(self.manager.get_sensor_health("camera").code, SensorErrorCode.out_of_range)

    def test_jacobian_matches_finite_difference(self) -> None:
        state = np.array([420.0, 250.0, 0.0, 0.0], dtype=float)
        analytical = self.manager.compute_jacobian("radar", state)

        epsilon = 1e-6
        numerical = np.zeros((2, 4), dtype=float)
        for index in range(2):
            perturbed_plus = state.copy()
            perturbed_minus = state.copy()
            perturbed_plus[index] += epsilon
            perturbed_minus[index] -= epsilon
            measurement_plus = self.manager.compute_measurement("radar", perturbed_plus)
            measurement_minus = self.manager.compute_measurement("radar", perturbed_minus)
            self.assertIsNotNone(measurement_plus)
            self.assertIsNotNone(measurement_minus)
            numerical[:, index] = (measurement_plus - measurement_minus) / (2.0 * epsilon)

        np.testing.assert_allclose(analytical[:, :2], numerical[:, :2], rtol=1e-5, atol=1e-5)

    def test_vessel_update_changes_ais_sensor_position(self) -> None:
        self.assertEqual(self.manager.get_sensor_position("ais"), (0.0, 0.0))

        self.manager.update_vessel_position(100.0, 50.0, time=12.0)
        self.assertEqual(self.manager.get_sensor_position("ais"), (100.0, 50.0))

        measurement = self.manager.compute_measurement("ais", np.array([160.0, 140.0, 0.0, 0.0]))
        self.assertAlmostEqual(measurement[0], np.hypot(60.0, 90.0), places=6)
        self.assertAlmostEqual(measurement[1], np.arctan2(90.0, 60.0), places=6)

    def test_manual_disable_skips_sensor_in_safe_mode(self) -> None:
        self.manager.disable_sensor("camera", reason="camera panic")

        measurement, jacobian, health = self.manager.compute_observation_safe(
            "camera",
            np.array([100.0, 100.0, 0.0, 0.0]),
        )

        self.assertIsNone(measurement)
        self.assertIsNone(jacobian)
        self.assertFalse(health.is_valid)
        self.assertEqual(health.code, SensorErrorCode.manual_disabled)

    def test_safe_mode_sets_out_of_range_error(self) -> None:
        measurement, jacobian, health = self.manager.compute_observation_safe(
            "camera",
            np.array([900.0, 300.0, 0.0, 0.0]),
        )

        self.assertIsNone(measurement)
        self.assertIsNone(jacobian)
        self.assertFalse(health.is_valid)
        self.assertEqual(health.code, SensorErrorCode.out_of_range)

    def test_clear_sensor_error_recovers_sensor(self) -> None:
        self.manager.disable_sensor("camera", reason="camera panic")
        self.manager.clear_sensor_error("camera")

        measurement, jacobian, health = self.manager.compute_observation_safe(
            "camera",
            np.array([100.0, 100.0, 0.0, 0.0]),
        )

        self.assertIsNotNone(measurement)
        self.assertIsNotNone(jacobian)
        self.assertTrue(health.is_valid)
        self.assertEqual(health.code, SensorErrorCode.ok)


if __name__ == "__main__":
    unittest.main()