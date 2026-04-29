from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from Project.CoordinateFrameManager import CoordinateFrameManager
    from Project.EKF.EKFTracker import EKFTracker
    from Project.ScenarioData import SensorId
    from Project.ScenarioLoader import ScenarioLoader
else:
    from .EKFTracker import EKFTracker
    from ..CoordinateFrameManager import CoordinateFrameManager
    from ..ScenarioData import SensorId
    from ..ScenarioLoader import ScenarioLoader


CHI2_2DOF_95_PERCENT_LOWER = 0.05063561596857975
CHI2_2DOF_95_PERCENT_UPPER = 7.377758908227871
SCENARIO_A_TRUE_X0 = np.array([800.0, 600.0, -2.0, -1.0], dtype=float)
SCENARIO_A_DURATION_S = 120.0
SCENARIO_A_RADAR_PD = 0.95
SCENARIO_A_RADAR_LAMBDA_FA = 3.0
SCENARIO_A_RMSE_LIMIT_M = 12.0
SCENARIO_A_NIS_FRACTION_LIMIT = 0.90


class CVMotionModel:
    """
    Constant-velocity model copied from the notebook so the test stays aligned
    with the motion assumptions used to generate Scenario A.
    """

    def __init__(self, dt: float, sigma_a: float = 0.05) -> None:
        self.dt = float(dt)
        self.sigma_a = float(sigma_a)

        dt2 = self.dt**2
        dt3 = self.dt**3
        dt4 = self.dt**4
        q = self.sigma_a**2

        self._F = np.array(
            [
                [1.0, 0.0, self.dt, 0.0],
                [0.0, 1.0, 0.0, self.dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        self._Q = q * np.array(
            [
                [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
                [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
                [dt3 / 2.0, 0.0, dt2, 0.0],
                [0.0, dt3 / 2.0, 0.0, dt2],
            ],
            dtype=float,
        )

    @property
    def F(self) -> np.ndarray:
        return self._F.copy()

    @property
    def Q(self) -> np.ndarray:
        return self._Q.copy()

class EKFTrackerScenarioATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        loader = ScenarioLoader()
        cls.output = loader.load_scenarios(str(repo_root / "harbour_sim_output" / "scenario_A.json"))
        cls.ground_truth = cls.output.ground_truth[0]
        cls.radar_hits = [
            measurement
            for measurement in cls.output.measurements
            if measurement.sensor_id == SensorId.radar.value
            and not measurement.is_false_alarm
            and measurement.target_id == 0
        ]

    def setUp(self) -> None:
        self.frame_manager = CoordinateFrameManager(copy.deepcopy(self.output.sensor_configs))

    def test_scenario_a_true_radar_track_meets_rmse_and_nis_targets(self) -> None:
        """
        Simple Scenario A test:
        1. Use the Scenario A start state from the assignment slide as x0.
        2. Run predict/update on the true radar hits.
        3. Print RMSE and NIS diagnostics.
        4. Check the task thresholds.
        """
        radar_hits = self.radar_hits

        # These values come directly from the Scenario A description in the assignment.
        x0 = SCENARIO_A_TRUE_X0.copy()
        P0 = np.diag([25.0**2, 25.0**2, 5.0**2, 5.0**2]).astype(float)
        dt = radar_hits[0].time

        tracker = EKFTracker(
            x0=x0,
            P0=P0,
            motion_model=CVMotionModel(dt=dt, sigma_a=0.05),
            frame_manager=self.frame_manager,
        )

        # Check that the loaded scenario matches the slide.
        np.testing.assert_allclose(self.ground_truth[0, 1:5], SCENARIO_A_TRUE_X0, atol=1e-9)
        self.assertAlmostEqual(self.output.t_end, SCENARIO_A_DURATION_S)
        self.assertAlmostEqual(self.output.sensor_configs.radar.pd, SCENARIO_A_RADAR_PD)
        self.assertAlmostEqual(self.output.sensor_configs.radar.lambda_fa, SCENARIO_A_RADAR_LAMBDA_FA)

        estimate_times = [0.0]
        previous_time = 0.0
        nis_values = []

        # Run the EKF on the true radar hits.
        for hit in radar_hits:
            dt = hit.time - previous_time
            tracker.motion_model = CVMotionModel(dt=dt, sigma_a=0.05)
            tracker.predict()

            z = np.array([hit.range_m, hit.bearing_rad], dtype=float)

            z_pred = tracker.frame_manager.compute_measurement("radar", tracker.x)
            H = tracker.frame_manager.compute_jacobian("radar", tracker.x)
            R = tracker.frame_manager.get_noise_covariance("radar")
            innovation = z - z_pred
            innovation[1] = np.arctan2(np.sin(innovation[1]), np.cos(innovation[1]))
            S = H @ tracker.P @ H.T + R
            nis = float(innovation.T @ np.linalg.solve(S, innovation))
            nis_values.append(nis)

            tracker.update(z)

            estimate_times.append(hit.time)
            previous_time = hit.time

        estimates = np.asarray(tracker.state_history, dtype=float)

        # Match each EKF estimate with the ground-truth row at the same time.
        aligned_truth = []
        gt_times = self.ground_truth[:, 0]
        for time in estimate_times:
            index = np.where(np.isclose(gt_times, time))[0]
            self.assertEqual(len(index), 1, f"Expected one ground-truth row at t={time}")
            aligned_truth.append(self.ground_truth[index[0]])
        aligned_truth = np.asarray(aligned_truth, dtype=float)

        # The assignment says steady-state RMSE, so skip the first five radar scans.
        steady_state_start = 6
        position_errors = estimates[steady_state_start:, :2] - aligned_truth[steady_state_start:, 1:3]
        rmse = float(np.sqrt(np.mean(np.sum(position_errors**2, axis=1))))

        nis_in_bounds = [
            CHI2_2DOF_95_PERCENT_LOWER <= nis <= CHI2_2DOF_95_PERCENT_UPPER
            for nis in nis_values
        ]
        nis_fraction_in_bounds = float(np.mean(nis_in_bounds))

        first_truth = aligned_truth[0]
        last_truth = aligned_truth[-1]
        first_estimate = estimates[0]
        last_estimate = estimates[-1]

        print()
        print("=== Scenario A EKF diagnostics ===")
        print(f"Scenario name: {self.output.scenario_name}")
        print("Purpose: validate single-sensor EKF, coordinate frame transform, and NIS consistency")
        print(f"Ground-truth dt_true: {self.output.dt_true:.3f} s")
        print(f"Scenario duration: {self.output.t_end:.1f} s")
        print(f"True radar detections used: {len(radar_hits)}")
        print(f"Radar config: pd={self.output.sensor_configs.radar.pd}, lambda_fa={self.output.sensor_configs.radar.lambda_fa}")
        print(f"Update times [s]: {estimate_times}")
        print("Track confirmation criterion from task: within 5 radar scans")
        print("Track confirmation status here: not checked in this test")
        print(f"Assignment start state x0 [pN, pE, vN, vE]: {np.array2string(SCENARIO_A_TRUE_X0, precision=3)}")
        print(f"Initial x0 [pN, pE, vN, vE]: {np.array2string(x0, precision=3)}")
        print(f"Initial P0 diagonal: {np.array2string(np.diag(P0), precision=3)}")
        print(f"Initial P0 matrix:\n{np.array2string(P0, precision=3)}")
        print("Motion model: CV, sigma_a=0.05")
        print(f"Steady-state RMSE starts at estimate index: {steady_state_start}")
        print(f"First estimate: {np.array2string(first_estimate, precision=3)}")
        print(f"First aligned truth row [t, pN, pE, vN, vE]: {np.array2string(first_truth, precision=3)}")
        print(f"Last estimate: {np.array2string(last_estimate, precision=3)}")
        print(f"Last aligned truth row [t, pN, pE, vN, vE]: {np.array2string(last_truth, precision=3)}")
        print(f"Steady-state position RMSE: {rmse:.3f} m  (task target: < {SCENARIO_A_RMSE_LIMIT_M:.1f} m)")
        print(
            "NIS in 95% chi^2(2) bounds: "
            f"{sum(nis_in_bounds)}/{len(nis_values)} = {nis_fraction_in_bounds:.3%}  "
            f"(task target: >= {SCENARIO_A_NIS_FRACTION_LIMIT:.1%})"
        )
        print(
            "NIS bounds used: "
            f"[{CHI2_2DOF_95_PERCENT_LOWER:.6f}, {CHI2_2DOF_95_PERCENT_UPPER:.6f}]"
        )
        print(f"All NIS values: {np.array2string(np.asarray(nis_values), precision=3)}")
        print("=== End diagnostics ===")

        self.assertLess(
            rmse,
            SCENARIO_A_RMSE_LIMIT_M,
            msg=(
                f"Scenario A RMSE requirement failed: rmse={rmse:.3f} m, "
                f"required < {SCENARIO_A_RMSE_LIMIT_M:.1f} m"
            ),
        )
        self.assertGreaterEqual(
            nis_fraction_in_bounds,
            SCENARIO_A_NIS_FRACTION_LIMIT,
            msg=(
                "Scenario A NIS consistency requirement failed: "
                f"fraction_in_bounds={nis_fraction_in_bounds:.3%}, "
                f"required >= {SCENARIO_A_NIS_FRACTION_LIMIT:.1%}"
            ),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
