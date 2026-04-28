from __future__ import annotations

from pathlib import Path

# Support both `python -m Project` and running this file directly from VS Code.
if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from Project.CoordinateFrameManager import CoordinateFrameManager
    from Project.ScenarioData import SensorId
    from Project.ScenarioLoader import ScenarioLoader
else:
    from .CoordinateFrameManager import CoordinateFrameManager
    from .ScenarioData import SensorId
    from .ScenarioLoader import ScenarioLoader


def _print_frame_manager_snapshot(output, manager: CoordinateFrameManager) -> None:
    target_ids = sorted(output.ground_truth.keys())
    if not target_ids:
        print("No target state found for h(x)/H snapshot.")
        return

    target_id = target_ids[0]
    state = output.ground_truth[target_id][0, 1:5]

    gnss_measurements = [
        measurement
        for measurement in output.measurements
        if measurement.sensor_id == SensorId.gnss.value
        and measurement.north_m is not None
        and measurement.east_m is not None
    ]
    if gnss_measurements:
        manager.update_vessel_position_from_gnss(gnss_measurements[-1])

    print(f"Target {target_id} state [pN, pE, vN, vE]: {state}")
    for sensor_id in (SensorId.radar.value, SensorId.camera.value, SensorId.ais.value):
        measurement_vector, jacobian, health = manager.compute_observation_safe(sensor_id, state)
        if not health.is_valid:
            print(
                f"{sensor_id.upper()} skipped: "
                f"code={health.code.value}, message={health.message}"
            )
            continue

        noise_covariance = manager.get_noise_covariance(sensor_id)

        print(
            f"{sensor_id.upper()} h(x): range={measurement_vector[0]:.3f} m, "
            f"bearing={measurement_vector[1]:.6f} rad"
        )
        print(f"{sensor_id.upper()} H:\n{jacobian}")
        print(f"{sensor_id.upper()} R:\n{noise_covariance}")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    for scenario_path in sorted((repo_root / "harbour_sim_output").glob("scenario_*.json")):
        loader = ScenarioLoader()
        output = loader.load_scenarios(str(scenario_path))
        manager = CoordinateFrameManager.from_simulation_output(output)

        print(f"Scenario: {output.scenario_name}")
        print(f"Duration: {output.t_end}s, dt_true: {output.dt_true}s")
        print(f"Targets in ground truth: {len(output.ground_truth)}")
        print(f"Measurements: {len(output.measurements)}")
        print(f"Vessel position samples: {output.vessel_positions.shape[0]}")
        _print_frame_manager_snapshot(output, manager)
        print("-" * 40)


if __name__ == "__main__":
    main()
