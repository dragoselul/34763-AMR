from __future__ import annotations

from pathlib import Path

# Support both `python -m Project` and running this file directly from VS Code.
if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from Project.ScenarioLoader import ScenarioLoader
else:
    from .ScenarioLoader import ScenarioLoader


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    scenario_path = repo_root / "harbour_sim_output" / "scenario_A.json"

    loader = ScenarioLoader()
    output = loader.load_scenarios(str(scenario_path))

    print(f"Scenario: {output.scenario_name}")
    print(f"Duration: {output.t_end}s, dt_true: {output.dt_true}s")
    print(f"Targets in ground truth: {len(output.ground_truth)}")
    print(f"Measurements: {len(output.measurements)}")
    print(f"Vessel position samples: {output.vessel_positions.shape[0]}")


if __name__ == "__main__":
    main()
