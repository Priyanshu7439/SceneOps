
import pytest

from sceneops.simulator.engine import SimulationStatus
from sceneops.simulator.runner import SimulationRunner
from sceneops.simulator.scenario_factory import (
    create_mass_casualty_scenario,
)


def test_runner_completes_scenario():
    scenario = create_mass_casualty_scenario()
    runner = SimulationRunner(scenario)

    summary = runner.run()

    assert summary.final_status == SimulationStatus.COMPLETED
    assert summary.elapsed_seconds == 600
    assert summary.tick_count == 10


def test_runner_processes_deterioration_events():
    scenario = create_mass_casualty_scenario()
    runner = SimulationRunner(scenario)

    summary = runner.run()

    assert summary.elapsed_seconds == scenario.duration_seconds
    assert runner.engine.patients["P005"].respiratory_rate == 35
    assert runner.engine.patients["P006"].follows_commands is False


def test_runner_allocates_available_ambulances():
    scenario = create_mass_casualty_scenario()
    runner = SimulationRunner(scenario)

    summary = runner.run()

    assert summary.allocation_count == 5


def test_runner_returns_triage_summary():
    scenario = create_mass_casualty_scenario()
    runner = SimulationRunner(scenario)

    summary = runner.run()

    assert sum(summary.triage_counts.values()) == 20


def test_runner_rejects_zero_tick_duration():
    scenario = create_mass_casualty_scenario()

    with pytest.raises(ValueError):
        SimulationRunner(scenario, tick_seconds=0)


def test_runner_cannot_run_twice():
    scenario = create_mass_casualty_scenario()
    runner = SimulationRunner(scenario)

    runner.run()

    with pytest.raises(RuntimeError):
        runner.run()
