
import pytest

from sceneops.simulator.clock import SimulationClock


def test_clock_starts_at_zero():
    clock = SimulationClock()

    assert clock.elapsed_seconds == 0
    assert clock.elapsed_minutes == 0
    assert clock.tick_count == 0


def test_clock_accepts_custom_start_time():
    clock = SimulationClock(start_seconds=120)

    assert clock.elapsed_seconds == 120
    assert clock.elapsed_minutes == 2


def test_clock_advances_by_default_60_seconds():
    clock = SimulationClock()

    snapshot = clock.advance()

    assert snapshot.elapsed_seconds == 60
    assert snapshot.tick_count == 1


def test_clock_advances_by_custom_duration():
    clock = SimulationClock()

    clock.advance(90)

    assert clock.elapsed_seconds == 90
    assert clock.elapsed_minutes == 1.5


def test_multiple_advances_accumulate_time():
    clock = SimulationClock()

    clock.advance(60)
    clock.advance(120)
    clock.advance(30)

    assert clock.elapsed_seconds == 210
    assert clock.tick_count == 3


def test_clock_rejects_negative_start_time():
    with pytest.raises(ValueError):
        SimulationClock(start_seconds=-10)


def test_clock_rejects_negative_advance():
    clock = SimulationClock()

    with pytest.raises(ValueError):
        clock.advance(-10)


def test_clock_rejects_zero_advance():
    clock = SimulationClock()

    with pytest.raises(ValueError):
        clock.advance(0)


def test_clock_rejects_non_integer_advance():
    clock = SimulationClock()

    with pytest.raises(TypeError):
        clock.advance(30.5)


def test_clock_rejects_boolean_advance():
    clock = SimulationClock()

    with pytest.raises(TypeError):
        clock.advance(True)


def test_snapshot_is_immutable():
    clock = SimulationClock()
    snapshot = clock.advance(60)

    with pytest.raises(AttributeError):
        snapshot.elapsed_seconds = 120


def test_reset_returns_clock_to_zero():
    clock = SimulationClock(start_seconds=120)
    clock.advance(60)

    snapshot = clock.reset()

    assert snapshot.elapsed_seconds == 0
    assert snapshot.tick_count == 0


def test_clock_does_not_change_without_advance():
    clock = SimulationClock()

    first_snapshot = clock.snapshot()
    second_snapshot = clock.snapshot()

    assert first_snapshot == second_snapshot
