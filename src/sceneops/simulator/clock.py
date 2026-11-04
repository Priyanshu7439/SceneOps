
from dataclasses import dataclass


@dataclass(frozen=True)
class ClockSnapshot:
    """Immutable snapshot of simulation time."""

    elapsed_seconds: int
    tick_count: int

    @property
    def elapsed_minutes(self) -> float:
        return self.elapsed_seconds / 60


class SimulationClock:
    """
    Deterministic simulation clock.

    Time advances only when advance() is explicitly called.
    It does not depend on the computer's wall-clock time.
    """

    def __init__(self, start_seconds: int = 0) -> None:
        self._validate_seconds(start_seconds, "start_seconds")

        self._elapsed_seconds = start_seconds
        self._tick_count = 0

    @staticmethod
    def _validate_seconds(value: int, field_name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{field_name} must be an integer.")

        if value < 0:
            raise ValueError(f"{field_name} cannot be negative.")

    @property
    def elapsed_seconds(self) -> int:
        return self._elapsed_seconds

    @property
    def elapsed_minutes(self) -> float:
        return self._elapsed_seconds / 60

    @property
    def tick_count(self) -> int:
        return self._tick_count

    def advance(self, seconds: int = 60) -> ClockSnapshot:
        """Advance simulation time by a positive number of seconds."""

        self._validate_seconds(seconds, "seconds")

        if seconds == 0:
            raise ValueError("Clock advance must be greater than zero.")

        self._elapsed_seconds += seconds
        self._tick_count += 1

        return self.snapshot()

    def snapshot(self) -> ClockSnapshot:
        """Return an immutable snapshot of the current clock state."""

        return ClockSnapshot(
            elapsed_seconds=self._elapsed_seconds,
            tick_count=self._tick_count,
        )

    def reset(self) -> ClockSnapshot:
        """Reset the simulation clock to its initial state."""

        self._elapsed_seconds = 0
        self._tick_count = 0

        return self.snapshot()
