"""The simulation clock. One tick == one day."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Clock:
    """Tracks elapsed simulation time in days, with a derived year."""

    day: int = 0
    days_per_year: int = 365

    def advance(self, days: int = 1) -> int:
        """Move the clock forward and return the new day number."""
        if days < 0:
            raise ValueError("Cannot advance the clock by a negative number of days.")
        self.day += days
        return self.day

    def reset(self) -> None:
        """Return to day zero."""
        self.day = 0

    @property
    def year(self) -> int:
        """Completed years since the world began."""
        return self.day // self.days_per_year

    @property
    def day_of_year(self) -> int:
        """Day index within the current year, starting at 0."""
        return self.day % self.days_per_year

    def days_to_years(self, days: int) -> float:
        """Convert a duration in days to years."""
        return days / self.days_per_year

    def __str__(self) -> str:
        return f"Day {self.day} (Year {self.year}, day {self.day_of_year})"
