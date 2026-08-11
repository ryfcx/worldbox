"""The food layer of the world.

:class:`ResourceField` stores, for every tile, how much food is currently
available, the maximum that tile can hold, and how fast it regrows. It knows
nothing about agents -- callers take food from it and it simply depletes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..config import ResourceConfig
from .terrain import Grid, TerrainType

FloatGrid = List[List[float]]


@dataclass
class ResourceField:
    """Per-tile food amounts with terrain-driven capacity and regrowth."""

    width: int
    height: int
    food: FloatGrid
    capacity: FloatGrid
    regen: FloatGrid

    @classmethod
    def from_terrain(cls, grid: Grid, config: ResourceConfig) -> "ResourceField":
        """Build a field whose capacities/regen rates follow ``grid``'s terrain."""
        height = len(grid)
        width = len(grid[0]) if height else 0
        capacity: FloatGrid = []
        regen: FloatGrid = []
        food: FloatGrid = []

        for row in grid:
            capacity_row = [config.food_capacity.get(tile.value, 0.0) for tile in row]
            regen_row = [config.food_regen.get(tile.value, 0.0) for tile in row]
            capacity.append(capacity_row)
            regen.append(regen_row)
            food.append([value * config.initial_fill for value in capacity_row])

        return cls(width=width, height=height, food=food, capacity=capacity, regen=regen)

    # -- queries ------------------------------------------------------------

    def food_at(self, x: int, y: int) -> float:
        """Food currently available on a tile (0.0 for out-of-bounds tiles)."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return 0.0
        return self.food[y][x]

    def total_food(self) -> float:
        """Sum of all food currently in the world."""
        return sum(sum(row) for row in self.food)

    def total_capacity(self) -> float:
        """Sum of every tile's maximum food."""
        return sum(sum(row) for row in self.capacity)

    # -- mutation -----------------------------------------------------------

    def take(self, x: int, y: int, amount: float) -> float:
        """Remove up to ``amount`` food from a tile and return what was taken."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return 0.0
        available = self.food[y][x]
        taken = min(available, max(0.0, amount))
        self.food[y][x] = available - taken
        return taken

    def regrow(self) -> None:
        """Advance one day of regrowth across the whole map.

        Growth is proportional to how depleted a tile is, so recovery slows as a
        tile approaches its capacity -- empty tiles never fully stall because a
        small flat floor of growth is always applied.
        """
        for y in range(self.height):
            food_row = self.food[y]
            capacity_row = self.capacity[y]
            regen_row = self.regen[y]
            for x in range(self.width):
                cap = capacity_row[x]
                if cap <= 0.0:
                    continue
                current = food_row[x]
                if current >= cap:
                    continue
                rate = regen_row[x]
                growth = rate * (0.25 + 0.75 * (1.0 - current / cap))
                food_row[x] = min(cap, current + growth)


def is_edible(field: ResourceField, x: int, y: int, minimum: float) -> bool:
    """True if the tile holds at least ``minimum`` food."""
    return field.food_at(x, y) >= minimum


def terrain_food_summary(grid: Grid, field: ResourceField) -> dict[str, float]:
    """Total food available per terrain type -- used by world statistics."""
    totals: dict[str, float] = {terrain.value: 0.0 for terrain in TerrainType}
    for y, row in enumerate(grid):
        for x, tile in enumerate(row):
            totals[tile.value] += field.food[y][x]
    return totals
