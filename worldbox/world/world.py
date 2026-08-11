"""The world: terrain grid + resource field + spatial queries.

:class:`World` is the environment agents live in. It owns no agents itself --
the simulation engine keeps those -- which keeps the environment reusable and
easy to test in isolation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

from ..config import Config
from .resources import ResourceField
from .terrain import Grid, TerrainType, WorldGenerationError, generate_terrain

Coord = Tuple[int, int]

# 8-way neighbourhood offsets, in a fixed order so movement stays deterministic.
_NEIGHBOUR_OFFSETS: Tuple[Coord, ...] = (
    (-1, -1), (0, -1), (1, -1),
    (-1, 0), (1, 0),
    (-1, 1), (0, 1), (1, 1),
)


@dataclass
class World:
    """A 2D grid world with terrain and a food layer."""

    width: int
    height: int
    seed: int
    grid: Grid
    resources: ResourceField
    _passable_tiles: List[Coord] = field(default_factory=list, repr=False)

    # -- construction -------------------------------------------------------

    @classmethod
    def generate(cls, config: Config, rng: random.Random, seed: int) -> "World":
        """Procedurally generate a world from ``config`` and a seeded ``rng``.

        Raises:
            WorldGenerationError: if the resulting map is largely unwalkable.
        """
        grid = generate_terrain(config.world, rng)
        resources = ResourceField.from_terrain(grid, config.resources)
        world = cls(
            width=config.world.width,
            height=config.world.height,
            seed=seed,
            grid=grid,
            resources=resources,
        )
        world._passable_tiles = [
            (x, y)
            for y in range(world.height)
            for x in range(world.width)
            if grid[y][x].passable
        ]
        if not world._passable_tiles:
            raise WorldGenerationError("Generated world has no passable tiles.")
        return world

    # -- spatial queries ----------------------------------------------------

    def in_bounds(self, x: int, y: int) -> bool:
        """True if the coordinate lies inside the map."""
        return 0 <= x < self.width and 0 <= y < self.height

    def terrain_at(self, x: int, y: int) -> Optional[TerrainType]:
        """Terrain of a tile, or ``None`` when out of bounds."""
        if not self.in_bounds(x, y):
            return None
        return self.grid[y][x]

    def is_passable(self, x: int, y: int) -> bool:
        """True if an agent may occupy this tile."""
        terrain = self.terrain_at(x, y)
        return terrain is not None and terrain.passable

    def neighbours(self, x: int, y: int, passable_only: bool = True) -> List[Coord]:
        """The up-to-8 tiles adjacent to ``(x, y)``, in a deterministic order."""
        result: List[Coord] = []
        for dx, dy in _NEIGHBOUR_OFFSETS:
            nx, ny = x + dx, y + dy
            if not self.in_bounds(nx, ny):
                continue
            if passable_only and not self.grid[ny][nx].passable:
                continue
            result.append((nx, ny))
        return result

    def tiles_within(self, x: int, y: int, radius: int) -> Iterator[Coord]:
        """Yield every in-bounds tile within a square ``radius`` of ``(x, y)``."""
        for dy in range(-radius, radius + 1):
            ny = y + dy
            if not 0 <= ny < self.height:
                continue
            for dx in range(-radius, radius + 1):
                nx = x + dx
                if 0 <= nx < self.width:
                    yield nx, ny

    def random_passable_tile(self, rng: random.Random) -> Coord:
        """A uniformly random walkable tile."""
        return rng.choice(self._passable_tiles)

    @property
    def passable_tile_count(self) -> int:
        """How many tiles agents can stand on."""
        return len(self._passable_tiles)

    # -- statistics ---------------------------------------------------------

    def terrain_counts(self) -> Dict[str, int]:
        """Tile count per terrain type, keyed by terrain name."""
        counts: Dict[str, int] = {terrain.value: 0 for terrain in TerrainType}
        for row in self.grid:
            for tile in row:
                counts[tile.value] += 1
        return counts

    # -- daily update -------------------------------------------------------

    def update(self, day: int) -> None:
        """Advance the environment by one day (phase 1 of the simulation tick)."""
        self.resources.regrow()
