"""Terrain types and deterministic procedural terrain generation.

The generator uses fractal *value noise*: a small lattice of random values is
sampled with smoothed bilinear interpolation, then several such lattices at
doubling frequencies are summed. This needs nothing beyond the standard library
and is fully reproducible from a seeded ``random.Random``.

Two independent noise fields are produced -- elevation and moisture -- and each
tile is classified from the pair:

    elevation < water_level      -> WATER      (impassable)
    elevation > mountain_level   -> MOUNTAIN   (impassable)
    moisture  > forest_moisture  -> FOREST
    otherwise                    -> GRASS
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import List

from ..config import WorldConfig

Grid = List[List["TerrainType"]]
NoiseField = List[List[float]]


class WorldGenerationError(RuntimeError):
    """Raised when the configured thresholds produce an unusable world."""


@dataclass(frozen=True)
class TerrainProperties:
    """Static, physical facts about a terrain type.

    Food values live in :class:`~worldbox.config.ResourceConfig` instead, since
    those are balance knobs rather than physical properties.
    """

    label: str
    glyph: str
    passable: bool
    discovery_name: str  # Reads naturally in "discovered a {discovery_name}".


class TerrainType(Enum):
    """The four terrain types of the world."""

    WATER = "water"
    GRASS = "grass"
    FOREST = "forest"
    MOUNTAIN = "mountain"

    @property
    def properties(self) -> TerrainProperties:
        return TERRAIN_PROPERTIES[self]

    @property
    def passable(self) -> bool:
        """True if an agent may stand on this terrain."""
        return self.properties.passable

    @property
    def label(self) -> str:
        """Human-readable name, e.g. ``"forest"``."""
        return self.properties.label

    @property
    def glyph(self) -> str:
        """Single character used for any future map rendering."""
        return self.properties.glyph

    @property
    def discovery_name(self) -> str:
        """Phrase used in discovery events, e.g. ``"grassland"``."""
        return self.properties.discovery_name


TERRAIN_PROPERTIES = {
    TerrainType.WATER: TerrainProperties("water", "~", False, "lake"),
    TerrainType.GRASS: TerrainProperties("grass", ".", True, "grassland"),
    TerrainType.FOREST: TerrainProperties("forest", "T", True, "forest"),
    TerrainType.MOUNTAIN: TerrainProperties("mountain", "^", False, "mountain"),
}


# ---------------------------------------------------------------------------
# Value noise
# ---------------------------------------------------------------------------


def _smoothstep(t: float) -> float:
    """Ease a 0..1 interpolation factor so lattice cells blend smoothly."""
    return t * t * (3.0 - 2.0 * t)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _sample_lattice(lattice: NoiseField, u: float, v: float) -> float:
    """Bilinearly sample ``lattice`` at normalised coordinates ``u, v`` in 0..1."""
    rows = len(lattice)
    cols = len(lattice[0])
    x = u * (cols - 1)
    y = v * (rows - 1)
    x0, y0 = int(x), int(y)
    x1, y1 = min(x0 + 1, cols - 1), min(y0 + 1, rows - 1)
    tx, ty = _smoothstep(x - x0), _smoothstep(y - y0)
    top = _lerp(lattice[y0][x0], lattice[y0][x1], tx)
    bottom = _lerp(lattice[y1][x0], lattice[y1][x1], tx)
    return _lerp(top, bottom, ty)


def generate_noise(
    rng: random.Random,
    width: int,
    height: int,
    octaves: int,
    base_frequency: int,
) -> NoiseField:
    """Build a ``height`` x ``width`` fractal noise field normalised to 0..1."""
    field: NoiseField = [[0.0] * width for _ in range(height)]
    amplitude = 1.0
    total_amplitude = 0.0
    frequency = max(1, base_frequency)

    for _ in range(max(1, octaves)):
        lattice = [
            [rng.random() for _ in range(frequency + 1)] for _ in range(frequency + 1)
        ]
        for y in range(height):
            v = y / max(1, height - 1)
            row = field[y]
            for x in range(width):
                u = x / max(1, width - 1)
                row[x] += amplitude * _sample_lattice(lattice, u, v)
        total_amplitude += amplitude
        amplitude *= 0.5
        frequency *= 2

    # Normalise to 0..1 across the whole field.
    lowest = min(min(row) for row in field)
    highest = max(max(row) for row in field)
    span = highest - lowest or 1.0
    for y in range(height):
        row = field[y]
        for x in range(width):
            row[x] = (row[x] - lowest) / span
    return field


# ---------------------------------------------------------------------------
# Terrain classification
# ---------------------------------------------------------------------------


def generate_terrain(config: WorldConfig, rng: random.Random) -> Grid:
    """Generate the terrain grid for ``config`` using ``rng``.

    The same seeded ``rng`` always yields the same map for the same config.

    Raises:
        WorldGenerationError: if too little of the map is walkable to be usable.
    """
    elevation = generate_noise(
        rng, config.width, config.height, config.noise_octaves, config.noise_base_frequency
    )
    moisture = generate_noise(
        rng, config.width, config.height, config.noise_octaves, config.noise_base_frequency
    )

    grid: Grid = []
    passable_tiles = 0
    for y in range(config.height):
        row: List[TerrainType] = []
        for x in range(config.width):
            height_value = elevation[y][x]
            if height_value < config.water_level:
                terrain = TerrainType.WATER
            elif height_value > config.mountain_level:
                terrain = TerrainType.MOUNTAIN
            elif moisture[y][x] > config.forest_moisture:
                terrain = TerrainType.FOREST
            else:
                terrain = TerrainType.GRASS
            if terrain.passable:
                passable_tiles += 1
            row.append(terrain)
        grid.append(row)

    total_tiles = config.width * config.height
    if passable_tiles < total_tiles * config.min_passable_fraction:
        raise WorldGenerationError(
            f"Only {passable_tiles}/{total_tiles} tiles are passable; "
            "lower water_level or raise mountain_level in WorldConfig."
        )
    return grid
