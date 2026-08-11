"""Central configuration for Worldbox.

Every tunable number in the simulation lives here. This module deliberately
imports nothing from the rest of the package so that it can never participate
in a circular import -- terrain-specific values are therefore keyed by terrain
*name* (``"grass"``, ``"forest"``, ...) rather than by the ``TerrainType`` enum.

All config objects are frozen dataclasses. To experiment, either edit the
defaults below or build a modified copy at runtime::

    import dataclasses
    cfg = dataclasses.replace(Config(), simulation=dataclasses.replace(
        Config().simulation, seed=99))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

# ---------------------------------------------------------------------------
# World generation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorldConfig:
    """Shape of the map and the thresholds used to classify each tile."""

    width: int = 90
    height: int = 40

    # Fractal value-noise settings. More octaves = more small-scale detail.
    noise_octaves: int = 4
    noise_base_frequency: int = 4

    # Elevation thresholds (noise is normalised to 0.0 - 1.0).
    water_level: float = 0.42
    mountain_level: float = 0.72

    # Anything above this moisture (and on land, below the mountain line)
    # becomes forest instead of grass.
    forest_moisture: float = 0.55

    # Generation fails loudly if fewer than this fraction of tiles is walkable.
    min_passable_fraction: float = 0.25


# ---------------------------------------------------------------------------
# Food / resources
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResourceConfig:
    """How much food each terrain holds and how quickly it grows back."""

    # Maximum food units a single tile of this terrain can hold.
    food_capacity: Dict[str, float] = field(
        default_factory=lambda: {
            "grass": 5.0,
            "forest": 10.0,
            "mountain": 0.0,
            "water": 0.0,
        }
    )

    # Food units regenerated per tile per day (logistic-ish, capped at capacity).
    food_regen: Dict[str, float] = field(
        default_factory=lambda: {
            "grass": 0.08,
            "forest": 0.18,
            "mountain": 0.0,
            "water": 0.0,
        }
    )

    # Fraction of capacity each tile starts the simulation with.
    initial_fill: float = 0.8

    # An agent will not bother eating a tile holding less than this.
    min_food_to_eat: float = 0.5

    # Maximum food units consumed in a single meal (one day).
    meal_size: float = 6.0


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentConfig:
    """Metabolism, perception, lifespan and reproduction rules."""

    initial_population: int = 100
    max_population: int = 1500  # Hard cap; protects runtime performance.

    # --- Needs -------------------------------------------------------------
    # Hunger runs 0 (full) -> 100 (starving).
    hunger_per_day: float = 3.5
    hunger_per_move: float = 0.5
    hunger_hungry_threshold: float = 45.0
    hunger_starving_threshold: float = 85.0
    hunger_restored_per_food: float = 8.0

    # Energy runs 100 (rested) -> 0 (exhausted).
    energy_per_day: float = 6.0
    energy_per_move: float = 3.0
    energy_tired_threshold: float = 30.0
    energy_restored_per_rest: float = 45.0

    # Health runs 100 -> 0. Reaching 0 kills the agent.
    health_loss_starving: float = 3.0
    health_loss_exhausted: float = 1.5
    health_regen: float = 0.8
    health_regen_max_hunger: float = 35.0
    health_regen_min_energy: float = 60.0

    # --- Perception & memory ----------------------------------------------
    vision_radius: int = 5
    memory_capacity: int = 12  # Remembered food locations per agent.
    memory_decay_days: int = 90  # Forget a location after this many days.
    personal_log_size: int = 8  # Recent personal events kept per agent.

    # --- Lifespan ----------------------------------------------------------
    lifespan_mean_years: float = 65.0
    lifespan_stddev_years: float = 9.0
    lifespan_min_years: float = 25.0

    # Ages of the founding generation (uniform, in years).
    starting_age_min_years: float = 5.0
    starting_age_max_years: float = 45.0

    # --- Reproduction ------------------------------------------------------
    adult_age_years: float = 16.0
    max_reproduction_age_years: float = 45.0
    reproduction_cooldown_days: int = 730
    reproduction_max_hunger: float = 35.0
    reproduction_min_energy: float = 50.0
    reproduction_min_health: float = 65.0

    # Newborn starting state.
    newborn_hunger: float = 20.0
    newborn_energy: float = 80.0

    # --- Temperament -------------------------------------------------------
    # Aggression (0.0 peaceful .. 1.0 warlike) drives fighting and fleeing. It
    # is inherited from the parents' average, plus a little drift, so tribes
    # slowly develop their own temperament.
    aggression_mean: float = 0.35
    aggression_stddev: float = 0.18
    aggression_inheritance_drift: float = 0.08


# ---------------------------------------------------------------------------
# Society: tribes, technology, diplomacy, war and disease
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupConfig:
    """How tribes form, grow, hold territory and split apart."""

    enabled: bool = True

    # Two unaffiliated adults who meet can found a tribe together. This is
    # deliberately rare: if founding is easy, everyone starts their own band of
    # three and no tribe ever grows large enough to build anything.
    founding_chance: float = 0.04
    # An unaffiliated agent adjacent to a tribe member may be recruited.
    recruit_chance: float = 0.55

    # A tribe this small is not viable; after a grace period it disbands and
    # its people drift into their neighbours. This is what consolidates a
    # scatter of tiny bands into a few real societies.
    min_viable_size: int = 4
    disband_grace_days: int = 400

    # Once a tribe passes this size it splinters, and the members furthest
    # from its centre leave to found a new one.
    split_size: int = 55
    split_fraction: float = 0.4
    min_split_size: int = 8

    # Chance an idle agent drifts back toward its tribe's centre rather than
    # wandering freely. This is what makes tribes hold territory at all.
    cohesion_chance: float = 0.55
    # Beyond this distance from the centre, cohesion always wins.
    territory_radius: int = 14


@dataclass(frozen=True)
class TechnologyConfig:
    """How quickly tribes accumulate research and invent new technologies."""

    enabled: bool = True

    # Research points per day = base * population^exponent * modifiers.
    base_research_rate: float = 0.04
    population_exponent: float = 0.75
    min_population_to_research: int = 4

    # Well-fed tribes think; starving tribes do not. Scales research by
    # between (1 - penalty) and 1.0 depending on average hunger.
    hunger_research_penalty: float = 0.75


@dataclass(frozen=True)
class SettlementConfig:
    """Permanent settlements and their food stores."""

    enabled: bool = True

    # A tribe founds its first settlement once it reaches this size.
    min_population_to_found: int = 8
    # A settlement will not grow to the next level until its granary is at
    # least this full -- civilisations are built on surplus.
    upgrade_store_fraction: float = 0.35

    # Farmers gather from tiles within this radius of the settlement.
    harvest_radius: int = 6
    harvest_per_farmer: float = 2.2
    harvest_per_tile: float = 1.5

    # Fraction of the granary lost each day; storage technology reduces it.
    spoilage_rate: float = 0.012

    # A hungry member near the settlement may draw this much from the store.
    ration_size: float = 5.0
    # How far from home a member can still claim a ration.
    ration_radius: int = 8


@dataclass(frozen=True)
class RoleConfig:
    """Division of labour within a tribe."""

    enabled: bool = True

    # Fractions of the working-age population assigned to each profession.
    warrior_share_at_peace: float = 0.10
    warrior_share_at_war: float = 0.32
    farmer_share: float = 0.32
    farmer_share_when_starving: float = 0.48
    scholar_share: float = 0.14
    healer_share: float = 0.08

    # Age past which an agent becomes an elder rather than a worker.
    elder_age_years: float = 55.0

    # What each profession contributes.
    scholar_research_bonus: float = 0.09  # Per scholar.
    elder_research_bonus: float = 0.02  # Per elder; accumulated wisdom.
    warrior_combat_bonus: float = 0.40
    noncombatant_penalty: float = 0.55
    healer_resistance: float = 1.6  # Scaled by healers per capita.
    max_healer_resistance: float = 0.35
    forager_vision_bonus: int = 2

    # A tribe counts as starving above this average hunger.
    starving_hunger: float = 55.0


@dataclass(frozen=True)
class CultureConfig:
    """How knowledge spreads between tribes, and who leads them."""

    enabled: bool = True

    # Neighbouring tribes on good terms copy each other's technology. This is
    # what lets civilisation advance across a region rather than in one tribe.
    diffusion_radius: int = 25
    diffusion_min_relation: float = -20.0
    diffusion_chance: float = 0.02

    # Friendly tribes send food to a neighbour whose granary has run dry.
    trade_min_relation: float = 20.0
    trade_fraction: float = 0.10

    # A chieftain's temperament pulls the tribe's own aggression toward theirs.
    chieftain_influence: float = 0.15


@dataclass(frozen=True)
class DiplomacyConfig:
    """Relations between tribes, and the thresholds for war and peace."""

    enabled: bool = True

    # Relations run -100 (blood feud) to +100 (close allies), starting at 0.
    starting_relation: float = 0.0

    # Tribes whose centres are closer than this compete for territory.
    territory_pressure_radius: int = 12
    pressure_per_day: float = 0.55
    # Competition bites harder when the world's food is running out.
    scarcity_multiplier: float = 2.0
    # Fraction of a grudge that fades each day. Relations settle where daily
    # pressure balances this decay, instead of sliding to permanent war.
    neutral_drift: float = 0.02

    # Each individual skirmish sours relations by this much.
    skirmish_relation_cost: float = 3.0

    war_threshold: float = -55.0
    peace_threshold: float = -20.0
    # Below this, tribes treat each other as enemies on sight. Kept well below
    # peace_threshold so ordinary cool relations do not mean daily bloodshed.
    hostility_threshold: float = -42.0
    min_war_days: int = 40
    # A war ends if either side loses this fraction of its members.
    war_exhaustion_fraction: float = 0.25
    # Relations after a peace treaty: wary, not friendly.
    post_war_relation: float = -15.0


@dataclass(frozen=True)
class CombatConfig:
    """Individual fights between members of rival tribes."""

    enabled: bool = True

    # Chance two adjacent enemies actually come to blows.
    war_engagement_chance: float = 0.35
    # Chance of an unprovoked skirmish while not at war, scaled by aggression.
    skirmish_chance: float = 0.02

    # Damage dealt to the loser and to the winner of a fight.
    loser_damage: float = 22.0
    winner_damage: float = 8.0
    # How much the strength ratio amplifies damage.
    damage_ratio_cap: float = 2.0

    # Below this health an agent would rather run than fight.
    flee_health: float = 40.0
    # Weapon technologies multiply strength by up to this much in total.
    max_tech_combat_bonus: float = 2.0


@dataclass(frozen=True)
class DiseaseConfig:
    """Outbreaks, contagion and immunity."""

    enabled: bool = True

    # Daily chance of a new outbreak somewhere in the world, scaled by
    # population (crowding breeds plague).
    outbreak_base_chance: float = 0.006
    crowding_reference_population: int = 150
    min_population_for_outbreak: int = 25
    # Never run more than this many simultaneous epidemics.
    max_active_outbreaks: int = 3

    # Contagion spreads to agents within this many tiles.
    infection_radius: int = 1
    # Recovered agents are immune to that disease for this long (0 = forever).
    immunity_days: int = 0

    # Medicine and related technologies reduce transmission and mortality by
    # up to this fraction.
    max_tech_resistance: float = 0.6


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulationConfig:
    """Clock, determinism and event-recording settings."""

    seed: int = 1337
    days_per_year: int = 365

    # Size of the in-memory event ring buffer.
    event_log_size: int = 500

    # Routine "found food" events would flood the log, so only this fraction
    # of successful meals is recorded.
    food_event_chance: float = 0.04

    # Seconds between ticks while the simulation is running unattended.
    autorun_tick_seconds: float = 0.12

    # Selectable tick delays for the live view, slowest to fastest. The last
    # entry (0.0) runs as fast as the machine allows.
    speed_steps: tuple[float, ...] = (1.0, 0.5, 0.25, 0.12, 0.06, 0.03, 0.01, 0.0)
    default_speed_index: int = 3

    # How often the live view repaints, in seconds.
    live_refresh_seconds: float = 0.1

    # The background runner always sleeps at least this long between batches.
    # Without it, an uncapped loop reacquires the lock before the display
    # thread can ever get in, and the live view freezes.
    min_tick_sleep: float = 0.002

    # Days simulated per lock acquisition when running at maximum speed.
    # Larger = faster, but the display updates less smoothly.
    max_speed_batch_days: int = 20

    # A safety valve for `advance` -- refuse absurd single commands.
    max_days_per_command: int = 1_000_000


# ---------------------------------------------------------------------------
# Top-level container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """The complete configuration passed to :class:`SimulationEngine`."""

    world: WorldConfig = field(default_factory=WorldConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    groups: GroupConfig = field(default_factory=GroupConfig)
    settlements: SettlementConfig = field(default_factory=SettlementConfig)
    roles: RoleConfig = field(default_factory=RoleConfig)
    culture: CultureConfig = field(default_factory=CultureConfig)
    technology: TechnologyConfig = field(default_factory=TechnologyConfig)
    diplomacy: DiplomacyConfig = field(default_factory=DiplomacyConfig)
    combat: CombatConfig = field(default_factory=CombatConfig)
    disease: DiseaseConfig = field(default_factory=DiseaseConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)

    def validate(self) -> None:
        """Raise :class:`ValueError` if the configuration is self-inconsistent.

        Called once by the engine at construction time so that bad numbers fail
        immediately rather than producing a silently broken world.
        """
        if self.world.width < 8 or self.world.height < 8:
            raise ValueError("World must be at least 8x8 tiles.")
        if not 0.0 < self.world.water_level < self.world.mountain_level < 1.0:
            raise ValueError("Require 0 < water_level < mountain_level < 1.")
        if self.agents.initial_population < 1:
            raise ValueError("initial_population must be at least 1.")
        if self.agents.initial_population > self.agents.max_population:
            raise ValueError("initial_population exceeds max_population.")
        if self.agents.vision_radius < 1:
            raise ValueError("vision_radius must be at least 1.")
        if self.simulation.days_per_year < 1:
            raise ValueError("days_per_year must be at least 1.")
        if self.agents.lifespan_min_years <= self.agents.adult_age_years:
            raise ValueError("lifespan_min_years must exceed adult_age_years.")
        if not self.simulation.speed_steps:
            raise ValueError("speed_steps must contain at least one delay.")
        if not 0 <= self.simulation.default_speed_index < len(self.simulation.speed_steps):
            raise ValueError("default_speed_index is outside speed_steps.")
        if self.simulation.live_refresh_seconds <= 0:
            raise ValueError("live_refresh_seconds must be positive.")
        if self.groups.split_size <= self.groups.min_split_size:
            raise ValueError("split_size must exceed min_split_size.")
        if not 0.0 < self.groups.split_fraction < 1.0:
            raise ValueError("split_fraction must be between 0 and 1.")
        if self.diplomacy.war_threshold >= self.diplomacy.peace_threshold:
            raise ValueError("war_threshold must be below peace_threshold.")
        if self.combat.damage_ratio_cap < 1.0:
            raise ValueError("damage_ratio_cap must be at least 1.0.")
