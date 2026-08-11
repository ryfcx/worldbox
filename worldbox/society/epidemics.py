"""Disease: outbreaks, contagion, mortality and immunity.

Epidemics are a population-level system in the same sense as war: they emerge
from many agents being close together. Crowding raises the chance of an
outbreak, contagion spreads between neighbouring agents, and a tribe's medical
technology reduces both transmission and mortality.

Agents who survive an illness are immune to that disease afterwards, so a
plague burns through a population and then dies out for lack of fresh hosts --
only to become dangerous again a generation later, once enough non-immune
people have been born.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..agents.agent import Agent
from ..agents.needs import clamp
from ..config import DiseaseConfig
from .groups import GroupRegistry


@dataclass(frozen=True)
class Disease:
    """A kind of illness."""

    id: str
    name: str
    transmission: float  # Chance per day of infecting an adjacent agent.
    health_drain: float  # Health lost per day while ill.
    energy_drain: float  # Extra energy lost per day while ill.
    duration_days: int  # How long the illness lasts if survived.
    lethality: float  # Extra daily chance of death while ill.


# The roster of illnesses. Roughly ordered from nuisance to catastrophe.
DISEASES: Tuple[Disease, ...] = (
    Disease("fever", "Wasting Fever", 0.16, 4.0, 4.0, 12, 0.004),
    Disease("flux", "Bloody Flux", 0.22, 6.0, 6.0, 9, 0.008),
    Disease("pox", "Speckled Pox", 0.28, 5.5, 3.0, 16, 0.010),
    Disease("plague", "Black Plague", 0.34, 9.0, 8.0, 10, 0.022),
    Disease("coughing", "Winter Cough", 0.30, 2.5, 5.0, 8, 0.002),
)

DISEASE_BY_ID: Dict[str, Disease] = {disease.id: disease for disease in DISEASES}


@dataclass
class Infection:
    """One agent's current illness."""

    disease_id: str
    days_left: int
    caught_on_day: int

    @property
    def disease(self) -> Disease:
        """The illness this infection is an instance of."""
        return DISEASE_BY_ID[self.disease_id]


@dataclass
class Outbreak:
    """A single epidemic: one disease, from first case to last."""

    id: int
    disease_id: str
    started_day: int
    origin: Tuple[int, int]
    ended_day: Optional[int] = None
    infections: int = 0
    deaths: int = 0
    peak_infected: int = 0

    @property
    def disease(self) -> Disease:
        """The illness driving this outbreak."""
        return DISEASE_BY_ID[self.disease_id]

    @property
    def active(self) -> bool:
        """True while at least one case remains."""
        return self.ended_day is None


@dataclass
class EpidemicReport:
    """What disease did on a single day, for the engine to turn into events."""

    started: List[Outbreak] = field(default_factory=list)
    ended: List[Outbreak] = field(default_factory=list)
    deaths: List[Tuple[Agent, Disease]] = field(default_factory=list)
    new_infections: int = 0

    @property
    def death_count(self) -> int:
        """How many agents the illness killed today."""
        return len(self.deaths)


class DiseaseSystem:
    """Owns every outbreak and advances contagion one day at a time."""

    def __init__(self, config: DiseaseConfig) -> None:
        self.config = config
        self.outbreaks: Dict[int, Outbreak] = {}
        self.history: List[Outbreak] = []
        self._next_id = 1

    # -- reporting ----------------------------------------------------------

    def active_outbreaks(self) -> List[Outbreak]:
        """Every epidemic currently running, oldest first."""
        return sorted(
            (outbreak for outbreak in self.outbreaks.values() if outbreak.active),
            key=lambda outbreak: outbreak.started_day,
        )

    def infected_count(self, agents: Sequence[Agent]) -> int:
        """How many agents are currently ill."""
        return sum(1 for agent in agents if agent.infection is not None)

    # -- resistance ---------------------------------------------------------

    def _resistance(self, agent: Agent, registry: GroupRegistry) -> float:
        """How much a tribe's medicine protects one of its members (0..max)."""
        group = registry.get(agent.group_id)
        if group is None:
            return 0.0
        # Technology and the tribe's healers both help.
        return min(
            self.config.max_tech_resistance,
            group.knowledge.effects.disease_resistance + group.healer_resistance,
        )

    # -- the daily update ---------------------------------------------------

    def update(
        self,
        agents: List[Agent],
        occupancy,
        world,
        registry: GroupRegistry,
        day: int,
        rng: random.Random,
    ) -> EpidemicReport:
        """Advance every infection, spread contagion, and maybe start an outbreak."""
        report = EpidemicReport()
        if not self.config.enabled:
            return report

        self._progress_infections(agents, registry, day, rng, report)
        self._spread(agents, occupancy, world, registry, day, rng, report)
        self._maybe_start_outbreak(agents, day, rng, report)
        self._close_finished_outbreaks(agents, day, report)
        return report

    def _progress_infections(
        self,
        agents: List[Agent],
        registry: GroupRegistry,
        day: int,
        rng: random.Random,
        report: EpidemicReport,
    ) -> None:
        """Apply the daily toll of illness, then recover or kill."""
        for agent in agents:
            infection = agent.infection
            if infection is None or not agent.alive:
                continue
            disease = infection.disease
            resistance = self._resistance(agent, registry)
            severity = 1.0 - resistance

            agent.needs.health = clamp(agent.needs.health - disease.health_drain * severity)
            agent.needs.energy = clamp(agent.needs.energy - disease.energy_drain * severity)

            outbreak = self.outbreaks.get(agent.outbreak_id or -1)

            # An illness can kill outright, on top of draining health.
            if rng.random() < disease.lethality * severity or agent.needs.health <= 0.0:
                agent.needs.health = 0.0
                report.deaths.append((agent, disease))
                if outbreak is not None:
                    outbreak.deaths += 1
                group = registry.get(agent.group_id)
                if group is not None:
                    group.plague_dead += 1
                continue

            infection.days_left -= 1
            if infection.days_left <= 0:
                agent.recover_from_illness(day, self.config.immunity_days)

    def _spread(
        self,
        agents: List[Agent],
        occupancy,
        world,
        registry: GroupRegistry,
        day: int,
        rng: random.Random,
        report: EpidemicReport,
    ) -> None:
        """Pass infections to susceptible neighbours."""
        radius = self.config.infection_radius
        carriers = [
            agent for agent in agents if agent.alive and agent.infection is not None
        ]
        for carrier in carriers:
            infection = carrier.infection
            if infection is None:
                continue
            disease = infection.disease
            for x, y in world.tiles_within(carrier.x, carrier.y, radius):
                for other in occupancy.get((x, y), ()):
                    if other.id == carrier.id or not other.alive:
                        continue
                    if not other.is_susceptible_to(disease.id, day):
                        continue
                    chance = disease.transmission * (1.0 - self._resistance(other, registry))
                    if rng.random() < chance:
                        other.infect(disease.id, disease.duration_days, day, carrier.outbreak_id)
                        report.new_infections += 1
                        outbreak = self.outbreaks.get(carrier.outbreak_id or -1)
                        if outbreak is not None:
                            outbreak.infections += 1

    def _maybe_start_outbreak(
        self,
        agents: List[Agent],
        day: int,
        rng: random.Random,
        report: EpidemicReport,
    ) -> None:
        """Roll for a fresh epidemic. Crowding makes one likelier."""
        population = len(agents)
        if population < self.config.min_population_for_outbreak:
            return
        if len(self.active_outbreaks()) >= self.config.max_active_outbreaks:
            return

        crowding = population / max(1, self.config.crowding_reference_population)
        if rng.random() >= self.config.outbreak_base_chance * crowding:
            return

        disease = rng.choice(DISEASES)
        candidates = [agent for agent in agents if agent.is_susceptible_to(disease.id, day)]
        if not candidates:
            return

        patient_zero = rng.choice(candidates)
        outbreak = Outbreak(
            id=self._next_id,
            disease_id=disease.id,
            started_day=day,
            origin=(patient_zero.x, patient_zero.y),
            infections=1,
        )
        self.outbreaks[outbreak.id] = outbreak
        self._next_id += 1

        patient_zero.infect(disease.id, disease.duration_days, day, outbreak.id)
        report.started.append(outbreak)

    def _close_finished_outbreaks(
        self, agents: List[Agent], day: int, report: EpidemicReport
    ) -> None:
        """Mark outbreaks over once nobody is carrying them any more."""
        live_counts: Dict[int, int] = {}
        for agent in agents:
            if agent.alive and agent.infection is not None and agent.outbreak_id is not None:
                live_counts[agent.outbreak_id] = live_counts.get(agent.outbreak_id, 0) + 1

        for outbreak in list(self.outbreaks.values()):
            if not outbreak.active:
                continue
            current = live_counts.get(outbreak.id, 0)
            outbreak.peak_infected = max(outbreak.peak_infected, current)
            if current == 0:
                outbreak.ended_day = day
                self.history.append(outbreak)
                del self.outbreaks[outbreak.id]
                report.ended.append(outbreak)
