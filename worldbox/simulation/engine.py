"""The simulation engine: owns the world, the agents, the clock and history.

This module is completely independent of any user interface. A frontend drives
it with :meth:`SimulationEngine.step` / :meth:`run` and reads back a
:class:`WorldStats` snapshot plus events from the log -- nothing more. That
contract is what will let a graphical frontend be added later without touching
the simulation.

A single day (one tick) runs these phases, in this order:

    1. Update the environment (food regrowth) and refresh tribe membership
    2. Update each agent's needs
    3. Let each agent decide what to do
    4. Execute their actions
    5. Process food / resources (resolve meals)
    6. Resolve combat between rival tribes
    7. Advance disease: illness, contagion, new outbreaks
    8. Process deaths, then births
    9. Update society: tribes form and split, research, diplomacy and war
   10. Record important events (daily milestones)
   11. Advance the simulation clock

Combat is resolved before deaths so that mortal wounds are attributed to the
right war, and society is updated after births so that newborns are counted.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from ..agents.agent import Agent, create_agent
from ..agents.behavior import (
    Goal,
    Occupancy,
    SocialContext,
    can_reproduce,
    decide,
    execute,
)
from ..agents.needs import Needs, apply_daily_upkeep, apply_meal, cause_of_death
from ..config import Config
from ..society import combat as combat_system
from ..society import culture as culture_system
from ..society import roles as role_system
from ..society.diplomacy import Diplomacy, War, pair_key
from ..society.epidemics import DiseaseSystem, EpidemicReport
from ..society.groups import Group, GroupRegistry
from ..society.roles import Role
from ..society.settlements import Settlement, SettlementSystem
from ..society.technology import ERAS, TechEffects, Technology, try_invent
from ..world.terrain import TerrainType
from ..world.world import World
from .chronicle import Chronicle, Milestone
from .clock import Clock
from .events import Event, EventKind, EventLog

Coord = Tuple[int, int]

# The modifiers an agent with no tribe gets: none at all.
NO_TECH_EFFECTS = TechEffects()


@dataclass(frozen=True)
class WorldStats:
    """An immutable snapshot of the simulation, for display or logging.

    Frontends should depend on this dataclass rather than on the live world or
    agent objects.
    """

    day: int
    year: int
    seed: int
    population: int
    births_today: int
    deaths_today: int
    total_births: int
    total_deaths: int
    average_age_years: float
    average_health: float
    average_hunger: float
    average_energy: float
    hungry: int
    resting: int
    seeking_food: int
    eating: int
    wandering: int
    seeking_mate: int
    fighting: int
    fleeing: int
    oldest_age_years: float
    total_food: float
    food_capacity: float
    terrain_counts: Dict[str, int]

    # -- society ------------------------------------------------------------
    tribes: int
    unaffiliated: int
    largest_tribe: Optional[str]
    largest_tribe_size: int
    most_advanced_era: str
    technologies_known: int  # Distinct technologies known anywhere.
    average_aggression: float

    # -- conflict -----------------------------------------------------------
    active_wars: int
    battles_today: int
    total_battles: int
    war_deaths: int

    # -- disease ------------------------------------------------------------
    active_outbreaks: int
    ill: int
    plague_deaths: int
    current_plague: Optional[str]

    # -- civilisation -------------------------------------------------------
    settlements: int
    largest_settlement: Optional[str]
    largest_settlement_level: Optional[str]
    total_food_stored: float
    role_counts: Dict[str, int]
    chronicle_entries: int


@dataclass
class DayReport:
    """What happened on a single simulated day."""

    day: int
    births: int = 0
    deaths: int = 0
    battles: int = 0
    war_deaths: int = 0
    plague_deaths: int = 0
    inventions: List[Tuple[str, Technology]] = field(default_factory=list)
    wars_declared: List[War] = field(default_factory=list)
    wars_ended: List[War] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)


class SimulationEngine:
    """Runs the Worldbox simulation, one day at a time."""

    def __init__(self, config: Optional[Config] = None, seed: Optional[int] = None) -> None:
        self.config = config or Config()
        self.config.validate()
        self.events = EventLog(self.config.simulation.event_log_size)
        self.seed = self.config.simulation.seed if seed is None else int(seed)

        # Populated by reset(); declared here so the attributes always exist.
        self.rng: random.Random = random.Random(self.seed)
        self.world: World
        self.clock: Clock
        self.agents: List[Agent] = []
        self.groups: GroupRegistry
        self.settlements: SettlementSystem
        self.diplomacy: Diplomacy
        self.diseases: DiseaseSystem
        self.chronicle = Chronicle()
        self.total_births = 0
        self.total_deaths = 0
        self.total_battles = 0
        self.total_war_deaths = 0
        self.total_plague_deaths = 0
        self.last_report = DayReport(day=0)

        # Events recorded during the current day, collected via a subscriber so
        # the engine never has to guess what the log kept or evicted.
        self._day_events: List[Event] = []
        self.events.subscribe(self._day_events.append)

        self.reset(self.seed)

    # -- lifecycle ----------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> None:
        """Rebuild the world and population from scratch.

        Passing the same seed always reproduces exactly the same simulation.
        """
        if seed is not None:
            self.seed = int(seed)

        self.rng = random.Random(self.seed)
        self.clock = Clock(day=0, days_per_year=self.config.simulation.days_per_year)
        self.events.clear()
        self.world = World.generate(self.config, self.rng, self.seed)

        # Society starts empty: nobody belongs to a tribe on day zero, and every
        # tribe in the world's history forms through agents meeting each other.
        self.groups = GroupRegistry(self.config.groups)
        self.settlements = SettlementSystem(self.config.settlements)
        self.diplomacy = Diplomacy(self.config.diplomacy)
        self.diseases = DiseaseSystem(self.config.disease)
        self.chronicle.clear()

        self._next_agent_id = 1
        self.total_births = 0
        self.total_deaths = 0
        self.total_battles = 0
        self.total_war_deaths = 0
        self.total_plague_deaths = 0
        self.agents = [self._spawn_founder() for _ in range(self.config.agents.initial_population)]
        self.last_report = DayReport(day=0)

        self.events.record(
            0,
            EventKind.SYSTEM,
            f"World created with seed {self.seed} "
            f"({self.world.width}x{self.world.height}, {len(self.agents)} agents)",
        )

    def _spawn_founder(self) -> Agent:
        """Create one member of the founding generation."""
        agent_config = self.config.agents
        days_per_year = self.config.simulation.days_per_year
        x, y = self.world.random_passable_tile(self.rng)
        age_days = int(
            self.rng.uniform(
                agent_config.starting_age_min_years, agent_config.starting_age_max_years
            )
            * days_per_year
        )
        agent = create_agent(
            agent_id=self._next_agent_id,
            x=x,
            y=y,
            age_days=age_days,
            rng=self.rng,
            config=agent_config,
            days_per_year=days_per_year,
            day=0,
            needs=Needs(hunger=self.rng.uniform(0.0, 30.0), energy=self.rng.uniform(70.0, 100.0)),
        )
        terrain = self.world.terrain_at(x, y)
        if terrain is not None:
            agent.memory.known_terrain.add(terrain.value)
        self._next_agent_id += 1
        return agent

    # -- the day loop -------------------------------------------------------

    def step(self) -> DayReport:
        """Simulate exactly one day and return a report of what happened."""
        day = self.clock.day
        report = DayReport(day=day)
        self._day_events.clear()

        # Phase 1: environment, then today's tribe membership and territory.
        self.world.update(day)
        for extinct in self.groups.refresh(self.agents):
            self.diplomacy.forget_group(extinct.id)
            self.settlements.abandon(extinct.id, day)
            self.events.record(
                day, EventKind.MILESTONE, f"The {extinct.name} died out"
            )
            self.chronicle.record(
                day, self.clock.year, Milestone.TRIBE, f"The {extinct.name} died out"
            )

        occupancy = self._build_occupancy()
        eaters: List[Agent] = []

        for agent in self.agents:
            if not agent.alive:
                continue
            try:
                social = self._social_context(agent, occupancy)

                # Phase 2: needs.
                agent.age_days += 1
                apply_daily_upkeep(
                    agent.needs, self.config.agents, self._tech(agent).health
                )
                agent.memory.forget_stale_food(day, self.config.agents)

                # Phase 3: decide.
                goal = decide(agent, self.world, self.config, day, social)
                agent.goal = goal.value

                # Phase 4: act.
                result = execute(
                    agent, goal, self.world, occupancy, self.config, self.rng, day, social
                )
                if result.wants_to_eat:
                    eaters.append(agent)
                if result.discovered_terrain:
                    self._record_discovery(agent, result.discovered_terrain, day)
            except Exception as error:  # One bad agent must not stop the world.
                self.events.record(
                    day,
                    EventKind.ERROR,
                    f"Agent #{agent.id} failed to act: {error}",
                    agent_id=agent.id,
                )

        # Phase 5: food and resources.
        self._process_meals(eaters, day)

        # Phase 6: combat. Positions changed during phase 4, so re-index first.
        occupancy = self._build_occupancy()
        report.battles = self._process_combat(occupancy, day)

        # Phase 7: disease.
        epidemic = self._process_disease(occupancy, day)
        report.plague_deaths = epidemic.death_count

        # Phase 8: deaths, then births.
        report.deaths, report.war_deaths = self._process_deaths(day)
        report.births = self._process_births(day)

        # Phase 9: society -- tribes, research, diplomacy.
        self._process_society(occupancy, day, report)

        # Phase 10: milestones worth remembering.
        self._record_milestones(day)
        report.events = list(self._day_events)

        # Phase 11: clock.
        self.clock.advance()
        self.last_report = report
        return report

    def run(
        self,
        days: int,
        on_day: Optional[Callable[[DayReport], None]] = None,
        stop_when_extinct: bool = True,
    ) -> int:
        """Simulate ``days`` days; return how many were actually simulated.

        ``on_day`` is invoked after each day, which lets a frontend render
        progress without the engine knowing anything about it.
        """
        if days < 0:
            raise ValueError("Cannot simulate a negative number of days.")
        limit = self.config.simulation.max_days_per_command
        if days > limit:
            raise ValueError(f"Refusing to simulate more than {limit:,} days at once.")

        simulated = 0
        for _ in range(days):
            report = self.step()
            simulated += 1
            if on_day is not None:
                on_day(report)
            if stop_when_extinct and not self.agents:
                break
        return simulated

    # -- phase helpers ------------------------------------------------------

    def _build_occupancy(self) -> Occupancy:
        """Index living agents by tile so neighbour lookups stay cheap."""
        occupancy: Occupancy = {}
        for agent in self.agents:
            if agent.alive:
                occupancy.setdefault(agent.position, []).append(agent)
        return occupancy

    def _tech(self, agent: Agent) -> TechEffects:
        """The technology modifiers an agent enjoys through their tribe."""
        group = self.groups.get(agent.group_id)
        return group.knowledge.effects if group is not None else NO_TECH_EFFECTS

    def _social_context(self, agent: Agent, occupancy: Occupancy) -> SocialContext:
        """Assemble the society-derived facts one agent needs for the day."""
        group = self.groups.get(agent.group_id)
        threat = None
        if self.config.combat.enabled and group is not None:
            threat = combat_system.find_enemy(
                agent, occupancy, self.world, self.diplomacy
            )
        if group is None:
            return SocialContext(threat=threat)

        # A tribe with a settlement clusters around it rather than around the
        # drifting centroid -- which is what makes territory stable.
        settlement = self.settlements.for_group(group.id)
        home = settlement.position if settlement is not None else group.centre

        fertility = group.knowledge.effects.fertility
        if settlement is not None:
            fertility += settlement.spec.fertility_bonus
        return SocialContext(
            threat=threat,
            home=home,
            reproduction_cooldown_scale=1.0 / (1.0 + fertility),
            vision_bonus=role_system.foraging_vision_bonus(agent, self.config.roles),
        )

    def _process_meals(self, eaters: List[Agent], day: int) -> None:
        """Resolve every eat intent against the shared food layer (phase 5).

        Agents are served in list order, so a crowded tile genuinely runs out.
        """
        meal_size = self.config.resources.meal_size
        for agent in eaters:
            taken = self.world.resources.take(agent.x, agent.y, meal_size)
            if taken <= 0.0:
                # The ground is bare -- fall back on the tribe's granary.
                taken = self._draw_ration(agent)
            if taken <= 0.0:
                # Nothing here and nothing stored: go looking.
                agent.goal = Goal.SEEK_FOOD.value
                agent.memory.forget_food(agent.position)
                continue
            # Farming, fire and pottery all mean more nutrition per unit eaten.
            nourishment = taken * (1.0 + self._tech(agent).food_yield)
            apply_meal(agent.needs, nourishment, self.config.agents)
            agent.memory.remember_food(agent.position, day, self.config.agents)
            if self.rng.random() < self.config.simulation.food_event_chance:
                self.events.record(
                    day,
                    EventKind.FOOD,
                    f"Agent #{agent.id} ({agent.name}) found food at ({agent.x}, {agent.y})",
                    agent_id=agent.id,
                )

    def _process_combat(self, occupancy: Occupancy, day: int) -> int:
        """Resolve every fight between rival tribes today (phase 6).

        Unprovoked skirmishes sour relations, which is how peacetime violence
        escalates into a declared war a few days later.
        """
        if not self.config.combat.enabled:
            return 0

        fights = combat_system.engagements(
            self.agents,
            occupancy,
            self.world,
            self.groups,
            self.diplomacy,
            self.config.combat,
            self.rng,
        )
        for attacker, defender, provoked in fights:
            result = combat_system.resolve_fight(
                attacker,
                defender,
                self.groups,
                self.config.combat,
                self.config.agents,
                self.config.roles,
                self.config.simulation.days_per_year,
                day,
                provoked,
                self.rng,
            )
            self.total_battles += 1
            if not provoked and result.winner_group and result.loser_group:
                self.diplomacy.adjust(
                    result.winner_group.id,
                    result.loser_group.id,
                    -self.config.diplomacy.skirmish_relation_cost,
                )
            if provoked and attacker.group_id is not None and defender.group_id is not None:
                war = self.diplomacy.wars.get(pair_key(attacker.group_id, defender.group_id))
                if war is not None:
                    war.battles += 1
        return len(fights)

    def _process_disease(self, occupancy: Occupancy, day: int) -> EpidemicReport:
        """Advance illness, contagion and outbreaks (phase 7)."""
        report = self.diseases.update(
            self.agents, occupancy, self.world, self.groups, day, self.rng
        )
        for outbreak in report.started:
            place = self.groups.get(
                next(
                    (
                        agent.group_id
                        for agent in self.agents
                        if agent.outbreak_id == outbreak.id
                    ),
                    None,
                )
            )
            where = f" among the {place.name}" if place else ""
            self.events.record(
                day,
                EventKind.PLAGUE,
                f"{outbreak.disease.name} broke out{where} at {outbreak.origin}",
            )
        for outbreak in report.ended:
            if outbreak.deaths >= 10:
                self.chronicle.record(
                    day,
                    self.clock.year,
                    Milestone.PLAGUE,
                    f"{outbreak.disease.name} killed {outbreak.deaths} over "
                    f"{day - outbreak.started_day} days",
                )
            self.events.record(
                day,
                EventKind.PLAGUE,
                f"The {outbreak.disease.name} burned out after "
                f"{day - outbreak.started_day} days: "
                f"{outbreak.infections} infected, {outbreak.deaths} dead",
            )
        return report

    def _process_deaths(self, day: int) -> Tuple[int, int]:
        """Kill agents whose health, wounds, illness or lifespan ran out.

        Returns ``(total_deaths, war_deaths)``. Causes are attributed in
        priority order: wounds taken today, then illness, then old age, then
        neglected needs.
        """
        survivors: List[Agent] = []
        deaths = 0
        war_deaths = 0

        for agent in self.agents:
            if not agent.alive:
                continue

            killed_in_battle = False
            if agent.needs.is_dead() and agent.wounded_on_day == day:
                agent.die("battle wounds", day)
                killed_in_battle = True
            elif agent.needs.is_dead() and agent.infection is not None:
                agent.die(f"the {agent.infection.disease.name}", day)
                self.total_plague_deaths += 1
            elif agent.age_days >= agent.lifespan_days:
                agent.die("old age", day)
            elif agent.needs.is_dead():
                agent.die(cause_of_death(agent.needs, self.config.agents), day)
            else:
                survivors.append(agent)
                continue

            if killed_in_battle:
                war_deaths += 1
                self.total_war_deaths += 1
                self._attribute_war_death(agent, day)

            deaths += 1
            self.total_deaths += 1
            years = agent.age_years(self.config.simulation.days_per_year)
            tribe = self.groups.get(agent.group_id)
            allegiance = f" of the {tribe.name}" if tribe else ""
            self.events.record(
                day,
                EventKind.DEATH,
                f"Agent #{agent.id} ({agent.name}){allegiance} died of "
                f"{agent.cause_of_death} at age {years:.1f}",
                agent_id=agent.id,
            )
        self.agents = survivors
        return deaths, war_deaths

    def _attribute_war_death(self, agent: Agent, day: int) -> None:
        """Charge a battle death to the war it was fought in."""
        tribe = self.groups.get(agent.group_id)
        if tribe is not None:
            tribe.war_dead += 1
        if agent.group_id is None or agent.wounded_by_group is None:
            return
        war = self.diplomacy.wars.get(pair_key(agent.group_id, agent.wounded_by_group))
        if war is not None:
            war.record_death(agent.group_id)

    def _process_births(self, day: int) -> int:
        """Pair up eligible adjacent adults and create children (phase 6b)."""
        agent_config = self.config.agents
        if len(self.agents) >= agent_config.max_population:
            return 0

        occupancy = self._build_occupancy()
        candidates = [
            agent
            for agent in self.agents
            if agent.goal == Goal.SEEK_MATE.value
            and can_reproduce(agent, self.config, day, self._cooldown_scale(agent))
        ]
        paired: set[int] = set()
        newborns: List[Agent] = []

        for agent in candidates:
            if agent.id in paired:
                continue
            if len(self.agents) + len(newborns) >= agent_config.max_population:
                break
            partner = self._find_partner(agent, occupancy, paired, day)
            if partner is None:
                continue

            paired.add(agent.id)
            paired.add(partner.id)
            agent.last_birth_day = day
            partner.last_birth_day = day

            child = create_agent(
                agent_id=self._next_agent_id,
                x=agent.x,
                y=agent.y,
                age_days=0,
                rng=self.rng,
                config=agent_config,
                days_per_year=self.config.simulation.days_per_year,
                day=day,
                parents=(agent.id, partner.id),
                needs=Needs(
                    hunger=agent_config.newborn_hunger,
                    energy=agent_config.newborn_energy,
                    health=100.0,
                ),
                # Children are born into their parents' tribe, and inherit the
                # average of their temperaments.
                group_id=agent.group_id if agent.group_id is not None else partner.group_id,
                inherited_aggression=(agent.aggression + partner.aggression) / 2.0,
                inherited_caution=(agent.caution + partner.caution) / 2.0,
                inherited_industry=(agent.industry + partner.industry) / 2.0,
                naming_style=self._naming_style(agent, partner),
                inherited_family=agent.family_name or partner.family_name,
            )
            terrain = self.world.terrain_at(child.x, child.y)
            if terrain is not None:
                child.memory.known_terrain.add(terrain.value)
            self._next_agent_id += 1

            agent.children.append(child.id)
            partner.children.append(child.id)
            agent.memory.note(day, f"had a child (#{child.id})")
            partner.memory.note(day, f"had a child (#{child.id})")
            newborns.append(child)

            self.total_births += 1
            self.events.record(
                day,
                EventKind.BIRTH,
                f"Agent #{agent.id} ({agent.name}) and #{partner.id} ({partner.name}) "
                f"had a child: #{child.id} ({child.name})",
                agent_id=child.id,
            )

        self.agents.extend(newborns)
        return len(newborns)

    def _process_society(self, occupancy: Occupancy, day: int, report: DayReport) -> None:
        """Tribes form, specialise, build, research and quarrel (phase 9)."""
        if self.config.groups.enabled:
            self._process_group_membership(occupancy, day)
            self._process_group_splits(day)
            self._process_disbanding(day)
        self._process_roles_and_leadership(day)
        if self.config.settlements.enabled:
            self._process_settlements(day)
        if self.config.technology.enabled:
            self._process_research(day, report)
        if self.config.culture.enabled:
            self._process_culture(day)
        if self.config.diplomacy.enabled:
            self._process_diplomacy(day, report)
        self.chronicle.note_population(day, self.clock.year, len(self.agents))

    def _members_of(self, group: Group) -> List[Agent]:
        """Every living member of a tribe, in list order."""
        return [agent for agent in self.agents if agent.group_id == group.id]

    def _process_disbanding(self, day: int) -> None:
        """Dissolve tribes too small to sustain themselves."""
        for group in self.groups.find_unviable(day):
            members = self._members_of(group)
            self.groups.disband(group, members)
            self.settlements.abandon(group.id, day)
            self.diplomacy.forget_group(group.id)
            self.events.record(
                day,
                EventKind.SOCIETY,
                f"The {group.name} disbanded; its {len(members)} survivors scattered",
            )

    def _process_roles_and_leadership(self, day: int) -> None:
        """Assign professions and pick each tribe's chieftain."""
        for group in self.groups.active():
            members = self._members_of(group)
            if not members:
                continue

            if self.config.roles.enabled:
                at_war = any(
                    war.active
                    for key, war in self.diplomacy.wars.items()
                    if group.id in key
                )
                starving = group.average_hunger >= self.config.roles.starving_hunger
                counts = role_system.assign_roles(
                    members,
                    group.knowledge,
                    at_war,
                    starving,
                    self.config.roles,
                    self.config.agents,
                    self.config.simulation.days_per_year,
                )
                group.role_counts = {role.value: count for role, count in counts.items()}
                group.healer_resistance = role_system.disease_resistance_bonus(
                    counts, len(members), self.config.roles
                )

            chieftain = culture_system.choose_chieftain(
                members, self.config.simulation.days_per_year
            )
            if chieftain is not None:
                if group.chieftain_id != chieftain.id:
                    group.chieftain_id = chieftain.id
                    chieftain.memory.note(day, f"became chieftain of the {group.name}")
                culture_system.apply_chieftain_influence(
                    group, chieftain, members, self.config.culture
                )

    def _process_settlements(self, day: int) -> None:
        """Found, grow, provision and abandon settlements."""
        for group in self.groups.active():
            settlement = self.settlements.for_group(group.id)

            if settlement is None:
                settlement = self.settlements.try_found(group, self.world, day, self.rng)
                if settlement is None:
                    continue
                group.settlement_id = settlement.id
                self.events.record(
                    day,
                    EventKind.SOCIETY,
                    f"The {group.name} founded {settlement.name} at "
                    f"({settlement.x}, {settlement.y})",
                )
                self.chronicle.note_settlement(
                    day, self.clock.year, settlement.id, settlement.name,
                    int(settlement.level), settlement.level_name, group.name,
                )
                continue

            # Farmers fill the granary; everything in it slowly spoils.
            members = self._members_of(group)
            farmers = [
                agent for agent in members if agent.role == Role.FARMER.value
            ]
            self.settlements.harvest(settlement, farmers, self.world, group.knowledge)
            self.settlements.spoil(settlement, group.knowledge)
            settlement.peak_population = max(settlement.peak_population, group.size)

            upgraded = self.settlements.try_upgrade(settlement, group, group.knowledge, day)
            if upgraded is not None:
                self.events.record(
                    day,
                    EventKind.SOCIETY,
                    f"{settlement.name} grew into a {upgraded.name} "
                    f"({group.size} people)",
                )
                self.chronicle.note_settlement(
                    day, self.clock.year, settlement.id, settlement.name,
                    int(settlement.level), settlement.level_name, group.name,
                )

    def _process_culture(self, day: int) -> None:
        """Spread knowledge and food between tribes."""
        for spread in culture_system.diffuse_knowledge(
            self.groups, self.diplomacy, self.config.culture, self.rng
        ):
            self.events.record(
                day,
                EventKind.INVENTION,
                f"The {spread.learner.name} learned {spread.technology.name} "
                f"from the {spread.teacher.name}",
            )
        for trade in culture_system.trade_food(
            self.groups, self.settlements, self.diplomacy, self.config.culture
        ):
            self.events.record(
                day,
                EventKind.SOCIETY,
                f"The {trade.giver.name} sent {trade.amount:.0f} food to the "
                f"{trade.receiver.name}",
            )

    def _process_group_membership(self, occupancy: Occupancy, day: int) -> None:
        """Let unaffiliated agents join a nearby tribe or found a new one."""
        for agent in self.agents:
            if agent.group_id is not None or not agent.alive:
                continue
            if not agent.is_adult(self.config.agents, self.config.simulation.days_per_year):
                continue

            neighbours: List[Agent] = []
            for tile in [agent.position] + self.world.neighbours(agent.x, agent.y):
                neighbours.extend(occupancy.get(tile, ()))

            outcome = self.groups.try_form_or_join(agent, neighbours, day, self.rng)
            if outcome is None:
                continue
            group, founded = outcome
            if founded:
                agent.memory.note(day, f"founded the {group.name}")
                self.events.record(
                    day,
                    EventKind.SOCIETY,
                    f"Agent #{agent.id} ({agent.name}) founded the {group.name}",
                    agent_id=agent.id,
                )

    def _process_group_splits(self, day: int) -> None:
        """Split any tribe that has outgrown its cohesion."""
        for group in self.groups.active():
            if group.size < self.config.groups.split_size:
                continue
            members = [agent for agent in self.agents if agent.group_id == group.id]
            splinter = self.groups.try_split(group, members, day, self.rng)
            if splinter is None:
                continue
            # A splinter group starts out resentful of the tribe it left.
            self.diplomacy.adjust(group.id, splinter.id, -20.0)
            self.events.record(
                day,
                EventKind.SOCIETY,
                f"The {group.name} splintered: {splinter.size} left to found "
                f"the {splinter.name}",
            )

    def _process_research(self, day: int, report: DayReport) -> None:
        """Advance every tribe's research and record what they invent."""
        for group in self.groups.active():
            # Scholars, elders and a developed settlement all multiply the
            # tribe's effective size for research purposes.
            counts = {
                Role(name): count for name, count in group.role_counts.items()
            }
            multiplier = role_system.research_multiplier(counts, self.config.roles)
            settlement = self.settlements.for_group(group.id)
            if settlement is not None:
                multiplier *= 1.0 + settlement.spec.research_bonus
            effective_population = max(1, int(group.size * multiplier))

            invention = try_invent(
                group.knowledge,
                effective_population,
                group.average_hunger,
                self.config.technology,
                self.rng,
            )
            if invention is None:
                continue
            report.inventions.append((group.name, invention))
            self.events.record(
                day,
                EventKind.INVENTION,
                f"The {group.name} invented {invention.name} ({invention.era})",
            )
            self.chronicle.note_invention(
                day, self.clock.year, group.name, invention.id, invention.name, invention.era
            )

    def _process_diplomacy(self, day: int, report: DayReport) -> None:
        """Move relations, declare wars and make peace."""
        capacity = self.world.resources.total_capacity()
        food_fraction = self.world.resources.total_food() / capacity if capacity else 1.0
        declared, concluded = self.diplomacy.update(self.groups, day, food_fraction)

        for war in declared:
            first = self.groups.get(war.tribes[0])
            second = self.groups.get(war.tribes[1])
            if first and second:
                report.wars_declared.append(war)
                message = f"War broke out between the {first.name} and the {second.name}"
                self.events.record(day, EventKind.WAR, message)
                self.chronicle.record(day, self.clock.year, Milestone.WAR, message)
        for war in concluded:
            first = self.groups.get(war.tribes[0])
            second = self.groups.get(war.tribes[1])
            if first and second:
                report.wars_ended.append(war)
                message = (
                    f"The {first.name} and the {second.name} made peace after "
                    f"{day - war.started_day} days and {war.total_casualties()} dead"
                )
                self.events.record(day, EventKind.WAR, message)
                self.chronicle.record(day, self.clock.year, Milestone.WAR, message)

    def _find_partner(
        self, agent: Agent, occupancy: Occupancy, paired: set[int], day: int
    ) -> Optional[Agent]:
        """Find an unpaired, eligible agent on or next to ``agent``'s tile."""
        tiles = [agent.position] + self.world.neighbours(agent.x, agent.y)
        for tile in tiles:
            for other in occupancy.get(tile, ()):
                if other.id == agent.id or other.id in paired or not other.alive:
                    continue
                if other.goal != Goal.SEEK_MATE.value:
                    continue
                if can_reproduce(other, self.config, day, self._cooldown_scale(other)):
                    return other
        return None

    def _cooldown_scale(self, agent: Agent) -> float:
        """How much a tribe's fertility technology shortens birth spacing."""
        fertility = self._tech(agent).fertility
        settlement = self.settlements.for_group(agent.group_id)
        if settlement is not None:
            fertility += settlement.spec.fertility_bonus
        return 1.0 / (1.0 + fertility)

    def _naming_style(self, first: Agent, second: Agent) -> Optional[int]:
        """The naming style a child is born into: its tribe's, if it has one."""
        group = self.groups.get(first.group_id) or self.groups.get(second.group_id)
        return group.naming_style if group is not None else None

    def _draw_ration(self, agent: Agent) -> float:
        """Let a hungry agent eat from their tribe's granary, if near enough.

        This is what a food store is *for*: it keeps a settlement alive through
        a season when the ground around it has been picked bare.
        """
        settlement = self.settlements.for_group(agent.group_id)
        if settlement is None:
            return 0.0
        if agent.distance_to(*settlement.position) > self.config.settlements.ration_radius:
            return 0.0
        return self.settlements.draw_ration(settlement, self.config.settlements.ration_size)

    def _record_discovery(self, agent: Agent, terrain_name: str, day: int) -> None:
        """Note the first time an agent sets foot on a new kind of terrain."""
        phrase = TerrainType(terrain_name).discovery_name
        agent.memory.note(day, f"discovered a {phrase}")
        self.events.record(
            day,
            EventKind.DISCOVERY,
            f"Agent #{agent.id} ({agent.name}) discovered a {phrase}",
            agent_id=agent.id,
        )

    def _record_milestones(self, day: int) -> None:
        """Record rare, world-level events worth keeping in history (phase 7)."""
        if not self.agents:
            self.events.record(day, EventKind.MILESTONE, "The population has died out")
            return
        if day > 0 and day % self.config.simulation.days_per_year == 0:
            self.events.record(
                day,
                EventKind.MILESTONE,
                f"Year {self.clock.year} ended with a population of {len(self.agents)}",
            )

    # -- read-only interface for frontends ----------------------------------

    @property
    def population(self) -> int:
        """Number of living agents."""
        return len(self.agents)

    def get_agent(self, agent_id: int) -> Optional[Agent]:
        """Look up a living agent by id, or ``None`` if there is no such agent."""
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        return None

    def stats(self) -> WorldStats:
        """Build a snapshot of the current simulation state."""
        agents = self.agents
        population = len(agents)
        days_per_year = self.config.simulation.days_per_year
        agent_config = self.config.agents

        total_age = total_health = total_hunger = total_energy = total_aggression = 0.0
        oldest = 0.0
        hungry = resting = seeking_food = eating = wandering = seeking_mate = 0
        fighting = fleeing = unaffiliated = ill = 0

        for agent in agents:
            total_age += agent.age_days
            total_health += agent.needs.health
            total_hunger += agent.needs.hunger
            total_energy += agent.needs.energy
            total_aggression += agent.aggression
            oldest = max(oldest, agent.age_days / days_per_year)
            if agent.needs.is_hungry(agent_config):
                hungry += 1
            if agent.group_id is None:
                unaffiliated += 1
            if agent.infection is not None:
                ill += 1
            goal = agent.goal
            if goal == Goal.REST.value:
                resting += 1
            elif goal == Goal.SEEK_FOOD.value:
                seeking_food += 1
            elif goal == Goal.EAT.value:
                eating += 1
            elif goal == Goal.SEEK_MATE.value:
                seeking_mate += 1
            elif goal == Goal.FIGHT.value:
                fighting += 1
            elif goal == Goal.FLEE.value:
                fleeing += 1
            else:
                wandering += 1

        divisor = float(population or 1)

        tribes = self.groups.active()
        largest = self.groups.largest()
        known_techs: set[str] = set()
        best_era = ERAS[0]
        for group in tribes:
            known_techs |= group.knowledge.known
            if ERAS.index(group.era) > ERAS.index(best_era):
                best_era = group.era
        outbreaks = self.diseases.active_outbreaks()
        settlements = self.settlements.active()
        biggest_settlement = self.settlements.largest()
        role_totals: Dict[str, int] = {}
        for group in tribes:
            for role_name, count in group.role_counts.items():
                role_totals[role_name] = role_totals.get(role_name, 0) + count
        return WorldStats(
            day=self.clock.day,
            year=self.clock.year,
            seed=self.seed,
            population=population,
            births_today=self.last_report.births,
            deaths_today=self.last_report.deaths,
            total_births=self.total_births,
            total_deaths=self.total_deaths,
            average_age_years=(total_age / divisor) / days_per_year,
            average_health=total_health / divisor,
            average_hunger=total_hunger / divisor,
            average_energy=total_energy / divisor,
            hungry=hungry,
            resting=resting,
            seeking_food=seeking_food,
            eating=eating,
            wandering=wandering,
            seeking_mate=seeking_mate,
            fighting=fighting,
            fleeing=fleeing,
            oldest_age_years=oldest,
            total_food=self.world.resources.total_food(),
            food_capacity=self.world.resources.total_capacity(),
            terrain_counts=self.world.terrain_counts(),
            tribes=len(tribes),
            unaffiliated=unaffiliated,
            largest_tribe=largest.name if largest else None,
            largest_tribe_size=largest.size if largest else 0,
            most_advanced_era=best_era,
            technologies_known=len(known_techs),
            average_aggression=total_aggression / divisor,
            active_wars=len(self.diplomacy.active_wars()),
            battles_today=self.last_report.battles,
            total_battles=self.total_battles,
            war_deaths=self.total_war_deaths,
            active_outbreaks=len(outbreaks),
            ill=ill,
            plague_deaths=self.total_plague_deaths,
            current_plague=outbreaks[0].disease.name if outbreaks else None,
            settlements=len(settlements),
            largest_settlement=biggest_settlement.name if biggest_settlement else None,
            largest_settlement_level=(
                biggest_settlement.level_name if biggest_settlement else None
            ),
            total_food_stored=sum(s.food_store for s in settlements),
            role_counts=role_totals,
            chronicle_entries=len(self.chronicle),
        )


class SimulationRunner:
    """Runs an engine continuously on a background thread.

    This exists so that a frontend can offer start/pause without blocking its
    input loop. It is interface-agnostic: the terminal UI uses it today and a
    GUI could use it unchanged.
    """

    def __init__(self, engine: SimulationEngine) -> None:
        self.engine = engine
        self.lock = threading.RLock()
        self.delay = engine.config.simulation.autorun_tick_seconds
        self.batch = 1  # Days simulated per lock acquisition.
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    @property
    def running(self) -> bool:
        """True while the background thread is ticking."""
        return self._thread is not None and self._thread.is_alive()

    def set_delay(self, seconds: float) -> None:
        """Change the pause between ticks; takes effect on the next tick.

        At the fastest setting the runner switches to simulating a batch of days
        per lock acquisition instead of spinning one day at a time, so a display
        thread can still get the lock between batches.
        """
        self.delay = max(0.0, float(seconds))
        settings = self.engine.config.simulation
        self.batch = settings.max_speed_batch_days if self.delay <= 0.0 else 1

    def start(self) -> bool:
        """Begin continuous simulation. Returns False if already running."""
        if self.running:
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="worldbox-sim", daemon=True)
        self._thread.start()
        return True

    def pause(self) -> bool:
        """Stop continuous simulation. Returns False if not running."""
        if not self.running:
            return False
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._thread = None
        return True

    def _loop(self) -> None:
        minimum_sleep = self.engine.config.simulation.min_tick_sleep
        stopped = False
        while not self._stop.is_set() and not stopped:
            with self.lock:
                for _ in range(max(1, self.batch)):
                    try:
                        self.engine.step()
                    except Exception as error:
                        self.engine.events.record(
                            self.engine.clock.day,
                            EventKind.ERROR,
                            f"Simulation halted: {error}",
                        )
                        stopped = True
                        break
                    if not self.engine.agents:
                        stopped = True
                        break
            # The lock is released before sleeping, and the sleep is never zero,
            # so a display thread is always able to acquire it between batches.
            time.sleep(max(self.delay, minimum_sleep))
