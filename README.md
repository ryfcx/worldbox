# worldbox

A civilisation simulator that runs in the terminal.

You start with 100 people on a procedurally generated map. They're hungry, they get
tired, and they die if you ignore either. Everything after that — tribes, villages,
farming, wars, plagues — is something they work out themselves.

Runs on Python 3.10+ with no dependencies.

## Running it

```bash
python3 -m worldbox.main                 # interactive
python3 -m worldbox.main --seed 42       # pick a seed
python3 -m worldbox.main --days 5000     # run headless, print a summary, exit
```

Type `start` for a live dashboard that repaints while the simulation runs. Space
pauses, `+`/`-` changes speed, `q` goes back to the prompt.

## What actually happens

People wander until they bump into each other. Two who meet might start a tribe;
after that, anyone nearby tends to get pulled in. Kids are born into their parents'
tribe and inherit a surname and a temperament.

A tribe of eight builds a settlement, and that changes things — people now cluster
around a fixed point instead of drifting, so tribes start holding actual territory.
Farmers fill a granary. The granary is what carries everyone through the weeks when
the ground nearby has been picked clean, and a settlement won't grow past its
current size until it has a surplus sitting in store.

Research comes from having people, feeding them, and eventually having scholars.
Tribes that get along copy each other's ideas, so a neighbourhood advances together
and a tribe off on its own falls behind. Grow past 55 and you splinter; drop below
four and you fold into whoever's nearby.

Wars come from crowding. Two tribes sharing ground get on worse, especially in a
bad year, and grudges fade slowly when they're apart. Fights make it worse. That
combination gives you waves of conflict rather than one endless war.

Plague needs crowds. It spreads between neighbours, medicine and healers blunt it,
and anyone who survives is immune — so an outbreak burns through, dies out for lack
of hosts, and comes back a generation later once enough new people have been born.

## Commands

| | |
| --- | --- |
| `start` | live dashboard |
| `step` / `advance <n>` | one day / n days (bare numbers work too) |
| `chronicle` | the history — what this is all for |
| `tribes` / `tribe <id>` | who exists, and one of them in detail |
| `minds` | what each tribe AI has decided to do |
| `tech` | the tech tree and who's got what |
| `wars` / `towns` | current conflicts, current settlements |
| `agent <id>` | one person: needs, memory, family, job |
| `view [days]` | build the graphical map and open it |
| `narrate` | have an AI write the chronicle up as prose |
| `export [days]` | record a run to JSON for the map viewer |
| `stats` / `events` | numbers, recent activity |
| `seed <n>` / `reset` | new world / same world again |

## The chronicle

The event log is a ring buffer, so old history scrolls away. The chronicle keeps
only turning points and keeps them forever:

```
Year    1 (day    650) | The Brananborn Band invented Fire
Year    1 (day    650) | The world entered the Stone Age
Year    6 (day   2272) | The Oleliel Folk invented Agriculture
Year    6 (day   2337) | Dunylmund Post grew into a Village
Year    6 (day   2387) | Black Plague killed 18 over 14 days
```

## Editing how agents think

`worldbox/rules.py` is the one file to edit. Agents score each thing they could
do from 0 to 1 and take the highest. Three ways in:

- Tune `WEIGHTS` at the top — raise `fight` and the world gets more violent.
- Rewrite a scorer — they're a few lines each.
- Add an action: put it in `Goal`, write a scorer, teach `execute()` to do it.

Then check what changed: `python3 tools/compare_runs.py --days 6000`

## Seeing it

Live, while it runs:

```bash
python3 -m worldbox.main --live
```

Opens a browser map that updates twice a second. Zoom with the `+`/`-` buttons
(or ctrl-scroll), click anywhere to inspect that tile — settlement, tribe,
chieftain, granary, professions, and whoever is standing there — and press
**Ask AI** for a written history of the run so far. The **Minds** panel shows
what each tribe has decided to do — farm, raid, research — and those decisions
change the simulation, not just the prose. `livemap` does the same from
inside the prompt.

Recorded, to scrub through afterwards:

```bash
python3 -m worldbox.main --view 20000
```

Simulates, builds a self-contained HTML page and opens it: a pixel map you can
scrub through, terminal-style readouts, an event log, and AI commentary that
follows the timeline. `view 20000` does the same from inside the prompt.

`--no-open` writes the page without launching a browser. The two steps are still
available separately as `export` plus `tools/build_viewer.py`.

## How agents decide

Each day every agent scores every action it could take — eat, look for food,
rest, fight, flee, seek a mate, wander — on 0 to 1, and does whichever scores
highest. Scores curve rather than step, so a mildly hungry agent isn't the same
as a starving one and behaviour changes smoothly instead of flipping at a line.

That matters more than it sounds. The first version was a priority ladder:
check threat, then tiredness, then hunger, first match wins. It couldn't weigh
anything against anything else — an agent at energy 30 rested while starving,
purely because tiredness sat higher in the list.

Two traits, `caution` and `industry`, weight those scores per agent and are
inherited with drift, so temperament is under selection rather than fixed.

Bodies age. Past 45 a ceiling comes down on how healthy anyone can be — about 78
at 55, 56 at 65 — so the old are permanently closer to the edge and are the first
to be finished off by famine, plague or a fight. Health is modelled as a ceiling
rather than a daily drain because a drain has to out-race the recovery term, and
the values needed to do that turn into a death spiral.

The old ladder is still there as `decide_by_rules`, behind
`UtilityConfig.enabled`. `tools/compare_runs.py` runs both on the same seeds
and prints the difference.

## AI minds

The point of Worldbox is to trap language models in a simulator and see what
they do with a people. Narration is optional colour; the minds are the players.

Each tribe that forms is handed to DeepSeek-V4.1-Flash (`deepseek-flash`). Every
few dozen days it looks at the tribe's food, neighbours, wars and granary, and
returns a policy. That policy is applied to the real systems:

- who farms, fights, studies or heals
- how readily members fight, flee or forage
- how tightly they hold territory
- daily diplomacy toward named neighbours (war, peace, trade)
- custom crafts invented from materials actually around the tribe — river,
  forest, mountain, granary, a war — paid for with research like Fire or Pottery.

Those inventions really change the tribe: fish traps raise food yield, a
hill-fort raises combat, and they show up in the **Discoveries** panel as they
land. A mind cannot will iron out of grassland; the proposal is dropped if the
materials are not there.

People still pick a daily action by scoring needs. The mind sets the weights
and the politics those scores live inside. Tiny bands stay on the ordinary
rules until they are large enough to be worth a call.

Put a key in `.env` at the project root (copy `.env.example`) and run it:

```bash
# .env
DEEPSEEK_API_KEY=your-key-here

python3 -m worldbox.main --live
```

`export DEEPSEEK_API_KEY=...` still works, and overrides the file if both are set.

Watch the **Minds** panel. Or from the prompt, `minds` and `tribe <id>`.
`narrate` still writes the chronicle up as prose, using the same model.

`--no-ai` turns the minds off even if a key is set, which is what you want for
a repeatable seed. No key, or a failed call, leaves a tribe on utility — the
world never stalls waiting for the network.

Nothing else in the simulation touches the network. The key is read from
`.env` or the environment, never from the repository. `.env` is gitignored.

## Layout

```
worldbox/
  config.py          every tunable number, imports nothing else
  world/             terrain generation, the grid, the food layer
  agents/            state, metabolism, decisions, names
  society/           tribes, settlements, tech, jobs, diplomacy, war, disease
  simulation/        the day loop, events, the chronicle, the clock
  cli/               prompt and live dashboard
  ai/                DeepSeek client, tribe minds, optional narration
  export.py          record a run to JSON
```

Dependencies only point one way: `cli → simulation → society → agents → world → config`.
Tribe minds sit beside the engine and write policies back into society; the
engine does not know the terminal exists, which is what makes a GUI possible
without touching any of the simulation.

## A day

1. Food regrows, tribe membership is recounted
2. Everyone gets hungrier, older, more tired
3. Everyone picks a goal
4. Everyone acts — move, rest, fight, run
5. Meals resolve against the ground, then the granary
6. Fights between rival tribes
7. Disease spreads
8. Deaths, then births
9. Tribes form, split, specialise, research, argue
10. Turning points recorded
11. Clock ticks

Combat runs before deaths so a killing blow gets attributed to the right war.
Society runs after births so newborns count toward tribe size.

## Seeds

One seeded RNG drives everything. Same seed, same history, down to the day. `reset`
replays it exactly.

Live mode is the exception — how many days pass depends on wall-clock time, so use
`advance n` if you want a run you can repeat.

## Tuning

`worldbox/config.py`. Map size, food, metabolism, lifespans, how easily tribes form,
how fast research goes, how quickly relations sour, how deadly each disease is.

Every system has an `enabled` flag. Turning one off is the fastest way to see what
it was contributing.

On seed 1337: Stone Age around year 27, farming and the first villages by year 82,
a few hundred people across a dozen tribes. Watch it with `chronicle`.
