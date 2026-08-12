"""A live map in the browser, served from the running simulation.

The recorded viewer replays a finished run. This serves the world *as it is
happening*: a tiny standard-library HTTP server hands the page a JSON snapshot
whenever it asks, and the page redraws. Start it at day zero and watch the whole
history unfold.

The separation still holds. The server only reads the engine through the same
lock the terminal uses, and the engine has no idea it exists.

Endpoints:
    GET /            the page itself
    GET /world       terrain and dimensions, fetched once
    GET /state       the current snapshot: agents, settlements, stats, log
    GET /control     pause, resume or change speed
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from ..simulation.engine import SimulationEngine, SimulationRunner
from ..world.terrain import TerrainType

TERRAIN_CODES: Dict[str, int] = {
    TerrainType.WATER.value: 0,
    TerrainType.GRASS.value: 1,
    TerrainType.FOREST.value: 2,
    TerrainType.MOUNTAIN.value: 3,
}

# Event kinds worth showing in the live feed.
LOGGED_KINDS = ("birth", "death", "war", "plague", "invention", "society", "milestone")


class LiveState:
    """Reads snapshots out of a running engine, safely.

    Tribe colour indices are allocated on first sight and kept, so a tribe keeps
    its colour for as long as it exists.
    """

    def __init__(self, engine: SimulationEngine, runner: SimulationRunner) -> None:
        self.engine = engine
        self.runner = runner
        self.tribe_index: Dict[int, int] = {}
        self.log: List[Dict[str, Any]] = []
        self._next_index = 0
        engine.events.subscribe(self._on_event)

    def _on_event(self, event) -> None:
        """Keep a rolling window of notable events for the feed."""
        if event.kind.value not in LOGGED_KINDS:
            return
        self.log.append({"d": event.day, "k": event.kind.value, "m": event.message})
        if len(self.log) > 400:
            del self.log[:-300]

    def _index_for(self, group_id: Optional[int]) -> int:
        if group_id is None:
            return -1
        if group_id not in self.tribe_index:
            self.tribe_index[group_id] = self._next_index
            self._next_index += 1
        return self.tribe_index[group_id]

    def world(self) -> Dict[str, Any]:
        """Terrain and dimensions. Constant for the life of a world."""
        with self.runner.lock:
            grid = self.engine.world.grid
            return {
                "width": self.engine.world.width,
                "height": self.engine.world.height,
                "seed": self.engine.seed,
                "terrain": [
                    "".join(str(TERRAIN_CODES[tile.value]) for tile in row) for row in grid
                ],
            }

    def snapshot(self, since_day: int = -1) -> Dict[str, Any]:
        """Everything the page needs to draw one frame."""
        with self.runner.lock:
            engine = self.engine
            stats = engine.stats()

            agents: List[int] = []
            for agent in engine.agents:
                agents.extend((agent.x, agent.y, self._index_for(agent.group_id)))

            settlements: List[int] = []
            for settlement in engine.settlements.active():
                settlements.extend(
                    (settlement.x, settlement.y, int(settlement.level),
                     self._index_for(settlement.group_id))
                )

            fresh = [entry for entry in self.log if entry["d"] > since_day][-60:]
            capacity = stats.food_capacity or 1.0
            return {
                "day": stats.day,
                "year": stats.year,
                "running": self.runner.running,
                "population": stats.population,
                "tribes": stats.tribes,
                "settlements": stats.settlements,
                "techs": stats.technologies_known,
                "era": stats.most_advanced_era,
                "wars": stats.active_wars,
                "ill": stats.ill,
                "warDeaths": stats.war_deaths,
                "plagueDeaths": stats.plague_deaths,
                "births": stats.total_births,
                "deaths": stats.total_deaths,
                "health": round(stats.average_health, 1),
                "hunger": round(stats.average_hunger, 1),
                "food": round(stats.total_food / capacity, 3),
                "agents": agents,
                "settlementList": settlements,
                "log": fresh,
            }

    def inspect(self, x: int, y: int) -> Dict[str, Any]:
        """Describe whatever is at a tile: settlement, tribe and people."""
        with self.runner.lock:
            engine = self.engine
            dpy = engine.config.simulation.days_per_year
            result: Dict[str, Any] = {"x": x, "y": y}

            terrain = engine.world.terrain_at(x, y)
            result["terrain"] = terrain.label if terrain else "void"
            result["food"] = round(engine.world.resources.food_at(x, y), 1)
            if 0 <= y < engine.world.height and 0 <= x < engine.world.width:
                result["fertility"] = round(engine.world.resources.fertility[y][x], 2)

            # Nearest settlement within a few tiles, so clicking is forgiving.
            best, best_distance = None, 4
            for settlement in engine.settlements.active():
                d = max(abs(settlement.x - x), abs(settlement.y - y))
                if d <= best_distance:
                    best, best_distance = settlement, d

            if best is not None:
                group = engine.groups.get(best.group_id)
                chief = engine.get_agent(group.chieftain_id) if group and group.chieftain_id else None
                result["settlement"] = {
                    "name": best.name,
                    "level": best.level_name,
                    "founded": best.founded_day,
                    "store": round(best.food_store),
                    "storePct": round(best.store_fraction() * 100),
                    "tribe": group.name if group else "abandoned",
                    "population": group.size if group else 0,
                    "era": group.era if group else "",
                    "techs": sorted(group.knowledge.known) if group else [],
                    "chieftain": chief.full_name if chief else None,
                    "chieftainAge": round(chief.age_years(dpy)) if chief else None,
                    "battlesWon": group.battles_won if group else 0,
                    "battlesLost": group.battles_lost if group else 0,
                    "warDead": group.war_dead if group else 0,
                    "plagueDead": group.plague_dead if group else 0,
                    "roles": {k: v for k, v in (group.role_counts or {}).items() if v} if group else {},
                }

            # A couple of the people standing here.
            people = []
            for agent in engine.agents:
                if max(abs(agent.x - x), abs(agent.y - y)) <= 1:
                    tribe = engine.groups.get(agent.group_id)
                    people.append({
                        "id": agent.id,
                        "name": agent.full_name,
                        "age": round(agent.age_years(dpy)),
                        "role": agent.role,
                        "goal": agent.goal,
                        "health": round(agent.needs.health),
                        "hunger": round(agent.needs.hunger),
                        "tribe": tribe.name if tribe else None,
                        "ill": agent.infection.disease.name if agent.infection else None,
                    })
                    if len(people) >= 6:
                        break
            result["people"] = people
            return result

    def explain(self) -> Dict[str, Any]:
        """Ask the AI what has been happening, from the chronicle.

        Runs outside the simulation lock: it is a network call taking seconds,
        and holding the lock would freeze the world while it waited.
        """
        from ..ai import narrator

        if not narrator.is_available():
            return {"ok": False, "error":
                    f"No API key. Set {narrator.ENV_KEY} before starting Worldbox."}
        with self.runner.lock:
            entries = list(self.engine.chronicle.entries)
            year = self.engine.clock.year
        if not entries:
            return {"ok": False, "error": "Nothing has happened yet."}
        result = narrator.narrate(entries, max_entries=80)
        if not result.ok:
            return {"ok": False, "error": result.error}
        return {"ok": True, "year": year, "text": result.text}

    def control(self, action: str, value: Optional[str]) -> Dict[str, Any]:
        """Pause, resume or set the speed from the browser."""
        if action == "pause":
            self.runner.pause()
        elif action == "resume":
            self.runner.start()
        elif action == "speed" and value is not None:
            try:
                self.runner.set_delay(float(value))
            except ValueError:
                pass
        return {"running": self.runner.running, "delay": self.runner.delay}


def _handler_factory(state: LiveState, page: str):
    """Build a request handler bound to one simulation."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:  # noqa: D102
            pass  # Silence per-request logging; it would spam the terminal.

        def _send(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: Dict[str, Any]) -> None:
            self._send(json.dumps(payload).encode("utf-8"), "application/json")

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    self._send(page.encode("utf-8"), "text/html; charset=utf-8")
                elif parsed.path == "/world":
                    self._json(state.world())
                elif parsed.path == "/state":
                    since = int(query.get("since", ["-1"])[0])
                    self._json(state.snapshot(since))
                elif parsed.path == "/inspect":
                    x = int(query.get("x", ["0"])[0])
                    y = int(query.get("y", ["0"])[0])
                    self._json(state.inspect(x, y))
                elif parsed.path == "/explain":
                    self._json(state.explain())
                elif parsed.path == "/control":
                    action = query.get("action", [""])[0]
                    value = query.get("value", [None])[0]
                    self._json(state.control(action, value))
                else:
                    self.send_error(404)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Browser navigated away mid-response.
            except Exception as error:  # Never let one bad request kill the server.
                try:
                    self.send_error(500, str(error))
                except OSError:
                    pass

    return Handler


class LiveServer:
    """Serves the live map on localhost while the simulation runs."""

    def __init__(
        self, engine: SimulationEngine, runner: SimulationRunner, port: int = 8765
    ) -> None:
        self.state = LiveState(engine, runner)
        self.port = port
        self.httpd: Optional[ThreadingHTTPServer] = None
        self.thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        """Where the page is served."""
        return f"http://127.0.0.1:{self.port}/"

    def start(self) -> str:
        """Start serving in a background thread; returns the URL.

        If the port is busy, tries the next few before giving up.
        """
        handler = _handler_factory(self.state, PAGE)
        last_error: Optional[OSError] = None
        for offset in range(8):
            try:
                self.httpd = ThreadingHTTPServer(("127.0.0.1", self.port + offset), handler)
                self.port += offset
                break
            except OSError as error:
                last_error = error
        if self.httpd is None:
            raise OSError(f"Could not bind a port near {self.port}: {last_error}")

        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return self.url

    def stop(self) -> None:
        """Shut the server down."""
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Worldbox — live</title>
<style>
  :root{
    --bg:#EDEFE9;--panel:#F8F9F5;--line:#CBD1C6;--ink:#10130F;--ink2:#4C554B;--ink3:#79826F;
    --accent:#1F6F78;
    --war:#A83A20;--plague:#6E4795;--invention:#A4761A;--society:#2C6E4B;
    --birth:#4C554B;--death:#79826F;--milestone:#1F6F78;
    --w:#A9BCC9;--g:#D6DAC6;--f:#B2C0A2;--m:#C3C1BA;
    --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
    --ui:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  }
  @media (prefers-color-scheme:dark){:root{
    --bg:#0E1210;--panel:#161B18;--line:#2A322C;--ink:#E5EAE3;--ink2:#A3ADA3;--ink3:#77817A;
    --accent:#57B3B8;
    --war:#E0714F;--plague:#B08BD8;--invention:#D9AC4A;--society:#5FB884;
    --birth:#A3ADA3;--death:#77817A;--milestone:#57B3B8;
    --w:#22323D;--g:#2B3427;--f:#36442E;--m:#3A3A37;
  }}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--ui);font-size:14px}
  .app{max-width:1240px;margin:0 auto;padding:18px clamp(10px,3vw,22px) 36px;
       display:flex;flex-direction:column;gap:12px}
  .bar{display:flex;align-items:baseline;justify-content:space-between;gap:14px;
       flex-wrap:wrap;border-bottom:1px solid var(--line);padding-bottom:9px}
  .bar h1{font-family:var(--mono);font-size:14px;letter-spacing:.22em;text-transform:uppercase;margin:0}
  .bar .meta{font-family:var(--mono);font-size:11.5px;color:var(--ink3)}
  .live{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--accent);
        margin-right:6px;vertical-align:middle}
  .live.off{background:var(--ink3)}
  .main{display:grid;grid-template-columns:minmax(0,1fr) 230px;gap:12px;align-items:start}
  @media(max-width:830px){.main{grid-template-columns:minmax(0,1fr)}}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:2px}
  canvas{display:block;width:100%;height:auto;image-rendering:pixelated}
  .ctl{display:flex;align-items:center;gap:9px;padding:9px 11px;border-top:1px solid var(--line);flex-wrap:wrap}
  button{font-family:var(--mono);font-size:12px;font-weight:600;color:var(--panel);
         background:var(--accent);border:none;border-radius:2px;padding:6px 13px;cursor:pointer}
  button:hover{filter:brightness(1.12)}
  button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
  select{font-family:var(--mono);font-size:12px;padding:4px 6px;background:var(--panel);
         color:var(--ink);border:1px solid var(--line);border-radius:2px}
  .stamp{font-family:var(--mono);font-size:12px;color:var(--ink2);font-variant-numeric:tabular-nums}
  .info{font-family:var(--mono);font-size:12px;padding:11px;line-height:1.75}
  .info h2{font-size:10px;letter-spacing:.16em;text-transform:uppercase;color:var(--ink3);
           margin:0 0 7px;font-weight:600}
  .info .r{display:flex;justify-content:space-between;gap:8px}
  .info .r span:first-child{color:var(--ink2)}
  .info .r span:last-child{font-variant-numeric:tabular-nums}
  .info hr{border:none;border-top:1px solid var(--line);margin:8px 0}
  .info .era{color:var(--accent);font-size:11px;letter-spacing:.09em;text-transform:uppercase}
  .head{font-family:var(--mono);font-size:10px;letter-spacing:.16em;text-transform:uppercase;
        color:var(--ink3);padding:9px 12px;border-bottom:1px solid var(--line);font-weight:600}
  #log{font-family:var(--mono);font-size:11.5px;line-height:1.7;height:210px;overflow-y:auto;
       padding:7px 12px;margin:0;list-style:none}
  #log li{display:flex;gap:9px}
  #log .d{font-variant-numeric:tabular-nums;white-space:nowrap}
  #log .m{color:var(--ink2)}
  footer{font-family:var(--mono);font-size:11px;color:var(--ink3);
         border-top:1px solid var(--line);padding-top:9px}
  .mapwrap{overflow:auto;max-height:70vh;background:var(--panel)}
  canvas{cursor:crosshair}
  .zoom{display:flex;align-items:center;gap:5px;font-family:var(--mono);font-size:11px;color:var(--ink3)}
  .zoom button{min-width:26px;padding:4px 8px}
  .lower{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:12px;align-items:start}
  @media(max-width:830px){.lower{grid-template-columns:minmax(0,1fr)}}
  .pad{padding:11px 13px}
  .insp{font-family:var(--mono);font-size:11.5px;line-height:1.65;max-height:210px;overflow-y:auto}
  .insp h3{font-size:13px;margin:0 0 2px;font-family:var(--ui)}
  .insp .sub{color:var(--ink3);margin-bottom:7px}
  .insp .r{display:flex;justify-content:space-between;gap:8px}
  .insp .r span:first-child{color:var(--ink2)}
  .insp .tags{margin-top:6px;color:var(--ink3);font-size:10.5px;line-height:1.5}
  .insp .who{border-top:1px solid var(--line);margin-top:7px;padding-top:6px}
  .hint{color:var(--ink3);font-style:italic}
  .ai{max-height:210px;overflow-y:auto;font-size:13px;color:var(--ink2);line-height:1.6}
  .ai p{margin:0 0 9px}
  .marker{pointer-events:none}
</style></head><body>
<div class="app">
  <div class="bar"><h1>Worldbox <span style="letter-spacing:0;text-transform:none;font-weight:400">live</span></h1>
    <div class="meta" id="meta"></div></div>
  <div class="main">
    <div class="card">
      <div class="mapwrap"><canvas id="map"></canvas></div>
      <div class="ctl">
        <button id="toggle">Pause</button>
        <span class="zoom"><button id="zoomout">-</button><span id="zlabel">1x</span><button id="zoomin">+</button></span>
        <label class="stamp">Speed
          <select id="speed">
            <option value="0.25">slow</option>
            <option value="0.06" selected>normal</option>
            <option value="0.01">fast</option>
            <option value="0">max</option>
          </select></label>
        <span class="stamp" id="stamp"></span>
      </div>
    </div>
    <div class="card info" id="info"></div>
  </div>
  <div class="lower">
    <div class="card"><div class="head">Inspector</div>
      <div class="insp pad" id="insp"><span class="hint">Click anywhere on the map to inspect it.</span></div></div>
    <div class="card"><div class="head">What is happening
      <button id="explain" style="float:right;margin-top:-3px">Ask AI</button></div>
      <div class="ai pad" id="ai"><span class="hint">Ask the AI to summarise the history so far.</span></div></div>
  </div>
  <div class="card"><div class="head">Event log</div><ul id="log"></ul></div>
  <footer id="foot">Polling the running simulation.</footer>
</div>
<script>
(function(){
"use strict";
var TER=["--w","--g","--f","--m"],W=0,H=0,cell=8,terrain=null,img=null,lastDay=-1,seed=0,lastState=null;
var cv=document.getElementById("map"),ctx=cv.getContext("2d");ctx.imageSmoothingEnabled=false;
function tok(n){return getComputedStyle(document.documentElement).getPropertyValue(n).trim()}
function col(i){return i<0?null:"hsl("+((i*47)%360)+" 62% "+(52+((i*13)%16))+"%)"}
var PERSON=[".#.","###",".#.","#.#"];
var BUILD=[["..x..",".x#x.","x###x","#####"],
           [".xxx.","x###x","#o#o#","#####"],
           ["x.xxx.x","xx###xx","x#o#o#x","x#####x","xxxxxxx"],
           [".x...x.","xxx.xxx","x#xxx#x","x#o#o#x","x#####x","x#o#o#x","xxxxxxx"],
           ["x.x.x.x.x","xxxxxxxxx","x#o#o#o#x","x#######x","x#o#o#o#x","x#######x","x#o#o#o#x","xxxxxxxxx"]];
var cache={};
function sh(c,d){var m=/hsl\((\d+) (\d+)% (\d+)%\)/.exec(c);if(!m)return c;
  return "hsl("+m[1]+" "+m[2]+"% "+Math.max(6,Math.min(94,+m[3]+d))+"%)"}
function spr(b,c,s){var k=b.length+"x"+b[0].length+c+s;if(cache[k])return cache[k];
  var e=document.createElement("canvas");e.width=b[0].length*s;e.height=b.length*s;
  var g=e.getContext("2d"),f={"#":c,"o":sh(c,14),"x":sh(c,-22)};
  for(var y=0;y<b.length;y++)for(var x=0;x<b[0].length;x++){var ch=b[y][x];if(ch===".")continue;
    g.fillStyle=f[ch]||c;g.fillRect(x*s,y*s,s,s)}
  cache[k]=e;return e}
var zoom=1;
function size(){var dpr=window.devicePixelRatio||1;
  var base=Math.max(3,Math.floor(cv.parentElement.clientWidth/W));
  cell=Math.max(2,Math.round(base*zoom));
  cv.style.width=cell*W+"px";cv.style.height=cell*H+"px";
  cv.width=Math.round(cell*W*dpr);cv.height=Math.round(cell*H*dpr);
  ctx.setTransform(dpr,0,0,dpr,0,0);ctx.imageSmoothingEnabled=false;img=null}
function paint(){var c=TER.map(tok);
  for(var y=0;y<H;y++){var r=terrain[y];for(var x=0;x<W;x++){ctx.fillStyle=c[+r[x]];
    ctx.fillRect(x*cell,y*cell,cell,cell)}}
  img=ctx.getImageData(0,0,cv.width,cv.height)}
function draw(s){if(img)ctx.putImageData(img,0,0);else paint();
  var grey=tok("--ink3"),bs=Math.max(1,Math.round(cell/4)),ps=Math.max(1,Math.round(cell/5));
  var L=s.settlementList;
  for(var k=0;k<L.length;k+=4){var lv=Math.max(0,Math.min(4,L[k+2]));
    var sp=spr(BUILD[lv],col(L[k+3])||grey,bs);
    ctx.drawImage(sp,Math.round(L[k]*cell+cell/2-sp.width/2),Math.round(L[k+1]*cell+cell/2-sp.height/2))}
  var A=s.agents;
  for(var j=0;j<A.length;j+=3){var p=spr(PERSON,col(A[j+2])||grey,ps);
    ctx.drawImage(p,Math.round(A[j]*cell+cell/2-p.width/2),Math.round(A[j+1]*cell+cell/2-p.height/2))}}
function markSelection(){if(!sel)return;
  ctx.strokeStyle=tok("--accent");ctx.lineWidth=Math.max(1,cell*0.2);
  ctx.strokeRect(sel.x*cell,sel.y*cell,cell,cell)}
function row(a,b){return '<div class="r"><span>'+a+'</span><span>'+b+'</span></div>'}
function info(s){document.getElementById("info").innerHTML=
  "<h2>World</h2>"+row("Day",s.day.toLocaleString())+row("Year",s.year)+
  row("Population",s.population.toLocaleString())+row("Births",s.births.toLocaleString())+
  row("Deaths",s.deaths.toLocaleString())+"<hr><h2>Society</h2>"+
  row("Tribes",s.tribes)+row("Settlements",s.settlements)+row("Technologies",s.techs+"/16")+
  '<div class="era">'+(s.era||"pre-technological")+"</div><hr><h2>Pressures</h2>"+
  row("Wars",s.wars)+row("Ill",s.ill)+row("War dead",s.warDeaths.toLocaleString())+
  row("Plague dead",s.plagueDeaths.toLocaleString())+row("Avg health",s.health)+
  row("Avg hunger",s.hunger)+row("Wild food",Math.round(s.food*100)+"%")}
var KIND={war:"--war",plague:"--plague",invention:"--invention",society:"--society",
          birth:"--birth",death:"--death",milestone:"--milestone"};
var logEl=document.getElementById("log");
function addLog(items){if(!items.length)return;
  var stick=logEl.scrollTop+logEl.clientHeight>=logEl.scrollHeight-16;
  items.forEach(function(e){var li=document.createElement("li");
    li.innerHTML='<span class="d" style="color:var('+(KIND[e.k]||"--ink3")+')">'+e.d+'</span>'+
      '<span class="m">'+e.m.replace(/&/g,"&amp;").replace(/</g,"&lt;")+'</span>';
    logEl.appendChild(li)});
  while(logEl.children.length>300)logEl.removeChild(logEl.firstChild);
  if(stick)logEl.scrollTop=logEl.scrollHeight}
var running=true;
function poll(){fetch("/state?since="+lastDay).then(function(r){return r.json()}).then(function(s){
    lastDay=s.day;running=s.running;lastState=s;
    draw(s);info(s);addLog(s.log);
    if(sel)markSelection();
    document.getElementById("stamp").textContent="Year "+s.year+" · day "+s.day.toLocaleString();
    document.getElementById("meta").innerHTML='<span class="live'+(running?"":" off")+
      '"></span>seed '+seed+" · "+W+"×"+H+(running?" · running":" · paused");
    document.getElementById("toggle").textContent=running?"Pause":"Resume";
  }).catch(function(){
    document.getElementById("foot").textContent="Lost contact with the simulation — is it still running?";
  })}
document.getElementById("toggle").addEventListener("click",function(){
  fetch("/control?action="+(running?"pause":"resume")).then(poll)});
document.getElementById("speed").addEventListener("change",function(){
  fetch("/control?action=speed&value="+this.value)});
function setZoom(z){var old=zoom;zoom=Math.max(1,Math.min(6,z));
  if(zoom===old)return;
  document.getElementById("zlabel").textContent=zoom+"x";
  cache={};size();paint();if(lastState)draw(lastState);}
document.getElementById("zoomin").addEventListener("click",function(){setZoom(zoom+1)});
document.getElementById("zoomout").addEventListener("click",function(){setZoom(zoom-1)});
cv.addEventListener("wheel",function(ev){if(!ev.ctrlKey&&!ev.metaKey)return;
  ev.preventDefault();setZoom(zoom+(ev.deltaY<0?1:-1))},{passive:false});

// --- click to inspect ----------------------------------------------------
var sel=null;
function esc(t){return String(t).replace(/&/g,"&amp;").replace(/</g,"&lt;")}
cv.addEventListener("click",function(ev){
  var r=cv.getBoundingClientRect();
  var x=Math.floor((ev.clientX-r.left)/cell),y=Math.floor((ev.clientY-r.top)/cell);
  if(x<0||y<0||x>=W||y>=H)return;
  sel={x:x,y:y};
  fetch("/inspect?x="+x+"&y="+y).then(function(r){return r.json()}).then(showInspect);
});
function showInspect(d){
  var h='';
  var s=d.settlement;
  if(s){
    h+='<h3>'+esc(s.name)+'</h3><div class="sub">'+esc(s.level)+' of the '+esc(s.tribe)+'</div>';
    h+=row2("Population",s.population)+row2("Founded","day "+s.founded);
    h+=row2("Granary",s.store+" ("+s.storePct+"%)");
    if(s.era)h+=row2("Era",esc(s.era));
    if(s.chieftain)h+=row2("Chieftain",esc(s.chieftain)+" ("+s.chieftainAge+")");
    if(s.battlesWon+s.battlesLost>0)h+=row2("Battles",s.battlesWon+" won / "+s.battlesLost+" lost");
    if(s.warDead)h+=row2("War dead",s.warDead);
    if(s.plagueDead)h+=row2("Plague dead",s.plagueDead);
    var roles=Object.keys(s.roles||{}).map(function(k){return k+" "+s.roles[k]}).join(", ");
    if(roles)h+='<div class="tags">'+esc(roles)+'</div>';
    if(s.techs&&s.techs.length)h+='<div class="tags">'+esc(s.techs.join(", "))+'</div>';
  }else{
    h+='<h3>('+d.x+', '+d.y+')</h3><div class="sub">'+esc(d.terrain)+'</div>';
    h+=row2("Food here",d.food);
    if(d.fertility!==undefined)h+=row2("Fertility",d.fertility+"x");
  }
  if(d.people&&d.people.length){
    h+='<div class="who">';
    d.people.forEach(function(p){
      h+='<div>'+esc(p.name)+' &middot; '+p.age+'y '+esc(p.role)+' &middot; '+esc(p.goal)+
         (p.ill?' &middot; <span style="color:var(--plague)">'+esc(p.ill)+'</span>':'')+'</div>';
    });
    h+='</div>';
  }else if(!s){h+='<div class="who hint">Nobody here.</div>'}
  document.getElementById("insp").innerHTML=h;
}
function row2(a,b){return '<div class="r"><span>'+a+'</span><span>'+b+'</span></div>'}

// --- AI ------------------------------------------------------------------
document.getElementById("explain").addEventListener("click",function(){
  var el=document.getElementById("ai");
  el.innerHTML='<span class="hint">Writing the history...</span>';
  fetch("/explain").then(function(r){return r.json()}).then(function(d){
    if(!d.ok){el.innerHTML='<span class="hint">'+esc(d.error)+'</span>';return}
    el.innerHTML='<div class="sub" style="font-family:var(--mono);font-size:10px;letter-spacing:.1em;'+
      'text-transform:uppercase;color:var(--accent);margin-bottom:7px">To year '+d.year+'</div>'+
      d.text.split(/\n+/).map(function(p){
        return '<p>'+esc(p.replace(/\*\*/g,"").replace(/^#+\s*/,""))+'</p>'}).join("");
  }).catch(function(){el.innerHTML='<span class="hint">Could not reach the simulation.</span>'});
});

window.addEventListener("resize",function(){size();paint();if(lastState)draw(lastState);});
fetch("/world").then(function(r){return r.json()}).then(function(w){
  W=w.width;H=w.height;terrain=w.terrain;seed=w.seed;
  size();paint();poll();setInterval(poll,500)});
})();
</script></body></html>
"""
