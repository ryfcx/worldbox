#!/usr/bin/env python3
"""Turn an exported run into a self-contained HTML map viewer.

    python3 -m worldbox.main --days 0        # then, at the prompt: export 20000
    python3 tools/build_viewer.py worldbox_run.json worldbox.html

The output is one file with the run data inlined, so it opens straight from disk
with no server and no network access.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TEMPLATE = Path(__file__).with_name("viewer_template.html")
PLACEHOLDER = "__RUN_DATA__"


def build(run_json: Path, output: Path, template: Path = TEMPLATE) -> Path:
    """Inline ``run_json`` into ``template`` and write the result to ``output``."""
    if not template.exists():
        raise FileNotFoundError(f"Missing viewer template: {template}")
    if not run_json.exists():
        raise FileNotFoundError(
            f"Missing run data: {run_json}. Run 'export <days>' in worldbox first."
        )

    html = template.read_text(encoding="utf-8")
    if PLACEHOLDER not in html:
        raise ValueError(f"Template has no {PLACEHOLDER} placeholder.")

    # "</" inside the JSON would close the <script> block early.
    data = run_json.read_text(encoding="utf-8").replace("</", "<\\/")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html.replace(PLACEHOLDER, data), encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Worldbox map viewer.")
    parser.add_argument("run", type=Path, nargs="?", default=Path("worldbox_run.json"),
                        help="Exported run JSON (default: worldbox_run.json)")
    parser.add_argument("output", type=Path, nargs="?", default=Path("worldbox.html"),
                        help="Where to write the viewer (default: worldbox.html)")
    args = parser.parse_args(argv)

    try:
        written = build(args.run, args.output)
    except (FileNotFoundError, ValueError) as error:
        print(f"Could not build the viewer: {error}", file=sys.stderr)
        return 1

    print(f"Wrote {written} ({written.stat().st_size / 1024:.0f} KB). Open it in a browser.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
