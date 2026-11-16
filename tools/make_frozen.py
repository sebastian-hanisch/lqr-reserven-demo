"""Erzeugt `data/res_frozen.json`: die vorgerechneten Messungen der App (Experimente 5 und 6).

Aufruf (aus dem Repo-Ordner):  python tools/make_frozen.py            schreibt die Datei
                               python tools/make_frozen.py --check    rechnet neu und vergleicht mit der Datei (Zählungen exakt, Gleitkommazahlen als Band)

Inhalt:
  lqg90   die 90 LQG-Entwürfe am Wagen (Ts = 1, Euler-Form; 3 Regler × 2 Formen × 3 Rauschrichtungen × 5 Rauschstärken), Zeile je Entwurf, und ihre Zusammenfassung
  bound   die Schranke min|1 + L| ≥ √(R/(R + BᵀPB)) an 3 000 Zufallssystemen (Zufalls-Seed 15, SplitMix64)
  kalman  die stetige Kalman-Ungleichung min|1 + L| ≥ 1 an 3 000 Zufallssystemen (Zufalls-Seed 15, SplitMix64)

Rechenzeit: etwa eine halbe Minute (vorgerechnet, damit die App nicht warten muss). Die Zufallssysteme kommen aus dem ganzzahligen SplitMix64 mit
exakten Logarithmus- und Kosinus-Routinen (res_rng.py), nicht aus dem NumPy-Generator: die Zählungen gelten für diesen Zufallsstrom."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import res_constants as C  # noqa: E402
import res_evaluation as ev  # noqa: E402
import res_margins as M  # noqa: E402

OUT = ROOT / "data" / "res_frozen.json"


def compute() -> dict:
    rows = ev.lqg90_rows()
    return {
        "lqg90": rows,
        "lqg90_summary": ev.lqg90_summary(rows),
        "bound": M.bound_check(C.FROZEN_SEED_BOUND, C.FROZEN_COUNT),
        "kalman": M.kalman_check(C.FROZEN_SEED_BOUND, C.FROZEN_COUNT),
    }


def main() -> int:
    t0 = time.time()
    data = compute()
    text = json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
    if "--check" in sys.argv:
        old = json.loads(OUT.read_text(encoding="utf-8"))
        new = json.loads(text)
        same_counts = all(old["bound"][k] == new["bound"][k] for k in ("seed", "requested", "ok", "good", "violations_bound", "violations_gain")) \
            and old["lqg90_summary"]["below_lqr"] == new["lqg90_summary"]["below_lqr"]
        print("Zählungen gleich:", same_counts, f"({time.time() - t0:.0f} s)")
        return 0 if same_counts else 1
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"geschrieben: {OUT} ({time.time() - t0:.0f} s)")
    s = data["lqg90_summary"]
    print("LQG 90:", s)
    print("Schranke:", data["bound"])
    print("Kalman:", data["kalman"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
