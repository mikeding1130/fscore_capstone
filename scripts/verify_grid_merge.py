"""Prove the 18-notebook -> 2-notebook merge left every number untouched.

The merge rests on a claim about seeding: because every random draw in
`fscore.grid` comes from a locally constructed `np.random.default_rng`, a
grid cell's result depends only on (market, k, N, seed), never on what ran
before it in the same process. That claim is cheap to state and easy to get
wrong - one stray `np.random.choice` would break it silently, and the damage
would look like a plausible result rather than an error.

So it is checked rather than asserted. Point this at a baseline copy of
`results/grid` taken before the merge; it compares every cell's CSVs against
the freshly written ones, value by value.

Run:  python scripts/verify_grid_merge.py <baseline_dir>

`<baseline_dir>` is a directory holding the pre-merge `grid/` folder.

It compares *whatever* two sets of grid outputs you point it at, so it stays
useful beyond the merge it was written for - re-run it after any refactor that
is supposed to leave the numbers alone. It is not a regression test against
the current results: those legitimately changed when each market's grid moved
onto the same statements as its main study.

Exit status is 0 only if every file matches.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
CURRENT = ROOT / "results" / "grid"
TOL = 1e-9          # bit-identical in practice; a hair of slack for float I/O


def compare(old: pathlib.Path, new: pathlib.Path) -> tuple[bool, str]:
    a = pd.read_csv(old, header=None, dtype=str).fillna("")
    b = pd.read_csv(new, header=None, dtype=str).fillna("")
    if a.shape != b.shape:
        return False, f"shape {a.shape} -> {b.shape}"
    same = a.values == b.values
    if same.all():
        return True, "identical"
    # Differences that are only float formatting are not differences.
    bad = []
    for i, j in zip(*np.where(~same)):
        x, y = a.values[i, j], b.values[i, j]
        try:
            if abs(float(x) - float(y)) <= TOL * max(1.0, abs(float(x))):
                continue
        except ValueError:
            pass
        bad.append(f"r{i}c{j}: {x!r} != {y!r}")
    if not bad:
        return True, "identical (float formatting only)"
    return False, "; ".join(bad[:4]) + (f" (+{len(bad)-4} more)" if len(bad) > 4 else "")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    base = pathlib.Path(sys.argv[1])
    old_grid = base / "grid" if (base / "grid").is_dir() else base
    if not old_grid.is_dir():
        print(f"[STOP] no baseline directory at {old_grid}")
        sys.exit(2)

    olds = sorted(p for p in old_grid.glob("*.csv"))
    if not olds:
        print(f"[STOP] baseline holds no CSVs: {old_grid}")
        sys.exit(2)

    ok, failed, missing = 0, [], []
    for o in olds:
        n = CURRENT / o.name
        if not n.exists():
            missing.append(o.name)
            continue
        good, why = compare(o, n)
        if good:
            ok += 1
        else:
            failed.append((o.name, why))

    new_only = sorted({p.name for p in CURRENT.glob("*.csv")}
                      - {p.name for p in olds})

    print(f"baseline : {old_grid}")
    print(f"current  : {CURRENT}")
    print(f"\nmatched  : {ok}/{len(olds)}")
    if missing:
        print(f"MISSING  : {len(missing)} file(s) the merge did not reproduce")
        for m in missing[:10]:
            print(f"   {m}")
    if failed:
        print(f"CHANGED  : {len(failed)} file(s)")
        for name, why in failed[:10]:
            print(f"   {name}: {why}")
    if new_only:
        print(f"new      : {len(new_only)} file(s) the merge adds "
              f"(not a regression)")
        for m in new_only[:10]:
            print(f"   {m}")

    if failed or missing:
        print("\nRESULT: the merge changed results - do not ship it")
        sys.exit(1)
    print("\nRESULT: every pre-merge number reproduced exactly")


if __name__ == "__main__":
    main()
