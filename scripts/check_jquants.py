"""Probe what the J-Quants FREE tier gives us under the **v2** API.

v2 replaced v1's token flow with a single API key:

    "V1 APIのトークン認証方式はV2 APIでは廃止されています。
     V2 APIではダッシュボードから発行するAPIキーをご利用ください。"

so a v1-style mailaddress/password login no longer applies, and a Google-
linked account is fine — issue a key from the dashboard instead.

Run it on your own machine; the key stays in the environment and is never
printed or written to disk:

    $env:JQUANTS_API_KEY = "...key from the dashboard..."
    python scripts\\check_jquants.py

Endpoint paths come from the official quick-start notebook
(github.com/J-Quants/jquants-api-quick-start) — the ones that matter here are
`/equities/master`, `/equities/bars/daily`, `/fins/summary` and
`/fins/details` (BS/PL/CF). Nothing about the free tier is assumed — reach,
history depth, delisting coverage and statement access are all measured:

  1. which endpoints the key can reach at all;
  2. the listed-issue master: size, fields, sector labels;
  3. how far back daily bars actually go, found by walking dates backwards;
  4. **delisted names** — whether the master can be queried as of a past
     date, and whether a code that has since delisted still returns prices
     (a master that merely lists them is useless for a backtest);
  5. overlap with the Nikkei panel already in data/;
  6. whether BS/PL/CF statements are included in the free tier.

Writes results/jquants_v2_check.json.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import requests

BASE = "https://api.jquants.com/v2"
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "jquants_v2_check.json"
TIMEOUT = 30

# Real v2 paths, taken from the official quick-start notebook
# (github.com/J-Quants/jquants-api-quick-start).
CANDIDATES = {
    "daily_bars": ["/equities/bars/daily"],
    "listed_master": ["/equities/master"],
    "fins_summary": ["/fins/summary"],        # headline financials
    "fins_details": ["/fins/details"],        # full BS/PL/CF — what we need
    "dividends": ["/fins/dividend"],
    "calendar": ["/markets/calendar"],
}
PROBE_CODE = "72030"     # Toyota, 5-digit v1/v2 style (4-digit code + 0)
PROBE_CODE_ALT = "7203"


def die(msg: str) -> None:
    print(f"\n[STOP] {msg}")
    sys.exit(1)


def key() -> str:
    k = os.environ.get("JQUANTS_API_KEY")
    if not k:
        die('set JQUANTS_API_KEY first:  $env:JQUANTS_API_KEY = "..."')
    return k


def call(path: str, api_key: str, **params):
    """GET one page. Returns (status, payload_rows, raw_json_or_text)."""
    try:
        r = requests.get(BASE + path, headers={"x-api-key": api_key},
                         params=params, timeout=TIMEOUT)
    except Exception as e:
        return 0, [], f"{type(e).__name__}: {e}"
    if r.status_code != 200:
        return r.status_code, [], r.text[:200]
    try:
        js = r.json()
    except ValueError:
        return r.status_code, [], r.text[:200]
    rows = next((v for v in js.values() if isinstance(v, list)), [])
    return 200, rows, js


def paged(path: str, api_key: str, **params):
    """Follow pagination_key through to the end (capped)."""
    rows, pk, guard = [], None, 0
    while True:
        p = dict(params)
        if pk:
            p["pagination_key"] = pk
        st, got, js = call(path, api_key, **p)
        if st != 200:
            return rows, st, js
        rows += got
        pk = js.get("pagination_key") if isinstance(js, dict) else None
        guard += 1
        if not pk or guard > 40:
            return rows, 200, ""
        time.sleep(0.2)


def main() -> None:
    api_key = key()
    report: dict = {"base": BASE}

    # ---- 1. which endpoints does this key reach? --------------------
    print("discovering v2 endpoints (probing candidates):\n")
    resolved: dict[str, str] = {}
    for label, paths in CANDIDATES.items():
        for p in paths:
            probe = ({"code": PROBE_CODE, "date": "20240701"}
                     if "bars" in p else {})
            st, rows, _ = call(p, api_key, **probe)
            mark = {200: "OK ", 403: "403", 404: "404"}.get(st, str(st))
            print(f"  [{mark}] {p}" + (f"  ({len(rows)} rows)" if st == 200 else ""))
            if st == 200 and label not in resolved:
                resolved[label] = p
            time.sleep(0.25)
    print()
    report["endpoints"] = resolved
    if not resolved:
        die("no candidate endpoint answered 200 — check the key, or open the "
            "dashboard's API reference and send me the exact v2 paths")

    # ---- 2. listed master -------------------------------------------
    master_path = resolved.get("listed_master")
    latest_codes: set[str] = set()
    if master_path:
        rows, st, _ = paged(master_path, api_key)
        latest_codes = {str(r.get("Code") or r.get("code")) for r in rows}
        print(f"listed master: {len(rows)} rows via {master_path}")
        if rows:
            print(f"  fields: {', '.join(sorted(rows[0])[:14])}")
            s = rows[0]
            print(f"  sample: {s}")
        report["listed_master"] = {"path": master_path, "n": len(rows),
                                   "fields": sorted(rows[0]) if rows else []}
    else:
        print("listed master: NOT reachable on the free tier")
        report["listed_master"] = None

    # ---- 3. how far back do daily bars go? --------------------------
    bars = resolved.get("daily_bars")
    depth = None
    if bars:
        print("\nprobing price history depth:")
        for year in range(2025, 2009, -1):
            st, rows, _ = call(bars, api_key, code=PROBE_CODE, date=f"{year}0701")
            if st == 200 and not rows:
                st2, rows2, _ = call(bars, api_key, code=PROBE_CODE_ALT,
                                     date=f"{year}0701")
                rows = rows2 or rows
            ok = st == 200 and len(rows) > 0
            print(f"  {year}-07-01: {'OK' if ok else 'empty'} (http {st})")
            if ok:
                depth = year
            elif depth:
                break
            time.sleep(0.25)
        report["price_history_back_to"] = depth
        print(f"  -> reachable back to {depth}")

    # ---- 4. delisted names ------------------------------------------
    print("\ndelisting check:")
    delist: dict = {}
    if master_path:
        # can the master be asked "as of" a past date? that is the only way
        # to recover names that have since delisted
        # the docs state the master can be retrieved "as of the past" — the
        # question is how far back the tier allows, so walk backwards
        dated_ok, as_of, old = False, None, set()
        for y in range(2025, 2011, -1):
            st, rows, _ = call(master_path, api_key, date=f"{y}0701")
            ok = st == 200 and len(rows) > 0
            print(f"  master as of {y}-07-01: {'OK (' + str(len(rows)) + ' rows)' if ok else 'empty'} (http {st})")
            if ok:
                dated_ok, as_of = True, f"{y}0701"
                old = {str(r.get("Code") or r.get("code")) for r in rows}
            elif dated_ok:
                break
            time.sleep(0.25)
        delist["dated_master"] = dated_ok
        delist["oldest_master"] = as_of
        if dated_ok:
            gone = sorted(old - latest_codes)
            print(f"  codes listed then, absent now: {len(gone)}")
            delist["n_delisted"] = len(gone)
            for code in gone[:5]:
                st, px, _ = (call(bars, api_key, code=code, date=as_of)
                             if bars else (0, [], ""))
                print(f"    {code}: prices on a past date -> {len(px)} rows (http {st})")
                delist.setdefault("samples", []).append(
                    {"code": code, "rows": len(px), "http": st})
                time.sleep(0.25)
            if delist.get("samples") and all(s["rows"] == 0 for s in delist["samples"]):
                print("    !! delisted codes carry NO price history —"
                      " survivorship bias cannot be fixed with this tier")
    report["delisting"] = delist

    # ---- 5. overlap with our panel ----------------------------------
    sec = ROOT / "data" / "japan_sectors.csv"
    if sec.exists() and latest_codes:
        import csv
        with open(sec, encoding="utf-8") as fh:
            ours = {r["ticker"].split(".")[0] for r in csv.DictReader(fh)}
        jq4 = {c[:4] for c in latest_codes}
        hit = len(ours & jq4)
        print(f"\npanel overlap: {hit}/{len(ours)} Nikkei codes present")
        report["panel_overlap"] = {"ours": len(ours), "matched": hit}

    # ---- 6. statements ----------------------------------------------
    st_path = resolved.get("fins_details") or resolved.get("fins_summary")
    print("\nfinancial statements (BS/PL/CF):")
    if st_path:
        rows, st, _ = paged(st_path, api_key, code=PROBE_CODE)
        if not rows:
            rows, st, _ = paged(st_path, api_key, code=PROBE_CODE_ALT)
        print(f"  {st_path}: {len(rows)} rows (http {st})")
        if rows:
            yrs = sorted({str(r.get("DisclosedDate", ""))[:4] for r in rows})
            print(f"  disclosed years: {yrs}")
            eq = [f for f in rows[0]
                  if any(w in f.lower() for w in ("issu", "stock", "capital",
                                                  "equity", "share"))]
            print(f"  {len(rows[0])} fields; equity-issuance candidates: {eq[:8]}")
            report["statements"] = {"path": st_path, "n": len(rows), "years": yrs}
    else:
        print("  NOT reachable on the free tier")
        report["statements"] = None

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")

    print("\n" + "=" * 64)
    print(f"listed master        : {'YES' if report.get('listed_master') else 'NO'}")
    print(f"price history to     : {report.get('price_history_back_to')}")
    print(f"delisted recoverable : "
          f"{'YES' if any(s['rows'] for s in delist.get('samples', [])) else 'NO'}")
    print(f"statements (BS/PL/CF): {'YES' if report.get('statements') else 'NO'}")
    print("=" * 64)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
