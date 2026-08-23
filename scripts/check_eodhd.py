"""Can EODHD supply what Japan is missing? Three questions, measured.

The Japanese main study is stuck at two formations because `FS_clean` has no
equity line, so the high-B/M universe can only start where the Yahoo cache
does (FY2021). EODHD's documentation says it carries `totalStockholderEquity`
and a cash-flow statement for non-US symbols from 2000. This checks whether
that holds for the names we actually use, before anyone pays for a year:

  1. **Which suffix resolves Japanese tickers** - the docs use `.TSE`, our
     panel is in Yahoo's `.T` form, and several variants exist.
  2. **How deep the history really is** for our own Nikkei names, not for the
     large-cap US examples the documentation is written around.
  3. **Whether the equity-issuance line is populated.** This is the one that
     decides whether Japan can stop using the share-count substitute for
     EQ_OFFER: on Apple the field `issuanceOfCapitalStock` came back null,
     so it cannot be taken on trust for Japanese filers either.

Each suffix is tried against both the fundamentals and the price endpoint,
because the two failure modes need different fixes: if prices resolve and
fundamentals do not, the symbol is right and the subscription is the problem.

Run it on your own machine; the key stays in the environment and is never
printed or written to disk:

    $env:EODHD_API_KEY = "...your key..."
    python scripts\\check_eodhd.py            # 5 tickers
    python scripts\\check_eodhd.py 20         # wider sample

Fundamentals requests are metered (EODHD counts one fundamentals call as
several ordinary ones), so the sample is small by default, every request is
counted, and the run stops early if the suffix probe fails.

Always writes results/eodhd_check.json, including on failure - the errors are
the diagnostic.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import pandas as pd
import requests

BASE = "https://eodhd.com/api"
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "eodhd_check.json"
TIMEOUT = 40

# Fields the study needs, by statement
NEEDED = {
    "Balance_Sheet": ["totalAssets", "totalCurrentAssets", "totalCurrentLiabilities",
                      "longTermDebt", "totalStockholderEquity",
                      "commonStockSharesOutstanding"],
    "Cash_Flow": ["totalCashFromOperatingActivities", "netIncome"],
    "Income_Statement": ["totalRevenue", "costOfRevenue"],
}
# The EQ_OFFER question: gross issuance first, netted figures only as fallback
ISSUANCE = ["issuanceOfCapitalStock", "salePurchaseOfStock", "netBorrowings"]
SUFFIXES = [".TSE", ".T", ".TYO", ".JP"]

calls = 0
report: dict = {}


def save() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    report["api_calls"] = calls
    OUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


def get(path: str, key: str, **params):
    """Return (payload, error). Network faults become errors, not tracebacks,
    so a failed probe still leaves a diagnosable file behind."""
    global calls
    calls += 1
    params.update({"api_token": key, "fmt": "json"})
    try:
        r = requests.get(f"{BASE}/{path}", params=params, timeout=TIMEOUT)
    except requests.RequestException as exc:
        return None, f"{type(exc).__name__}: {str(exc)[:120]}"
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}: {r.text[:160]}"
    try:
        return r.json(), ""
    except ValueError:
        return None, f"not JSON: {r.text[:160]}"


def key() -> str:
    k = os.environ.get("EODHD_API_KEY")
    if not k:
        print('\n[STOP] set EODHD_API_KEY first:  $env:EODHD_API_KEY = "..."')
        sys.exit(1)
    return k


def panel_tickers(n: int) -> list[str]:
    """Our own Japanese names, four-digit codes without the Yahoo suffix."""
    p = ROOT / "results" / "panel" / "japan_fscore_panel.csv"
    if p.exists():
        codes = (pd.read_csv(p).ticker.astype(str)
                 .str.replace(".T", "", regex=False).unique().tolist())
    else:
        codes = ["7203", "6758", "9432", "8306", "4502"]
    return codes[:n]


def fill(store: dict, fields: list[str]) -> dict:
    """Share of annual periods in which each field is actually populated."""
    if not store:
        return {f: 0.0 for f in fields}
    return {f: sum(1 for y in store.values()
                   if y.get(f) not in (None, "", "0", 0)) / len(store)
            for f in fields}


def probe_suffix(k: str) -> str | None:
    print("resolving the Japanese ticker suffix (7203 = Toyota):")
    suffix, attempts = None, []
    for suf in SUFFIXES:
        d, err = get(f"fundamentals/7203{suf}", k)
        ok = isinstance(d, dict) and "Financials" in d
        px, perr = get(f"eod/7203{suf}", k, **{"from": "2024-01-04",
                                               "to": "2024-01-31"})
        px_ok = isinstance(px, list) and len(px) > 0
        attempts.append({"suffix": suf, "fundamentals_ok": ok,
                         "fundamentals_error": err,
                         "prices_ok": px_ok, "prices_error": perr,
                         "top_keys": list(d)[:8] if isinstance(d, dict) else None})
        print(f"  7203{suf:5s} fundamentals : "
              f"{'OK' if ok else (err or 'no Financials key')[:80]}")
        print(f"  {'':10s} prices       : "
              f"{f'OK ({len(px)} rows)' if px_ok else (perr or 'empty')[:80]}")
        if ok and suffix is None:
            suffix = suf
        time.sleep(0.3)
    report["suffix"] = suffix
    report["attempts"] = attempts

    if suffix is None:
        any_px = any(a["prices_ok"] for a in attempts)
        print("\n[STOP] no suffix returned fundamentals.")
        if any_px:
            good = [a["suffix"] for a in attempts if a["prices_ok"]]
            print(f"  prices DID resolve for {good} - the symbol form is fine,")
            print("  so this is a subscription entitlement, not a mapping bug.")
            report["diagnosis"] = "symbol ok, fundamentals not entitled"
        else:
            print("  prices failed too - the key is limited, or Tokyo is not")
            print("  covered by this plan. Check the account line above.")
            report["diagnosis"] = "neither prices nor fundamentals available"
        save()
        print(f"  details -> {OUT}")
    return suffix


def main() -> None:
    k = key()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    # ---- 0. does the key work, and what does the plan cover? --------
    who, err = get("user", k)
    if isinstance(who, dict):
        plan = {kk: who.get(kk) for kk in
                ("name", "subscriptionType", "apiRequests", "apiRequestsDate",
                 "dailyRateLimit", "extraLimit", "inviteToken")
                if kk != "inviteToken"}
        print(f"account : {plan}\n")
        report["account"] = plan
    else:
        print(f"account lookup failed: {err}\n")
        report["account_error"] = err

    # ---- 1. which suffix resolves? ----------------------------------
    suffix = probe_suffix(k)
    if suffix is None:
        sys.exit(1)
    print(f"  -> using {suffix}\n")

    # ---- 2/3. depth, field fill, issuance ---------------------------
    rows = []
    for code in panel_tickers(n):
        d, err = get(f"fundamentals/{code}{suffix}", k)
        if not isinstance(d, dict) or "Financials" not in d:
            print(f"  {code}: {err or 'no data'}")
            rows.append({"code": code, "ok": False, "error": err})
            continue
        fin = d["Financials"]
        bs = fin.get("Balance_Sheet", {}).get("yearly", {})
        cf = fin.get("Cash_Flow", {}).get("yearly", {})
        inc = fin.get("Income_Statement", {}).get("yearly", {})
        years = sorted(bs)
        fb = fill(bs, NEEDED["Balance_Sheet"])
        fc = fill(cf, NEEDED["Cash_Flow"])
        fi = fill(inc, NEEDED["Income_Statement"])
        iss = fill(cf, ISSUANCE)
        filed = sum(1 for y in bs.values() if y.get("filing_date")) / max(len(bs), 1)
        rows.append({
            "code": code, "ok": True, "annual_periods": len(bs),
            "earliest": years[0] if years else None,
            "latest": years[-1] if years else None,
            "equity_fill": round(fb["totalStockholderEquity"], 2),
            "assets_fill": round(fb["totalAssets"], 2),
            "cfo_fill": round(fc["totalCashFromOperatingActivities"], 2),
            "cogs_fill": round(fi["costOfRevenue"], 2),
            "shares_fill": round(fb["commonStockSharesOutstanding"], 2),
            "filing_date_fill": round(filed, 2),
            **{f"iss_{f}": round(iss[f], 2) for f in ISSUANCE},
        })
        r = rows[-1]
        print(f"  {code}{suffix}: {r['annual_periods']:2d} annual periods "
              f"{r['earliest']} -> {r['latest']} | equity {r['equity_fill']:.0%}"
              f" | issuance {r['iss_issuanceOfCapitalStock']:.0%}")
        report["per_ticker"] = rows
        save()
        time.sleep(0.4)

    df = pd.DataFrame(rows)
    good = df[df.ok] if "ok" in df.columns else df

    print("\n" + "=" * 64)
    if len(good):
        reach = sum(1 for r in rows
                    if r.get("earliest") and str(r["earliest"])[:4] <= "2011")
        print(f"tickers resolved        : {len(good)}/{len(rows)}")
        print(f"median annual periods   : {good.annual_periods.median():.0f}")
        print(f"reach FY2011 or earlier : {reach}/{len(rows)}   "
              f"(needed for a 2012 formation)")
        print(f"book value populated    : {good.equity_fill.mean():.0%}   "
              f"<- the blocker for Japan's high-B/M universe")
        print(f"true filing_date        : {good.filing_date_fill.mean():.0%}   "
              f"<- would replace the 3-month lag assumption")
        print("equity-issuance line    :")
        for f in ISSUANCE:
            print(f"    {f:26s} {good[f'iss_{f}'].mean():.0%}")
        report["verdict"] = {
            "resolved": f"{len(good)}/{len(rows)}",
            "median_periods": float(good.annual_periods.median()),
            "reach_2011": int(reach),
            "equity_fill": float(good.equity_fill.mean()),
            "filing_date_fill": float(good.filing_date_fill.mean()),
            "issuance_fill": float(good.iss_issuanceOfCapitalStock.mean()),
        }
    else:
        print("no ticker resolved - see per_ticker errors in the JSON")
    print(f"API calls used          : {calls}")
    print("=" * 64)

    save()
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # keep the diagnostic even on an unexpected fault
        report["crash"] = f"{type(exc).__name__}: {exc}"
        save()
        raise
