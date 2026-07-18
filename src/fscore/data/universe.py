"""Constituent lists and sector maps for the developed pair (US, Japan).

Universe *membership* comes from index constituent lists (S&P 500, Nikkei 225);
the point-in-time eligible universe (~100-150 names per formation date) is then
selected downstream on liquidity and data availability — see
`fscore.pipeline.build_universe`.

Sector labels feed the sector-constrained GMV (20% cap):
  - US:    GICS Sector from the S&P 500 table
  - Japan: Nikkei industry classification (section headings)

Caveat (documented limitation): these are *current* constituents, so the
universe inherits index-inclusion survivorship. True point-in-time membership
and delisted names require a commercial source (see data/README.md gating
checks); the pipeline is written against canonical frames so that source can
be swapped in without touching downstream code.
"""
from __future__ import annotations

import io
import re

import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (capstone research; mike.ding.work@gmail.com)"}

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NIKKEI_URL = "https://en.wikipedia.org/wiki/Nikkei_225"


def sp500_constituents() -> pd.DataFrame:
    """Current S&P 500 members: [ticker, name, sector] (tickers in Yahoo form)."""
    html = requests.get(SP500_URL, headers=UA, timeout=30).text
    table = pd.read_html(io.StringIO(html))[0]
    out = pd.DataFrame({
        "ticker": table["Symbol"].str.replace(".", "-", regex=False),  # BRK.B -> BRK-B
        "name": table["Security"],
        "sector": table["GICS Sector"],
    })
    return out.drop_duplicates("ticker").reset_index(drop=True)


def nikkei225_constituents() -> pd.DataFrame:
    """Current Nikkei 225 members: [ticker, name, sector] ('7203.T' form).

    Parsed from the Components section: industry sub-headings (h3) followed by
    company lists whose TSE codes appear in jpx.co.jp search links.
    """
    html = requests.get(NIKKEI_URL, headers=UA, timeout=30).text
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    rows: list[dict] = []
    sector = None
    for el in soup.find_all(["h2", "h3", "li"]):
        if el.name in ("h2", "h3"):
            sector = el.get_text(strip=True)
            continue
        text = el.get_text(" ", strip=True)
        m = re.search(r"topSearchStr=(\d{4})", str(el))
        if m and sector and "TYO" in text:
            name = re.split(r"\s*\(", text)[0].strip()
            rows.append({"ticker": f"{m.group(1)}.T", "name": name, "sector": sector})
    out = pd.DataFrame(rows).drop_duplicates("ticker").reset_index(drop=True)
    if len(out) < 200:
        raise RuntimeError(f"Nikkei parse looks wrong: only {len(out)} names")
    return out


def constituents(market: str) -> pd.DataFrame:
    market = market.lower()
    if market in ("us", "united states"):
        return sp500_constituents()
    if market in ("japan", "jp"):
        return nikkei225_constituents()
    raise NotImplementedError(f"no constituent source wired for {market!r}")
