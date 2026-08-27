"""Where the Vietnamese preprocessing pipeline reads and writes.

One module resolves every location, so a notebook never spells out a relative
path and moving the tree again means editing this file and nothing else. That
is the lesson of the move that created it: the pipeline used to live in a
sibling repository and address everything as `../fscore.db`,
`../data/preprocessing_pipeline_results/...`, which meant the paths were only
correct from one working directory in one checkout.

Layout under the repository root:

    data/vietnam_pipeline/            DATA    — crawled universe files
    data/vietnam_pipeline/results/    RESULTS — every panel the pipeline writes

`fscore.db` is the exception. It is ~1.7 GB of crawled statements, is not
version-controlled, and is not copied around: point `FSCORE_DB` at wherever
your copy lives. The default is the sibling `thesis` checkout, which is where
it was crawled — so an existing working tree keeps running untouched, and a
machine that has moved the file only has to export one variable.

Every path is overridable:

    FSCORE_DB        absolute path to fscore.db
    FSCORE_VN_DATA   the DATA directory
    FSCORE_VN_RESULTS  the RESULTS directory
"""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]              # src/fscore_vietnam -> src -> repo

DATA = Path(os.environ.get("FSCORE_VN_DATA",
                           REPO_ROOT / "data" / "vietnam_pipeline"))
RESULTS = Path(os.environ.get("FSCORE_VN_RESULTS", DATA / "results"))

# Not in the repository: see the module docstring.
_DEFAULT_DB = REPO_ROOT.parent / "thesis" / "fscore.db"
DB = Path(os.environ.get("FSCORE_DB", _DEFAULT_DB))


def require_db() -> Path:
    """`DB`, or a message saying exactly how to point at it.

    The pipeline's first four notebooks read the crawl; failing here with the
    remedy beats failing three cells later inside sqlite3 with `unable to
    open database file`.
    """
    if not DB.exists():
        raise FileNotFoundError(
            f"fscore.db not found at {DB}.\n"
            "It is ~1.7 GB of crawled statements and is not kept in the "
            "repository. Point at your copy with:\n"
            "    export FSCORE_DB=/path/to/fscore.db\n"
            "Notebooks downstream of `fields_extract` do not need it — they "
            f"read the panels in {RESULTS}."
        )
    return DB


def ensure_dirs() -> None:
    """Create the output tree. Safe to call from every notebook."""
    DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
