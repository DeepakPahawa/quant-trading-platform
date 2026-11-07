"""Stock universes for swing research: Nifty 50 and Nifty 500.

Constituent lists come from NSE Indices (niftyindices.com) as CSV files. Put them at:
    backend/data/universe/nifty50.csv
    backend/data/universe/nifty500.csv

Accepted columns: `Symbol` (NSE format, e.g. RELIANCE) or `symbol` (FYERS format,
e.g. NSE:RELIANCE-EQ). Both are converted to FYERS format.
"""

import csv
from pathlib import Path
from typing import List

UNIVERSE_DIR = Path("backend/data/universe")
BENCHMARK_SYMBOL = "NSE:NIFTY50-INDEX"
KNOWN_UNIVERSES = ("nifty50", "nifty500")


def nse_to_fyers(symbol: str) -> str:
    """'RELIANCE' -> 'NSE:RELIANCE-EQ'. Values already in FYERS format pass through."""
    s = symbol.strip().upper()
    if not s:
        raise ValueError("Empty symbol")
    if ":" in s:
        return s
    return f"NSE:{s}-EQ"


def load_universe(name: str, base_dir: Path = UNIVERSE_DIR) -> List[str]:
    """Load a universe by name ('nifty50' or 'nifty500') as sorted FYERS symbols."""
    # Any `<name>.csv` in base_dir is a valid universe, so larger or custom sets can be tested later.
    if not name or not name.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"Invalid universe name {name!r}. Use letters, digits, '_' or '-' (e.g. {KNOWN_UNIVERSES}).")
    path = base_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the constituent list from niftyindices.com and save it there."
        )

    symbols: List[str] = []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        col = next((c for c in (reader.fieldnames or []) if c.strip().lower() == "symbol"), None)
        if col is None:
            raise ValueError(f"{path} has no 'Symbol' column (found {reader.fieldnames})")
        for row in reader:
            raw = (row.get(col) or "").strip()
            if raw:
                symbols.append(nse_to_fyers(raw))

    unique = sorted(set(symbols))
    if not unique:
        raise ValueError(f"{path} contained no symbols")
    return unique
