"""
Audit and sanitise notebook outputs before committing to GitHub.

Scans the saved .ipynb for outputs that contain supplied-dataset
content, then optionally clears the outputs of ONLY those cells.

Run this LOCALLY. It reads the private dataset to build its needle
list and never writes any sensitive value anywhere.

Usage:
    python audit_notebook.py                 # report only
    python audit_notebook.py --clean         # clear flagged cells
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pipeline as P

NOTEBOOK = Path("Message_Classification.ipynb")
MESSAGES = Path("data/messages.csv")


def build_needles():
    """Raw dataset strings that must not appear in committed output."""
    df = pd.read_csv(MESSAGES, dtype=str, keep_default_na=False)

    sensitive_values = P.harvest_sensitive_values(df["message"])

    # Full raw messages, and their prefix-stripped cores.
    raw = set()
    for m in df["message"]:
        m = m.strip()
        if len(m) >= 25:
            raw.add(m)
            core, _ = P.strip_noise_prefixes(m)
            if len(core) >= 25:
                raw.add(core)

    # Cores of sensitive messages get their own bucket: these are the
    # most serious to leak.
    sens_cores = set()
    for m in df.loc[df["message"].map(
            lambda x: P.detect_sensitive(x)["is_sensitive"]), "message"]:
        core, _ = P.strip_noise_prefixes(m.strip())
        sens_cores.add(core)
        sens_cores.add(m.strip())

    return sensitive_values, raw, sens_cores


def cell_output_text(cell):
    """Concatenate every piece of rendered output from one cell."""
    parts = []
    for out in cell.get("outputs", []):
        if "text" in out:
            parts.append("".join(out["text"]))
        data = out.get("data", {})
        for key in ("text/plain", "text/html"):
            if key in data:
                parts.append("".join(data[key]))
        if "traceback" in out:
            parts.append("".join(out["traceback"]))
    return "\n".join(parts)


def main():
    clean = "--clean" in sys.argv

    if not NOTEBOOK.exists():
        print(f"ERROR: {NOTEBOOK} not found. Run from the project root.")
        return 1
    if not MESSAGES.exists():
        print(f"ERROR: {MESSAGES} not found.")
        return 1

    sens_values, raw_msgs, sens_cores = build_needles()
    print(f"Needles built: {len(sens_values)} sensitive values, "
          f"{len(raw_msgs)} raw message strings\n")

    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = nb["cells"]

    flagged = []
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        text = cell_output_text(cell)
        if not text:
            continue

        n_sens = sum(1 for v in sens_values if v in text)
        n_sens_core = sum(1 for c in sens_cores if c in text)
        n_raw = sum(1 for m in raw_msgs if m in text)

        if n_sens or n_sens_core or n_raw:
            first_line = (cell["source"][0].strip()
                          if cell["source"] else "")[:52]
            flagged.append((idx, n_sens, n_sens_core, n_raw, first_line))

    print(f"Code cells with output : "
          f"{sum(1 for c in cells if c.get('cell_type')=='code' and c.get('outputs'))}")
    print(f"Cells flagged          : {len(flagged)}\n")

    if flagged:
        print(f"{'cell':>5}  {'sensval':>7} {'senstxt':>7} {'rawmsg':>7}   source")
        print("-" * 78)
        for idx, ns, nsc, nr, src in flagged:
            print(f"{idx:>5}  {ns:>7} {nsc:>7} {nr:>7}   {src}")
        print("\n(counts only - no values are printed)")
    else:
        print("No dataset content found in any cell output.")

    if clean and flagged:
        for idx, *_ in flagged:
            cells[idx]["outputs"] = []
            cells[idx]["execution_count"] = None
        backup = NOTEBOOK.with_suffix(".ipynb.bak")
        backup.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        NOTEBOOK.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print(f"\nCleared outputs of {len(flagged)} cell(s).")
        print(f"Backup written to {backup}")
        print("Re-run this script without --clean to verify.")
    elif clean:
        print("\nNothing to clean.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
