#!/usr/bin/env python3
"""
Appendix B table builder (SOC toy sim artifacts)

Inputs:
  - data/module1/fold_runs/                 (drop)
  - data/module1/truncated_fold_runs/       (reset)

Outputs (per your preference that paper/ is correct):
  - paper/tables/

REMARK (outputs):
Writes 4 files into paper/tables/:
  1) appendixB_drop_summary.csv   (one row per run file; header metrics)
  2) appendixB_reset_summary.csv
  3) appendixB_drop_table.tex     (LaTeX table wrapper)
  4) appendixB_reset_table.tex
Overwrites existing files with the same names.
"""

from __future__ import annotations
from pathlib import Path
import csv
import re
import sys
from typing import Dict, List, Tuple, Optional

DROP_SUMMARY_CSV  = "appendixB_drop_summary.csv"
RESET_SUMMARY_CSV = "appendixB_reset_summary.csv"
DROP_TABLE_TEX    = "appendixB_drop_table.tex"
RESET_TABLE_TEX   = "appendixB_reset_table.tex"

# filename pattern you’re using
FNAME_RE = re.compile(
    r".*?_N(?P<N>\d+)_a(?P<a>[0-9.]+)_k(?P<k>[0-9.]+)_j(?P<j>[0-9.]+)_seed(?P<seed>\d+)_(?P<date>\d{8})_(?P<time>\d{6})\.csv$"
)

# Output columns (omit C..U block)
OUTPUT_COLS = [
    "mode",
    "N",
    "events",
    "event_fraction",
    "max_avalanche",
    "mean_avalanche_nonzero",
    "total_topples"
]

# --- formatting rules for output table ---
FORMAT_SPECS = {
    "event_fraction": ("float", 3),
    "mean_avalanche_nonzero": ("float", 1),
}

def _fmt_value(key: str, val: str) -> str:
    """Format selected numeric fields as strings for nicer tables."""
    if val is None:
        return ""
    s = str(val).strip()
    if s == "":
        return ""

    spec = FORMAT_SPECS.get(key)
    if not spec:
        return s

    kind, ndp = spec
    if kind == "float":
        try:
            x = float(s)
            return f"{x:.{ndp}f}"
        except ValueError:
            return s  # leave untouched if it isn't parseable
    return s


def die(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    raise SystemExit(code)

def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for d in [start] + list(start.parents):
        if (d / "data" / "module1").exists():
            return d
        if (d / ".git").exists():
            return d
    die(f"Could not find repo root from: {start}")
    return start

def out_dir(repo_root: Path) -> Path:
    d = repo_root / "paper" / "tables"
    d.mkdir(parents=True, exist_ok=True)
    return d

def escape_latex(s: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in str(s))

def parse_filename_params(filename: str) -> Dict[str, str]:
    m = FNAME_RE.match(filename)
    return m.groupdict() if m else {}

def read_header_kv(csv_path: Path) -> Dict[str, str]:
    """
    Reads the leading '# ...' block formatted as CSV rows:
      # key,value
    Stops at '# data' line (inclusive).
    """
    kv: Dict[str, str] = {}

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for raw in f:
            line = raw.strip()
            if not line.startswith("#"):
                # header ended unexpectedly (still fine)
                break

            # strip '#'
            content = line.lstrip("#").strip()

            # stop marker
            if content.lower() == "data":
                break

            # Try "key: value" first
            if ":" in content:
                k, v = content.split(":", 1)
                kv[k.strip()] = v.strip()
                continue

            # Otherwise treat as CSV-ish "key,value"
            # (handles "# events,63964" exactly like your file)
            parts = [p.strip() for p in content.split(",")]
            if len(parts) >= 2 and parts[0]:
                kv[parts[0]] = parts[1]
            else:
                # single token like "# summary" — store as flag
                if parts and parts[0]:
                    kv[parts[0]] = ""

    return kv

def summarize_file(p: Path, mode_label: str) -> Dict[str, str]:
    row: Dict[str, str] = {"mode": mode_label, "source_file": p.name}

    # filename params
    row.update(parse_filename_params(p.name))

    # header kv (THIS is where your meaningful stats live)
    hdr = read_header_kv(p)

    # normalize some keys if you ever vary capitalization
    # (we’ll keep original too, but these are the ones we use)
    for k, v in hdr.items():
        row[k] = v

    return row

def summarize_folder(folder: Path, mode_label: str) -> List[Dict[str, str]]:
    if not folder.exists():
        die(f"Missing input folder: {folder}")
    files = sorted(folder.glob("*.csv"))
    if not files:
        die(f"No CSV inputs found in: {folder}")
    return [summarize_file(p, mode_label) for p in files]

def column_order(rows: List[Dict[str, str]]) -> List[str]:
    keys = set()
    for r in rows:
        keys.update(r.keys())

    # Only keep what user wants (drop the whole middle block)
    cols = [c for c in OUTPUT_COLS if c in keys]
    return cols


def write_csv(rows: List[Dict[str, str]], out_csv: Path) -> List[str]:
    cols = column_order(rows)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            rr = {k: _fmt_value(k, r.get(k, "")) for k in cols}
            w.writerow(rr)
    return cols


def infer_col_spec(cols: List[str]) -> str:
    numericish = {
        "n","a","k","j","seed","steps","warmup","drive_amount",
        "kappa","kappa_jitter","events","event_fraction","max_avalanche",
        "mean_avalanche_nonzero","total_topples","max_queue","runtime_sec"
    }
    spec = []
    for c in cols:
        spec.append("r" if c.lower() in numericish else "l")
    return "".join(spec)

def write_latex_table(rows: List[Dict[str, str]], cols: List[str], out_tex: Path,
                      caption: str, label: str) -> None:
    col_spec = infer_col_spec(cols)
    lines: List[str] = []
    lines += [
        r"\begin{table}[h]",
        r"\centering",
        r"\scriptsize",
        fr"\caption{{{escape_latex(caption)}}}",
        fr"\label{{{escape_latex(label)}}}",
        r"\begin{tabular}{" + col_spec + r"}",
        r"\toprule",
        " & ".join(escape_latex(c) for c in cols) + r" \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(" & ".join(escape_latex(r.get(c, "")) for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    out_tex.write_text("\n".join(lines), encoding="utf-8")

def main() -> int:
    repo_root = find_repo_root(Path(__file__).resolve().parent)

    drop_dir  = repo_root / "data" / "module1" / "fold_runs"
    reset_dir = repo_root / "data" / "module1" / "truncated_fold_runs"
    out = out_dir(repo_root)

    drop_rows  = summarize_folder(drop_dir, "drop")
    reset_rows = summarize_folder(reset_dir, "reset")

    drop_rows.sort(key=lambda r: int(r.get("N", 10**18)))
    reset_rows.sort(key=lambda r: int(r.get("N", 10**18)))


    drop_cols = write_csv(drop_rows, out / DROP_SUMMARY_CSV)
    reset_cols = write_csv(reset_rows, out / RESET_SUMMARY_CSV)

    write_latex_table(drop_rows, drop_cols, out / DROP_TABLE_TEX,
                      "Appendix B results summary (mode=drop).", "tab:appB-drop")
    write_latex_table(reset_rows, reset_cols, out / RESET_TABLE_TEX,
                      "Appendix B results summary (mode=reset).", "tab:appB-reset")

    print("Wrote to:", out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
