#!/usr/bin/env python3
"""
Make Appendix B summary tables from SOC toy sim run CSVs.

Inputs (repo-root relative):
  - data/module1/fold_runs/               (mode=drop)
  - data/module1/truncated_fold_runs/     (mode=reset)

Outputs (repo-root relative):
  - paper/tables/ (preferred) OR code/Module1/tables/ (fallback)
    - appendixB_drop_summary.csv
    - appendixB_reset_summary.csv
    - appendixB_drop_table.tex
    - appendixB_reset_table.tex

No third-party deps. Safe to run from any working directory.
"""

from __future__ import annotations
from pathlib import Path
import csv
import sys
import re
from typing import Dict, List, Tuple, Optional

# ---- config: filenames ----
DROP_SUMMARY_CSV  = "appendixB_drop_summary.csv"
RESET_SUMMARY_CSV = "appendixB_reset_summary.csv"
DROP_TABLE_TEX    = "appendixB_drop_table.tex"
RESET_TABLE_TEX   = "appendixB_reset_table.tex"

# ---- columns we'd like to show first (if present) ----
PREFERRED_COLS = [
    "mode", "N", "steps", "events", "topples", "rate",
    "run_id", "git_commit", "git_dirty", "host", "source_file"
]


def find_repo_root(start: Path) -> Path:
    """
    Walk upward from 'start' until we find a directory that looks like the repo root.
    Criteria: contains data/module1 OR contains .git
    """
    start = start.resolve()
    for d in [start] + list(start.parents):
        if (d / "data" / "module1").exists():
            return d
        if (d / ".git").exists():
            return d
    raise RuntimeError(f"Could not locate repo root starting from: {start}")


def pick_output_dir(repo_root: Path) -> Path:
    """
    Prefer paper/tables if paper/ exists; otherwise code/Module1/tables.
    """
    paper_dir = repo_root / "paper"
    if paper_dir.exists():
        return paper_dir / "tables"
    return repo_root / "code" / "Module1" / "tables"


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
    out = []
    for ch in str(s):
        out.append(repl.get(ch, ch))
    return "".join(out)


def read_provenance_and_rows(csv_path: Path) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    """
    Reads a CSV that may begin with comment provenance lines:
      # key: value
    Returns (provenance_dict, list_of_rows_dicts).
    """
    prov: Dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        # Read lines to separate header comments from CSV data
        lines = f.readlines()

    i = 0
    while i < len(lines) and lines[i].lstrip().startswith("#"):
        line = lines[i].lstrip()[1:].strip()
        if ":" in line:
            k, v = line.split(":", 1)
            prov[k.strip()] = v.strip()
        i += 1

    body = lines[i:]
    if not body:
        return prov, []

    reader = csv.DictReader(body)
    rows = list(reader)
    return prov, rows


def norm_key(k: str) -> str:
    return re.sub(r"\s+", "", k.strip().lower())


def get_last_value(rows: List[Dict[str, str]], wanted_key: str) -> Optional[str]:
    """
    Fetch a value for wanted_key from the LAST row, case-insensitively.
    Returns None if not found or empty.
    """
    if not rows:
        return None

    want = norm_key(wanted_key)
    last = rows[-1]
    for k, v in last.items():
        if norm_key(k) == want:
            if v is None or str(v).strip() == "":
                return None
            return str(v).strip()
    return None


def summarize_folder(folder: Path, mode_label: str) -> List[Dict[str, str]]:
    """
    Create one summary row per input CSV file in folder.
    """
    if not folder.exists():
        raise RuntimeError(f"Input folder not found: {folder}")

    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"No CSV files found in: {folder}")

    summaries: List[Dict[str, str]] = []

    for p in csv_files:
        prov, rows = read_provenance_and_rows(p)

        row: Dict[str, str] = {}
        row["mode"] = mode_label
        row["source_file"] = p.name

        # provenance headers you said you write
        for k in ("git_commit", "git_dirty", "run_id", "host"):
            if k in prov and prov[k] != "":
                row[k] = prov[k]

        # pull common metrics from last row if present
        for k in ("N", "steps", "events", "topples", "rate", "run_id", "host"):
            v = get_last_value(rows, k)
            if v is not None:
                row[k] = v

        # if the CSV itself is already a 1-row summary, capture its columns too
        if len(rows) == 1:
            for k, v in rows[0].items():
                if v is None:
                    continue
                vv = str(v).strip()
                if vv == "":
                    continue
                # keep original header name but normalize common ones already in row
                if k not in row:
                    row[k] = vv

        summaries.append(row)

    return summaries


def write_summary_csv(rows: List[Dict[str, str]], out_csv: Path) -> List[str]:
    """
    Writes union-of-keys CSV and returns the column order used.
    """
    keys = set()
    for r in rows:
        keys.update(r.keys())

    cols = [c for c in PREFERRED_COLS if c in keys]
    cols += sorted([c for c in keys if c not in cols])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    return cols


def infer_col_spec(cols: List[str]) -> str:
    """
    Basic LaTeX alignment: numbers right, text left.
    """
    numeric = {"n", "steps", "events", "topples", "rate"}
    spec = []
    for c in cols:
        if norm_key(c) in numeric:
            spec.append("r")
        else:
            spec.append("l")
    return "".join(spec)


def write_latex_table(rows: List[Dict[str, str]], cols: List[str], out_tex: Path,
                      caption: str, label: str) -> None:
    """
    Writes a complete LaTeX table environment using booktabs.
    """
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    col_spec = infer_col_spec(cols)
    header = " & ".join(escape_latex(c) for c in cols) + r" \\"

    lines: List[str] = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(fr"\caption{{{escape_latex(caption)}}}")
    lines.append(fr"\label{{{escape_latex(label)}}}")
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\toprule")
    lines.append(header)
    lines.append(r"\midrule")

    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            cells.append(escape_latex(v))
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    out_tex.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    repo_root = find_repo_root(script_dir)

    drop_dir  = repo_root / "data" / "module1" / "fold_runs"
    reset_dir = repo_root / "data" / "module1" / "truncated_fold_runs"
    out_dir   = pick_output_dir(repo_root)

    # Summarize
    drop_rows  = summarize_folder(drop_dir,  "drop")
    reset_rows = summarize_folder(reset_dir, "reset")

    # Write CSV summaries
    drop_csv  = out_dir / DROP_SUMMARY_CSV
    reset_csv = out_dir / RESET_SUMMARY_CSV
    drop_cols  = write_summary_csv(drop_rows, drop_csv)
    reset_cols = write_summary_csv(reset_rows, reset_csv)

    # Write LaTeX tables
    drop_tex  = out_dir / DROP_TABLE_TEX
    reset_tex = out_dir / RESET_TABLE_TEX

    write_latex_table(drop_rows,  drop_cols,  drop_tex,
                      "Appendix B results summary (mode=drop).", "tab:appB-drop")
    write_latex_table(reset_rows, reset_cols, reset_tex,
                      "Appendix B results summary (mode=reset).", "tab:appB-reset")

    print("Repo root: ", repo_root)
    print("Wrote:")
    print(" ", drop_csv)
    print(" ", reset_csv)
    print(" ", drop_tex)
    print(" ", reset_tex)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise SystemExit(1)
