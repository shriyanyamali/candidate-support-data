# Code/combine_csv.py
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


CYCLE_RE = re.compile(r"_(\d{2})\.csv$", re.IGNORECASE)  # e.g. final_support_table_20.csv


def infer_cycle(filename: str) -> str | None:
    m = CYCLE_RE.search(filename)
    return m.group(1) if m else None


def combine_csvs(input_dir: Path, output_path: Path) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No .csv files found in: {input_dir}")

    frames: list[pd.DataFrame] = []
    for f in csv_files:
        df = pd.read_csv(f, dtype=str, low_memory=False)  # keep everything as text to avoid type conflicts
        df.columns = [c.strip() for c in df.columns]

        df["source_file"] = f.name
        cyc = infer_cycle(f.name)
        df["cycle"] = cyc if cyc is not None else ""

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True, sort=False)

    # Optional: drop exact duplicate rows
    combined = combined.drop_duplicates()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    print(f"Combined {len(csv_files)} files -> {output_path}")
    print(f"Rows: {len(combined):,} | Columns: {len(combined.columns):,}")


def main() -> None:
    default_input = Path(r"C:\Users\sruja\Downloads\Data Collection\FEC_Data\final_output_files")
    default_output = default_input / "final_support_table_ALL.csv"

    ap = argparse.ArgumentParser(description="Combine all CSVs in final_output_files into one CSV.")
    ap.add_argument("--input-dir", type=Path, default=default_input, help="Folder containing the CSV files")
    ap.add_argument("--output", type=Path, default=default_output, help="Output CSV path")
    args = ap.parse_args()

    combine_csvs(args.input_dir, args.output)


if __name__ == "__main__":
    main()