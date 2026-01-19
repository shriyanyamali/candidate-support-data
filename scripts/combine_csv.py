from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


CYCLE_RE = re.compile(r"_(\d{2})\.csv$", re.IGNORECASE)  # e.g. final_support_table_20.csv


def infer_cycle(filename: str) -> str | None:
    m = CYCLE_RE.search(filename)
    return m.group(1) if m else None


def infer_office_type(filepath: Path) -> str:
    """Infer office type from directory structure or filename prefix."""
    # Check parent directory name
    parent = filepath.parent.name.lower()
    if parent in ("senate", "presidential", "total"):
        return parent
    
    # Check filename prefix
    name_lower = filepath.name.lower()
    if name_lower.startswith("senate_"):
        return "senate"
    elif name_lower.startswith("presidential_"):
        return "presidential"
    elif name_lower.startswith("total_"):
        return "total"
    
    return "unknown"


def combine_csvs(input_dir: Path, output_path: Path, recursive: bool = False) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if recursive:
        # Recursively find all CSV files in subdirectories
        csv_files = sorted(input_dir.rglob("*.csv"))
    else:
        csv_files = sorted(input_dir.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No .csv files found in: {input_dir}")

    frames: list[pd.DataFrame] = []
    for f in csv_files:
        df = pd.read_csv(f, dtype=str, low_memory=False)  # keep everything as text to avoid type conflicts
        df.columns = [c.strip() for c in df.columns]

        df["source_file"] = f.name
        df["source_path"] = str(f.relative_to(input_dir))
        
        cyc = infer_cycle(f.name)
        df["cycle"] = cyc if cyc is not None else ""
        
        office_type = infer_office_type(f)
        df["office_type"] = office_type

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True, sort=False)

    # Optional: drop exact duplicate rows
    combined = combined.drop_duplicates()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    print(f"Combined {len(csv_files)} files -> {output_path}")
    print(f"Rows: {len(combined):,} | Columns: {len(combined.columns):,}")
    
    # Show breakdown by office type
    if "office_type" in combined.columns:
        print("\nBreakdown by office type:")
        office_counts = combined["office_type"].value_counts()
        for office, count in office_counts.items():
            print(f"  {office}: {count:,} rows")


def combine_by_type(input_dir: Path, output_dir: Path) -> None:
    """
    Combine CSVs separately for each office type (senate, presidential, total).
    Creates three separate combined files.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Look for subdirectories
    senate_dir = input_dir / "senate"
    presidential_dir = input_dir / "presidential"
    total_dir = input_dir / "total"
    
    for subdir, office_name in [(senate_dir, "senate"), 
                                  (presidential_dir, "presidential"), 
                                  (total_dir, "total")]:
        if not subdir.exists():
            print(f"[WARN] Directory not found: {subdir}")
            continue
        
        csv_files = sorted(subdir.glob("*.csv"))
        if not csv_files:
            print(f"[WARN] No CSV files found in: {subdir}")
            continue
        
        print(f"\nProcessing {office_name} files...")
        output_file = output_dir / f"combined_{office_name}_ALL.csv"
        
        frames: list[pd.DataFrame] = []
        for f in csv_files:
            df = pd.read_csv(f, dtype=str, low_memory=False)
            df.columns = [c.strip() for c in df.columns]
            
            df["source_file"] = f.name
            cyc = infer_cycle(f.name)
            df["cycle"] = cyc if cyc is not None else ""
            df["office_type"] = office_name
            
            frames.append(df)
        
        combined = pd.concat(frames, ignore_index=True, sort=False)
        combined = combined.drop_duplicates()
        
        combined.to_csv(output_file, index=False)
        print(f"  Combined {len(csv_files)} files -> {output_file}")
        print(f"  Rows: {len(combined):,} | Columns: {len(combined.columns):,}")


def main() -> None:
    default_input = Path(r"C:\Users\sruja\Downloads\Data Collection\FEC_Data\final_output_files")
    default_output = default_input / "final_support_table_ALL.csv"

    ap = argparse.ArgumentParser(
        description="Combine CSVs from final_output_files. Can combine all into one file or separate by office type."
    )
    ap.add_argument("--input-dir", type=Path, default=default_input, help="Folder containing the CSV files")
    ap.add_argument("--output", type=Path, default=default_output, help="Output CSV path (for single file mode)")
    ap.add_argument("--output-dir", type=Path, help="Output directory (for by-type mode)")
    ap.add_argument("--recursive", action="store_true", help="Recursively search for CSV files in subdirectories")
    ap.add_argument("--by-type", action="store_true", help="Create separate combined files for senate/presidential/total")
    args = ap.parse_args()

    if args.by_type:
        output_dir = args.output_dir or args.input_dir
        combine_by_type(args.input_dir, output_dir)
    else:
        combine_csvs(args.input_dir, args.output, recursive=args.recursive)


if __name__ == "__main__":
    main()
