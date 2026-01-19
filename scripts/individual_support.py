## 04

import pandas as pd
from pathlib import Path
from config import TARGET_ELECTION_YR, write_csv_no_blank_line, get_output_dir, get_output_prefix

def _find_file(folder: Path, startswith: str) -> Path:
    for ext in ("*.txt", "*.dat"):
        for p in folder.glob(ext):
            if p.name.lower().startswith(startswith.lower()):
                return p
    cands = list(folder.glob("*.txt")) + list(folder.glob("*.dat"))
    if not cands:
        raise FileNotFoundError(f"No data files found in {folder}")
    return max(cands, key=lambda p: p.stat().st_size)

def _build_cmte_to_cand(ccl: pd.DataFrame) -> dict:
    """
    Deterministic mapping CMTE_ID -> CAND_ID.
    Prefer CMTE_DSGN == 'P' if present, else first observed.
    """
    ccl = ccl.copy()
    ccl["CMTE_DSGN"] = ccl["CMTE_DSGN"].fillna("")
    ccl["__is_principal"] = (ccl["CMTE_DSGN"] == "P").astype(int)
    ccl = ccl.sort_values(["CMTE_ID", "__is_principal"], ascending=[True, False])
    chosen = ccl.dropna(subset=["CMTE_ID", "CAND_ID"]).drop_duplicates("CMTE_ID", keep="first")
    return dict(zip(chosen["CMTE_ID"], chosen["CAND_ID"]))

def main(office_filter=None, cfg=None):
    """
    Generate individual contribution support data.
    
    Args:
        office_filter: Set of office codes to include (e.g., {'S'}, {'P'}, or {'S', 'P'})
        cfg: Optional config dict (for testing/flexibility)
    """
    if cfg is None:
        from config import CCL_DIR, CN_DIR, INDIV_DIR, CCL_COLS, CN_COLS, INDIV_COLS, SUFFIX, VALID_OFFICES, CHUNKSIZE
    else:
        CCL_DIR = cfg['CCL_DIR']
        CN_DIR = cfg['CN_DIR']
        INDIV_DIR = cfg['INDIV_DIR']
        CCL_COLS = cfg['CCL_COLS']
        CN_COLS = cfg['CN_COLS']
        INDIV_COLS = cfg['INDIV_COLS']
        SUFFIX = cfg['SUFFIX']
        VALID_OFFICES = cfg['VALID_OFFICES']
        CHUNKSIZE = cfg['CHUNKSIZE']
    
    # Use provided office_filter or default to all valid offices
    if office_filter is None:
        office_filter = VALID_OFFICES
    office_filter = set(office_filter)  # Ensure it's a set
    
    # Get appropriate output directory and prefix
    out_dir = get_output_dir(office_filter)
    prefix = get_output_prefix(office_filter)
    
    ccl_path = _find_file(CCL_DIR, "ccl")
    cn_path = _find_file(CN_DIR, "cn")
    indiv_path = _find_file(INDIV_DIR, "itcont")

    print(f"[individual_support][{prefix}] Loading ccl linkage:", ccl_path)
    ccl = pd.read_csv(ccl_path, sep="|", header=None, names=CCL_COLS, dtype=str, encoding_errors="ignore")
    cmte_to_cand = _build_cmte_to_cand(ccl)

    print(f"[individual_support][{prefix}] Loading candidate master:", cn_path)

    cn = pd.read_csv(cn_path, sep="|", header=None, names=CN_COLS, dtype=str, encoding_errors="ignore")
    
    # Filter to specified offices
    cn = cn[cn["CAND_OFFICE"].isin(office_filter)].copy()
    print(f"[individual_support][{prefix}] After office filter {sorted(office_filter)}: {len(cn):,} candidates")

    # Election-year restriction
    cn["CAND_ELECTION_YR"] = cn["CAND_ELECTION_YR"].astype(str).str.extract(r"(\d{4})", expand=False)
    before = len(cn)
    cn = cn[cn["CAND_ELECTION_YR"] == TARGET_ELECTION_YR].copy()
    print(f"[individual_support][{prefix}] After year filter {TARGET_ELECTION_YR}: {before:,} -> {len(cn):,}")

    valid_cand_ids = set(cn["CAND_ID"].dropna().unique())
    cn_index = cn.set_index("CAND_ID")

    totals = {}

    print(f"[individual_support][{prefix}] Streaming itcont:", indiv_path)
    reader = pd.read_csv(
        indiv_path, sep="|", header=None, names=INDIV_COLS,
        dtype=str, chunksize=CHUNKSIZE, encoding_errors="ignore",
        on_bad_lines="skip"
    )

    for i, chunk in enumerate(reader, start=1):
        chunk = chunk[(chunk["TRANSACTION_TP"].isin(["15", "15E"])) & (chunk["ENTITY_TP"] == "IND")].copy()
        if chunk.empty:
            continue

        # Map committee -> candidate
        chunk["CAND_ID"] = chunk["CMTE_ID"].map(cmte_to_cand)
        chunk = chunk[chunk["CAND_ID"].notna()]
        if chunk.empty:
            continue

        # Filter to valid candidates for this office type
        chunk = chunk[chunk["CAND_ID"].isin(valid_cand_ids)]
        if chunk.empty:
            continue

        amt = pd.to_numeric(chunk["TRANSACTION_AMT"], errors="coerce")
        mask = amt.notna() & (amt > 0)
        if not mask.any():
            continue

        chunk = chunk.loc[mask]
        amt = amt.loc[mask]

        grp = amt.groupby(chunk["CAND_ID"]).sum()
        for cand_id, val in grp.items():
            totals[cand_id] = totals.get(cand_id, 0.0) + float(val)

        if i % 5 == 0:
            print(f"[individual_support][{prefix}] chunks: {i:,} | candidates so far: {len(totals):,}")

    rows = [{"CAND_ID": k, "INDIVIDUAL_SUPPORT": v} for k, v in totals.items()]
    out = (
        pd.DataFrame(rows, columns=["CAND_ID", "INDIVIDUAL_SUPPORT"])
          .merge(cn_index, left_on="CAND_ID", right_index=True, how="left")
          .sort_values("INDIVIDUAL_SUPPORT", ascending=False)
    )

    from config import SUFFIX
    out_path = out_dir / f"{prefix}_individual_support_{SUFFIX}.csv"
    write_csv_no_blank_line(out, out_path, index=False)
    print(f"[individual_support][{prefix}] Wrote:", out_path)

if __name__ == "__main__":
    main()
