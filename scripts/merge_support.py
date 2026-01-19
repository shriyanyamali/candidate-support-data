## 05

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

def _safe_read_csv(path: Path, cols: list, dtypes=None) -> pd.DataFrame:
    """
    Read a CSV if it exists; otherwise return empty DF with requested cols.
    Ensures requested cols exist (fills missing numeric cols with 0).
    """
    if not path.exists():
        print(f"[merge_support][WARN] Missing file: {path} (using zeros)")
        return pd.DataFrame(columns=cols)

    df = pd.read_csv(path, dtype=dtypes)

    for c in cols:
        if c not in df.columns:
            # Default missing numeric cols to 0; missing keys to NaN
            if c in ("CAND_ID", "CAND_ELECTION_YR"):
                df[c] = pd.NA
            else:
                df[c] = 0.0

    return df[cols]

def _coerce_year(series: pd.Series) -> pd.Series:
    """
    Keep election year as string but normalize to 4-digit where possible.
    """
    s = series.astype(str)
    # clean obvious junk
    s = s.replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
    # keep only digits if present
    s = s.str.extract(r"(\d{4})", expand=False)
    return s

def _collapse_support(df: pd.DataFrame, name: str, key_cols: list, sum_cols: list) -> pd.DataFrame:
    """
    Ensure one row per key by summing numeric support columns.
    Prints diagnostics if duplicates were found/collapsed.
    """
    if df.empty:
        return df

    # normalize key cols existence
    for k in key_cols:
        if k not in df.columns:
            df[k] = pd.NA

    # coerce numeric cols
    for c in sum_cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    dup_mask = df.duplicated(key_cols, keep=False)
    if dup_mask.any():
        n_groups = df.loc[dup_mask].groupby(key_cols).ngroups
        n_rows = int(dup_mask.sum())
        print(f"[merge_support][INFO] {name}: collapsing {n_groups:,} duplicate key groups ({n_rows:,} rows) by summing columns {sum_cols}")

        # show one example group
        example = df.loc[dup_mask].groupby(key_cols).size().sort_values(ascending=False).head(1)
        ex_key = example.index[0]
        ex_n = int(example.iloc[0])
        print(f"[merge_support][INFO] {name}: example duplicate group {ex_key} has {ex_n} rows")

    collapsed = (
        df.groupby(key_cols, as_index=False)[sum_cols].sum()
    )

    return collapsed

def main(office_filter=None):
    """
    Merge support files for a specific office type.
    
    Args:
        office_filter: Set of office codes to include (e.g., {'S'}, {'P'}, or {'S', 'P'})
    """
    from config import CN_DIR, CN_COLS, SUFFIX, VALID_OFFICES
    
    # Use provided office_filter or default to all valid offices
    if office_filter is None:
        office_filter = VALID_OFFICES
    office_filter = set(office_filter)  # Ensure it's a set
    
    # Get appropriate output directory and prefix
    out_dir = get_output_dir(office_filter)
    prefix = get_output_prefix(office_filter)
    
    # Inputs - now using office-specific prefixes
    superpac_path = out_dir / f"{prefix}_superpac_ie_support_{SUFFIX}.csv"
    indiv_path = out_dir / f"{prefix}_individual_support_{SUFFIX}.csv"
    pac_path = out_dir / f"{prefix}_pac_support_corp_nonconnected_{SUFFIX}.csv"

    cn_path = _find_file(CN_DIR, "cn")

    print(f"[merge_support][{prefix}] Reading:")
    print("  cn:", cn_path)
    print("  superpac:", superpac_path)
    print("  indiv:", indiv_path)
    print("  pac:", pac_path)

    # ---------------------------
    # Load candidate master (authoritative universe)
    # ---------------------------
    cn = pd.read_csv(
        cn_path, sep="|", header=None, names=CN_COLS,
        dtype=str, encoding_errors="ignore"
    )

    # Restrict to specified offices
    before = len(cn)
    cn = cn[cn["CAND_OFFICE"].isin(office_filter)].copy()
    print(f"[merge_support][{prefix}] Candidate master: {before:,} rows -> {len(cn):,} after office filter {sorted(office_filter)}")

    # Normalize election year
    cn["CAND_ELECTION_YR"] = _coerce_year(cn["CAND_ELECTION_YR"])

    before_yr = len(cn)
    cn = cn[cn["CAND_ELECTION_YR"] == TARGET_ELECTION_YR].copy()
    print(f"[merge_support][{prefix}] Candidate master: {before_yr:,} rows -> {len(cn):,} after election-year filter == {TARGET_ELECTION_YR}")

    # Diagnostics: candidate IDs spanning multiple election years
    multi_year = cn.groupby("CAND_ID")["CAND_ELECTION_YR"].nunique(dropna=True)
    n_multi = int((multi_year > 1).sum())
    if n_multi > 0:
        print(f"[merge_support][{prefix}][WARN] {n_multi:,} CAND_IDs appear in multiple election years in cn.")
        print(f"[merge_support][{prefix}][WARN] Example multi-year CAND_IDs (up to 10):")
        ex_ids = multi_year[multi_year > 1].sort_values(ascending=False).head(10).index.tolist()
        print("  ", ", ".join(ex_ids))
    else:
        print(f"[merge_support][{prefix}] OK: No CAND_ID spans multiple election years in cn after filtering.")

    # Enforce one row per (CAND_ID, CAND_ELECTION_YR)
    # Prefer "best" administrative record when duplicates exist.
    cn = cn.copy()
    cn["CAND_STATUS"] = cn["CAND_STATUS"].fillna("")
    cn["CAND_PCC"] = cn["CAND_PCC"].fillna("")

    # Scoring: prefer rows that have a PCC; then prefer status == 'C' if present
    cn["__has_pcc"] = (cn["CAND_PCC"].str.len() > 0).astype(int)
    cn["__is_status_C"] = (cn["CAND_STATUS"] == "C").astype(int)

    # Sort so the first row in each group is the one we keep
    cn = cn.sort_values(
        ["CAND_ID", "CAND_ELECTION_YR", "__has_pcc", "__is_status_C"],
        ascending=[True, True, False, False]
    )

    # Count duplicates before collapsing
    dup_mask = cn.duplicated(["CAND_ID", "CAND_ELECTION_YR"], keep=False)
    n_dup_rows = int(dup_mask.sum())
    if n_dup_rows > 0:
        n_dup_groups = int(cn.loc[dup_mask].groupby(["CAND_ID", "CAND_ELECTION_YR"]).ngroups)
        print(f"[merge_support][{prefix}][INFO] Found {n_dup_groups:,} duplicate (CAND_ID, CAND_ELECTION_YR) groups ({n_dup_rows:,} rows). Collapsing to 1 per group.")
        print(f"[merge_support][{prefix}][INFO] Example duplicate groups (up to 5):")
        for (cid, yr), g in cn.loc[dup_mask].groupby(["CAND_ID", "CAND_ELECTION_YR"]):
            print(f"  - {cid} / {yr}: {len(g)} rows | statuses={sorted(set(g['CAND_STATUS']))} | pcc_present={sorted(set((g['CAND_PCC'].str.len()>0).astype(int)))}")
            if len(g) > 0:
                break
    else:
        print(f"[merge_support][{prefix}] OK: No duplicate (CAND_ID, CAND_ELECTION_YR) groups in cn.")

    cn_labels = cn.drop_duplicates(["CAND_ID", "CAND_ELECTION_YR"], keep="first")[
        ["CAND_ID", "CAND_ELECTION_YR", "CAND_NAME", "CAND_PTY_AFFILIATION", "CAND_OFFICE", "CAND_OFFICE_ST"]
    ].copy()

    # Hard assertion (prints friendly error then raises)
    if cn_labels.duplicated(["CAND_ID", "CAND_ELECTION_YR"]).any():
        bad = cn_labels[cn_labels.duplicated(["CAND_ID", "CAND_ELECTION_YR"], keep=False)].head(10)
        print(f"[merge_support][{prefix}][ERROR] Duplicate candidate-year records remain after collapse. Examples:")
        print(bad.to_string(index=False))
        raise ValueError("Duplicate (CAND_ID, CAND_ELECTION_YR) found in cn_labels")

    print(f"[merge_support][{prefix}] Candidate universe for merge: {len(cn_labels):,} unique candidate-years")

    # ---------------------------
    # Read support files (prefer candidate-year merges if available)
    # ---------------------------
    # Try reading with CAND_ELECTION_YR; if not present, will be NA and we fall back to ID-only merge.
    superpac = _safe_read_csv(
        superpac_path,
        cols=["CAND_ID", "CAND_ELECTION_YR", "SUPERPAC_IE_SUPPORT"],
        dtypes={"CAND_ID": str}
    )
    indiv = _safe_read_csv(
        indiv_path,
        cols=["CAND_ID", "CAND_ELECTION_YR", "INDIVIDUAL_SUPPORT"],
        dtypes={"CAND_ID": str}
    )
    pac = _safe_read_csv(
        pac_path,
        cols=["CAND_ID", "CAND_ELECTION_YR", "CORP_PAC_SUPPORT", "NONCONNECTED_PAC_SUPPORT"],
        dtypes={"CAND_ID": str}
    )

    # Collapse duplicates in support files so merges never discard values
    key_cols = ["CAND_ID", "CAND_ELECTION_YR"]

    superpac = _collapse_support(
        superpac, f"{prefix}_superpac",
        key_cols=key_cols,
        sum_cols=["SUPERPAC_IE_SUPPORT"]
    )

    indiv = _collapse_support(
        indiv, f"{prefix}_indiv",
        key_cols=key_cols,
        sum_cols=["INDIVIDUAL_SUPPORT"]
    )

    pac = _collapse_support(
        pac, f"{prefix}_pac",
        key_cols=key_cols,
        sum_cols=["CORP_PAC_SUPPORT", "NONCONNECTED_PAC_SUPPORT"]
    )

    # Normalize years if present
    for df_name, df in [(f"{prefix}_superpac", superpac), (f"{prefix}_indiv", indiv), (f"{prefix}_pac", pac)]:
        if "CAND_ELECTION_YR" in df.columns:
            df["CAND_ELECTION_YR"] = _coerce_year(df["CAND_ELECTION_YR"])

    # Determine merge strategy
    has_year_superpac = superpac["CAND_ELECTION_YR"].notna().any()
    has_year_indiv = indiv["CAND_ELECTION_YR"].notna().any()
    has_year_pac = pac["CAND_ELECTION_YR"].notna().any()

    use_year_merge = bool(has_year_superpac and has_year_indiv and has_year_pac)

    if use_year_merge:
        print(f"[merge_support][{prefix}] Merge strategy: using keys (CAND_ID, CAND_ELECTION_YR) for all support files.")
        merged = (
            cn_labels
            .merge(indiv, on=["CAND_ID", "CAND_ELECTION_YR"], how="left")
            .merge(pac, on=["CAND_ID", "CAND_ELECTION_YR"], how="left")
            .merge(superpac, on=["CAND_ID", "CAND_ELECTION_YR"], how="left")
        )
    else:
        print(f"[merge_support][{prefix}][WARN] One or more support files missing CAND_ELECTION_YR; falling back to CAND_ID-only merge.")
        print(f"  superpac has year? {has_year_superpac} | indiv has year? {has_year_indiv} | pac has year? {has_year_pac}")
        merged = (
            cn_labels
            .merge(indiv.drop(columns=["CAND_ELECTION_YR"], errors="ignore"), on="CAND_ID", how="left")
            .merge(pac.drop(columns=["CAND_ELECTION_YR"], errors="ignore"), on="CAND_ID", how="left")
            .merge(superpac.drop(columns=["CAND_ELECTION_YR"], errors="ignore"), on="CAND_ID", how="left")
        )

    # ---------------------------
    # Numeric cleanup + totals
    # ---------------------------
    support_cols = [
        "INDIVIDUAL_SUPPORT",
        "CORP_PAC_SUPPORT",
        "NONCONNECTED_PAC_SUPPORT",
        "SUPERPAC_IE_SUPPORT",
    ]
    for col in support_cols:
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    merged["TOTAL_SUPPORT"] = merged[support_cols].sum(axis=1)
    merged["HAS_MONEY"] = (merged["TOTAL_SUPPORT"] > 0).astype(int)

    # ---------------------------
    # Post-merge diagnostics
    # ---------------------------
    # Check for duplicates after merge
    if merged.duplicated(["CAND_ID", "CAND_ELECTION_YR"]).any():
        bad = merged[merged.duplicated(["CAND_ID", "CAND_ELECTION_YR"], keep=False)].head(20)
        print(f"[merge_support][{prefix}][ERROR] Duplicate rows in merged output by (CAND_ID, CAND_ELECTION_YR). Examples:")
        print(bad.to_string(index=False))
        raise ValueError("Duplicates in merged output by (CAND_ID, CAND_ELECTION_YR)")

    # Check for any invalid office that slipped in
    bad_office = merged.loc[~merged["CAND_OFFICE"].isin(office_filter)]
    if not bad_office.empty:
        print(f"[merge_support][{prefix}][ERROR] Found candidates outside office_filter in merged output. Examples:")
        print(bad_office.head(10).to_string(index=False))
        raise ValueError("Invalid office found in merged output")

    # Summaries
    print(f"[merge_support][{prefix}] Money summary:")
    print(f"  Candidates with money: {int((merged['HAS_MONEY'] == 1).sum()):,}")
    print(f"  Candidates with zero : {int((merged['HAS_MONEY'] == 0).sum()):,}")
    print(f"  Total candidates     : {len(merged):,}")
    print(f"  Total $ support      : {merged['TOTAL_SUPPORT'].sum():,.2f}")

    # ---------------------------
    # Sorting + outputs
    # ---------------------------
    merged_sorted = merged.sort_values(["CAND_OFFICE_ST", "TOTAL_SUPPORT"], ascending=[True, False])

    with_money = merged_sorted[merged_sorted["HAS_MONEY"] == 1].copy()
    no_money = merged_sorted[merged_sorted["HAS_MONEY"] == 0].copy()

    out_with_money = out_dir / f"{prefix}_final_support_table_{SUFFIX}.csv"
    out_no_money = out_dir / f"{prefix}_candidates_no_support_{SUFFIX}.csv"
    out_all_flag = out_dir / f"{prefix}_candidates_all_with_flag_{SUFFIX}.csv"

    write_csv_no_blank_line(with_money, out_with_money, index=False)
    write_csv_no_blank_line(no_money, out_no_money, index=False)
    write_csv_no_blank_line(merged_sorted, out_all_flag, index=False)

    print(f"[merge_support][{prefix}] Wrote:")
    print("  ", out_with_money)
    print("  ", out_no_money)
    print("  ", out_all_flag)

    print(f"\n[merge_support][{prefix}] Preview (top 25 with money):")
    print(with_money.head(25).to_string(index=False))

if __name__ == "__main__":
    main()
