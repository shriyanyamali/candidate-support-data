## 04

import pandas as pd
from pathlib import Path
from config import (
    CCL_DIR, CN_DIR, INDIV_DIR, OUT_DIR, SENATE_ONLY, CHUNKSIZE,
    CCL_COLS, CN_COLS, INDIV_COLS
)

def _find_file(folder: Path, startswith: str) -> Path:
    for ext in ("*.txt", "*.dat"):
        for p in folder.glob(ext):
            if p.name.lower().startswith(startswith.lower()):
                return p
    cands = list(folder.glob("*.txt")) + list(folder.glob("*.dat"))
    if not cands:
        raise FileNotFoundError(f"No data files found in {folder}")
    return max(cands, key=lambda p: p.stat().st_size)

def main():
    ccl_path = _find_file(CCL_DIR, "ccl")
    cn_path = _find_file(CN_DIR, "cn")

    # indiv file can be named itcont.txt or similar
    indiv_path = _find_file(INDIV_DIR, "itcont")

    print("[individual_support] Loading ccl linkage:", ccl_path)
    ccl = pd.read_csv(ccl_path, sep="|", header=None, names=CCL_COLS, dtype=str, encoding_errors="ignore")
    # Many committees link to a candidate; keep mapping for all.
    cmte_to_cand = dict(zip(ccl["CMTE_ID"], ccl["CAND_ID"]))

    print("[individual_support] Loading candidate master:", cn_path)
    cn = pd.read_csv(cn_path, sep="|", header=None, names=CN_COLS, dtype=str, encoding_errors="ignore")
    if SENATE_ONLY:
        cn = cn[cn["CAND_OFFICE"] == "S"]
    valid_cand_ids = set(cn["CAND_ID"].dropna().unique())
    cn_index = cn.set_index("CAND_ID")

    totals = {}

    print("[individual_support] Streaming itcont:", indiv_path)
    reader = pd.read_csv(
        indiv_path, sep="|", header=None, names=INDIV_COLS,
        dtype=str, chunksize=CHUNKSIZE, encoding_errors="ignore"
    )

    for i, chunk in enumerate(reader, start=1):
        # Individual contribution records
        chunk = chunk[(chunk["TRANSACTION_TP"] == "15") & (chunk["ENTITY_TP"] == "IND")].copy()
        if chunk.empty:
            continue

        # Map committee -> candidate
        chunk["CAND_ID"] = chunk["CMTE_ID"].map(cmte_to_cand)
        chunk = chunk[chunk["CAND_ID"].notna()]
        if chunk.empty:
            continue

        if SENATE_ONLY:
            chunk = chunk[chunk["CAND_ID"].isin(valid_cand_ids)]
            if chunk.empty:
                continue

        amt = pd.to_numeric(chunk["TRANSACTION_AMT"], errors="coerce")
        chunk = chunk[amt.notna()]
        amt = amt.loc[chunk.index]
        chunk = chunk[amt > 0]  # keep positive support only
        if chunk.empty:
            continue

        grp = amt.loc[chunk.index].groupby(chunk["CAND_ID"]).sum()
        for cand_id, val in grp.items():
            totals[cand_id] = totals.get(cand_id, 0.0) + float(val)

        if i % 5 == 0:
            print(f"[individual_support] chunks: {i:,} | candidates so far: {len(totals):,}")

    out = (
        pd.DataFrame([{"CAND_ID": k, "INDIVIDUAL_SUPPORT": v} for k, v in totals.items()])
          .merge(cn_index, left_on="CAND_ID", right_index=True, how="left")
          .sort_values("INDIVIDUAL_SUPPORT", ascending=False)
    )

    out_path = OUT_DIR / "individual_support.csv"
    out.to_csv(out_path, index=False)
    print("[individual_support] Wrote:", out_path)

if __name__ == "__main__":
    main()
