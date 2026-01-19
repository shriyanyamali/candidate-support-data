## 03

import pandas as pd
from pathlib import Path
from config import (
    CM_DIR, CN_DIR, PAS2_DIR, OUT_DIR, SENATE_ONLY, CHUNKSIZE,
    CM_COLS, CN_COLS, ITPAS2_COLS
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
    cm_path = _find_file(CM_DIR, "cm")
    cn_path = _find_file(CN_DIR, "cn")
    itpas2_path = _find_file(PAS2_DIR, "itpas2")

    print("[pac_support_corp_union] Loading committee master:", cm_path)
    cm = pd.read_csv(cm_path, sep="|", header=None, names=CM_COLS, dtype=str, encoding_errors="ignore")
    cm_type = cm.set_index("CMTE_ID")["CMTE_TP"].to_dict()

    print("[pac_support_corp_union] Loading candidate master:", cn_path)
    cn = pd.read_csv(cn_path, sep="|", header=None, names=CN_COLS, dtype=str, encoding_errors="ignore")
    if SENATE_ONLY:
        cn = cn[cn["CAND_OFFICE"] == "S"]
    valid_cand_ids = set(cn["CAND_ID"].dropna().unique())
    cn_index = cn.set_index("CAND_ID")

    corp_totals = {}
    nonconn_totals = {}

    print("[pac_support_corp_union] Streaming itpas2:", itpas2_path)
    reader = pd.read_csv(
        itpas2_path, sep="|", header=None, names=ITPAS2_COLS,
        dtype=str, chunksize=CHUNKSIZE, encoding_errors="ignore"
    )

    for i, chunk in enumerate(reader, start=1):
        # Exclude independent expenditures (Super PAC IE support is handled elsewhere)
        chunk = chunk[~chunk["TRANSACTION_TP"].isin(["24E", "24O"])]
        if chunk.empty:
            continue

        # Candidate restriction (Senate only if configured)
        if SENATE_ONLY:
            chunk = chunk[chunk["CAND_ID"].isin(valid_cand_ids)]
            if chunk.empty:
                continue

        # Map committee type
        chunk = chunk.copy()
        chunk["CMTE_TP"] = chunk["CMTE_ID"].map(cm_type)

        # Amount numeric and positive only (avoid refunds/negatives for "support")
        amt = pd.to_numeric(chunk["TRANSACTION_AMT"], errors="coerce")
        mask = amt.notna() & (amt > 0)
        if not mask.any():
            continue

        chunk = chunk.loc[mask]
        chunk["AMT"] = amt.loc[mask]

        # Corporate-connected PACs (FEC CMTE_TP = 'C')
        corp = chunk[chunk["CMTE_TP"] == "C"]
        if not corp.empty:
            grp = corp["AMT"].groupby(corp["CAND_ID"]).sum()
            for cand_id, val in grp.items():
                corp_totals[cand_id] = corp_totals.get(cand_id, 0.0) + float(val)

        # Nonconnected PACs (FEC CMTE_TP = 'N')
        nonconn = chunk[chunk["CMTE_TP"] == "N"]
        if not nonconn.empty:
            grp = nonconn["AMT"].groupby(nonconn["CAND_ID"]).sum()
            for cand_id, val in grp.items():
                nonconn_totals[cand_id] = nonconn_totals.get(cand_id, 0.0) + float(val)

        if i % 5 == 0:
            print(
                f"[pac_support_corp_union] chunks: {i:,} | "
                f"corp cands: {len(corp_totals):,} | nonconn cands: {len(nonconn_totals):,}"
            )

    out = (
        pd.DataFrame({"CAND_ID": list(set(corp_totals) | set(nonconn_totals))})
          .assign(
              CORP_PAC_SUPPORT=lambda d: d["CAND_ID"].map(corp_totals).fillna(0.0),
              NONCONNECTED_PAC_SUPPORT=lambda d: d["CAND_ID"].map(nonconn_totals).fillna(0.0),
          )
          .merge(cn_index, left_on="CAND_ID", right_index=True, how="left")
          .sort_values(["CORP_PAC_SUPPORT", "NONCONNECTED_PAC_SUPPORT"], ascending=False)
    )

    out_path = OUT_DIR / "pac_support_corp_nonconnected.csv"
    out.to_csv(out_path, index=False)
    print("[pac_support_corp_union] Wrote:", out_path)

if __name__ == "__main__":
    main()