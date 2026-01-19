## 02

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
    # fallback largest
    cands = list(folder.glob("*.txt")) + list(folder.glob("*.dat"))
    if not cands:
        raise FileNotFoundError(f"No data files found in {folder}")
    return max(cands, key=lambda p: p.stat().st_size)

def main():
    cm_path = _find_file(CM_DIR, "cm")
    cn_path = _find_file(CN_DIR, "cn")
    itpas2_path = _find_file(PAS2_DIR, "itpas2")

    print("[superpac_ie_support] Loading committee master:", cm_path)
    cm = pd.read_csv(cm_path, sep="|", header=None, names=CM_COLS, dtype=str, encoding_errors="ignore")
    superpac_ids = set(cm.loc[cm["CMTE_TP"] == "O", "CMTE_ID"].dropna().unique())
    print(f"[superpac_ie_support] Super PAC committees (CMTE_TP='O'): {len(superpac_ids):,}")

    print("[superpac_ie_support] Loading candidate master:", cn_path)
    cn = pd.read_csv(cn_path, sep="|", header=None, names=CN_COLS, dtype=str, encoding_errors="ignore")
    if SENATE_ONLY:
        cn = cn[cn["CAND_OFFICE"] == "S"]
    valid_cand_ids = set(cn["CAND_ID"].dropna().unique())
    cn_index = cn.set_index("CAND_ID")

    totals = {}

    print("[superpac_ie_support] Streaming itpas2:", itpas2_path)
    reader = pd.read_csv(
        itpas2_path, sep="|", header=None, names=ITPAS2_COLS,
        dtype=str, chunksize=CHUNKSIZE, encoding_errors="ignore"
    )

    for i, chunk in enumerate(reader, start=1):
        # IE SUPPORT
        chunk = chunk[chunk["TRANSACTION_TP"] == "24E"]
        if chunk.empty:
            continue

        # Super PAC spenders
        chunk = chunk[chunk["CMTE_ID"].isin(superpac_ids)]
        if chunk.empty:
            continue

        # Candidate filter
        if SENATE_ONLY:
            chunk = chunk[chunk["CAND_ID"].isin(valid_cand_ids)]
            if chunk.empty:
                continue

        amt = pd.to_numeric(chunk["TRANSACTION_AMT"], errors="coerce").fillna(0.0)
        grp = amt.groupby(chunk["CAND_ID"]).sum()
        for cand_id, val in grp.items():
            totals[cand_id] = totals.get(cand_id, 0.0) + float(val)

        if i % 5 == 0:
            print(f"[superpac_ie_support] processed chunks: {i:,} | candidates so far: {len(totals):,}")

    out = (
        pd.DataFrame([{"CAND_ID": k, "SUPERPAC_IE_SUPPORT": v} for k, v in totals.items()])
          .merge(cn_index, left_on="CAND_ID", right_index=True, how="left")
          .sort_values("SUPERPAC_IE_SUPPORT", ascending=False)
    )

    out_path = OUT_DIR / "superpac_ie_support.csv"
    out.to_csv(out_path, index=False)
    print("[superpac_ie_support] Wrote:", out_path)

if __name__ == "__main__":
    main()
