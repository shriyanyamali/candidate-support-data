## 02

import pandas as pd
from pathlib import Path
from config import TARGET_ELECTION_YR, write_csv_no_blank_line

def _find_file(folder: Path, startswith: str) -> Path:
    for ext in ("*.txt", "*.dat"):
        for p in folder.glob(ext):
            if p.name.lower().startswith(startswith.lower()):
                return p
    cands = list(folder.glob("*.txt")) + list(folder.glob("*.dat"))
    if not cands:
        raise FileNotFoundError(f"No data files found in {folder}")
    return max(cands, key=lambda p: p.stat().st_size)

def main(cfg=None):
    if cfg is None:
        from config import CM_DIR, CN_DIR, PAS2_DIR, OUT_DIR, CM_COLS, CN_COLS, ITPAS2_COLS, SUFFIX, VALID_OFFICES, CHUNKSIZE
    else:
        CM_DIR = cfg['CM_DIR']
        CN_DIR = cfg['CN_DIR']
        PAS2_DIR = cfg['PAS2_DIR']
        OUT_DIR = cfg['OUT_DIR']
        CM_COLS = cfg['CM_COLS']
        CN_COLS = cfg['CN_COLS']
        ITPAS2_COLS = cfg['ITPAS2_COLS']
        SUFFIX = cfg['SUFFIX']
        VALID_OFFICES = cfg['VALID_OFFICES']
        CHUNKSIZE = cfg['CHUNKSIZE']
    
    cm_path = _find_file(CM_DIR, "cm")
    cn_path = _find_file(CN_DIR, "cn")
    itpas2_path = _find_file(PAS2_DIR, "itpas2")

    print("[superpac_ie_support] Loading committee master:", cm_path)
    cm = pd.read_csv(cm_path, sep="|", header=None, names=CM_COLS, dtype=str, encoding_errors="ignore")
    superpac_ids = set(cm.loc[cm["CMTE_TP"] == "O", "CMTE_ID"].dropna().unique())
    print(f"[superpac_ie_support] IE-only committees (CMTE_TP='O'): {len(superpac_ids):,}")

    print("[superpac_ie_support] Loading candidate master:", cn_path)
    cn = pd.read_csv(cn_path, sep="|", header=None, names=CN_COLS, dtype=str, encoding_errors="ignore")

    cn = pd.read_csv(cn_path, sep="|", header=None, names=CN_COLS, dtype=str, encoding_errors="ignore")
    cn = cn[cn["CAND_OFFICE"].isin(VALID_OFFICES)].copy()

    cn["CAND_ELECTION_YR"] = cn["CAND_ELECTION_YR"].astype(str).str.extract(r"(\d{4})", expand=False)
    before = len(cn)
    cn = cn[cn["CAND_ELECTION_YR"] == TARGET_ELECTION_YR].copy()
    print(f"[superpac_ie_support] cn after year filter {TARGET_ELECTION_YR}: {before:,} -> {len(cn):,}")

    valid_cand_ids = set(cn["CAND_ID"].dropna().unique())
    cn_index = cn.set_index("CAND_ID")

    totals = {}

    print("[superpac_ie_support] Streaming itpas2:", itpas2_path)
    reader = pd.read_csv(
        itpas2_path, sep="|", header=None, names=ITPAS2_COLS,
        dtype=str, chunksize=CHUNKSIZE, encoding_errors="ignore",
        on_bad_lines="skip"
    )

    for i, chunk in enumerate(reader, start=1):
        # IE support
        chunk = chunk[chunk["TRANSACTION_TP"] == "24E"]
        if chunk.empty:
            continue

        # IE-only committees
        chunk = chunk[chunk["CMTE_ID"].isin(superpac_ids)]
        if chunk.empty:
            continue

        # ✅ Drop House candidates
        chunk = chunk[chunk["CAND_ID"].isin(valid_cand_ids)]
        if chunk.empty:
            continue

        amt = pd.to_numeric(chunk["TRANSACTION_AMT"], errors="coerce")
        mask = amt.notna() & (amt > 0)
        if not mask.any():
            continue

        amt = amt.loc[mask]
        chunk = chunk.loc[mask]

        grp = amt.groupby(chunk["CAND_ID"]).sum()
        for cand_id, val in grp.items():
            totals[cand_id] = totals.get(cand_id, 0.0) + float(val)

        if i % 5 == 0:
            print(f"[superpac_ie_support] chunks: {i:,} | candidates so far: {len(totals):,}")

    rows = [{"CAND_ID": k, "SUPERPAC_IE_SUPPORT": v} for k, v in totals.items()]
    out = (
        pd.DataFrame(rows, columns=["CAND_ID", "SUPERPAC_IE_SUPPORT"])
          .merge(cn_index, left_on="CAND_ID", right_index=True, how="left")
          .sort_values("SUPERPAC_IE_SUPPORT", ascending=False)
    )

    out_path = OUT_DIR / f"superpac_ie_support_{SUFFIX}.csv"
    write_csv_no_blank_line(out, out_path, index=False)
    print("[superpac_ie_support] Wrote:", out_path)

if __name__ == "__main__":
    main()