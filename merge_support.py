## 05

import pandas as pd
from config import OUT_DIR, CN_DIR, CN_COLS, SENATE_ONLY

def main():
    # Inputs
    superpac_path = OUT_DIR / "superpac_ie_support.csv"
    indiv_path = OUT_DIR / "individual_support.csv"
    pac_path = OUT_DIR / "pac_support_corp_nonconnected.csv"
    cn_path = CN_DIR / "cn.txt"

    print("[merge_support] Reading:")
    print("  ", cn_path)
    print("  ", superpac_path)
    print("  ", indiv_path)
    print("  ", pac_path)

    # Candidate labels (authoritative)
    cn = pd.read_csv(
        cn_path, sep="|", header=None, names=CN_COLS,
        dtype=str, encoding_errors="ignore"
    )
    if SENATE_ONLY:
        cn = cn[cn["CAND_OFFICE"] == "S"].copy()

    cn_labels = cn[
        ["CAND_ID", "CAND_NAME", "CAND_PTY_AFFILIATION", "CAND_OFFICE", "CAND_OFFICE_ST"]
    ].drop_duplicates("CAND_ID")

    # Support files
    superpac = pd.read_csv(superpac_path, dtype={"CAND_ID": str})
    indiv = pd.read_csv(indiv_path, dtype={"CAND_ID": str})
    pac = pd.read_csv(pac_path, dtype={"CAND_ID": str})

    superpac_key = superpac[["CAND_ID", "SUPERPAC_IE_SUPPORT"]]
    indiv_key = indiv[["CAND_ID", "INDIVIDUAL_SUPPORT"]]
    pac_key = pac[["CAND_ID", "CORP_PAC_SUPPORT", "NONCONNECTED_PAC_SUPPORT"]]

    # Merge from cn_labels so every Senate candidate is represented
    merged = (
        cn_labels
        .merge(indiv_key, on="CAND_ID", how="left")
        .merge(pac_key, on="CAND_ID", how="left")
        .merge(superpac_key, on="CAND_ID", how="left")
    )

    # Fill numeric columns
    support_cols = [
        "INDIVIDUAL_SUPPORT",
        "CORP_PAC_SUPPORT",
        "NONCONNECTED_PAC_SUPPORT",
        "SUPERPAC_IE_SUPPORT",
    ]
    for col in support_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    merged["TOTAL_SUPPORT"] = merged[support_cols].sum(axis=1)
    merged["HAS_MONEY"] = (merged["TOTAL_SUPPORT"] > 0).astype(int)

    # ---------- SORTING RULE ----------
    # Alphabetical by state, then descending by money
    sort_cols = ["CAND_OFFICE_ST", "TOTAL_SUPPORT"]
    sort_orders = [True, False]

    merged_sorted = merged.sort_values(sort_cols, ascending=sort_orders)

    with_money = merged_sorted[merged_sorted["HAS_MONEY"] == 1].copy()
    no_money = merged_sorted[merged_sorted["HAS_MONEY"] == 0].copy()

    # Outputs
    out_with_money = OUT_DIR / "final_support_table.csv"
    out_no_money = OUT_DIR / "candidates_no_support.csv"
    out_all_flag = OUT_DIR / "candidates_all_with_flag.csv"

    with_money.to_csv(out_with_money, index=False)
    no_money.to_csv(out_no_money, index=False)
    merged_sorted.to_csv(out_all_flag, index=False)

    print("[merge_support] Wrote:")
    print("  ", out_with_money)
    print("  ", out_no_money)
    print("  ", out_all_flag)

    print(
        f"[merge_support] Counts: "
        f"with_money={len(with_money):,} | "
        f"no_money={len(no_money):,} | "
        f"total={len(merged_sorted):,}"
    )

    print("\n[merge_support] Preview (AL → AK → …):")
    print(with_money.head(25).to_string(index=False))

if __name__ == "__main__":
    main()
