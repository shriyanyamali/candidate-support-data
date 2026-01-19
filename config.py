## 00

from pathlib import Path
import re

# ---- EDIT THESE TWO LINES ONLY ----
BASE_DIR = Path(r"C:\Users\sruja\Downloads\Data Collection\FEC_Data")
CYCLE_LABEL = "16"   # e.g., "02", "04", "18", "20" (will expand to 2001_2002, 2003_2004, etc.)
# ----------------------------------

def _expand_cycle_label(label: str) -> str:
    """
    Convert short form to full form:
    "02" -> "2001_2002"
    "04" -> "2003_2004"
    "18" -> "2017_2018"
    "20" -> "2019_2020"
    """
    label = label.strip()
    
    # If already in full format (YYYY_YYYY), return as-is
    if re.match(r"^\d{4}_\d{4}$", label):
        return label
    
    # If in short format (XX), expand it
    if re.match(r"^\d{2}$", label):
        suffix = int(label)
        even_year = 2000 + suffix
        odd_year = even_year - 1
        return f"{odd_year}_{even_year}"
    
    raise ValueError(f"CYCLE_LABEL must be either 'XX' (e.g., '02', '04') or 'YYYY_YYYY'. Got: {label}")

def _cycle_suffix(cycle_label: str) -> str:
    """
    Extract the two-digit suffix from the cycle label.
    "2001_2002" -> "02"
    "2003_2004" -> "04"
    """
    m = re.match(r"^\s*(\d{4})\s*_\s*(\d{4})\s*$", cycle_label)
    if not m:
        raise ValueError(f"Invalid cycle format: {cycle_label}")
    end_year = int(m.group(2))
    return f"{end_year % 100:02d}"

# Expand the cycle label if needed
CYCLE_LABEL = _expand_cycle_label(CYCLE_LABEL)
SUFFIX = _cycle_suffix(CYCLE_LABEL)
TARGET_ELECTION_YR = str(int("20" + SUFFIX))  # "06"->"2006", "16"->"2016"

CYCLE_DIR = BASE_DIR / CYCLE_LABEL
CODE_DIR = BASE_DIR / "Code"

# Input folders (auto-select based on SUFFIX)
CM_DIR = CYCLE_DIR / f"cm{SUFFIX}"
CN_DIR = CYCLE_DIR / f"cn{SUFFIX}"
CCL_DIR = CYCLE_DIR / f"ccl{SUFFIX}"
INDIV_DIR = CYCLE_DIR / f"indiv{SUFFIX}"
PAS2_DIR = CYCLE_DIR / f"pas2{SUFFIX}"

# Output folder (always inside cycle folder)
OUT_DIR = CYCLE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Behavior
VALID_OFFICES = {"S", "P"}    # ✅ Senate + Presidential only (no House)
CHUNKSIZE = 2_000_000

# ---- File schemas ----
CM_COLS = [
    "CMTE_ID","CMTE_NM","TRES_NM","CMTE_ST1","CMTE_ST2","CMTE_CITY","CMTE_ST",
    "CMTE_ZIP","CMTE_DSGN","CMTE_TP","CMTE_PTY_AFFILIATION","CMTE_FILING_FREQ",
    "ORG_TP","CONNECTED_ORG_NM","CAND_ID"
]
CN_COLS = [
    "CAND_ID","CAND_NAME","CAND_PTY_AFFILIATION","CAND_ELECTION_YR","CAND_OFFICE_ST",
    "CAND_OFFICE","CAND_OFFICE_DISTRICT","CAND_ICI","CAND_STATUS","CAND_PCC",
    "CAND_ST1","CAND_ST2","CAND_CITY","CAND_ST","CAND_ZIP"
]
CCL_COLS = [
    "CAND_ID","CAND_ELECTION_YR","FEC_ELECTION_YR","CMTE_ID","CMTE_TP","CMTE_DSGN","LINKAGE_ID"
]
INDIV_COLS = [
    "CMTE_ID","AMNDT_IND","RPT_TP","TRANSACTION_PGI","IMAGE_NUM","TRANSACTION_TP","ENTITY_TP",
    "NAME","CITY","STATE","ZIP_CODE","EMPLOYER","OCCUPATION","TRANSACTION_DT","TRANSACTION_AMT",
    "OTHER_ID","TRAN_ID","FILE_NUM","MEMO_CD","MEMO_TEXT","SUB_ID"
]
ITPAS2_COLS = [
    "CMTE_ID","AMNDT_IND","RPT_TP","TRANSACTION_PGI","IMAGE_NUM","TRANSACTION_TP","ENTITY_TP",
    "NAME","CITY","STATE","ZIP_CODE","EMPLOYER","OCCUPATION","TRANSACTION_DT","TRANSACTION_AMT",
    "OTHER_ID","CAND_ID","TRAN_ID","FILE_NUM","MEMO_CD","MEMO_TEXT","SUB_ID"
]

def write_csv_no_blank_line(df, path, **kwargs):
    """
    Write DataFrame to CSV without trailing blank line.
    """
    import pandas as pd
    df.to_csv(path, **kwargs)
    
    # Remove trailing blank lines
    with open(path, 'rb') as f:
        content = f.read()
    
    # Strip all trailing newlines
    content = content.rstrip(b'\r\n')
    
    with open(path, 'wb') as f:
        f.write(content)