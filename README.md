# FEC Campaign Finance Pipeline: Comprehensive Documentation

## Table of Contents

1. [Overview](#overview)
2. [Data Sources](#data-sources)
3. [What This Pipeline Does](#what-this-pipeline-does)
4. [Included Data](#included-data)
5. [Support Categories Explained](#support-categories-explained)
6. [How the Math Works](#how-the-math-works)
7. [Filters and Restrictions](#filters-and-restrictions)
8. [Output Files Structure](#output-files-structure)
9. [Column Definitions](#column-definitions)
10. [Installation and Setup](#installation-and-setup)
11. [How to Use](#how-to-use)
12. [Validation](#validation)
13. [Known Limitations](#known-limitations)
14. [Methodology Notes](#methodology-notes)

---

## Overview

This pipeline processes **FEC bulk transaction data** to create **candidate-level campaign finance datasets** for Senate and Presidential elections. It aggregates individual contributions, PAC contributions, and independent expenditures into clean, non-overlapping support categories. This repository already contains the extracted data for the 2001-2002, 2003-2004,...,2017-2018, 2019-2020 data sets

**Key Features:**
- Transaction-level data
- Non-overlapping support categories
- Separate outputs for Senate, Presidential, and totals
- Election-year restricted
- Fully reproducible from FEC bulk files

---

| Cycle Years | CYCLE_LABEL | TARGET_ELECTION_YR | Elections |
|-------------|-------------|-------------------|-----------|
| 2001-2002 | `02` | `2002` | Senate midterm |
| 2003-2004 | `04` | `2004` | Presidential + Senate |
| 2005-2006 | `06` | `2006` | Senate midterm |
| 2007-2008 | `08` | `2008` | Presidential + Senate |
| 2009-2010 | `10` | `2010` | Senate midterm |
| 2011-2012 | `12` | `2012` | Presidential + Senate |
| 2013-2014 | `14` | `2014` | Senate midterm |
| 2015-2016 | `16` | `2016` | Presidential + Senate |
| 2017-2018 | `18` | `2018` | Senate midterm |
| 2019-2020 | `20` | `2020` | Presidential + Senate |
| 2021-2022 | `22` | `2022` | Senate midterm |
| 2023-2024 | `24` | `2024` | Presidential + Senate |

---

## Data Sources

All data comes from the **Federal Election Commission (FEC) bulk data portal**:

**Source URL:** https://www.fec.gov/data/browse-data/?tab=bulk-data

### Required FEC Files (by cycle)

For each election cycle (e.g., 2015-2016 for the "16" cycle), download these files:

| File Type | FEC Name | Purpose | Our Usage |
|-----------|----------|---------|-----------|
| **Candidate Master** | `cn##.zip` | Candidate information, election year, office | Defines candidate universe, filters by office and year |
| **Committee Master** | `cm##.zip` | Committee information, types | Identifies PAC types (C=corporate, N=nonconnected) and Super PACs (O=IE-only) |
| **Candidate-Committee Linkages** | `ccl##.zip` | Links committees to candidates | Maps individual contributions (committee → candidate) |
| **Individual Contributions** | `indiv##.zip` | Itemized individual donations | Individual support category |
| **Committee Contributions & IEs** | `pas2##.zip` | PAC-to-candidate contributions and independent expenditures | PAC support and Super PAC IE support |

**Example for 2015-2016:**
- `cn16.zip` → Candidate master
- `cm16.zip` → Committee master
- `ccl16.zip` → Candidate-committee linkages
- `indiv16.zip` (also called `itcont16.zip`) → Individual contributions
- `pas216.zip` (also called `itpas216.zip`) → All other contributions and IEs

### File Structure After Download

```
FEC_Data/
└── 2015_2016/
    ├── cn16/
    │   └── cn.txt
    ├── cm16/
    │   └── cm.txt
    ├── ccl16/
    │   └── ccl.txt
    ├── indiv16/
    │   └── itcont.txt
    └── pas216/
        └── itpas2.txt
```

**Note:** The FEC uses `##` to denote the 2-digit cycle year (e.g., `16` for 2015-2016).

---

## What This Pipeline Does

### Input → Processing → Output

```
FEC Bulk Files (5 files)
         ↓
    Pipeline Processing
    - Filter candidates (Senate & Presidential only)
    - Filter to target election year
    - Classify contributions by type
    - Remove duplicates
    - Aggregate to candidate level
         ↓
    18 Output Files
    - 6 for Senate
    - 6 for Presidential  
    - 6 for Total (combined)
```

### Processing Steps

1. **Load Candidate Universe**
   - Read `cn.txt` (candidate master)
   - Filter to Senate (`CAND_OFFICE = 'S'`) and Presidential (`CAND_OFFICE = 'P'`)
   - Filter to target election year (e.g., `CAND_ELECTION_YR = 2016`)
   - Remove duplicate candidate records (keep best administrative record)

2. **Classify Committees**
   - Read `cm.txt` (committee master)
   - Identify Super PACs: `CMTE_TP = 'O'` (IE-only committees)
   - Identify regular PACs: `CMTE_TP IN ('Q', 'N')` (qualified/nonqualified)
   - Within PACs, classify by `ORG_TP`:
     - `'C'` = Corporate-connected
     - `''` (blank) = Nonconnected

3. **Process Individual Contributions**
   - Read `itcont.txt` (individual contributions)
   - Filter: `TRANSACTION_TP = '15'` AND `ENTITY_TP = 'IND'`
   - Map committee → candidate using `ccl.txt`
   - Sum by candidate

4. **Process PAC Contributions**
   - Read `itpas2.txt` (PAC contributions and IEs)
   - Exclude independent expenditures (`TRANSACTION_TP NOT IN ('24E', '24A')`)
   - Filter to PAC committees only
   - Split by corporate (`ORG_TP = 'C'`) vs nonconnected (`ORG_TP = ''`)
   - Sum by candidate

5. **Process Super PAC Independent Expenditures**
   - Read `itpas2.txt` again
   - Filter: `TRANSACTION_TP = '24E'` AND committee in Super PAC list
   - Sum by candidate

6. **Merge and Calculate Totals**
   - Join all support categories on candidate ID
   - Calculate `TOTAL_SUPPORT` = sum of all categories
   - Create flags and split into final output files

---

## Included Data

### Included

#### Candidates
- **Senate candidates** with `CAND_OFFICE = 'S'`
- **Presidential candidates** with `CAND_OFFICE = 'P'`
- Only candidates with `CAND_ELECTION_YR` matching target year (e.g., 2016)

#### Transactions
- **Individual contributions** (transaction type 15, individual entities)
- **Corporate PAC contributions** (from PACs with `ORG_TP = 'C'`)
- **Nonconnected PAC contributions** (from PACs with `ORG_TP = ''`)
- **Super PAC independent expenditures** (transaction type 24E, from IE-only committees)

#### Support
- All positive transaction amounts (`TRANSACTION_AMT > 0`)
- Contributions to principal campaign committees
- Contributions to authorized committees linked via `ccl`
- Independent expenditures supporting the candidate

### Excluded

#### Candidates
- **House candidates** (`CAND_OFFICE = 'H'`)
- Candidates from other election years
- Candidates with post-election activity but who didn't run in target year

#### Transactions
- **Negative amounts** (refunds, adjustments)
- **Zero amounts**
- **Contributions to parties** (not candidate-specific)
- **Coordinated expenditures** (transaction type 24K and others)
- **Non-independent expenditures** from Super PACs
- **Loans**
- **In-kind contributions** (unless itemized as contributions)
- **Party committee transfers**

#### Committees
- Party committees (e.g., DNC, RNC)
- Leadership PACs (unless contributing to candidates)
- Joint fundraising committees (contributions attributed to underlying committees)
- Unlinked committees (committees not connected to any candidate)

### Individual Contributions (itcont.txt)

| Type | Description | Included? |
|------|-------------|-----------|
| 15 | Contribution | ✅ Yes |
| 15E | Earmarked contribution | ❌ No (handled separately) |
| 15J | JFC contribution | ❌ No (complex attribution) |

### PAC/Committee Transactions (itpas2.txt)

| Type | Description | Included? |
|------|-------------|-----------|
| 24A | Independent expenditure (against) | ❌ No |
| 24E | Independent expenditure (for) | ✅ Yes (Super PAC IE only) |
| 24F | Communication cost | ❌ No |
| 24K | Direct contribution (coordinated) | ❌ No |
| 24N | Electioneering communication | ❌ No |
| 24R | Electioneering communication (request) | ❌ No |
| 24Z | In-kind contribution | ❌ No (unless coded differently) |

## Support Categories Explained

### 1. Individual Support (`INDIVIDUAL_SUPPORT`)

**What it includes:**
- Direct donations from individuals to candidate committees
- Small-dollar contributions
- Large individual donations (up to legal limit)

**FEC Criteria:**
- `TRANSACTION_TP = '15'` (contribution from individual)
- `ENTITY_TP = 'IND'` (entity is an individual)
- Committee is linked to candidate via `ccl` file

**Example:**
- John Doe donates $2,700 to Marco Rubio's campaign
- This appears in `itcont.txt` with transaction type 15
- Rubio's committee ID is linked to his candidate ID in `ccl.txt`
- $2,700 added to Rubio's `INDIVIDUAL_SUPPORT`

**NOT included:**
- Contributions from PACs (even if sourced from individuals)
- Candidate self-financing through committees
- Party committee transfers

---

### 2. Corporate PAC Support (`CORP_PAC_SUPPORT`)

**What it includes:**
- Contributions from corporate-connected PACs
- Contributions from trade association PACs
- Contributions from membership organization PACs with corporate ties

**FEC Criteria:**
- Committee has `CMTE_TP IN ('Q', 'N')` (PAC designation)
- Committee has `ORG_TP = 'C'` (corporate-connected)
- `TRANSACTION_TP` is a contribution type (NOT 24E or 24A)
- Recipient is a candidate (has `CAND_ID`)

**Example:**
- AT&T's PAC donates $5,000 to Pat Toomey's campaign
- AT&T PAC has `CMTE_TP = 'Q'` and `ORG_TP = 'C'`
- $5,000 added to Toomey's `CORP_PAC_SUPPORT`

**Key distinction:**
- These are **direct contributions** to candidates
- Subject to $5,000 per election limit
- Cannot be independent expenditures

---

### 3. Nonconnected PAC Support (`NONCONNECTED_PAC_SUPPORT`)

**What it includes:**
- Contributions from ideological PACs (e.g., Emily's List)
- Contributions from issue-based PACs
- Contributions from PACs without corporate/union affiliation

**FEC Criteria:**
- Committee has `CMTE_TP IN ('Q', 'N')` (PAC designation)
- Committee has `ORG_TP = ''` (blank = nonconnected)
- `TRANSACTION_TP` is a contribution type (NOT 24E or 24A)
- Recipient is a candidate

**Example:**
- Emily's List donates $5,000 to Kelly Ayotte's campaign
- Emily's List has `CMTE_TP = 'Q'` and `ORG_TP = ''`
- $5,000 added to Ayotte's `NONCONNECTED_PAC_SUPPORT`

**Key distinction:**
- Not affiliated with corporations or unions
- Often ideological or issue-based
- Subject to same $5,000 limit as corporate PACs

**Note on Union PACs:**
- Union-connected PACs would have `ORG_TP = 'L'` or `'M'`
- This pipeline does **not** currently have a separate union category
- Union PACs would need to be identified with additional filtering

---

### 4. Super PAC Independent Expenditure Support (`SUPERPAC_IE_SUPPORT`)

**What it includes:**
- Independent expenditures **supporting** the candidate
- Spending by Super PACs (IE-only committees)
- Ad buys, mailers, and other communications

**FEC Criteria:**
- Committee has `CMTE_TP = 'O'` (independent expenditure-only)
- `TRANSACTION_TP = '24E'` (independent expenditure)
- `CAND_ID` links to a candidate

**Example:**
- Priorities USA (Super PAC) spends $1,000,000 on TV ads supporting Hillary Clinton
- Priorities USA has `CMTE_TP = 'O'`
- Transaction has `TRANSACTION_TP = '24E'` and `CAND_ID = Clinton's ID`
- $1,000,000 added to Clinton's `SUPERPAC_IE_SUPPORT`

**Key distinctions:**
- **Independent** = no coordination with candidate
- **Unlimited** = no contribution limits
- Only from "IE-only" Super PACs (not hybrid PACs)
- Includes both pro-candidate and anti-opponent spending (if coded to candidate)

**What this does NOT include:**
- Spending by hybrid PACs (they contribute directly, not IEs)
- Party committee IEs
- Non-Super PAC independent expenditures
- Electioneering communications (different transaction type)

---

### 5. Total Support (`TOTAL_SUPPORT`)

**Calculation:**
```
TOTAL_SUPPORT = INDIVIDUAL_SUPPORT 
              + CORP_PAC_SUPPORT 
              + NONCONNECTED_PAC_SUPPORT 
              + SUPERPAC_IE_SUPPORT
```

**Properties:**
- Non-overlapping categories (no double-counting)
- Comprehensive coverage of major funding sources
- Excludes party transfers, loans, and other non-contribution funding

---

## How the Math Works

### Aggregation Process

#### Step 1: Transaction-Level Filtering
```python
# Example for individual contributions
transactions = read_file("itcont.txt")
filtered = transactions[
    (transactions['TRANSACTION_TP'] == '15') &
    (transactions['ENTITY_TP'] == 'IND') &
    (transactions['TRANSACTION_AMT'] > 0)
]
```

#### Step 2: Map to Candidates
```python
# Link committee → candidate
candidate_map = build_map_from_ccl()
filtered['CAND_ID'] = filtered['CMTE_ID'].map(candidate_map)
```

#### Step 3: Sum by Candidate
```python
# Group by candidate and sum
support = filtered.groupby('CAND_ID')['TRANSACTION_AMT'].sum()
```

#### Step 4: Repeat for Each Category
- Same process for corporate PAC, nonconnected PAC, and Super PAC
- Each creates a separate support column

#### Step 5: Merge All Categories
```python
final = candidates.merge(individual_support, on='CAND_ID', how='left')
                  .merge(corp_pac_support, on='CAND_ID', how='left')
                  .merge(nonconn_pac_support, on='CAND_ID', how='left')
                  .merge(superpac_ie_support, on='CAND_ID', how='left')
```

#### Step 6: Fill Missing Values
```python
# Candidates with no support in a category get 0
support_columns.fillna(0, inplace=True)
```

#### Step 7: Calculate Total
```python
final['TOTAL_SUPPORT'] = (
    final['INDIVIDUAL_SUPPORT'] +
    final['CORP_PAC_SUPPORT'] +
    final['NONCONNECTED_PAC_SUPPORT'] +
    final['SUPERPAC_IE_SUPPORT']
)
```

### Handling Duplicates

#### Problem: Multiple Administrative Records
Some candidates have multiple rows in `cn.txt` for the same election.

**Solution:**
1. Sort by preference: Principal committee > No committee
2. Sort by status: Active > Other statuses  
3. Keep first record per (CAND_ID, CAND_ELECTION_YR)

#### Problem: Duplicate Support Entries
Some transactions may appear multiple times (amendments, etc.)

**Solution:**
1. Group by (CAND_ID, CAND_ELECTION_YR)
2. **Sum** all support values (don't drop duplicates)
3. Ensures no money is lost

### Handling Missing Data

- **Missing candidate info:** Candidate excluded from output
- **Missing committee link:** Transaction ignored (can't attribute to candidate)
- **Missing transaction amount:** Treated as 0
- **Missing organization type:** Treated as nonconnected PAC

---

## Filters and Restrictions

### Office Filter

**Applied to:** Candidate master (`cn.txt`)

**Logic:**
```python
VALID_OFFICES = {'S', 'P'}  # Senate and Presidential only
candidates = candidates[candidates['CAND_OFFICE'].isin(VALID_OFFICES)]
```

**Effect:**
- House candidates (`CAND_OFFICE = 'H'`) excluded entirely
- Only Senate and Presidential races analyzed

**Rationale:**
- House has 435 seats vs. 35 Senate seats and 1 Presidential
- House races are structurally different (smaller scale, local)
- Keeps dataset focused and manageable

---

### Election Year Filter

**Applied to:** Candidate master (`cn.txt`)

**Logic:**
```python
TARGET_ELECTION_YR = '2016'  # Set in config based on cycle
candidates = candidates[candidates['CAND_ELECTION_YR'] == TARGET_ELECTION_YR]
```

**Effect:**
- Only candidates **running in that specific election** included
- Excludes candidates who ran in other years
- Excludes candidates with post-election administrative activity

**Example of what's excluded:**
- Dan Sullivan ran for Senate in **2014**
- His committee filed amendments in **2016**
- He appears in 2015-2016 bulk files
- But he's **excluded** from 2016 dataset because `CAND_ELECTION_YR = 2014`

**Rationale:**
- Creates clean election-specific datasets
- Prevents conflating election participation with reporting activity
- Enables valid election-to-election comparisons

---

### Transaction Type Filters

**Individual Contributions:**
```python
TRANSACTION_TP == '15'   # Contribution
ENTITY_TP == 'IND'       # From individual
```

**PAC Contributions:**
```python
CMTE_TP IN ('Q', 'N')           # Committee is a PAC
TRANSACTION_TP NOT IN ('24E', '24A')  # Not an IE
```

**Super PAC IEs:**
```python
CMTE_TP == 'O'          # IE-only committee
TRANSACTION_TP == '24E'  # Independent expenditure
```

---

### Amount Filters

**All categories:**
```python
TRANSACTION_AMT > 0  # Positive amounts only
```

**Effect:**
- Refunds (negative amounts) excluded
- Adjustments (negative) excluded
- Zero-dollar transactions excluded

**Rationale:**
- Focus on actual financial support flowing **to** candidates
- Negative amounts represent money flowing **away** (corrections, refunds)
- Including negatives would understate support

---

### Committee Linkage Filter

**Applied to:** Individual contributions

**Logic:**
```python
# Only include if committee is linked to a candidate
transactions = transactions[transactions['CMTE_ID'].isin(linked_committees)]
```

**Effect:**
- Contributions to unlinked committees excluded
- Contributions to party committees excluded
- Only candidate-committee contributions counted

**Rationale:**
- Can only attribute support if committee-candidate link exists
- Prevents including party/other committee fundraising
- Ensures support is candidate-specific

---

## Output Files Structure

### Directory Organization

```
outputs/
├── senate/
│   ├── senate_superpac_ie_support_##.csv
│   ├── senate_individual_support_##.csv
│   ├── senate_pac_support_corp_nonconnected_##.csv
│   ├── senate_final_support_table_##.csv
│   ├── senate_candidates_no_support_##.csv
│   └── senate_candidates_all_with_flag_##.csv
│
├── presidential/
│   ├── presidential_superpac_ie_support_##.csv
│   ├── presidential_individual_support_##.csv
│   ├── presidential_pac_support_corp_nonconnected_##.csv
│   ├── presidential_final_support_table_##.csv
│   ├── presidential_candidates_no_support_##.csv
│   └── presidential_candidates_all_with_flag_##.csv
│
└── total/
    ├── total_superpac_ie_support_##.csv
    ├── total_individual_support_##.csv
    ├── total_pac_support_corp_nonconnected_##.csv
    ├── total_final_support_table_##.csv
    ├── total_candidates_no_support_##.csv
    └── total_candidates_all_with_flag_##.csv
```

**Total: 18 output files** (6 per office + total type)

---

### File Descriptions

#### 1. `{prefix}_superpac_ie_support_{cycle}.csv`

**Purpose:** Intermediate file showing Super PAC IE support per candidate

**Columns:**
- `CAND_ID`: FEC candidate ID
- `SUPERPAC_IE_SUPPORT`: Total Super PAC independent expenditures
- `CAND_NAME`: Candidate name
- `CAND_PTY_AFFILIATION`: Party (DEM, REP, etc.)
- `CAND_ELECTION_YR`: Election year
- `CAND_OFFICE_ST`: State (for Senate) or blank (for Presidential)
- `CAND_OFFICE`: Office (S or P)
- Other candidate fields from `cn.txt`

---

#### 2. `{prefix}_individual_support_{cycle}.csv`

**Purpose:** Intermediate file showing individual contribution support per candidate

**Columns:**
- `CAND_ID`: FEC candidate ID
- `INDIVIDUAL_SUPPORT`: Total individual contributions
- Candidate info fields (name, party, state, etc.)

---

#### 3. `{prefix}_pac_support_corp_nonconnected_{cycle}.csv`

**Purpose:** Intermediate file showing PAC support per candidate (split by type)

**Columns:**
- `CAND_ID`: FEC candidate ID
- `CORP_PAC_SUPPORT`: Total corporate PAC contributions
- `NONCONNECTED_PAC_SUPPORT`: Total nonconnected PAC contributions
- Candidate info fields

---

#### 4. `{prefix}_final_support_table_{cycle}.csv` ⭐ **PRIMARY OUTPUT**

**Purpose:** **Main analysis file** - Complete support data for candidates who received money

**Columns:**
- `CAND_ID`: FEC candidate ID
- `CAND_ELECTION_YR`: Election year (e.g., 2016)
- `CAND_NAME`: Candidate name
- `CAND_PTY_AFFILIATION`: Party affiliation
- `CAND_OFFICE`: Office sought (S or P)
- `CAND_OFFICE_ST`: State (for Senate) or blank (Presidential)
- `INDIVIDUAL_SUPPORT`: Individual contributions total
- `CORP_PAC_SUPPORT`: Corporate PAC contributions total
- `NONCONNECTED_PAC_SUPPORT`: Nonconnected PAC contributions total
- `SUPERPAC_IE_SUPPORT`: Super PAC IE total
- `TOTAL_SUPPORT`: Sum of all support categories
- `HAS_MONEY`: 1 (always 1 in this file)

**Filtering:**
- Only candidates with `TOTAL_SUPPORT > 0`
- Sorted by state and total support (descending)

**Example rows (Senate 2016):**
```
CAND_ID      CAND_NAME              CAND_OFFICE  CAND_OFFICE_ST  TOTAL_SUPPORT  INDIVIDUAL_SUPPORT  CORP_PAC_SUPPORT
S0FL00338    RUBIO, MARCO           S            FL              24,785,695     18,234,521          2,456,789
S4PA00121    TOOMEY, PATRICK JOSEPH S            PA              24,075,292     16,890,443          3,112,654
```

---

#### 5. `{prefix}_candidates_no_support_{cycle}.csv`

**Purpose:** Candidates who ran but received zero financial support

**Columns:**
- Same as `final_support_table`
- All support columns = 0
- `HAS_MONEY` = 0

**Who uses this:**
- Researchers studying non-viable candidates
- Completeness checking
- Understanding full candidate field

**Filtering:**
- Only candidates with `TOTAL_SUPPORT = 0`

**Typical row count:**
- Senate: ~20-50 (fringe/late withdrawal candidates)
- Presidential: ~5-15 (fringe candidates)

**Why these candidates exist:**
- Filed with FEC but never fundraised
- Withdrew before raising money
- Very late entry candidates
- Fringe/protest candidates

---

#### 6. `{prefix}_candidates_all_with_flag_{cycle}.csv`

**Purpose:** Complete candidate universe (funded + unfunded)

**Columns:**
- Same as `final_support_table`
- `HAS_MONEY`: 1 if funded, 0 if unfunded

**Who uses this:**
- Researchers needing complete candidate counts
- Studies of candidate entry/viability
- Denominator for "% of candidates who raised money"

**Filtering:**
- All candidates from target election year and office

**Typical row count:**
- Senate: ~180-220 total candidates
- Presidential: ~30-50 total candidates

**Relationship:**
```
candidates_all = final_support_table + candidates_no_support
```

---

## Column Definitions

### Candidate Identification Columns

**`CAND_ID`** (string)
- FEC candidate identification number
- Format: `[S|P|H][0-9][A-Z]{2}[0-9]{5}`
- Example: `S0FL00338` (Marco Rubio)
- **Primary key** (unique per candidate per election year)

**`CAND_NAME`** (string)
- Candidate full name (LAST, FIRST format)
- Example: `RUBIO, MARCO`

**`CAND_ELECTION_YR`** (integer as string)
- Year of the election the candidate is running in
- Format: `YYYY` (e.g., `2016`)
- **Important:** This is the election year, not the file vintage year

---

### Candidate Attributes

**`CAND_PTY_AFFILIATION`** (string)
- Three-letter party code
- Common values: `DEM`, `REP`, `LIB`, `GRE`, `IND`
- May be blank for some candidates

**`CAND_OFFICE`** (string)
- Office sought
- Values in outputs: `S` (Senate) or `P` (Presidential)
- `H` (House) excluded by design

**`CAND_OFFICE_ST`** (string)
- Two-letter state code for Senate candidates
- Examples: `FL`, `PA`, `NY`, `CA`
- Blank for Presidential candidates
- `US` may appear for Presidential in some FEC files

**`CAND_OFFICE_DISTRICT`** (string)
- Congressional district (House only)
- Always `00` or blank for Senate/Presidential

**`CAND_ICI`** (string)
- Incumbent/Challenger/Open status
- Values: `I` (Incumbent), `C` (Challenger), `O` (Open seat)
- May be blank

**`CAND_STATUS`** (string)
- Candidate status code
- Common values: `C` (Continuing), `F` (Future), `N` (Not yet a candidate)
- Used internally to resolve duplicate records

**`CAND_PCC`** (string)
- Principal Campaign Committee ID
- Links to committee in `cm.txt`
- Format: `C[0-9]{8}`
- May be blank if no PCC

---

### Support Columns (all in US dollars)

**`INDIVIDUAL_SUPPORT`** (float)
- Total individual contributions to candidate
- Source: `itcont.txt`, transaction type 15
- Typical range: $0 to $50,000,000
- Includes all individual donors

**`CORP_PAC_SUPPORT`** (float)
- Total corporate PAC contributions to candidate
- Source: `itpas2.txt`, PACs with `ORG_TP = 'C'`
- Typical range: $0 to $5,000,000
- Limited to $5,000 per PAC per election

**`NONCONNECTED_PAC_SUPPORT`** (float)
- Total nonconnected PAC contributions to candidate
- Source: `itpas2.txt`, PACs with `ORG_TP = ''`
- Typical range: $0 to $3,000,000
- Includes ideological and issue PACs

**`SUPERPAC_IE_SUPPORT`** (float)
- Total Super PAC independent expenditures supporting candidate
- Source: `itpas2.txt`, transaction type 24E from IE-only committees
- Typical range: $0 to $150,000,000 (Presidential) or $20,000,000 (Senate)
- No legal limits

**`TOTAL_SUPPORT`** (float)
- Sum of all four support categories
- Calculation: `INDIVIDUAL + CORP_PAC + NONCONNECTED_PAC + SUPERPAC_IE`
- Typical range: $0 to $350,000,000 (top Presidential candidates)

---

### Flags and Indicators

**`HAS_MONEY`** (integer: 0 or 1)
- Indicates whether candidate received any financial support
- `1` = `TOTAL_SUPPORT > 0`
- `0` = `TOTAL_SUPPORT = 0`
- Used to split files (final vs. no_support)

---

### Less Common Columns (in some files)

Additional columns from `cn.txt` may appear in output:

- `CAND_ST1`, `CAND_ST2`: Candidate mailing address
- `CAND_CITY`, `CAND_ST`, `CAND_ZIP`: Candidate location
- `TRES_NM`: Treasurer name (in some merge scenarios)

These are generally not used for analysis but are retained for reference.

---

## Installation and Setup

### Prerequisites

**Software:**
- Python 3.8 or higher
- pip (Python package manager)

**Python packages:**
```bash
pip install pandas
```

**Disk space:**
- ~10 GB per cycle for raw FEC files (compressed)
- ~20 GB per cycle for unzipped files
- ~1 MB for outputs

---

### Setup Steps

#### 1. Clone or Download Pipeline

```bash
git clone https://github.com/shriyanyamali/fec-cn-support-etl.git
cd fec-cn-support-etl
```

---

#### 2. Create Directory Structure

```bash
mkdir -p FEC_Data/2015_2016/{cn16,cm16,ccl16,indiv16,pas216,outputs,Code}
```

For other cycles, adjust the year and cycle number (e.g., `2019_2020` and `20`).

---

#### 3. Download FEC Data

Visit: https://www.fec.gov/data/browse-data/?tab=bulk-data

**For 2015-2016 cycle:**

1. Download `cn16.zip` → Extract to `FEC_Data/2015_2016/cn16/`
2. Download `cm16.zip` → Extract to `FEC_Data/2015_2016/cm16/`
3. Download `ccl16.zip` → Extract to `FEC_Data/2015_2016/ccl16/`
4. Download `indiv16.zip` → Extract to `FEC_Data/2015_2016/indiv16/`
   - May be named `itcont16.zip`
5. Download `pas216.zip` → Extract to `FEC_Data/2015_2016/pas216/`
   - May be named `itpas216.zip`

**File naming:** The extracted `.txt` files are usually named after the file type (e.g., `cn.txt`, `cm.txt`). The pipeline auto-detects the largest `.txt` or `.dat` file in each directory.

---

#### 4. Place Code Files

Copy all `.py` files to `FEC_Data/Code/`:

```
FEC_Data/
└── Code/
    ├── config.py
    ├── superpac_ie_support.py
    ├── individual_support.py
    ├── pac_support_corp_union.py
    ├── merge_support.py
    ├── run_all.py
    ├── combine_csv.py
    └── validate_outputs.py
```

---

#### 5. Configure Pipeline

Edit `FEC_Data/Code/config.py`:

```python
BASE_DIR = Path(r"C:\Users\YourName\FEC_Data")  # Change this path
CYCLE_LABEL = "16"  # Two-digit cycle year
```

The pipeline will automatically:
- Set `TARGET_ELECTION_YR = 2016`
- Look for input folders: `cn16`, `cm16`, `ccl16`, `indiv16`, `pas216`
- Create output folders: `senate`, `presidential`, `total`

---

## How to Use

### Basic Usage

#### Run Complete Pipeline

```bash
cd FEC_Data/Code
python run_all.py
```

**What it does:**
1. Processes Senate candidates → `outputs/senate/`
2. Processes Presidential candidates → `outputs/presidential/`
3. Processes combined dataset → `outputs/total/`

**Expected runtime:** 10-30 minutes depending on hardware and cycle size

**Expected output:**
```
████████████████████████████████████████
█ PIPELINE: SENATE
████████████████████████████████████████
... [processing messages]
✓ SENATE pipeline completed successfully

████████████████████████████████████████
█ PIPELINE: PRESIDENTIAL  
████████████████████████████████████████
... [processing messages]
✓ PRESIDENTIAL pipeline completed successfully

████████████████████████████████████████
█ PIPELINE: TOTAL (SENATE + PRESIDENTIAL)
████████████████████████████████████████
... [processing messages]
✓ TOTAL pipeline completed successfully

█ ALL PIPELINES COMPLETED SUCCESSFULLY
```

---

#### Validate Outputs

```bash
python validate_outputs.py
```

**What it checks:**
- All 18 files exist
- No duplicates
- Office filters correct
- Totals calculated correctly
- Senate + Presidential = Total

**Expected output:**
```
✅ ALL VALIDATIONS PASSED
```

See validation section below for details.

---

#### Combine Multiple Cycles

If you have processed multiple cycles (e.g., 2012, 2014, 2016), combine them:

```bash
# Move final files to a central location first
mkdir FEC_Data/final_output_files
cp 2015_2016/outputs/senate/senate_final_support_table_16.csv final_output_files/
cp 2013_2014/outputs/senate/senate_final_support_table_14.csv final_output_files/
# ... etc.

# Combine all files
python combine_csv.py --input-dir final_output_files --output combined_all_cycles.csv --recursive
```

## Validation

### Automated Validation

Run the validation script after every pipeline execution:

```bash
python validate_outputs.py
```

#### Validation Checks Performed

1. **File Existence** (18 files)
   - All expected output files present
   - Files are readable

2. **No Duplicates**
   - Each (CAND_ID, CAND_ELECTION_YR) appears exactly once
   - Checks all final, all, and no_support files

3. **Office Filter Accuracy**
   - Senate files contain only `CAND_OFFICE = 'S'`
   - Presidential files contain only `CAND_OFFICE = 'P'`
   - Total files contain both

4. **Election Year Filter**
   - All candidates have correct `CAND_ELECTION_YR`
   - No candidates from other years

5. **Total Calculations**
   - `TOTAL_SUPPORT` = sum of four categories
   - Difference < $0.01 (floating point tolerance)

6. **HAS_MONEY Flag**
   - Flag = 1 when `TOTAL_SUPPORT > 0`
   - Flag = 0 when `TOTAL_SUPPORT = 0`

7. **File Consistency**
   - `final + no_support = all` (for each office type)
   - Row counts match
   - Candidate IDs match

8. **Senate + Presidential = Total**
   - Row counts: Senate + Presidential = Total
   - Money totals: Senate + Presidential = Total
   - No overlap (no candidate in both files)

9. **Intermediate Files**
   - All candidate IDs in intermediate files appear in final files

10. **Sample Verification**
    - Displays top candidates for manual spot-checking

**Validation Output Example:**

```
✅ PASSED CHECKS (45):
  ✅ PASS: Found and loaded senate_final_support_table_16.csv (192 rows)
  ... [more checks]
  ✅ PASS: Senate (192) + Presidential (43) = Total (235)
  ✅ PASS: Support totals match: Senate ($453.7M) + Presidential ($854.3M) = Total ($1,308.0M)

ℹ️  INFORMATION (6):
  ℹ️  INFO: senate_final: Top candidate: RUBIO, MARCO ($24,785,695.00)
  ... [more info]

================================================================================
✅ ALL VALIDATIONS PASSED
================================================================================
```

---

### Manual Validation

#### Spot Check Known Candidates

For 2016 cycle, verify these well-known candidates:

**Presidential:**
- Hillary Clinton: `P00003392`
- Donald Trump: `P80001571`
- Bernie Sanders: `P60007168`

**Senate:**
- Marco Rubio (FL): `S0FL00338`
- Pat Toomey (PA): `S4PA00121`
- Rob Portman (OH): `S0OH00133`

**Check:**
```python
import pandas as pd

df = pd.read_csv("outputs/total/total_final_support_table_16.csv")

# Find Clinton
clinton = df[df['CAND_ID'] == 'P00003392']
print(clinton[['CAND_NAME', 'CAND_OFFICE', 'TOTAL_SUPPORT']])

# Should show:
# CAND_NAME: CLINTON, HILLARY RODHAM
# CAND_OFFICE: P
# TOTAL_SUPPORT: ~$300,000,000+
```

#### Verify Row Count Math

```python
senate = pd.read_csv("outputs/senate/senate_final_support_table_16.csv")
pres = pd.read_csv("outputs/presidential/presidential_final_support_table_16.csv")
total = pd.read_csv("outputs/total/total_final_support_table_16.csv")

print(f"Senate:       {len(senate):,}")  # e.g., 192
print(f"Presidential: {len(pres):,}")    # e.g., 43
print(f"Total:        {len(total):,}")   # Should be 192 + 43 = 235

assert len(senate) + len(pres) == len(total)
```

#### Check Support Breakdown

```python
# Verify TOTAL_SUPPORT calculation
df = pd.read_csv("outputs/senate/senate_final_support_table_16.csv")

calculated = (
    df['INDIVIDUAL_SUPPORT'] + 
    df['CORP_PAC_SUPPORT'] + 
    df['NONCONNECTED_PAC_SUPPORT'] + 
    df['SUPERPAC_IE_SUPPORT']
)

diff = (calculated - df['TOTAL_SUPPORT']).abs()
assert diff.max() < 0.01, f"Calculation error: max diff = ${diff.max():.2f}"
print("✅ Support breakdown verified")
```