import pandas as pd
import sys
from pathlib import Path

# Import config
try:
    from config import SENATE_OUT_DIR, PRESIDENTIAL_OUT_DIR, TOTAL_OUT_DIR, SUFFIX, TARGET_ELECTION_YR
except ImportError:
    print("ERROR: Could not import config. Make sure config.py is in the same directory.")
    sys.exit(1)


def verify_data():
    """Comprehensive data verification with detailed checks."""
    
    print("\n" + "█"*80)
    print("█ COMPREHENSIVE DATA VERIFICATION")
    print("█"*80)
    print(f"\nCycle: {SUFFIX}")
    print(f"Target Election Year: {TARGET_ELECTION_YR}")
    
    errors = []
    warnings = []
    info = []
    
    # Load data files
    print("\n" + "="*80)
    print("Loading data files...")
    print("="*80)
    
    try:
        df_total = pd.read_csv(TOTAL_OUT_DIR / f"total_final_support_table_{SUFFIX}.csv")
        df_senate = pd.read_csv(SENATE_OUT_DIR / f"senate_final_support_table_{SUFFIX}.csv")
        df_pres = pd.read_csv(PRESIDENTIAL_OUT_DIR / f"presidential_final_support_table_{SUFFIX}.csv")
        print(f"✅ Loaded all files successfully")
    except Exception as e:
        print(f"❌ ERROR loading files: {e}")
        return False
    
    # Test 1: Overall Statistics
    print("\n" + "="*80)
    print("[1/10] OVERALL STATISTICS")
    print("="*80)
    
    total_candidates = len(df_total)
    total_money = df_total['TOTAL_SUPPORT'].sum()
    mean_support = df_total['TOTAL_SUPPORT'].mean()
    median_support = df_total['TOTAL_SUPPORT'].median()
    max_support = df_total['TOTAL_SUPPORT'].max()
    
    print(f"\nTotal candidates:    {total_candidates:,}")
    print(f"Total money raised:  ${total_money:,.2f}")
    print(f"Mean support:        ${mean_support:,.2f}")
    print(f"Median support:      ${median_support:,.2f}")
    print(f"Max support:         ${max_support:,.2f}")
    
    # Expected ranges for 2016
    if SUFFIX == "16":
        if not (1_000_000_000 < total_money < 2_000_000_000):
            warnings.append(f"Total money ${total_money:,.0f} outside expected $1.0B-$2.0B range for 2016")
        else:
            info.append(f"Total money ${total_money:,.2f} within expected range")
        
        if not (150 < total_candidates < 300):
            warnings.append(f"Candidate count {total_candidates} outside expected 150-300 for 2016")
        else:
            info.append(f"Candidate count {total_candidates} within expected range")
    
    # Test 2: Support Breakdown
    print("\n" + "="*80)
    print("[2/10] SUPPORT BREAKDOWN")
    print("="*80)
    
    indiv_total = df_total['INDIVIDUAL_SUPPORT'].sum()
    corp_pac_total = df_total['CORP_PAC_SUPPORT'].sum()
    nonconn_pac_total = df_total['NONCONNECTED_PAC_SUPPORT'].sum()
    superpac_total = df_total['SUPERPAC_IE_SUPPORT'].sum()
    
    indiv_pct = (indiv_total / total_money * 100) if total_money > 0 else 0
    corp_pac_pct = (corp_pac_total / total_money * 100) if total_money > 0 else 0
    nonconn_pac_pct = (nonconn_pac_total / total_money * 100) if total_money > 0 else 0
    superpac_pct = (superpac_total / total_money * 100) if total_money > 0 else 0
    
    print(f"\nIndividual Support:       ${indiv_total:>15,.2f} ({indiv_pct:5.1f}%)")
    print(f"Corporate PAC Support:    ${corp_pac_total:>15,.2f} ({corp_pac_pct:5.1f}%)")
    print(f"Nonconnected PAC Support: ${nonconn_pac_total:>15,.2f} ({nonconn_pac_pct:5.1f}%)")
    print(f"Super PAC IE Support:     ${superpac_total:>15,.2f} ({superpac_pct:5.1f}%)")
    
    # Expected percentages for 2016
    if SUFFIX == "16":
        if not (60 < indiv_pct < 80):
            warnings.append(f"Individual support {indiv_pct:.1f}% outside expected 60-80% (2016)")
        else:
            info.append(f"Individual support {indiv_pct:.1f}% within expected range")
        
        if not (15 < superpac_pct < 35):
            warnings.append(f"Super PAC IE {superpac_pct:.1f}% outside expected 15-35% (2016)")
        else:
            info.append(f"Super PAC IE {superpac_pct:.1f}% within expected range")
        
        if corp_pac_pct + nonconn_pac_pct > 15:
            warnings.append(f"Total PAC support {corp_pac_pct + nonconn_pac_pct:.1f}% unexpectedly high (>15%)")
    
    # Test 3: Senate + Presidential = Total
    print("\n" + "="*80)
    print("[3/10] SENATE + PRESIDENTIAL = TOTAL")
    print("="*80)
    
    senate_count = len(df_senate)
    pres_count = len(df_pres)
    senate_money = df_senate['TOTAL_SUPPORT'].sum()
    pres_money = df_pres['TOTAL_SUPPORT'].sum()
    
    print(f"\nSenate:       {senate_count:>5,} candidates, ${senate_money:>15,.2f}")
    print(f"Presidential: {pres_count:>5,} candidates, ${pres_money:>15,.2f}")
    print(f"Total:        {total_candidates:>5,} candidates, ${total_money:>15,.2f}")
    
    # Check counts
    if senate_count + pres_count != total_candidates:
        errors.append(f"Row count mismatch: {senate_count} + {pres_count} = {senate_count + pres_count} ≠ {total_candidates}")
    else:
        info.append(f"Row counts match: {senate_count} + {pres_count} = {total_candidates}")
    
    # Check money
    money_diff = abs((senate_money + pres_money) - total_money)
    if money_diff > 0.01:
        errors.append(f"Money mismatch: ${senate_money:,.0f} + ${pres_money:,.0f} = ${senate_money + pres_money:,.0f} ≠ ${total_money:,.0f} (diff: ${money_diff:,.2f})")
    else:
        info.append(f"Money totals match (diff: ${money_diff:.6f})")
    
    # Check Presidential dominates (for Presidential years)
    pres_pct = (pres_money / total_money * 100) if total_money > 0 else 0
    print(f"\nPresidential: {pres_pct:.1f}% of total money")
    
    if SUFFIX in ["16", "20", "12", "08", "04"]:  # Presidential years
        if not (60 < pres_pct < 80):
            warnings.append(f"Presidential {pres_pct:.1f}% outside expected 60-80% for Presidential year")
        else:
            info.append(f"Presidential {pres_pct:.1f}% within expected range for Presidential year")
    
    # Test 4: Known Candidate Verification
    print("\n" + "="*80)
    print("[4/10] KNOWN CANDIDATE VERIFICATION")
    print("="*80)
    
    known_candidates = {
        "16": [
            ("P00003392", "Clinton", 200_000_000, 400_000_000, "Presidential"),
            ("P80001571", "Trump", 50_000_000, 200_000_000, "Presidential"),
            ("P60007168", "Sanders", 200_000_000, 250_000_000, "Presidential"),
            ("S0FL00338", "Rubio", 15_000_000, 30_000_000, "Senate"),
            ("S4PA00121", "Toomey", 15_000_000, 30_000_000, "Senate"),
        ],
        "20": [
            ("P00009795", "Biden", 800_000_000, 1_200_000_000, "Presidential"),
            ("P80001571", "Trump", 500_000_000, 900_000_000, "Presidential"),
        ],
        "14": [
            ("S4KY00249", "McConnell", 20_000_000, 35_000_000, "Senate"),
        ],
    }
    
    if SUFFIX in known_candidates:
        print(f"\nChecking known candidates for cycle {SUFFIX}:")
        
        for cand_id, name, min_exp, max_exp, office in known_candidates[SUFFIX]:
            cand = df_total[df_total['CAND_ID'] == cand_id]
            
            if cand.empty:
                errors.append(f"Known candidate {name} ({cand_id}) not found in output!")
                print(f"  ❌ {name} ({office}): NOT FOUND")
            else:
                amount = cand.iloc[0]['TOTAL_SUPPORT']
                
                if not (min_exp < amount < max_exp):
                    warnings.append(f"{name}: ${amount:,.0f} outside expected ${min_exp:,.0f}-${max_exp:,.0f}")
                    print(f"  ⚠️  {name} ({office}): ${amount:,.2f} (expected ${min_exp:,.0f}-${max_exp:,.0f})")
                else:
                    info.append(f"{name}: ${amount:,.2f} within expected range")
                    print(f"  ✅ {name} ({office}): ${amount:,.2f}")
    else:
        print(f"\nNo known candidates defined for cycle {SUFFIX}")
        info.append("No known candidate checks for this cycle")
    
    # Test 5: Check for Unexpected Zeros
    print("\n" + "="*80)
    print("[5/10] CHECKING FOR UNEXPECTED ZEROS")
    print("="*80)
    
    # Top 50 candidates shouldn't have zeros
    top_50 = df_total.nlargest(50, 'TOTAL_SUPPORT')
    zero_indiv = (top_50['INDIVIDUAL_SUPPORT'] == 0).sum()
    zero_total_in_top = (top_50['TOTAL_SUPPORT'] == 0).sum()
    
    print(f"\nTop 50 candidates:")
    print(f"  With $0 individual support: {zero_indiv}")
    print(f"  With $0 total support:      {zero_total_in_top}")
    
    if zero_total_in_top > 0:
        errors.append(f"{zero_total_in_top} candidates in top 50 have $0 total support")
    else:
        info.append("No top-50 candidates with $0 total")
    
    if zero_indiv > 5:
        warnings.append(f"{zero_indiv} top-50 candidates have $0 individual support (unusual)")
    elif zero_indiv > 0:
        info.append(f"{zero_indiv} top-50 candidates have $0 individual support (some candidates avoid small donations)")
    
    # Test 6: Calculation Accuracy
    print("\n" + "="*80)
    print("[6/10] CHECKING CALCULATION ACCURACY")
    print("="*80)
    
    calculated_total = (
        df_total['INDIVIDUAL_SUPPORT'] + 
        df_total['CORP_PAC_SUPPORT'] + 
        df_total['NONCONNECTED_PAC_SUPPORT'] + 
        df_total['SUPERPAC_IE_SUPPORT']
    )
    
    diff = (calculated_total - df_total['TOTAL_SUPPORT']).abs()
    max_diff = diff.max()
    num_diff = (diff > 0.01).sum()
    
    print(f"\nMax calculation difference: ${max_diff:.6f}")
    print(f"Rows with difference > $0.01: {num_diff}")
    
    if max_diff > 0.01:
        errors.append(f"Calculation errors found: max diff = ${max_diff:.2f} in {num_diff} rows")
    else:
        info.append(f"All calculations accurate (max diff: ${max_diff:.6f})")
    
    # Test 7: Duplicate Check
    print("\n" + "="*80)
    print("[7/10] CHECKING FOR DUPLICATES")
    print("="*80)
    
    dupes = df_total.duplicated(['CAND_ID', 'CAND_ELECTION_YR']).sum()
    
    print(f"\nDuplicate (CAND_ID, CAND_ELECTION_YR) combinations: {dupes}")
    
    if dupes > 0:
        errors.append(f"Found {dupes} duplicate candidate-year combinations")
    else:
        info.append("No duplicates found")
    
    # Test 8: Distribution Sanity
    print("\n" + "="*80)
    print("[8/10] CHECKING DISTRIBUTION")
    print("="*80)
    
    print(f"\nDistribution statistics:")
    print(f"  Mean:   ${mean_support:,.2f}")
    print(f"  Median: ${median_support:,.2f}")
    print(f"  Ratio:  {mean_support / median_support if median_support > 0 else 0:.2f}")
    
    # Should be right-skewed (mean > median)
    if mean_support <= median_support:
        warnings.append("Unusual distribution: mean ≤ median (expected right-skewed)")
    else:
        ratio = mean_support / median_support
        if ratio < 1.5:
            warnings.append(f"Low skew: mean/median = {ratio:.2f} (expected > 1.5 for campaign finance)")
        else:
            info.append(f"Distribution appropriately right-skewed (mean/median = {ratio:.2f})")
    
    # Check quantiles
    q25 = df_total['TOTAL_SUPPORT'].quantile(0.25)
    q75 = df_total['TOTAL_SUPPORT'].quantile(0.75)
    q95 = df_total['TOTAL_SUPPORT'].quantile(0.95)
    
    print(f"\nQuantiles:")
    print(f"  25th percentile: ${q25:,.2f}")
    print(f"  75th percentile: ${q75:,.2f}")
    print(f"  95th percentile: ${q95:,.2f}")
    
    # Test 9: Office-Specific Checks
    print("\n" + "="*80)
    print("[9/10] OFFICE-SPECIFIC CHECKS")
    print("="*80)
    
    # Check Senate candidates are in correct states
    senate_states = df_senate['CAND_OFFICE_ST'].nunique()
    print(f"\nSenate candidates in {senate_states} different states")
    
    if SUFFIX in ["16", "20", "12", "08"]:  # Presidential + ~34 Senate seats
        if not (25 < senate_states < 45):
            warnings.append(f"Senate state count {senate_states} outside expected 25-45")
    
    # Check Presidential candidates don't have state
    pres_with_state = df_pres[df_pres['CAND_OFFICE_ST'].notna() & (df_pres['CAND_OFFICE_ST'] != '')].shape[0]
    if pres_with_state > 0:
        warnings.append(f"{pres_with_state} Presidential candidates have state codes (should be blank)")
    else:
        info.append("No Presidential candidates have state codes")
    
    # Test 10: Top Candidates Check
    print("\n" + "="*80)
    print("[10/10] TOP CANDIDATES")
    print("="*80)
    
    print("\nTop 10 fundraisers:")
    top_10 = df_total.nlargest(10, 'TOTAL_SUPPORT')[['CAND_NAME', 'CAND_OFFICE', 'CAND_OFFICE_ST', 'TOTAL_SUPPORT']]
    for idx, row in top_10.iterrows():
        state = row['CAND_OFFICE_ST'] if pd.notna(row['CAND_OFFICE_ST']) else ''
        print(f"  {row['CAND_NAME']:30s} ({row['CAND_OFFICE']}-{state:2s}): ${row['TOTAL_SUPPORT']:>15,.2f}")
    
    # For 2016, top should include Clinton, Trump, Sanders, etc.
    if SUFFIX == "16":
        top_10_ids = set(top_10.index.map(lambda i: df_total.loc[i, 'CAND_ID']))
        expected_in_top = ['P00003392', 'P80001571', 'P60007168']  # Clinton, Trump, Sanders
        
        missing = [cid for cid in expected_in_top if cid not in df_total['CAND_ID'].values]
        for cid in missing:
            errors.append(f"Expected top candidate {cid} not found in data")
    
    # Print Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    print(f"\nChecks performed: 10")
    print(f"Errors:   {len(errors)}")
    print(f"Warnings: {len(warnings)}")
    print(f"Info:     {len(info)}")
    
    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        for i, err in enumerate(errors, 1):
            print(f"  {i}. {err}")
    
    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for i, warn in enumerate(warnings, 1):
            print(f"  {i}. {warn}")
    
    if info:
        print(f"\n✅ PASSED CHECKS ({len(info)}):")
        for i, inf in enumerate(info, 1):
            print(f"  {i}. {inf}")
    
    print("\n" + "="*80)
    if not errors and not warnings:
        print("✅ ALL VERIFICATIONS PASSED - DATA IS CORRECT!")
        print("="*80)
        return True
    elif not errors:
        print("⚠️  PASSED WITH WARNINGS - Review warnings above")
        print("="*80)
        return True
    else:
        print("❌ VERIFICATION FAILED - Fix errors before using data")
        print("="*80)
        return False


def main():
    """Run comprehensive verification."""
    print("\nThis script performs comprehensive data verification.")
    print("Run this AFTER validate_outputs.py passes.")
    print("\nThis checks:")
    print("  - Overall statistics")
    print("  - Support breakdowns")
    print("  - Known candidate verification")
    print("  - Distribution sanity")
    print("  - And more...")
    
    success = verify_data()
    
    if success:
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("\n1. Spot-check 2-3 candidates on FEC.gov")
        print("   - Go to https://www.fec.gov/data/candidates/")
        print("   - Search by candidate ID")
        print("   - Compare individual contributions")
        print("\n2. If everything looks good, you're ready to analyze!")
        print("   - Use: outputs/total/total_final_support_table_{}.csv".format(SUFFIX))
        print("\n3. Document your data:")
        print("   - Note the cycle and download date")
        print("   - Keep validation reports")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)