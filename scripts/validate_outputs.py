"""
Checks for:
1. File existence and structure
2. No duplicate candidates
3. Office filter accuracy
4. Total calculations
5. Cross-file consistency
6. Senate + Presidential = Total validation
7. Sample candidate verification
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Import config for paths
from config import SENATE_OUT_DIR, PRESIDENTIAL_OUT_DIR, TOTAL_OUT_DIR, SUFFIX, TARGET_ELECTION_YR


class ValidationReport:
    """Collect and display validation results."""
    
    def __init__(self):
        self.error_messages = []
        self.warning_messages = []
        self.info_messages = []
        self.passed_messages = []
    
    def error(self, msg: str):
        self.error_messages.append(f"❌ ERROR: {msg}")
    
    def warning(self, msg: str):
        self.warning_messages.append(f"⚠️  WARNING: {msg}")
    
    def info(self, msg: str):
        self.info_messages.append(f"ℹ️  INFO: {msg}")
    
    def success(self, msg: str):
        self.passed_messages.append(f"✅ PASS: {msg}")
    
    def print_summary(self):
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        
        if self.passed_messages:
            print(f"\n✅ PASSED CHECKS ({len(self.passed_messages)}):")
            for msg in self.passed_messages:
                print(f"  {msg}")
        
        if self.info_messages:
            print(f"\nℹ️  INFORMATION ({len(self.info_messages)}):")
            for msg in self.info_messages:
                print(f"  {msg}")
        
        if self.warning_messages:
            print(f"\n⚠️  WARNINGS ({len(self.warning_messages)}):")
            for msg in self.warning_messages:
                print(f"  {msg}")
        
        if self.error_messages:
            print(f"\n❌ ERRORS ({len(self.error_messages)}):")
            for msg in self.error_messages:
                print(f"  {msg}")
        
        print("\n" + "="*80)
        if not self.error_messages and not self.warning_messages:
            print("✅ ALL VALIDATIONS PASSED")
        elif not self.error_messages:
            print("⚠️  PASSED WITH WARNINGS")
        else:
            print("❌ VALIDATION FAILED")
        print("="*80 + "\n")
        
        return len(self.error_messages) == 0


def check_files_exist(report: ValidationReport) -> Dict[str, pd.DataFrame]:
    """Check that all expected output files exist and load them."""
    print("\n" + "="*80)
    print("CHECK 1: File Existence")
    print("="*80)
    
    files_to_check = {
        'senate_final': SENATE_OUT_DIR / f"senate_final_support_table_{SUFFIX}.csv",
        'senate_no_support': SENATE_OUT_DIR / f"senate_candidates_no_support_{SUFFIX}.csv",
        'senate_all': SENATE_OUT_DIR / f"senate_candidates_all_with_flag_{SUFFIX}.csv",
        'senate_superpac': SENATE_OUT_DIR / f"senate_superpac_ie_support_{SUFFIX}.csv",
        'senate_indiv': SENATE_OUT_DIR / f"senate_individual_support_{SUFFIX}.csv",
        'senate_pac': SENATE_OUT_DIR / f"senate_pac_support_corp_nonconnected_{SUFFIX}.csv",
        
        'pres_final': PRESIDENTIAL_OUT_DIR / f"presidential_final_support_table_{SUFFIX}.csv",
        'pres_no_support': PRESIDENTIAL_OUT_DIR / f"presidential_candidates_no_support_{SUFFIX}.csv",
        'pres_all': PRESIDENTIAL_OUT_DIR / f"presidential_candidates_all_with_flag_{SUFFIX}.csv",
        'pres_superpac': PRESIDENTIAL_OUT_DIR / f"presidential_superpac_ie_support_{SUFFIX}.csv",
        'pres_indiv': PRESIDENTIAL_OUT_DIR / f"presidential_individual_support_{SUFFIX}.csv",
        'pres_pac': PRESIDENTIAL_OUT_DIR / f"presidential_pac_support_corp_nonconnected_{SUFFIX}.csv",
        
        'total_final': TOTAL_OUT_DIR / f"total_final_support_table_{SUFFIX}.csv",
        'total_no_support': TOTAL_OUT_DIR / f"total_candidates_no_support_{SUFFIX}.csv",
        'total_all': TOTAL_OUT_DIR / f"total_candidates_all_with_flag_{SUFFIX}.csv",
        'total_superpac': TOTAL_OUT_DIR / f"total_superpac_ie_support_{SUFFIX}.csv",
        'total_indiv': TOTAL_OUT_DIR / f"total_individual_support_{SUFFIX}.csv",
        'total_pac': TOTAL_OUT_DIR / f"total_pac_support_corp_nonconnected_{SUFFIX}.csv",
    }
    
    loaded_data = {}
    
    for key, filepath in files_to_check.items():
        if filepath.exists():
            try:
                df = pd.read_csv(filepath)
                loaded_data[key] = df
                report.success(f"Found and loaded {filepath.name} ({len(df):,} rows)")
            except Exception as e:
                report.error(f"Failed to load {filepath.name}: {e}")
        else:
            report.error(f"Missing file: {filepath}")
    
    return loaded_data


def check_no_duplicates(data: Dict[str, pd.DataFrame], report: ValidationReport):
    """Check for duplicate candidate-year combinations."""
    print("\n" + "="*80)
    print("CHECK 2: No Duplicate Candidates")
    print("="*80)
    
    key_cols = ['CAND_ID', 'CAND_ELECTION_YR']
    
    for name, df in data.items():
        if name.endswith('_final') or name.endswith('_all') or name.endswith('_no_support'):
            if not all(col in df.columns for col in key_cols):
                report.warning(f"{name}: Missing key columns {key_cols}")
                continue
            
            dupes = df.duplicated(key_cols, keep=False)
            if dupes.any():
                n_dupes = dupes.sum()
                report.error(f"{name}: Found {n_dupes:,} duplicate rows by {key_cols}")
                # Show examples
                examples = df[dupes].head(5)
                print(f"\n  Examples from {name}:")
                print(examples[key_cols + ['CAND_NAME', 'CAND_OFFICE']].to_string(index=False))
            else:
                report.success(f"{name}: No duplicates by {key_cols}")


def check_office_filters(data: Dict[str, pd.DataFrame], report: ValidationReport):
    """Check that office filters are applied correctly."""
    print("\n" + "="*80)
    print("CHECK 3: Office Filter Accuracy")
    print("="*80)
    
    checks = [
        ('senate_final', 'S'),
        ('senate_all', 'S'),
        ('senate_no_support', 'S'),
        ('pres_final', 'P'),
        ('pres_all', 'P'),
        ('pres_no_support', 'P'),
    ]
    
    for key, expected_office in checks:
        if key not in data:
            continue
        
        df = data[key]
        if 'CAND_OFFICE' not in df.columns:
            report.warning(f"{key}: Missing CAND_OFFICE column")
            continue
        
        offices = df['CAND_OFFICE'].unique()
        if len(offices) == 1 and offices[0] == expected_office:
            report.success(f"{key}: Only contains office '{expected_office}' as expected")
        else:
            report.error(f"{key}: Expected only '{expected_office}', found {sorted(offices)}")
    
    # Check total contains both
    if 'total_final' in data:
        df = data['total_final']
        if 'CAND_OFFICE' in df.columns:
            offices = set(df['CAND_OFFICE'].unique())
            if 'S' in offices or 'P' in offices:
                report.success(f"total_final: Contains offices {sorted(offices)}")
            else:
                report.error(f"total_final: Expected S and/or P, found {sorted(offices)}")


def check_election_year(data: Dict[str, pd.DataFrame], report: ValidationReport):
    """Check that all candidates are from target election year."""
    print("\n" + "="*80)
    print("CHECK 4: Election Year Filter")
    print("="*80)
    
    for name, df in data.items():
        if name.endswith('_final') or name.endswith('_all') or name.endswith('_no_support'):
            if 'CAND_ELECTION_YR' not in df.columns:
                report.warning(f"{name}: Missing CAND_ELECTION_YR column")
                continue
            
            years = df['CAND_ELECTION_YR'].unique()
            if len(years) == 1 and str(years[0]) == TARGET_ELECTION_YR:
                report.success(f"{name}: All candidates from election year {TARGET_ELECTION_YR}")
            else:
                years_str = ', '.join(sorted([str(y) for y in years]))
                report.error(f"{name}: Expected only {TARGET_ELECTION_YR}, found: {years_str}")


def check_total_calculations(data: Dict[str, pd.DataFrame], report: ValidationReport):
    """Verify TOTAL_SUPPORT is calculated correctly."""
    print("\n" + "="*80)
    print("CHECK 5: Total Support Calculations")
    print("="*80)
    
    support_cols = [
        'INDIVIDUAL_SUPPORT',
        'CORP_PAC_SUPPORT',
        'NONCONNECTED_PAC_SUPPORT',
        'SUPERPAC_IE_SUPPORT'
    ]
    
    for name, df in data.items():
        if name.endswith('_final') or name.endswith('_all'):
            if 'TOTAL_SUPPORT' not in df.columns:
                report.warning(f"{name}: Missing TOTAL_SUPPORT column")
                continue
            
            # Check all support columns exist
            missing_cols = [col for col in support_cols if col not in df.columns]
            if missing_cols:
                report.warning(f"{name}: Missing columns {missing_cols}")
                continue
            
            # Calculate total
            calculated = df[support_cols].sum(axis=1)
            
            # Allow small floating point differences
            diff = (calculated - df['TOTAL_SUPPORT']).abs()
            max_diff = diff.max()
            
            if max_diff < 0.01:
                report.success(f"{name}: TOTAL_SUPPORT correctly calculated (max diff: {max_diff:.6f})")
            else:
                report.error(f"{name}: TOTAL_SUPPORT mismatch (max diff: {max_diff:.2f})")
                # Show problem rows
                problem_rows = df[diff > 0.01].head(3)
                if not problem_rows.empty:
                    print(f"\n  Examples from {name}:")
                    cols_to_show = ['CAND_ID', 'CAND_NAME'] + support_cols + ['TOTAL_SUPPORT']
                    print(problem_rows[cols_to_show].to_string(index=False))


def check_has_money_flag(data: Dict[str, pd.DataFrame], report: ValidationReport):
    """Check HAS_MONEY flag is consistent."""
    print("\n" + "="*80)
    print("CHECK 6: HAS_MONEY Flag Consistency")
    print("="*80)
    
    for name, df in data.items():
        if name.endswith('_all'):
            if 'HAS_MONEY' not in df.columns or 'TOTAL_SUPPORT' not in df.columns:
                report.warning(f"{name}: Missing HAS_MONEY or TOTAL_SUPPORT column")
                continue
            
            # Check flag matches actual support
            expected_flag = (df['TOTAL_SUPPORT'] > 0).astype(int)
            mismatches = (df['HAS_MONEY'] != expected_flag).sum()
            
            if mismatches == 0:
                report.success(f"{name}: HAS_MONEY flag consistent with TOTAL_SUPPORT")
            else:
                report.error(f"{name}: {mismatches:,} rows have inconsistent HAS_MONEY flag")


def check_final_vs_all_consistency(data: Dict[str, pd.DataFrame], report: ValidationReport):
    """Check that final + no_support = all."""
    print("\n" + "="*80)
    print("CHECK 7: Final vs All File Consistency")
    print("="*80)
    
    checks = [
        ('senate_final', 'senate_no_support', 'senate_all'),
        ('pres_final', 'pres_no_support', 'pres_all'),
        ('total_final', 'total_no_support', 'total_all'),
    ]
    
    for final_key, no_support_key, all_key in checks:
        if not all(k in data for k in [final_key, no_support_key, all_key]):
            continue
        
        final_df = data[final_key]
        no_support_df = data[no_support_key]
        all_df = data[all_key]
        
        # Check row counts
        combined_count = len(final_df) + len(no_support_df)
        all_count = len(all_df)
        
        if combined_count == all_count:
            report.success(f"{final_key} + {no_support_key} = {all_key} ({combined_count:,} rows)")
        else:
            report.error(f"{final_key} ({len(final_df):,}) + {no_support_key} ({len(no_support_df):,}) "
                        f"= {combined_count:,} but {all_key} has {all_count:,} rows")
        
        # Check candidate IDs match
        final_ids = set(final_df['CAND_ID'].unique())
        no_support_ids = set(no_support_df['CAND_ID'].unique())
        all_ids = set(all_df['CAND_ID'].unique())
        
        combined_ids = final_ids | no_support_ids
        if combined_ids == all_ids:
            report.success(f"{final_key} + {no_support_key} candidate IDs match {all_key}")
        else:
            missing = all_ids - combined_ids
            extra = combined_ids - all_ids
            if missing:
                report.error(f"{all_key} has {len(missing)} IDs not in final+no_support")
            if extra:
                report.error(f"final+no_support has {len(extra)} IDs not in {all_key}")


def check_senate_plus_presidential_equals_total(data: Dict[str, pd.DataFrame], report: ValidationReport):
    """Check that Senate + Presidential = Total."""
    print("\n" + "="*80)
    print("CHECK 8: Senate + Presidential = Total")
    print("="*80)
    
    if not all(k in data for k in ['senate_final', 'pres_final', 'total_final']):
        report.warning("Missing files for Senate + Presidential = Total check")
        return
    
    senate_df = data['senate_final']
    pres_df = data['pres_final']
    total_df = data['total_final']
    
    # Row count check
    combined_count = len(senate_df) + len(pres_df)
    total_count = len(total_df)
    
    if combined_count == total_count:
        report.success(f"Senate ({len(senate_df):,}) + Presidential ({len(pres_df):,}) = Total ({total_count:,})")
    else:
        report.error(f"Senate ({len(senate_df):,}) + Presidential ({len(pres_df):,}) = {combined_count:,} "
                    f"but Total has {total_count:,} rows")
    
    # Candidate ID check
    senate_ids = set(senate_df['CAND_ID'].unique())
    pres_ids = set(pres_df['CAND_ID'].unique())
    total_ids = set(total_df['CAND_ID'].unique())
    
    combined_ids = senate_ids | pres_ids
    
    if combined_ids == total_ids:
        report.success("Senate + Presidential candidate IDs exactly match Total")
    else:
        missing = total_ids - combined_ids
        extra = combined_ids - total_ids
        if missing:
            report.error(f"Total has {len(missing)} candidate IDs not in Senate+Presidential")
            print(f"  Missing IDs: {list(missing)[:5]}")
        if extra:
            report.error(f"Senate+Presidential has {len(extra)} candidate IDs not in Total")
            print(f"  Extra IDs: {list(extra)[:5]}")
    
    # Check for overlap (should be none)
    overlap = senate_ids & pres_ids
    if overlap:
        report.error(f"Found {len(overlap)} candidates appearing in BOTH Senate and Presidential files")
        print(f"  Overlapping IDs: {list(overlap)[:10]}")
    else:
        report.success("No candidates appear in both Senate and Presidential files")
    
    # Support total check
    senate_total = senate_df['TOTAL_SUPPORT'].sum()
    pres_total = pres_df['TOTAL_SUPPORT'].sum()
    combined_support = senate_total + pres_total
    total_support = total_df['TOTAL_SUPPORT'].sum()
    
    diff = abs(combined_support - total_support)
    if diff < 0.01:
        report.success(f"Support totals match: Senate (${senate_total:,.2f}) + Presidential (${pres_total:,.2f}) "
                      f"= Total (${total_support:,.2f})")
    else:
        report.error(f"Support totals don't match: Senate+Presidential = ${combined_support:,.2f} "
                    f"but Total = ${total_support:,.2f} (diff: ${diff:,.2f})")


def check_support_intermediate_files(data: Dict[str, pd.DataFrame], report: ValidationReport):
    """Check intermediate support files."""
    print("\n" + "="*80)
    print("CHECK 9: Intermediate Support Files")
    print("="*80)
    
    # Check Senate intermediate files
    if all(k in data for k in ['senate_final', 'senate_superpac', 'senate_indiv', 'senate_pac']):
        final_ids = set(data['senate_final']['CAND_ID'].unique())
        
        for key in ['senate_superpac', 'senate_indiv', 'senate_pac']:
            intermediate_ids = set(data[key]['CAND_ID'].unique())
            
            # All intermediate IDs should be in final
            extra = intermediate_ids - final_ids
            if extra:
                report.warning(f"{key} has {len(extra)} candidate IDs not in senate_final")
            else:
                report.success(f"{key}: All candidate IDs appear in senate_final")


def print_summary_statistics(data: Dict[str, pd.DataFrame]):
    """Print summary statistics for each dataset."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for name in ['senate_final', 'pres_final', 'total_final']:
        if name not in data:
            continue
        
        df = data[name]
        print(f"\n{name.upper()}:")
        print(f"  Total candidates: {len(df):,}")
        print(f"  Total support:    ${df['TOTAL_SUPPORT'].sum():,.2f}")
        print(f"  Average support:  ${df['TOTAL_SUPPORT'].mean():,.2f}")
        print(f"  Median support:   ${df['TOTAL_SUPPORT'].median():,.2f}")
        print(f"  Max support:      ${df['TOTAL_SUPPORT'].max():,.2f}")
        
        if 'CAND_OFFICE' in df.columns:
            print(f"  Offices: {sorted(df['CAND_OFFICE'].unique())}")
        
        # Support breakdown
        support_cols = ['INDIVIDUAL_SUPPORT', 'CORP_PAC_SUPPORT', 'NONCONNECTED_PAC_SUPPORT', 'SUPERPAC_IE_SUPPORT']
        if all(col in df.columns for col in support_cols):
            print("\n  Support breakdown:")
            for col in support_cols:
                total = df[col].sum()
                pct = (total / df['TOTAL_SUPPORT'].sum() * 100) if df['TOTAL_SUPPORT'].sum() > 0 else 0
                print(f"    {col:30s}: ${total:15,.2f} ({pct:5.1f}%)")


def spot_check_sample_candidates(data: Dict[str, pd.DataFrame], report: ValidationReport):
    """Display sample candidates for manual verification."""
    print("\n" + "="*80)
    print("CHECK 10: Sample Candidates for Manual Verification")
    print("="*80)
    
    for name in ['senate_final', 'pres_final']:
        if name not in data:
            continue
        
        df = data[name]
        
        # Top 5 by total support
        print(f"\n{name.upper()} - Top 5 by Total Support:")
        top5 = df.nlargest(5, 'TOTAL_SUPPORT')[
            ['CAND_ID', 'CAND_NAME', 'CAND_OFFICE', 'CAND_OFFICE_ST', 'TOTAL_SUPPORT']
        ]
        print(top5.to_string(index=False))
        
        report.info(f"{name}: Top candidate: {top5.iloc[0]['CAND_NAME']} (${top5.iloc[0]['TOTAL_SUPPORT']:,.2f})")


def main():
    """Run all validation checks."""
    print("\n" + "█"*80)
    print("█ FEC CAMPAIGN FINANCE PIPELINE VALIDATION")
    print("█"*80)
    print(f"\nTarget Election Year: {TARGET_ELECTION_YR}")
    print(f"Cycle Suffix: {SUFFIX}")
    print(f"\nDirectories:")
    print(f"  Senate:       {SENATE_OUT_DIR}")
    print(f"  Presidential: {PRESIDENTIAL_OUT_DIR}")
    print(f"  Total:        {TOTAL_OUT_DIR}")
    
    report = ValidationReport()
    
    # Load all data
    data = check_files_exist(report)
    
    if not data:
        report.error("No data files loaded - cannot proceed with validation")
        report.print_summary()
        return False
    
    # Run all checks
    check_no_duplicates(data, report)
    check_office_filters(data, report)
    check_election_year(data, report)
    check_total_calculations(data, report)
    check_has_money_flag(data, report)
    check_final_vs_all_consistency(data, report)
    check_senate_plus_presidential_equals_total(data, report)
    check_support_intermediate_files(data, report)
    
    # Summary stats and spot checks
    print_summary_statistics(data)
    spot_check_sample_candidates(data, report)
    
    # Print final report
    success = report.print_summary()
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)