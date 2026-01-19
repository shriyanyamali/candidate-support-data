## 06

import sys

def run_step(name, fn, office_filter):
    """Run a pipeline step with the specified office filter."""
    office_desc = "+".join(sorted(office_filter))
    print("\n" + "="*80)
    print(f"RUNNING: {name} [{office_desc}]")
    print("="*80)
    fn(office_filter=office_filter)

def run_full_pipeline(office_filter, label):
    """Run the complete pipeline for a specific office type."""
    print("\n" + "█"*80)
    print(f"█ PIPELINE: {label}")
    print("█"*80)
    
    import superpac_ie_support
    import individual_support
    import pac_support_corp_union
    import merge_support

    run_step("superpac_ie_support.py", superpac_ie_support.main, office_filter)
    run_step("individual_support.py", individual_support.main, office_filter)
    run_step("pac_support_corp_union.py", pac_support_corp_union.main, office_filter)
    run_step("merge_support.py", merge_support.main, office_filter)
    
    print(f"\n✓ {label} pipeline completed successfully\n")

def main():
    """Run the complete pipeline for Senate, Presidential, and Total (combined)."""
    print("\n" + "="*80)
    print("FEC CAMPAIGN FINANCE PIPELINE")
    print("="*80)
    print("\nThis will generate three sets of outputs:")
    print("  1. Senate candidates only")
    print("  2. Presidential candidates only")
    print("  3. Total (Senate + Presidential combined)")
    print("="*80)
    
    try:
        # Run for Senate only
        run_full_pipeline({"S"}, "SENATE")
        
        # Run for Presidential only
        run_full_pipeline({"P"}, "PRESIDENTIAL")
        
        # Run for Total (both)
        run_full_pipeline({"S", "P"}, "TOTAL (SENATE + PRESIDENTIAL)")
        
        print("\n" + "█"*80)
        print("█ ALL PIPELINES COMPLETED SUCCESSFULLY")
        print("█"*80)
        print("\nOutput directories:")
        from config import SENATE_OUT_DIR, PRESIDENTIAL_OUT_DIR, TOTAL_OUT_DIR
        print(f"  Senate:       {SENATE_OUT_DIR}")
        print(f"  Presidential: {PRESIDENTIAL_OUT_DIR}")
        print(f"  Total:        {TOTAL_OUT_DIR}")
        print("="*80)
        
    except Exception as e:
        print("\n" + "█"*80)
        print("█ EXECUTION TERMINATED WITH ERROR")
        print("█"*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
