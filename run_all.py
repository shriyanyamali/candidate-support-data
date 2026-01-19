## 06

import sys

def run_step(name, fn):
    print("\n" + "="*80)
    print(f"RUNNING: {name}")
    print("="*80)
    fn()

def main():
    # Import inside main so failures show up clearly
    import superpac_ie_support
    import individual_support
    import pac_support_corp_union
    import merge_support

    run_step("superpac_ie_support.py", superpac_ie_support.main)
    run_step("individual_support.py", individual_support.main)
    run_step("pac_support_corp_union.py", pac_support_corp_union.main)
    run_step("merge_support.py", merge_support.main)

    print("\nProgram executed successfully")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nExecution terminated with error(s)", e)
        sys.exit(1)
