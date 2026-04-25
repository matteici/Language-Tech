"""
09_gpu_followup_reliability.py — CLI entry point for Section 9.

The actual implementation lives in gpu_followup_reliability.py (importable).
This wrapper exists so the script can be invoked as:
    python src/09_gpu_followup_reliability.py [--smoke-test] [...]
"""
from gpu_followup_reliability import main

if __name__ == "__main__":
    main()
