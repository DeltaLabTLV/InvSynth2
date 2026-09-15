"""Compatibility alias for the focused ICASSP U-Net runner.

The earlier Transformer grid is available in Git history but is not a paper
artifact because its contrastive target admitted a positional shortcut.
"""

from scripts.run_icassp_study import main


if __name__ == "__main__":
    main()
