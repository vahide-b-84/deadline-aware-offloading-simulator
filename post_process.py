# post_process.py
"""
Post-process launcher (run from project root).
Runs final aggregation/post-processing on results/.
"""

import os
import sys


def _add_project_root_to_syspath():
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def main():
    _add_project_root_to_syspath()
    from io_utils.post_process_results import main as post_main
    post_main()


if __name__ == "__main__":
    main()
