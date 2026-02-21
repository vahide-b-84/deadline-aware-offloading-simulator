# pre_process.py
"""
Pre-process launcher (run from project root).
Generates input Excel files into data/:
- homogeneous_server_info.xlsx
- heterogeneous_server_info.xlsx
- task_parameters.xlsx
"""

import os
import sys


def _add_project_root_to_syspath():
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def main():
    _add_project_root_to_syspath()
    from tools.generate_server_and_task_parameters import main as gen_main
    gen_main()


if __name__ == "__main__":
    main()
