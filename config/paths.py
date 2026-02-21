"""config.paths

Single source of truth for project paths.

Why?
- Avoids 'File not found' errors after moving files into folders.
- Makes the project runnable from ANY working directory.

Rule:
- NEVER hardcode file paths in core/agents/io/tools.
- Always build paths from PROJECT_ROOT.
"""

import os

# project_root = folder containing Project_main.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Keep legacy support: older scripts used to drop results in root.
# Now everything should go into RESULTS_DIR.

def ensure_dirs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
