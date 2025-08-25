# Ensure project root is on sys.path for imports like `services.*`
import os
import sys

repo_root = os.path.dirname(__file__)
if repo_root and repo_root not in sys.path:
    sys.path.insert(0, repo_root)
