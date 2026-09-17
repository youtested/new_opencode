import sys
from pathlib import Path

# ensure the source package is importable even without the editable install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
