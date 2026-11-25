"""Run every test from the repo root: the code uses relative paths like models/sign_model.pt."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
