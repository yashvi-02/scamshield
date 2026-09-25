import sys
from pathlib import Path

# Project root: scamshield/
PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mod7 import explain_message


def detect_message(message: str):
    """
    Runs the ScamShield detection pipeline through Module 7.
    """
    return explain_message(message)