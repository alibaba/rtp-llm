"""Shared Python/Java launch defaults stored with the Java resources."""
from pathlib import Path

FLEXLB_DIR = Path(__file__).resolve().parents[5]

def load_defaults():
    path = FLEXLB_DIR / 'flexlb-mock-engine/src/main/resources/load-client-defaults.properties'
    return dict(line.split('=', 1) for line in path.read_text().splitlines()
                if line and not line.startswith('#'))


DEFAULTS = load_defaults()
