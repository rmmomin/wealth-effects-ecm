"""Subdivide the frozen ECM wealth channel using S&P AI/other capitalizations."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wealth_effects.ecm_ai_publication import build_ai_split


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/ai_split')
    parser.add_argument('--config', type=Path, default=ROOT / 'config/ai_split.json')
    args = parser.parse_args()
    print(json.dumps(build_ai_split(args.output, args.config), indent=2))
