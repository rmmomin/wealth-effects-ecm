"""Create GDP and three-part wealth column charts from public saved outputs."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wealth_effects.ecm_column_charts import build_column_charts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/ai_split/columns')
    parser.add_argument('--config', type=Path, default=ROOT / 'config/ai_split.json')
    args = parser.parse_args()
    print(json.dumps(build_column_charts(args.output, args.config), indent=2))
