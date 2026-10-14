"""Run the portable actual-data ECM showcase without private data APIs."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wealth_effects.ecm_publication import build_ecm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--config", type=Path, default=ROOT / "config/ecm.json")
    parser.add_argument("--rebuild-bootstrap", action="store_true",
        help="Resample and refit both stages instead of restoring supplied draws")
    arguments = parser.parse_args()
    print(json.dumps(build_ecm(arguments.output, arguments.config, arguments.rebuild_bootstrap), indent=2))
