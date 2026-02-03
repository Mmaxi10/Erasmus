"""Run FFIEC data formatting and cleaning in sequence.

Usage:
    python3 RunDataPipeline.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_step(script_path: Path) -> None:
    print(f"Running {script_path.name}...")
    subprocess.run([sys.executable, str(script_path)], check=True)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    formatting = base_dir / "DataFormatting.py"
    cleaning = base_dir / "DataCleaning.py"

    if not formatting.exists():
        raise FileNotFoundError(f"Missing {formatting}")
    if not cleaning.exists():
        raise FileNotFoundError(f"Missing {cleaning}")

    run_step(formatting)
    run_step(cleaning)


if __name__ == "__main__":
    main()
