"""5-fold OOF AUC for the current LightGBM baseline (no test leakage)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import cv_and_predict, load_frames  # noqa: E402


def main() -> None:
    train, test = load_frames()
    cv_and_predict(train, test)


if __name__ == "__main__":
    main()
