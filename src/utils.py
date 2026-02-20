from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np


@dataclass(frozen=True)
class Paths:
    root: Path
    data: Path
    raw: Path
    processed: Path
    artifacts: Path
    figs: Path
    report: Path


def get_paths() -> Paths:
    """
    Repo root assumed as: <repo_root>/src/utils.py -> parents[1] is repo_root
    """
    root = Path(__file__).resolve().parents[1]
    data = root / "data"
    return Paths(
        root=root,
        data=data,
        raw=data / "raw",
        processed=data / "processed",
        artifacts=root / "artifacts",
        figs=root / "figs",
        report=root / "report",
    )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger()
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def save_json(obj: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))