# lib/utils/config.py

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import yaml


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_path(project_root: Path, maybe_relative: Union[Path, str]) -> Path:
    p = Path(maybe_relative)
    if not p.is_absolute():
        p = project_root / p
    return p

def get_device(gpu: Optional[Union[int, str]] = None) -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"