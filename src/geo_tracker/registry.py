from __future__ import annotations
import yaml
from pathlib import Path
from typing import Iterable, Dict, Any

class PromptRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        with open(self.path, "r", encoding="utf-8") as f:
            self._data = yaml.safe_load(f)

    def panel(self, name: str) -> Iterable[Dict[str, Any]]:
        return list(self._data.get("panel_versions", {}).get(name, []))
