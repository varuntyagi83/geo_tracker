from abc import ABC, abstractmethod
from typing import Dict, Any

class LLMProvider(ABC):
    name: str
    @abstractmethod
    def generate(self, prompt: str, model: str) -> Dict[str, Any]:
        ...
