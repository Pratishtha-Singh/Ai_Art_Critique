import faiss
from kedro.io import AbstractDataset
from typing import Any, Dict
import os

class FaissIndexDataset(AbstractDataset):
    def __init__(self, filepath: str):
        self._filepath = filepath

    def _load(self) -> Any:
        if not os.path.exists(self._filepath):
            raise FileNotFoundError(f"Faiss index file not found: {self._filepath}")
        return faiss.read_index(self._filepath)

    def _save(self, data: Any) -> None:
        faiss.write_index(data, self._filepath)

    def _exists(self) -> bool:
        return os.path.exists(self._filepath)

    def _describe(self) -> Dict[str, Any]:
        return {"filepath": self._filepath, "type": "FaissIndex"}