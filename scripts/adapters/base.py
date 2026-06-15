"""Abstract base class for publisher adapters."""

from abc import ABC, abstractmethod


class PublisherAdapter(ABC):
    @abstractmethod
    def search(self, max_results: int) -> list[dict]:
        """Return list of {title, paper_id, abstract, source} dicts."""

    @abstractmethod
    def fetch_full_text(self, paper: dict) -> dict | None:
        """Return {sections: dict[str, str], tables: list[str]} or None."""
