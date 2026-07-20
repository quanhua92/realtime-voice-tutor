"""Markdown + frontmatter content loader.

Loads scenario and vocabulary files into memory at startup and exposes
filter helpers (by category, tag, difficulty, CEFR level).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import frontmatter

from .config import DATA_DIR


class DataLoader:
    """Loads and filters Markdown content files with YAML frontmatter."""

    def __init__(self, data_dir: str | Path = DATA_DIR) -> None:
        self.data_dir = Path(data_dir)
        self.scenarios: dict[str, dict] = {}
        self.vocabulary: dict[str, dict] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Load all markdown files into memory at construction time."""
        scenarios_dir = self.data_dir / "scenarios"
        if scenarios_dir.exists():
            for f in sorted(scenarios_dir.glob("*.md")):
                post = frontmatter.load(f)
                self.scenarios[f.stem] = {
                    "name": f.stem,
                    "meta": dict(post.metadata),
                    "content": post.content,
                }

        vocab_dir = self.data_dir / "vocabulary"
        if vocab_dir.exists():
            for f in sorted(vocab_dir.glob("*.md")):
                post = frontmatter.load(f)
                self.vocabulary[f.stem] = {
                    "name": f.stem,
                    "meta": dict(post.metadata),
                    "content": post.content,
                }

    def find_scenarios(
        self,
        category: Optional[str] = None,
        tag: Optional[str] = None,
        difficulty: Optional[str] = None,
    ) -> list[dict]:
        """Filter scenarios by category, tag, or difficulty (any combination)."""
        results: list[dict] = []
        for data in self.scenarios.values():
            meta = data["meta"]
            if category and meta.get("category") != category:
                continue
            if tag and tag not in meta.get("tags", []):
                continue
            if difficulty and difficulty not in meta.get("difficulty", []):
                continue
            results.append(data)
        return results

    def find_vocabulary(
        self,
        level: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> list[dict]:
        """Filter vocabulary by CEFR level or tag."""
        results: list[dict] = []
        for data in self.vocabulary.values():
            meta = data["meta"]
            if level and meta.get("level") != level:
                continue
            if tag and tag not in meta.get("tags", []):
                continue
            results.append(data)
        return results

    def get_all_categories(self) -> list[str]:
        """Return all available scenario categories (sorted, unique)."""
        cats = {d["meta"].get("category") for d in self.scenarios.values()}
        return sorted(c for c in cats if c)

    def get_all_tags(self) -> list[str]:
        """Return all unique tags across scenarios (sorted)."""
        tags: set[str] = set()
        for d in self.scenarios.values():
            tags.update(d["meta"].get("tags", []))
        return sorted(tags)

    def get_all_levels(self) -> list[str]:
        """Return all CEFR levels present in vocabulary files (sorted)."""
        levels = {d["meta"].get("level") for d in self.vocabulary.values()}
        return sorted(l for l in levels if l)
