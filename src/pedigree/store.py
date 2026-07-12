"""JSON-based lineage store with graph queries."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .model import BreedingMethod, BreedingRecord, LineageTree, Model


class LineageStore:
    """File-backed store for model lineage records.

    Persists to a single JSON file. Supports graph queries:
    ancestry, descendants, siblings, common ancestors.
    """

    def __init__(self, path: str | Path = "lineage.json"):
        self.path = Path(path)
        self._models: dict[str, Model] = {}
        self._records: list[BreedingRecord] = []
        self._load()

    # ── Persistence ──────────────────────────────────────────────

    def _load(self) -> None:
        if self.path.exists():
            data = json.loads(self.path.read_text())
            for m in data.get("models", []):
                model = Model.from_dict(m)
                self._models[model.id] = model
            for r in data.get("records", []):
                self._records.append(BreedingRecord(
                    child_id=r["child_id"],
                    sire_id=r.get("sire_id"),
                    dam_id=r.get("dam_id"),
                    method=r.get("method", ""),
                    timestamp=r.get("timestamp"),
                    notes=r.get("notes"),
                    parameters=r.get("parameters", {}),
                ))

    def _save(self) -> None:
        data = {
            "models": [m.to_dict() for m in self._models.values()],
            "records": [r.to_dict() for r in self._records],
        }
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    # ── Registration ─────────────────────────────────────────────

    def add_model(self, model: Model) -> None:
        """Register or update a model."""
        self._models[model.id] = model
        if model.method and (model.sire or model.dam):
            self._records.append(BreedingRecord(
                child_id=model.id,
                sire_id=model.sire,
                dam_id=model.dam,
                method=model.method,
                timestamp=model.birth_date,
                notes=model.notes,
            ))
        self._save()

    def get_model(self, model_id: str) -> Optional[Model]:
        return self._models.get(model_id)

    def all_models(self) -> list[Model]:
        return list(self._models.values())

    # ── Graph Queries ────────────────────────────────────────────

    def get_children(self, model_id: str) -> list[Model]:
        """Direct children of a model."""
        return [
            m for m in self._models.values()
            if model_id in m.parents
        ]

    def get_descendants(self, model_id: str, max_depth: int = 100) -> set[str]:
        """All descendants of a model."""
        descendants: set[str] = set()
        queue = [(model_id, 0)]
        while queue:
            mid, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            for child in self.get_children(mid):
                if child.id not in descendants:
                    descendants.add(child.id)
                    queue.append((child.id, depth + 1))
        return descendants

    def get_siblings(self, model_id: str) -> list[Model]:
        """Models sharing at least one parent."""
        model = self._models.get(model_id)
        if not model:
            return []
        siblings = []
        for m in self._models.values():
            if m.id == model_id:
                continue
            if set(m.parents) & set(model.parents):
                siblings.append(m)
        return siblings

    def resolve_lineage(self, model_id: str, max_depth: int = 10) -> Optional[LineageTree]:
        """Resolve full ancestry tree for a model."""
        model = self._models.get(model_id)
        if model is None:
            return None

        sire_tree = None
        dam_tree = None

        if model.sire and max_depth > 0:
            sire_tree = self.resolve_lineage(model.sire, max_depth - 1)
        if model.dam and max_depth > 0:
            dam_tree = self.resolve_lineage(model.dam, max_depth - 1)

        return LineageTree(model=model, sire_tree=sire_tree, dam_tree=dam_tree)

    def get_ancestors(self, model_id: str, generations: int = 100) -> set[str]:
        """All ancestor model IDs within N generations."""
        tree = self.resolve_lineage(model_id, max_depth=generations)
        if tree is None:
            return set()
        return tree.all_ancestors

    def common_ancestors(self, id_a: str, id_b: str, generations: int = 100) -> set[str]:
        """Find shared ancestors between two models."""
        anc_a = self.get_ancestors(id_a, generations)
        anc_b = self.get_ancestors(id_b, generations)
        return anc_a & anc_b

    def ancestor_path(self, descendant: str, ancestor: str) -> list[list[str]]:
        """Find all paths from descendant to ancestor.

        Returns list of paths, each a list of model IDs from descendant to ancestor.
        """
        model = self._models.get(descendant)
        if model is None:
            return []

        if descendant == ancestor:
            return [[descendant]]

        paths: list[list[str]] = []
        for parent in model.parents:
            for sub_path in self._ancestor_path_recursive(parent, ancestor, set()):
                paths.append([descendant] + sub_path)
        return paths

    def _ancestor_path_recursive(self, current: str, target: str,
                                  visited: set[str]) -> list[list[str]]:
        if current in visited:
            return []
        visited = visited | {current}

        if current == target:
            return [[current]]

        model = self._models.get(current)
        if model is None:
            return []

        paths: list[list[str]] = []
        for parent in model.parents:
            for sub_path in self._ancestor_path_recursive(parent, target, visited):
                paths.append([current] + sub_path)
        return paths

    # ── Inbreeding ───────────────────────────────────────────────

    def inbreeding_coefficient(self, id_a: str, id_b: str, generations: int = 10) -> float:
        """Estimate inbreeding coefficient between two models.

        Uses a simplified Wright's coefficient of relationship:
        For each common ancestor, compute the contribution based on
        path length from each model to the common ancestor.

        CRC = Σ (1/2)^(n1+n2) for each common ancestor c
        where n1, n2 are the number of generations from each model to c.

        Returns 0.0–1.0 (0 = no relationship, 1 = identical ancestry).
        """
        # Check if one model is a direct ancestor of the other
        anc_a = self.get_ancestors(id_a, generations)
        anc_b = self.get_ancestors(id_b, generations)

        total = 0.0

        # Direct lineage: id_b is an ancestor of id_a (or vice versa)
        if id_b in anc_a:
            paths = self.ancestor_path(id_a, id_b)
            for pa in paths:
                n = max(len(pa) - 1, 0)
                if n > 0:
                    total += 0.5 ** n
        if id_a in anc_b:
            paths = self.ancestor_path(id_b, id_a)
            for pb in paths:
                n = max(len(pb) - 1, 0)
                if n > 0:
                    total += 0.5 ** n

        # Common ancestors in both lineages
        common = anc_a & anc_b
        for ancestor_id in common:
            paths_a = self.ancestor_path(id_a, ancestor_id)
            paths_b = self.ancestor_path(id_b, ancestor_id)

            for pa in paths_a:
                for pb in paths_b:
                    n1 = max(len(pa) - 1, 0)
                    n2 = max(len(pb) - 1, 0)
                    # Avoid counting self-relationship
                    if n1 == 0 and n2 == 0:
                        continue
                    total += (0.5 ** (n1 + n2))

        return min(total, 1.0)

    # ── Breeding Recommendations ─────────────────────────────────

    def recommend_outcrosses(self, model_id: str, generations: int = 10,
                              top_k: int = 5) -> list[tuple[str, float]]:
        """Find models with lowest inbreeding coefficient for breeding.

        Returns list of (model_id, diversity_score) sorted by diversity (desc).
        diversity_score = 1 - inbreeding_coefficient
        """
        results: list[tuple[str, float]] = []
        for candidate in self._models:
            if candidate == model_id:
                continue
            # Skip direct ancestors / descendants
            if candidate in self.get_ancestors(model_id, generations):
                continue
            if model_id in self.get_ancestors(candidate, generations):
                continue
            coeff = self.inbreeding_coefficient(model_id, candidate, generations)
            diversity = 1.0 - coeff
            results.append((candidate, round(diversity, 4)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
