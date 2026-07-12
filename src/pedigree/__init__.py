"""Pedigree: Model lineage tracking as bloodline records.

Tracks model bloodlines like a breeder tracks dog pedigrees.
Base model → fine-tunes → merges → quantizations → adapters.

Usage:
    from pedigree import Pedigree

    p = Pedigree("lineage.json")
    p.register("my-model", sire="base-model", method="sft")
    p.print_lineage("my-model")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .model import BreedingMethod, BreedingRecord, LineageTree, Model
from .store import LineageStore
from .visualizer import render_ascii, render_dot

__version__ = "0.1.0"
__all__ = [
    "Pedigree",
    "Model",
    "LineageTree",
    "BreedingRecord",
    "BreedingMethod",
    "LineageStore",
]


@dataclass
class BreedingRecommendation:
    """A recommended breeding partner for maximizing diversity."""

    model_id: str
    diversity_score: float
    common_ancestors: list[str]


class Pedigree:
    """Main interface for model lineage tracking.

    Wraps LineageStore with a clean, high-level API.

    Args:
        store_path: Path to the JSON lineage file. Created on first save.
    """

    def __init__(self, store_path: str | Path = "lineage.json"):
        self.store = LineageStore(store_path)

    # ── Registration ─────────────────────────────────────────────

    def register(
        self,
        id: str,
        sire: Optional[str] = None,
        dam: Optional[str] = None,
        method: Optional[str] = None,
        birth_date: Optional[str] = None,
        notes: Optional[str] = None,
        **metadata,
    ) -> Model:
        """Register a new model in the lineage.

        Args:
            id:        Unique model identifier (e.g., "llama-3-70b").
            sire:      Father / primary base model ID.
            dam:       Mother / secondary base model ID (for merges).
            method:    How this model was produced ("sft", "merge", ...).
            birth_date: ISO date string (e.g., "2024-07-12").
            notes:     Free-form description.
            **metadata: Additional key-value metadata.

        Returns:
            The registered Model.
        """
        model = Model(
            id=id,
            sire=sire,
            dam=dam,
            method=method,
            birth_date=birth_date,
            notes=notes,
            metadata=metadata,
        )
        self.store.add_model(model)
        return model

    def get(self, model_id: str) -> Optional[Model]:
        """Retrieve a model by ID."""
        return self.store.get_model(model_id)

    def all_models(self) -> list[Model]:
        """List all registered models."""
        return self.store.all_models()

    # ── Lineage Queries ──────────────────────────────────────────

    def get_lineage(self, model_id: str, generations: int = 10) -> Optional[LineageTree]:
        """Get the full ancestry tree for a model.

        Args:
            model_id: The model to trace.
            generations: Maximum depth to trace.

        Returns:
            LineageTree or None if model not found.
        """
        return self.store.resolve_lineage(model_id, max_depth=generations)

    def get_children(self, model_id: str) -> list[Model]:
        """Get direct children of a model."""
        return self.store.get_children(model_id)

    def get_siblings(self, model_id: str) -> list[Model]:
        """Get models sharing at least one parent."""
        return self.store.get_siblings(model_id)

    def get_ancestors(self, model_id: str, generations: int = 100) -> set[str]:
        """Get all ancestor model IDs."""
        return self.store.get_ancestors(model_id, generations)

    def common_ancestry(self, id_a: str, id_b: str) -> list[str]:
        """Find common ancestors between two models."""
        return sorted(self.store.common_ancestors(id_a, id_b))

    # ── Inbreeding & Diversity ───────────────────────────────────

    def check_inbreeding(self, id_a: str, id_b: str, generations: int = 10) -> float:
        """Calculate the inbreeding coefficient between two models.

        Returns 0.0 (unrelated) to 1.0 (identical ancestry).
        Values > 0.25 indicate significant shared heritage.

        Args:
            id_a: First model ID.
            id_b: Second model ID.
            generations: How many generations back to check.

        Returns:
            Coefficient of relationship (0.0–1.0).
        """
        return self.store.inbreeding_coefficient(id_a, id_b, generations)

    def recommend_breeding(
        self, model_id: str, top_k: int = 5, generations: int = 10
    ) -> list[BreedingRecommendation]:
        """Recommend outcross partners to maximize genetic diversity.

        Finds models with the lowest inbreeding coefficient relative
        to the given model — i.e., the freshest blood.

        Args:
            model_id: The model you want to breed.
            top_k:    Number of recommendations.
            generations: Depth of ancestry check.

        Returns:
            List of BreedingRecommendation sorted by diversity (best first).
        """
        raw = self.store.recommend_outcrosses(
            model_id, generations=generations, top_k=top_k
        )
        recs: list[BreedingRecommendation] = []
        for candidate_id, diversity in raw:
            common = self.common_ancestry(model_id, candidate_id)
            recs.append(BreedingRecommendation(
                model_id=candidate_id,
                diversity_score=diversity,
                common_ancestors=common,
            ))
        return recs

    # ── Visualization ────────────────────────────────────────────

    def print_lineage(self, model_id: str, generations: int = 10) -> str:
        """Print and return the ASCII lineage tree.

        Args:
            model_id: Model to visualize.
            generations: Max depth.

        Returns:
            The ASCII tree as a string.
        """
        tree = self.get_lineage(model_id, generations)
        if tree is None:
            return f"Model '{model_id}' not found."
        result = render_ascii(tree, max_depth=generations)
        print(result)
        return result

    def export_dot(
        self, model_id: str, output_path: str, generations: int = 10
    ) -> str:
        """Export lineage tree as a GraphViz DOT file.

        Args:
            model_id: Model to trace.
            output_path: Where to write the .dot file.
            generations: Max depth.

        Returns:
            The DOT graph as a string.
        """
        tree = self.get_lineage(model_id, generations)
        if tree is None:
            raise ValueError(f"Model '{model_id}' not found")
        dot = render_dot(tree, title=f"Lineage: {model_id}")
        Path(output_path).write_text(dot)
        return dot
