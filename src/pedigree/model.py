"""Core data classes for Pedigree."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class BreedingMethod(str, Enum):
    """How a model was produced."""

    PRETRAIN = "pretrain"
    SFT = "sft"
    RLHF = "rlhf"
    MERGE = "merge"
    QUANTIZE = "quantize"
    ADAPTER = "adapter"
    DISTILL = "distill"
    CONTINUE_PRETRAIN = "continue_pretrain"


@dataclass
class Model:
    """A registered model with lineage information.

    Uses breeder terminology:
      - sire:  the parent model (base / father)
      - dam:   second parent (for merges, the other base model / mother)
      - method: how this model was bred from its parent(s)
    """

    id: str
    sire: Optional[str] = None        # Father / primary base model
    dam: Optional[str] = None         # Mother / secondary base (merges)
    method: Optional[str] = None      # BreedingMethod value
    birth_date: Optional[str] = None  # ISO date string
    notes: Optional[str] = None       # Free-form description
    metadata: dict = field(default_factory=dict)

    @property
    def parents(self) -> list[str]:
        """All non-None parents."""
        return [p for p in (self.sire, self.dam) if p is not None]

    @property
    def is_foundation(self) -> bool:
        """True if this model has no parents (trained from scratch)."""
        return self.sire is None and self.dam is None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "sire": self.sire,
            "dam": self.dam,
            "method": self.method,
            "birth_date": self.birth_date,
            "notes": self.notes,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Model":
        return cls(
            id=d["id"],
            sire=d.get("sire"),
            dam=d.get("dam"),
            method=d.get("method"),
            birth_date=d.get("birth_date"),
            notes=d.get("notes"),
            metadata=d.get("metadata", {}),
        )


@dataclass
class LineageTree:
    """A resolved ancestry tree for a single model.

    Each node contains a Model and its recursively-resolved ancestors.
    """

    model: Model
    sire_tree: Optional["LineageTree"] = None
    dam_tree: Optional["LineageTree"] = None

    @property
    def all_ancestors(self) -> set[str]:
        """All ancestor model IDs in this tree."""
        ancestors: set[str] = set()
        if self.sire_tree:
            ancestors.add(self.sire_tree.model.id)
            ancestors |= self.sire_tree.all_ancestors
        if self.dam_tree:
            ancestors.add(self.dam_tree.model.id)
            ancestors |= self.dam_tree.all_ancestors
        return ancestors

    def to_ascii(self, prefix: str = "", is_last: bool = True, depth: int = 0,
                 max_depth: int = 10) -> str:
        """Render this subtree as ASCII art."""
        if depth > max_depth:
            return ""

        connector = "└── " if is_last else "├── "
        method_tag = f" [{self.model.method}]" if self.model.method else ""
        lines = [f"{prefix}{connector}{self.model.id}{method_tag}"]

        children = []
        if self.sire_tree:
            children.append(self.sire_tree)
        if self.dam_tree:
            children.append(self.dam_tree)

        for i, child in enumerate(children):
            child_is_last = i == len(children) - 1
            child_prefix = prefix + ("    " if is_last else "│   ")
            line = child.to_ascii(child_prefix, child_is_last, depth + 1, max_depth)
            if line:
                lines.append(line)

        return "\n".join(lines)


@dataclass
class BreedingRecord:
    """Record of a breeding event (merge, fine-tune, etc.)."""

    child_id: str
    sire_id: Optional[str]
    dam_id: Optional[str]
    method: str
    timestamp: Optional[str] = None
    notes: Optional[str] = None
    parameters: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "child_id": self.child_id,
            "sire_id": self.sire_id,
            "dam_id": self.dam_id,
            "method": self.method,
            "timestamp": self.timestamp,
            "notes": self.notes,
            "parameters": dict(self.parameters),
        }
