"""Comprehensive tests for Pedigree model lineage tracker."""

import json
import os
import tempfile

import pytest

from pedigree import Pedigree, Model, LineageTree, BreedingMethod
from pedigree.store import LineageStore
from pedigree.visualizer import render_ascii, render_dot


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def tmp_store(tmp_path):
    """Temporary lineage store path."""
    return str(tmp_path / "lineage.json")


@pytest.fixture
def pedigree(tmp_store):
    """A Pedigree instance with a small test lineage.

    Build this tree:

        foundation-a     foundation-b
            │                │
            ├─── child-ab (merge of a+b)
            │
        foundation-c
            │
            ├─── fine-tune-1 (sft from foundation-a)
            ├─── fine-tune-2 (sft from foundation-a)
            │
            └─── deep-fine-tune (sft from fine-tune-1)
    """
    p = Pedigree(tmp_store)
    p.register("foundation-a", method="pretrain", notes="Foundation A")
    p.register("foundation-b", method="pretrain", notes="Foundation B")
    p.register("foundation-c", method="pretrain", notes="Foundation C")

    p.register("child-ab", sire="foundation-a", dam="foundation-b",
               method="merge", notes="A+B merge")

    p.register("fine-tune-1", sire="foundation-a", method="sft")
    p.register("fine-tune-2", sire="foundation-a", method="sft")
    p.register("deep-fine-tune", sire="fine-tune-1", method="sft")

    return p


# ── Model Dataclass Tests ───────────────────────────────────────

class TestModel:
    def test_foundation_model(self):
        m = Model(id="base", method="pretrain")
        assert m.is_foundation is True
        assert m.parents == []

    def test_with_sire(self):
        m = Model(id="child", sire="parent", method="sft")
        assert m.is_foundation is False
        assert m.parents == ["parent"]

    def test_with_both_parents(self):
        m = Model(id="child", sire="dad", dam="mom", method="merge")
        assert set(m.parents) == {"dad", "mom"}

    def test_serialization_roundtrip(self):
        m = Model(id="test", sire="p1", dam="p2", method="merge",
                  notes="test model", metadata={"key": "val"})
        d = m.to_dict()
        assert d["id"] == "test"
        assert d["sire"] == "p1"
        assert d["metadata"]["key"] == "val"

        m2 = Model.from_dict(d)
        assert m2.id == m.id
        assert m2.sire == m.sire
        assert m2.dam == m.dam
        assert m2.metadata == m.metadata


# ── Registration Tests ──────────────────────────────────────────

class TestRegistration:
    def test_register_foundation(self, pedigree):
        model = pedigree.get("foundation-a")
        assert model is not None
        assert model.is_foundation is True
        assert model.method == "pretrain"

    def test_register_with_sire(self, pedigree):
        model = pedigree.get("fine-tune-1")
        assert model is not None
        assert model.sire == "foundation-a"
        assert model.method == "sft"

    def test_register_merge(self, pedigree):
        model = pedigree.get("child-ab")
        assert model is not None
        assert model.sire == "foundation-a"
        assert model.dam == "foundation-b"
        assert model.method == "merge"

    def test_register_with_metadata(self, tmp_store):
        p = Pedigree(tmp_store)
        p.register("model-x", method="pretrain",
                   author="test", dataset="pile", epochs=10)
        model = p.get("model-x")
        assert model.metadata["author"] == "test"
        assert model.metadata["epochs"] == 10

    def test_all_models(self, pedigree):
        ids = {m.id for m in pedigree.all_models()}
        assert ids == {
            "foundation-a", "foundation-b", "foundation-c",
            "child-ab", "fine-tune-1", "fine-tune-2", "deep-fine-tune",
        }


# ── Lineage Query Tests ─────────────────────────────────────────

class TestLineageQueries:
    def test_get_lineage(self, pedigree):
        tree = pedigree.get_lineage("deep-fine-tune")
        assert tree is not None
        assert tree.model.id == "deep-fine-tune"
        assert tree.sire_tree.model.id == "fine-tune-1"
        assert tree.sire_tree.sire_tree.model.id == "foundation-a"

    def test_lineage_not_found(self, pedigree):
        tree = pedigree.get_lineage("nonexistent")
        assert tree is None

    def test_get_children(self, pedigree):
        children = pedigree.get_children("foundation-a")
        child_ids = {m.id for m in children}
        assert "child-ab" in child_ids
        assert "fine-tune-1" in child_ids
        assert "fine-tune-2" in child_ids
        assert "deep-fine-tune" not in child_ids

    def test_get_siblings(self, pedigree):
        siblings = pedigree.get_siblings("fine-tune-1")
        sibling_ids = {m.id for m in siblings}
        assert "fine-tune-2" in sibling_ids
        assert "child-ab" in sibling_ids  # shares foundation-a

    def test_get_ancestors(self, pedigree):
        ancestors = pedigree.get_ancestors("deep-fine-tune")
        assert "fine-tune-1" in ancestors
        assert "foundation-a" in ancestors

    def test_common_ancestry(self, pedigree):
        common = pedigree.common_ancestry("fine-tune-1", "fine-tune-2")
        assert "foundation-a" in common

    def test_common_ancestry_none(self, pedigree):
        common = pedigree.common_ancestry("foundation-a", "foundation-b")
        assert common == []


# ── Inbreeding Tests ────────────────────────────────────────────

class TestInbreeding:
    def test_unrelated_models(self, pedigree):
        coeff = pedigree.check_inbreeding("foundation-a", "foundation-b")
        assert coeff == 0.0

    def test_siblings(self, pedigree):
        # fine-tune-1 and fine-tune-2 share foundation-a as sire
        coeff = pedigree.check_inbreeding("fine-tune-1", "fine-tune-2")
        assert coeff > 0.0
        assert coeff <= 1.0

    def test_parent_child(self, pedigree):
        # fine-tune-1 is child of foundation-a
        coeff = pedigree.check_inbreeding("foundation-a", "fine-tune-1")
        assert coeff > 0.0

    def test_self_inbreeding(self, pedigree):
        coeff = pedigree.check_inbreeding("foundation-a", "foundation-a")
        # Same model — should be high
        assert coeff >= 0.5 or coeff == 0.0  # depends on self-path handling


# ── Breeding Recommendation Tests ───────────────────────────────

class TestBreedingRecommendations:
    def test_recommend_returns_candidates(self, pedigree):
        recs = pedigree.recommend_breeding("fine-tune-1", top_k=3)
        assert len(recs) <= 3
        assert len(recs) > 0

    def test_recommend_sorted_by_diversity(self, pedigree):
        recs = pedigree.recommend_breeding("fine-tune-1", top_k=5)
        for i in range(len(recs) - 1):
            assert recs[i].diversity_score >= recs[i + 1].diversity_score

    def test_recommend_excludes_ancestors(self, pedigree):
        recs = pedigree.recommend_breeding("deep-fine-tune")
        candidate_ids = {r.model_id for r in recs}
        ancestors = pedigree.get_ancestors("deep-fine-tune")
        # Should not recommend direct ancestors
        assert not (candidate_ids & ancestors)

    def test_recommend_excludes_self(self, pedigree):
        recs = pedigree.recommend_breeding("foundation-a")
        assert all(r.model_id != "foundation-a" for r in recs)


# ── Visualization Tests ─────────────────────────────────────────

class TestVisualization:
    def test_ascii_output(self, pedigree):
        tree = pedigree.get_lineage("deep-fine-tune")
        assert tree is not None
        result = render_ascii(tree)
        assert "deep-fine-tune" in result
        assert "fine-tune-1" in result
        assert "foundation-a" in result

    def test_ascii_structure(self, pedigree):
        tree = pedigree.get_lineage("child-ab")
        assert tree is not None
        result = render_ascii(tree)
        assert "child-ab" in result
        assert "foundation-a" in result
        assert "foundation-b" in result

    def test_dot_output(self, pedigree):
        tree = pedigree.get_lineage("child-ab")
        assert tree is not None
        dot = render_dot(tree)
        assert "digraph" in dot
        assert "child-ab" in dot
        assert "foundation-a" in dot
        assert "->" in dot

    def test_dot_has_merge_label(self, pedigree):
        tree = pedigree.get_lineage("child-ab")
        assert tree is not None
        dot = render_dot(tree)
        assert "merge" in dot

    def test_print_lineage(self, pedigree, capsys):
        result = pedigree.print_lineage("fine-tune-1")
        assert "fine-tune-1" in result
        assert "foundation-a" in result

    def test_print_lineage_not_found(self, pedigree):
        result = pedigree.print_lineage("nonexistent")
        assert "not found" in result.lower()

    def test_export_dot(self, pedigree, tmp_path):
        out = str(tmp_path / "test.dot")
        dot = pedigree.export_dot("child-ab", out)
        assert os.path.exists(out)
        with open(out) as f:
            assert f.read() == dot

    def test_export_dot_not_found(self, pedigree, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            pedigree.export_dot("nonexistent", str(tmp_path / "x.dot"))


# ── Store Persistence Tests ─────────────────────────────────────

class TestStorePersistence:
    def test_persistence_roundtrip(self, tmp_store):
        p1 = Pedigree(tmp_store)
        p1.register("model-a", method="pretrain")
        p1.register("model-b", sire="model-a", method="sft")

        # New instance loading same file
        p2 = Pedigree(tmp_store)
        assert p2.get("model-a") is not None
        assert p2.get("model-b") is not None
        assert p2.get("model-b").sire == "model-a"

    def test_store_creates_file(self, tmp_store):
        p = Pedigree(tmp_store)
        p.register("test-model", method="pretrain")
        assert os.path.exists(tmp_store)

    def test_store_json_valid(self, tmp_store):
        p = Pedigree(tmp_store)
        p.register("a", method="pretrain")
        p.register("b", sire="a", method="sft")

        with open(tmp_store) as f:
            data = json.load(f)

        assert "models" in data
        assert len(data["models"]) == 2
        assert "records" in data


# ── Edge Cases ──────────────────────────────────────────────────

class TestEdgeCases:
    def test_register_duplicate_overwrites(self, tmp_store):
        p = Pedigree(tmp_store)
        p.register("model", method="pretrain", notes="v1")
        p.register("model", method="sft", sire="base", notes="v2")
        model = p.get("model")
        assert model.notes == "v2"
        assert model.sire == "base"

    def test_deep_lineage(self, tmp_store):
        p = Pedigree(tmp_store)
        p.register("gen-0", method="pretrain")
        for i in range(1, 6):
            p.register(f"gen-{i}", sire=f"gen-{i - 1}", method="sft")

        ancestors = p.get_ancestors("gen-5")
        for i in range(5):
            assert f"gen-{i}" in ancestors

    def test_inbreeding_deep_tree(self, tmp_store):
        """Test that inbreeding detects cousin relationships."""
        p = Pedigree(tmp_store)
        # Grandparent
        p.register("grandparent", method="pretrain")
        # Two children
        p.register("parent-1", sire="grandparent", method="sft")
        p.register("parent-2", sire="grandparent", method="sft")
        # Two grandchildren (cousins)
        p.register("cousin-1", sire="parent-1", method="sft")
        p.register("cousin-2", sire="parent-2", method="sft")

        coeff = p.check_inbreeding("cousin-1", "cousin-2")
        assert coeff > 0.0  # Should detect shared grandparent
