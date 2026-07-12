# Pedigree: Model Lineage Tracking as Bloodline Records

> Tracks model bloodlines like a breeder tracks dog pedigrees. Register every birth, trace ancestry, detect inbreeding before you merge two cousins.

[![Python](https://img.shields.io/python/required-version-toml?toml=pyproject.toml)](https://python.org)
[![License](https://img.shields.io/github/license/SuperInstance/pedigree)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](tests/)

Every model has a family tree. Base models beget fine-tunes. Fine-tunes beget merges. Merges beget quantizations. Adapters stack on adapters. After three layers, nobody remembers who descended from whom — or whether the two models you're about to merge share a common ancestor with known limitations. Pedigree treats this seriously, applying the rigor of animal breeding registries to model lineage: registration, bloodline trees, inbreeding coefficients, and outcross recommendations.

## What It Does

Pedigree maintains a JSON-backed registry of models and their breeding history. Every model gets a lineage record with sire (base model), dam (training data lineage), birth date, and breeding method. The registry supports full ancestry trees — trace any model's bloodline back N generations, visualized as ASCII art in the terminal or exported as GraphViz DOT files for publication-quality diagrams.

The inbreeding coefficient (Wright's coefficient of relationship, adapted for model lineage) detects when two models share too much common ancestry. Before you merge two fine-tunes, check their inbreeding coefficient — if it's high, the merge will likely reinforce shared weaknesses rather than complement them. Pedigree recommends outcross candidates to maximize genetic diversity, suggesting models from different lineages that would bring fresh capabilities.

The visualization system produces both ASCII trees (for terminal workflows) and GraphViz DOT exports (for documentation and presentations). The JSON store is simple file-based persistence — no database required, human-readable, and git-friendly for version control of your breeding program.

## Install

```bash
pip install pedigree-tracker
```

For development:

```bash
git clone https://github.com/SuperInstance/pedigree.git
cd pedigree
pip install -e ".[dev]"
```

## Quick Start

```python
from pedigree import Pedigree

p = Pedigree("lineage.json")

# Register a foundation model (no parents — root of the tree)
p.register("llama-3-70b", method="pretrain",
           notes="Meta foundation model, 70B parameters")

# Register an official fine-tune (sire = base model)
p.register("llama-3-70b-chat", sire="llama-3-70b", method="sft",
           notes="Official instruction-tuned variant")

# Register a community fine-tune
p.register("noromaid-70b", sire="llama-3-70b", method="sft",
           notes="Sao10K creative writing fine-tune")

# Register a merge of two fine-tunes
p.register("creative-merge-v1",
           sire="llama-3-70b-chat",
           dam="noromaid-70b",
           method="merge",
           notes="SLERP merge for creative + instruction following")

# Check inbreeding BEFORE merging
coeff = p.check_inbreeding("llama-3-70b-chat", "noromaid-70b")
print(f"Inbreeding coefficient: {coeff:.2%}")
# → 50.00% (half-siblings sharing llama-3-70b as sire)

# Get recommendations for outcrossing (fresh blood)
recs = p.recommend_breeding("llama-3-70b-chat")
for r in recs[:5]:
    print(f"  {r.model_id}: diversity={r.diversity_score:.2f}")
    print(f"    common ancestors: {r.common_ancestors}")

# Visualize bloodline in terminal
p.print_lineage("creative-merge-v1")
# ════════════════════════════════════════════════
#   Bloodline: creative-merge-v1
# ════════════════════════════════════════════════
#   Gen 2: creative-merge-v1 (merge)
#   ├── Gen 1: llama-3-70b-chat (sft)
#   │   └── Gen 0: llama-3-70b (pretrain)
#   └── Gen 1: noromaid-70b (sft)
#       └── Gen 0: llama-3-70b (pretrain)
# ════════════════════════════════════════════════

# Export GraphViz for publication-quality diagrams
p.export_dot("creative-merge-v1", "lineage.dot")
# Then: dot -Tpng lineage.dot -o lineage.png
```

## Breeding Methods

| Method | Description | Typical Use |
|--------|-------------|-------------|
| `pretrain` | Foundation model trained from scratch | Root of a lineage tree |
| `sft` | Supervised fine-tuning | Instruction following, domain adaptation |
| `rlhf` | RL from human feedback | Alignment, preference optimization |
| `merge` | Model merging (SLERP, DARE, TIES) | Combining capabilities |
| `quantize` | Quantization (GGUF, AWQ, GPTQ) | Deployment optimization |
| `adapter` | LoRA / QLoRA adapter | Lightweight specialization |
| `distill` | Knowledge distillation | Smaller model from larger teacher |
| `continue_pretrain` | Continued pre-training | Domain expansion |

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    Pedigree                           │
│                                                       │
│  ┌──────────────┐    ┌─────────────────────────┐     │
│  │  LineageStore│    │     Pedigree (API)      │     │
│  │  (JSON file) │───▶│                         │     │
│  │              │    │  .register()            │     │
│  │ models: {}   │    │  .get_lineage()         │     │
│  │ records: []  │    │  .check_inbreeding()    │     │
│  └──────────────┘    │  .recommend_breeding()  │     │
│                      │  .print_lineage()       │     │
│                      │  .export_dot()          │     │
│                      └────────────┬────────────┘     │
│                                   │                   │
│                      ┌────────────▼────────────┐     │
│                      │    Visualizer           │     │
│                      │  ├─ render_ascii()      │     │
│                      │  └─ render_dot()        │     │
│                      └─────────────────────────┘     │
└──────────────────────────────────────────────────────┘
```

## API Reference

### `Pedigree`

```python
class Pedigree:
    def __init__(self, store_path: str | Path = "lineage.json")

    # Registration
    def register(self, id: str, *,
                 sire: str | None = None,
                 dam: str | None = None,
                 method: BreedingMethod = BreedingMethod.PRETRAIN,
                 notes: str = "",
                 traits: dict | None = None) -> Model

    # Querying
    def get_model(self, id: str) -> Model | None
    def get_lineage(self, id: str, generations: int = 10) -> LineageTree
    def get_all_models(self) -> list[Model]
    def get_descendants(self, id: str) -> list[Model]

    # Diversity
    def check_inbreeding(self, a: str, b: str) -> float
    def recommend_breeding(self, id: str) -> list[BreedingRecommendation]

    # Visualization
    def print_lineage(self, id: str, max_generations: int = 5)
    def export_dot(self, id: str, output_path: str | Path)
```

### `BreedingRecommendation`

```python
@dataclass
class BreedingRecommendation:
    model_id: str               # recommended partner
    diversity_score: float      # 0.0-1.0 (higher = more diverse)
    common_ancestors: list[str] # shared ancestors (fewer = better)
```

### `Model` and `LineageTree`

```python
@dataclass
class Model:
    id: str
    sire: str | None        # father (base model)
    dam: str | None         # mother (training data lineage)
    method: BreedingMethod
    registered_at: str      # ISO timestamp
    notes: str
    traits: dict            # benchmark scores, capabilities

@dataclass
class LineageTree:
    root: Model
    generations: list[list[Model]]  # by generation level
    total_ancestors: int
    unique_founders: int
```

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v

# Test inbreeding calculation
pytest tests/test_inbreeding.py -v

# Test visualization
pytest tests/test_visualizer.py -v
```

## Philosophy

Pedigree registries are how humanity learned to do selective breeding instead of random mutation. Before studbooks, animal breeding was superstition and luck. After studbooks, it became a science. The same transition is happening in AI right now — from ad-hoc fine-tuning to structured breeding programs. Pedigree brings 300 years of animal husbandry wisdom to model development.

The key innovation is the **inbreeding coefficient**. In animal breeding, this number determines whether a pairing will produce healthy offspring or reinforce genetic disorders. In model breeding, it determines whether a merge will produce emergent capabilities or amplify shared weaknesses. The math is the same; the domain is different.

For more on the breeding paradigm, see [AI-Writings](https://github.com/SuperInstance/AI-Writings).

## Ecosystem

| Repo | Role |
|------|------|
| **[pedigree](https://github.com/SuperInstance/pedigree)** | **This repo** — bloodline tracking |
| [lineage-tracker](https://github.com/SuperInstance/lineage-tracker) | Provenance records (complementary tracking) |
| [breed-registry](https://github.com/SuperInstance/breed-registry) | Breed assessment and selection |
| [baton](https://github.com/SuperInstance/baton) | Generational handoff (connects pedigree generations) |
| [vetcheck](https://github.com/SuperInstance/vetcheck) | Health checks for registered models |



## Pedigree vs Lineage Tracker: When to Use Which

Both [Pedigree](https://github.com/SuperInstance/pedigree) and [Lineage Tracker](https://github.com/SuperInstance/lineage-tracker) track model ancestry. They're complementary tools with different primary strengths:

| Concern | Pedigree | Lineage Tracker |
|---------|----------|----------------|
| **Mental model** | Breeder's studbook | MLOps audit trail |
| **Inbreeding coefficient** | ✅ Wright's coefficient of relationship | ❌ Trait diversity only |
| **Visualization** | ✅ ASCII trees + GraphViz DOT | ❌ Programmatic queries only |
| **Merge safety check** | ✅ Pre-merge inbreeding screening | ❌ Post-hoc trait comparison |
| **Rich training metadata** | Basic (method, notes) | ✅ Full (LoRA rank, LR, epochs, merge strategy) |
| **Outcross recommendations** | ✅ Diversity-scored partners | ✅ Criteria-based recommendations |
| **Best for** | "Is this merge genetically safe?" | "What hyperparameters produced this model?" |

**Rule of thumb:** Use Pedigree *before* a merge to check for inbreeding. Use Lineage Tracker *after* a fine-tune to record what happened. Use both for a complete breeding program.

## Advanced Scenario: Merge Planning

```python
from pedigree import Pedigree

p = Pedigree("lineage.json")

# You want to merge two community fine-tunes
# Check if they're too closely related first
model_a = "noromaid-70b"          # creative writing FT
model_b = "openchat-3.5-70b"      # conversation FT

coeff = p.check_inbreeding(model_a, model_b)
print(f"Inbreeding coefficient: {coeff:.2%}")

if coeff > 0.25:
    print("⚠ HIGH INBREEDING — merge will likely reinforce shared weaknesses")
    print("  Recommendations for outcrossing:")
    recs = p.recommend_breeding(model_a)
    for r in recs[:5]:
        print(f"    {r.model_id}: diversity={r.diversity_score:.2f}")
        print(f"      common ancestors: {r.common_ancestors}")
else:
    print("✓ Safe to merge — diverse enough bloodlines")
```

### Multi-Generation Visualization

```python
from pedigree import Pedigree

p = Pedigree("breeding-program.json")

# Register a 4-generation breeding program
p.register("llama-3-70b", method="pretrain", notes="Foundation")
p.register("instruct-v1", sire="llama-3-70b", method="sft", notes="Instruction tuning")
p.register("code-v1", sire="llama-3-70b", method="sft", notes="Code specialization")
p.register("reasoning-v1", sire="llama-3-70b", method="rlhf", notes="Reasoning focus")
p.register("instruct-code-v2", sire="instruct-v1", dam="code-v1", method="merge",
           notes="SLERP merge: instruction + code")
p.register("instruct-reasoning-v2", sire="instruct-v1", dam="reasoning-v1", method="merge",
           notes="SLERP merge: instruction + reasoning")
p.register("apex-v3", sire="instruct-code-v2", dam="instruct-reasoning-v2", method="merge",
           notes="Triple-merge: code + instruction + reasoning")

# Visualize the full tree
p.print_lineage("apex-v3", max_generations=5)
# ═══════════════════════════════════════════════════
#   Bloodline: apex-v3
# ═══════════════════════════════════════════════════
#   Gen 3: apex-v3 (merge)
#   ├── Gen 2: instruct-code-v2 (merge)
#   │   ├── Gen 1: instruct-v1 (sft)
#   │   │   └── Gen 0: llama-3-70b (pretrain)
#   │   └── Gen 1: code-v1 (sft)
#   │       └── Gen 0: llama-3-70b (pretrain)
#   └── Gen 2: instruct-reasoning-v2 (merge)
#       ├── Gen 1: instruct-v1 (sft)
#       │   └── Gen 0: llama-3-70b (pretrain)
#       └── Gen 1: reasoning-v1 (rlhf)
#           └── Gen 0: llama-3-70b (pretrain)
# ═══════════════════════════════════════════════════

# Check inbreeding on the final merge
coeff = p.check_inbreeding("instruct-code-v2", "instruct-reasoning-v2")
print(f"Pre-merge inbreeding: {coeff:.2%}")
# → 50.00% (both share instruct-v1 and llama-3-70b as ancestors)
# This is expected — they're half-siblings through instruct-v1

# Export for a presentation
p.export_dot("apex-v3", "breeding_tree.dot")
# dot -Tpng breeding_tree.dot -o breeding_tree.png
```

### Fleet Diversity Audit

```python
from pedigree import Pedigree

p = Pedigree("fleet.json")

# Get all models in the fleet
all_models = p.get_all_models()

# Find the foundation models (roots of all trees)
founders = [m for m in all_models if m.sire is None and m.dam is None]
print(f"Foundation models ({len(founders)}):")
for f in founders:
    descendants = p.get_descendants(f.id)
    print(f"  {f.id}: {len(descendants)} descendants")

# Check pairwise inbreeding across the fleet
print("\nFleet inbreeding matrix:")
for i, a in enumerate(all_models):
    for b in all_models[i+1:]:
        coeff = p.check_inbreeding(a.id, b.id)
        if coeff > 0.3:
            print(f"  ⚠ {a.id} × {b.id}: {coeff:.2%}")

# Recommend the most diverse pairings
print("\nMost diverse pairings for next breeding cycle:")
for model in all_models[:3]:
    recs = p.recommend_breeding(model)
    if recs:
        best = recs[0]
        print(f"  {model.id} × {best.model_id}: diversity={best.diversity_score:.2f}")
```

## The Inbreeding Coefficient Explained

The inbreeding coefficient is the most important number Pedigree computes. Here's what it means in practice:

| Coefficient | Animal Breeding Interpretation | Model Breeding Interpretation |
|-------------|-------------------------------|-------------------------------|
| **0%** | Completely unrelated | No shared base model ancestry |
| **6.25%** | First cousins once removed | Distant lineage overlap |
| **12.5%** | Half-siblings or uncle/niece | One shared parent model |
| **25%** | Full siblings or parent/child | Very close — will reinforce shared traits |
| **50%+** | Highly inbred (problematic) | Same base model, likely to amplify weaknesses |

**What this means for merges:** When you merge two models with a high inbreeding coefficient, you're not adding new capabilities — you're reinforcing what both models already share. This is good if what they share is a strength. It's catastrophic if what they share is a weakness (e.g., both can't do math, both hallucinate on dates, both share a tokenizer bug).

The outcross recommendation system finds models from *different* bloodlines — maximizing the chance that what one model lacks, the other provides. This is how animal breeders produce healthy offspring: not by breeding the best to the best, but by breeding complementary to complementary.

## License

MIT — see [LICENSE](LICENSE).
