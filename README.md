# Pedigree: Model Lineage Tracking as Bloodline Records

> Tracks model bloodlines like a breeder tracks dog pedigrees.

Part of the **Working Animal Architecture** — treating AI models like working animals with traceable lineage, breeding records, and genetic diversity tracking.

## Why?

Every model has a family tree. Base models beget fine-tunes. Fine-tunes beget merges. Merges beget quantizations. Adapters stack on top of adapters. After three layers nobody remembers who descended from whom.

Pedigree treats this seriously. Register every model birth. Track sire (base) and dam (training data lineage). Detect inbreeding before you merge two cousins. Recommend fresh blood for outcrossing.

## Features

- **Pedigree registration** — every model gets a lineage record (sire, dam, birth date, method)
- **Lineage trees** — trace ancestry N generations back
- **Inbreeding coefficient** — Wright's coefficient of relationship, adapted for model lineage
- **Breeding recommendations** — find outcross candidates to maximize genetic diversity
- **ASCII visualization** — print bloodline trees in the terminal
- **GraphViz DOT export** — render publication-quality lineage diagrams
- **JSON store** — simple file-based persistence, no database required

## Install

```bash
pip install pedigree-tracker
```

## Quick Start

```python
from pedigree import Pedigree

p = Pedigree("lineage.json")

# Register a foundation model
p.register("llama-3-70b", method="pretrain", notes="Meta foundation model")

# Register a fine-tune (sire = base model)
p.register("llama-3-70b-chat", sire="llama-3-70b", method="sft",
           notes="Official instruction-tuned variant")

# Register a community fine-tune
p.register("noromaid-70b", sire="llama-3-70b", method="sft",
           notes="Sao10K creative writing fine-tune")

# Check inbreeding before merging
coeff = p.check_inbreeding("llama-3-70b-chat", "noromaid-70b")
print(f"Inbreeding coefficient: {coeff:.2%}")
# → 50.00% (half-siblings sharing llama-3-70b as sire)

# Get recommendations for outcrossing
recs = p.recommend_breeding("llama-3-70b-chat")
for r in recs:
    print(f"{r.model_id}: diversity={r.diversity_score:.2f}")

# Visualize
p.print_lineage("noromaid-70b")      # ASCII tree
p.export_dot("noromaid-70b", "lineage.dot")  # GraphViz
```

## Breeding Methods

| Method | Description |
|--------|-------------|
| `pretrain` | Foundation model trained from scratch |
| `sft` | Supervised fine-tuning |
| `rlhf` | Reinforcement learning from human feedback |
| `merge` | Model merging (SLERP, DARE, TIES, etc.) |
| `quantize` | Quantization (GGUF, AWQ, GPTQ) |
| `adapter` | LoRA / QLoRA adapter |
| `distill` | Knowledge distillation |
| `continue_pretrain` | Continued pre-training |

## License

MIT
