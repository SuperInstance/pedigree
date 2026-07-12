"""Render lineage as ASCII tree and DOT graph."""

from __future__ import annotations

from typing import Optional

from .model import LineageTree
from .store import LineageStore


def render_ascii(tree: LineageTree, max_depth: int = 10) -> str:
    """Render a LineageTree as an ASCII bloodline chart.

    Example output:
        └── noromaid-70b [sft]
            └── llama-3-70b [pretrain]
    """
    # The tree's root has no connector — start with a clean root
    lines: list[str] = []

    method_tag = f" [{tree.model.method}]" if tree.model.method else ""
    lines.append(f"{tree.model.id}{method_tag}")

    children = []
    if tree.sire_tree:
        children.append(("sire", tree.sire_tree))
    if tree.dam_tree:
        children.append(("dam", tree.dam_tree))

    for i, (role, child) in enumerate(children):
        is_last = i == len(children) - 1
        sub = child.to_ascii(prefix="", is_last=is_last, depth=1, max_depth=max_depth)
        if sub:
            # Prefix role tag
            tagged = _tag_first_line(sub, f"({role}) ")
            lines.append(tagged)

    return "\n".join(lines)


def _tag_first_line(text: str, tag: str) -> str:
    """Insert a role tag after the tree connector on the first line."""
    lines = text.split("\n")
    if not lines:
        return text
    # Find where the model name starts (after connectors)
    first = lines[0]
    # Insert tag after the connector characters
    idx = 0
    while idx < len(first) and first[idx] in "├└│─ ":
        idx += 1
    lines[0] = first[:idx] + tag + first[idx:]
    return "\n".join(lines)


def render_dot(tree: LineageTree, title: str = "Model Lineage") -> str:
    """Render a LineageTree as a GraphViz DOT graph.

    Produces a left-to-right digraph with method-labeled edges.
    """
    lines = [
        'digraph lineage {',
        '  rankdir=LR;',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica"];',
        f'  label="{title}";',
        '  labelloc=t;',
        '  fontsize=14;',
        '',
    ]

    edges: set[tuple[str, str, str]] = set()
    nodes: set[str] = set()
    _collect_dot(tree, nodes, edges)

    # Node declarations with styling
    for node_id in sorted(nodes):
        is_root = node_id == tree.model.id
        fill = "#4CAF50" if is_root else "#E1BEE7"
        fontcolor = "white" if is_root else "black"
        lines.append(
            f'  "{node_id}" [fillcolor="{fill}", '
            f'fontcolor="{fontcolor}"];'
        )

    lines.append("")

    # Edges with method labels
    for sire, child, method in sorted(edges):
        label = f' [label="{method}"]' if method else ""
        lines.append(f'  "{sire}" -> "{child}"{label};')

    lines.append("}")

    return "\n".join(lines)


def _collect_dot(tree: LineageTree, nodes: set[str],
                 edges: set[tuple[str, str, str]]) -> None:
    """Recursively collect nodes and edges for DOT output."""
    nodes.add(tree.model.id)
    if tree.sire_tree:
        method = tree.model.method or ""
        edges.add((tree.sire_tree.model.id, tree.model.id, method))
        _collect_dot(tree.sire_tree, nodes, edges)
    if tree.dam_tree:
        method = tree.model.method or ""
        edges.add((tree.dam_tree.model.id, tree.model.id, method))
        _collect_dot(tree.dam_tree, nodes, edges)
