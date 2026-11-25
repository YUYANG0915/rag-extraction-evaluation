#!/usr/bin/env python3
"""Build a descriptive table; protocols and budgets differ for the completed paper-linked attacks."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


MARKER = re.compile(r"Clinic-([0-9a-f]{12})", re.I)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def marker_map(corpus_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    id_to_marker: dict[str, str] = {}
    marker_to_id: dict[str, str] = {}
    for row in read_jsonl(corpus_path):
        match = MARKER.search(row["text"])
        if not match:
            raise RuntimeError(f"missing marker for {row['_id']}")
        marker = match.group(1).lower()
        id_to_marker[row["_id"]] = marker
        marker_to_id[marker] = row["_id"]
    return id_to_marker, marker_to_id


def attack_summary(name: str, rows: list[dict], id_to_marker: dict[str, str], marker_to_id: dict[str, str]) -> dict:
    retrieved: set[str] = set()
    disclosed: set[str] = set()
    slots = 0
    disclosed_slots = 0
    marker_mentions = 0
    for row in rows:
        ids = row.get("context_ids") or row.get("doc_ids") or []
        ids = [str(item) for item in ids]
        retrieved.update(ids)
        slots += len(ids)
        answer_markers = [m.lower() for m in MARKER.findall(row.get("answer", ""))]
        marker_mentions += len(answer_markers)
        answer_marker_set = set(answer_markers)
        disclosed.update(marker_to_id[m] for m in answer_marker_set if m in marker_to_id)
        disclosed_slots += sum(id_to_marker.get(doc_id) in answer_marker_set for doc_id in ids)
    result = {
        "attack": name,
        "queries": len(rows),
        "retrieved_unique_docs": len(retrieved),
        "retrieved_corpus_coverage": len(retrieved) / len(id_to_marker),
        "answer_disclosed_unique_docs": len(disclosed),
        "answer_disclosed_corpus_coverage": len(disclosed) / len(id_to_marker),
        "marker_mentions": marker_mentions,
        "retrieved_slots": slots,
        "retrieved_slot_marker_disclosure_rate": disclosed_slots / slots if slots else 0.0,
    }
    if name == "ragcrawler" and rows:
        result.update({
            "final_graph_nodes": rows[-1].get("graph_nodes"),
            "final_graph_edges": rows[-1].get("graph_edges"),
            "total_new_graph_items": sum(row.get("new_graph_items", 0) for row in rows),
        })
    return result


def defense_summary(root: Path) -> dict:
    output: dict[str, dict] = {}
    for attack in ["pide", "rtf", "tgtb"]:
        rows = read_jsonl(root / attack / "records.jsonl")
        output[attack] = {}
        for arm in sorted({row["arm"] for row in rows}):
            chosen = [row for row in rows if row["arm"] == arm]
            marker_ids = [marker for row in chosen for marker in row.get("marker_hit_ids", [])]
            output[attack][arm] = {
                "queries": len(chosen),
                "blocked_rate": sum(bool(row.get("blocked")) for row in chosen) / len(chosen),
                "marker_hits": len(marker_ids),
                "unique_marker_ids": len(set(marker_ids)),
            }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[3] / "supplementary/paper_baselines/analysis")
    args = parser.parse_args()
    project = args.project
    args.output.mkdir(parents=True, exist_ok=True)
    id_to_marker, marker_to_id = marker_map(project / "supplementary/paper_baselines/corpus.jsonl")
    baseline = project / "supplementary/paper_baselines"
    leakdojo = list((baseline / "ikea_por").glob("**/*.jsonl"))
    paths = {
        "ikea": next(path for path in leakdojo if path.name.startswith("IKEA_")),
        "por": next(path for path in leakdojo if path.name.startswith("POR_")),
        "dgea": baseline / "dgea/records.jsonl",
        "copybreakrag": baseline / "copybreak/records.jsonl",
        "ragcrawler": project / "supplementary/paper_baselines/ragcrawler/records.jsonl",
    }
    report = {
        "corpus_docs": len(id_to_marker),
        "metric_note": "Exact transformed Clinic marker disclosure; retrieved coverage is reported separately.",
        "attacks": {name: attack_summary(name, read_jsonl(path), id_to_marker, marker_to_id) for name, path in paths.items()},
        "defense_matrix": defense_summary(baseline / "defense_matrix"),
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Paper-linked attack and defense results",
        "",
        "Exact Clinic markers measure answer disclosure; retrieved-document coverage is shown separately. Protocols, budgets, and retrievers differ: this is not a matched ranking or an evaluation of the new defense. Blocking rates on attacks do not measure benign service quality.",
        "",
        "| Attack | Queries | Retrieved docs | Retrieved coverage | Disclosed docs | Disclosed coverage | Retrieved-slot disclosure |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, x in report["attacks"].items():
        lines.append(
            f"| {name} | {x['queries']} | {x['retrieved_unique_docs']} | {x['retrieved_corpus_coverage']:.3f} "
            f"| {x['answer_disclosed_unique_docs']} | {x['answer_disclosed_corpus_coverage']:.3f} "
            f"| {x['retrieved_slot_marker_disclosure_rate']:.3f} |"
        )
    lines += ["", "## Defense stress panel", "", "| Attack | Arm | Queries | Blocked rate | Marker hits | Unique markers |", "|---|---|---:|---:|---:|---:|"]
    for attack, arms in report["defense_matrix"].items():
        for arm, x in arms.items():
            lines.append(f"| {attack} | {arm} | {x['queries']} | {x['blocked_rate']:.3f} | {x['marker_hits']} | {x['unique_marker_ids']} |")
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
