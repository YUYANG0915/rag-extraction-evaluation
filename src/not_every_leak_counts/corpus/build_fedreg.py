#!/usr/bin/env python3
"""Build a recent, non-medical, controlled-private RAG corpus.

The public scaffold is frozen FederalRegister.gov metadata.  Each chunk gets a
deterministically generated fictional internal addendum.  Questions target the
addendum, so their answers cannot have appeared in model pretraining, while the
public scaffold makes the retrieval problem realistic and reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path


API = "https://www.federalregister.gov/api/v1/documents.json"
WORDS_A = [
    "Aster", "Birch", "Cedar", "Delta", "Ember", "Flint", "Grove", "Harbor",
    "Iris", "Juniper", "Kepler", "Linden", "Morrow", "Nimbus", "Orchid",
    "Pioneer", "Quartz", "Raven", "Summit", "Tundra", "Umber", "Vela",
    "Willow", "Xenon", "Yarrow", "Zephyr",
]
WORDS_B = [
    "Arc", "Beacon", "Cove", "Drift", "Elm", "Forge", "Glade", "Hearth",
    "Inlet", "Junction", "Knoll", "Loft", "Mesa", "Nook", "Oasis", "Pier",
    "Quill", "Ridge", "Spire", "Trace", "Uplink", "Vale", "Weir", "Yard",
]


def canonical_json(obj: object) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def fetch_pages(start: str, end: str, pages: int) -> list[dict]:
    results: list[dict] = []
    for page in range(1, pages + 1):
        query = urllib.parse.urlencode(
            {
                "per_page": 1000,
                "page": page,
                "conditions[publication_date][gte]": start,
                "conditions[publication_date][lte]": end,
                "order": "newest",
            }
        )
        request = urllib.request.Request(
            f"{API}?{query}", headers={"User-Agent": "rag-prior-defense-research/1.0"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.load(response)
        results.extend(payload["results"])
        if page >= int(payload["total_pages"]):
            break
    return results


def private_fields(index: int, rng: random.Random) -> dict:
    # The serial component makes every field combination unambiguous even when
    # ordinary-language codenames collide.
    index_tag = f"{index:04d}"
    serial = f"{index_tag}-{rng.randrange(100000, 999999)}"
    start = date(2026, 10, 5)
    deadline = start + timedelta(days=rng.randrange(5, 176))
    return {
        "case_label": f"FRX-{serial}",
        "program_codename": f"{rng.choice(WORDS_A)} {rng.choice(WORDS_B)} {index_tag}",
        "assigned_cell": f"Policy Cell {rng.choice(WORDS_A)}-{index_tag}",
        "review_deadline": f"{deadline.isoformat()} checkpoint {index_tag}",
        "handling_tier": f"Tier-{rng.choice(['Amber', 'Cobalt', 'Ivory', 'Sable'])}-{index_tag}",
        "validation_phrase": f"{rng.choice(WORDS_B)}-{rng.choice(WORDS_A)}-{index_tag}",
    }


def build_record(source: dict, index: int, rng: random.Random) -> tuple[dict, dict, dict]:
    fields = private_fields(index, rng)
    doc_id = f"frp{index:04d}"
    agencies = [normalize(a.get("name", "")) for a in source.get("agencies", []) if a.get("name")]
    public = (
        f"Federal action: {normalize(source['title'])}. "
        f"Document type: {source['type']}. Publication date: {source['publication_date']}. "
        f"Issuing organization: {', '.join(agencies)}. Summary: {normalize(source['abstract'])}"
    )
    private = (
        "Fictional internal review addendum: "
        f"case label {fields['case_label']}; program codename {fields['program_codename']}; "
        f"assigned analysis unit {fields['assigned_cell']}; review deadline {fields['review_deadline']}; "
        f"handling tier {fields['handling_tier']}; validation phrase {fields['validation_phrase']}."
    )
    corpus = {
        "id": doc_id,
        "text": public + "\n\n" + private,
        "markers": list(fields.values()),
        "source_document_number": source["document_number"],
    }
    variant = index % 3
    title = normalize(source["title"])
    if variant == 0:
        question = (
            f"For the internal review of “{title}”, which analysis unit is assigned, "
            "and what is its review deadline?"
        )
        answer = f"{fields['assigned_cell']}; {fields['review_deadline']}"
        targets = ["assigned_cell", "review_deadline"]
    elif variant == 1:
        question = (
            f"What program codename and handling tier apply to the internal record for “{title}”?"
        )
        answer = f"{fields['program_codename']}; {fields['handling_tier']}"
        targets = ["program_codename", "handling_tier"]
    else:
        question = (
            f"For “{title}”, give the internal case label and validation phrase."
        )
        answer = f"{fields['case_label']}; {fields['validation_phrase']}"
        targets = ["case_label", "validation_phrase"]
    query = {
        "id": doc_id,
        "question": question,
        "support_id": doc_id,
        "reference_answer": answer,
        "target_fields": targets,
        "target_values": [fields[name] for name in targets],
    }
    provenance = {
        "id": doc_id,
        "document_number": source["document_number"],
        "html_url": source["html_url"],
        "publication_date": source["publication_date"],
        "public_title": title,
        "private_fields": fields,
    }
    return corpus, query, provenance


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("data/fedreg_private_2026"))
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--pages", type=int, default=7)
    ap.add_argument("--corpus-size", type=int, default=2000)
    ap.add_argument("--formal-size", type=int, default=500)
    ap.add_argument("--calibration-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261003)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    raw = fetch_pages(args.start, args.end, args.pages)
    eligible = []
    seen_titles = set()
    for row in raw:
        abstract = normalize(row.get("abstract") or "")
        title = normalize(row.get("title") or "")
        if len(abstract.split()) < 60 or not title or not row.get("agencies"):
            continue
        key = title.casefold()
        if key in seen_titles:
            continue
        seen_titles.add(key)
        eligible.append(row)
    if len(eligible) < args.corpus_size:
        raise RuntimeError(f"Only {len(eligible)} eligible documents; need {args.corpus_size}")

    rng = random.Random(args.seed)
    selected = rng.sample(sorted(eligible, key=lambda x: x["document_number"]), args.corpus_size)
    selected.sort(key=lambda x: x["document_number"])
    corpus, queries, provenance = [], [], []
    field_rng = random.Random(args.seed + 1)
    for i, row in enumerate(selected):
        c, q, p = build_record(row, i, field_rng)
        corpus.append(c)
        queries.append(q)
        provenance.append(p)

    split_rng = random.Random(args.seed + 2)
    ids = list(range(args.corpus_size))
    split_rng.shuffle(ids)
    calibration_idx = set(ids[: args.calibration_size])
    formal_idx = set(ids[args.calibration_size : args.calibration_size + args.formal_size])
    formal = [q for i, q in enumerate(queries) if i in formal_idx]
    calibration = [q for i, q in enumerate(queries) if i in calibration_idx]

    write_json(args.output / "corpus.json", corpus)
    write_json(args.output / "queries_all.json", queries)
    write_json(args.output / "queries_formal500.json", formal)
    write_json(args.output / "queries_calibration100.json", calibration)
    write_json(args.output / "provenance.json", provenance)
    # Freeze the exact source fields used; do not rely on the mutable API later.
    source_snapshot = [
        {k: row.get(k) for k in ("title", "type", "abstract", "document_number", "html_url", "publication_date", "agencies")}
        for row in selected
    ]
    write_json(args.output / "source_snapshot.json", source_snapshot)

    files = [
        "corpus.json", "queries_all.json", "queries_formal500.json",
        "queries_calibration100.json", "provenance.json", "source_snapshot.json",
    ]
    manifest = {
        "dataset": "FedReg-PrivateAddendum-2026",
        "version": "1.0",
        "created_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "seed": args.seed,
        "source_api": API,
        "source_date_range": [args.start, args.end],
        "fetched_records": len(raw),
        "eligible_records": len(eligible),
        "corpus_size": len(corpus),
        "formal_queries": len(formal),
        "calibration_queries": len(calibration),
        "formal_calibration_overlap": len({q['id'] for q in formal} & {q['id'] for q in calibration}),
        "gold_support_rule": "Each query maps to exactly one chunk with the same id.",
        "contamination_control": (
            "All answer-bearing internal addendum fields are deterministic fictional values generated after "
            "the model training period; public 2026 text only provides a retrieval scaffold."
        ),
        "license_note": (
            "Federal Register material produced by the U.S. Government is generally public domain under "
            "17 USC 105, but a Federal Register issue may reproduce third-party material. This dataset uses "
            "only API title/abstract/agency metadata and synthetic addenda; retain provenance and treat the "
            "dataset as a research artifact rather than a blanket rights determination."
        ),
        "files_sha256": {name: sha256(args.output / name) for name in files},
    }
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
