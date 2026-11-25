"""Precompute QueryBlock decisions; decisions depend only on frozen queries."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def request(url, system, prompt):
    payload = {
        "model": "attacker",
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": 20261007,
        "max_tokens": 2,
    }
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as response:
        choice = json.load(response)["choices"][0]
    text = choice["message"].get("content", "").strip()
    if not text or choice["finish_reason"] not in {"stop", "length"}:
        raise RuntimeError("incomplete QueryBlock decision")
    return text, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--queries-root", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    system = (args.prompts / "query_block_system.txt").read_text()
    template = (args.prompts / "query_block_template.txt").read_text()
    records = args.out / "queryblock.jsonl"
    completed = set()
    if records.exists():
        for line in records.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                completed.add((row["attack"], int(row["turn"])))
    attack_tasks = []
    for attack in ("tgtb", "pide", "rtf"):
        source = args.queries_root / attack / "records.jsonl"
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        if len(rows) != 200 or any(row["attack"] != attack for row in rows):
            raise RuntimeError(f"invalid frozen {attack} query set")
        attack_tasks.extend((attack, turn, row) for turn, row in enumerate(rows)
                            if (attack, turn) not in completed)
    with records.open("a", buffering=1) as handle, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(request, args.url, system, template.format(prompt=row["query"])): (attack, turn, row)
                   for attack, turn, row in attack_tasks}
        for index, future in enumerate(as_completed(futures), 1):
            attack, turn, row = futures[future]
            verdict, seconds = future.result()
            handle.write(json.dumps({
                "attack": attack, "turn": turn, "query": row["query"],
                "verdict": verdict, "blocked": not verdict.lower().startswith("y"),
                "detector_seconds": seconds,
            }, ensure_ascii=False) + "\n")
            print(f"QUERYBLOCK {index}/{len(attack_tasks)}", flush=True)
    final_rows = [json.loads(line) for line in records.read_text().splitlines() if line.strip()]
    keys = {(row["attack"], int(row["turn"])) for row in final_rows}
    if len(final_rows) != 600 or len(keys) != 600:
        raise RuntimeError(f"expected 600 unique QueryBlock decisions, got {len(final_rows)} / {len(keys)}")
    (args.out / "summary.json").write_text(json.dumps({
        "status": "QUERYBLOCK_PRECOMPUTE_COMPLETE",
        "records": 600,
        "request_workers": args.workers,
        "queries_sha256": {
            attack: hashlib.sha256((args.queries_root / attack / "records.jsonl").read_bytes()).hexdigest()
            for attack in ("tgtb", "pide", "rtf")
        },
        "prompt_sha256": {
            name: hashlib.sha256((args.prompts / name).read_bytes()).hexdigest()
            for name in ("query_block_system.txt", "query_block_template.txt")
        },
    }, indent=2) + "\n")

    # Utility-side evaluation for the same paper baseline and frozen detector.
    # These benign query sets were fixed before this detector run.
    benign_sets = [
        ("fedreg500", args.queries_root.parents[1] / "data/fedreg_private_2026/queries_formal500.json"),
        ("hcm200", args.queries_root.parents[1] / "data/hcm_formal200/queries.json"),
        ("hcm300_extension", args.queries_root.parents[1] / "data/hcm_formal300_extension/queries.json"),
    ]
    benign_path = args.out / "queryblock_benign.jsonl"
    benign_done = set()
    if benign_path.exists():
        for line in benign_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                benign_done.add((row["set"], row["id"]))
    benign_tasks = []
    for set_name, source in benign_sets:
        rows = json.loads(source.read_text())
        for index, row in enumerate(rows):
            identifier = str(row.get("id", index))
            if (set_name, identifier) not in benign_done:
                benign_tasks.append((set_name, identifier, row))
    with benign_path.open("a", buffering=1) as handle, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(request, args.url, system, template.format(prompt=row["question"])): (set_name, identifier, row)
                   for set_name, identifier, row in benign_tasks}
        for index, future in enumerate(as_completed(futures), 1):
            set_name, identifier, row = futures[future]
            verdict, seconds = future.result()
            handle.write(json.dumps({
                "set": set_name, "id": identifier, "question": row["question"],
                "verdict": verdict, "blocked": not verdict.lower().startswith("y"),
                "detector_seconds": seconds,
            }, ensure_ascii=False) + "\n")
            print(f"QUERYBLOCK_BENIGN {index}/{len(benign_tasks)}", flush=True)
    benign_rows = [json.loads(line) for line in benign_path.read_text().splitlines() if line.strip()]
    benign_keys = {(row["set"], row["id"]) for row in benign_rows}
    if len(benign_rows) != 1000 or len(benign_keys) != 1000:
        raise RuntimeError(f"expected 1000 unique benign decisions, got {len(benign_rows)} / {len(benign_keys)}")
    (args.out / "benign_summary.json").write_text(json.dumps({
        "status": "QUERYBLOCK_BENIGN_UTILITY_COMPLETE",
        "records": 1000,
        "request_workers": args.workers,
        "sets": {
            name: {
                "n": sum(row["set"] == name for row in benign_rows),
                "blocked": sum(row["set"] == name and row["blocked"] for row in benign_rows),
            }
            for name, _ in benign_sets
        },
        "source_sha256": {
            name: hashlib.sha256(source.read_bytes()).hexdigest() for name, source in benign_sets
        },
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
