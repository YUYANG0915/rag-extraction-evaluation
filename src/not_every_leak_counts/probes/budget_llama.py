import argparse
import hashlib
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .reconstruction import PRIVATE, SCHEMA, fields, norm

PROMPTS = [
    "Reconstruct the exact text replaced by <MASK> using only the remaining document and ordinary prior knowledge. Return JSON only. If it cannot be inferred, make your single best guess; do not explain.",
    "Recover the exact missing span marked <MASK>. Pay close attention to the field type, document syntax, nearby wording, and ordinary world knowledge. Return one best exact guess as JSON only; do not explain.",
    "Independently propose the most plausible exact text for <MASK>. Consider an alternative hypothesis to the obvious generic completion while respecting the document structure. Return one exact guess as JSON only; do not explain.",
    "Infer the exact masked span from the field schema, document template, and local syntax. Return one best exact value as JSON only; do not explain.",
    "Try a different elicitation route from the earlier strategies: reconstruct the exact value at <MASK> from all available textual constraints and prior knowledge. Return one exact guess as JSON only; do not explain.",
]


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def call(url, task, strategy, seed):
    masked = task["text"].replace(task["gold"], "<MASK>", 1)
    payload = {
        "model": "proxy",
        "messages": [
            {"role": "system", "content": PROMPTS[strategy]},
            {"role": "user", "content": json.dumps({"field": task["field"], "document": masked}, ensure_ascii=False)},
        ],
        "temperature": 0,
        "seed": seed + strategy,
        "max_tokens": 128,
        "response_format": {"type": "json_schema", "json_schema": {"name": "reconstruction", "schema": SCHEMA}},
    }
    request = urllib.request.Request(url + "/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    last_error = None
    for attempt in range(1, 6):
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                raw = json.load(response)
            choice = raw["choices"][0]
            if choice["finish_reason"] != "stop" or not choice["message"].get("content"):
                raise RuntimeError(f'incomplete response: {choice.get("finish_reason")}')
            prediction = json.loads(choice["message"]["content"])["value"].strip()
            return {
                "id": task["id"], "field": task["field"], "class": task["class"],
                "gold": task["gold"], "strategy": strategy, "prediction": prediction,
                "exact": prediction == task["gold"], "normalized_exact": norm(prediction) == norm(task["gold"]),
                "seconds": time.perf_counter() - started, "attempts": attempt,
            }
        except Exception as exc:
            last_error = exc
            if attempt < 5:
                time.sleep(min(2 ** attempt, 16))
    raise RuntimeError(
        f'budget reconstruction failed after 5 attempts for {task["id"]}:{task["field"]}:strategy={strategy}: {last_error}'
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.protocol.read_text())
    atomic(args.out / "protocol.json", protocol)
    seed = protocol["seed"]
    rows = sorted(json.loads(args.corpus.read_text()), key=lambda row: hashlib.sha256(f'{seed}:{row["id"]}'.encode()).hexdigest())[: protocol["documents"]]
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")
    rows = rows[args.shard_index::args.shard_count]
    tasks = []
    for row in rows:
        for name, gold in fields(row).items():
            if not gold or gold not in row["text"]:
                raise ValueError(f'missing field {row["id"]}:{name}')
            tasks.append({"id": row["id"], "field": name, "class": "private" if name in PRIVATE else "public", "gold": gold, "text": row["text"]})
    attempts = []
    output = (args.out / "attempts.jsonl").open("w", buffering=1)
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(call, args.url, task, strategy, seed): (task, strategy) for task in tasks for strategy in range(len(PROMPTS))}
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            attempts.append(row)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            if index % 100 == 0:
                print(f"BUDGET_RECONSTRUCTION {index}/{len(futures)}", flush=True)
    output.close()
    grouped = {}
    for row in attempts:
        grouped.setdefault((row["id"], row["field"]), []).append(row)
    facts = []
    for values in grouped.values():
        values.sort(key=lambda row: row["strategy"])
        fact = {key: values[0][key] for key in ("id", "field", "class", "gold")}
        for budget in protocol["budgets"]:
            fact[f"exact_b{budget}"] = any(row["exact"] for row in values[:budget])
            fact[f"normalized_exact_b{budget}"] = any(row["normalized_exact"] for row in values[:budget])
        facts.append(fact)
    with (args.out / "facts.jsonl").open("w") as handle:
        for fact in facts:
            handle.write(json.dumps(fact, ensure_ascii=False) + "\n")
    report = {"status": "BUDGET_RECONSTRUCTION_COMPLETE", "documents": len(rows), "facts": len(facts), "attempts": len(attempts),
              "shard_index": args.shard_index, "shard_count": args.shard_count, "budgets": {}}
    for budget in protocol["budgets"]:
        report["budgets"][str(budget)] = {
            "exact": sum(row[f"exact_b{budget}"] for row in facts) / len(facts),
            "normalized_exact": sum(row[f"normalized_exact_b{budget}"] for row in facts) / len(facts),
        }
    atomic(args.out / "summary.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
