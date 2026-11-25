"""DGEA protocol port using the frozen Snowflake embedding space."""
import argparse
import json
from pathlib import Path
import random
import time
import urllib.request

import numpy as np


def call(url, model, prompt, temperature=0, max_tokens=1536):
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "temperature": temperature, "seed": 20261007, "max_tokens": max_tokens}
    if model == "generator":
        payload["reasoning_effort"] = "none"
    request = urllib.request.Request(url + "/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=900) as response:
        result = json.load(response)
    choice = result["choices"][0]
    text = choice["message"].get("content", "").strip()
    if not text or choice["finish_reason"] not in {"stop", "length"}:
        raise RuntimeError("incomplete response")
    return text


class ToolLLM:
    def __init__(self, url):
        self.url = url

    def infer(self, prompt, temperature=None, top_p=None, sysprompt=None):
        body = prompt if sysprompt is None else sysprompt + "\n\n" + prompt
        return call(self.url, "attacker", body, 0 if temperature is None else temperature), None

    def batch_infer(self, prompts, temperature=None, top_p=None, sysprompt=None):
        pairs = [self.infer(prompt, temperature, top_p, sysprompt) for prompt in prompts]
        return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def main():
    import pandas as pd
    import torch
    from sentence_transformers import CrossEncoder
    from src.skuas import DGEAQueryGenerator

    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-url", required=True)
    parser.add_argument("--attacker-url", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--count", type=int, default=200)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    models = json.loads(args.models.read_text())
    rows = [json.loads(line) for line in args.corpus.read_text().splitlines()]
    texts = [row["text"] for row in rows]
    tool = ToolLLM(args.attacker_url)

    # This distribution is the DGEA target distribution, recomputed in the
    # frozen non-Chinese embedding space rather than reusing incompatible BGE
    # statistics from the public implementation.
    from src.components.utils import get_embed_model
    embedding = get_embed_model("hf", models["embedder"]["path"], device=args.device)
    corpus_vectors = np.asarray(embedding.embed_documents(texts), dtype=np.float32)
    stats = args.out / "snowflake_embedding_statistics.csv"
    pd.DataFrame({"mean": corpus_vectors.mean(axis=0),
                  "variance": corpus_vectors.var(axis=0)}).to_csv(stats, index=False)

    dgea = DGEAQueryGenerator(tool, vectors_num=args.count, device=args.device,
                              embedding_model_name=models["embedder"]["path"],
                              df_csv_path=str(stats))
    # Reuse the already computed vectors and the exact same Snowflake model.
    reranker = CrossEncoder(models["reranker"]["path"], device="cpu")
    history = []
    if args.resume:
        for line in args.resume.read_text().splitlines():
            if line.strip():
                history.append(json.loads(line))
        if len(history) > args.count:
            raise ValueError(f"resume has {len(history)} records, count is {args.count}")
    output_path = args.out / "records.jsonl"
    with output_path.open("w") as copied:
        for record in history:
            copied.write(json.dumps(record, ensure_ascii=False) + "\n")
    output = output_path.open("a", buffering=1)
    embedding_space = []
    last_query = dgea.prefix
    for record in history:
        contents = dgea.extract_or_fetch_content(record["answer"])
        embedding_space = dgea.embed_and_store_unique_contents(contents, embedding_space)
        last_query = record["query"]
    if history:
        print(f"DGEA_RESUME {len(history)} {args.resume}", flush=True)
    try:
        for index in range(len(history), args.count):
            started = time.perf_counter()
            if index == 0:
                target = torch.tensor(dgea.Vectors[index], device=args.device)
            else:
                target = dgea.get_next_target(index=index, embedding_space=embedding_space,
                                              last_query=last_query,
                                              embedding_model=dgea.embedding_model,
                                              vector_dim=len(dgea.Vectors[0]), device=args.device)
                target = torch.as_tensor(target, dtype=torch.float32, device=args.device)
            suffix, best_loss, _ = dgea.gcqAttack(target, iterations=3, topk=128)
            query = dgea.prefix + " " + suffix
            query_vector = np.asarray(dgea.embedding_model.embed_query(query), dtype=np.float32)
            query_vector /= max(np.linalg.norm(query_vector), 1e-12)
            normalized = corpus_vectors / np.maximum(np.linalg.norm(corpus_vectors, axis=1, keepdims=True), 1e-12)
            indices = np.argsort(-(normalized @ query_vector), kind="stable")[:20]
            candidates = [texts[int(i)] for i in indices]
            scores = np.asarray(reranker.predict([(query, text) for text in candidates],
                                                 activation_fn=torch.nn.Sigmoid())).reshape(-1)
            order = np.argsort(-scores, kind="stable")[:5]
            selected = [int(indices[int(i)]) for i in order]
            context = "\n".join(texts[i] for i in selected)
            answer = call(args.generator_url, "generator", f"context: {context}\n\nquestion: {query}\n\nanswer:")
            contents = dgea.extract_or_fetch_content(answer)
            embedding_space = dgea.embed_and_store_unique_contents(contents, embedding_space)
            output.write(json.dumps({
                "attack": "dgea", "turn": index, "query": query, "best_loss": best_loss,
                "context_ids": [rows[i]["_id"] for i in selected], "answer": answer,
                "seconds": time.perf_counter() - started,
            }, ensure_ascii=False) + "\n")
            last_query = query
            print(f"DGEA_PROTOCOL_PORT {index + 1}/{args.count}", flush=True)
    finally:
        output.close()
    (args.out / "summary.json").write_text(json.dumps({
        "status": "DGEA_PROTOCOL_PORT_COMPLETE", "queries": args.count,
        "licensed_source": "LeakDojo Apache-2.0 commit ae3ac6d",
        "algorithm": {"iterations": 3, "topk": 128},
        "declared_substitution": "Snowflake corpus-estimated embedding distribution replaces BGE statistics",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
