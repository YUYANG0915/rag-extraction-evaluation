"""Run licensed LeakDojo IKEA and PoR protocol ports with shared services."""
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import torch


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def ready(process, url, model):
    deadline = time.monotonic() + 7200
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{model} service exited {process.returncode}")
        try:
            with urllib.request.urlopen(url + "/models", timeout=5) as response:
                if any(item["id"] == model for item in json.load(response)["data"]):
                    return
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError(f"{model} readiness timeout")


def main():
    root = Path(os.environ["RAG_EXPERIMENT_ROOT"])
    models = json.loads(Path(os.environ["RAG_MODELS_JSON"]).read_text())
    job = os.environ["SLURM_JOB_ID"]
    group = os.environ.get("BASELINE_GROUP", "adaptive")
    attacks = {"adaptive": ["por", "ikea"], "ikea_only": ["ikea"]}[group]
    run = root / "results" / f"leakdojo-{group}-{job}-{time.time_ns()}"
    run.mkdir(parents=True)
    work = run / "leakdojo"
    shutil.copytree(root / "external/leakdojo", work)
    (work / "data/hcm").mkdir(parents=True)
    shutil.copy2(root / "data/hcm_leakdojo1000/corpus.jsonl", work / "data/hcm/corpus.jsonl")

    # LeakDojo resumes from the number of records already present in its
    # JSONL output. Seed a restarted allocation with the longest prior PoR
    # trajectory so scheduler cancellation does not discard completed calls.
    prior_outputs = []
    for path in root.glob("results/leakdojo-adaptive-*/leakdojo/results/**/*por_attack.jsonl"):
        try:
            count = sum(1 for _ in path.open(encoding="utf-8"))
        except OSError:
            continue
        if count:
            prior_outputs.append((count, path))
    if prior_outputs:
        resume_count, resume_file = max(prior_outputs, key=lambda item: item[0])
        resume_root = next(
            parent for parent in resume_file.parents
            if parent.name == "results" and parent.parent.name == "leakdojo"
        )
        shutil.copytree(resume_root, work / "results", dirs_exist_ok=True)
        print(f"ADAPTIVE_RESUME {resume_count} {resume_file}", flush=True)
    print(f"RUN_DIR={run}", flush=True)

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2").split(",")
    if len(visible) < 2:
        raise RuntimeError("adaptive baseline group requires generator and attacker GPUs")
    generator_count = int(os.environ.get("ADAPTIVE_GENERATOR_GPUS", "2" if len(visible) >= 3 else "1"))
    if generator_count < 1 or generator_count >= len(visible):
        raise RuntimeError("invalid ADAPTIVE_GENERATOR_GPUS split")
    small_gpu = min(torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())) < 150 * 1024**3
    processes = []
    handles = []
    urls = {}
    try:
        for model, devices in (("generator", visible[:generator_count]), ("attacker", visible[generator_count:])):
            port = free_port()
            urls[model] = f"http://127.0.0.1:{port}/v1"
            command = ["vllm", "serve", models[model]["path"], "--served-model-name", model,
                       "--host", "127.0.0.1", "--port", str(port), "--tensor-parallel-size", str(len(devices)),
                       "--max-model-len", "16384", "--max-num-seqs", "8", "--max-num-batched-tokens", "8192",
                       "--gpu-memory-utilization", "0.90", "--generation-config", "vllm",
                       "--no-enable-prefix-caching"]
            if model == "generator":
                command += ["--reasoning-parser", "mistral", "--language-model-only", "--config-format", "mistral",
                            "--load-format", "mistral", "--tokenizer-mode", "mistral"]
                if small_gpu or (len(devices) == 1 and os.environ.get("RAG_GENERATOR_FP8") == "1"):
                    command += ["--quantization", "fp8"]
            else:
                command += ["--config-format", "hf", "--load-format", "safetensors"]
                if small_gpu:
                    command += ["--quantization", "fp8"]
                # vLLM 0.30 cannot combine its UVA CPU offloader with online
                # FP8 initialization (parameters are still meta tensors).
                # The FP8 attacker fits on a 96 GiB-class GPU without it.
                offload = None if small_gpu else os.environ.get("RAG_ATTACKER_CPU_OFFLOAD_GB")
                if offload:
                    command += ["--cpu-offload-gb", offload]
            handle = (run / f"{model}.log").open("w")
            handles.append(handle)
            process = subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(devices)),
                                       stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(process)
            ready(process, urls[model], model)

        router_port = free_port()
        router = subprocess.Popen([sys.executable, str(root / "src/not_every_leak_counts/baselines/leakdojo_router.py"), "--port", str(router_port),
                                   "--generator", urls["generator"], "--attacker", urls["attacker"]],
                                  stdout=(run / "router.log").open("w"), stderr=subprocess.STDOUT,
                                  start_new_session=True)
        processes.append(router)
        router_url = f"http://127.0.0.1:{router_port}/v1"
        time.sleep(2)

        environment = dict(os.environ, PYTHONPATH=str(work),
                           LEAKDOJO_EMBEDDER=models["embedder"]["path"],
                           LEAKDOJO_RERANKER=models["reranker"]["path"])
        commands = []
        for attack in attacks:
            command = [sys.executable, "main.py", "--cfg_name", "hcm", "--attack", attack,
                       "--attack_num", "200", "--batch_size", "1", "--device", "cpu",
                       "--llm_model", "generator", "--llm_base_url", router_url,
                       "--llm_api_key", "EMPTY", "--llm_temperature", "0", "--llm_top_p", "1",
                       "--llm_max_gen_len", "1536", "--reranker"]
            log = (run / f"{attack}.log").open("w")
            handles.append(log)
            commands.append(command)
            subprocess.run(command, cwd=work, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        (run / "summary.json").write_text(json.dumps({
            "status": "LEAKDOJO_ADAPTIVE_PROTOCOL_PORT_COMPLETE", "attacks": attacks,
            "queries_per_attack": 200, "licensed_source": "LeakDojo Apache-2.0 commit ae3ac6d",
            "substitutions": ["HCM transformed corpus", "Mistral target", "Llama attacker",
                              "Snowflake embedder", "Mixedbread reranker"],
            "commands": commands,
        }, indent=2) + "\n")
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in reversed(processes):
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()
