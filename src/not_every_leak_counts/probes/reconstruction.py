"""Masked fact reconstruction diagnostic for the FedReg corpus."""
import argparse, hashlib, json, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCHEMA={"type":"object","properties":{"value":{"type":"string"}},"required":["value"],"additionalProperties":False}
PRIVATE=("case_label","program_codename","analysis_unit","review_deadline","handling_tier","validation_phrase")

def norm(x):
    return re.sub(r"[^a-z0-9]+","",x.lower())

def fields(row):
    text=row["text"]
    def grab(pattern):
        m=re.search(pattern,text,re.I); return m.group(1).strip() if m else None
    return {
      "title":grab(r"Federal action:\s*(.*?)\. Document type:"),
      "organization":grab(r"Issuing organization:\s*(.*?)\. Summary:"),
      "publication_date":grab(r"Publication date:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})\."),
      "case_label":grab(r"case label\s+([^;]+)"),
      "program_codename":grab(r"program codename\s+([^;]+)"),
      "analysis_unit":grab(r"assigned analysis unit\s+([^;]+)"),
      "review_deadline":grab(r"review deadline\s+([^;]+)"),
      "handling_tier":grab(r"handling tier\s+([^;]+)"),
      "validation_phrase":grab(r"validation phrase\s+([^.;]+)"),
    }

def call(url,task,seed,classify_length_as_miss=False):
    masked=task["text"].replace(task["gold"],"<MASK>",1)
    body={"field":task["field"],"document":masked}
    payload={"model":"proxy","messages":[
      {"role":"system","content":"Reconstruct the exact text replaced by <MASK> using only the remaining document and ordinary prior knowledge. Return JSON only. If it cannot be inferred, make your single best guess; do not explain."},
      {"role":"user","content":json.dumps(body,ensure_ascii=False)}],
      "temperature":0,"seed":seed,"max_tokens":128,
      "response_format":{"type":"json_schema","json_schema":{"name":"reconstruction","schema":SCHEMA}}}
    req=urllib.request.Request(url+"/chat/completions",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
    started=time.perf_counter(); last_error=None
    # A deterministic request can occasionally end without content when the
    # serving engine aborts one sequence. Retry only transport/incomplete
    # responses; keep the prompt, seed, and decoding parameters unchanged.
    for attempt in range(1,6):
      try:
        with urllib.request.urlopen(req,timeout=900) as response: raw=json.load(response)
        choice=raw["choices"][0]
        if choice["finish_reason"] == "length" and classify_length_as_miss:
          return {k:v for k,v in task.items() if k!="text"}|{"prediction":"","exact":False,"normalized_exact":False,
            "seconds":time.perf_counter()-started,"attempts":attempt,"finish_reason":"length","generation_failure":"output_budget_exhausted"}
        if choice["finish_reason"]!="stop" or not choice["message"].get("content"):
          raise RuntimeError(f'incomplete response: {choice.get("finish_reason")}')
        pred=json.loads(choice["message"]["content"])["value"].strip()
        return {k:v for k,v in task.items() if k!="text"}|{"prediction":pred,"exact":pred==task["gold"],"normalized_exact":norm(pred)==norm(task["gold"]),"seconds":time.perf_counter()-started,"attempts":attempt,"finish_reason":"stop"}
      except Exception as exc:
        last_error=exc
        if attempt<5: time.sleep(min(2**attempt,16))
    raise RuntimeError(f'reconstruction failed after 5 attempts for {task["id"]}:{task["field"]}: {last_error}')

def atomic(path,obj):
    tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n"); tmp.replace(path)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--url",required=True); p.add_argument("--corpus",type=Path,required=True); p.add_argument("--protocol",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    p.add_argument("--shard-index",type=int,default=0); p.add_argument("--shard-count",type=int,default=1)
    p.add_argument("--resume",action="store_true"); p.add_argument("--classify-length-as-miss",action="store_true")
    a=p.parse_args()
    protocol=json.loads(a.protocol.read_text())
    a.out.mkdir(parents=True,exist_ok=a.resume)
    protocol_path=a.out/"protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol: raise ValueError("resume protocol mismatch")
    atomic(protocol_path,protocol)
    rows=json.loads(a.corpus.read_text()); seed=protocol["seed"]
    rows=sorted(rows,key=lambda r:hashlib.sha256(f'{seed}:{r["id"]}'.encode()).hexdigest())[:protocol["documents"]]
    if not 0 <= a.shard_index < a.shard_count: raise ValueError("invalid shard")
    rows=rows[a.shard_index::a.shard_count]
    tasks=[]
    for row in rows:
      for name,gold in fields(row).items():
        if not gold or gold not in row["text"]: raise ValueError(f'missing field {row["id"]}:{name}')
        tasks.append({"id":row["id"],"field":name,"class":"private" if name in PRIVATE else "public","gold":gold,"text":row["text"],"rule_predicts_known":name not in PRIVATE})
    output_path=a.out/"reconstructions.jsonl"; results=[]
    if a.resume and output_path.exists():
      with output_path.open() as existing:
        results=[json.loads(line) for line in existing if line.strip()]
    result_keys=[(row["id"],row["field"]) for row in results]
    if len(result_keys)!=len(set(result_keys)): raise ValueError("duplicate records in resume file")
    completed={(row["id"],row["field"]) for row in results}
    pending=[task for task in tasks if (task["id"],task["field"]) not in completed]
    output=output_path.open("a" if a.resume else "w",buffering=1)
    with ThreadPoolExecutor(max_workers=16) as pool:
      future={pool.submit(call,a.url,t,seed,a.classify_length_as_miss):t for t in pending}
      for i,f in enumerate(as_completed(future),len(results)+1):
        row=f.result(); results.append(row); output.write(json.dumps(row,ensure_ascii=False)+"\n")
        if i%100==0: print(f'RECONSTRUCTION {i}/{len(tasks)}',flush=True)
    output.close()
    groups={}
    for r in results:
      key=(r["class"],r["field"]); groups.setdefault(key,[]).append(r)
    report={"status":"RECONSTRUCTION_COMPLETE","documents":len(rows),"tasks":len(results),"shard_index":a.shard_index,"shard_count":a.shard_count,"groups":{
      f"{c}:{f}":{"n":len(g),"exact":sum(x["exact"] for x in g)/len(g),"normalized_exact":sum(x["normalized_exact"] for x in g)/len(g),"rule_disagreements":sum(x["rule_predicts_known"]!=x["normalized_exact"] for x in g)} for (c,f),g in groups.items()}}
    atomic(a.out/"summary.json",report); print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__": main()
