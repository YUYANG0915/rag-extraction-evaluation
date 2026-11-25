import argparse,hashlib,json,random,time,urllib.request
from pathlib import Path
from . import prompts

QSCHEMA={"type":"object","properties":{"question":{"type":"string","minLength":8}},"required":["question"],"additionalProperties":False}
JSCHEMA={"type":"object","properties":{"equivalent":{"type":"boolean"},"reason":{"type":"string"}},"required":["equivalent","reason"],"additionalProperties":False}
def atomic(p,x):
 t=p.with_suffix(p.suffix+".tmp");t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n");t.replace(p)
def public(xs):return [{"id":x["id"],"text":x["text"]} for x in xs]
class Client:
 def __init__(self,g,a,out):self.u={"generator":g,"attacker":a};self.log=(out/"api_calls.jsonl").open("w",buffering=1)
 def call(self,role,purpose,system,body,seed,temp,max_tokens,schema=None):
  p={"model":role,"messages":[{"role":"system","content":system},{"role":"user","content":json.dumps(body,ensure_ascii=False)}],"temperature":temp,"seed":seed,"max_tokens":max_tokens}
  if role=="generator":p["reasoning_effort"]="none"
  if schema:p["response_format"]={"type":"json_schema","json_schema":{"name":purpose,"schema":schema}}
  last_error=None
  for attempt in range(1,6):
   start=time.perf_counter();entry={"role":role,"purpose":purpose,"request":p,"attempt":attempt}
   try:
    req=urllib.request.Request(self.u[role]+"/chat/completions",data=json.dumps(p).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=900) as response:r=json.load(response)
    c=r["choices"][0];ans=c["message"].get("content","").strip()
    if not ans or c["finish_reason"] not in {"stop","length"}:raise RuntimeError(f'incomplete response: {c.get("finish_reason")}')
    parsed=json.loads(ans) if schema else ans
    entry.update(status="ok",response=r);return parsed
   except Exception as e:
    last_error=e;entry.update(status="error",error=str(e))
   finally:
    entry["seconds"]=time.perf_counter()-start;self.log.write(json.dumps(entry,ensure_ascii=False)+"\n")
   if attempt<5:time.sleep(min(2**attempt,16))
  raise RuntimeError(f"{role}:{purpose} failed after 5 attempts: {last_error}")
def hits(answer,rows):return sorted({m for r in rows for m in r["markers"] if m.lower() in answer.lower()})
def main():
 import numpy as np,torch
 from sentence_transformers import SentenceTransformer,CrossEncoder
 p=argparse.ArgumentParser();p.add_argument("--generator-url",required=True);p.add_argument("--attacker-url",required=True);p.add_argument("--states",type=Path,required=True);p.add_argument("--protocol",type=Path,required=True);p.add_argument("--models",type=Path,required=True);p.add_argument("--corpus",type=Path,required=True);p.add_argument("--embedding-cache",type=Path);p.add_argument("--out",type=Path,required=True);p.add_argument("--shard-index",type=int,required=True);p.add_argument("--shard-count",type=int,required=True);a=p.parse_args()
 a.out.mkdir(parents=True,exist_ok=False);protocol=json.loads(a.protocol.read_text());states=json.loads(a.states.read_text())[a.shard_index::a.shard_count];atomic(a.out/"protocol.json",protocol);atomic(a.out/"states.json",states)
 rows=json.loads(a.corpus.read_text());byid={r["id"]:r for r in rows};models=json.loads(a.models.read_text());embed=SentenceTransformer(models["embedder"]["path"],device="cpu");rerank=CrossEncoder(models["reranker"]["path"],device="cpu");torch.set_num_threads(8)
 corpus_sha=hashlib.sha256(a.corpus.read_bytes()).hexdigest();vec=None
 if a.embedding_cache and a.embedding_cache.exists():
  meta_path=a.embedding_cache.with_suffix(".json")
  if not meta_path.exists():raise RuntimeError(f"embedding cache metadata missing: {meta_path}")
  meta=json.loads(meta_path.read_text())
  expected={"corpus_sha256":corpus_sha,"embedder_path":models["embedder"]["path"],"rows":len(rows),"normalized":True}
  if any(meta.get(k)!=v for k,v in expected.items()):raise RuntimeError(f"embedding cache metadata mismatch: {meta}")
  vec=np.load(a.embedding_cache)
  if vec.shape[0]!=len(rows):raise RuntimeError(f"embedding cache row mismatch: {vec.shape}")
  print(f"EMBEDDING_CACHE_LOADED rows={len(rows)} path={a.embedding_cache}",flush=True)
 else:
  vec=embed.encode([r["text"] for r in rows],normalize_embeddings=True,batch_size=32,show_progress_bar=True)
 client=Client(a.generator_url,a.attacker_url,a.out);out=(a.out/"branches.jsonl").open("w",buffering=1);done=0
 for state in states:
  rank={x:i for i,x in enumerate(state["top20_ids"])};core=[byid[x] for x in state["core_ids"]];common=[byid[x] for x in state["common_filler_ids"]]
  reference=client.call("generator","core_reference",prompts.ANSWER,{"question":state["question"],"evidence":public(core)},protocol["seed"],0,1024)
  immediate={}
  initials={}
  for cid in state["candidate_ids"]:
   context=sorted(core+common+[byid[cid]],key=lambda r:rank[r["id"]]);answer=client.call("generator","branch_answer",prompts.ANSWER,{"question":state["question"],"evidence":public(context)},protocol["seed"],0,1024)
   quality=client.call("attacker","quality",prompts.QUALITY,{"question":state["question"],"reference_A":reference,"candidate_B":answer},protocol["seed"],0,512,JSCHEMA)
   initials[cid]=(answer,[r["id"] for r in context],quality);immediate[cid]=hits(answer,rows)
  immediate_union=sorted({x for xs in immediate.values() for x in xs})
  for cid in state["candidate_ids"]:
   for rollseed in protocol["rollout_seeds"]:
    answer,context_ids,quality=initials[cid];question=state["question"];future=[];trace=[];history=state["history"][-3:]+[{"question":question,"answer":answer}]
    for step in range(protocol["continuation_turns"]):
     q=client.call("attacker","next_question",prompts.NEXT,{"history":history[-4:]},rollseed+step,.7,512,QSCHEMA)["question"].strip()
     qv=embed.encode([q],prompt_name="query",normalize_embeddings=True)[0];scores=vec@qv;idx=np.argsort(-scores,kind="stable")[:20];initial=[rows[int(i)] for i in idx];rs=np.asarray(rerank.predict([(q,r["text"]) for r in initial],activation_fn=torch.nn.Sigmoid())).reshape(-1);cand=sorted([dict(r,rerank_score=float(s)) for r,s in zip(initial,rs)],key=lambda r:(-r["rerank_score"],r["id"]));ctx=cand[:5]
     ans=client.call("generator","continuation_answer",prompts.ANSWER,{"question":q,"evidence":public(ctx)},rollseed+step,0,1024);hh=hits(ans,rows);future.extend(hh);trace.append({"step":step,"question":q,"answer":ans,"context_ids":[r["id"] for r in ctx],"marker_hits":hh});history.append({"question":q,"answer":ans});question,answer=q,ans
    record={"state_id":state["state_id"],"candidate_id":cid,"candidate_prior_nll":state["candidate_prior_nll"][cid],"rollout_seed":rollseed,"initial_question":state["question"],"initial_answer":initials[cid][0],"context_ids":context_ids,"quality":quality,"immediate_hits":immediate[cid],"state_immediate_union":immediate_union,"future_hits":sorted(set(future)),"future_excluding_union":sorted(set(future)-set(immediate_union)),"total_hits":sorted(set(immediate[cid])|set(future)),"trace":trace}
    out.write(json.dumps(record,ensure_ascii=False)+"\n");done+=1;print(f"BRANCH {done}/{len(states)*len(state['candidate_ids'])*len(protocol['rollout_seeds'])}",flush=True)
 out.close();client.log.close();atomic(a.out/"summary.json",{"status":"PAIRED_BRANCH_SHARD_COMPLETE","shard_index":a.shard_index,"shard_count":a.shard_count,"states":len(states),"records":done})
if __name__=="__main__":main()
