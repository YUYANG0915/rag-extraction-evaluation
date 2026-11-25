"""Freeze paired branch states from completed FedReg H2 trajectories."""
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
IN=ROOT/"data"/"development"
prior={r["id"]:r["nll"] for r in map(json.loads,(IN/"prior.jsonl").read_text().splitlines())}
states=[]
for seed_index in range(1,4):
    rows=[json.loads(x) for x in (IN/f"seed{seed_index}.jsonl").read_text().splitlines()]
    low={r["turn"]:r for r in rows if r["arm"]=="low_information"}
    eligible=[]
    for turn,row in sorted(low.items()):
        slots=row["selection"].get("surplus",0); pool=row["selection"].get("pool_ids",[])
        if row.get("core_available") and slots>0 and len(pool)>=slots+3:
            eligible.append(row)
    # Spread ten states across the completed trajectory rather than taking an early prefix.
    chosen=[eligible[round(i*(len(eligible)-1)/9)] for i in range(10)]
    for row in chosen:
        pool=sorted(row["selection"]["pool_ids"],key=lambda x:(prior[x],x))
        slots=row["selection"]["surplus"]
        common=pool[:slots-1]; remaining=[x for x in pool if x not in common]
        idx=sorted({round(i*(len(remaining)-1)/3) for i in range(4)})
        if len(idx)<4: raise RuntimeError("candidate quantiles collapsed")
        candidates=[remaining[i] for i in idx]
        history=[low[t] for t in sorted(low) if t<row["turn"]][-5:]
        states.append({
          "state_id":f"s{seed_index}-t{row['turn']:03d}","source_seed":seed_index,"source_turn":row["turn"],
          "question":row["question"],"core_ids":row["core_ids"],"top20_ids":row["top20_ids"],
          "surplus_slots":slots,"common_filler_ids":common,"candidate_ids":candidates,
          "candidate_prior_nll":{x:prior[x] for x in candidates},
          "history":[{"question":x["question"],"answer":x["answer"]} for x in history]
        })
out=ROOT/"data"/"development"/"states.json"
out.write_text(json.dumps(states,ensure_ascii=False,indent=2)+"\n")
print("BRANCH_STATES_OK",len(states),hashlib.sha256(out.read_bytes()).hexdigest())
