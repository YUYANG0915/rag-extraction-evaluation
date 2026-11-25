"""Check packaged hashes and complete experimental grids."""
from pathlib import Path
import hashlib,json,itertools
R=Path(__file__).resolve().parents[3]
def rows(p): return [json.loads(x) for x in (R/p).read_text().splitlines() if x.strip()]
for line in (R/'MANIFEST.sha256').read_text().splitlines():
 digest,name=line.split('  ',1)
 assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,name
p=rows('data/service_pairs/records.jsonl');t=rows('data/adaptive_service/records.jsonl');c=rows('data/cross_family/records.jsonl')
keys=lambda data,fields:{tuple(x[f] for f in fields) for x in data}
fields=('case_id','world','arm'); pk=keys(p,fields)
assert len(pk)==len(p)==1680 and pk==keys(c,fields)
assert pk==set(itertools.product(*[sorted({x[f] for x in p}) for f in fields]))
fields=('case_id','seed','arm','turn');tk=keys(t,fields)
assert len(tk)==len(t)==5040 and tk==set(itertools.product(*[sorted({x[f] for x in t}) for f in fields]))
for model in ['gemma','llama']:
 facts=rows(f'data/budgeted_knowledge/{model}/facts.jsonl')
 assert len(facts)==1800
 for x in facts: assert x['exact_b1']<=x['exact_b3']<=x['exact_b5']
for arm in {x['arm'] for x in p}:
 chosen=[x for x in p if x['arm']==arm]
 assert all(0<=x['quality_exact']<=1 for x in chosen)
by={ (x['case_id'],x['world']):x['answer'] for x in p if x['arm']=='ideal'}
worlds=sorted({x['world'] for x in p})
assert all(by[(cid,worlds[0])]==by[(cid,worlds[1])] for cid in {x['case_id'] for x in p})
print('PASS: all payload hashes; complete paired/cross-family and adaptive grids; budget monotonicity; ideal paired identity.')

# Source decomposition must conserve coverage and share the main estimands.
d=json.loads((R/'results/adaptive_decomposition.json').read_text())
s=json.loads((R/'results/summary.json').read_text())
assert d==s['adaptive_decomposition']
assert len(d['trajectory_observations'])==252
for r in d['trajectory_observations']:
 assert r['start_document']+r['other_documents']==r['total']
for arm,v in d['arms'].items():
 assert abs(v['gap_start_document']['mean']+v['gap_other_documents']['mean']-v['gap_total']['mean'])<1e-10
 assert abs(v['total']['mean']-s['adaptive'][arm]['total_coverage']['mean'])<1e-10
 assert abs(v['gap_total']['mean']-s['adaptive'][arm]['real_minus_ideal']['mean'])<1e-10
 assert all(x['n_units']==12 for x in v.values())
for name,digest in d['input_sha256'].items():
 assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,name
print('PASS: decomposition conserves coverage; paired estimands match main results; 12 clusters and input hashes verified.')
