"""Fit a small development-only relative MEV scorer. No evaluation claims on training data."""
import collections,hashlib,json,re
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
def features(row,question,known,nll):
 words=lambda s:set(re.findall(r'[a-z]{3,}',s.lower()))
 a,b=words(question),words(row['text'].split('Fictional internal')[0])
 return [nll,len(row['text'].split())/500,len(a&b)/max(1,len(a|b)),sum(m not in known for m in row['markers'])/6]
def main():
 path=ROOT/'data/development/branches.jsonl'
 records=[json.loads(x) for x in path.open()];assert len(records)==120
 states={s['state_id']:s for s in json.loads((ROOT/'data/development/states.json').read_text())}
 corpus={r['id']:r for r in json.loads((ROOT/'data/fedreg_private_2026/corpus.json').read_text())}
 grouped=collections.defaultdict(lambda:collections.defaultdict(list))
 for r in records:grouped[r['state_id']][r['candidate_id']].append(r)
 X=[];Y=[]
 for sid,candidates in grouped.items():
  assert len(candidates)==4 and all(len(g)==3 for g in candidates.values())
  state=states[sid];history=json.dumps(state['history']);known={m for row in corpus.values() for m in row['markers'] if m in history}
  x=[];y=[]
  for cid,g in candidates.items():
   x.append(features(corpus[cid],state['question'],known,g[0]['candidate_prior_nll']));y.append(np.mean([len(r['total_hits']) for r in g]))
  x=np.asarray(x);y=np.asarray(y);X.extend(x-x.mean(axis=0));Y.extend(y-y.mean())
 X=np.asarray(X);Y=np.asarray(Y);scale=np.maximum(X.std(axis=0),1e-6);Xs=X/scale
 # Fixed ridge regularization, no test-driven hyperparameter selection.
 beta=np.linalg.solve(Xs.T@Xs+10*np.eye(Xs.shape[1]),Xs.T@Y)/scale
 model={'version':'development-relative-mev-ridge-v1','features':['prior_nll','word_count_over_500','public_question_jaccard','unknown_marker_fraction'],'coefficients':beta.tolist(),'ridge_lambda':10,'training_states':list(grouped),'training_candidates':len(Y),'training_records':len(records),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'label':'state-centered mean total unique markers over three continuation seeds','limitations':'10 development states, no claimed train-set performance; controlled-schema novelty, no quality theorem; full development candidate pools excluded from evaluation'}
 (ROOT/'data/protocol/mev_model.json').write_text(json.dumps(model,indent=2)+'\n')
 print('FIT_OK',len(Y),'candidate labels',len(grouped),'states')
if __name__=='__main__':main()
