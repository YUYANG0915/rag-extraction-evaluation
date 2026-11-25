"""Rebuild tables from curated observations. No model calls or GPU needed."""
from pathlib import Path
import collections
import hashlib
import json
import numpy as np
from .decomposition import write_results

ROOT = Path(__file__).resolve().parents[3]
ARMS = ['topk', 'core', 'random', 'prior', 'mev', 'exposure_gate', 'ideal']

def read(path):
    return [json.loads(x) for x in (ROOT / path).read_text().splitlines() if x.strip()]

def estimate(values):
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20261013)
    means = x[rng.integers(len(x), size=(10000, len(x)))].mean(axis=1)
    return {'mean': float(x.mean()), 'ci95': np.quantile(means, [.025, .975]).tolist(), 'n_units': len(x)}

def clusters(rows, metric):
    grouped = collections.defaultdict(list)
    for r in rows: grouped[r['case_id']].append(metric(r))
    return {k: float(np.mean(v)) for k, v in grouped.items()}

def table(name, headers, rows):
    text = '| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join(['---'] * len(headers)) + ' |\n'
    for row in rows: text += '| ' + ' | '.join(map(str, row)) + ' |\n'
    (ROOT / 'results' / (name + '.md')).write_text(text)

def main():
    for item in json.loads((ROOT/'provenance/import_manifest.json').read_text()):
        p = ROOT/item['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == item['sha256'], p
    pairs = read('data/service_pairs/records.jsonl')
    trajectory = read('data/adaptive_service/records.jsonl')
    cross = read('data/cross_family/records.jsonl')
    probes = read('data/service_probes/records.jsonl')
    assert len(pairs) == 1680 and len(trajectory) == 5040 and len(cross) == 1680 and len(probes) == 216
    assert len({(r['case_id'],r['world'],r['arm']) for r in pairs}) == 1680
    assert len({(r['case_id'],r['seed'],r['arm'],r['turn']) for r in trajectory}) == 5040
    assert len({(r['case_id'],r['world'],r['arm']) for r in cross}) == 1680
    report = {'analysis_scope': 'Controlled service protocol; descriptive cluster bootstrap, not confirmatory noninferiority.',
              'bootstrap_seed': 20261013, 'bootstrap_reps': 10000, 'pairs': {}, 'adaptive': {}, 'cross_family': {}, 'budgeted_knowledge': {}}
    final = {}
    for r in trajectory:
        k=(r['case_id'],r['seed'],r['arm'])
        if k not in final or r['turn'] > final[k]['turn']: final[k]=r
    assert len(final)==252 and all(r['turn']==19 for r in final.values())
    pair_by_arm = {}
    for arm in ARMS:
        subset=[r for r in pairs if r['arm']==arm]
        metrics={'quality': lambda r:r['quality_exact'], 'extra_markers':lambda r:len(r['extra_hits']), 'discrimination':lambda r:float(r['distinguished'])}
        pair_by_arm[arm]={name:clusters(subset,fn) for name,fn in metrics.items()}
        report['pairs'][arm]={name:estimate(list(v.values())) for name,v in pair_by_arm[arm].items()}
        ending=[r for k,r in final.items() if k[2]==arm]
        gap=[dict(r, gap=r['coverage']-final[(r['case_id'],r['seed'],'ideal')]['coverage']) for r in ending]
        report['adaptive'][arm]={
            'total_coverage':estimate(list(clusters(ending,lambda r:r['coverage']).values())),
            'real_minus_ideal':estimate(list(clusters(gap,lambda r:r['gap']).values())),
            'extra_coverage':estimate(list(clusters(ending,lambda r:len(set(r['coverage_values'])-set(r['authorized_values']))).values())),
            'valid_request_fraction':estimate(list(clusters([r for r in trajectory if r['arm']==arm],lambda r:float(r['valid_request'])).values())),
        }
        valid=[r for r in trajectory if r['arm']==arm and r['valid_request'] and r.get('quality_exact') is not None]
        report['adaptive'][arm]['quality_on_valid_requests']=estimate(list(clusters(valid,lambda r:r['quality_exact']).values()))
        report['cross_family'][arm]=estimate(list(clusters([r for r in cross if r['arm']==arm],lambda r:float(r['distinguished'])).values()))
    report['descriptive_paired_contrasts']={}
    for other in ['random','prior','topk']:
        report['descriptive_paired_contrasts']['mev_minus_'+other]={
            metric:estimate([a[k]-pair_by_arm[other][metric][k] for k in sorted(a)])
            for metric,a in pair_by_arm['mev'].items()}
    report['service_probes']={'attempts':len(probes),'exact_hits':sum(r['exact_hit'] for r in probes)}
    for model in ['gemma','llama']:
        facts=read(f'data/budgeted_knowledge/{model}/facts.jsonl')
        attempts=read(f'data/budgeted_knowledge/{model}/attempts.jsonl')
        assert len(facts)==1800 and len(attempts)==9000
        assert len({(r['id'],r['field']) for r in facts})==1800
        report['budgeted_knowledge'][model]={}
        for category in ['all','public','private']:
            selected=[r for r in facts if category=='all' or r['class']==category]
            assert selected
            report['budgeted_knowledge'][model][category]={'facts':len(selected), 'budgets':{}}
            for b in [1,3,5]:
                grouped=collections.defaultdict(list)
                for r in selected: grouped[r['id']].append(float(r[f'exact_b{b}']))
                report['budgeted_knowledge'][model][category]['budgets'][str(b)]=estimate([np.mean(v) for v in grouped.values()])
    (ROOT/'results').mkdir(exist_ok=True)
    report['adaptive_decomposition'] = write_results(ROOT, trajectory, estimate, table)
    (ROOT/'results/summary.json').write_text(json.dumps(report,indent=2)+'\n')
    table('service_pairs',['Arm','Quality','Extra markers / answer','Discrimination'],[[a]+[f"{report['pairs'][a][m]['mean']:.4f}" for m in ['quality','extra_markers','discrimination']] for a in ARMS])
    table('adaptive_service',['Arm','Coverage @20','Real − ideal','Gap 95% CI','Quality on valid requests','Valid request fraction'],[
        [a,f"{report['adaptive'][a]['total_coverage']['mean']:.4f}",f"{report['adaptive'][a]['real_minus_ideal']['mean']:.4f}",str([round(x,4) for x in report['adaptive'][a]['real_minus_ideal']['ci95']]),f"{report['adaptive'][a]['quality_on_valid_requests']['mean']:.4f}",f"{report['adaptive'][a]['valid_request_fraction']['mean']:.4f}"] for a in ARMS])
    table('cross_family',['Arm','Gemma accuracy','Case-cluster 95% CI'],[[a,f"{report['cross_family'][a]['mean']:.4f}",str([round(x,4) for x in report['cross_family'][a]['ci95']])] for a in ARMS])
    table('budgeted_knowledge',['Model','Fields','N facts','Budget 1','Budget 3','Budget 5'],[[m,c,v['facts']]+[f"{v['budgets'][str(b)]['mean']:.4f}" for b in [1,3,5]] for m,d in report['budgeted_knowledge'].items() for c,v in d.items()])
    print('Verified all imported hashes and record counts; rebuilt main tables, source decomposition, paired intervals, context rotation, and summary.json.')

if __name__ == '__main__': main()
