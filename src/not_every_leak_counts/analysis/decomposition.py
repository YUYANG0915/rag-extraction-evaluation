"""Post-hoc document-source decomposition of completed adaptive trajectories."""
import collections
import hashlib
import json
import numpy as np


def decompose(trajectory, corpus, estimate):
    owner = {}
    for doc in corpus:
        for marker in doc['markers']:
            assert marker not in owner, f'Ambiguous marker: {marker}'
            owner[marker] = doc['id']
    groups = collections.defaultdict(list)
    for row in trajectory:
        groups[row['case_id'], row['seed'], row['arm']].append(row)
    observations = []
    for (case, seed, arm), rows in sorted(groups.items()):
        rows.sort(key=lambda row: row['turn'])
        assert [r['turn'] for r in rows] == list(range(20))
        final = rows[-1]
        values = set(final['coverage_values'])
        assert values <= owner.keys(), 'Unmapped recovered marker'
        assert len(values) == final['coverage']
        valid = [r for r in rows if r['valid_request']]
        assert len(valid) >= 2, 'Context transitions require two valid requests'
        contexts = [set(r['context_ids']) - {case} for r in valid]
        within = sum(owner[v] == case for v in values)
        outside = sum(owner[v] != case for v in values)
        assert within + outside == final['coverage']
        # Count response mentions separately from cumulative unique recovery.
        per_response = [sum(m.lower() in r['answer'].lower()
                            for m, doc in owner.items() if doc != case)
                        for r in rows]
        observations.append(dict(case_id=case, seed=seed, arm=arm,
            start_document=within, other_documents=outside, total=len(values),
            distinct_surplus_documents=len(set().union(*contexts)),
            context_change_rate=sum(a != b for a, b in zip(contexts, contexts[1:])) / (len(contexts)-1),
            cross_document_markers_per_response=float(np.mean(per_response)),
            valid_requests=len(valid), context_transitions=len(contexts)-1))
    metrics = ['start_document', 'other_documents', 'total',
               'distinct_surplus_documents', 'context_change_rate',
               'cross_document_markers_per_response']
    grouped = collections.defaultdict(list)
    for r in observations:
        grouped[r['arm'], r['case_id']].append(r)
    cases = sorted({r['case_id'] for r in observations})
    arms = sorted({r['arm'] for r in observations})
    assert len(cases) == 12
    assert all(len(grouped[a, c]) == 3 for a in arms for c in cases)
    means = {a: {m: np.array([np.mean([r[m] for r in grouped[a, c]])
                              for c in cases]) for m in metrics} for a in arms}
    report = {
        'analysis_status': 'Post-hoc descriptive analysis; no multiplicity correction.',
        'partition': 'Starting document versus other documents; not a proven authorization closure.',
        'bootstrap': {'unit': 'starting document (case_id)', 'clusters': 12,
                      'seeds_per_cluster': 3, 'replicates': 10000, 'seed': 20261013,
                      'interval': 'two-sided percentile 95%',
                      'pairing': 'Average seeds within each start, then resample paired starts across arms.'},
        'denominators': {'context_change_rate': 'Successive valid requests within each trajectory; then equal-weight seeds and starts.',
                         'cross_document_markers_per_response': 'All 20 responses, including invalid requests; repeated mentions count per response.'},
        'arms': {}, 'paired_contrasts': {}, 'trajectory_observations': observations}
    for arm in arms:
        report['arms'][arm] = {m: estimate(means[arm][m]) for m in metrics}
        for m in metrics[:3]:
            report['arms'][arm]['gap_' + m] = estimate(means[arm][m] - means['ideal'][m])
        assert np.allclose(means[arm]['total'] - means['ideal']['total'],
            means[arm]['start_document'] - means['ideal']['start_document'] +
            means[arm]['other_documents'] - means['ideal']['other_documents'])
    for a, b in [('topk', 'core'), ('mev', 'topk'), ('exposure_gate', 'core')]:
        report['paired_contrasts'][a + '_minus_' + b] = {
            m: estimate(means[a][m] - means[b][m]) for m in metrics[:3]}
    return report


def write_results(root, trajectory, estimate, table):
    corpus_path = root / 'data/corpus/fedreg.json'
    report = decompose(trajectory, json.loads(corpus_path.read_text()), estimate)
    report['input_sha256'] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [corpus_path, root / 'data/adaptive_service/records.jsonl']}
    (root / 'results/adaptive_decomposition.json').write_text(json.dumps(report, indent=2)+'\n')
    def cell(v):
        return f"{v['mean']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]"
    metrics = ['start_document','other_documents','gap_start_document','gap_other_documents','gap_total']
    table('adaptive_decomposition', ['Arm','Start facts (95% CI)','Other facts (95% CI)',
        'Start gap vs ideal','Other gap vs ideal','Total gap vs ideal'],
        [[a]+[cell(d[m]) for m in metrics] for a,d in report['arms'].items()])
    table('adaptive_decomposition_contrasts', ['Paired contrast','Start difference (95% CI)',
        'Other difference (95% CI)','Total difference (95% CI)'],
        [[a]+[cell(d[m]) for m in ['start_document','other_documents','total']]
         for a,d in report['paired_contrasts'].items()])
    table('adaptive_context_rotation', ['Arm','Distinct surplus documents (95% CI)',
        'Context change fraction (95% CI)','Other markers / response (95% CI)'],
        [[a]+[cell(d[m]) for m in ['distinct_surplus_documents','context_change_rate',
                                  'cross_document_markers_per_response']]
         for a,d in report['arms'].items()])
    return report
