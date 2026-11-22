import json, copy, ast
from pathlib import Path
src=Path('/Users/flexonafft/Downloads/enveda-casmi-stack-aprior-uniq-lb-0-498.ipynb')
out=Path('/Users/flexonafft/MLKaggleTasks/competitions/public-comp/enveda-casmi26')
base=json.loads(src.read_text())

def md(text): return {'cell_type':'markdown','metadata':{},'source':text.splitlines(True)}
def code(text): return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':text.splitlines(True)}

adapt=copy.deepcopy(base)
adapt['metadata']['title']='CASMI26 — APRIOR + UNIQ + E180X adaptation'
# Clear stale run outputs from the supplied notebook; this is a rerunnable Kaggle copy.
for c in adapt['cells']:
    if c['cell_type']=='code':
        c['execution_count']=None; c['outputs']=[]
# Replace first descriptive title while retaining source attribution and all provenance.
s=''.join(adapt['cells'][1]['source'])
s=s.replace('# Enveda CASMI | Stack + E180X + APRIOR + UNIQ | LB 0.501', '# CASMI26 | Adaptation of APRIOR + UNIQ + E180X')
s=s.replace('**Public LB 0.501** (MRR@25)', '**Source notebook reported public LB 0.501** (MRR@25)')
s=s.replace('This version of the notebook is submitted as well, and t', 'This local adaptation preserves the upstream pipeline and experimental layers for Kaggle reruns. It has not been submitted from this copy. The source notebook reported that t')
adapt['cells'][1]['source']=s.splitlines(True)
adapt['cells'].insert(2,md('''## Local adaptation notes\n\nThis copy preserves the supplied notebook's pipeline, APRIOR adduct prior, UNIQ duplicate-leader logic, E180X natural-product prior, comparisons, and attribution. It is intended for Kaggle **Copy & Edit** with the source competition inputs and datasets attached.\n\nAll saved execution outputs were cleared so the notebook will show results from your own run. The source-reported leaderboard scores below are historical observations, not a score guarantee for this copy.\n'''))
(out/'casmi26_aprior_uniq_adaptation.ipynb').write_text(json.dumps(adapt,ensure_ascii=False,indent=1)+'\n')

hyp=copy.deepcopy(adapt)
hyp['metadata']['title']='CASMI26 — Hypothesis: evidence-calibrated rank fusion'
hyp['cells'].insert(2,md('''## Hypothesis notebook · evidence-calibrated candidate ordering\n\n**Hypothesis (unverified):** the source pipeline's fixed APRIOR and synthetic-library rank rules may overreact when evidence is weak. A bounded, evidence-calibrated reranker could preserve the baseline order when evidence is sparse, while acting more strongly when independent evidence agrees.\n\nThis notebook keeps the adapted pipeline as its baseline and adds a controlled experiment design. No leaderboard gain is claimed. The experiment below is deliberately an offline, labelled-panel test; enable the reranker in a submission only after it improves held-out MRR@25 without damaging rank-1 accuracy.\n\n**Primary test:** compare baseline, current fixed-rule output, and evidence-calibrated output on molecule-disjoint folds. Report MRR@25, top-1 accuracy, per-evidence-coverage results, and bootstrap intervals. Tune thresholds only inside training folds.\n'''))
hyp['cells'].insert(3,md('### Experiment cell · bounded score fusion\n\nThis generic, testable utility defines the proposed fusion. It is not wired into the production submission path: the source notebook does not expose aligned per-candidate evidence scores and held-out labels in a safe common schema. Connect those only in the offline evaluation section after verifying molecule and candidate keys.'))
hyp['cells'].insert(4,code('''import numpy as np\n\ndef evidence_calibrated_rerank(candidate_ids, baseline_scores, evidence_scores, evidence_reliability, max_shift=0.15):\n    """Return a stable reranking score; callers must supply fold-safe, aligned evidence."""\n    candidate_ids = np.asarray(candidate_ids)\n    baseline_scores = np.asarray(baseline_scores, dtype=float)\n    evidence_scores = np.asarray(evidence_scores, dtype=float)\n    reliability = np.asarray(evidence_reliability, dtype=float)\n    if not (candidate_ids.ndim == baseline_scores.ndim == evidence_scores.ndim == reliability.ndim == 1):\n        raise ValueError("All inputs must be one-dimensional and candidate-aligned")\n    if not (len(candidate_ids) == len(baseline_scores) == len(evidence_scores) == len(reliability)):\n        raise ValueError("Candidate and score arrays must have equal lengths")\n    if len(set(candidate_ids.tolist())) != len(candidate_ids):\n        raise ValueError("Candidate IDs must be unique within a molecule")\n    if not all(np.isfinite(x).all() for x in (baseline_scores, evidence_scores, reliability)):\n        raise ValueError("Scores must be finite")\n    if np.any((reliability < 0) | (reliability > 1)) or not 0 <= max_shift <= 1:\n        raise ValueError("Reliability and max_shift must be in [0, 1]")\n    # Center evidence per molecule and bound its influence to protect the baseline ordering.\n    centered = evidence_scores - evidence_scores.mean()\n    scale = np.max(np.abs(centered))\n    normalized = centered / scale if scale > 0 else centered\n    return baseline_scores + max_shift * reliability * normalized\n\n# Small invariant check: no evidence means exactly the baseline scores.\n_ids = np.array(["a", "b", "c"])\n_base = np.array([3.0, 2.0, 1.0])\nassert np.array_equal(evidence_calibrated_rerank(_ids, _base, [0, 1, 2], [0, 0, 0]), _base)\n'''))
hyp['cells'].insert(5,md('''### Required validation before submission use\n\n1. Build a molecule-disjoint offline panel with candidate IDs, baseline ranks, per-candidate evidence, and known answers.\n2. Produce out-of-fold evidence scores; do not fit calibrators on the evaluated molecules.\n3. Compare baseline and reranked MRR@25 and top-1 accuracy overall and by evidence coverage.\n4. Bootstrap by molecule and inspect harmful rank-1 changes.\n5. Keep the baseline if confidence intervals include no gain or rank-1 accuracy drops materially.\n\nThe proposed score utility is only a hypothesis mechanism. It is intentionally not applied to the Kaggle submission in this notebook.\n'''))
(out/'casmi26_evidence_calibrated_hypothesis.ipynb').write_text(json.dumps(hyp,ensure_ascii=False,indent=1)+'\n')
# Validate notebook JSON and parse every Python code cell without executing competition pipeline.
for path in (out/'casmi26_aprior_uniq_adaptation.ipynb',out/'casmi26_evidence_calibrated_hypothesis.ipynb'):
 n=json.loads(path.read_text()); assert n['nbformat']==4
 for i,c in enumerate(n['cells']):
  if c['cell_type']=='code':
   try: ast.parse(''.join(c['source']))
   except SyntaxError as e: raise RuntimeError(f'{path.name} cell {i}: {e}')
 print(path.name, len(n['cells']), 'cells; code syntax OK')
