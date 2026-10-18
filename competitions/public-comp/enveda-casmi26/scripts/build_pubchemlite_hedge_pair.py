'''Last-stage-only PubChemLite experiments with atomic checked submission handoff.'''
import ast
import copy
import json
from build_pubchemlite_pair import build as source_build
from build_v22_adaptation import SOURCE as V22_SOURCE
from build_flow_experiments import ROOT,prose,validate
from build_metric_safe_adaptation import code_cell

source_v22=json.loads(V22_SOURCE.read_text())
tree=ast.parse(''.join(source_v22['cells'][36]['source']))
METRIC_ROWS=ast.unparse(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)
    and n.name in ('e2r2_make_key','e2r2_row')],type_ignores=[]))+'\n'

FINALIZER='''
import os

class CheckedSubmission:
    def __init__(self,sample,control,active,identity):
        self.sample,self.control,self.active,self.identity=sample,control,active,identity

    def validate(self,frame):
        assert frame.columns.tolist()==['molecule_id','smiles'], 'Wrong submission columns'
        assert frame.molecule_id.is_unique and self.sample.molecule_id.is_unique, 'Duplicate molecule IDs'
        assert frame.molecule_id.tolist()==self.sample.molecule_id.tolist(), 'Sample ID order/membership differs'
        assert frame.molecule_id.tolist()==self.control.molecule_id.tolist(), 'Control IDs differ'
        assert frame.smiles.notna().all()
        for row,old in zip(frame.itertuples(index=False),self.control.itertuples(index=False)):
            tokens=row.smiles.split(';')
            assert 1<=len(tokens)<=25 and all(t.strip() for t in tokens), ('Empty/oversized candidate list',row.molecule_id)
            if row.molecule_id in self.active:
                assert all(self.identity(s) is not None for s in tokens), ('Invalid candidate',row.molecule_id)
            else:
                assert row.smiles==old.smiles, ('Inactive placeholder changed',row.molecule_id)
            assert tokens[0]==old.smiles.split(';')[0], ('Final hedge changed top1',row.molecule_id)

    def save(self,frame,path,exact_bytes=None):
        path=Path(path)
        temporary=path.with_name(path.stem+'_checked.tmp.csv')
        self.validate(frame)
        try:
            if exact_bytes is None:
                frame.to_csv(temporary,index=False)
            else:
                temporary.write_bytes(exact_bytes)
            readback=pd.read_csv(temporary,dtype=str,keep_default_na=False)
            self.validate(readback)
            assert readback.equals(frame), 'CSV readback differs from intended output'
            assert temporary.stat().st_size>0
            os.replace(temporary,path)
            assert path.is_file() and path.stat().st_size>0
        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise

class PubChemLiteHedgeExperiment:
    def __init__(self,experiment,mode):
        assert mode in ('metric_exact_rank2','no_final_rank2')
        self.scope,self.mode=experiment.scope,mode

    def run(self):
        scope=self.scope
        control_bytes=Path('submission.csv').read_bytes()
        pre_bytes=Path('submission_e006.csv').read_bytes()
        Path('submission_pubchemlite_rank2_control.csv').write_bytes(control_bytes)
        Path('submission_pubchemlite_e006_control.csv').write_bytes(pre_bytes)
        control=pd.read_csv('submission.csv',dtype=str,keep_default_na=False)
        before=pd.read_csv('submission_e006.csv',dtype=str,keep_default_na=False)
        sample=pd.read_csv(Path(scope['COMP'])/'sample_submission.csv',dtype=str,keep_default_na=False)
        active=set(scope['te'].molecule_id.astype(str))
        checker=CheckedSubmission(sample,control,active,scope['metric_identity'])
        checker.validate(before)
        rows,audit=[],[]
        key=e2r2_make_key(scope['metric_identity'])
        for row,old in zip(before.itertuples(index=False),control.itertuples(index=False)):
            tokens=row.smiles.split(';')
            revised,tag,moved=list(tokens),'hedge_disabled',None
            if self.mode=='metric_exact_rank2' and row.molecule_id in active:
                engine=scope['ENG'].get(row.molecule_id)
                if engine is None and row.molecule_id.strip().lstrip('-').isdigit():
                    engine=scope['ENG'].get(int(row.molecule_id))
                candidates=engine.get('smiles') or [] if isinstance(engine,dict) else []
                revised,tag,moved=e2r2_row(tokens,candidates,key)
                # Preserve the source helper's guarantee: do not introduce metric duplicates.
                def duplicate_count(values):
                    identities=[scope['metric_identity'](s) for s in values]
                    return len(identities)-len(set(identities))
                assert duplicate_count(revised)<=duplicate_count(tokens)
            elif row.molecule_id not in active:
                tag='inactive'
            value=';'.join(revised)
            rows.append((row.molecule_id,value))
            audit.append(dict(molecule_id=row.molecule_id,tag=tag,moved_from=moved,
                changed_vs_native_rank2=value!=old.smiles,changed_vs_e006=value!=row.smiles))
        final=pd.DataFrame(rows,columns=['molecule_id','smiles'])
        # Check/write/reload before replacing the known-good source submission.csv.
        checker.save(final,'submission.csv',exact_bytes=pre_bytes if self.mode=='no_final_rank2' else None)
        pd.DataFrame(audit).to_csv('pubchemlite_hedge_comparison.csv',index=False)
        report=dict(mode=self.mode,measured_score=None,
            changed_rows_vs_native_rank2=sum(a['changed_vs_native_rank2'] for a in audit),
            changed_rows_vs_e006=sum(a['changed_vs_e006'] for a in audit),
            changed_top1=0,submission_checked_and_read_back=True,
            control_sha256=hashlib.sha256(control_bytes).hexdigest(),
            submission_sha256=hashlib.sha256(Path('submission.csv').read_bytes()).hexdigest(),
            note='Native PubChemLite source control from the same run. No new neural inference.')
        Path('pubchemlite_hedge_audit.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
        print('READY:',str(Path('submission.csv').resolve()),Path('submission.csv').stat().st_size,'bytes',flush=True)
        if not report['changed_rows_vs_native_rank2']:
            print('NO-OP versus the source control; do not spend a duplicate submission.')
        return report
'''


def build():
    base=source_build()[0]
    outputs=[]
    for mode,title,method in (
        ('metric_exact_rank2','PubChemLite — Metric-Exact Final Hedge',
         'Keep the complete public PubChemLite recipe and old prvsiyan BIO. Replace only the final '
         'source-key hedge with the V22 pure metric-exact row helper: stereo/tautomer twins of '
         'rank1 cannot take rank2, and duplicate identities of the inserted candidate are removed. '
         'No V22 BIO, fragment-gap or runner-up policy is added.'),
        ('no_final_rank2','PubChemLite — No Final Hedge',
         'Keep the same PubChemLite recipe but use its complete E006 output BEFORE the last rank2 '
         'insertion. Engine2 remains in retrieval, simulator inputs and metric fusion. Restore '
         'those exact E006 CSV bytes after validation; this is not disabling engine2 itself.')):
        nb=copy.deepcopy(base)
        nb['cells'][0]=code_cell(prose('CASMI26: Closing the Evidence Gap\n'+title+'\n'+method+'\n'
            'Independent last-stage-only treatment. The underlying adaptation has been submitted '
            'by the user; its new score is not yet known. No treatment score is claimed. '
            'Run the original recipe first; retain both final and pre-hedge control CSVs. '
            'Our additions are OOP experiment control, the isolated hedge policy and atomic checked '
            'CSV finalization, not the original models or candidate datasets.'))
        nb['cells'][1]['source']+=prose('Additional hedge attribution\n'
            '@antriksharya — original rank2 insertion; Omer Faruk Merey — metric-exact version '
            'attributed in the supplied V22 file. Source link retained: '
            'https://www.kaggle.com/code/antriksharya/casmi26-lb-0-443-v7c-engine-2-rank-2-insert . '
            'The V22 e2r2_make_key/e2r2_row helpers are reused verbatim as parsed Python functions; '
            'their metric_identity dependency is the identical upstream official-metric helper. '
            'All earlier credits and licences remain.').splitlines(keepends=True)
        nb['cells'].append(code_cell(prose('Finalize one experiment, validate and read the actual file\n'
            +method+'\nBoth notebooks use prvsiyan/chebi-lipidmaps-casmi26 ONLY as BIO, plus '
            'PubChemLite and the inputs of the submitted adaptation. Disconnect sergiomalv BIO '
            'for this pair. Keep both RDKit resources for Python3.13. No new datasets. '
            'The finalizer checks sample ID order, nonempty lists of 1..25 valid candidates, '
            'unchanged top1/placeholders and CSV readback. On validation/write failure it raises '
            'without replacing the original source submission. GPU inference and rendered outputs '
            'still require Kaggle; synthetic file integration tests are not a full-model run.')
            +METRIC_ROWS+FINALIZER+f'hedge_experiment = PubChemLiteHedgeExperiment(experiment,{mode!r})\n'
            +'hedge_result = hedge_experiment.run()\n'))
        nb['metadata']['casmi_experiment'].update(mode=mode,measured_score=None,
            validation='Synthetic native pipeline/hedge/atomic CSV handoff; full Kaggle model run pending')
        validate(nb);outputs.append(nb)
    return outputs


if __name__=='__main__':
    for name,nb in zip(('casmi26_pubchemlite_metric_exact_hedge.ipynb','casmi26_pubchemlite_no_final_hedge.ipynb'),build()):
        path=ROOT/'research/evidence-routing'/name
        path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
        print(path)
