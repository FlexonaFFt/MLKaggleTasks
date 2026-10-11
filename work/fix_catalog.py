from pathlib import Path
import json, ast, tempfile
p=Path('competitions/monthly-comp/predicting-airline-satisfaction/s6e10_blender_research.ipynb')
n=json.loads(p.read_text())
c=n['cells'][10]
s=''.join(c['source'])
a=s.index('    def directory('); b=s.index('    def record(',a)
s=s[:a]+'''    def directory(self, slug, required=False):
        markers = {
            'playground-series-s6e10': 'sample_submission.csv',
            's6e10-oof-library': 'oof_predictions.parquet',
            's6e10-tfm-oof-and-test-predictions': 'oof_*.npy',
            's6e10-stack-oof-predictions': '*.npz',
            's6e10-tabpfn-route-categories-lb-0-96160': 'oof_members.csv',
        }
        mounts = [p for p in self.root.rglob('*') if p.is_dir() and p.name == slug]
        if self.root.name == slug:
            mounts.append(self.root)
        matches = set()
        for mount in mounts:
            matches.update(p.parent for p in mount.rglob(markers[slug]) if p.is_file())
        # Competition and ID-keyed library schemas have distinctive filenames.
        if not matches and slug in ('playground-series-s6e10', 's6e10-oof-library'):
            matches.update(p.parent for p in self.root.rglob(markers[slug]) if p.is_file())
        if len(matches) > 1:
            raise ValueError(f'Ambiguous input {slug}: {sorted(map(str, matches))}. Keep one pinned version.')
        if not matches and required:
            raise FileNotFoundError(f'Attach {slug}; expected {markers[slug]} under {self.root}')
        return next(iter(matches)) if matches else None

'''+s[b:]
s=s.replace("directory.glob('oof_*.npy')", "directory.rglob('oof_*.npy')")
s=s.replace("test_path = directory/('test_'+path.name[4:])", "test_path = path.with_name('test_'+path.name[4:])")
s=s.replace("directory.glob('*.npz')", "directory.rglob('*.npz')")
s=s.replace("        if len(self.rows)<2:raise ValueError('Attach at least two valid paired OOF/test members')", '''        if len(self.rows) < 2:
            visible = sorted(str(p.relative_to(self.root)) for p in self.root.rglob('*') if p.is_file())
            rejected = [row for row in self.audit if row['status'] != 'accepted']
            raise ValueError(
                f'Loaded {len(self.rows)} unique valid OOF/test members. '
                'Competition train/test CSVs alone do not contain model predictions. '
                'In Kaggle, use Add Input to attach megayak/s6e10-oof-library and '
                'cdeotte/s6e10-tfm-oof-and-test-predictions, then Run All. '
                f'Input root: {self.root}. Visible files (first 40): {visible[:40]}. '
                f'Rejected or duplicate members: {rejected[:20]}'
            )''')
ast.parse(s)
c['source']=s.splitlines(keepends=True)
# Existing outputs belong to the earlier run; retain them as documented research evidence.
n['cells'][2]['source'] += ['\n\n**Kaggle input setup:** Competition CSV files alone are insufficient. Use **Add Input** to attach both public prediction datasets above, then **Run All**. The loader supports nested dataset directories and reports visible files when predictions are missing.\n']
p.write_text(json.dumps(n,ensure_ascii=False,indent=1)+'\n')
# Exercise discovery without loading large prediction libraries.
namespace={'Path':Path}
exec(s,namespace)
Catalog=namespace['InputCatalog']
with tempfile.TemporaryDirectory() as temp:
    obj=Catalog.__new__(Catalog); obj.root=Path(temp)
    nested=obj.root/'datasets/owner/s6e10-tfm-oof-and-test-predictions/v1'
    nested.mkdir(parents=True); (nested/'oof_model.npy').touch()
    assert obj.directory('s6e10-tfm-oof-and-test-predictions') == nested
    mirror=obj.root/'renamed-library'; mirror.mkdir(); (mirror/'oof_predictions.parquet').touch()
    assert obj.directory('s6e10-oof-library') == mirror
    assert obj.directory('s6e10-stack-oof-predictions') is None
    duplicate=obj.root/'another-library'; duplicate.mkdir(); (duplicate/'oof_predictions.parquet').touch()
    try: obj.directory('s6e10-oof-library')
    except ValueError as error: assert 'Ambiguous' in str(error)
    else: raise AssertionError('Ambiguous inputs accepted')
for cell in n['cells']:
    if cell['cell_type']=='code': ast.parse(''.join(cell['source']))
print('Passed nested discovery, renamed library, missing source, ambiguity, and notebook syntax checks.')
