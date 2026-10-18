'''Build self-contained reproduction notebooks from an immutable public notebook.'''

import argparse
import ast
import hashlib
import json
from pathlib import Path
import textwrap

import nbformat


class ReproductionBuilder:
    def __init__(self, source, api_metadata):
        self.root = Path(__file__).resolve().parents[1]
        self.source_path = Path(source)
        self.source = nbformat.read(self.source_path, as_version=4)
        self.metadata = json.loads(Path(api_metadata).read_text())
        assignments = {}
        for cell in self.source.cells:
            if cell.cell_type != 'code':
                continue
            for node in ast.parse(cell.source).body:
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in ('PAYLOAD', 'DIGESTS'):
                            assignments[target.id] = json.loads(ast.literal_eval(node.value.args[0]))
        self.payload = assignments['PAYLOAD']
        self.digests = assignments['DIGESTS']
        if self.payload.keys() != self.digests.keys():
            raise ValueError('Incomplete upstream payload')
        for name, source_text in self.payload.items():
            if Path(name).name != name:
                raise ValueError(name)
            if hashlib.sha256(source_text.encode()).hexdigest() != self.digests[name]:
                raise ValueError(f'Upstream digest mismatch: {name}')

    @staticmethod
    def cell(title, source=''):
        return nbformat.v4.new_code_cell(f"'''{title}\n'''\n" + source.rstrip() + '\n')

    def archive(self):
        directory = self.root / 'sources/public_0557'
        directory.mkdir(parents=True, exist_ok=True)
        for name, source_text in self.payload.items():
            (directory / name).write_bytes(source_text.encode())
        origin = {
            'source_file': self.source_path.name,
            'source_file_sha256': hashlib.sha256(self.source_path.read_bytes()).hexdigest(),
            'upstream_url': 'https://www.kaggle.com/code/lightningv08/0-557-lora-smollm3-tabpfn',
            'reference_url': 'https://www.kaggle.com/code/obstacledeveloper/local-candidate-adaptive-centered-stylegram?scriptVersionId=359313348',
            'payload_digests': self.digests,
            'original_kaggle_metadata': dict(self.source.metadata.get('kaggle', {})),
            'original_description': self.source.cells[0].source,
            'api_metadata_note': 'The live upstream notebook has changed. Scientific code is frozen from the user-provided file; API metadata is used only for resource slugs and Docker image.',
        }
        (directory / 'origin.json').write_text(json.dumps(origin, indent=2) + '\n')
        (directory / 'resource-metadata.json').write_text(json.dumps(self.metadata, indent=2) + '\n')

    def runtime_cells(self):
        runtime = (self.root / 'scripts/reproduction_runtime.py').read_text()
        tree = ast.parse(runtime)
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        lines = runtime.splitlines(keepends=True)
        start = min([classes[0].lineno] + [d.lineno for d in classes[0].decorator_list])
        cells = [self.cell('Name That LLM | Imports', ''.join(lines[1:start - 1]))]
        for node in classes:
            begin = min([node.lineno] + [d.lineno for d in node.decorator_list])
            cells.append(self.cell(f'Name That LLM | {node.name}', ''.join(lines[begin - 1:node.end_lineno])))
        return cells

    def notebook(self, number, full):
        title = 'SmolLM3 + Protected TabPFN Reproduction' if full else 'Public Reference Ensemble Reproduction'
        intro = textwrap.dedent(f'''\
            Name That LLM | Iteration {number:02d}: {title}
            Goal: reproduce the strong public inference pipeline with its original
            models, coefficients, cluster centering, balancing, and Claude experts.
            {'This attempt adds the original Smol CENTER 0.03 and protected PFN 0.10.' if full else 'This control runs the original reference without the optional Smol/PFN stages.'}
            This is an OOP orchestration of frozen upstream modules. Their numerical
            implementation is preserved verbatim; it has not been fully refactored
            into methods. Source hashes are checked before any model is executed.
            Our score has not been measured. No fallback classifier is used.
            Runtime: GPU T4 x2, internet disabled, pinned upstream Docker image.
            Output: /kaggle/working/submission.parquet, validated against test.csv.
            The two attempts are independent and can run in separate sessions.''')
        credit = textwrap.dedent('''\
            Name That LLM | Credits and Source Provenance
            Vladislav Lavrentev: original LoRA, SmolLM3 training, TabPFN extension,
            and the supplied inference notebook:
            https://www.kaggle.com/code/lightningv08/0-557-lora-smollm3-tabpfn
            Obstacle Developer: adaptive centered stylegram reference, pinned source:
            https://www.kaggle.com/code/obstacledeveloper/local-candidate-adaptive-centered-stylegram?scriptVersionId=359313348
            Lucas Boesen: backbone ensemble, features, attention heads, and centering:
            https://www.kaggle.com/code/lucasboesen/lora-x-adaptive-centering
            Zbigniew Kirchner: additional Qwen LoRA and Claude experts:
            https://www.kaggle.com/code/zbigniewkirchner/0-552-adaptive-centering-own-lora-claude-gat
            seyitkaangunes: public Claude API passages. pjmathematician: UltraBERT
            fold weights. kirilltym: token-attention recipe. Upstream notices and
            component licenses are preserved; this wrapper does not relicense them.
            SmolLM3: Apache 2.0. TabPFN weights: Prior Labs Non-Commercial License.
            Changes: OOP runtime, readable notebook sections, early asset checks,
            final output validation, and quarantine of partial outputs on failure.
            The original source is provided by the user; the current live notebook
            has a newer method. We deliberately reproduce the supplied version.''')
        cells = [self.cell(intro), self.cell(credit)] + self.runtime_cells()
        cells.append(self.cell('Name That LLM | Frozen Source Manifest',
                               'SOURCE_DIGESTS = ' + repr(self.digests) + '\nbundle = PublicSourceBundle(SOURCE_DIGESTS)'))
        for name, source_text in self.payload.items():
            normalized = source_text.replace('\r\n', '\n')
            if "'''" in normalized or normalized.endswith('\\'):
                raise ValueError(f'Cannot embed this upstream file as a readable raw string: {name}')
            crlf = tuple(i for i, line in enumerate(source_text.splitlines(keepends=True)) if line.endswith('\r\n'))
            if crlf and len(crlf) == normalized.count('\n'):
                crlf = True
            cells.append(self.cell(
                f'Name That LLM | Upstream Module: {name}\nPreserved source dependency; verified against its original SHA-256.',
                f"bundle.add({name!r}, r'''\n" + normalized + f"\n'''[1:-1], crlf={crlf!r})"))
        name = f'ntl_{number:02d}_' + ('smol_protected_pfn' if full else 'public_reference')
        cls = 'FullEnsembleExperiment' if full else 'ReferenceEnsembleExperiment'
        cells.append(self.cell('Name That LLM | Experiment Configuration',
            f"config = ExperimentConfig(name={name!r}, full_ensemble={full!r})\nexperiment = {cls}(config, bundle)\nbundle.verify()\nprint(f'Frozen source integrity verified: {{len(bundle.sources)}} files')"))
        cells.append(self.cell('Name That LLM | Run and Validate Submission\nRun in a clean Kaggle session. Missing assets or incompatible versions stop\nbefore inference; partial output is quarantined if the run fails.',
                               'submission = experiment.run()\nsubmission.head()'))
        metadata = {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python'},
            'kaggle': dict(self.source.metadata.get('kaggle', {})),
        }
        metadata['kaggle']['isGpuEnabled'] = True
        metadata['kaggle']['isInternetEnabled'] = False
        result = nbformat.v4.new_notebook(cells=cells, metadata=metadata)
        nbformat.validate(result)
        return result

    def build(self):
        self.archive()
        for number, full, filename, folder, slug, title in [
            (3, False, '03_public_reference_oop.ipynb', 'reference', 'name-that-llm-03-public-reference-oop', 'Name That LLM 03 Public Reference OOP'),
            (4, True, '04_public_0557_reproduction_oop.ipynb', 'full', 'name-that-llm-04-smol-tabpfn-reproduction', 'Name That LLM 04 Smol TabPFN Reproduction'),
        ]:
            nbformat.write(self.notebook(number, full), self.root / filename)
            metadata = dict(self.metadata)
            metadata.pop('id_no', None)
            metadata.update(id=f'flexonafft/{slug}', title=title, code_file='../../' + filename,
                            is_private=True, enable_gpu=True, enable_tpu=False,
                            enable_internet=False, machine_shape='NvidiaTeslaT4',
                            docker_image_pinning_type='original')
            metadata['dataset_sources'] = [s for s in metadata['dataset_sources']
                                           if s != 'lightningv08/ntl-tabicl-offline-assets']
            metadata['dataset_sources'] = [s + '/1' if s == 'lightningv08/ntl-smollm3-final-inference-assets' else s
                                           for s in metadata['dataset_sources']]
            if not full:
                metadata['dataset_sources'] = [s for s in metadata['dataset_sources']
                                               if not s.startswith(('lightningv08/ntl-smollm3-', 'lightningv08/ntl-tabular-'))]
                metadata['kernel_sources'] = [s for s in metadata['kernel_sources']
                                              if s != 'lightningv08/smollm3-3b-base-offline-weights']
            directory = self.root / 'kaggle' / folder
            directory.mkdir(parents=True, exist_ok=True)
            (directory / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
            print(self.root / filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('api_metadata')
    args = parser.parse_args()
    ReproductionBuilder(args.source, args.api_metadata).build()
