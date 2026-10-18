'''OOP execution and artifact validation for the frozen public ensemble.'''

import ast
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    full_ensemble: bool
    input_root: Path = Path('/kaggle/input')
    work_dir: Path = Path('/kaggle/working')
    temp_root: Path = Path('/kaggle/temp')
    timeout_seconds: int = 32400

    @property
    def source_dir(self):
        return self.temp_root / self.name


class PublicSourceBundle:
    def __init__(self, digests):
        self.digests = digests
        self.sources = {}

    def add(self, name, source, crlf=False):
        if Path(name).name != name or name in self.sources:
            raise ValueError(f'Invalid or repeated source filename: {name}')
        if crlf is True:
            source = source.replace('\n', '\r\n')
        elif crlf:
            lines = source.splitlines(keepends=True)
            source = ''.join(line[:-1] + '\r\n' if i in crlf and line.endswith('\n') else line
                             for i, line in enumerate(lines))
        self.sources[name] = source

    def verify(self):
        if self.sources.keys() != self.digests.keys():
            raise ValueError('The source bundle is incomplete')
        for name, source in self.sources.items():
            if hashlib.sha256(source.encode()).hexdigest() != self.digests[name]:
                raise ValueError(f'Source integrity check failed: {name}')
            if name.endswith('.py'):
                ast.parse(source, filename=name)

    def materialize(self, directory):
        self.verify()
        directory.mkdir(parents=True, exist_ok=True)
        for name, source in self.sources.items():
            path = directory / name
            expected = source.encode()
            if path.exists() and path.read_bytes() != expected:
                raise FileExistsError(f'Refusing to overwrite different source: {path}')
            path.write_bytes(expected)


class KagglePreflight:
    def __init__(self, config, bundle):
        self.config = config
        self.bundle = bundle

    def check(self):
        if not self.config.input_root.is_dir():
            raise RuntimeError('Run on Kaggle with the attached model inputs and two T4 GPUs')
        import pandas as pd
        import torch
        if torch.cuda.device_count() != 2 or any(
            'T4' not in torch.cuda.get_device_name(i) for i in range(2)
        ):
            raise RuntimeError('This reproduction requires GPU T4 x2')
        data_dirs = [
            self.config.input_root / 'competitions/name-that-llm',
            self.config.input_root / 'name-that-llm',
        ]
        data = next((p for p in data_dirs if (p / 'train.csv').is_file()), None)
        if data is None or not (data / 'test.csv').is_file():
            raise FileNotFoundError('Attach the Name That LLM competition data')
        train = pd.read_csv(data / 'train.csv')
        test = pd.read_csv(data / 'test.csv')
        if not {'id', 'text', 'model', 'prompt_id'}.issubset(train.columns):
            raise ValueError('Unexpected training data schema')
        if not {'id', 'text'}.issubset(test.columns) or test.empty:
            raise ValueError('Unexpected or empty test data')
        for frame in (train, test):
            if frame['id'].isna().any() or frame['id'].duplicated().any():
                raise ValueError('IDs must be unique and non-null')
        labels = json.loads(self.bundle.sources['smol-spec.json'])['classes']
        if set(train['model']) != set(labels):
            raise ValueError('Expected all 18 official author labels')
        patterns = [
            '**/glm4_9b_trunc25/config.json',
            '**/deberta-v3-large/config.json',
            '**/electra-large-discriminator/config.json',
            '**/desklib-detector/config.json',
            '**/lora24/final/COMPLETE.json',
            '**/lora_out/job_final/COMPLETE.json',
            '**/ntl-qwen3-8b-lora-claude-head/**/final/COMPLETE.json',
            '**/tokattn_final.pt',
            '**/tokattn_debl23_final.pt',
            '**/ultrabert-380m-ntl/fold*/config.json',
            '**/ntl_extra_passages_claude.csv',
        ]
        missing = [pattern for pattern in patterns if not next(self.config.input_root.glob(pattern), None)]
        if missing:
            raise FileNotFoundError('Missing public ensemble assets: ' + ', '.join(missing))
        if self.config.full_ensemble:
            self._check_full_assets(data)
        return data, test, labels

    def _check_full_assets(self, data):
        spec = json.loads(self.bundle.sources['smol-spec.json'])
        actual = {name: importlib.metadata.version(name) for name in spec['core_versions']}
        if actual != spec['core_versions']:
            raise RuntimeError(f'Use the pinned source Docker image. Expected {spec["core_versions"]}; got {actual}')
        if hashlib.sha256((data / 'train.csv').read_bytes()).hexdigest() != spec['train_sha256']:
            raise ValueError('Official training data differs from the Smol checkpoint training data')
        optional = json.loads(self.bundle.sources['optional-config.json'])
        markers = {
            'asset_manifest.json': optional['compact_asset_manifest_sha256'],
            'smollm3_staging_complete.json': spec['base_manifest_sha256'],
        }
        for marker, digest in markers.items():
            matches = [p for p in self.config.input_root.rglob(marker)
                       if hashlib.sha256(p.read_bytes()).hexdigest() == digest]
            if len(matches) != 1:
                raise FileNotFoundError(f'Attach exactly one matching public Smol asset: {marker}')
        runtime = json.loads(self.bundle.sources['runtime_config.json'])
        tabular = list(self.config.input_root.rglob(runtime['tabpfn']['checkpoint_filename']))
        if len(tabular) != 1 or not (tabular[0].parent / 'runtime-addon-lock.txt').is_file():
            raise FileNotFoundError('Attach the pinned NTL Tabular Offline Assets dataset')


class SubmissionArtifact:
    def __init__(self, config):
        self.path = config.work_dir / 'submission.parquet'

    def validate(self, test, labels):
        import pandas as pd
        if not self.path.is_file():
            raise FileNotFoundError(f'The ensemble did not produce {self.path}')
        result = pd.read_parquet(self.path)
        if list(result.columns) != ['id', 'model'] or len(result) != len(test):
            raise ValueError('Submission schema or row count differs from the official test data')
        if result.isna().any().any() or result['id'].duplicated().any():
            raise ValueError('Submission contains null values or repeated IDs')
        if not result['id'].equals(test['id'].reset_index(drop=True)):
            raise ValueError('Submission ID order differs from test.csv')
        if not result['model'].isin(labels).all():
            raise ValueError('Submission contains an unknown author label')
        return result

    def quarantine(self):
        if self.path.exists():
            destination = self.path.with_name(f'failed_run_{time.time_ns()}.parquet')
            self.path.rename(destination)


class ReferenceEnsembleExperiment:
    def __init__(self, config, bundle):
        self.config = config
        self.bundle = bundle
        self.artifact = SubmissionArtifact(config)

    def command(self):
        return [sys.executable, '-u', str(self.config.source_dir / 'ntl_reference.py')]

    def run(self):
        self.bundle.materialize(self.config.source_dir)
        data, test, labels = KagglePreflight(self.config, self.bundle).check()
        self.config.work_dir.mkdir(parents=True, exist_ok=True)
        if self.artifact.path.exists():
            raise FileExistsError('submission.parquet already exists; start a clean notebook session')
        environment = os.environ.copy()
        environment.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
        started = time.monotonic()
        try:
            subprocess.run(self.command(), cwd=self.config.work_dir, env=environment,
                           check=True, timeout=self.config.timeout_seconds)
            submission = self.artifact.validate(test, labels)
        except BaseException:
            self.artifact.quarantine()
            raise
        summary = {
            'experiment': self.config.name,
            'method': 'full_smol_protected_tabpfn' if self.config.full_ensemble else 'reference_centered_stylegram',
            'test_rows': len(test),
            'elapsed_seconds': time.monotonic() - started,
            'reference_sha256': self.bundle.digests['ntl_reference.py'],
            'submission_sha256': hashlib.sha256(self.artifact.path.read_bytes()).hexdigest(),
            'source_modules_preserved': True,
            'fallback_used': False,
            'score_verified': False,
        }
        (self.config.work_dir / 'reproduction_summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return submission


class FullEnsembleExperiment(ReferenceEnsembleExperiment):
    def command(self):
        return [sys.executable, '-u', str(self.config.source_dir / 'run_candidate_optional.py')]
