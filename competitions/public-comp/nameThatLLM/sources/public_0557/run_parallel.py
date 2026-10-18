"""Optional UUID-isolated GPU0 PFN128 and GPU1 owned Smol; no shared-GPU loading."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

here = Path(__file__).resolve().parent
optional = json.loads((here / 'optional-config.json').read_text())
config = json.loads((here / 'runtime_config.json').read_text())
assert optional['candidate_admitted'] and optional['selected_by_root'] and config['candidate_admitted']
assert optional['assets_public_audited'] and optional['diagnostic_actual_PASS_bound']
assert config['raw_extension'] == optional['raw_extension'] == 'smol_center_add'
assert config['tabpfn']['n_estimators'] == 128
assert os.name != 'nt' and Path('/kaggle/input').is_dir()
mapping = []
for line in subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name',
                                     '--format=csv,noheader,nounits'], text=True).strip().splitlines():
    index, uuid, name = [value.strip() for value in line.split(',', 2)]
    mapping.append(dict(physical_index=int(index), uuid=uuid, name=name))
mapping.sort(key=lambda gpu: gpu['physical_index'])
assert [gpu['physical_index'] for gpu in mapping] == [0, 1]
assert all('T4' in gpu['name'] and gpu['uuid'].startswith('GPU-') for gpu in mapping)
assert mapping[0]['uuid'] != mapping[1]['uuid']
out = Path('/kaggle/working/smol_pfn_parallel')
out.mkdir(exist_ok=False)
children, logs, records = [], [], []
started = time.monotonic()
try:
    for index, route in enumerate(('PFN128', 'SmolCENTER')):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = mapping[index]['uuid']
        env['HF_HUB_OFFLINE'] = env['TRANSFORMERS_OFFLINE'] = '1'
        # Preserve the already installed PFN-only add-on environment; Smol uses the original runtime.
        if route == 'PFN128':
            env['PYTHONPATH'] = os.environ['NTL_TABULAR_SITE']
        command = ([sys.executable, '-u', str(here / 'tabular_worker.py'), 'tabpfn'] if index == 0
                   else [sys.executable, '-u', str(here / 'compute_center.py')])
        log = (out / (route + '.log')).open('w', buffering=1)
        logs.append(log)
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(child)
        records.append(dict(route=route, PID=child.pid, GPU=mapping[index],
                            started_seconds=time.monotonic()-started, finished_seconds=None))
    tick, overlap_samples = 0, 0
    observed_compute_PIDs = set()
    while True:
        elapsed = time.monotonic()-started
        codes = [child.poll() for child in children]
        for i, code in enumerate(codes):
            if code is not None and records[i]['finished_seconds'] is None:
                records[i]['finished_seconds'] = elapsed
            if code is None:
                limit = config['model_timeout_seconds'] if i == 0 else optional['smol_timeout_seconds']
                assert elapsed < limit, (records[i]['route'], 'timeout; no automatic retry')
        if any(code is not None and code != 0 for code in codes):
            raise RuntimeError(f'Parallel worker failed{codes}; no finalization or automatic retry')
        sample = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.used',
                                         '--format=csv,noheader,nounits'], text=True, timeout=15)
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
                                       '--format=csv,noheader,nounits'], text=True, timeout=15)
        active = {}
        for line in sample.strip().splitlines():
            index, uuid, utilization, memory = [value.strip() for value in line.split(',')]
            active[uuid] = float(utilization) > 0
        matched = []
        for record in records:
            exact = any(fields[0].strip() == str(record['PID']) and fields[1].strip() == record['GPU']['uuid']
                        for fields in (line.split(',') for line in apps.strip().splitlines()) if len(fields) >= 2)
            if exact:
                observed_compute_PIDs.add(record['PID'])
            matched.append(exact)
        both_compute = all(code is None for code in codes) and all(matched) and all(active.get(gpu['uuid'], False) for gpu in mapping)
        overlap_samples += int(both_compute)
        with (out / 'gpu-telemetry.jsonl').open('a') as handle:
            handle.write(json.dumps(dict(elapsed_seconds=elapsed, GPU=sample, compute_apps=apps,
                                         worker_codes=codes, both_workers_matched_active=both_compute))+'\n')
        if tick % 6 == 0:
            print(json.dumps(dict(stage='PFN_SMOL_PROGRESS', elapsed_seconds=elapsed, workers=records,
                                  return_codes=codes, devices=sample)), flush=True)
        if all(code == 0 for code in codes):
            break
        tick += 1
        time.sleep(5)
    assert all(record['PID'] in observed_compute_PIDs for record in records), 'Actual per-worker GPU mapping not observed'
    report = json.loads(Path('/kaggle/working/smol_center_inference_audit.json').read_text())
    assert report['status'] == 'OWNED_FINAL_CENTER_FEATURE_PASS_COMPLETE' and report['query_cuda_event_ms'] > 0
    assert report['CUDA_VISIBLE_DEVICES'] == mapping[1]['uuid'] and report['PID'] == children[1].pid
    (out / 'COMPLETE.json').write_text(json.dumps(dict(status='BOTH_OPTIONAL_COMPONENTS_COMPLETE', workers=records,
        overlap_active_samples=overlap_samples, actual_compute_overlap_observed=overlap_samples > 0,
        missing_sample_caveat=None if overlap_samples else 'No overlapping active sample captured; no claim of measured compute overlap.',
        wall_seconds=time.monotonic()-started, automatic_retry=False), indent=2))
finally:
    for child in children:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
    for child in children:
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
    for log in logs:
        log.close()
