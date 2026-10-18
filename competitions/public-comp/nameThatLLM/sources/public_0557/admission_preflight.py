"""Stdlib-only immutable diagnostic evidence and explicit candidate admission guards."""
import hashlib
import json
from pathlib import Path


def validate(here):
    here = Path(here)
    config = json.loads((here / 'optional-config.json').read_text())
    assert config['candidate_admitted'] and config['selected_by_root']
    assert config['assets_public_audited'] and config['diagnostic_actual_PASS_bound']
    assert config['root_decision_binding'] and config['root_decision_binding']['explicit_candidate_admission']
    binding = config['diagnostic_PASS_binding']
    assert binding['notebook_id'] == 139353455 and binding['version'] == 3 and binding['script_version_id'] == 359581988
    raw = (here / 'actual-diagnostic-audit.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == binding['audit_sha256']
    audit = json.loads(raw)
    assert audit['notebook_id'] == binding['notebook_id'] and audit['version'] == 3
    assert audit['component_sha256'] == config['source_sha256']['component.py']
    assert audit['both_physical_T4_forward_CUDA_events_proven'] and audit['training_updates'] == 0
    assert not audit['candidate_or_production_admitted']
    for rows in ('3', '5'):
        report = audit['reports'][rows]
        assert report['status'] == 'PASS_INFERENCE_DIAGNOSTIC_ONLY'
        assert report['component_sha256'] == config['source_sha256']['component.py']
        assert report['owned_path_sha256'] == config['source_sha256']['owned_path.py']
        assert report['observed_query_batch_sizes'] == ([3] if rows == '3' else [3, 2])
        assert report['prediction_cuda_event_ms'] > 0 and report['finite'] and report['id_order_preserved']
        replay = report['anchor_replay']
        assert replay['feature_rtol'] == replay['feature_atol'] == 1e-5 and replay['probability_atol'] == 1e-6
        assert replay['feature_maxabs'] == 0 and replay['probability_maxabs'] == {'plain': 0, 'center': 0}
    return config
