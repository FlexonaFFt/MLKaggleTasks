"""Remote-only plain-component adapter; no submission, training, or automatic admission."""
import argparse
import json
from pathlib import Path
from component import SmolFinalPlain, validate_contract

parser = argparse.ArgumentParser()
parser.add_argument("--spec", default="inference-spec.json")
parser.add_argument("--artifact-root", required=True)
parser.add_argument("--base-root", required=True)
parser.add_argument("--anchor-train-csv", required=True)
parser.add_argument("--test-csv")
parser.add_argument("--output", default="/kaggle/working/smollm3_plain_logp.npz")
parser.add_argument("--validate-only", action="store_true")
args = parser.parse_args()
spec = json.loads(Path(args.spec).read_text())
validate_contract(spec, args.artifact_root, args.base_root)
if args.validate_only:
    print(json.dumps({"status": "PASS_FULL_FINAL_FILES_NO_MODEL_INIT", "Torch_imported": False}), flush=True)
else:
    assert args.test_csv
    model = SmolFinalPlain(args.spec, args.artifact_root, args.base_root, args.anchor_train_csv)
    import numpy as np
    import pandas as pd
    test = pd.read_csv(args.test_csv, usecols=["id", "text"], dtype=str)
    assert test.notna().all().all() and test.id.is_unique
    result = model.predict_plain_logp(test.id.to_numpy(dtype=str), test.text.tolist())
    np.testing.assert_array_equal(result["ids"], test.id.to_numpy(dtype=str))
    np.savez_compressed(args.output, **result)
    Path(args.output).with_suffix(".json").write_text(json.dumps({"status": "PLAIN_COMPONENT_ONLY",
        "rows": len(test), "anchor_replay": model.anchor_replay,
        "training_or_submission_performed": False, "centering_or_calibration_applied": False}, indent=2))
