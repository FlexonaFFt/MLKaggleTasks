"""Disabled-by-default portable Smol final plain component; no training, targets, or optimizer."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_contract(spec, artifact_root, base_root, diagnostic_only=False):
    """Stdlib-only full-final admission and immutable portable file checks."""
    assert spec["actual_final_binding"], "Source only: actual full-final QA/binding missing"
    binding = spec["actual_final_binding"]
    assert binding["actual_full_QA_passed"] is True
    if diagnostic_only:
        assert spec["diagnostic_only_authorized"] is True and spec["inference_authorized"] is False
        assert binding["paired_gain_admitted_by_root"] is False, "Diagnostic must not claim candidate admission"
    else:
        assert spec["inference_authorized"] is True and binding["paired_gain_admitted_by_root"] is True
    assert binding["notebook_id"] == 139347470 and binding["version"] == 1
    assert binding["script_version_id"] > 0
    assert binding["worker_sha256"] == spec["training_worker_sha256"]
    assert binding["checkpoint_relative"] == "final/finished_checkpoint"
    artifact_root, base_root = Path(artifact_root), Path(base_root)
    complete_path = artifact_root / "final/COMPLETE.json"
    report_path = artifact_root / "final/continuation-report.json"
    assert sha(complete_path) == binding["complete_sha256"] and sha(report_path) == binding["report_sha256"]
    complete, report = json.loads(complete_path.read_text()), json.loads(report_path.read_text())
    assert complete["report_sha256"] == binding["report_sha256"]
    assert complete["status"] == report["status"] == "COMPLETE_SMOL_EXACT_CONTINUATION"
    assert complete["job"] == report["job"] == "final" and complete["successful_updates"] == report["successful_updates"] == 450
    assert report["full_planned_updates"] == 450 and report["resumed_successful_updates"] == 20
    assert report["new_successful_updates"] == 430 and report["completed_epochs"] == report["full_planned_epochs"] == 3
    assert report["final_is_not_OOF"] and report["fit_rows"] == list(range(2699)) and report["valid_rows"] == []
    assert len(report["fit_ids"]) == len(set(report["fit_ids"])) == 2699 and report["valid_ids"] == []
    assert report["classes"] == spec["classes"] and report["train_sha256"] == spec["train_sha256"]
    assert report["worker_sha256"] == spec["training_worker_sha256"]
    assert report["base_revision"] == spec["base_revision"] and report["base_manifest_sha256"] == spec["base_manifest_sha256"]
    assert report["versions"] == spec["core_versions"] and report["pretrained_base_only"] and not report["quantization"]
    assert report["strict_reload_replay_passed"] and report["inference_artifact_ready"] and not report["validation_scored"]
    assert complete["inference_artifacts_ready"] and report["checkpoint"] == "finished_checkpoint"
    assert report["query_artifact"] is None and complete["query_artifact"] is None
    assert max(report["probability_replay_maxabs"].values()) <= 1e-6
    assert [record["update"] for record in report["updates"]] == list(range(1, 451))
    assert all(record["finite_gradients"] for record in report["updates"])
    checkpoint = artifact_root / binding["checkpoint_relative"]
    required = {"adapter/adapter_model.safetensors", "adapter/adapter_config.json", "heads.safetensors",
                "readouts.npz", "replay_anchors.npz"}
    assert required <= set(binding["portable_files_sha256"])
    for relative, digest in binding["portable_files_sha256"].items():
        path = checkpoint / relative
        assert path.resolve().is_relative_to(checkpoint.resolve())
        assert relative != "continuation_state.pt", "Optimizer state is unnecessary for inference"
        assert digest == complete["checkpoint_files_sha256"][relative] and sha(path) == digest
    config = json.loads((checkpoint / "adapter/adapter_config.json").read_text())
    assert config["r"] == 16 and config["lora_alpha"] == 32 and config["lora_dropout"] == .05
    assert config["bias"] == "none" and config["peft_type"] == "LORA"
    stage_path = base_root / "smollm3_staging_complete.json"
    assert sha(stage_path) == spec["base_manifest_sha256"]
    stage = json.loads(stage_path.read_text())
    assert stage["status"] == "COMPLETE_PINNED_PUBLIC_BASE_ONLY" and stage["revision"] == spec["base_revision"]
    assert stage["license"] == "Apache-2.0" and not stage["competition_data"] and not stage["model_initialized"]
    base_dir = base_root / "smollm3-base"
    assert len(stage["files"]) == 9
    for record in stage["files"]:
        assert sha(base_dir / record["name"]) == record["sha256"]
        assert (base_dir / record["name"]).stat().st_size == spec["base_files"][record["name"]]["size"]
        pinned = spec["base_files"][record["name"]].get("lfs_sha256")
        assert not pinned or pinned == record["sha256"]
    return checkpoint, base_dir, report


class SmolFinalPlain:
    def __init__(self, spec_path, artifact_root, base_root, anchor_train_csv, diagnostic_only=False):
        self.spec = json.loads(Path(spec_path).read_text())
        for name, digest in self.spec["source_hashes"].items():
            assert sha(Path(__file__).parent / name) == digest
        checkpoint, base_dir, report = validate_contract(self.spec, artifact_root, base_root, diagnostic_only=diagnostic_only)
        assert os.name != "nt" and Path("/kaggle/input").is_dir(), "Remote-only component"
        assert sha(anchor_train_csv) == self.spec["train_sha256"]
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import importlib.util
        finder = importlib.util.find_spec
        importlib.util.find_spec = lambda name, *a, **kw: None if name.split(".")[0] == "torchao" else finder(name, *a, **kw)
        # Keep the successful FULL worker's torchao mask active through PEFT dispatch.
        import torch
        from torch import nn
        from peft import PeftModel
        import peft.import_utils as peft_imports
        peft_imports.is_torchao_available = lambda *a, **kw: False
        import numpy as np
        import pandas as pd
        from safetensors.torch import load_file
        from scipy.special import log_softmax, softmax
        from owned_path import load_owned_base, make_representation, tokenize_owned

        assert torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (7, 5)
        assert {name: importlib.metadata.version(name) for name in self.spec["core_versions"]} == self.spec["core_versions"]
        torch.set_num_threads(2)
        loading = []
        base = load_owned_base(base_dir, loading, lambda: None)
        self.model = PeftModel.from_pretrained(base, checkpoint / "adapter", is_trainable=False, local_files_only=True).eval()
        assert not any(parameter.requires_grad for parameter in self.model.parameters())
        assert all(parameter.dtype == torch.float32 for name, parameter in self.model.named_parameters() if "lora_" in name)
        self.layers = [6, 12, 18, 24, 30, 36]
        self.norms = nn.ModuleList([nn.LayerNorm(2048) for _ in self.layers]).cuda().float()
        saved = load_file(checkpoint / "heads.safetensors")
        assert all(value.dtype == torch.float32 and bool(torch.isfinite(value).all()) for value in saved.values())
        self.norms.load_state_dict({key.removeprefix("norms."): value for key, value in saved.items() if key.startswith("norms.")}, strict=True)
        self.norms.eval()
        self.norms.requires_grad_(False)
        with np.load(checkpoint / "readouts.npz", allow_pickle=False) as readouts:
            self.readouts = {key: readouts[key].copy() for key in readouts.files}
        np.testing.assert_array_equal(self.readouts["classes"], np.asarray(self.spec["classes"], dtype=str))
        np.testing.assert_array_equal(self.readouts["pool_layers"], self.layers)
        assert self.readouts["plain_coef"].shape == self.readouts["center_coef"].shape == (18, 12288)
        assert self.readouts["plain_intercept"].shape == self.readouts["center_intercept"].shape == (18,)
        for route, prefix in (("plain", "plain"), ("center", "centered")):
            for key, state_key in (("coef", "weight"), ("intercept", "bias")):
                array = self.readouts[route + "_" + key]
                assert array.dtype == np.float32 and np.isfinite(array).all()
                np.testing.assert_array_equal(array, saved[prefix + "." + state_key].numpy())
        # Read only anchor IDs/text, never a target/model column. B3 and padding match the saved replay.
        anchor_data = pd.read_csv(anchor_train_csv, usecols=["id", "text"], dtype=str)
        assert len(anchor_data) == 2699 and anchor_data.id.tolist() == report["fit_ids"]
        self._diagnostic_only = diagnostic_only
        self._diagnostic_ids = anchor_data.id.iloc[:5].tolist()
        self._diagnostic_texts = anchor_data.text.iloc[:5].tolist()
        self.tokenizer, tokens = tokenize_owned(base_dir, anchor_data.text.iloc[:3].tolist())
        self._torch, self._np = torch, np
        self._make_representation, self._tokenize_owned = make_representation, tokenize_owned
        self._log_softmax = log_softmax
        with np.load(checkpoint / "replay_anchors.npz", allow_pickle=False) as anchors:
            np.testing.assert_array_equal(anchors["rows"], [0, 1, 2])
            np.testing.assert_array_equal(anchors["ids"], anchor_data.id.iloc[:3].to_numpy(dtype=str))
            np.testing.assert_array_equal(anchors["classes"], self.readouts["classes"])
            expected, recorded_reloaded = anchors["features"].copy(), anchors["reloaded_features"].copy()
        fresh = self._features(anchor_data.text.iloc[:3].tolist(), sort_by_length=False)
        assert expected.dtype == fresh.dtype == np.float32 and np.isfinite(expected).all()
        np.testing.assert_allclose(fresh, expected, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(fresh, recorded_reloaded, rtol=1e-5, atol=1e-5)
        parity = {}
        for route in ("plain", "center"):
            a = expected if route == "plain" else expected - expected.mean(0)
            b = fresh if route == "plain" else fresh - fresh.mean(0)
            coef, intercept = self.readouts[route + "_coef"], self.readouts[route + "_intercept"]
            before = softmax(log_softmax(a @ coef.T + intercept, axis=1), axis=1)
            after = softmax(log_softmax(b @ coef.T + intercept, axis=1), axis=1)
            parity[route] = float(np.max(np.abs(before - after)))
            assert parity[route] <= 1e-6
        self.anchor_replay = {"feature_maxabs": float(np.max(np.abs(fresh - expected))), "probability_maxabs": parity,
                              "feature_rtol": 1e-5, "feature_atol": 1e-5, "probability_atol": 1e-6}

    def _features(self, texts, sort_by_length=True):
        torch, np = self._torch, self._np
        tokenizer, tokens = self._tokenize_owned(None, texts, tokenizer=self.tokenizer)
        representation = self._make_representation(self.model, tokenizer, tokens, self.layers, self.norms)
        features = np.empty((len(tokens), 12288), dtype=np.float32)
        order = np.argsort(np.asarray([len(row) for row in tokens]), kind="stable") if sort_by_length else np.arange(len(tokens))
        with torch.inference_mode():
            for start in range(0, len(order), 3):
                positions = order[start:start + 3]
                features[positions] = representation(positions)[0].cpu().numpy()
        assert np.isfinite(features).all()
        return features

    def predict_plain_logp(self, ids, texts):
        np = self._np
        ids = np.asarray(ids, dtype=str)
        assert ids.ndim == 1 and len(ids) == len(texts) > 0 and len(np.unique(ids)) == len(ids)
        if self._diagnostic_only:
            assert len(ids) in (3, 5) and ids.tolist() == self._diagnostic_ids[:len(ids)]
            assert list(texts) == self._diagnostic_texts[:len(ids)], "Only exact owned3/5-row diagnostic fixtures admitted"
        features = self._features(texts)
        logp = self._log_softmax(features @ self.readouts["plain_coef"].T + self.readouts["plain_intercept"], axis=1)
        assert logp.shape == (len(ids), 18) and logp.dtype == np.float32 and np.isfinite(logp).all()
        return {"ids": ids, "classes": self.readouts["classes"].copy(), "plain_logp": logp}
