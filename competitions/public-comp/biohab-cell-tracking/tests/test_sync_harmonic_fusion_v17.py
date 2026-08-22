import unittest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sync_harmonic_fusion_v17 import (
    CJK,
    fix_guard_print,
    require_input_guard,
    preset_cell,
    require_preset_disables_validator,
    strip_notebook_notes,
    validate_notebook,
)

UPSTREAM_SNIPPET = "\n".join((
    "SUPPORT = 'biohub-tracking-support-pack-50ep-v1'",
    "SECONDARY = 'biohub-temporal-unet3d-seed314159-v1'",
    "os.environ['BIOHUB_SECONDARY_DETECTION_WEIGHT'] = '0.475'",
    "secondary_detection_weight = 0.475",
    "os.environ['BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT'] = '0.30'",
    'os.environ["BIOHUB_BIDIRECTIONAL_FUSION_MODE"] = "harmonic_probability"',
    '"BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT": 0.30,',
    "_bidirectional_weight_guard, 0.30, rel_tol=0.0",
    "os.environ['BIOHUB_DEEPCENTER_CHECKPOINT'] ="
    " '/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center/best.pt'",
    "_support_expected_manifest_sha256 = '0' * 64",
    'raise RuntimeError("Support repo manifest checksum mismatch")',
    "missing_datasets = sorted(expected - seen)",
    'raise RuntimeError("Retention-guard submission schema changed")',
    'VALIDATOR_ENABLE = os.environ.get("BIOHUB_VALIDATOR_ENABLE", "1") != "0"',
))


class HarmonicFusionSyncTest(unittest.TestCase):
    def test_validates_fusion_stack(self):
        validate_notebook({"cells": [{"source": UPSTREAM_SNIPPET}]})

    def test_rejects_non_fusion_notebook(self):
        with self.assertRaisesRegex(RuntimeError, "Unexpected upstream"):
            validate_notebook({"cells": [{"source": "print('hi')"}]})

    def test_rejects_wrong_fusion_mode(self):
        bad = UPSTREAM_SNIPPET.replace(
            '"harmonic_probability"', '"arithmetic_logit"'
        )
        with self.assertRaisesRegex(RuntimeError, "harmonic_probability"):
            validate_notebook({"cells": [{"source": bad}]})

    def test_rejects_wrong_bidirectional_weight(self):
        bad = UPSTREAM_SNIPPET.replace(
            "_bidirectional_weight_guard, 0.30, rel_tol=0.0",
            "_bidirectional_weight_guard, 0.20, rel_tol=0.0",
        )
        with self.assertRaisesRegex(RuntimeError, "weight guard is not 0.30"):
            validate_notebook({"cells": [{"source": bad}]})

    def test_removes_markdown_comments_and_cjk(self):
        notebook = {"cells": [
            {"cell_type": "markdown", "source": "# 日本語"},
            {"cell_type": "code", "source": "x = 1  # コメント\nprint('完了')\n"},
        ]}
        strip_notebook_notes(notebook)
        self.assertEqual(len(notebook["cells"]), 1)
        source = "".join(notebook["cells"][0]["source"])
        self.assertNotIn("#", source)
        self.assertIsNone(CJK.search(source))

    def test_fixes_guard_log_weight(self):
        notebook = {"cells": [
            {"source": "print('Reverse-time association weight: 0.200')\n"}
        ]}
        fix_guard_print(notebook)
        self.assertIn(
            "Reverse-time association weight: 0.300",
            "".join(notebook["cells"][0]["source"]),
        )

    def test_fix_guard_print_requires_upstream_line(self):
        notebook = {"cells": [{"source": "print('unrelated')\n"}]}
        with self.assertRaisesRegex(RuntimeError, "Guard log line not found"):
            fix_guard_print(notebook)

    def test_preset_disables_validator(self):
        notebook = {"cells": [preset_cell(17)]}
        require_preset_disables_validator(notebook)
        source = "".join(notebook["cells"][0]["source"])
        self.assertIn("BIOHUB_VALIDATOR_ENABLE'] = '0'", source)
        self.assertIn("yunusgmsoy/kimi-notebook-v17", source)

    def test_preset_guards_required_inputs(self):
        notebook = {"cells": [preset_cell(17)]}
        require_input_guard(notebook)
        source = "".join(notebook["cells"][0]["source"])
        for slug in (
            "biohub-tracking-support-pack-50ep-v1",
            "biohub-temporal-unet3d-seed314159-v1",
            "biohub-deepcenter-unet3d-center-prior-v1",
        ):
            self.assertIn(f"'/kaggle/input/{slug}'", source)
        self.assertIn("Missing Kaggle inputs", source)
        self.assertIn("yunusgmsoy/kimi-notebook-v17", source)


if __name__ == "__main__":
    unittest.main()
