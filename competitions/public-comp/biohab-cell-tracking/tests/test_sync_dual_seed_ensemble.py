import unittest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sync_dual_seed_ensemble import (
    CJK,
    add_partial_submission_guard,
    strip_notebook_notes,
    validate_notebook,
)


class DualSeedSyncTest(unittest.TestCase):
    def test_validates_and_guards_submission(self):
        notebook = {"cells": [{"source": "\n".join((
            "biohub-temporal-unet3d-seed314159-v1",
            "biohub-tracking-support-pack-50ep-v1",
            "BIOHUB_SECONDARY_WEIGHTS",
            "secondary_model",
            "secondary_detection_weight",
            "missing_datasets",
            "n_rows = write_submission(results, OUT)",
        ))}]}
        validate_notebook(notebook)
        add_partial_submission_guard(notebook)
        self.assertIn("Refusing partial submission", "".join(notebook["cells"][0]["source"]))

    def test_rejects_non_ensemble_notebook(self):
        with self.assertRaisesRegex(RuntimeError, "Unexpected upstream"):
            validate_notebook({"cells": [{"source": "n_rows = write_submission(results, OUT)"}]})

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


if __name__ == "__main__":
    unittest.main()
