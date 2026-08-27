import unittest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_harmonic_safe_div_veto import OVERRIDE, inject_override, validate


class HarmonicSafeDivVetoTest(unittest.TestCase):
    def test_injects_before_configuration_is_read(self):
        notebook = {"cells": [
            {"cell_type": "code", "source": "GAP_CLOSE_MAX_GAP = int(os.environ.get('BIOHUB_GAP_CLOSE_MAX_GAP', '1'))\n"},
            {"cell_type": "code", "source": "BIOHUB_BIDIRECTIONAL_FUSION_MODE\n"},
        ]}
        inject_override(notebook)
        self.assertEqual("".join(notebook["cells"][0]["source"]), OVERRIDE)
        validate(notebook)


if __name__ == "__main__":
    unittest.main()
