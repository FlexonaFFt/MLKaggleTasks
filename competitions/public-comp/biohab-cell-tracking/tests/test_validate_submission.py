import csv
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from validate_submission import COLUMNS, validate


class ValidateSubmissionTest(unittest.TestCase):
    def write_rows(self, rows):
        tmp = tempfile.NamedTemporaryFile(mode="w", newline="", suffix=".csv", delete=False)
        with tmp:
            writer = csv.DictWriter(tmp, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        return Path(tmp.name)

    def test_valid_division(self):
        rows = [
            dict(zip(COLUMNS, [0, "sample", "node", 1, 0, 2, 3, 4, -1, -1])),
            dict(zip(COLUMNS, [1, "sample", "node", 2, 1, 2, 3, 5, -1, -1])),
            dict(zip(COLUMNS, [2, "sample", "node", 3, 1, 2, 4, 4, -1, -1])),
            dict(zip(COLUMNS, [3, "sample", "edge", -1, -1, -1, -1, -1, 1, 2])),
            dict(zip(COLUMNS, [4, "sample", "edge", -1, -1, -1, -1, -1, 1, 3])),
        ]
        self.assertEqual(validate(self.write_rows(rows)), [])

    def test_rejects_multiple_parents_and_frame_gap(self):
        rows = [
            dict(zip(COLUMNS, [0, "sample", "node", 1, 0, 0, 0, 0, -1, -1])),
            dict(zip(COLUMNS, [1, "sample", "node", 2, 0, 0, 0, 1, -1, -1])),
            dict(zip(COLUMNS, [2, "sample", "node", 3, 2, 0, 0, 2, -1, -1])),
            dict(zip(COLUMNS, [3, "sample", "edge", -1, -1, -1, -1, -1, 1, 3])),
            dict(zip(COLUMNS, [4, "sample", "edge", -1, -1, -1, -1, -1, 2, 3])),
        ]
        errors = validate(self.write_rows(rows))
        self.assertTrue(any("consecutive" in error for error in errors))
        self.assertTrue(any("parents" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
