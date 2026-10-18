'''Launch a private Kaggle reproduction once a batch GPU slot is available.'''

import argparse
import json
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


class ReproductionLauncher:
    def __init__(self):
        self.root = Path(__file__).resolve().parents[1]
        self.api = KaggleApi()
        self.api.authenticate()

    def launch(self, mode):
        directory = self.root / 'kaggle' / mode
        response = self.api.kernels_push(str(directory), timeout='32400', acc='NvidiaTeslaT4')
        result = json.loads(str(response))
        (directory / 'launch.json').write_text(json.dumps(result, indent=2) + '\n')
        if result.get('error'):
            raise RuntimeError(result['error'])
        if not result.get('url') or not result.get('versionNumber'):
            raise RuntimeError(f'Kaggle did not confirm a notebook version: {result}')
        print(json.dumps(result, indent=2))
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['reference', 'full'])
    args = parser.parse_args()
    ReproductionLauncher().launch(args.mode)
