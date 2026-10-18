# Name That LLM

## Current experiments

| Notebook | Method | Measured public score |
|---|---|---|
| `01_public_ensemble_oop.ipynb` | Lightweight word/character TF-IDF baseline; the legacy filename does not describe a neural ensemble reproduction | 0.316, user reported |
| `02_soft_claude_refinement.ipynb` | TF-IDF plus a Claude-only specialist | 0.318, user reported |
| `03_public_reference_oop.ipynb` | Frozen centered-stylegram reference ensemble: Qwen LoRAs, frozen model features, attention heads, UltraBERT, centering, balancing, Claude experts | Not measured |
| `04_public_0557_reproduction_oop.ipynb` | Same reference plus the supplied SmolLM3 centered extension and protected TabPFN residual | Not measured |

Attempts 03 and 04 have independent runtime directories and the same final output name, `submission.parquet`, in their separate Kaggle sessions. They require **GPU T4 x2**, internet disabled, and all inputs in the corresponding `kaggle/*/kernel-metadata.json`. No TF-IDF fallback replaces a failed neural stage.

The new notebooks use classes for configuration, source management, resource checks, execution, and output validation. Scientific modules are preserved as upstream dependencies with exact SHA-256 checks. Their numerical code has **not** been completely rewritten into class methods. This is an explicit reproducibility choice; the fully refactored implementation is still outstanding.

## Sources and credits

The frozen scientific source comes from the user-provided `0-557-lora-smollm3-tabpfn.ipynb`, not the latest live version. The upstream live notebook has subsequently changed.

- [Vladislav Lavrentev: supplied SmolLM3/TabPFN notebook](https://www.kaggle.com/code/lightningv08/0-557-lora-smollm3-tabpfn): original Qwen LoRA contribution, SmolLM3 fine-tuning, and tabular extension.
- [Obstacle Developer: reference source, script version 359313348](https://www.kaggle.com/code/obstacledeveloper/local-candidate-adaptive-centered-stylegram?scriptVersionId=359313348): adaptive centered stylegram.
- [Lucas Boesen: base ensemble](https://www.kaggle.com/code/lucasboesen/lora-x-adaptive-centering): frozen features, blend, adaptive centering, LoRA and attention heads.
- [Zbigniew Kirchner: Claude experts](https://www.kaggle.com/code/zbigniewkirchner/0-552-adaptive-centering-own-lora-claude-gat).
- seyitkaangunes: public Claude API passages; pjmathematician: UltraBERT fold weights; kirilltym: learned token-attention recipe. Other upstream component credits and notices remain in the archived source.

`sources/public_0557/origin.json` records the supplied notebook hash, original description, source hashes, and Kaggle input version IDs. The 24 original payload files are archived beside it, including license notices. SmolLM3 uses Apache 2.0; TabPFN weights retain the Prior Labs Non-Commercial License; other components retain their licenses. This repository does not relicense those assets.

## Launch

From the repository root, using the existing authenticated Kaggle environment:

```bash
.venv/bin/python competitions/public-comp/nameThatLLM/scripts/launch_reproductions.py reference
.venv/bin/python competitions/public-comp/nameThatLLM/scripts/launch_reproductions.py full
```

Both launches create **private** notebooks. They do not automatically submit to the competition or publish publicly. Use only a successfully completed notebook version that contains the validated `submission.parquet` for a competition submission. The launch script stops on a quota or API error; it does not silently retry.

The pinned Docker image and model/resource references are configured in `kaggle/reference/kernel-metadata.json` and `kaggle/full/kernel-metadata.json`. The Smol inference dataset is pinned to version 1. Other resource slugs come from upstream API metadata; runtime checks catch missing assets and hash/version mismatches. Full remote inference is still required to verify compatibility.

## Validation and launch status

- Both new notebooks compile and their source bundles materialize byte-for-byte identically to the 24 supplied files, including mixed line endings.
- Output schema and ID alignment checks reject invalid submissions; partial files are quarantined after failed inference.
- The local machine is correctly rejected as missing the required Kaggle GPUs and model assets. No local neural inference score is claimed.
- On 2026-10-18 both launch requests were rejected with `Maximum batch GPU session count of 2 reached.` Neither new notebook version was launched, and no competition submission was made. Responses are saved as `kaggle/*/launch.json`.
- The supplied notebook recorded about 56 minutes for its small interactive test pool. Full hidden-test runtime and our scores have not been measured.

## Rebuild from the supplied source

```bash
.venv/bin/python competitions/public-comp/nameThatLLM/scripts/build_reproductions.py \
  /Users/flexonafft/Downloads/0-557-lora-smollm3-tabpfn.ipynb \
  competitions/public-comp/nameThatLLM/sources/public_0557/resource-metadata.json
```

The notebooks are self-contained after generation: runtime execution does not depend on local repository files. Resource metadata is archived with the scientific source; the original downloaded notebook is needed only to rebuild the files.
