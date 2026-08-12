# AI Agent Security: path to 95-100 after failed suppression batch

Date: 2026-08-12

Goal: identify whether there is a credible path from our current post-reset
ceiling (`85.995`) toward `95-100`, using public discussions, public notebooks,
leaderboard shape, and our failed submissions.

## Executive Summary

- **The current EXFIL-throughput family is probably capped below `95`.** Our
  recent batches tested short prompts, multi-post tails, mechanical wording,
  direct output suppression, and frame-only variants. None beat `85.995`; many
  collapsed to `47-72`. More wording/cap sweeps are low expected value.
- **Public high-score notebooks do not expose a `100+` recipe.** The strongest
  public code line is still live validation-fill with single `http.post` EXFIL,
  latency split, frame only on the slow row, and replay-safe sizing. Visible
  public versions mostly explain `85-90`, not `95-100`.
- **The important public-code lesson is fire efficiency, not prompt length.**
  Dimong/Foysal/Canqiang lineage explicitly says the `47 -> 60` jump came from
  keeping only candidates that fired and sizing to each model's real speed, not
  from payload, candidate cap, or template speed.
- **A real `95-100` path likely needs a new scored surface or a better slow-row
  breakthrough.** To reach `100`, the current `18 raw` single-post primitive
  needs roughly `1112` clean fired-event equivalents. Our best line is around
  `956`; the missing `150+` clean events are too many for small template edits.

## Current Position

Leaderboard snapshot from the saved Kaggle CSV on 2026-08-10:

| Threshold | Teams at or above |
|---:|---:|
| 130 | 1 |
| 120 | 2 |
| 110 | 5 |
| 105 | 16 |
| 100 | 25 |
| 95 | 47 |
| 90 | 129 |
| 86 | 450 |

Rank markers:

| Rank | Score | Team |
|---:|---:|---|
| 1 | 137.130 | Please Hire Me |
| 10 | 107.750 | juliencst |
| 20 | 103.170 | Antoniu Gugu |
| 50 | 94.185 | Saicharan Ramineni |
| 100 | 90.675 | zzx Hee |
| 450 | 85.995 | Игорь Жаров |

Interpretation: `90` is still a recovery milestone. `95-100` is a different
regime; top teams are not just one safe-cap tweak ahead.

## What Our Last Two Batches Proved

### Short/mechanical/multi-post batch

| Version | Score | Read |
|---|---:|---|
| `front01_micro_single_overfill` | 80.910 | Shorter prompt did not help. |
| `front02_legacy_front_k2_tail` | 56.060 | K=2 tail is bad. |
| `front03_legacy_front_k3_tail` | 60.610 | K=3 tail is also bad. |
| `front05_gemma_braceproof_single` | 58.275 | Braceproof wording is bad. |
| `code02_pilkwang_mechanical_probe` | 56.970 | Mechanical public-note wording did not transfer. |

### Output suppression batch

| Version | Score | Read |
|---|---:|---|
| `sup01_legacy_no_answer` | 49.590 | Removing `Then answer OK only` breaks reliability. |
| `sup02_legacy_stop_after_tool` | 47.205 | Explicit `stop` is worse. |
| `sup03_legacy_no_final_text` | 52.650 | "No final response" is not a useful suppressor. |
| `sup04_frame_only` | 71.550 | Frame is alive, but not as a universal template. |
| `sup05_frame_after_safe_front` | 55.720 | Frame tail did not add enough; safe front did most of the work. |

Decision: stop spending attempts on prompt shortening, direct suppressors,
frame-only, and K2/K3 tails.

## Public Sources And Signals

| Source | Signal | Impact |
|---|---|---|
| [Kaggle evaluator update 733058](https://www.kaggle.com/competitions/ai-agent-security-multi-step-tool-attacks/discussion/733058) | Parser/replay behavior changed; replay timeout now keeps partial score, but `attack.py` timeout still kills. | Overfill can be valid, but generation must return quickly. Old scores are not comparable. |
| [Dimong public notebook](https://www.kaggle.com/code/dimong4/ai-agent-security) | Public code says the `47 -> 60` jump came from live validation-fill: keep only candidates that fired, and size to each model's measured speed. It also uses `FRAME_TEMPLATE`, `REPLAY_SAFE_FRAC=0.97`, latency split, and the reliable verbose `http.post` wording. | Good explanation of `60 -> high 80s` lineage, not a direct `100+` recipe. |
| [Foysal v15](https://www.kaggle.com/code/foysalemonshanto/ai-agent-security-v15) and [Canqiang 0721](https://www.kaggle.com/code/canqiang/aiagsec-ea-b-0721) | Same practical family: live validation-fill, latency split, frame on slow path, replay-safe sizing. Canqiang note says `N=600` runtime-exceeded in one run. | Confirms the mechanism is fragile and model-speed-bound. |
| [Pilkwang working note](https://www.kaggle.com/code/pilkwang/ai-agent-working-note) | Public note frames reasoning-suppression around a `52.87` base and says `>56` would be meaningful; it is not presented as the team's exact `100+` solution. | Mechanical wording is an idea source only. Our `code02` result closes the direct transfer. |
| [Anvith AISec Pilk](https://www.kaggle.com/code/anvithpothula/aisec-pilk) | Raw/sec selector, selected-template seed, and measured replay-cost sizing. User-reported public notebook score after refresh is `85.410`, despite team API score being higher. | Useful selector discipline, not proof of `90+`. |
| [Tetsutani two-probe recovery](https://www.kaggle.com/code/tetsutani/ai-agent-sec-adaptive-uniform-two-probe-recovery) | Same attack bytes reportedly produced hosted scores around `88.5-89.1`; timeouts are treated as negative controls. | Reinforces run variance and timeout fragility around high 80s. |
| [Yusuke Tail8 notebook](https://www.kaggle.com/code/yusuketogashi/lb60-525-july-safe-edge-prune-tail8-upgrade) | Its own notes say short URL hurt; keeps verbose OK-only fallback, no K2 preseed, no bare prompt, no static bank. Tail8 moved a `60.125` line only slightly. | Public tail ideas are small edge tuning, not `95-100`. |

## Why Public Code Does Not Explain 100+

The exposed public notebooks mostly cluster around one known recipe:

1. single-message candidates;
2. one intended `http.post`;
3. `SECRET_MARKER` in `data`;
4. live validation-fill;
5. keep only fired candidates;
6. latency split between model rows;
7. `FRAME_TEMPLATE` only for the slow/reasoning path;
8. replay-safe cap around `0.97`;
9. verbose `Then answer OK only` wording.

That recipe explains the move from weak static lists to `80-89`. It does not
explain scores above `100`, because our own best already includes much of this
family and still sits at `85.995` post-reset.

The likely gap is not "one better sentence". It is one of:

| Hypothesis | Why plausible | Evidence gap |
|---|---|---|
| New scored predicate | `EXFIL` gives about `18 raw` per clean event; `100+` may require more raw per replay second. | We have not systematically mapped all predicates in the updated scorer. |
| Slow-row breakthrough | Public code says fast row can run wide, while slow `gpt_oss` row drags the mean. | Frame-only and direct suppressors failed; maybe the winning method is a different slow-row frame/turn structure. |
| Better model-specific branching | Top teams may separate Gemma/GPT-OSS more accurately than our median latency split. | We lack replay logs for leader solutions and cannot infer exact branching from public code. |
| Hidden private notebook delta | Public notebooks may be old, redacted, or intentionally not exact. | Visible public code often has scores far below the team score. |

## What Is Closed

Stop spending submissions on:

- shorter input prompt;
- removing `OK only`;
- direct `stop` / `no final response`;
- frame-only for every candidate;
- frame tail after static safe front;
- K2/K3 multi-post tail;
- Pilkwang mechanical wording as a direct clone;
- duplicate/cache stacking;
- raw public notebook clones without source-level reason.

## What Still Has A Credible Path

### 1. Restore the working dynamic baseline

This is not a `100+` plan, but it is required hygiene. Our current `multi-post-up`
file is a failed diagnostic branch. The only live baseline worth using is the
dynamic validated-fill family around `legacy89_partial_overfill` / old r14-v38:

- live validation;
- fired candidate keep;
- `SPLIT_CLASSIFY_N=5`;
- median latency split;
- `FRAME_TEMPLATE` only after slow-row classification;
- warmup/probe candidates kept when fired;
- dynamic replay cap.

Any future attempt should diff against that, not against the latest failed
suppression file.

### 2. Build a predicate map before spending more submits

For `95-100`, the highest expected-value research is not another Kaggle submit.
It is a source-level scorer audit:

1. inspect `aicomp_sdk` / `kaggle_evaluation` scorer and guardrail code;
2. list every predicate and raw weight;
3. classify each as reachable, blocked, slow, or untested;
4. identify whether any single-call predicate beats `18 raw` per replay-cost unit;
5. only then design a submit.

This is the first genuinely new route. If every reachable predicate collapses
back to EXFIL-only, `100+` probably requires a hidden model-specific replay trick.

### 3. Revisit slow-row classification, not wording

If scorer audit finds no new predicate, the next best target is the slow row:

- compare current median classifier with stronger model fingerprinting;
- test whether selected slow-row frame variants can be validated live before use;
- never use frame blindly for Gemma;
- keep the verbose OK-only template for fast rows.

This is still more likely to recover `90` than reach `100`.

## Recommended Next Work

Do not submit immediately. Spend one work session on:

| Priority | Work item | Output |
|---:|---|---|
| 1 | Restore exact best dynamic baseline locally | `attack.py` + notebook matching `85.995` family |
| 2 | Audit scorer predicates in the competition SDK | Markdown table of predicates, weights, guardrails, reachability |
| 3 | Compare public high-score notebook code against our baseline | Small diff table, not copied code |
| 4 | Pick at most 2 submit candidates | One baseline restoration, one genuinely new predicate/slow-row test |

## Bottom Line

The honest read is: `90` may still be reachable by restoring and slightly tuning
the validated dynamic baseline. `95-100` is unlikely through the current
`http.post SECRET_MARKER` wording/cap/tail family. The next credible `100+`
research path is a scorer/predicate audit, not more prompt engineering.

