# AI Agent Security - Research Summary

Last updated: 2026-07-24.

Goal: push public score above 100 while avoiding replay timeout / blank submissions.

Current confirmed best: about `89` from `/Users/flexonafft/Downloads/89.py`.

Best recent valid research branch: `--r3-selected-seed-pilkplain` with `82.080`, but its R4 same-logic control blanked. Treat it as useful evidence, not a stable default.

Current code state: `attack.py` is `89.py` with `PROBE_REPS = 2` for `--r5-89-probe2`.

## Working Mental Model

This competition is mostly a replay-throughput problem.

- A clean single-post EXFIL candidate scores about `18 raw = 16 severity + 2 novelty`.
- Normalized contribution is about `18 / 200 = 0.09` public score per fired candidate.
- `87.705` is roughly `974` fired-candidate equivalents.
- Score `100` needs roughly `1112` fired-candidate equivalents.
- The gap is about `+14%` throughput.

The replay wall is the main constraint. Too many candidates, too slow candidates, or underestimated replay cost can blank the whole submission instead of giving partial credit. Kaggle API may show these runs as `complete` with no public score; the UI shows `Submission Format Error`.

## Active Files

- `agressive-score-up/attack.py`
- `agressive-score-up/ultra-optimized-agressive.ipynb`

The notebook embeds `attack.py` as base64 in `ATTACK_PY_B64`. Always update the notebook after editing `attack.py`.

Before every Kaggle submission:

1. Edit `agressive-score-up/attack.py`.
2. Regenerate `ATTACK_PY_B64` in `agressive-score-up/ultra-optimized-agressive.ipynb`.
3. Run `python3 -m py_compile agressive-score-up/attack.py`.
4. Verify notebook base64 decodes byte-for-byte to `attack.py`.
5. Dry-run `AttackAlgorithm().run(None, config)` and check 5 one-message candidates.
6. Use a descriptive submission label.

## Current Recovery Baseline

Use this as the next starting point unless a new public notebook gives a stronger reason:

- single-message candidates only
- single-post EXFIL with `SECRET_MARKER` in `http.post` data
- short `http://aa.co` style hosts
- live template probing
- `PROBE_REPS = 3`
- `REPLAY_SAFE = 0.99`
- start from `/Users/flexonafft/Downloads/89.py` or a carefully reset clean branch
- old binary `_effective_cost` selector, not `raw/sec`
- old all-selected-latency `fill_unit`, not fired-only fill pacing

Do not rebuild from failed R4 branches.

## Submission History

| Date | Label | Public score | Result | Notes |
|---|---|---:|---|---|
| 2026-07-15 | `--try attack on 256` | 23.040 | valid | early run |
| 2026-07-15 | `ok --let's try` | 8.640 | valid | early run |
| 2026-07-15 | `ok another try` | 60.255 | valid | early run |
| 2026-07-16 | `ok --let's try with public params` | 76.230 | valid | older baseline |
| 2026-07-17 | `--update params --to small minimize agress` | - | SFE/blank | no public score |
| 2026-07-18 | suppress `OK` | 72.990 | valid | worse |
| 2026-07-18 | shortened suppress + removed template | 71.865 | valid | worse |
| 2026-07-19 | unknown variant | 75.375 | valid | not a lead |
| 2026-07-20 | baseline repeat | 80.190 | valid | run noise / repeat |
| 2026-07-20 | baseline repeat | 78.435 | valid | run noise / repeat |
| 2026-07-20 | `hopdiag` | 83.610 | valid | behavior-identical diagnostic; confirms noise |
| 2026-07-20 | `adaptiveprobe` | - | SFE/blank | likely replay wall |
| 2026-07-20 | `percentile` | 74.925 | valid | over-conservative |
| 2026-07-21 | `--initial score version` | - | SFE/blank | same-day control failed at replay wall |
| 2026-07-21 | `--r2-selected-seed` | 81.495 | valid | best valid R2 |
| 2026-07-21 | `--r2-rawsec` | 73.665 | valid | bad; close direction |
| 2026-07-21 | `--r2-pilkwang-plain` | 80.505 | valid | confounded by `rawsec`; not clean Pilkwang test |
| 2026-07-21 | `--r2-fired-fillunit` | - | SFE/blank | close direction |
| 2026-07-22 | `--r3-selected-seed-control` | - | SFE/blank | old plain at `0.99` still risky |
| 2026-07-22 | `--r3-selected-seed-safe985` | - | SFE/blank | `0.985` did not save old plain |
| 2026-07-22 | `--r3-selected-seed-pilkplain` | 82.080 | valid | best recent branch |
| 2026-07-22 | `--r3-pilkplain-safe985` | 79.380 | valid | valid but lower |
| 2026-07-22 | `--r3-probe3-safe985` | - | SFE/blank | close direction |
| 2026-07-23 | `--r4-control-selected-pilkplain` | - | SFE/blank | same-day control failed; `selected-seed + pilkplain` is unstable at this cap |
| 2026-07-23 | `--r4-hop1-coef180` | 75.285 | valid | 1-hop passed but scored poorly |
| 2026-07-23 | `--r4-hop1-coef143` | - | SFE/blank | aggressive 1-hop accounting failed |
| 2026-07-23 | `--r4-extra-templates-only` | 76.995 | valid | extra templates hurt |
| 2026-07-23 | `--r4-hop1-coef160` | - | SFE/blank | middle 1-hop accounting failed |
| 2026-07-24 | external / downloaded `89.py` | ~89 | valid | same five templates as `87.705`, main behavior change is `PROBE_REPS=3`; new baseline |

## What Has Worked

| Idea | Evidence | Keep? |
|---|---|---|
| Single-post EXFIL with `SECRET_MARKER` in `data` | all strong local scores | yes |
| One user message per candidate | avoids known format/0-byte risks | yes |
| Short `.co` URLs | baseline and Pilkwang branches | yes |
| Live template probing | adapts across replay models | yes |
| Replay-safe sizing by measured latency | necessary for non-blank runs | yes |
| `PROBE_REPS = 3` in the exact uniform branch | `89.py` improved over `87.705` by reducing probe overhead | yes |
| `PROBE_REPS = 5` | still useful as older stability control | control only |
| `REPLAY_SAFE = 0.99` | best risk/reward when it lands | yes, but risky |
| Selected-arm seed only | `81.495` R2 and part of `82.080` R3 | yes |
| Pilkwang-style natural language `plain` | clean combo scored `82.080`; R4 control blanked | maybe, but not a stable default |

## Closed Or Low-EV Directions

| Direction | Result / reason |
|---|---|
| `raw/sec` selector | `73.665`; likely overcounts noisy multi-EXFIL traces or rewards slow candidates. |
| Fired-only `fill_unit` | SFE/blank; likely underestimates replay cost. |
| Mixed `PROBE_REPS = 3` branches | SFE/blank happened when mixed with selected-bank / safety changes; exact `89.py` is not closed. |
| `REPLAY_SAFE = 0.985` as generic fix | did not save old plain; valid Pilkwang-safe branch scored lower. |
| 1-hop search/fill with replay coefficients | `1.80` valid but only `75.285`; `1.43` and `1.60` blanked. Close direction. |
| Extra Pilkwang templates: `call_syntax`, `inj_empty`, `inj_done` | `76.995`; valid but worse than control-quality branches. Close direction. |
| Short suppress text: `OK`, `Routine call; no analysis.` | scored worse: `72.990` and `71.865`. |
| Removing useful templates too early | lost fallback diversity. |
| `90th percentile fill_unit` | scored `74.925`, likely over-conservative. |
| Adaptive early-accept probe | blank / no score, likely filled too aggressively. |
| Multi-message candidates | format/0-byte risk. Do not use. |
| Batch/multi-post fan-out | likely burns hops and replay time; no evidence it beats single-post. |
| `read secret.txt -> post` | public guardrail blocks and it is slower. |
| UTA/destructive attacks | public guardrail arithmetic makes them structurally bad. |
| Obfuscation/exotic payloads | no evidence of better throughput than simple post. |

## Current Kaggle Watchlist

Fresh public notebooks checked on 2026-07-23:

- `pilkwang/ai-agent-v3-1-2-single-post-exfiltration`
- `kaiwalyaatulraut/ai-agent-security-competition-solution`
- `kaiwalyaatulraut/ai-agent-security-solution`
- `tetsutani/ai-agent-sec-adaptive-uniform-three-probe-race`

Notes:

- Pilkwang is still the most relevant source because our best R3 branch confirms selected-seed + Pilkwang plain are compatible.
- Tetsutani's three-probe idea did not transfer cleanly to our branch (`--r3-probe3-safe985` blanked), so it should not be copied blindly.
- Kaiwalya's solution decodes to the same `uniform_three_probe` attack as Tetsutani: `PROBE_REPS=3`, old plain, all-bank seed, no selected-bank, no Pilkwang plain, no 1-hop. This exact branch has not been cleanly tested in our submissions.
- Any new public code needs source-level diffing, not just copying parameters.

## Next Useful Directions

Best practical next steps:

1. Reset `attack.py` away from `--r4-hop1-coef160`.
2. Rebuild from `/Users/flexonafft/Downloads/89.py` or another confirmed stable baseline.
3. Stop spending attempts on Pilkwang extra templates, 1-hop accounting, raw/sec, fired-only fill, and mixed three-probe variants.
4. Repeat exact `89.py` before trying new hybrids; run noise is large enough that repeats are useful.
5. Look for a genuinely new public mechanism or a safer replay-cap strategy before more speculative submissions.

Today's planned R5 attempts:

| Order | Label | Source | Change | Purpose |
|---:|---|---|---|---|
| 1 | `--r5-89-exact-a` | `/Users/flexonafft/Downloads/89.py` | none | confirm / repeat new best baseline |
| 2 | `--r5-89-exact-b` | `/Users/flexonafft/Downloads/89.py` | none | exploit run variance on strongest branch |
| 3 | `--r5-89-probe2` | `89.py` | `PROBE_REPS=2` | one controlled upside test: less probe overhead, noisier selector |
| 4 | `--r5-89-safe985` | `89.py` | `REPLAY_SAFE=0.985` | safer replay cap, likely slightly lower score |
| 5 | `--r5-087705-control` | `/Users/flexonafft/Downloads/87.705.py` | none | old strong-control fallback |

Reasonable experiment candidates:

- Conservative valid-score branch from the original `87.705.py`, not from the failed R4 code.
- Selector priority branch: keep binary `_effective_cost`, but add explicit deterministic preference for historically strong templates when costs tie.
- Candidate-text micro-branch: compare only `plain` wording while keeping all replay logic identical.
- Exact-EXFIL counting branch: count only traces that include the selected URL and marker; this could reduce false optimism, but may also shrink throughput.

What we may be missing:

- A genuinely faster high-fire template, not another replay cap tweak.
- A validated multi-hit trace that still fits replay time. We do not have evidence for this yet.
- Public notebook changes after 2026-07-23 that introduce a new attack surface or a lower-latency payload.

## Strategic Read

The recent experiments are useful because they closed several tempting branches. We now know the biggest mistake would be continuing to tune around `rawsec`, fired-only pacing, three-probe probing, 1-hop replay coefficients, or extra Pilkwang templates.

To reach 100+, the current single-post approach needs about 14% more effective throughput. Small safety and wording changes are unlikely to create that alone. The next real upside must come from either a lower-latency prompt/tool-call pattern or a new public-code mechanism that increases fired candidates without increasing replay cost.
