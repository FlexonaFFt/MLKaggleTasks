# AI Agent Security: strategy report after leaderboard reset

Date: 2026-08-10

Goal: recover above public score 90, then find a credible path above 100.

## Public leaderboard snapshot

Downloaded with the official Kaggle leaderboard API on 2026-08-10
(`ai-agent-security-multi-step-tool-attacks-publicleaderboard-2026-08-10T09:50:44.csv`).

| Rank marker | Score | Team |
|---:|---:|---|
| 1 | 137.130 | Please Hire Me |
| 3 | 114.610 | feng_jack777 |
| 5 | 110.205 | Gerwyn |
| 10 | 107.750 | juliencst |
| 20 | 103.170 | Antoniu Gugu |
| 50 | 94.185 | Saicharan Ramineni |
| 100 | 90.675 | zzx Hee |
| 150 | 89.450 | อธิคม หวังเจริญวงศ์ |
| 300 | 87.750 | Mr RRR |
| 450 | 85.995 | Игорь Жаров |

Threshold counts from 962 teams:

| Threshold | Teams at or above |
|---:|---:|
| 110 | 5 |
| 105 | 16 |
| 100 | 25 |
| 95 | 47 |
| 90 | 129 |
| 86 | 450 |

Read: `85.995` is not close to the competitive front anymore; it is around rank
450 / 962. Crossing 90 gets us into about top 100, but the top 20 already needs
about 103+. This means 90 is a recovery milestone, not the final strategic goal.

## Executive summary

We are not stuck because of one bad parameter. The public game changed after the
leaderboard refresh, and several approaches that looked plausible before the
refresh are now confirmed low-value.

The most important public-discussion signal is official: replay timeouts now
preserve partial score, but timeout inside `attack.py` still kills the run.
That changes the best strategy from "avoid replay timeout at all cost" to
"front-load high-confidence candidates and let risky tails die late if needed".

Our own latest results line up with the public discussions:

| Experiment family | Best recent score | Read |
|---|---:|---|
| Ordered / partial-overfill single-post | 85.995 | Current best post-reset direction. |
| Burst density single-post | 84.690 | Useful but not enough. |
| Blend burst then single tail | 85.635 | Tail/blend can help a little, but not enough yet. |
| Exact duplicate stacking | 58.250 | Closed. Duplicate/cache stack does not solve it. |
| Same-cell jitter stacking | 43.230 | Closed. Jitter made it worse. |
| Naive hop / adaptive target | 49.545 | Closed under current Gemma/tool parsing. |
| Unique-chain hop | SFE | Format/replay unsafe. |

The next serious path is not more naive multi-post. It is:

1. Front-load a proven single-post EXFIL block.
2. Cut per-candidate generation tokens as hard as possible.
3. Put risky innovation only after the safe front block.
4. Use public submissions as controlled experiments, one variable at a time.

My practical probability read:

| Target | Probability with current primitives | Why |
|---|---:|---|
| >90 | Medium | We are only 4.005 points below 90 from `legacy89_partial_overfill`; a faster single-post front or better ordering can plausibly close that gap. |
| >100 | Low without a new primitive | Single-post throughput alone likely needs too many more fired candidates; >100 probably requires a tail that adds multi-fire or another predicate without hurting the front block. |

## What I checked in public discussions

I pulled Kaggle discussion data through Kaggle's public `discussions` API because
the HTML pages are mostly SPA shells. Covered pages:

| Kaggle sort requested | API sort checked | Pages | Notes |
|---|---|---:|---|
| Recent Comments | `active` | 1-3 | High-signal current threads. |
| Hotness | `hot` | 1-3 | Found official/reset and evaluator threads. |
| Recently Posted | `new` | 1-3 | Found current Gemma/debug threads. |
| Most Votes | `top` | 1-3 | Found scoring-model and local-validation threads. |
| Most Comments | derived from `totalMessages` | 61 detailed topics | Ranked locally because no separate API sort was exposed. |

Local evidence files:

- `research/discussion_733058.json`
- `research/discussion_index_pages.json`
- `research/discussion_topic_details.json`
- `research/topic_full/*.json`
- `research/public-code-analysis-2026-08-10.md`

## Source-backed findings

| Source | Signal | Impact on us |
|---|---|---|
| [733058: official evaluator update](https://www.kaggle.com/competitions/ai-agent-security-multi-step-tool-attacks/discussion/733058) | Officially says Gemma parser normalization was added, replay timeout now keeps partial score, old and new scores are not comparable. | Old 89.415 is not a real current baseline. Partial-overfill is now strategically valid. |
| [733058 comments](https://www.kaggle.com/competitions/ai-agent-security-multi-step-tool-attacks/discussion/733058) | Users still report Gemma double-brace tool-call wrapping after the update, especially with multiple repeated calls. | Naive multi-post/hop is still dangerous. Our hop/multi-post failures match this. |
| [723698: throughput mental model](https://www.kaggle.com/competitions/ai-agent-security-multi-step-tool-attacks/discussion/723698) | Public notebook authors frame the score as fired exfil events per replay second. A K=2 multi-post diagnostic fired both calls, but full live-fill scored roughly half a single-hop baseline because generation time doubled. | Multi-post must beat generation-time economics, not just fire in toy tests. |
| [727895: scored surface collapses](https://www.kaggle.com/competitions/ai-agent-security-multi-step-tool-attacks/discussion/727895) | Public writeup argues the reachable high-value public surface mostly reduces to EXFIL throughput. Comment discussion says continuation/multi-post can help only if it amortizes fixed wrap-up cost. | Mainline remains EXFIL throughput; multi-call belongs in a late tail, not the front. |
| [718209: scorer reverse engineering](https://www.kaggle.com/competitions/ai-agent-security-multi-step-tool-attacks/discussion/718209) | Public score is discussed as mean of public rows, clean single-post EXFIL gives about 18 raw, and the hard lever is throughput under replay wall. | Our score gap is mostly event throughput, not classifier magic. |
| [712642: evaluator FAQ](https://www.kaggle.com/competitions/ai-agent-security-multi-step-tool-attacks/discussion/712642) | Reconfirms 9,000s phase budget and 15h global limit; larger candidate counts can surface as SFE/empty output. | SFE is not always CSV format. We should keep `attack.py` fast and let replay partial-credit handle tail risk. |
| [708272: runtime budgets](https://www.kaggle.com/competitions/ai-agent-security-multi-step-tool-attacks/discussion/708272) | Attack generation and replay are separate; algorithms should return early and leave headroom. | Never spend time searching inside `attack.py`; precomputed lists are right. |
| [725073: SFE may be timeout](https://www.kaggle.com/competitions/ai-agent-security-multi-step-tool-attacks/discussion/725073) | Community code reading says timeout/failure before final write can show as Submission Format Error. | Treat SFE as "too risky/too slow" unless local syntax is obviously broken. |
| [708629: public/private env](https://www.kaggle.com/competitions/ai-agent-security-multi-step-tool-attacks/discussion/708629) | Private fixtures/guardrails differ like an eval dataset; overfitting public specifics is risky. | Keep final-selected candidate robust, not only public-harness-specific. |
| [733915: Gemma debug](https://www.kaggle.com/competitions/ai-agent-security-multi-step-tool-attacks/discussion/733915) | User reports Gemma still double-wraps many calls when adding multiple queries. | Confirms our multi-step/multi-post failures are probably structural, not one bad implementation. |

## What is now closed

| Direction | Evidence | Decision |
|---|---|---|
| Exact duplicate cache stacking | `cache01_exact_dup_2000` scored 58.250. | Stop. |
| Same-cell jitter cache stacking | `cache02_samecell_jitter_2000` scored 43.230. | Stop. |
| Naive adaptive hop | `hop01_adaptive_target925` scored 49.545. | Stop. |
| Unique-chain hop | `hop02_unique_chain_143` returned SFE. | Stop until Gemma parser behavior is visibly fixed. |
| Broad multi-post front blocks | Prior `r14_mp*`, `density*`, and `blend*` scores stayed 77-86. | Do not put multi-post in the front block. |
| Replaying old pre-reset 89 exactly | `legacy89_safe987_exact` scored 80.955 after reset. | Old 89 is dead as a direct submit. |
| Local validation as exact score oracle | Gemma artifact/chat-template uncertainty is public, and user-local validation already stalled. | Use it only for format sanity, not for strategy choice. |

## Current best mental model

The public score is mostly:

```text
score ~= fired_exfil_events_per_replay_second * replay_seconds * row_mean_factor
```

For a clean single-post EXFIL event, public discussions repeatedly use the same
rough reward model:

```text
raw ~= 16 severity + 2 novelty = 18
normalized contribution ~= 18 / 200 = 0.09
```

That means:

| Public target | Rough fired-event equivalent |
|---|---:|
| 86 | about 956 clean events |
| 90 | about 1000 clean events |
| 100 | about 1112 clean events |

This is approximate because rows/models differ and Kaggle changed replay
timeout handling. It is still useful because it tells us what kind of gain is
needed:

- To go from 85.995 to 90: about 45 extra clean-event equivalents.
- To go from 85.995 to 100: about 156 extra clean-event equivalents.

So >90 can plausibly come from small throughput/order improvements. >100 likely
needs either a real multi-fire tail or a new predicate family.

## Why our recent "innovations" failed

| Attempt | What it tested | Why the result matters |
|---|---|---|
| `cache01_exact_dup_2000` | Can duplicate identical successful candidates stack cheaply? | No. 58.250 says dedupe/cache/cell effects or refusal dynamics kill it. |
| `cache02_samecell_jitter_2000` | Can tiny same-cell mutations bypass duplicate weakness? | No. 43.230 is worse, so this family is likely poisoning replay. |
| `hop01_adaptive_target925` | Can an adaptive hop/multi-event template get extra events? | No. 49.545 is too low; likely Gemma/parser/generation cost dominates. |
| `hop02_unique_chain_143` | Can unique chained calls avoid duplicate effects? | No. SFE means this implementation is not submission-safe. |

The useful lesson is negative but important: do not chase scoring cache tricks.
The next work must attack replay efficiency and tail optionality.

## Next 5 research submissions

These are ordered by expected value. Each one should be one variable, not a
bundle of guesses.

| Priority | Label | What to submit | What it teaches | Success read |
|---:|---|---|---|---|
| 1 | `front01_micro_single_overfill` | Proven single-post EXFIL only, unique candidates, shortest natural-language/tool-call prompt we can make, overfilled list, high-confidence candidates first. | Whether token-minimized single-post throughput alone can cross 90. | If score beats 85.995 by 2-5 points, keep optimizing this family. If it crosses 90, this is new baseline. |
| 2 | `front02_legacy_front_k2_tail` | First ~950-1050 proven single-post candidates, then K=2 multi-post tail only. | Whether partial replay lets GPT-OSS harvest a late dense tail while Gemma keeps front credit. | If above `front01`, tail contributes. If same/lower, K=2 tail closed. |
| 3 | `front03_legacy_front_k3_tail` | Same safe front, but K=3 tail with brace-free natural language, no JSON-looking syntax. | Tests if K=3 has enough raw gain to offset generation cost when isolated to tail. | If >90 or >front02, multi-fire tail is the >100 path. If low/SFE, close K>2 under current parser. |
| 4 | `front04_deputy_tail_after_exfil` | Safe EXFIL front, then single-call non-EXFIL/confused-deputy style tail candidates. | Tests if another predicate can add score without risking front block. | Any lift over safe front means >100 may require predicate portfolio. No lift means public surface is still EXFIL-only. |
| 5 | `front05_gemma_braceproof_single` | Single-post only, but wording specifically avoids raw braces, semicolons, and repeated command syntax. | Whether Gemma parser failures/latency are costing us even on single-call variants. | If this beats `front01`, Gemma-safe phrasing becomes default. |

If only 4 attempts are available, drop priority 4. Deputy-tail is the least
likely to help public score, but useful for private/generalization.

## Implementation rules for next submissions

Keep the implementation boring:

1. Use only `multi-post-up/attack.py` for these research variants.
2. Do not create a folder per version.
3. Keep `AttackAlgorithm.run()` deterministic and fast.
4. Return a precomputed list, no live model probing inside `attack.py`.
5. Sync notebook `ATTACK_PY_B64` every time.
6. Run `python3 -m py_compile`.
7. Run a smoke check that `AttackAlgorithm().run(None, config)` returns a non-empty list of `AttackCandidate`.

## Decision tree after the next batch

| Result pattern | Interpretation | Next move |
|---|---|---|
| `front01` crosses 90 | We solved the immediate problem with single-post throughput. | Use it as baseline; spend remaining attempts on tail >100. |
| `front01` improves but stays 87-89 | Single-post efficiency still has headroom. | Iterate prompt length/order/candidate count, not multi-hop. |
| `front02` or `front03` beats `front01` | Tail multi-fire is real when isolated after safe front. | Move toward >100 with more tail density and better K ordering. |
| `front02/front03` low or SFE | Multi-post remains structurally bad under current Gemma/parser. | Stop multi-post entirely until organizer confirms parser fix. |
| `front04` beats safe front | Another predicate family is alive. | Build a portfolio tail, then test private-safe variants. |
| All variants stay 83-86 | Public ceiling for current primitive is real. | Need a genuinely new one-call predicate or external public notebook idea; stop burning attempts on parameter sweeps. |

## Bottom line

The best chance to cross 90 is not another full-list multi-post experiment. It
is a high-confidence single-post front block with aggressive but ordered
overfill, plus one or two tail-only experiments that cannot poison the first
~1000 candidates.

The best chance to cross 100 is tail optionality: either K=2/K=3 multi-fire
after a safe front, or a second single-call predicate family. If those do not
move the score, the honest conclusion is that our current primitive has a
public ceiling around the mid/high 80s and we need a new attack surface rather
than more tuning.
