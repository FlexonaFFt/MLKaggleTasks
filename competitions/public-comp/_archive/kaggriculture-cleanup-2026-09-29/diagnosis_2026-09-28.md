# Kaggriculture submission diagnosis — 2026-09-28

## Verified live identity

| Agent | Submission | Notebook script version | Rating observed | Completed non-self episodes | W/D/L |
|---|---:|---:|---:|---:|---|
| C95 | 56638524 | 353569068 | 781.1 | 34 | 15/0/19 |
| Herd-Safe v3 | 56638501 | 353568962 | 1675.8 | 52 | 27/3/22 |

Source: authenticated Kaggle Competition API, `competition_submissions` and
`competition_list_episodes`. These are a changing sample, not final ratings.
The screenshot labels do not identify the actual agents correctly. Notebook
source and source checksums establish the mapping above.

C95 median final cash was 90,535, with a median opponent margin of -1,006.
Herd-Safe median final cash was 90,749.5, with a median margin of +56.
Cash earned in a game is not the leaderboard rating.

## Runtime findings

The submitted C95 notebook's saved starter smoke test printed [1.0, 2530.0].
The former test checked DONE status only, so that result was not rejected.
Local official kaggle-environments 1.32.7, seed 18590000, instead returns
[156886, 3634] using the same C95 source and file runner. The cause of that
historical smoke discrepancy is unresolved; it is not evidence of a specific
confirmed runtime bug. Actual live episodes show substantial final cash.

Both notebooks now require complete games and a win against starter in both
seats before packaging. Missing engine support fails the check instead of
silently skipping it. Archives are named submission_c95.tar.gz and
submission_herd_safe_v3.tar.gz. Source-agent bytes remain unchanged.

## Rejected experiment

Only `_HP_WINDOW` changed from 4 to 2 in Herd-Safe. Compare paired_screening.json
and paired_holdout.json. There were 32 games total: each version played C95 and
V38 in both seats on four seeds. Both versions won all 16 of their games.
The change improved mean cash margin by 50.75 on screening seeds but reduced
it by 6.25 on held-out seeds. This provides no convincing reason to submit it.
The opponent pool is small and related; it does not predict a 2400 rating.

Reproduce with the workspace .kaggle-venv Python:

```sh
.kaggle-venv/bin/python competitions/public-comp/Kaggriculture/check_candidates.py
.kaggle-venv/bin/python competitions/public-comp/Kaggriculture/check_candidates.py --seeds 280903 280904 --output paired_holdout.json
```

Decision: do not resubmit unchanged C95 as a recovery strategy. Keep Herd-Safe
as a comparison baseline. A replacement needs evidence against recent stronger
opponents; no new high-score strategy has been validated in this audit.
