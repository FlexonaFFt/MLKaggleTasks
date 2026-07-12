# EXP-005 — bounded beam v1

- Kaggle kernel: `flexonafft/working-note-try-7-100`
- Submission ref: `54573450`
- Public score: `7.110`
- Reported rank after scoring: `235`
- Previous anchor: `7.148`
- Absolute RMSE gain: `0.038`
- Relative RMSE gain: `0.532%`
- Runtime: `719.3 s`
- Notebook source SHA-256: `db1a3d85a8e49b88d55e6a263ba636372329d9af00b12018b408217ba1a6b013`
- Visible submission SHA-256: `fdf4a8175b6ec6a70c9b78fd6916ac3c317e43f7e9c08bbca87cd02314801ca9`

Visible three overlap wells were skipped by bounded beam and remained fully contact-overridden. Score gain therefore came from hidden rerun behavior. Beam v1 becomes new anchor.

Next one-factor candidates:

1. Weight `0.20`, same gates and clip.
2. Weight `0.10`, same gates and clip.
3. Weight `0.15`, tighter prefix-backtest RMSE limit `6.0`.

Submit one candidate at a time. Do not combine changes before attribution.
