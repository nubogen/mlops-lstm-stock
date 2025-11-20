# Experiment: Phase C — Balanced Feature Set

**Branch**: `experiment/phase-c-balanced-set`  
**Purpose**: Add and validate the balanced 8–12 feature set for LSTM training (price + trend + volatility + volume signals).

## Goal
- Implement the balanced feature set.
- Run a short training job (smoke test) to confirm pipeline stability.
- Log runs to MLflow and compare baseline vs new features.

## Features included (initial):
1. log_ret (log return)
2. ret_lag1 (lag-1 return)
3. sma5 (short SMA)
4. sma20 (long SMA)
5. dist_sma20 (close - sma20)
6. atr14 (ATR with window 14)
7. obv (On-Balance Volume)
8. vol_z (rolling z-score of volume)
9. money_flow (log_ret * volume) — optional
10. ret_std_30 (rolling std of returns) — optional

## Minimal experiment steps
1. Implement `add_balanced_features(df)` in `src/data/preprocess.py`.
2. Run a quick training: small epochs (1–3) and small batch size to smoke test.
3. Commit and push to this branch.
4. Review MLflow run and compare to baseline.

## Notes
- Always shift rolling features by 1 to avoid leakage.
- Fit scalers on training set only.
- Keep `mlruns/`, `data/`, `models/` ignored.
