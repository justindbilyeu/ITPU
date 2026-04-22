# ITPU Estimator Guide

## What This Document Is

A decision guide for choosing the right estimator and surrogate type for your use case. Read this before writing code that calls `ITPU.mutual_info()` or `surrogate_test()`.

---

## Estimator Selection

### Histogram MI (`method="hist"`)

**Use when:**
- Data is approximately stationary
- Speed matters more than precision
- n is large relative to bins (see bias rule below)
- You need a fast sanity check before committing to KSG

**Bias rule — required reading:**

Histogram MI has positive bias proportional to `bins² / n`. At `bins=64, n=5000` the bias is approximately 0.4 nats — large enough to dominate a real signal. The safe operating condition is `bins² / n < 0.1`. At `bins=16, n=10000` the bias is approximately 0.011 nats — acceptable for most applications.

Check before using:

```python
assert bins**2 / n < 0.1, f"Histogram bias may dominate: bins²/n = {bins**2/n:.3f}"
```

**Do not use when:**
- Data is high-dimensional (histogram MI is defined for 1D marginals only in current implementation)
- You need publication-ready MI estimates without bias correction
- n < 1000

---

### KSG MI (`method="ksg"`)

**Use when:**
- Data distribution is unknown or non-Gaussian
- You need a non-parametric estimate
- n ≥ 1000 (reliable behavior)
- Dimensionality is low to moderate (d ≤ 4 recommended)

**Metric:** Chebyshev (L∞) in joint space, Euclidean in marginals. This is correct per KSG variant I. Do not interpret metric warnings as bugs.

**The k parameter:**

Current implementation uses fixed k=5. This is a reasonable default for n=1000–50000 in d=2. Known limitations:
- In high dimensions, fixed k leads to density collapse — marginal neighbor counts drop to zero silently (now warned, not silent)
- Adaptive k-selection is on the roadmap (Phase 2) — not yet implemented

**High-dimensional degradation heuristic:**

A warning fires when `N < 10^d`. Take this seriously. At d=2, you need at least 100 samples for reliable KSG. At d=4, at least 10,000. Beyond d=4, validate carefully against known ground truth before trusting results.

**Do not use when:**
- n < 200 (too few samples for reliable k-NN statistics)
- You see warnings about zero marginal neighbors and cannot increase n
- d > 6 without explicit validation

---

## Surrogate Type Selection

Once you have an MI estimate, `surrogate_test()` tells you whether it's distinguishable from noise. The surrogate type determines what "noise" means.

### Decision Tree

```
Is your data a time series?
├── No → use "shuffle"
└── Yes → Does it have autocorrelation structure?
    ├── No (i.i.d. samples) → use "shuffle"
    └── Yes → Is it stationary?
        ├── No (non-stationary) → use "block"
        └── Yes (stationary, autocorrelated) → use "iaaft"
```

---

### Shuffle (`surrogate_type="shuffle"`)

**What it destroys:** All temporal and statistical structure. Each surrogate is an independent random permutation of y.

**Valid for:** i.i.d. data. Any data where temporal order is meaningless.

**Not valid for:** Time series with autocorrelation. Shuffle destroys autocorrelation, which inflates the apparent MI signal — your observed MI will look more significant than it is because the null distribution is too low.

**Calibration status:** Validated. KS test against Uniform(0,1): statistic=0.0565, p=0.1497. Threshold locked at p > 0.05.

---

### Block Bootstrap (`surrogate_type="block"`)

**What it preserves:** Local temporal structure within blocks. What it destroys: long-range dependence across blocks.

**Valid for:** Non-stationary time series where local structure matters but global structure does not.

**Block size guidance:** Default is `max(1, len(x)//20)` — 5% of the signal length. For signals with known autocorrelation length τ, set `block_size ≥ 2τ`.

**Not valid for:** Stationary autocorrelated signals where the autocorrelation structure extends beyond the block size. Use IAAFT instead.

---

### IAAFT (`surrogate_type="iaaft"`)

**What it preserves:** Power spectrum (and therefore autocorrelation structure) and amplitude distribution (rank-preserved to within numerical tolerance).

**What it randomizes:** Phase relationships — the temporal ordering of events, not their statistical properties.

**Valid for:** Stationary autocorrelated time series. EEG, LFP, and other neural signals. Any signal where you need to test MI while controlling for autocorrelation.

**Important:** IAAFT preserves autocorrelation by design. Do not expect surrogates to have different lag-1 autocorrelation than the original — they will not, and should not. The null hypothesis being tested is phase independence, not temporal independence.

**Convergence:** Default `n_iterations=100`. For signals with complex spectral structure, increase to 200–500. Convergence can be checked by monitoring the spectral relative error across iterations — not currently exposed in the API but on the roadmap.

**Calibration status:** AR(1) calibration run pending — this is the final gate to close issue #13. Do not use IAAFT for publication results until that run completes.

---

## Known Limitations and Failure Modes

| Issue | Severity | Estimator | Status |
|---|---|---|---|
| Histogram positive bias at high bins/low n | High | hist | Documented — check `bins²/n < 0.1` |
| Fixed k degradation in high dimensions | Medium | ksg | Warned — adaptive k in Phase 2 |
| Short window bias in windowed MI | Medium | both | Documented — use median + null correction |
| No metric callable support | Low | ksg | Architecture ceiling, not immediate |
| IAAFT AR(1) calibration pending | Medium | iaaft | Issue #13 — run before publication use |
| No real domain validation | Medium | all | R1 acceptable — R2 needs partnerships |

---

## The Calibration Principle

Every estimator in this library has been validated against known ground truth before being declared ready. The validation approach:
- H₀ truth: Independent Gaussians → p-values should be uniform under the null
- H₁ truth: Known correlation structures → power should increase with sample size
- Gate: KS test against Uniform(0,1), threshold p > 0.05, set before running, never adjusted after

If you add a new estimator or surrogate type, this validation is required before the issue closes. See `tests/test_surrogate_validation.py` and `CONTRIBUTING.md`.

---

## Quick Reference

| Situation | Estimator | Surrogate |
|---|---|---|
| Fast check, large n, 1D | hist (bins=16, n≥10k) | shuffle |
| Non-parametric, low-d | ksg | shuffle |
| Time series, no autocorrelation | ksg | shuffle |
| Time series, non-stationary | ksg | block |
| EEG/LFP, stationary autocorrelated | ksg | iaaft |
| High-dimensional (d>4) | validate carefully | validate carefully |

---

*ITPU Estimator Guide*
*Last updated: April 2026*
*Calibration results: docs/collaboration.md — Decision Log*
