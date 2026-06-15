"""
26× speedup re-audit for the R2 FPGA targeting decision.

This script audits the original claim that KSG is 26× slower than histogram MI.
The audit controls for **accuracy** — comparing at matched bias, not arbitrary bins.

Pre-registered claim: "The 26× gap is an artifact of unmatched accuracy."

Decision criteria (§7):
  R2 premise STANDS     if speedup ≥ 10× at matched accuracy, non-overlapping 95% CIs.
  R2 premise FALSIFIED  if speedup <  10× or CIs overlap.

Both outcomes are load-bearing — report honestly. Note the asymmetry:
if histogram cannot reach KSG accuracy at any bin count (likely for continuous
data), that STRENGTHENS the case for accelerating KSG, not weakens it.

Usage:
    python validation/ksg/bench_audit.py

Output:
    Prints the verdict table + writes bench_audit_<sha>.json to results/
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy

sys.path.insert(0, os.path.dirname(__file__))
import ground_truth as gt
import ksg as ksg_module

try:
    from itpu.sdk import ITPU as _ITPU
    _ITPU_AVAILABLE = True
except ImportError:
    _ITPU_AVAILABLE = False

# ── Parameters ────────────────────────────────────────────────────────────────
N_BENCH = 10_000
RHO = 0.5
K_CORE = 4
S_TIMING = 30        # timing reps (warm cache)
BIN_CANDIDATES = [8, 16, 32, 64, 128, 256, 512]


def _git_sha() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _time_ksg(x: np.ndarray, y: np.ndarray, k: int, S: int) -> np.ndarray:
    """Wall-clock S timing reps for ksg_mi."""
    times = np.empty(S)
    for i in range(S):
        t0 = time.perf_counter()
        ksg_module.ksg_mi(x, y, k=k, jitter_seed=i)
        times[i] = time.perf_counter() - t0
    return times


def _time_hist(x: np.ndarray, y: np.ndarray, bins: int, S: int) -> np.ndarray:
    """Wall-clock S timing reps for histogram MI (via itpu.sdk if available)."""
    times = np.empty(S)
    if _ITPU_AVAILABLE:
        itpu = _ITPU(device="software")
        for i in range(S):
            t0 = time.perf_counter()
            itpu.mutual_info(x, y, method="hist", bins=bins)
            times[i] = time.perf_counter() - t0
    else:
        # Fallback: inline histogram MI
        for i in range(S):
            t0 = time.perf_counter()
            _hist_mi_inline(x, y, bins)
            times[i] = time.perf_counter() - t0
    return times


def _hist_mi_inline(x: np.ndarray, y: np.ndarray, bins: int) -> float:
    hxy, _, _ = np.histogram2d(x, y, bins=bins)
    hx = hxy.sum(axis=1)
    hy = hxy.sum(axis=0)
    def _H(h):
        p = h / h.sum()
        p = p[p > 0]
        return -float(np.sum(p * np.log(p)))
    return _H(hx) + _H(hy) - _H(hxy.ravel())


def _hist_bias(x: np.ndarray, y: np.ndarray, bins: int, I_true: float) -> float:
    if _ITPU_AVAILABLE:
        itpu = _ITPU(device="software")
        mi = float(itpu.mutual_info(x, y, method="hist", bins=bins))
    else:
        mi = _hist_mi_inline(x, y, bins)
    return abs(mi - I_true)


def main() -> None:
    sha = _git_sha()
    print("=" * 66)
    print(f"26× Speedup Audit  |  N={N_BENCH}  ρ={RHO}  k={K_CORE}  S={S_TIMING}")
    print(f"SHA: {sha}")
    print("=" * 66)

    rng = np.random.default_rng(0x4155_4449)  # "AUDI"
    x, y = gt.generate_bivariate_gaussian(N_BENCH, RHO, rng)
    I_true = gt.gaussian_mi(RHO)

    # ── Step 1: KSG bias at core point ───────────────────────────────────────
    mi_ksg, _ = ksg_module.ksg_mi(x, y, k=K_CORE, jitter_seed=0)
    ksg_bias = abs(mi_ksg - I_true)
    print(f"\nKSG: MI={mi_ksg:.5f} nats  bias={ksg_bias:.5f} nats  (I_true={I_true:.5f})")

    # ── Step 2: Match histogram accuracy ─────────────────────────────────────
    print("\nHistogram bias scan (target: within 20% of KSG bias):")
    print(f"  KSG bias = {ksg_bias:.5f} nats  (target ≤ {ksg_bias * 1.20:.5f})")
    print(f"  {'bins':>6}  {'hist_bias':>12}  {'ratio':>8}  {'match':>6}")

    matched_bins: int | None = None
    for bins in BIN_CANDIDATES:
        h_bias = _hist_bias(x, y, bins, I_true)
        ratio = h_bias / max(ksg_bias, 1e-8)
        match = abs(ratio - 1.0) <= 0.20
        print(f"  {bins:>6}  {h_bias:>12.5f}  {ratio:>8.3f}  {'YES' if match else 'no':>6}")
        if match and matched_bins is None:
            matched_bins = bins

    verdict_regime: str
    if matched_bins is None:
        print(
            "\n  FINDING: Histogram cannot match KSG accuracy at any tested bin count.\n"
            "  This STRENGTHENS the case for accelerating KSG — 'faster' was comparing\n"
            "  fast-but-wrong to slow-but-right. R2 premise stands on accuracy grounds."
        )
        verdict_regime = "hist_cannot_match_ksg_accuracy"
        # Use the closest bin count for timing comparison (informational)
        biases = [(_hist_bias(x, y, b, I_true), b) for b in BIN_CANDIDATES]
        _, matched_bins = min(biases)
    else:
        print(f"\n  Matched at bins={matched_bins}")
        verdict_regime = "accuracy_matched"

    # ── Step 3: Timing comparison ─────────────────────────────────────────────
    print(f"\nTiming at bins={matched_bins} (S={S_TIMING} reps, warm cache):")

    ksg_times = _time_ksg(x, y, K_CORE, S_TIMING)
    hist_times = _time_hist(x, y, matched_bins, S_TIMING)

    ksg_med = float(np.median(ksg_times))
    hist_med = float(np.median(hist_times))
    speedup = ksg_med / hist_med if hist_med > 0 else float("inf")

    # 95% CI via bootstrap
    rng_boot = np.random.default_rng(42)
    speedup_boot = np.empty(2000)
    for b in range(2000):
        k_s = np.median(rng_boot.choice(ksg_times, size=S_TIMING, replace=True))
        h_s = np.median(rng_boot.choice(hist_times, size=S_TIMING, replace=True))
        speedup_boot[b] = k_s / h_s if h_s > 0 else float("inf")
    ci_lo = float(np.percentile(speedup_boot, 2.5))
    ci_hi = float(np.percentile(speedup_boot, 97.5))

    print(f"  KSG  median: {ksg_med * 1000:.2f} ms")
    print(f"  hist median: {hist_med * 1000:.2f} ms")
    print(f"  Speedup:     {speedup:.1f}×  (95% CI [{ci_lo:.1f}×, {ci_hi:.1f}×])")

    # ── Step 4: Verdict ───────────────────────────────────────────────────────
    ci_non_overlapping_10x = ci_lo >= 10.0
    verdict_stands = speedup >= 10.0 and ci_non_overlapping_10x

    if verdict_regime == "hist_cannot_match_ksg_accuracy":
        verdict = "R2_PREMISE_STANDS (accuracy-grounds: hist cannot reach KSG quality)"
    elif verdict_stands:
        verdict = "R2_PREMISE_STANDS (≥10× at matched accuracy, non-overlapping CI)"
    else:
        verdict = (
            "R2_PREMISE_FALSIFIED (speedup < 10× or CIs overlap — "
            "revisit whether KSG is the right acceleration target)"
        )

    print(f"\n  VERDICT: {verdict}")

    # ── Write JSON ────────────────────────────────────────────────────────────
    outdir = Path(__file__).parent / "results"
    outdir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime, timezone
    utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = outdir / f"bench_audit_{sha[:8]}_{utc}.json"

    payload = {
        "git_sha": sha,
        "lib_versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "config": {
            "N": N_BENCH,
            "rho": RHO,
            "k": K_CORE,
            "S_timing": S_TIMING,
        },
        "ksg_bias_nats": ksg_bias,
        "matched_bins": matched_bins,
        "verdict_regime": verdict_regime,
        "ksg_median_s": ksg_med,
        "hist_median_s": hist_med,
        "speedup_median": speedup,
        "speedup_95ci": [ci_lo, ci_hi],
        "verdict": verdict,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nResults → {out}")


if __name__ == "__main__":
    main()
