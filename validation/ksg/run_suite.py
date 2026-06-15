"""
KSG validation suite orchestrator.

Runs all sweeps, writes results/run_<sha>_<utc>.json and per-seed CSVs.

Usage
-----
    # Run everything (slow — ~10 min for full GATE battery):
    python validation/ksg/run_suite.py

    # Run only the T3 pilot to freeze τ_var (then update test_ksg.py):
    python validation/ksg/run_suite.py --pilot-only

    # Use a custom results directory:
    python validation/ksg/run_suite.py --outdir /path/to/results
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import scipy.stats as sp_stats

sys.path.insert(0, os.path.dirname(__file__))
import ground_truth as gt
import ksg as ksg_module
import oracles

# ── Core operating point ─────────────────────────────────────────────────────
N_CORE = 10_000
K_CORE = 4
S_CORE = 100
MASTER_ENTROPY = 0x4954_5055  # "ITPU"
NULL_ENTROPY = 0x4E554C4C    # "NULL"
T6_ENTROPY = 0x5436_5F4F     # "T6_O"


# ── Utilities ─────────────────────────────────────────────────────────────────

def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _lib_versions() -> dict:
    import platform
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def _make_result(name: str, tier: str, stat: float, threshold: float,
                 passed: bool, localization: str) -> dict:
    return {
        "name": name,
        "tier": tier,
        "statistic": round(stat, 9),
        "threshold": threshold,
        "pass": passed,
        "localization": localization,
    }


def _run_seeds(rho: float, N: int, k: int, S: int, entropy: int) -> np.ndarray:
    master = np.random.SeedSequence(entropy=entropy)
    child_seeds = master.spawn(S)
    estimates = np.empty(S)
    for i, cs in enumerate(child_seeds):
        rng = np.random.default_rng(cs)
        x, y = gt.generate_bivariate_gaussian(N, rho, rng)
        mi, _ = ksg_module.ksg_mi(x, y, k=k, jitter_seed=int(cs.entropy & 0xFFFF))
        estimates[i] = mi
    return estimates


def _save_csv(path: Path, headers: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


# ── Individual test runners ───────────────────────────────────────────────────

def run_t7() -> dict:
    """T7 — digamma identities."""
    try:
        oracles.assert_digamma()
        passed = True
        stat = 0.0
        loc = "ψ(1)+γ and ψ(2)−(1−γ) both < 1e-12"
    except AssertionError as e:
        passed = False
        stat = float("nan")
        loc = str(e)
    return _make_result("T7_digamma", "GATE", stat, 1e-12, passed, loc)


def run_t6() -> dict:
    """T6 — brute-force oracle agreement."""
    rng = np.random.default_rng(T6_ENTROPY)
    x, y = gt.generate_bivariate_gaussian(500, 0.6, rng)
    mi_prod, _ = ksg_module.ksg_mi(x, y, k=K_CORE, jitter_seed=42)
    mi_brute = oracles.mi_bruteforce(x, y, k=K_CORE, jitter_seed=42)
    diff = abs(mi_prod - mi_brute)
    passed = diff <= 1e-9
    return _make_result(
        "T6_oracle", "GATE", diff, 1e-9, passed,
        "indexing/metric/counting bug in fast path" if not passed else "exact match"
    )


def run_t5() -> list[dict]:
    """T5 — reparameterization invariance."""
    I_TRUE = gt.GAUSSIAN_TABLE[0.7]
    rng = np.random.default_rng(0x54355F49)
    x, y = gt.generate_bivariate_gaussian(N_CORE, 0.7, rng)
    mi_id, _ = ksg_module.ksg_mi(x, y, k=K_CORE, jitter_seed=7)

    transforms = [
        ("identity",  lambda x, y: (x, y),               True),
        ("scale_10x", lambda x, y: (10 * x, 0.1 * y),    True),
        ("cube_x",    lambda x, y: (x ** 3, y),           True),
        ("sinh_x",    lambda x, y: (np.sinh(x), y),       False),
        ("exp_y",     lambda x, y: (x, np.exp(y)),        False),
    ]

    results = []
    for label, tf, gate in transforms:
        tx, ty = tf(x, y)
        mi_t, _ = ksg_module.ksg_mi(tx, ty, k=K_CORE, jitter_seed=7)
        rel_vs_id = abs(mi_t - mi_id) / max(abs(mi_id), 1e-9)
        rel_vs_true = abs(mi_t - I_TRUE) / I_TRUE
        worst_rel = max(rel_vs_id, rel_vs_true)
        passed = worst_rel <= 0.10 if gate else True  # DIAG always passes
        loc = (
            "OK" if worst_rel <= 0.10
            else ("missing C5 standardization" if label == "scale_10x"
                  else "C4 jitter / tie handling under nonlinear spacing")
        )
        results.append(_make_result(
            f"T5_{label}", "GATE" if gate else "DIAG",
            worst_rel, 0.10, passed, loc
        ))
    return results


def run_t4(outdir: Path) -> list[dict]:
    """T4 — independence floor."""
    results = []
    for tag, entropy, generator in [
        ("gaussian", MASTER_ENTROPY, lambda rng: gt.generate_bivariate_gaussian(N_CORE, 0.0, rng)),
        ("uniform",  NULL_ENTROPY,   lambda rng: gt.generate_bivariate_uniform(N_CORE, rng)),
    ]:
        master = np.random.SeedSequence(entropy=entropy)
        child_seeds = master.spawn(S_CORE)
        estimates = np.empty(S_CORE)
        for i, cs in enumerate(child_seeds):
            rng = np.random.default_rng(cs)
            x, y = generator(rng)
            mi, _ = ksg_module.ksg_mi(x, y, k=K_CORE, jitter_seed=int(cs.entropy & 0xFFFF))
            estimates[i] = mi

        _save_csv(
            outdir / f"t4_{tag}_seeds.csv",
            ["seed_idx", "mi_nats"],
            [[i, estimates[i]] for i in range(S_CORE)],
        )

        mean_e = float(np.mean(estimates))
        q25, q75 = float(np.percentile(estimates, 25)), float(np.percentile(estimates, 75))
        abs_mean = abs(mean_e)
        floor_ok = q25 <= 0.0 <= q75
        passed = abs_mean <= 0.01 and floor_ok
        results.append(_make_result(
            f"T4_{tag}", "GATE", abs_mean, 0.01, passed,
            f"IQR=[{q25:.5f},{q75:.5f}]  0_in_iqr={floor_ok}"
        ))

    # MI_floor deliverable
    master = np.random.SeedSequence(entropy=NULL_ENTROPY)
    child_seeds = master.spawn(S_CORE)
    null_ests = np.empty(S_CORE)
    for i, cs in enumerate(child_seeds):
        rng = np.random.default_rng(cs)
        x, y = gt.generate_bivariate_uniform(N_CORE, rng)
        mi, _ = ksg_module.ksg_mi(x, y, k=K_CORE, jitter_seed=int(cs.entropy & 0xFFFF))
        null_ests[i] = mi
    mi_floor = 3.0 * max(abs(float(np.mean(null_ests))), float(np.std(null_ests, ddof=0)))
    return results, mi_floor


def run_t1(outdir: Path) -> list[dict]:
    """T1 — known-answer bias grid."""
    results = []
    grid = [(0.3, 0.05), (0.5, 0.05), (0.7, 0.05), (0.9, 0.10)]
    for rho, threshold in grid:
        estimates = _run_seeds(rho, N_CORE, K_CORE, S_CORE, MASTER_ENTROPY)
        _save_csv(
            outdir / f"t1_rho{rho}_seeds.csv",
            ["seed_idx", "mi_nats", "i_true"],
            [[i, estimates[i], gt.gaussian_mi(rho)] for i in range(S_CORE)],
        )
        mean_e = float(np.mean(estimates))
        I_true = gt.gaussian_mi(rho)
        rel_bias = abs(mean_e - I_true) / I_true
        passed = rel_bias <= threshold
        results.append(_make_result(
            f"T1_rho{rho}", "GATE", rel_bias, threshold, passed,
            f"mean={mean_e:.5f} I_true={I_true:.5f} bias={mean_e-I_true:+.5f}"
        ))
    return results


def run_t2(outdir: Path) -> dict:
    """T2 — consistency / convergence."""
    rho, I_true = 0.5, gt.gaussian_mi(0.5)
    Ns = [1_250, 2_500, 5_000, 10_000, 20_000]
    biases, means = [], []
    for N in Ns:
        estimates = _run_seeds(rho, N, K_CORE, S_CORE, MASTER_ENTROPY)
        mean_e = float(np.mean(estimates))
        bias = abs(mean_e - I_true)
        means.append(mean_e)
        biases.append(bias)
        _save_csv(
            outdir / f"t2_N{N}_seeds.csv",
            ["seed_idx", "mi_nats"],
            [[i, estimates[i]] for i in range(S_CORE)],
        )

    bias_ratio = biases[1] / max(biases[-1], 1e-15)  # bias(2500)/bias(20000)
    log_N = np.log(Ns)
    log_bias = np.log(np.maximum(biases, 1e-8))
    slope, *_ = sp_stats.linregress(log_N, log_bias)
    slope = float(slope)

    passed_a = biases[-1] < biases[1] / 2.0
    passed_b = slope < -0.30
    passed = passed_a and passed_b
    loc = (
        f"bias_ratio(2.5k/20k)={bias_ratio:.2f} slope={slope:.3f} "
        f"{'OK' if passed else 'FAIL — structural inconsistency, not tuning'}"
    )
    return _make_result("T2_consistency", "GATE", max(abs(slope) - 0.30, 0), 0.30, passed, loc)


def run_t3_pilot() -> tuple[float, dict]:
    """T3 pilot — run ONE time to freeze τ_var."""
    estimates = _run_seeds(0.5, N_CORE, K_CORE, S_CORE, MASTER_ENTROPY)
    sd = float(np.std(estimates, ddof=0))
    ci_half = 1.96 * sd / np.sqrt(S_CORE)
    result = _make_result(
        "T3_variance_pilot", "GATE", sd, float("nan"), True,
        f"SD={sd:.5f} nats | 95% CI ±{ci_half:.5f}"
    )
    return sd, result


def run_t8() -> list[dict]:
    """T8 DIAG — saturation sanity."""
    rng = np.random.default_rng(0x54385F53)
    results = []
    prev_mi = None
    for rho in [0.99, 0.999]:
        x, y = gt.generate_bivariate_gaussian(N_CORE, rho, rng)
        mi, _ = ksg_module.ksg_mi(x, y, k=K_CORE, jitter_seed=8)
        finite_ok = np.isfinite(mi)
        mono_ok = prev_mi is None or mi > prev_mi
        results.append(_make_result(
            f"T8_rho{rho}", "DIAG", mi, float("inf"),
            finite_ok and mono_ok,
            f"finite={finite_ok} monotone_vs_prev={mono_ok}"
        ))
        prev_mi = mi

    x = rng.standard_normal(N_CORE)
    mi_exact, _ = ksg_module.ksg_mi(x, x, k=K_CORE, jitter_seed=8)
    results.append(_make_result(
        "T8_YeqX", "DIAG", mi_exact, 0.0, mi_exact > 0,
        f"Y=X MI={mi_exact:.4f} (saturates high, must be positive)"
    ))
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="KSG validation suite runner")
    parser.add_argument(
        "--pilot-only", action="store_true",
        help="Run only the T3 pilot (print τ_var + SHA, then exit)"
    )
    parser.add_argument(
        "--outdir", type=Path,
        default=Path(__file__).parent / "results",
        help="Directory for JSON + CSV outputs"
    )
    parser.add_argument(
        "--no-slow", action="store_true",
        help="Skip T1, T2, T4 (fast run — only T5, T6, T7)"
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    sha = _git_sha()

    if args.pilot_only:
        print("=" * 60)
        print("T3 PILOT — one-time run to set τ_var")
        print(f"  git SHA: {sha}")
        print("=" * 60)
        sd, result = run_t3_pilot()
        print(f"\n  SD (τ_var) = {sd:.6f} nats")
        print(f"  95% CI ±  {1.96 * sd / np.sqrt(S_CORE):.6f} nats")
        print(f"\n  → In test_ksg.py, set:")
        print(f"      _TAU_VAR_PILOT_SHA = \"{sha}\"")
        print(f"      _TAU_VAR = {sd:.6f}")
        print(f"\n  Commit and do not re-run the pilot.")
        return

    print(f"KSG validation suite  |  SHA={sha}")
    print(f"Core point: N={N_CORE}, k={K_CORE}, S={S_CORE}")
    start = time.monotonic()

    all_results: list[dict] = []

    print("\n[T7] Digamma identities ...", end=" ", flush=True)
    all_results.append(run_t7())
    print("PASS" if all_results[-1]["pass"] else "FAIL")

    print("[T6] Brute-force oracle ...", end=" ", flush=True)
    all_results.append(run_t6())
    print("PASS" if all_results[-1]["pass"] else "FAIL")

    print("[T5] Reparameterization invariance ...", end=" ", flush=True)
    t5_results = run_t5()
    all_results.extend(t5_results)
    gate_t5 = [r for r in t5_results if r["tier"] == "GATE"]
    print("PASS" if all(r["pass"] for r in gate_t5) else "FAIL")

    mi_floor = float("nan")

    if not args.no_slow:
        print("[T4] Independence floor ...", end=" ", flush=True)
        t4_results, mi_floor = run_t4(args.outdir)
        all_results.extend(t4_results)
        print("PASS" if all(r["pass"] for r in t4_results) else "FAIL")

        print("[T1] Known-answer bias (4 ρ values × 100 seeds) ...", end=" ", flush=True)
        t1_results = run_t1(args.outdir)
        all_results.extend(t1_results)
        print("PASS" if all(r["pass"] for r in t1_results) else "FAIL")

        print("[T2] Consistency sweep ...", end=" ", flush=True)
        all_results.append(run_t2(args.outdir))
        print("PASS" if all_results[-1]["pass"] else "FAIL")

        print("[T8] Saturation DIAG ...", end=" ", flush=True)
        all_results.extend(run_t8())
        print("done")

    gate_pass = all(r["pass"] for r in all_results if r["tier"] == "GATE")
    elapsed = time.monotonic() - start

    # ── Write results JSON ────────────────────────────────────────────────────
    utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"run_{sha[:8]}_{utc}"
    out_json = args.outdir / f"{run_id}.json"

    payload = {
        "run_id": run_id,
        "config": {
            "N_core": N_CORE,
            "k_core": K_CORE,
            "S_core": S_CORE,
            "master_entropy": MASTER_ENTROPY,
        },
        "git_sha": sha,
        "lib_versions": _lib_versions(),
        "master_seed": str(MASTER_ENTROPY),
        "MI_floor": mi_floor,
        "tau_var_pilot_sha": "see test_ksg.py",
        "elapsed_seconds": round(elapsed, 1),
        "tests": all_results,
        "overall_gate_pass": gate_pass,
    }

    out_json.write_text(json.dumps(payload, indent=2))
    print(f"\nResults → {out_json}")
    print(f"Overall GATE pass: {'YES' if gate_pass else 'NO'}")
    print(f"Elapsed: {elapsed:.1f}s")

    if not gate_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
