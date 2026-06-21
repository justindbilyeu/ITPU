"""
ITPU R1 KSG Estimator Validation — pytest battery.

Run the fast GATE tests (T6, T7):
    pytest validation/ksg/test_ksg.py -m "not slow and not diag"

Run all GATE tests (includes S=100 sweeps — ~2 min):
    pytest validation/ksg/test_ksg.py -m "not diag"

Run everything including diagnostics:
    pytest validation/ksg/test_ksg.py

Markers
-------
  (none)   fast GATE tests (T6, T7)
  slow     GATE tests requiring S=100 × N=10000 sweeps (T1–T5 with seeds)
  diag     Non-blocking characterization tests (T8, T9, k-sweep)

T3 variance threshold
---------------------
_TAU_VAR is set by exactly one pilot run:
    python validation/ksg/run_suite.py --pilot-only

After the pilot, fill in _TAU_VAR and _TAU_VAR_PILOT_SHA below and commit.
Do not re-run the pilot after seeing results.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

import ground_truth as gt
import ksg as ksg_module
import oracles

# ── Core operating point (all mandatory tests unless noted) ──────────────────
N_CORE = 10_000
K_CORE = 4
S_CORE = 100

# ── T3 variance threshold — frozen from ONE pilot run ────────────────────────
# Pilot procedure: python validation/ksg/run_suite.py --pilot-only
# After the pilot prints the SD and SHA, fill in both constants and commit.
# Do not adjust after seeing the confirmatory run.
_TAU_VAR_PILOT_SHA: str = "f4e6d3c6cb674f494678a98da27292feb330ad4a"
_TAU_VAR: float | None = 0.009651      # SD from pilot (nats; regression gate)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _run_seeds(rho: float, N: int = N_CORE, k: int = K_CORE, S: int = S_CORE) -> np.ndarray:
    """Return S KSG estimates for bivariate Gaussian(rho) at (N, k)."""
    master = np.random.SeedSequence(entropy=0x4954_5055)  # "ITPU" in hex
    child_seeds = master.spawn(S)
    estimates = np.empty(S)
    for i, cs in enumerate(child_seeds):
        rng = np.random.default_rng(cs)
        x, y = gt.generate_bivariate_gaussian(N, rho, rng)
        mi, _ = ksg_module.ksg_mi(x, y, k=k, jitter_seed=int(cs.entropy & 0xFFFF))
        estimates[i] = mi
    return estimates


def _run_seeds_uniform(N: int = N_CORE, k: int = K_CORE, S: int = S_CORE) -> np.ndarray:
    """Return S KSG estimates for independent Uniform(0,1)² (null, T4b)."""
    master = np.random.SeedSequence(entropy=0x4E554C4C)  # "NULL"
    child_seeds = master.spawn(S)
    estimates = np.empty(S)
    for i, cs in enumerate(child_seeds):
        rng = np.random.default_rng(cs)
        x, y = gt.generate_bivariate_uniform(N, rng)
        mi, _ = ksg_module.ksg_mi(x, y, k=k, jitter_seed=int(cs.entropy & 0xFFFF))
        estimates[i] = mi
    return estimates


# ── T7 — Digamma identities [GATE, fast] ────────────────────────────────────

def test_t7_digamma_identities():
    """T7: ψ(1)=−γ and ψ(2)=1−γ to machine precision. Catches ln-for-ψ (C2)."""
    oracles.assert_digamma()


# ── T6 — Brute-force oracle agreement [GATE, fast] ──────────────────────────

def test_t6_oracle_agreement():
    """T6: |I_brute − I_prod| ≤ 1e-9 for N=500, ρ=0.6, single seed."""
    rng = np.random.default_rng(0x54365F4F)  # "T6_O" mnemonic
    x, y = gt.generate_bivariate_gaussian(500, 0.6, rng)

    mi_prod, _ = ksg_module.ksg_mi(x, y, k=K_CORE, jitter_seed=42)
    mi_brute = oracles.mi_bruteforce(x, y, k=K_CORE, jitter_seed=42)

    diff = abs(mi_prod - mi_brute)
    assert diff <= 1e-9, (
        f"T6 FAIL: |I_prod − I_brute| = {diff:.3e} (threshold ≤ 1e-9)\n"
        f"  I_prod = {mi_prod:.9f}, I_brute = {mi_brute:.9f}\n"
        f"  Localization: indexing/metric/counting bug in the fast path."
    )


# ── T5 — Reparameterization invariance [GATE for first three; DIAG rest] ────

@pytest.mark.parametrize(
    "label,transform,gate",
    [
        ("identity",    lambda x, y: (x, y),          True),
        ("scale_10x",   lambda x, y: (10 * x, 0.1 * y), True),
        ("cube_x",      lambda x, y: (x ** 3, y),     True),
        ("sinh_x",      lambda x, y: (np.sinh(x), y), False),  # DIAG
        ("exp_y",       lambda x, y: (x, np.exp(y)),  False),  # DIAG
    ],
)
def test_t5_invariance(label, transform, gate):
    """
    T5: I(g(X),Y) = I(X,Y) for invertible transforms.

    GATE transforms: ±10% relative of identity AND I_true.
    DIAG transforms: report only, never fail the suite.
    """
    I_TRUE = gt.GAUSSIAN_TABLE[0.7]
    rng = np.random.default_rng(0x54355F49)  # "T5_I"
    x, y = gt.generate_bivariate_gaussian(N_CORE, 0.7, rng)

    # Identity estimate (reference)
    mi_id, _ = ksg_module.ksg_mi(x, y, k=K_CORE, jitter_seed=7)

    # Transformed estimate
    tx, ty = transform(x, y)
    mi_t, _ = ksg_module.ksg_mi(tx, ty, k=K_CORE, jitter_seed=7)

    rel_vs_identity = abs(mi_t - mi_id) / max(abs(mi_id), 1e-9)
    rel_vs_true = abs(mi_t - I_TRUE) / I_TRUE

    if gate:
        assert rel_vs_identity <= 0.10, (
            f"T5 FAIL [{label}]: |I_transform − I_identity| / I_identity = "
            f"{rel_vs_identity:.4f} > 0.10\n"
            f"  I_transform={mi_t:.5f}, I_identity={mi_id:.5f}\n"
            f"  Localization: if scale_10x fails → missing C5 (standardization). "
            f"If cube_x fails → C4 jitter / tie handling under nonlinear spacing."
        )
        assert rel_vs_true <= 0.10, (
            f"T5 FAIL [{label}]: |I_transform − I_true| / I_true = "
            f"{rel_vs_true:.4f} > 0.10\n"
            f"  I_transform={mi_t:.5f}, I_true={I_TRUE:.5f}"
        )
    else:
        # DIAG: always pass, just print
        print(
            f"\nT5 DIAG [{label}]: I={mi_t:.5f}, "
            f"rel_vs_id={rel_vs_identity:.4f}, rel_vs_true={rel_vs_true:.4f}"
        )


# ── T4 — Independence floor [GATE, slow] ─────────────────────────────────────

@pytest.mark.slow
def test_t4_independence_floor_gaussian():
    """T4a: |mean(I)| ≤ 0.01 and 0 within IQR for ρ=0 Gaussian null."""
    estimates = _run_seeds(rho=0.0, N=N_CORE, k=K_CORE, S=S_CORE)
    mean_est = float(np.mean(estimates))
    q25, q75 = float(np.percentile(estimates, 25)), float(np.percentile(estimates, 75))

    assert abs(mean_est) <= 0.01, (
        f"T4a FAIL: |mean(I)|={abs(mean_est):.5f} > 0.01 nats "
        f"(Gaussian null, ρ=0). Possible systematic bias."
    )
    assert q25 <= 0.0 <= q75, (
        f"T4a FAIL: 0 not within IQR [{q25:.5f}, {q75:.5f}]. "
        f"Null distribution displaced from zero."
    )


@pytest.mark.slow
def test_t4_independence_floor_uniform():
    """T4b: |mean(I)| ≤ 0.01 and 0 within IQR for Uniform(0,1)² null."""
    estimates = _run_seeds_uniform(N=N_CORE, k=K_CORE, S=S_CORE)
    mean_est = float(np.mean(estimates))
    q25, q75 = float(np.percentile(estimates, 25)), float(np.percentile(estimates, 75))

    assert abs(mean_est) <= 0.01, (
        f"T4b FAIL: |mean(I)|={abs(mean_est):.5f} > 0.01 nats "
        f"(Uniform null). Floor is not a Gaussian artifact."
    )
    assert q25 <= 0.0 <= q75, (
        f"T4b FAIL: 0 not within IQR [{q25:.5f}, {q75:.5f}]."
    )

    # MI_floor: smallest MI the estimator may assert downstream
    sd_null = float(np.std(estimates, ddof=0))
    mi_floor = 3.0 * max(abs(mean_est), sd_null)
    print(f"\n  MI_floor = {mi_floor:.5f} nats (T4 deliverable)")


# ── T1 — Known-answer bias [GATE, slow] ──────────────────────────────────────

@pytest.mark.slow
@pytest.mark.parametrize(
    "rho,threshold",
    [(0.3, 0.05), (0.5, 0.05), (0.7, 0.05), (0.9, 0.10)],
)
def test_t1_known_answer_bias(rho, threshold):
    """
    T1: |mean(I_hat) − I_true| / I_true ≤ threshold.

    ρ≤0.7: threshold 5%.  ρ=0.9: 10% (pre-registered looser bar; KSG
    is known to underestimate high MI — if exceeded, it is a finding
    in the expected negative direction, not a surprise).
    """
    estimates = _run_seeds(rho=rho, N=N_CORE, k=K_CORE, S=S_CORE)
    mean_est = float(np.mean(estimates))
    I_true = gt.gaussian_mi(rho)
    rel_bias = abs(mean_est - I_true) / I_true

    assert rel_bias <= threshold, (
        f"T1 FAIL [ρ={rho}]: rel_bias={rel_bias:.4f} > {threshold}\n"
        f"  mean_est={mean_est:.5f}, I_true={I_true:.5f}, bias={mean_est-I_true:+.5f}\n"
        f"  Localization: uniform same-sign offset → C1 (metric) or C2 (digamma). "
        f"Bias only at high ρ → expected KSG degradation."
    )


# ── T2 — Consistency / convergence [GATE, slow] ──────────────────────────────

@pytest.mark.slow
def test_t2_consistency():
    """
    T2: Bias decreasing in N for ρ=0.5.

    (a) |bias(N=20000)| < |bias(N=1250)| by ≥ 2× margin.
    (b) OLS slope of log|bias| on log N < −0.30.
    """
    rho = 0.5
    I_true = gt.gaussian_mi(rho)
    Ns = [1_250, 2_500, 5_000, 10_000, 20_000]
    biases = []

    for N in Ns:
        ests = _run_seeds(rho=rho, N=N, k=K_CORE, S=S_CORE)
        biases.append(abs(float(np.mean(ests)) - I_true))

    bias_1250 = biases[0]
    bias_20000 = biases[-1]

    # Compare endpoints (N=1250 vs N=20000) — N=2500 can land below its own
    # noise floor with S=100 seeds, making the 2× check unreliable there.
    assert bias_20000 < bias_1250 / 2.0, (
        f"T2a FAIL: bias(20k)={bias_20000:.5f} not < bias(1.25k)/2={bias_1250/2:.5f}\n"
        f"  Biases: {dict(zip(Ns, [f'{b:.5f}' for b in biases]))}\n"
        f"  Localization: flat/non-decreasing bias → structural bug "
        f"(likely C3 counting or C6 approximate counts). Not tuning."
    )

    log_N = np.log(Ns)
    log_bias = np.log(np.maximum(biases, 1e-8))  # guard against log(0)
    slope, _, _, _, _ = stats.linregress(log_N, log_bias)

    assert slope < -0.30, (
        f"T2b FAIL: OLS slope of log|bias| on log N = {slope:.4f} (threshold < -0.30)\n"
        f"  Biases: {dict(zip(Ns, [f'{b:.5f}' for b in biases]))}\n"
        f"  Localization: positive or near-zero slope → inconsistent estimator → "
        f"structural bug, not parameter tuning."
    )


# ── T3 — Variance / precision [GATE, slow, pilot-frozen τ_var] ───────────────

@pytest.mark.slow
def test_t3_variance():
    """
    T3: SD ≤ τ_var (frozen from one pilot run).

    τ_var is set by running: python validation/ksg/run_suite.py --pilot-only
    The pilot result is frozen in _TAU_VAR at the top of this file.
    A T3 pass means "met the pilot-frozen empirical bar" — not "provably precise."
    """
    if _TAU_VAR is None:
        pytest.skip(
            "T3 SKIPPED: _TAU_VAR not yet frozen. "
            "Run: python validation/ksg/run_suite.py --pilot-only"
        )

    estimates = _run_seeds(rho=0.5, N=N_CORE, k=K_CORE, S=S_CORE)
    sd = float(np.std(estimates, ddof=0))
    ci_half = 1.96 * sd / np.sqrt(S_CORE)

    assert sd <= _TAU_VAR, (
        f"T3 FAIL: SD={sd:.5f} nats > τ_var={_TAU_VAR:.5f} "
        f"(pilot SHA={_TAU_VAR_PILOT_SHA})\n"
        f"  95% CI half-width: {ci_half:.5f} nats"
    )
    print(
        f"\n  T3 SD={sd:.5f} nats ≤ τ_var={_TAU_VAR:.5f}  "
        f"(95% CI ±{ci_half:.5f})"
    )


# ── T8 — Saturation sanity [DIAG, slow] ──────────────────────────────────────

@pytest.mark.diag
@pytest.mark.slow
def test_t8_saturation():
    """
    T8 DIAG: I_hat increases with ρ, finite, no NaN/inf. Y=X saturates high.
    """
    rng = np.random.default_rng(0x54385F53)  # "T8_S"
    results = {}

    for rho in [0.99, 0.999]:
        x, y = gt.generate_bivariate_gaussian(N_CORE, rho, rng)
        mi, _ = ksg_module.ksg_mi(x, y, k=K_CORE, jitter_seed=8)
        results[rho] = mi
        assert np.isfinite(mi), f"T8 DIAG: NaN/inf at ρ={rho}"

    # Y=X exactly
    x = rng.standard_normal(N_CORE)
    mi_exact, _ = ksg_module.ksg_mi(x, x, k=K_CORE, jitter_seed=8)
    assert np.isfinite(mi_exact), "T8 DIAG: NaN/inf for Y=X"
    assert mi_exact > 0, f"T8 DIAG: Y=X returned non-positive MI ({mi_exact:.4f})"

    assert results[0.99] < results[0.999], (
        f"T8 DIAG: MI not strictly increasing: "
        f"ρ=0.99 → {results[0.99]:.4f}, ρ=0.999 → {results[0.999]:.4f}"
    )

    print(
        f"\n  T8 DIAG: ρ=0.99 → {results[0.99]:.4f}, "
        f"ρ=0.999 → {results[0.999]:.4f}, "
        f"Y=X → {mi_exact:.4f}"
    )


# ── T9 — Dimensionality stress [DIAG] ────────────────────────────────────────

@pytest.mark.diag
@pytest.mark.slow
def test_t9_dimensionality():
    """
    T9 DIAG: 2D Gaussian, C=[[0.6,0.1],[0.1,0.5]], I_true=0.394719.
    Report relative bias at N=10000 and N=40000.
    """
    I_true = gt.MI_2D_TRUE

    for N in [10_000, 40_000]:
        rng = np.random.default_rng(0x54395F44)  # "T9_D"
        X, Y = gt.generate_multidim_gaussian(N, gt.C_2D, rng)

        # Flatten to scalar MI approximation using the first component pair
        # (full d-dim KSG would require a separate implementation — this
        # serves as a rough probe; a complete 2D KSG is left as R2 follow-on)
        mi_approx_list = []
        for d in range(X.shape[1]):
            mi_d, _ = ksg_module.ksg_mi(X[:, d], Y[:, d], k=K_CORE, jitter_seed=9)
            mi_approx_list.append(mi_d)
        mi_approx = float(np.sum(mi_approx_list))  # independence approximation

        rel_bias = (mi_approx - I_true) / I_true
        print(
            f"\n  T9 DIAG [N={N}]: I_approx={mi_approx:.5f}, "
            f"I_true={I_true:.5f}, rel_bias={rel_bias:+.4f} "
            f"(note: component-sum approximation, not full 2D KSG)"
        )


# ── k-sweep [DIAG] ────────────────────────────────────────────────────────────

@pytest.mark.diag
@pytest.mark.slow
def test_k_sensitivity_sweep():
    """
    k-sweep DIAG: bias and SD at ρ=0.5, N=10000, k ∈ {1,2,4,8,16}.

    Output sets the k parameter for R2 hardware (kNN cost scales with k).
    """
    rho = 0.5
    I_true = gt.gaussian_mi(rho)
    ks = [1, 2, 4, 8, 16]
    print(f"\n  k-sweep [ρ={rho}, N={N_CORE}, S={S_CORE}]")
    print(f"  {'k':>4}  {'mean':>8}  {'bias':>8}  {'SD':>8}")

    for k in ks:
        ests = _run_seeds(rho=rho, N=N_CORE, k=k, S=S_CORE)
        mean_e = float(np.mean(ests))
        sd_e = float(np.std(ests, ddof=0))
        bias_e = mean_e - I_true
        print(f"  {k:>4}  {mean_e:>8.5f}  {bias_e:>+8.5f}  {sd_e:>8.5f}")
