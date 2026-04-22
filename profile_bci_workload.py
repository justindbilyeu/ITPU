"""Profile BCI workload components for ITPU R2 baseline.

Profiles six components and writes results to profile_results.txt.
Run directly: python profile_bci_workload.py
"""
import cProfile
import io
import pstats
import time

import numpy as np

from itpu.sdk import ITPU
from itpu.stats.surrogates import iaaft_surrogate, shuffle_surrogate
from itpu.stats.surrogate_test import surrogate_test

RNG = np.random.default_rng(42)
N = 1000
N_SURROGATES = 499

x = RNG.standard_normal(N)
y = 0.6 * x + 0.4 * RNG.standard_normal(N)
sdk = ITPU(device="software")


def profile_component(label, fn):
    pr = cProfile.Profile()
    pr.enable()
    t0 = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - t0
    pr.disable()

    buf = io.StringIO()
    ps = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
    ps.print_stats(5)
    top5 = "\n".join(buf.getvalue().splitlines()[8:13])

    line = f"\n{'='*60}\n{label}\nwall time: {elapsed:.3f}s\ntop 5 cumtime:\n{top5}"
    print(line)
    return line


results = []

results.append(profile_component(
    "surrogate_test() — KSG + IAAFT",
    lambda: surrogate_test(x, y, method="ksg", n_surrogates=N_SURROGATES,
                           surrogate_type="iaaft", rng=0),
))

results.append(profile_component(
    "surrogate_test() — histogram + IAAFT",
    lambda: surrogate_test(x, y, method="hist", n_surrogates=N_SURROGATES,
                           surrogate_type="iaaft", rng=0),
))

results.append(profile_component(
    "KSG MI alone (499 calls, n=1000)",
    lambda: [sdk.mutual_info(x, RNG.permutation(y), method="ksg")
             for _ in range(N_SURROGATES)],
))

results.append(profile_component(
    "Histogram MI alone (499 calls, n=1000)",
    lambda: [sdk.mutual_info(x, RNG.permutation(y), method="hist")
             for _ in range(N_SURROGATES)],
))

results.append(profile_component(
    "IAAFT generation alone (499 surrogates, n=1000)",
    lambda: iaaft_surrogate(y, n_surrogates=N_SURROGATES, rng=0),
))

results.append(profile_component(
    "Shuffle generation alone (499 surrogates, n=1000)",
    lambda: shuffle_surrogate(y, n_surrogates=N_SURROGATES, rng=0),
))

with open("profile_results.txt", "w") as f:
    f.write("ITPU BCI Workload Profile\n")
    f.write(f"n={N}, n_surrogates={N_SURROGATES}\n")
    f.write("".join(results))

print("\nResults written to profile_results.txt")
