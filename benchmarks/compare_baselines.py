import time, numpy as np
from itpu.sdk import ITPU
try:
    from sklearn.feature_selection import mutual_info_regression
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

def benchmark_mi_methods(n_samples=10000, n_trials=3):
    print(f"Benchmarking ({n_samples:,} samples, {n_trials} trials)")
    rng = np.random.default_rng(42)
    x = rng.normal(size=n_samples)
    y = 0.6*x + 0.4*rng.normal(size=n_samples)

    results = {}
    itpu = ITPU(device="software")

    # histogram
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        mi = itpu.mutual_info(x, y, method="hist", bins=64)
        times.append(time.perf_counter() - t0)
    results['itpu_hist'] = (np.mean(times), mi)

    # ksg
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        mi = itpu.mutual_info(x, y, method="ksg", k=5)
        times.append(time.perf_counter() - t0)
    results['itpu_ksg'] = (np.mean(times), mi)

    if HAS_SKLEARN:
        times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            mi = mutual_info_regression(x.reshape(-1,1), y, random_state=0)[0]
            times.append(time.perf_counter() - t0)
        results['sklearn'] = (np.mean(times), mi)

    # print table
    baseline = results['sklearn'][0] if HAS_SKLEARN else results['itpu_hist'][0]
    print(f"\n{'Method':<12} {'Time (ms)':>10} {'MI':>10} {'Speedup':>10}")
    print("-"*46)
    for name, (t, mi) in results.items():
        speed = baseline / t
        print(f"{name:<12} {t*1000:>10.2f} {mi:>10.3f} {speed:>10.1f}x")

if __name__ == "__main__":
    benchmark_mi_methods()
