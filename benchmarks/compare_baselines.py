import time, numpy as np
from itpu.sdk import ITPU

try:
    from sklearn.feature_selection import mutual_info_regression
    HAS_SK = True
except Exception:
    HAS_SK = False

def bench(n=100_000, trials=3):
    rng = np.random.default_rng(123)
    x = rng.normal(size=n)
    y = 0.6 * x + 0.4 * rng.normal(size=n)
    itpu = ITPU(device="software")

    def t(fn):
        times = []
        val = None
        for _ in range(trials):
            t0 = time.perf_counter()
            val = fn()
            times.append(time.perf_counter() - t0)
        return np.mean(times), val

    th, mih = t(lambda: itpu.mutual_info(x, y, method="hist", bins=64))
    print(f"ITPU hist:  {mih:.3f} nats, {th*1000:.1f} ms")

    try:
        tk, mik = t(lambda: itpu.mutual_info(x, y, method="ksg", k=5))
        print(f"ITPU KSG:   {mik:.3f} nats, {tk*1000:.1f} ms")
    except Exception as e:
        print(f"ITPU KSG:   skipped ({e})")

    if HAS_SK:
        ts, mis = t(lambda: mutual_info_regression(x.reshape(-1,1), y, random_state=0)[0])
        print(f"sklearn:     {mis:.3f} (units vary), {ts*1000:.1f} ms")
    else:
        print("sklearn:     not installed (pip install -e '.[dev]')")

if __name__ == "__main__":
    bench()
