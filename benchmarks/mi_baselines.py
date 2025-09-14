# benchmarks/mi_baselines.py
import time, numpy as np
from itpu.sdk import ITPU
from scipy.stats import entropy

def hist_mi_numpy(x, y, bins=64):
  hx, _ = np.histogram(x, bins=bins)
  hy, _ = np.histogram(y, bins=bins)
  hxy, _, _ = np.histogram2d(x, y, bins=bins)
  def H(c): 
    p = c / c.sum()
    p = p[p>0]
    return -(p*np.log(p)).sum()
  return H(hx)+H(hy)-H(hxy)

def main():
  rng = np.random.default_rng(0)
  for n in [50_000, 200_000, 1_000_000]:
    x = rng.standard_normal(n)
    y = 0.6*x + 0.4*rng.standard_normal(n)

    t0 = time.time(); mi_ref = hist_mi_numpy(x,y,64); t1=time.time()
    itpu = ITPU()
    t2 = time.time(); mi_itpu = itpu.mutual_info(x,y,method="hist",bins=64); t3=time.time()

    print(f"n={n:,} | ref {t1-t0:.3f}s | itpu {t3-t2:.3f}s | MI ref {mi_ref:.3f} | MI itpu {mi_itpu:.3f}")

if __name__ == "__main__":
  main()
