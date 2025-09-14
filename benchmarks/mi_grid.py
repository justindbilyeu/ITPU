import numpy as np, time
from itpu.sdk import ITPU

def analytic_mi_gauss(rho):
    return -0.5*np.log(1 - rho**2)

def make_pair(n, rho, rng):
    x = rng.standard_normal(n)
    eps = rng.standard_normal(n)
    # y = a*x + b*eps with correlation rho = a/sqrt(a^2 + b^2) -> choose a=rho, b=sqrt(1-rho^2)
    a = rho; b = np.sqrt(max(1e-12, 1 - rho**2))
    y = a*x + b*eps
    return x, y

def main():
    rng = np.random.default_rng(0)
    itpu = ITPU()
    ns = [50_000, 200_000]
    rhos = [0.0, 0.3, 0.6, 0.9]
    print("n,rho,analytic,hist64,ksg5,time_hist_s,time_ksg_s")
    for n in ns:
        for rho in rhos:
            x, y = make_pair(n, rho, rng)
            t0=time.time(); mi_h=itpu.mutual_info(x,y,method="hist",bins=64); t1=time.time()
            xs, ys = x[:min(n, 50_000)], y[:min(n, 50_000)]
            t2=time.time(); mi_k=itpu.mutual_info(xs,ys,method="ksg",k=5); t3=time.time()
            print(f"{n},{rho:.2f},{analytic_mi_gauss(rho):.4f},{mi_h:.4f},{mi_k:.4f},{t1-t0:.3f},{t3-t2:.3f}")

if __name__ == "__main__":
    main()
