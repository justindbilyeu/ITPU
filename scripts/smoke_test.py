#!/usr/bin/env python3
"""
Smoke test for ITPU software baseline.
Quick check that histogram MI and windowed MI work correctly.
"""

import sys
import numpy as np

def main():
    print("ITPU Smoke Test")
    print("=" * 40)
    
    try:
        # Test basic imports
        from itpu.sdk import ITPU
        from itpu.utils.windowed import windowed_mi
        print("✓ Imports successful")
        
        # Generate test data with known relationship
        rng = np.random.default_rng(42)
        n = 10_000
        x = rng.normal(size=n)
        y = 0.7 * x + 0.3 * rng.normal(size=n)  # Strong correlation
        
        # Test histogram MI
        itpu = ITPU(device="software")
        mi_hist = itpu.mutual_info(x, y, method="hist", bins=64)
        print(f"✓ Histogram MI: {mi_hist:.3f} nats")
        
        # Basic sanity checks
        if mi_hist <= 0:
            print("✗ ERROR: MI should be positive for correlated data")
            return 1
        if mi_hist > 2.0:
            print("✗ WARNING: MI seems unusually high")
            
        # Test windowed MI
        starts, mi_vals = windowed_mi(x, y, window_size=1000, hop_size=200, bins=32)
        print(f"✓ Windowed MI: {len(mi_vals)} windows, mean={np.mean(mi_vals):.3f}")
        
        if len(mi_vals) == 0:
            print("✗ ERROR: No windowed results generated")
            return 1
            
        # Test with independent data (should give low MI)
        x_indep = rng.normal(size=1000)
        y_indep = rng.normal(size=1000)
        mi_indep = itpu.mutual_info(x_indep, y_indep, method="hist", bins=32)
        print(f"✓ Independent MI: {mi_indep:.3f} nats (should be near zero)")
        
        if mi_indep > 0.1:
            print("✗ WARNING: Independent variables show high MI")
            
        print("\n🎉 Basic functionality working!")
        print("Next: try 'python examples/eeg_streaming_demo.py'")
        return 0
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Make sure you're in the repo root and have installed dependencies:")
        print("pip install numpy scipy matplotlib")
        return 1
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())import numpy as np
from itpu.sdk import ITPU
from itpu.utils.windowed import windowed_mi

def main():
    rng = np.random.default_rng(0)
    x = rng.normal(size=50_000)
    y = 0.6 * x + 0.4 * rng.normal(size=50_000)

    itpu = ITPU(device="software")

    mi_hist = itpu.mutual_info(x, y, method="hist", bins=64)
    print(f"[hist] MI (nats): {mi_hist:.3f}")

    starts, vals = windowed_mi(x, y, window_size=5000, hop_size=1000, bins=64)
    print(f"[hist] windowed count: {len(vals)}, mean MI: {vals.mean():.3f}")

    # Optional: try KSG if available
    try:
        mi_ksg = itpu.mutual_info(x, y, method="ksg", k=5)
        print(f"[ksg]  MI (nats): {mi_ksg:.3f}")
    except Exception as e:
        print(f"[ksg]  skipped ({e})")

if __name__ == "__main__":
    main()
