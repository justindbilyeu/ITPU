#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
ITPU smoke test - verify basic functionality works.
"""

import sys
import numpy as np

def main():
    print("ITPU Smoke Test")
    print("=" * 40)
    
    try:
        # Test imports
        from itpu.sdk import ITPU
        from itpu.utils.windowed import windowed_mi
        print("✓ Imports successful")
        
        # Test data
        rng = np.random.default_rng(42)
        x = rng.normal(size=5000)
        y = 0.7 * x + 0.3 * rng.normal(size=5000)
        
        # Test histogram MI
        itpu = ITPU(device="software")
        mi = itpu.mutual_info(x, y, method="hist", bins=32)
        print(f"✓ Histogram MI: {mi:.3f} nats")
        
        if mi <= 0:
            print("✗ ERROR: MI should be positive for correlated data")
            return 1
            
        # Test windowed MI
        starts, mi_vals = windowed_mi(x, y, window_size=500, hop_size=100, bins=16)
        print(f"✓ Windowed MI: {len(mi_vals)} windows, mean={np.mean(mi_vals):.3f}")
        
        if len(mi_vals) == 0:
            print("✗ ERROR: No windowed results")
            return 1
            
        # Test independent data
        x_indep = rng.normal(size=1000)
        y_indep = rng.normal(size=1000)
        mi_indep = itpu.mutual_info(x_indep, y_indep, method="hist", bins=16)
        print(f"✓ Independent MI: {mi_indep:.3f} nats (should be ~0)")
        
        print("\n🎉 Basic functionality working!")
        return 0
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nTroubleshooting:")
        print("- Make sure you're in the repo root (contains itpu/ folder)")
        print("- Install dependencies: pip install numpy scipy matplotlib")
        return 1

if __name__ == "__main__":
    sys.exit(main())
