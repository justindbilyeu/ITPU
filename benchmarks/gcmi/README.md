# benchmarks/gcmi

Placeholder for ITPU vs. GCMI (Gaussian-Copula MI) comparison benchmarks.

## Scope

These benchmarks will compare ITPU's KSG and histogram MI estimators against
the GCMI library across varying sample sizes, correlation strengths, and noise
levels. Target metrics: wall-clock time, bias, and variance of MI estimates.

## Status

Not yet implemented. Outputs go to `results/benchmarks/gcmi/`.

## Usage (once implemented)

```bash
python benchmarks/gcmi/run_comparison.py
```
