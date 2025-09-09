# Kernel Specs (MVP)

- HIST_BUILD / HIST_REDUCE
  - Input: streams of (x,y) or batched arrays
  - Output: joint/marginal hist, entropy H, conditional H(X|Y)
  - Notes: on-chip SRAM tiling, integer atomics, saturation counters

- REDUCE_MI
  - Input: joint & marginal histograms
  - Output: I(X;Y) with numerically-stable log/exp

- KNN_QUERY (for KSG)
  - Input: points (metric-agnostic), k
  - Output: counts within epsilon in marginals; supports streaming windows

- LOG/EXP
  - Range reduction; Kahan-compensated accumulation

- WITNESS_FLUX_DERIV
  - Streaming d/dt for entropy/MI; windowed or EWMA options
