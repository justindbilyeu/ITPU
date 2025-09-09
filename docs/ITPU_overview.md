# ITPU Overview

The Information-Theoretic Processing Unit (ITPU) accelerates entropy and mutual information (MI) computations. Today this repo ships a software SDK baseline; the same API will target an FPGA/ASIC card.

- **Why:** GPUs are great at matrix math, weak at irregular probability ops (histograms, k-NN).
- **What:** Kernels for HIST/ENTROPY, MI (plug-in, KSG), k-NN, and streaming derivatives.
- **Where:** Neuroscience/BCI, medical imaging (MI registration), causal ML, quantum/QEC.
