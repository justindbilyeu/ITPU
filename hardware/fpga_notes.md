# FPGA Notes (Pathfinder)

- Tiled joint-hist blocks with local SRAM (e.g., 128x128 bins)
- DMA engines to feed tiles; overlap compute & transfer
- Accumulate partial histograms; reduce on-chip
- Vector math for log/exp and MI reduction
- Perf counters: GB/s, occupancy, stall reasons
