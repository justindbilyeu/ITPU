#!/usr/bin/env python3
"""
Real-time EEG Mutual Information Dashboard & Benchmark

Run:
  python examples/eeg_realtime_dashboard.py
"""

from __future__ import annotations
import time
from collections import deque
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button

from itpu.sdk import ITPU
from itpu.utils.windowed import windowed_mi


# ----------------------------- EEG Simulator ----------------------------- #
class EEGSimulator:
    """Simulates realistic EEG-like data with controllable states."""

    def __init__(self, n_channels: int = 8, sample_rate: int = 250):
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        self.t = 0.0
        self.state = "eyes_open"  # "eyes_open", "eyes_closed", "attention"
        self.rng = np.random.default_rng(0)
        # Simplified 10-20 names
        self.channel_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4'][:n_channels]

        # Per-state parameters
        self.state_params = {
            "eyes_open":   {"alpha": 0.5, "beta": 1.0, "gamma": 0.3, "corr": 0.3},
            "eyes_closed": {"alpha": 2.0, "beta": 0.5, "gamma": 0.2, "corr": 0.7},
            "attention":   {"alpha": 0.3, "beta": 1.5, "gamma": 0.8, "corr": 0.5},
        }

    def set_state(self, new_state: str) -> None:
        if new_state in self.state_params:
            self.state = new_state
            print(f"[EEG] state -> {new_state}")

    def generate_sample(self, duration: float = 0.1) -> np.ndarray:
        """Generate (n_channels, n_samples) for given duration in seconds."""
        n_samples = int(duration * self.sample_rate)
        t = np.linspace(self.t, self.t + duration, n_samples, endpoint=False)
        self.t += duration

        p = self.state_params[self.state]
        # Base mixture for correlation
        base = (p["alpha"] * np.sin(2*np.pi*10*t + self.rng.random()*2*np.pi) +
                p["beta"]  * np.sin(2*np.pi*20*t + self.rng.random()*2*np.pi) +
                p["gamma"] * np.sin(2*np.pi*40*t + self.rng.random()*2*np.pi) +
                self.rng.normal(0, 0.5, size=n_samples))

        X = np.empty((self.n_channels, n_samples), dtype=float)
        for i, name in enumerate(self.channel_names):
            noise = self.rng.normal(0, 0.5, size=n_samples)
            sig = p["corr"] * base + (1 - p["corr"]) * noise
            if name.startswith("Fp"):   # more theta frontally
                sig += 0.3 * np.sin(2*np.pi*5*t)
            if name.startswith("P"):    # more alpha parietally
                sig += 0.5 * np.sin(2*np.pi*10*t)
            X[i] = sig
        return X


# ----------------------------- MI Utilities ----------------------------- #
def mi_matrix_hist(X: np.ndarray, window_size: int = 500, bins: int = 64) -> np.ndarray:
    """
    Pairwise histogram MI over the last 'window_size' samples.
    X: (C, N) channels by time.
    """
    C, N = X.shape
    if N < window_size:
        return np.zeros((C, C), dtype=float)
    itpu = ITPU()
    seg = X[:, -window_size:]
    M = np.zeros((C, C), dtype=float)
    for i in range(C):
        for j in range(i+1, C):
            mij = itpu.mutual_info(seg[i], seg[j], method="hist", bins=bins)
            M[i, j] = M[j, i] = mij
    return M


# ----------------------------- Dashboard ----------------------------- #
class RealTimeMIDashboard:
    """Real-time dashboard showing MI heatmap + a tracked-pair MI time series."""

    def __init__(self, eeg_sim: EEGSimulator, window_size: int = 500, update_interval_ms: int = 100):
        self.eeg_sim = eeg_sim
        self.ws = window_size
        self.dt = update_interval_ms
        self.fs = eeg_sim.sample_rate

        # Rolling buffer of recent samples (for all channels)
        self.X = np.zeros((eeg_sim.n_channels, 0))
        # Time series for a tracked pair (Fp1-Fp2 if available)
        self.track_pair = (0, 1)
        self.ts_t, self.ts_mi = [], []

        # Figure & axes
        self.fig = plt.figure(figsize=(14, 9))
        self.ax_heat = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
        self.ax_ts   = plt.subplot2grid((3, 3), (2, 0), colspan=2)
        self.ax_raw  = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
        self.ax_ctrl = plt.subplot2grid((3, 3), (2, 2)); self.ax_ctrl.axis("off")

        # Buttons
        self.btn_open     = Button(plt.axes([0.72, 0.78, 0.22, 0.1]), "Eyes Open")
        self.btn_closed   = Button(plt.axes([0.72, 0.62, 0.22, 0.1]), "Eyes Closed")
        self.btn_attention= Button(plt.axes([0.72, 0.46, 0.22, 0.1]), "Attention")
        self.btn_open.on_clicked(lambda _: self.eeg_sim.set_state("eyes_open"))
        self.btn_closed.on_clicked(lambda _: self.eeg_sim.set_state("eyes_closed"))
        self.btn_attention.on_clicked(lambda _: self.eeg_sim.set_state("attention"))

        # Heatmap
        C = self.eeg_sim.n_channels
        self.MI = np.zeros((C, C))
        self.im = self.ax_heat.imshow(self.MI, aspect="auto", vmin=0, vmax=1.0)
        self.ax_heat.set_title("Real-time MI (current window)")
        self.ax_heat.set_xticks(range(C)); self.ax_heat.set_xticklabels(self.eeg_sim.channel_names, rotation=45)
        self.ax_heat.set_yticks(range(C)); self.ax_heat.set_yticklabels(self.eeg_sim.channel_names)
        plt.colorbar(self.im, ax=self.ax_heat, label="MI (nats)")

        # Time series for tracked pair
        self.line_ts, = self.ax_ts.plot([], [], lw=2)
        self.ax_ts.set_title(
            f"MI over time: {self.eeg_sim.channel_names[self.track_pair[0]]}-"
            f"{self.eeg_sim.channel_names[self.track_pair[1]]}"
        )
        self.ax_ts.set_xlabel("Time (s)"); self.ax_ts.set_ylabel("MI (nats)")
        self.ax_ts.set_xlim(0, 20); self.ax_ts.set_ylim(0, 1.2); self.ax_ts.grid(True, alpha=0.3)

        # Raw preview for the tracked pair (last 2s)
        self.line_raw1, = self.ax_raw.plot([], [], label=self.eeg_sim.channel_names[self.track_pair[0]])
        self.line_raw2, = self.ax_raw.plot([], [], label=self.eeg_sim.channel_names[self.track_pair[1]])
        self.ax_raw.legend()
        self.ax_raw.set_xlim(0, 2); self.ax_raw.set_ylim(-6, 6)
        self.ax_raw.set_title("Raw EEG (last 2s)")

        plt.tight_layout()

        # Animation
        self.ani = animation.FuncAnimation(self.fig, self._step, interval=self.dt, blit=False, cache_frame_data=False)

    def _step(self, _frame):
        # Generate new chunk
        hop_s = self.dt / 1000.0
        chunk = self.eeg_sim.generate_sample(hop_s)  # (C, hop)
        self.X = np.concatenate([self.X, chunk], axis=1)
        # Trim history (keep last ~4*window for safety)
        if self.X.shape[1] > 4 * self.ws:
            self.X = self.X[:, -4 * self.ws:]

        # MI matrix for current window
        self.MI = mi_matrix_hist(self.X, window_size=self.ws, bins=64)
        self.im.set_data(self.MI)
        vmax = max(1e-6, float(np.percentile(self.MI, 95)))
        self.im.set_clim(0, vmax)

        # Windowed MI time series for the tracked pair (median to reduce small-window bias)
        starts, vals = windowed_mi(self.X[self.track_pair[0]], self.X[self.track_pair[1]],
                                   window_size=self.ws, hop_size=max(1, int(self.fs * hop_s)), bins=64)
        if len(vals):
            t_s = self.X.shape[1] / self.fs
            self.ts_t.append(t_s)
            self.ts_mi.append(float(np.median(vals[-5:])))
            # keep last 20s
            while self.ts_t and self.ts_t[-1] - self.ts_t[0] > 20:
                self.ts_t.pop(0); self.ts_mi.pop(0)
            self.line_ts.set_data(self.ts_t, self.ts_mi)
            self.ax_ts.set_xlim(max(0, self.ts_t[-1] - 20), self.ts_t[-1] + 0.01)

        # Raw preview for last 2s
        L = min(self.X.shape[1], 2 * self.fs)
        tt = np.linspace(0, L / self.fs, L, endpoint=False)
        self.line_raw1.set_data(tt, self.X[self.track_pair[0], -L:])
        self.line_raw2.set_data(tt, self.X[self.track_pair[1], -L:])

        return self.im, self.line_ts, self.line_raw1, self.line_raw2

    def start(self):
        print("Starting real-time EEG MI dashboard… (close the window to exit)")
        plt.show()


# ----------------------------- Benchmark ----------------------------- #
class BenchmarkComparison:
    """Compare traditional batch MI vs ITPU streaming MI."""

    def __init__(self):
        self.itpu = ITPU()

    def simulate_traditional_batch(self, data: np.ndarray) -> Tuple[float, List[float]]:
        """
        Batch-style MI for all channel pairs over the FULL duration.
        Uses SciPy discrete MI if available, else falls back to ITPU hist MI.
        """
        C, N = data.shape
        t0 = time.perf_counter()
        mi_vals = []
        try:
            from scipy.stats import mutual_info_score
            from sklearn.preprocessing import KBinsDiscretizer
            disc = KBinsDiscretizer(n_bins=32, encode="ordinal", strategy="uniform")
            # discretize each channel once
            disc_data = []
            for i in range(C):
                xi = disc.fit_transform(data[i].reshape(-1, 1)).astype(int).ravel()
                disc_data.append(xi)
            for i in range(C):
                for j in range(i+1, C):
                    mi_vals.append(float(mutual_info_score(disc_data[i], disc_data[j])))
        except Exception:
            # fallback to ITPU histogram on the full signals
            for i in range(C):
                for j in range(i+1, C):
                    mi_vals.append(self.itpu.mutual_info(data[i], data[j], method="hist", bins=64))
        dt = time.perf_counter() - t0
        return dt, mi_vals

    def simulate_itpu_streaming(self, data: np.ndarray, window_size: int, hop_size: int) -> Tuple[float, List[np.ndarray]]:
        """Streaming MI matrices over windows (histogram MI)."""
        C, N = data.shape
        t0 = time.perf_counter()
        mats: List[np.ndarray] = []
        for s in range(0, N - window_size + 1, hop_size):
            seg = data[:, s:s + window_size]
            mats.append(mi_matrix_hist(seg, window_size=window_size, bins=64))
        dt = time.perf_counter() - t0
        return dt, mats

    def run(self, duration_s: int = 30, n_channels: int = 8, fs: int = 250) -> dict:
        sim = EEGSimulator(n_channels=n_channels, sample_rate=fs)
        data = sim.generate_sample(duration_s)  # (C, N)
        ws = 2 * fs  # 2-second windows
        hop = fs // 4  # 250 ms

        print("Running batch baseline…")
        t_batch, _ = self.simulate_traditional_batch(data)
        print("Running ITPU streaming…")
        t_stream, mats = self.simulate_itpu_streaming(data, ws, hop)

        n_windows = len(mats)
        per_window = t_stream / max(1, n_windows)
        capable = per_window < (hop / fs)

        out = dict(
            duration_s=duration_s, n_channels=n_channels, fs=fs,
            n_samples=int(duration_s * fs),
            window_size=ws, hop_size=hop, n_windows=n_windows,
            batch_time_s=round(t_batch, 4),
            stream_time_s=round(t_stream, 4),
            per_window_latency_s=round(per_window, 4),
            real_time_capable=bool(capable),
            speedup_factor=round((t_batch / t_stream) if t_stream > 0 else float("inf"), 2),
        )
        for k, v in out.items():
            print(f"{k}: {v}")
        return out


# ----------------------------- Main ----------------------------- #
def main():
    print("=== ITPU Real-time EEG MI ===")
    print("1) Real-time dashboard (interactive)")
    print("2) Performance comparison (benchmark)")
    print("3) Both")
    try:
        choice = input("Enter choice (1-3): ").strip()
    except KeyboardInterrupt:
        return

    if choice in {"1", "3"}:
        dash = RealTimeMIDashboard(EEGSimulator(n_channels=8, sample_rate=250), window_size=500, update_interval_ms=100)
        dash.start()

    if choice in {"2", "3"}:
        bench = BenchmarkComparison()
        results = bench.run(duration_s=30, n_channels=8, fs=250)
        try:
            import json
            with open("eeg_benchmark_results.json", "w") as f:
                json.dump(results, f, indent=2)
            print("Saved: eeg_benchmark_results.json")
        except Exception as e:
            print(f"Could not save results: {e}")


if __name__ == "__main__":
    main()
