# Datasets (for local demos)

This repo ships **offline** examples. To run them with real data, download a CSV and place it under `data/`.

## EEG Eye State (UCI, id=264) — **Recommended**

- **What:** 14 EEG channels + binary label (eyes open/closed), 14,980 rows.
- **File:** `data/eeg_eye_state.csv` (comma-separated, header optional)
- **Columns:** `ch1,...,ch14,label` (label in {0,1}); if your file has different names, adjust the demo's column selection.

**How to get it (one simple way)**  
Download the CSV from a mirror or export using your browser from the UCI site, then save as:
data/eeg_eye_state.csv

## Notes

- Keep files small(ish). We aim for quick, laptop-friendly demos.
- If you can’t download now, the example will fall back to synthetic data and still run.
