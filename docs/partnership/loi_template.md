# Letter of Intent (LOI) Template

Use this template for partner evaluations. Customize the bracketed sections for each organization.

---

**Subject:** Non-binding Letter of Intent — ITPU Pilot Evaluation

This non-binding LOI expresses **[ORGANIZATION]**'s intent to evaluate the Information-Theoretic Processing Unit (ITPU) for accelerating mutual information (MI) and entropy computations in **[BRIEF USE CASE, e.g., high-density EEG decoding]**.

## Scope

- **Dataset:** [description, e.g., 64–256-channel EEG at 1 kHz, 30–120 min sessions]
- **Workload:** Sliding-window MI (histogram, bins=128) and KSG MI (k=5–15); optional conditional MI
- **Integration:** Python SDK drop-in; no changes to existing acquisition stack

## Success Criteria (any two constitute success)

- **Throughput:** ≥100M samples/s effective on 128×128 histograms (single host) OR ≥10× faster than current toolchain on identical data
- **Latency:** KSG MI for N=10^5 samples in ≤5 ms (per pair) on prototype hardware OR ≤50 ms in software stub
- **Accuracy:** MI estimates within ±10% of JIDT/ground-truth baselines on test suite

## Commercial Intent (non-binding)

Upon meeting success criteria, **[ORGANIZATION]** intends to:
- Enter a paid pilot (USD $50k–$100k) for 8–12 weeks, OR
- Purchase an ITPU Dev Kit (USD $5k–$15k) plus annual SDK license (USD $20k–$50k), subject to procurement
- **Timeline:** Decision on pilot/purchase by **[DATE, typically 90 days post-evaluation]**

## Logistics

- **Target window:** [dates]
- **Data access:** Secure transfer or on-prem evaluation
- **Points of contact:** [names/emails]

## Legal

This LOI is non-binding and does not create exclusivity or obligations to purchase.

**Signed,**
[Name, Title]                           [Date]
[Organization]
