# WS1 Coding Manual: Geometric Pattern Classification

## Purpose
This manual provides standardized rules for classifying phenomenological reports of geometric visual experiences into invariant categories: **lattice, tunnel, spiral**. It ensures inter-rater reliability (Cohen’s κ ≥ 0.6) and reproducibility.

---

## 1. Operational Definitions

### Lattice
- **Definition:** Repeating, grid-like, tessellated patterns.
- **Keywords/Descriptors:** "net," "grid," "honeycomb," "checkerboard," "weblike."
- **Spatial Character:** Flat or tiled; no strong sense of depth or motion.

### Tunnel
- **Definition:** Radial or concentric patterns implying inward/outward depth.
- **Keywords/Descriptors:** "tunnel," "funnel," "vortex pulling inward," "concentric rings," "drawn into."
- **Spatial Character:** Sense of motion through space, expansion, or contraction.

### Spiral
- **Definition:** Rotational symmetry with twisting or pinwheel-like quality.
- **Keywords/Descriptors:** "spiral," "swirl," "vortex rotating," "pinwheel."
- **Spatial Character:** Dynamic rotation, twisting motion; often expanding or contracting.

---

## 2. Boundary Decision Rules

- **Spiral vs Tunnel:** 
  - If rotation is present → classify as Spiral.
  - If purely radial/inward/outward without rotation → classify as Tunnel.

- **Lattice vs Spiral:** 
  - If repeating regular units dominate → classify as Lattice.
  - If twisting dynamic dominates → classify as Spiral.

- **Lattice vs Tunnel:** 
  - If depth/motion dominates → Tunnel.
  - If flat grid dominates → Lattice.

- **Mixed/Edge Cases:** 
  - If equally strong features from multiple categories, mark as **Mixed** and rate each category (0–10).

---

## 3. Edge Case Handling Protocols

- **Ambiguous Language:**  
  - Raters should consult examples in this manual; if still unclear, use Mixed.  
- **Multiple Descriptors in One Report:**  
  - Assign primary category by dominance; secondary ratings record strength of others.  
- **No Clear Geometric Content:**  
  - Classify as **None**.

---

## 4. Secondary Rating Scales

For each transcript, in addition to primary category, assign **0–10 scale ratings** for:
- **Lattice-likeness** (0 = none, 10 = perfect repeating grid)
- **Tunnel-likeness** (0 = none, 10 = vivid tunnel/funnel)
- **Spiral-likeness** (0 = none, 10 = vivid spiral/vortex)

Also record:
- **Clarity** (0–10)
- **Dynamics** (0–10; degree of motion/change)
- **Immersion** (0–10; sense of being inside the pattern)

---

## 5. Inter-Rater Reliability Calculation

- **Primary Categories:**  
  - Use **Cohen’s κ** to measure inter-rater reliability.  
  - Compute κ separately for (a) Lattice/Tunnel/Spiral only, and (b) including Mixed/None.

- **Secondary Scales:**  
  - Use **Intraclass Correlation Coefficient (ICC)** to evaluate agreement.  
  - Report ICC with 95% CI.

### Example Calculation (Python, scikit-learn)
```python
from sklearn.metrics import cohen_kappa_score
import numpy as np

# Example rater classifications
rater1 = ["lattice", "spiral", "tunnel", "spiral"]
rater2 = ["lattice", "spiral", "tunnel", "lattice"]

kappa = cohen_kappa_score(rater1, rater2)
print("Cohen’s κ:", kappa)
```

---

## 6. Reliability Thresholds & Remediation

- **κ ≥ 0.6** → Proceed to WS3 analyses.
- **κ < 0.6** → Trigger remediation:
  - Review disagreements.
  - Refine coding rules with boundary examples.
  - Retrain raters with additional transcripts.
  - Repeat pilot until κ ≥ 0.6 or determine NO-GO.

---

## 7. Example Phrases by Category

- **Lattice:** "It looked like a honeycomb," "I saw endless grids," "like a checkerboard wrapping around me."
- **Tunnel:** "I felt pulled into a tunnel," "concentric rings expanding," "like a funnel spinning outward."
- **Spiral:** "A spiral rotating endlessly," "pinwheel patterns twisting," "swirling vortex."

---

## 8. Documentation
Each rater must document:
- Transcript ID
- Primary category
- Secondary scores (0–10)
- Notes on ambiguous cases
- Timestamp of coding

---

**End of WS1 Coding Manual**
