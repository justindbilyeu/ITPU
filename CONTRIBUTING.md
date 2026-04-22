# Contributing to ITPU

## Read This First

This repository operates under a research-grade collaboration standard. Before opening an issue, submitting a PR, or proposing a change, read and internalize the following.

We do not merge code that produces numbers.
We merge code that produces credible numbers.
That distinction is the entire point of this project.

---

## The Standard

ITPU applies the Research Assistant Charter to all contributions. The charter governs how ideas become testable, how tests become valid, and how results become credible. It is not a style guide. It is the epistemological foundation of the project.

The constitutional principle:

> No Claims Without Tests.
> No Tests Without Thresholds.
> No Thresholds Without Numbers.
> No Numbers Without a Run.

If your contribution cannot satisfy these four conditions, it is not ready.

---

## Hard Rules

These are non-negotiable. They apply to every contributor — human or AI.

**1. Tests before implementation.**
Write the test specification before writing any code. The test defines what correct behavior looks like. Implementation validates against it. This order is not reversed for any reason.

**2. Thresholds locked before running.**
Numeric thresholds in tests are set before results are seen. They are never adjusted after seeing results. Ever. A threshold that moves after a run is not a threshold — it is a rationalization.

**3. Calibration required for statistical claims.**
Any contribution touching the surrogate testing framework requires a calibration check — not just a smoke test. The distinction: a smoke test confirms the code runs. A calibration check confirms the code is correct. Both are required. Only the calibration check closes the issue.

**4. Failures are artifacts.**
Document failures with the same rigor as successes. A test that fails and reveals a real bug is more valuable than ten tests that pass. Negative results are data, not embarrassments. The calibration failure at KS p=0.0137 that exposed the SDK's shadow KSG implementation is in the decision log because it belongs there.

**5. No silent failures.**
If your code produces a plausible-looking result under conditions where the result is not trustworthy, it must warn. Clamping to zero without a warning is not acceptable. Degrading silently in high dimensions is not acceptable. The user must know when to distrust the output.

**6. Mechanism before optimization.**
Do not optimize what you cannot yet explain. Do not tune an estimator before you have a validity test for it. Do not spec hardware before the software stack is calibrated. The sequence matters.

---

## Collaboration Model

ITPU welcomes contributions from human researchers, AI systems, and human-AI teams. The standards above apply equally to all contributors. An AI system that generates plausible code without meeting the gates is not a contributor — it is a liability.

What productive AI collaboration looks like in this repository:
- Tests written and reviewed by humans before implementation is delegated
- Thresholds set in conversation, documented in the decision log, locked before any run
- Calibration checks run and reported with actual numbers, not just pass/fail
- Failures surfaced immediately and diagnosed before proceeding

What it does not look like:
- Generating implementations and tests in the same task
- Adjusting thresholds to make failing tests pass
- Treating narrative coherence as evidence of correctness
- Closing issues before the calibration gate is satisfied

---

## Before You Open a PR

Ask yourself:
1. Did I write the tests before the implementation?
2. Are my thresholds set and documented before I ran anything?
3. If this touches statistical machinery — does it include a calibration check, not just a smoke test?
4. Have I documented any failures or unexpected results in the PR description?
5. Does my code warn when its output should not be trusted?

If the answer to any of these is no, the PR is not ready.

---

## Decision Log

Every significant technical decision in this project is recorded in the collaboration document with rationale and date. When you make a decision that affects the library's behavior, correctness, or interface — document it. Future contributors need to know not just what was decided but why, and what was considered and rejected.

---

## The Failure That Earned This Standard

During R1 development, the surrogate testing calibration check failed with KS p=0.0137 — well below the locked threshold of 0.05. Investigation revealed two bugs: the SDK contained a shadow KSG implementation using Euclidean distance instead of Chebyshev, and a clipping operation was corrupting the null distribution comparison. Neither bug was detectable by the H₁ power test alone — a correlation of r=0.6 is strong enough to survive both. Only the calibration check, with its threshold set before the run, exposed them.

This is why the standard exists. Compelling results are not evidence of correctness. Calibration is.

---

## Getting Started

1. Read the collaboration document — full technical context, decision log, known issues
2. Read the estimator guide — when to use histogram MI, KSG, and which surrogate type
3. Check open issues — R1 items are tagged `r1-blocker`, R2 items are tagged `r2`
4. Open a discussion before starting significant work — alignment before implementation

---

*CONTRIBUTING.md — ITPU*
*Standard: Research Assistant Charter v2.0*
*Effective: April 2026*
