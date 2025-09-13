"""
WS1 Phenomenology Tools
------------------------
Helper functions for coding geometric phenomenology transcripts, computing
inter-rater reliability (Cohen’s κ, ICC), and generating reliability reports.

Author: Geometric Experience
"""

import pandas as pd
from sklearn.metrics import cohen_kappa_score
import numpy as np

def compute_kappa(rater1, rater2):
    """Compute Cohen’s κ between two raters.

    Args:
        rater1 (list/array): classifications by rater 1
        rater2 (list/array): classifications by rater 2

    Returns:
        float: Cohen’s κ
    """
    return cohen_kappa_score(rater1, rater2)


def compute_icc(data, raters=2):
    """Compute ICC (Intraclass Correlation Coefficient) for continuous ratings.

    Args:
        data (pd.DataFrame): rows = items, cols = raters
        raters (int): number of raters

    Returns:
        float: ICC estimate
    """
    # Shrout & Fleiss ICC(2,1)
    n, k = data.shape
    mean_ratings = data.mean(axis=1)
    grand_mean = data.values.flatten().mean()

    # Between targets
    ss_between = k * ((mean_ratings - grand_mean) ** 2).sum()
    ms_between = ss_between / (n - 1)

    # Between raters
    ss_rater = n * ((data.mean(axis=0) - grand_mean) ** 2).sum()
    ms_rater = ss_rater / (k - 1)

    # Residual
    ss_total = ((data - grand_mean) ** 2).sum().sum()
    ss_error = ss_total - ss_between - ss_rater
    ms_error = ss_error / ((n - 1) * (k - 1))

    icc = (ms_between - ms_error) / (ms_between + (k - 1) * ms_error + (k * (ms_rater - ms_error) / n))
    return icc


def reliability_report(rater1, rater2, continuous_data=None):
    """Generate reliability report for categorical and continuous ratings.

    Args:
        rater1 (list): categorical codes from rater 1
        rater2 (list): categorical codes from rater 2
        continuous_data (pd.DataFrame): optional, continuous scores

    Returns:
        dict: reliability metrics
    """
    results = {}
    results['kappa'] = compute_kappa(rater1, rater2)
    if continuous_data is not None:
        results['icc'] = compute_icc(continuous_data)
    return results


if __name__ == "__main__":
    # Example usage
    r1 = ["lattice", "spiral", "tunnel", "spiral"]
    r2 = ["lattice", "spiral", "tunnel", "lattice"]
    print("Cohen’s κ:", compute_kappa(r1, r2))

    df = pd.DataFrame({
        "rater1": [7, 5, 9, 6],
        "rater2": [8, 6, 8, 5]
    })
    print("ICC:", compute_icc(df))
