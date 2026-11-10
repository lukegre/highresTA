"""Reproduce the degree-one salinity component of a mean-aggregated ensemble."""
import numpy as np
from sklearn.linear_model import LinearRegression


def bottle_baselines(data, blocks, member_splits, row_ids):
    """Average the split-specific OLS predictions with the same weights as the members."""
    salinity = data[["salinity"]]
    sample_salinity = salinity.iloc[row_ids]
    predictions = []
    splits, counts = np.unique(member_splits, return_counts=True)
    for split in splits:
        train = (blocks != 0) & (blocks != split)
        linear = LinearRegression().fit(salinity.loc[train], data.loc[train, "talk"])
        predictions.append(linear.predict(sample_salinity))
    return np.average(predictions, weights=counts, axis=0)
