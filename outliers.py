import pandas as pd
from scipy import stats
import numpy as np

def remove_outliers():
    data = pd.read_csv('./preprocessed/encoded_dataset.csv')

    z_scores = np.abs(stats.zscore(data))
    threshold = 5

    outliers = (z_scores > threshold).any(axis=1)

    data_cleaned = data[~outliers]

    data_cleaned.to_csv('./preprocessed/cleaned_dataset.csv', index=False)

    print(data_cleaned.head())