import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from functools import lru_cache


fields = ['Observational Half-Light Radius',
          'Central Surface Brightness',
          'Central Velocity Dispersion',
          'Total V-band luminosity',
          'Half-Mass Relaxation Time',
          'Observational Core Radius']

data = pd.read_csv('./simulations.csv').dropna(axis=0, how='any')
# Count up the total number of black holes
data['Total Binaries'] = data['Single Black Holes'] + 2 * (data['Binary Black Holes']) + data['Black Hole Other Count']
data['BHS'] = data['Total Binaries'] > 15

# Calculate the surface brightness within the V-Band
data['V-Band Ratio'] = data['Total V-band luminosity'] / data['Total Luminosity of Cluster at 12 Gyrs']
data['Central Surface Brightness'] = data['V-Band Ratio'] * data['Central Surface Brightness Total']

def estimate_median_relaxation_time(data):
    # Estimate the median relaxation time as done in the Harris catalogue
    data['Median Relaxation Time'] = 2.055e6 * 1 / np.log(0.4 * (2 * data['Total V-band luminosity']) / (1/3)) \
                                            * 3 * (2 * data['Total V-band luminosity'])**0.5 * data['Observational Half-Light Radius'] ** 1.5
    data['Median Relaxation Time'] = data['Median Relaxation Time'] / 10**6


estimate_median_relaxation_time(data)


@lru_cache(maxsize=32)
def make_classifier(use_relaxation_time_estimate=True,
                    train_with_central_velocity_dispersion=True,
                    fallback_enabled=False):
    """
    use_relaxation_time_estimate (bool): Use the Harris estimate for median
        relaxation time instead of the half-mass relaxation time.
        (This argument will force a different set of hyperparameters)

    train_with_central_velocity_dispersion (bool): Whether to include the
        central velocity dispersion column in the training data or not. This
        must be set to false when predicting entries with missing CVD values in
        the Harris dataset.

    fallback_enabled (bool): Set to true if the classifier should only be
        trained on simulations where mass fallback is enabled. If false, all
        simulations will be used in the training process.

    Returns: the trained classifier
    """
    training_data = data.copy()

    if use_relaxation_time_estimate:
        training_data['Half-Mass Relaxation Time'] = training_data['Median Relaxation Time']
        hyperparameters = {
            'random_state': 4718913, 'subsample': 1.0,
            'n_estimators': 850, 'max_depth': 6, 'colsample_bytree': 1.0,
            'eval_metric': 'map', 'learning_rate': 0.3
        }
    else:
        hyperparameters = {
            'random_state': 4718913, 'subsample': 0.975, 
            'n_estimators': 500, 'max_depth': 9, 'colsample_bytree': 0.9, 
            'eval_metric': 'map', 'learning_rate': 0.325
        }

    X = training_data
    if fallback_enabled:
        X = X.loc[X['Fallback Enabled'] == 1]

    y = X['BHS']
    X = X.drop('BHS', axis=1)[fields]

    if not train_with_central_velocity_dispersion:
        X = X.drop(columns=['Central Velocity Dispersion'])

    clf = XGBClassifier(presort=True, n_jobs=-1, **hyperparameters)
    clf.use_relaxation_time_estimate = use_relaxation_time_estimate
    
    return clf.fit(X, y)


def predict(clf, X):
    X = X.copy()

    if clf.use_relaxation_time_estimate:
        estimate_median_relaxation_time(X)
        X['Half-Mass Relaxation Time'] = X['Median Relaxation Time']

    return clf.predict(X[fields])