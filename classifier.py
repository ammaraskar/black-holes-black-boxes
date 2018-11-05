import pandas as pd
import numpy as np
from xgboost import XGBClassifier


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


def make_classifier(use_relaxation_time_estimate=True, fallback_enabled=False):
    training_data = data.copy()

    if use_relaxation_time_estimate:
        training_data['Half-Mass Relaxation Time'] = training_data['Median Relaxation Time']

    X = training_data
    if fallback_enabled:
        X = X.loc[X['Fallback Enabled'] == 1]

    y = X['BHS']
    X = X.drop('BHS', axis=1)[fields]

    clf = XGBClassifier(presort=True, random_state=4718913,
                    subsample=1.0, n_estimators=850, max_depth=6, colsample_bytree=1.0, 
                    eval_metric='map', n_jobs=-1, learning_rate=0.3)
    clf.use_relaxation_time_estimate = use_relaxation_time_estimate
    
    return clf.fit(X, y)


def predict(clf, X):
    X = X.copy()

    if clf.use_relaxation_time_estimate:
        estimate_median_relaxation_time(X)
        X['Half-Mass Relaxation Time'] = X['Median Relaxation Time']

    print(X)

    return clf.predict(X[fields])