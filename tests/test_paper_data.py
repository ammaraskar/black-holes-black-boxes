import classifier
import pandas as pd
import pytest


# MAJOR CAVEAT:
# XGBoost has different random number generation across different operating
# systems. The original code was run on a Windows computer, thus this test
# checks that ~95% of the results match instead of all of them.

def arrays_almost_same(predictions, real):
    return ((predictions == real).sum() / float(len(predictions))) > 0.95

def test_holger_paper_data():
    # Load up the real holger data from the paper
    data = pd.read_csv('tests/table_holger.csv')

    paper_prediction = data['BHS']
    paper_prediction_fallback = data['BHS (Fallback)']

    data = data.drop(columns=['BHS', 'BHS (Fallback)', 'Cluster Name'])

    clf = classifier.make_classifier(use_relaxation_time_estimate=False)
    clf_fallback = classifier.make_classifier(use_relaxation_time_estimate=False, fallback_enabled=True)
    assert clf is not None
    assert clf_fallback is not None

    assert arrays_almost_same(paper_prediction.values, clf.predict(data))
    assert arrays_almost_same(paper_prediction_fallback.values, clf_fallback.predict(data))

def test_harris_paper_data():
    # Load up the real harris data from the paper
    data = pd.read_csv('tests/table_harris.csv')

    paper_prediction = data['BHS']
    paper_prediction_fallback = data['BHS (Fallback)']

    data = data.drop(columns=['BHS', 'BHS (Fallback)', 'Cluster Name'])

    clf = classifier.make_classifier()
    clf_fallback = classifier.make_classifier(fallback_enabled=True)
    assert clf is not None
    assert clf_fallback is not None

    assert arrays_almost_same(paper_prediction.values, clf.predict(data))
    assert arrays_almost_same(paper_prediction_fallback.values, clf_fallback.predict(data))

def test_harris_no_velocity_paper_data():
    # Load up the real harris data from the paper
    data = pd.read_csv('tests/table_harris_novelocity.csv')

    paper_prediction = data['BHS']
    paper_prediction_fallback = data['BHS (Fallback)']

    data = data.drop(columns=['BHS', 'BHS (Fallback)', 'Cluster Name'])

    clf = classifier.make_classifier(train_with_central_velocity_dispersion=False)
    clf_fallback = classifier.make_classifier(train_with_central_velocity_dispersion=False, fallback_enabled=True)
    assert clf is not None
    assert clf_fallback is not None

    assert arrays_almost_same(paper_prediction.values, clf.predict(data))
    assert arrays_almost_same(paper_prediction_fallback.values, clf_fallback.predict(data))