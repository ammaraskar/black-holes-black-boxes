import classifier
import pandas as pd
import pytest


dragoon_data = pd.DataFrame({
    'Cluster Name':                    ['D1-R7-IMF93', 'D2-R7-IMF01', 'D3-R7-ROT'],
    'Observational Half-Light Radius': [8.7,           14.4,           13.4],
    'Central Surface Brightness':      [2.5e2,         0.7e2,          1.3e2],
    'Central Velocity Dispersion':     [4.5,           3.8,            4.0],
    'Total V-band luminosity':         [1.86e5,        1.11e5,         1.22e5],
    'Observational Core Radius':       [4.8,           10.0,           11.9]
})


def test_make_classifier():
    clf = classifier.make_classifier()
    assert clf is not None

    predictions = classifier.predict(clf, dragoon_data)
    assert predictions[0] == True
    assert predictions[1] == True

def test_classifier_with_mass_fallback():
    clf = classifier.make_classifier(fallback_enabled=True)
    assert clf is not None

    predictions = classifier.predict(clf, dragoon_data)
    assert predictions[0] == True
    assert predictions[1] == True

def test_classifier_no_estimate():
    clf = classifier.make_classifier(use_relaxation_time_estimate=False)
    assert clf is not None

    # Should raise since dragoon_data lacks Half-Mass Relaxation Time
    with pytest.raises(KeyError):
        predictions = classifier.predict(clf, dragoon_data)

    # Let's add some fake values and make sure it works
    dragoon_copy = dragoon_data.copy()
    dragoon_copy['Half-Mass Relaxation Time'] = 10000

    predictions = classifier.predict(clf, dragoon_copy)
    assert len(predictions) == 3