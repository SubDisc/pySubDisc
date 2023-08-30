# Tests for the implemented target types

import pytest
import pysubdisc
import pandas
import io
import numpy as np

@pytest.fixture
def adult_data():
  return pandas.read_csv('tests/adult.txt')

def test_compute_threshold(adult_data):
  sd = pysubdisc.singleNominalTarget(adult_data, 'target', 'gr50K')
  # We need to use a relatively high amount to get higher precision
  t = sd.computeThreshold(amount=10000, verbose=False)
  assert t == pytest.approx(0.1004, abs=0.001)


def test_getmodel(adult_data):
  sd = pysubdisc.singleNumericTarget(adult_data, 'age')
  sd.run(verbose=False)
  df = sd.getModel(0, verbose=False)
  assert df['base'].mean() == pytest.approx(0.010283, abs=1e-6)
  assert df['base'].std() == pytest.approx(0.0104838, abs=1e-6)
  assert df[0].mean() == pytest.approx(0.003866, abs=1e-6)
  assert df[0].std() == pytest.approx(0.004420, abs=1e-6)

  df = sd.getModel([0, 1], verbose=False)
  assert df['base'].mean() == pytest.approx(0.010283, abs=1e-6)
  assert df['base'].std() == pytest.approx(0.0104838, abs=1e-6)
  assert df[0].mean() == pytest.approx(0.003866, abs=1e-6)
  assert df[0].std() == pytest.approx(0.004420, abs=1e-6)
  assert df[1].mean() == pytest.approx(0.004555, abs=1e-6)
  assert df[1].std() == pytest.approx(0.005234, abs=1e-6)