# Tests that SubDisc behaviour matches pySubDisc expectations

import pytest
import pysubdisc

def test_JVM():
  pysubdisc.java.ensureJVMStarted()

def test_classes():
  # Test (a non-exhaustive set of) imports
  pysubdisc.java.ensureJVMStarted()
  from nl.liacs.subdisc import Table, Process, TargetType, TargetConcept
  from nl.liacs.subdisc import ProbabilityDensityFunction2, Validation, NormalDistribution, QualityMeasure
  from nl.liacs.subdisc.gui import ResultTableModel, RandomQualitiesWindow

def test_enums():
  # Test string versions of defaults for enums
  pysubdisc.java.ensureJVMStarted()
  from nl.liacs.subdisc import TargetType, QM, SearchStrategy, NumericOperatorSetting, NumericStrategy
  from nl.liacs.subdisc.gui import RandomQualitiesWindow
  assert str(RandomQualitiesWindow.SWAP_RANDOMIZATION).lower() ==  'swap-randomization'
  assert TargetType.SINGLE_NOMINAL.toString() == 'single nominal'
  assert TargetType.SINGLE_NUMERIC.toString() == 'single numeric'
  assert str(QM.CORTANA_QUALITY.toString()).lower() == 'cortana quality'
  assert str(QM.Z_SCORE.toString()).lower() == 'z-score'
  assert str(QM.REGRESSION_SSD_COMPLEMENT.toString()).lower() == 'sign. of slope diff. (complement)'
  assert str(SearchStrategy.BEAM.toString()).lower() == 'beam'
  assert str(NumericStrategy.NUMERIC_BINS.toString()).lower() == 'bins'
  x = NumericOperatorSetting.NORMAL 
  
def test_ResultTableModel():
  # Test that ResultTableModel columns 4 and 5 correspond to secondary/tertiary
  # result statistics
  pysubdisc.java.ensureJVMStarted()
  from nl.liacs.subdisc import TargetType
  from nl.liacs.subdisc.gui import ResultTableModel
  assert ResultTableModel.getColumnName(4, TargetType.SINGLE_NOMINAL) == 'Target Share'
  assert ResultTableModel.getColumnName(5, TargetType.SINGLE_NOMINAL) == 'Positives'

def test_pdf2():
  from nl.liacs.subdisc import ProbabilityDensityFunction
  assert ProbabilityDensityFunction.USE_ProbabilityDensityFunction2
