from .java import ensureJVMStarted, redirectSystemOutErr

class SubgroupDiscovery(object):
  def __init__(self, targetConcept, data, index):
    ensureJVMStarted()
    self._targetConcept = targetConcept
    self._table = data
    self._index = index
    self._runCalled = False

  @property
  def targetType(self):
    return self._targetConcept.getTargetType().toString()

  def _initSearchParameters(self, *, qualityMeasure='CORTANA_QUALITY', searchDepth=1, minimumCoverage=2, maximumCoverageFraction=0.9, minimumSupport=0, maximumSubgroups=1000, filterSubgroups=True, minimumImprovement=0.0, maximumTime=0, searchStrategy='BEAM', nominalSets=False, numericOperatorSetting='NORMAL', numericStrategy='NUMERIC_BINS', searchStrategyWidth=100, nrBins=8, nrThreads=1):
    # use inspect to avoid duplicating the argument list
    from inspect import signature
    sig = signature(self._initSearchParameters)
    for arg in sig.parameters:
      if arg == 'self':
        continue
      setattr(self, arg, locals()[arg])

    from nl.liacs.subdisc import QM
    if not isinstance(qualityMeasure, QM):
      if hasattr(QM, qualityMeasure.upper()):
        qualityMeasure = getattr(QM, qualityMeasure.upper())
      else:
        raise ValueError("Invalid qualityMeasure")
    self.qualityMeasureMinimum = float(str(qualityMeasure.MEASURE_DEFAULT))

  def _createSearchParametersObject(self):
    from nl.liacs.subdisc import SearchParameters
    from nl.liacs.subdisc import QM, SearchStrategy, NumericOperatorSetting, NumericStrategy

    qualityMeasure = self.qualityMeasure
    if not isinstance(qualityMeasure, QM):
      if hasattr(QM, qualityMeasure.upper()):
        qualityMeasure = getattr(QM, qualityMeasure.upper())
      else:
        raise ValueError("Invalid qualityMeasure")
    searchStrategy = self.searchStrategy
    if not isinstance(searchStrategy, SearchStrategy):
      if hasattr(SearchStrategy, searchStrategy.upper()):
        searchStrategy = getattr(SearchStrategy, searchStrategy.upper())
      else:
        raise ValueError("Invalid searchStrategy")
    numericOperatorSetting = self.numericOperatorSetting
    if not isinstance(numericOperatorSetting, NumericOperatorSetting):
      if hasattr(NumericOperatorSetting, numericOperatorSetting.upper()):
        numericOperatorSetting = getattr(NumericOperatorSetting, numericOperatorSetting.upper())
      else:
        raise ValueError("Invalid numericOperatorSetting")
    numericStrategy = self.numericStrategy
    if not isinstance(numericStrategy, NumericStrategy):
      if hasattr(NumericStrategy, numericStrategy.upper()):
        numericStrategy = getattr(NumericStrategy, numericStrategy.upper())
      else:
        raise ValueError("Invalid numericStrategy")


    sp = SearchParameters()
    sp.setTargetConcept(self._targetConcept)
    sp.setQualityMeasure(qualityMeasure)
    sp.setQualityMeasureMinimum(self.qualityMeasureMinimum)
    sp.setSearchDepth(self.searchDepth)
    sp.setMinimumCoverage(self.minimumCoverage)
    sp.setMaximumCoverageFraction(self.maximumCoverageFraction)
    sp.setMaximumSubgroups(self.maximumSubgroups)
    sp.setSearchStrategy(searchStrategy)
    sp.setNominalSets(self.nominalSets)
    sp.setNumericOperators(numericOperatorSetting)
    sp.setNumericStrategy(numericStrategy)
    sp.setSearchStrategyWidth(self.searchStrategyWidth)
    sp.setFilterSubgroups(self.filterSubgroups)
    sp.setNrBins(self.nrBins)
    sp.setNrThreads(self.nrThreads)

    return sp

  def _setColumnType(self, columns, t):
    if isinstance(columns, str) or not hasattr(columns, '__iter__'):
      columns = [ columns ]
    for c in columns:
      self._table.getColumn(c).setType(t)

  def makeColumnsBinary(self, columns):
    from nl.liacs.subdisc import AttributeType
    self._setColumnType(columns, AttributeType.BINARY)

  def makeColumnsNominal(self, columns):
    from nl.liacs.subdisc import AttributeType
    self._setColumnType(columns, AttributeType.NOMINAL)

  def makeColumnsNumeric(self, columns):
    from nl.liacs.subdisc import AttributeType
    self._setColumnType(columns, AttributeType.NUMERIC)

  def describeColumns(self):
    import pandas as pd
    L = [ [ str(column.getName()), column.getCardinality(), str(column.getType()) ] for column in self._table.getColumns() ]

    df = pd.DataFrame(L, columns=['Attribute', 'Cardinality', 'Type'])
    return df

  def getSearchParameterDescription(self):
    sp = self._createSearchParametersObject()
    return str(self._targetConcept) + str(sp)

  def computeThreshold(self, *, significanceLevel=0.05, method='SWAP_RANDOMIZATION', amount=100, setAsMinimum=False, verbose=False):
    sp = self._createSearchParametersObject()

    threshold = redirectSystemOutErr(computeThreshold, sp, self._targetConcept, self._table, significanceLevel=significanceLevel, method=method, amount=amount, verbose=verbose)

    if setAsMinimum:
      self.qualityMeasureMinimum = threshold

    return threshold

  def getBaseModel(self, verbose=True):
    return self.getModel(None, includeBase=True, verbose=verbose)

  def _ensurePostRun(self):
    if not self._runCalled:
      raise RuntimeError("This function is only available after a succesfull call of run()")

  def run(self, verbose=True):
    sp = self._createSearchParametersObject()
    # TODO: check functionality of nrThreads via sp.setNrThreads vs as argument to runSubgroupDiscovery
    from nl.liacs.subdisc import Process
    sd = redirectSystemOutErr(Process.runSubgroupDiscovery, self._table, 0, None, sp, False, self.nrThreads, None, verbose=verbose)
    self._runCalled = True
    self._sd = sd

  def asDataFrame(self):
    self._ensurePostRun()
    return generateResultDataFrame(self._sd, self._targetConcept.getTargetType())

  def getSubgroupMembers(self, index):
    self._ensurePostRun()
    import pandas
    subgroups = list(self._sd.getResult())
    members = subgroups[index].getMembers()
    return pandas.Series(map(members.get, range(self._index.size)), index=self._index)

  def getModel(self, index, includeBase=True, verbose=True, **kwargs):
    if index is None:
      # small hack to allow being called by getBaseModel() pre-run
      index = []
      sd = None
    else:
      self._ensurePostRun()
      sd = self._sd

    from nl.liacs.subdisc import TargetType
    if self._targetConcept.getTargetType() == TargetType.SINGLE_NUMERIC:
      return redirectSystemOutErr(getModelSingleNumeric, self._targetConcept, sd, index, includeBase=includeBase, verbose=verbose, **kwargs)
    if self._targetConcept.getTargetType() == TargetType.DOUBLE_REGRESSION:
      return redirectSystemOutErr(getModelDoubleRegression, self._targetConcept, sd, index, dfIndex=self._index, includeBase=includeBase, verbose=verbose, **kwargs)
    else:
      raise NotImplementedError("getModel() is not implemented for this target type")

def generateResultDataFrame(sd, targetType):
  import pandas as pd

  L = [ [ r.getDepth(), r.getCoverage(), r.getMeasureValue(), r.getSecondaryStatistic(), r.getTertiaryStatistic(), r.getPValue(), str(r) ] for r in sd.getResult() ]

  from nl.liacs.subdisc.gui import ResultTableModel
  rtm = ResultTableModel(sd.getResult(), targetType)

  secondaryName = str(rtm.getColumnName(4))
  tertiaryName = str(rtm.getColumnName(5))

  df = pd.DataFrame(L, columns=['Depth', 'Coverage', 'Quality', secondaryName, tertiaryName, 'p-Value', 'Conditions'], copy=True)
  return df



def computeThreshold(sp, targetConcept, table, *, significanceLevel, method, amount, setAsMinimum=False):
    from nl.liacs.subdisc import TargetType, QualityMeasure, Validation, NormalDistribution
    from nl.liacs.subdisc.gui import RandomQualitiesWindow
    import scipy.stats

    methods = [ 'RANDOM_DESCRIPTIONS', 'RANDOM_SUBSETS', 'SWAP_RANDOMIZATION' ]
    if method.upper() not in methods:
      raise ValueError("Invalid method. Options are: " + ", ".join(methods))
    method = getattr(RandomQualitiesWindow, method).toString()

    # Logic duplicated from java MiningWindow
    if targetConcept.getTargetType() == TargetType.SINGLE_NOMINAL:
      positiveCount = targetConcept.getPrimaryTarget().countValues(targetConcept.getTargetValue(), None)
      qualityMeasure = QualityMeasure(sp.getQualityMeasure(), table.getNrRows(), positiveCount)
    elif targetConcept.getTargetType() == TargetType.SINGLE_NUMERIC:
      from nl.liacs.subdisc import QM, Stat, ProbabilityDensityFunction2
      from java.util import BitSet
      target = targetConcept.getPrimaryTarget()
      qm = sp.getQualityMeasure()
      b = BitSet(table.getNrRows())
      b.set(0, table.getNrRows())
      statistics = target.getStatistics(None, b, qm == QM.MMAD, QM.requiredStats(qm).contains(Stat.COMPL))
      # TODO: ProbabilityDensityFunction or ProbabilityDensityFunction2?
      pdf = ProbabilityDensityFunction2(target, None)
      pdf.smooth()
      qualityMeasure = QualityMeasure(qm, table.getNrRows(),
                                      statistics.getSubgroupSum(),
                                      statistics.getSubgroupSumSquaredDeviations(),
                                      pdf)
    elif targetConcept.getTargetType() == TargetType.DOUBLE_REGRESSION:
      qualityMeasure = None
    elif targetConcept.getTargetType() == TargetType.DOUBLE_BINARY:
      qualityMeasure = None
    else:
      raise NotImplementedError()

    validation = Validation(sp, table, None, qualityMeasure)
    qualities = validation.getQualities([ method, str(amount) ])
    if qualities is None:
      # TODO: Check how to handle this
      raise RuntimeError()

    distro = NormalDistribution(qualities)

    threshold = distro.getMu() + scipy.stats.norm.ppf(1 - significanceLevel) * distro.getSigma()

    return threshold

def getModelSingleNumeric(targetConcept, sd, index, relative=True, includeBase=True):
  from nl.liacs.subdisc import TargetType, ProbabilityDensityFunction2
  from pandas import DataFrame
  import numpy as np

  assert targetConcept.getTargetType() == TargetType.SINGLE_NUMERIC
  if not hasattr(index, '__iter__'):
    index = [ index ]

  pdfBase = ProbabilityDensityFunction2(targetConcept.getPrimaryTarget(), None)
  pdfBase.smooth()
  if includeBase:
    L = [ pdfBase ]
    columns = [ 'base' ]
    scales = [ 1. ]
  else:
    L = []
    columns = []
    scales = []

  subgroups = None
  for i in index:
    if subgroups is None:
      # small hack to avoid calling getResult() if index is empty
      subgroups = list(sd.getResult())
    s = subgroups[i]
    pdfSub = ProbabilityDensityFunction2(pdfBase, s.getMembers())
    pdfSub.smooth()
    assert pdfSub.size() == pdfBase.size()
    L.append(pdfSub)
    columns.append(i)
    if relative:
      scales.append( pdfSub.getAbsoluteCount() / pdfBase.getAbsoluteCount() )
    else:
      scales.append(1.)

  rows = np.zeros((pdfBase.size(), ), dtype=float)
  for i in range(pdfBase.size()):
    rows[i] = pdfBase.getMiddle(i)

  data = np.zeros((pdfBase.size(), len(L)), dtype=float)
  for j, pdf in enumerate(L):
    for i in range(pdfBase.size()):
      data[i, j] = pdf.getDensity(i) * scales[j]

  df = DataFrame(data=data, index=rows, columns=columns, copy=True)

  return df

def getModelDoubleRegression(targetConcept, sd, index, dfIndex=None, includeBase=True):
  from nl.liacs.subdisc import TargetType, QM, RegressionMeasure
  from pandas import DataFrame
  import numpy as np

  assert targetConcept.getTargetType() == TargetType.DOUBLE_REGRESSION
  if not hasattr(index, '__iter__'):
    index = [ index ]

  # Create a dataframe with one row per sample.
  # Columns: 'x', the primary target column
  #          'y base', the secondary target column
  #          'pred base', the predicted value with the base regression model
  # and for each requested subgroup nr #:
  #          'y #', NaN if sample is not in subgroup, otherwise y value
  #          'pred #', the predicted value with the subgroup's regression model


  if includeBase:
    columns = [ 'x', 'y base', 'pred base' ]
    baseCols = 3
  else:
    columns = [ 'x' ]
    baseCols = 1

  L = []
  subgroups = None
  for i in index:
    if subgroups is None:
      # small hack to avoid calling getResult() if index is empty
      subgroups = list(sd.getResult())
    s = subgroups[i]
    L.append(s)
    columns.extend([f'y {i}', f'pred {i}'])

  xcoords = np.array(targetConcept.getPrimaryTarget().getFloats())
  nrRows = xcoords.shape[0]

  # REGRESSION_SSD_COMPLEMENT is the default here, but doesn't really matter
  RM = RegressionMeasure(QM.REGRESSION_SSD_COMPLEMENT, targetConcept.getPrimaryTarget(), targetConcept.getSecondaryTarget())
  slope = RM.getSlope()
  intercept = RM.getIntercept()

  if dfIndex is not None:
    rows = range(nrRows)
  else:
    rows = dfIndex

  data = np.zeros((nrRows, len(columns)), dtype=float)

  data[:, 0] = xcoords
  if includeBase:
    data[:, 1] = targetConcept.getSecondaryTarget().getFloats()
    data[:, 2] = intercept + slope * data[:, 0]

  for j, s in enumerate(L):
    members = s.getMembers()
    slope = s.getSecondaryStatistic()
    intercept = s.getTertiaryStatistic()
    data[:, 2*j+baseCols] = targetConcept.getSecondaryTarget().getFloats()
    data[:, 2*j+baseCols+1] = intercept + slope * data[:, 0]
    for i in range(data.shape[0]):
      if not members.get(i):
        data[i, 2*j+baseCols] = np.NaN

  df = DataFrame(data=data, index=rows, columns=columns, copy=True)

  return df

