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

  def _initSearchParameters(self, *, qualityMeasure='cortana quality', searchDepth=1, minimumCoverage=2, maximumCoverageFraction=0.9, minimumSupport=0, maximumSubgroups=1000, filterSubgroups=True, minimumImprovement=0.0, maximumTime=0, searchStrategy='beam', nominalSets=False, numericOperatorSetting='normal', numericStrategy='bins', searchStrategyWidth=100, nrBins=8, nrThreads=1):
    # use inspect to avoid duplicating the argument list
    from inspect import signature
    sig = signature(self._initSearchParameters)
    for arg in sig.parameters:
      if arg == 'self':
        continue
      setattr(self, arg, locals()[arg])

    from nl.liacs.subdisc import QM
    if not isinstance(qualityMeasure, QM):
      qualityMeasure = QM.fromString(qualityMeasure)
    self.qualityMeasureMinimum = float(str(qualityMeasure.MEASURE_DEFAULT))

  def _createSearchParametersObject(self):
    from nl.liacs.subdisc import SearchParameters
    from nl.liacs.subdisc import QM, SearchStrategy, NumericOperatorSetting, NumericStrategy

    qualityMeasure = self.qualityMeasure
    if not isinstance(qualityMeasure, QM):
      qualityMeasure = QM.fromString(qualityMeasure)
    searchStrategy = self.searchStrategy
    if not isinstance(searchStrategy, SearchStrategy):
      searchStrategy = SearchStrategy.fromString(searchStrategy)
    numericOperatorSetting = self.numericOperatorSetting
    if not isinstance(numericOperatorSetting, NumericOperatorSetting):
      # The string values of NumericOperatorSetting are in HTML (to encode symbols for greater-equal, etc.).
      # Instead, we look up the names of enum constants. The only attributes
      # of the class that are full caps are the enum constants.
      if hasattr(NumericOperatorSetting, numericOperatorSetting.upper()):
        numericOperatorSetting = getattr(NumericOperatorSetting, numericOperatorSetting.upper())
      else:
        raise ValueError("Invalid numericOperatorSetting")
    numericStrategy = self.numericStrategy
    if not isinstance(numericStrategy, NumericStrategy):
      numericStrategy = NumericStrategy.fromString(numericStrategy)


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

  def getSearchParameterDescription(self):
    sp = self._createSearchParametersObject()
    return str(self._targetConcept) + str(sp)

  def computeThreshold(self, *, significanceLevel=0.05, method='swap-randomization', amount=100, setAsMinimum=False, verbose=False):
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

  def getModel(self, index, includeBase=True, relative=True, verbose=True):
    if index is None:
      # small hack to allow being called by getBaseModel() pre-run
      index = []
      sd = None
    else:
      self._ensurePostRun()
      sd = self._sd

    from nl.liacs.subdisc import TargetType
    if self._targetConcept.getTargetType() == TargetType.SINGLE_NUMERIC:
      return redirectSystemOutErr(getModelSingleNumeric, self._targetConcept, sd, index, includeBase=includeBase, relative=relative, verbose=verbose)
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



def computeThreshold(sp, targetConcept, table, *, significanceLevel=0.05, method='swap-randomization', amount=100, setAsMinimum=False):
    from nl.liacs.subdisc import TargetType, QualityMeasure, Validation, NormalDistribution
    from nl.liacs.subdisc.gui import RandomQualitiesWindow
    import scipy.stats

    methods = [ RandomQualitiesWindow.RANDOM_DESCRIPTIONS,
                RandomQualitiesWindow.RANDOM_SUBSETS,
                RandomQualitiesWindow.SWAP_RANDOMIZATION ]
    try:
      method = next( str(m.toString()) for m in methods if str(m.toString()).lower() == method.lower() )
    except StopIteration:
      method = None
    if method is None:
      raise ValueError("Invalid method. Options are: " + ", ".join(str(m.toString()).lower() for m in methods))

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

