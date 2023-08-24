import jpype
from .java import ensureJVMStarted

class Result(object):
  def __init__(self, result, index, targetType):
    # We store the index to be able to return subgroup members of a bool Series.
    # We store the targetType since it affects the column names for the results.
    # We convert the SubgroupSet (a Java TreeSet) into a list to better support the operations of our Result class.
    self.result = result
    self.subgroups = result.getResult()
    self.index = index
    self.targetType = targetType
  def asDataFrame(self):
    return generateResultDataFrame(self.result, self.targetType)
  def getSubgroupMembers(self, index):
    import pandas
    members = self.subgroups[index].getMembers()
    return pandas.Series(map(members.get, range(self.index.size)), index=self.index)

class SubgroupDiscovery(object):
  def __init__(self, targetConcept, data, index):
    ensureJVMStarted()
    self._targetConcept = targetConcept
    self._table = data
    self._index = index

  @property
  def targetType(self):
    return self._targetConcept.getTargetType().toString()

  def _initSearchParameters(self, *, qualityMeasure='cortana quality', qualityMeasureMinimum=0.1, searchDepth=1, minimumCoverage=2, maximumCoverageFraction=0.9, minimumSupport=0, maximumSubgroups=1000, filterSubgroups=True, minimumImprovement=0.0, maximumTime=0, searchStrategy='beam', nominalSets=False, numericOperatorSetting='normal', numericStrategy='bins', searchStrategyWidth=100, nrBins=8, nrThreads=1):
    # use inspect to avoid duplicating the argument list
    from inspect import signature
    sig = signature(self._initSearchParameters)
    for arg in sig.parameters:
      if arg == 'self':
        continue
      setattr(self, arg, locals()[arg])

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
    sp.setNrBins(self.nrBins)
    sp.setNrThreads(self.nrThreads)

    return sp

  def computeThreshold(self, *, significanceLevel=0.05, method='swap-randomization', amount=100, setAsMinimum=False, verbose=False):
    sp = self._createSearchParametersObject()

    threshold = _redirectSystemOutErr(computeThreshold, sp, self._targetConcept, self._table, significanceLevel=significanceLevel, method=method, amount=amount, verbose=verbose)

    if setAsMinimum:
      self.qualityMeasureMinimum = threshold

    return threshold

  def run(self, verbose=True):
    sp = self._createSearchParametersObject()
    # TODO: check functionality of nrThreads via sp.setNrThreads vs as argument to runSubgroupDiscovery
    from nl.liacs.subdisc import Process
    sd = _redirectSystemOutErr(Process.runSubgroupDiscovery, self._table, 0, None, sp, False, 1, None, verbose=verbose)
    return Result(sd, self._index, self._targetConcept.getTargetType())

# TODO: Reduce code duplication between these factory functions
def singleNominalTarget(data, targetColumn, targetValue):
  ensureJVMStarted()

  from nl.liacs.subdisc import TargetConcept, TargetType, Table
  from math import ceil

  if not isinstance(data, Table):
    index = data.index
    data = createTableFromDataFrame(data)
  else:
    index = pd.RangeIndex(data.getNrRows())

  targetType = TargetType.SINGLE_NOMINAL

  # can use column index or column name
  target = data.getColumn(targetColumn)

  targetConcept = TargetConcept()
  targetConcept.setTargetType(targetType)
  targetConcept.setPrimaryTarget(target)
  targetConcept.setTargetValue(targetValue)

  sd = SubgroupDiscovery(targetConcept, data, index)

  sd._initSearchParameters(minimumCoverage = ceil(0.1 * data.getNrRows()))

  return sd

def singleNumericTarget(data, targetColumn):
  ensureJVMStarted()

  from nl.liacs.subdisc import TargetConcept, TargetType, Table
  from math import ceil

  if not isinstance(data, Table):
    index = data.index
    data = createTableFromDataFrame(data)
  else:
    index = pd.RangeIndex(data.getNrRows())

  targetType = TargetType.SINGLE_NUMERIC

  # can use column index or column name
  target = data.getColumn(targetColumn)

  targetConcept = TargetConcept()
  targetConcept.setTargetType(targetType)
  targetConcept.setPrimaryTarget(target)

  sd = SubgroupDiscovery(targetConcept, data, index)

  sd._initSearchParameters(qualityMeasure = 'z-score', minimumCoverage = ceil(0.1 * data.getNrRows()))

  return sd


def _redirectSystemOutErr(f, *args, verbose=True, **kwargs):
  from java.lang import System
  from java.io import PrintStream, File
  # TODO: Consider capturing output to return to caller
  #from java.io import ByteArrayOutputStream

  if not verbose:
    import os
    oldOut = System.out
    oldErr = System.err
    System.out.flush()
    System.err.flush()
    System.setOut(PrintStream(File(os.devnull)))
    System.setErr(PrintStream(File(os.devnull)))

  ret = f(*args, **kwargs)

  if not verbose:
    System.out.flush()
    System.err.flush()
    System.setOut(oldOut)
    System.setErr(oldErr)

  return ret

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

def generateResultDataFrame(sd, targetType):
  import pandas as pd

  L = [ [ r.getDepth(), r.getCoverage(), r.getMeasureValue(), r.getSecondaryStatistic(), r.getTertiaryStatistic(), r.getPValue(), str(r) ] for r in sd.getResult() ]

  # TODO: Would be nice to have these names available outside of the gui module
  try:
    from nl.liacs.subdics.gui import ResultTableModel
    rtm = ResultTableModel(sd, targetType)
  except:
    rtm = None

  if rtm is not None:
    secondaryName = rtm.getColumnName(4)
    tertiaryName = rtm.getColumnName(5)
  else:
    from nl.liacs.subdisc import TargetType
    name2Dict = {
      TargetType.SINGLE_NOMINAL: 'Target Share',
      TargetType.SINGLE_NUMERIC: 'Average',
      # TODO
    }
    name3Dict = {
      TargetType.SINGLE_NOMINAL: 'Positives',
      TargetType.SINGLE_NUMERIC: 'St. Dev.',
      # TODO
    }
    secondaryName = name2Dict[targetType]
    tertiaryName = name3Dict[targetType]

  df = pd.DataFrame(L, columns=['Depth', 'Coverage', 'Quality', secondaryName, tertiaryName, 'p-Value', 'Conditions'], copy=True)
  return df

def createTableFromDataFrame(data):
  """Create subdisc Table from pandas DataFrame.
  """
  # TODO: Consider adding a 'dtype' keyword arg to override data types (cf pd.read_csv)
  # TODO: Also, consider an option to convert integer columns to nominals (sklearn data sets use this) (pd.api.types.is_integer_dtype)
  from nl.liacs.subdisc import Column
  from nl.liacs.subdisc import AttributeType
  from nl.liacs.subdisc import Table
  from java.io import File
  import pandas as pd

  dummyfile = File('pandas.DataFrame')
  nrows, ncols = data.shape
  table = Table(dummyfile, nrows, ncols)
  columns = table.getColumns()
  index = pd.RangeIndex(nrows)

  for i, name in enumerate(data.columns):
    #print(i, name, data.dtypes[name], pd.api.types.is_numeric_dtype(data.dtypes[name]), pd.api.types.is_string_dtype(data.dtypes[name]), pd.api.types.is_bool_dtype(data.dtypes[name]))
    if pd.api.types.is_string_dtype(data.dtypes[name]):
      atype = AttributeType.NOMINAL
      ctype = str
    elif pd.api.types.is_bool_dtype(data.dtypes[name]):
      atype = AttributeType.BINARY
      ctype = bool
    elif pd.api.types.is_numeric_dtype(data.dtypes[name]):
      atype = AttributeType.NUMERIC
      ctype = float
    else:
      raise ValueError(f"""Unsupported column type '{data.dtypes[name]}' for column '{name}'""")
    column = Column(name, name, atype, i, nrows)
    column.setData(data[name].set_axis(index).astype(ctype))
    columns.add(column)

  table.update()
  return table

