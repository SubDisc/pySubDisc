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

def runSubgroupDiscovery(data, targetType, targetColumn, *, targetValue=None, qualityMeasure='cortana quality', qualityMeasureMinimum=0.1, searchDepth=1, minimumCoverage=2, maximumCoverageFraction=1.0, minimumSupport=0, maximumSubgroups=1000, filterSubgroups=True, minimumImprovement=0.0, maximumTime=1000, searchStrategy='beam', nominalSets=False, numericOperatorSetting='normal', numericStrategy='best', searchStrategyWidth=10, nrBins=8, nrThreads=1, verbose=True):

  ensureJVMStarted()

  from nl.liacs.subdisc import TargetConcept
  from nl.liacs.subdisc import TargetType
  from nl.liacs.subdisc import SearchParameters
  from nl.liacs.subdisc import QM
  from nl.liacs.subdisc import SearchStrategy
  from nl.liacs.subdisc import NumericOperatorSetting
  from nl.liacs.subdisc import NumericStrategy
  from nl.liacs.subdisc import Table

  if not isinstance(targetType, TargetType):
    targetType = TargetType.fromString(targetType)
  if not isinstance(qualityMeasure, QM):
    qualityMeasure = QM.fromString(qualityMeasure)
  if not isinstance(searchStrategy, SearchStrategy):
    searchStrategy = SearchStrategy.fromString(searchStrategy)
  if not isinstance(numericOperatorSetting, NumericOperatorSetting):
    # The string values of NumericOperatorSetting are in HTML (to encode symbols for greater-equal, etc.).
    # Instead, we look up the names of enum constants. The only attributes
    # of the class that are full caps are the enum constants.
    if hasattr(NumericOperatorSetting, numericOperatorSetting.upper()):
      numericOperatorSetting = getattr(NumericOperatorSetting, numericOperatorSetting.upper())
    else:
      raise ValueError("Invalid numericOperatorSetting")
  if not isinstance(numericStrategy, NumericStrategy):
    numericStrategy = NumericStrategy.fromString(numericStrategy)

  if not isinstance(data, Table):
    index = data.index
    data = createTableFromDataFrame(data)
  else:
    index = pd.RangeIndex(data.getNrRows())

  # can use column index or column name
  target = data.getColumn(targetColumn)

  targetConcept = TargetConcept()
  targetConcept.setTargetType(targetType)
  targetConcept.setPrimaryTarget(target)
  if targetValue is not None:
    targetConcept.setTargetValue(targetValue)

  sp = SearchParameters()
  sp.setTargetConcept(targetConcept)
  sp.setQualityMeasure(qualityMeasure)
  sp.setQualityMeasureMinimum(qualityMeasureMinimum)
  sp.setSearchDepth(searchDepth)
  sp.setMinimumCoverage(minimumCoverage)
  sp.setMaximumCoverageFraction(maximumCoverageFraction)
  sp.setMaximumSubgroups(maximumSubgroups)
  sp.setSearchStrategy(searchStrategy)
  sp.setNominalSets(nominalSets)
  sp.setNumericOperators(numericOperatorSetting)
  sp.setNumericStrategy(numericStrategy)
  sp.setSearchStrategyWidth(searchStrategyWidth)
  sp.setNrBins(nrBins)
  sp.setNrThreads(nrThreads)

  sd = _runSubgroupDiscovery(data, 0, None, sp, False, 1, None, verbose=verbose)

  return Result(sd, index, targetType)

def _runSubgroupDiscovery(*args, verbose=True):
  from nl.liacs.subdisc import Process

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

  sd = Process.runSubgroupDiscovery(*args)

  if not verbose:
    System.out.flush()
    System.err.flush()
    System.setOut(oldOut)
    System.setErr(oldErr)

  return sd


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

