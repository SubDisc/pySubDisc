from .java import ensureJVMStarted
from .core import SubgroupDiscovery

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

  sd._initSearchParameters(qualityMeasure = 'CORTANA_QUALITY', minimumCoverage = ceil(0.1 * data.getNrRows()))

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

  sd._initSearchParameters(qualityMeasure = 'Z_SCORE', minimumCoverage = ceil(0.1 * data.getNrRows()))

  return sd

def doubleRegressionTarget(data, primaryTargetColumn, secondaryTargetColumn):
  ensureJVMStarted()

  from nl.liacs.subdisc import TargetConcept, TargetType, Table
  from math import ceil

  if not isinstance(data, Table):
    index = data.index
    data = createTableFromDataFrame(data)
  else:
    index = pd.RangeIndex(data.getNrRows())

  targetType = TargetType.DOUBLE_REGRESSION

  # can use column index or column name
  primaryTarget = data.getColumn(primaryTargetColumn)
  secondaryTarget = data.getColumn(secondaryTargetColumn)

  targetConcept = TargetConcept()
  targetConcept.setTargetType(targetType)
  targetConcept.setPrimaryTarget(primaryTarget)
  targetConcept.setSecondaryTarget(secondaryTarget)

  sd = SubgroupDiscovery(targetConcept, data, index)

  sd._initSearchParameters(qualityMeasure = 'REGRESSION_SSD_COMPLEMENT', minimumCoverage = ceil(0.1 * data.getNrRows()))

  return sd

def doubleBinaryTarget(data, primaryTargetColumn, secondaryTargetColumn):
  ensureJVMStarted()

  from nl.liacs.subdisc import TargetConcept, TargetType, Table
  from math import ceil

  if not isinstance(data, Table):
    index = data.index
    data = createTableFromDataFrame(data)
  else:
    index = pd.RangeIndex(data.getNrRows())

  targetType = TargetType.DOUBLE_BINARY

  # can use column index or column name
  primaryTarget = data.getColumn(primaryTargetColumn)
  secondaryTarget = data.getColumn(secondaryTargetColumn)

  targetConcept = TargetConcept()
  targetConcept.setTargetType(targetType)
  targetConcept.setPrimaryTarget(primaryTarget)
  targetConcept.setSecondaryTarget(secondaryTarget)

  sd = SubgroupDiscovery(targetConcept, data, index)

  sd._initSearchParameters(qualityMeasure = 'RELATIVE_WRACC', minimumCoverage = ceil(0.1 * data.getNrRows()))

  return sd
