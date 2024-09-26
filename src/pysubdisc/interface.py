from .java import ensureJVMStarted
from .core import SubgroupDiscovery

def loadDataFrame(data):
  """Create subdisc Table from pandas DataFrame."""
  ensureJVMStarted()

  from nl.liacs.subdisc import Column
  from nl.liacs.subdisc import AttributeType
  from nl.liacs.subdisc import Table as sdTable
  from jpype import JArray, JString, JBoolean, JFloat
  from java.io import File
  import pandas as pd
  from .core import Table

  dummyfile = File('pandas.DataFrame')
  nrows, ncols = data.shape
  table = sdTable(dummyfile, nrows, ncols)
  columns = table.getColumns()
  index = pd.RangeIndex(nrows)

  for i, name in enumerate(data.columns):
    #print(i, name, data.dtypes[name], pd.api.types.is_numeric_dtype(data.dtypes[name]), pd.api.types.is_string_dtype(data.dtypes[name]), pd.api.types.is_bool_dtype(data.dtypes[name]))
    if pd.api.types.is_string_dtype(data.dtypes[name]):
      atype = AttributeType.NOMINAL
      ctype = str
      jtype = JString
    elif pd.api.types.is_bool_dtype(data.dtypes[name]):
      atype = AttributeType.BINARY
      ctype = bool
      jtype = JBoolean
    elif pd.api.types.is_numeric_dtype(data.dtypes[name]):
      atype = AttributeType.NUMERIC
      ctype = float
      jtype = JFloat
    else:
      raise ValueError(f"""Unsupported column type '{data.dtypes[name]}' for column '{name}'""")
    column = Column(name, name, atype, i, nrows)
    column.setData(JArray(jtype)@data[name].set_axis(index).astype(ctype))
    columns.add(column)

  table.update()

  t = Table(table, data.index)

  return t

def _createTable(data):
  from nl.liacs.subdisc import Table as sdTable
  from .core import Table

  if isinstance(data, sdTable):
    import pandas as pd
    index = pd.RangeIndex(data.getNrRows())
    table = Table(data, index)
  elif isinstance(data, Table):
    table = data
  else:
    table = loadDataFrame(data)

  return table


# TODO: Reduce code duplication between these factory functions
def singleNominalTarget(data, targetColumn, targetValue):
  """Create subdisc interface of type 'single nominal'.
     Arguments:
     data -- the data as a DataFrame
     targetColumn -- the name/index of the target (nominal) column
     targetValue -- the target value
  """
  ensureJVMStarted()

  from nl.liacs.subdisc import TargetConcept, TargetType

  table = _createTable(data)

  targetType = TargetType.SINGLE_NOMINAL

  # can use column index or column name
  target = table._table.getColumn(targetColumn)
  if target is None:
    raise ValueError(f"Unknown column '{targetColumn}'")

  targetConcept = TargetConcept()
  targetConcept.setTargetType(targetType)
  targetConcept.setPrimaryTarget(target)
  targetConcept.setTargetValue(targetValue)

  sd = SubgroupDiscovery(targetConcept, table)

  sd._initSearchParameters(qualityMeasure = 'CORTANA_QUALITY')

  sd._checkColumnTypes()

  return sd

def singleNumericTarget(data, targetColumn):
  """Create subdisc interface of type 'single numeric'.
     Arguments:
     data -- the data as a DataFrame
     targetColumn -- the name/index of the target (numeric) column
  """
  ensureJVMStarted()

  from nl.liacs.subdisc import TargetConcept, TargetType

  table = _createTable(data)

  targetType = TargetType.SINGLE_NUMERIC

  # can use column index or column name
  target = table._table.getColumn(targetColumn)
  if target is None:
    raise ValueError(f"Unknown column '{targetColumn}'")

  targetConcept = TargetConcept()
  targetConcept.setTargetType(targetType)
  targetConcept.setPrimaryTarget(target)

  sd = SubgroupDiscovery(targetConcept, table)

  sd._initSearchParameters(qualityMeasure = 'Z_SCORE')

  sd._checkColumnTypes()

  return sd

def doubleRegressionTarget(data, primaryTargetColumn, secondaryTargetColumn):
  """Create subdisc interface of type 'double regression'.
     Arguments:
     data -- the data as a DataFrame
     primaryTargetColumn -- the name/index of the primary target (numeric)
     secondaryTargetColumn -- the name/index of the secondary target (numeric)
  """
  ensureJVMStarted()

  from nl.liacs.subdisc import TargetConcept, TargetType

  table = _createTable(data)

  targetType = TargetType.DOUBLE_REGRESSION

  # can use column index or column name
  primaryTarget = table._table.getColumn(primaryTargetColumn)
  secondaryTarget = table._table.getColumn(secondaryTargetColumn)
  if primaryTarget is None:
    raise ValueError(f"Unknown column '{primaryTargetColumn}'")
  if secondaryTarget is None:
    raise ValueError(f"Unknown column '{secondaryTargetColumn}'")

  targetConcept = TargetConcept()
  targetConcept.setTargetType(targetType)
  targetConcept.setPrimaryTarget(primaryTarget)
  targetConcept.setSecondaryTarget(secondaryTarget)

  sd = SubgroupDiscovery(targetConcept, table)

  sd._initSearchParameters(qualityMeasure = 'REGRESSION_SSD_COMPLEMENT')

  sd._checkColumnTypes()

  return sd

def doubleBinaryTarget(data, primaryTargetColumn, secondaryTargetColumn):
  """Create subdisc interface of type 'double binary'.
     Arguments:
     data -- the data as a DataFrame
     primaryTargetColumn -- the name/index of the primary target (binary)
     secondaryTargetColumn -- the name/index of the secondary target (binary)
  """
  ensureJVMStarted()

  from nl.liacs.subdisc import TargetConcept, TargetType

  table = _createTable(data)

  targetType = TargetType.DOUBLE_BINARY

  # can use column index or column name
  primaryTarget = table._table.getColumn(primaryTargetColumn)
  secondaryTarget = table._table.getColumn(secondaryTargetColumn)
  if primaryTarget is None:
    raise ValueError(f"Unknown column '{primaryTargetColumn}'")
  if secondaryTarget is None:
    raise ValueError(f"Unknown column '{secondaryTargetColumn}'")

  targetConcept = TargetConcept()
  targetConcept.setTargetType(targetType)
  targetConcept.setPrimaryTarget(primaryTarget)
  targetConcept.setSecondaryTarget(secondaryTarget)

  sd = SubgroupDiscovery(targetConcept, table)

  sd._initSearchParameters(qualityMeasure = 'RELATIVE_WRACC')

  sd._checkColumnTypes()

  return sd

def doubleCorrelationTarget(data, primaryTargetColumn, secondaryTargetColumn):
  """Create subdisc interface of type 'double correlation'.
     Arguments:
     data -- the data as a DataFrame
     primaryTargetColumn -- the name/index of the primary target (numeric)
     secondaryTargetColumn -- the name/index of the secondary target (numeric)
  """
  ensureJVMStarted()

  from nl.liacs.subdisc import TargetConcept, TargetType

  table = _createTable(data)

  targetType = TargetType.DOUBLE_CORRELATION

  # can use column index or column name
  primaryTarget = table._table.getColumn(primaryTargetColumn)
  secondaryTarget = table._table.getColumn(secondaryTargetColumn)
  if primaryTarget is None:
    raise ValueError(f"Unknown column '{primaryTargetColumn}'")
  if secondaryTarget is None:
    raise ValueError(f"Unknown column '{secondaryTargetColumn}'")

  targetConcept = TargetConcept()
  targetConcept.setTargetType(targetType)
  targetConcept.setPrimaryTarget(primaryTarget)
  targetConcept.setSecondaryTarget(secondaryTarget)

  sd = SubgroupDiscovery(targetConcept, table)

  sd._initSearchParameters(qualityMeasure = 'CORRELATION_R')

  sd._checkColumnTypes()

  return sd

def multiNumericTarget(data, targetColumns):
  """Create subdisc interface of type 'multi numeric'.
     Arguments:
     data -- the data as a DataFrame
     targetColumns -- list of name/index of the target columns (numeric)
  """
  ensureJVMStarted()

  from nl.liacs.subdisc import TargetConcept, TargetType
  from java.util import ArrayList

  table = _createTable(data)

  targetType = TargetType.MULTI_NUMERIC

  L = ArrayList()
  for c in targetColumns:
    # can use column index or column name

    target = table._table.getColumn(c)
    if target is None:
      raise ValueError(f"Unknown column '{c}'")
    L.add(target)

  if L.size() < 2:
    raise ValueError("At least 2 columns must be selected")

  targetConcept = TargetConcept()
  targetConcept.setTargetType(targetType)
  targetConcept.setMultiTargets(L)

  sd = SubgroupDiscovery(targetConcept, table)

  if L.size() == 2:
    # This qualityMeasure is only available for 2D
    qm = 'SQUARED_HELLINGER_2D'
  else:
    qm = 'L2'

  sd._initSearchParameters(qualityMeasure = qm)

  sd._checkColumnTypes()

  return sd
