from .version import __version__

from .interface import singleNominalTarget
from .interface import singleNumericTarget
from .interface import doubleRegressionTarget
from .interface import doubleBinaryTarget
from .interface import doubleCorrelationTarget
from .interface import multiNumericTarget
from .interface import loadDataFrame

__all__ = [ singleNominalTarget,
            singleNumericTarget,
            doubleRegressionTarget,
            doubleBinaryTarget,
            doubleCorrelationTarget,
            multiNumericTarget,
            loadDataFrame ]
