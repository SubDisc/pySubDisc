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


# Set package version
from os import path
from re import match
curdir = path.abspath(path.dirname(__file__))
with open(path.join(curdir, 'VERSION')) as version_file:
    __version__ = match("^pySubDisc (?P<version>.+)", version_file.read().strip()).group('version')
