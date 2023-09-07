from .interface import singleNominalTarget
from .interface import singleNumericTarget
from .interface import doubleRegressionTarget
from .interface import doubleBinaryTarget

__all__ = [ singleNominalTarget,
            singleNumericTarget,
            doubleRegressionTarget,
            doubleBinaryTarget ]


# Set package version
from os import path
curdir = path.abspath(path.dirname(__file__))
with open(path.join(curdir, 'VERSION')) as version_file:
    __version__ = version_file.read().strip()
