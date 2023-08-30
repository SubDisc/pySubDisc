from .interface import singleNominalTarget, singleNumericTarget, doubleRegressionTarget

__all__ = [ singleNominalTarget, singleNumericTarget, doubleRegressionTarget ]


# Set package version
from os import path
curdir = path.abspath(path.dirname(__file__))
with open(path.join(curdir, 'VERSION')) as version_file:
    __version__ = version_file.read().strip()
