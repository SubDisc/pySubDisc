from .interface import singleNominalTarget, singleNumericTarget

__all__ = [ singleNominalTarget, singleNumericTarget ]


# Set package version
from os import path
curdir = path.abspath(path.dirname(__file__))
with open(path.join(curdir, 'VERSION')) as version_file:
    __version__ = version_file.read().strip()
