from os import path
from re import match
curdir = path.abspath(path.dirname(__file__))
with open(path.join(curdir, 'VERSION')) as version_file:
    __version__ = match("^pySubDisc (?P<version>.+)", version_file.read().strip()).group('version')
