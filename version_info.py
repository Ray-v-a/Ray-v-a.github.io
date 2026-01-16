#
# Version information for src.outbreak_probabilities_model.
#
# See: https://packaging.python.org/guides/single-sourcing-package-version/
#
# Version as a tuple (major, minor, revision)
#  - Changes to major are rare
#  - Changes to minor indicate new features, possible slight backwards
#    incompatibility
#  - Changes to revision indicate bugfixes, tiny new features
VERSION_INT = 1, 1, 0

# String version of the version number
VERSION = '.'.join([str(x) for x in VERSION_INT])
__version__ = VERSION
