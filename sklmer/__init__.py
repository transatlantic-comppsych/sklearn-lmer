from ._estimators import LmerRegressor
#from ._template import TemplateClassifier

from ._version import __version__

__all__ = ['LmerRegressor',
           '__version__']

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
