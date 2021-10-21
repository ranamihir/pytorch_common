from pkg_resources import get_distribution

from .decorators import *

# Set package information
__version__ = get_distribution("pytorch_common").version
__author__ = "Mihir Rana"
