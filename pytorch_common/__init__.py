from pkg_resources import get_distribution
from .decorators import *

# Set package version
__version__ = get_distribution("pytorch_common").version
