import sys
import os

ds = os.path.join(os.getcwd(), 'utils')
if ds not in sys.path:
    sys.path.append(ds)

from utils import util

Finder = util.Finder
