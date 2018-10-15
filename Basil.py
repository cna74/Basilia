import sys
import os

ds = os.path.join(os.getcwd(), 'dataset')
if ds not in sys.path:
    sys.path.append(ds)

from dataset import util

Finder = util.Finder
