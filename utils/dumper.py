from os.path import exists, join, split
import numpy.core.defchararray as char
from numpy import genfromtxt
from utils import config
from glob2 import glob
import pandas as pd
import numpy as np
import json
import sys
import os

"""
dump the origin dataset's csv files
"""

DUMP_DIR = join(os.sep, split(__file__)[0], "dumped")
try:
    os.makedirs(DUMP_DIR)
except:
    pass


def annotation_loader(dir_, folder_name) -> pd.DataFrame:
    dumped = join(DUMP_DIR, "{}-annotations-bbox".format(folder_name.lower()))
    if exists(dumped):
        df = pd.read_pickle(dumped)
    else:
        dst = join(dir_, '{}/{}-annotations-bbox.csv'.format(folder_name, folder_name.lower()))
        df = pd.read_csv(dst,  usecols=config.DF_COLS, dtype='str')
        # df.to_pickle(dumped)
    return df


def img_dirs(resource, dir_) -> np.ndarray:
    dumped = join(DUMP_DIR, "img_dirs_{}.npy".format(resource))
    if exists(dumped):
        return np.load(dumped)
    if resource == "jpg":
        pathname = dir_+"*/*/*.jpg"
        if sys.platform == "win32":
            pathname = pathname.replace("\\", "/")
        dirs = glob(pathname=pathname)
        dirs = np.array(dirs)
        # dirs.dump(dumped)

    elif resource == "csv":
        pathname = dir_+"*/*images*.csv"
        if sys.platform == "win32":
            pathname = pathname.replace("\\", "/")
        dirs = glob(pathname=pathname)
        dfs = np.array(['image_name', 'image_url'])
        for c in dirs:
            dfs = np.append(dfs, genfromtxt(c, dtype=np.str, delimiter=",")[1:])
        dirs = dfs.reshape((-1, 2))[1:]
        dirs[:, 0] = char.replace(dirs[:, 0], ".jpg", "")
        # dirs.dump(dumped)
    else:
        raise FileNotFoundError("can't find {} images".format(resource))
    return dirs


def label_loader(dir_) -> pd.DataFrame:
    dumped = join(DUMP_DIR, "labels")
    if exists(dumped):
        df = pd.read_pickle(dumped)
    else:
        dst = join(dir_, 'class-descriptions-boxable.csv')
        df = pd.read_csv(dst, dtype='str', names=['code', 'name'], index_col=[1])
        df.to_pickle(dumped)
    return df


def json_loader(dir_=config.DATA_DIR, name='bbox_labels_600_hierarchy.json') -> json.JSONDecoder:
    js = join(dir_, name)
    if not exists(js):
        raise FileNotFoundError('MISSING {}'.format(js))
    json_ = json.load(open(js))['Subcategory']
    return json_


if __name__ == '__main__':
    pass
