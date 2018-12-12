from os.path import exists, join, split
import numpy.core.defchararray as char
from numpy import genfromtxt
from utils import config
from glob2 import glob
import pandas as pd
import numpy as np
import json
import sys

"""
dump the origin dataset's csv files
"""

DUMP_DIR = join(split(__file__)[0], "dumped")


def annotation_loader(folder_name) -> pd.DataFrame:
    dumped = join(DUMP_DIR, "{}-annotations-bbox".format(folder_name.lower()))
    if exists(dumped):
        df = pd.read_pickle(dumped)
    else:
        dst = join(config.DATA_DIR, '{}/{}-annotations-bbox.csv'.format(folder_name, folder_name.lower()))
        df = pd.read_csv(dst,  usecols=config.DF_COLS, dtype='str')
        df.to_pickle(dumped)

    return df


def img_dirs() -> np.ndarray:
    dumped = join(DUMP_DIR, "img_dirs.npy")
    if exists(dumped):
        dirs = np.load(dumped)
    else:
        if config.AVAILABLE_AS == "jpg":
            pathname = config.DATA_DIR+"*/*/*.jpg"
            if sys.platform == "win32":
                pathname = pathname.replace("\\", "/")
            dirs = glob(pathname=pathname)
            dirs = np.array(dirs)
            # dirs = char.replace(dirs, ".jpg", "")
            np.save(dumped, dirs)
        elif config.AVAILABLE_AS == "csv":
            pathname = config.DATA_DIR+"*/*images*.csv"
            if sys.platform == "win32":
                pathname = pathname.replace("\\", "/")
            dirs = glob(pathname=pathname)
            dfs = np.array(['image_name', 'image_url'])
            for c in dirs:
                dfs = np.append(dfs, genfromtxt(c, dtype=np.str, delimiter=",")[1:])
            dfs = dfs.reshape((-1, 2))[1:]
            dfs[:, 0] = char.replace(dfs[:, 0], ".jpg", "")
            np.save(dumped, dfs)

        # todo take care of this part later
        elif config.AVAILABLE_AS == "zip":
            dirs = None
        else:
            raise FileNotFoundError("can't find {} images".format(config.AVAILABLE_AS))
    return dirs


def label_loader() -> pd.DataFrame:
    dumped = join(DUMP_DIR, "labels")
    if exists(dumped):
        df = pd.read_pickle(dumped)
    else:
        dst = join(config.DATA_DIR, 'class-descriptions-boxable.csv')
        df = pd.read_csv(dst, dtype='str', names=['code', 'name'], index_col=[1])
        df.to_pickle(dumped)

    return df


def json_loader():
    json_ = json.load(open('{}/dumped/bbox_labels_600_hierarchy.json'.format(split(__file__)[0])))['Subcategory']
    return json_


def check_requirements():
    if not exists(join(DUMP_DIR, 'bbox_labels_600_hierarchy.json')):
        raise FileNotFoundError('MISSING bbox_labels_600_hierarchy.json')


if __name__ == '__main__':
    for i in ["Train", "Test", "Validation"]:
        train = annotation_loader(i)
    imgs = img_dirs()
    labels = label_loader()
    pass
