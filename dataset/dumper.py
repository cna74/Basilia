import numpy.core.defchararray as char
from dataset import config
from zipfile import ZipFile
import pandas as pd
import numpy as np
import pickle
import sys
import os

"""
dump the origin dataset's csv files
"""

DUMP_DIR = os.path.split(__file__)[0]
DATA_DIR = config.DATA_DIR
DF_COLS = config.DF_COLS


def bbox_dumper(dst: str = None):
    sys.stdout.write('dumping bbox\n')
    if not dst:
        dst = os.path.join(DATA_DIR, 'Train/train-annotations-bbox.csv')
    df = pd.read_csv(dst,  usecols=DF_COLS, dtype='str')

    with open('{}/dumped/dumped_bbox'.format(DUMP_DIR), 'wb') as f:
        pickle.dump(df, f)


def img_dumper(dst: str = None):
    """
    dumps image directories as numpy array
    :param dst: dataset directory
    """
    sys.stdout.write('dumping image dirs\n')

    if not dst:
        dst = os.path.join(DATA_DIR, 'Train/train_0{}.zip')
    lst = np.array([])
    for i in range(9):
        file = dst.format(i)
        with ZipFile(file, 'r') as zip_:
            lst = np.append(lst, zip_.namelist()[1:])
    lst = char.replace(lst, '.jpg', '')
    with open('{}/dumped/dumped_img_dirs'.format(DUMP_DIR), 'wb') as f:
        pickle.dump(lst, f)


def label_dumper(dst: str = None):
    sys.stdout.write('dumping labels\n')

    if not dst:
        dst = os.path.join(DATA_DIR, 'class-descriptions-boxable.csv')
    df = pd.read_csv(dst, dtype='str', names=['code', 'name'], index_col=[1])

    with open('{}/dumped/dumped_labels'.format(DUMP_DIR), 'wb') as f:
        pickle.dump(df, f)


def img_loader(dst: str = None) -> list:
    if not dst:
        dst = '{}/dumped/dumped_img_dirs'.format(DUMP_DIR)
    with open(dst, 'rb') as f:
        return pickle.load(f)


def bbox_loader(dst: str = None) -> pd.DataFrame:
    if not dst:
        dst = '{}/dumped/dumped_bbox'.format(DUMP_DIR)
    with open(dst, 'rb') as f:
        return pickle.load(f)


def label_loader(dst: str = None) -> pd.DataFrame:
    if not dst:
        dst = '{}/dumped/dumped_labels'.format(DUMP_DIR)
    with open(dst, 'rb') as f:
        return pickle.load(f)


if not os.path.exists(os.path.join(DUMP_DIR, 'dumped/bbox_labels_600_hierarchy.json')):
    raise FileNotFoundError('MISSING bbox_labels_600_hierarchy.json')
if not os.path.exists(os.path.join(DUMP_DIR, 'dumped/dumped_bbox')):
    bbox_dumper()
if not os.path.exists(os.path.join(DUMP_DIR, 'dumped/dumped_img_dirs')):
    img_dumper()
if not os.path.exists(os.path.join(DUMP_DIR, 'dumped/dumped_labels')):
    label_dumper()

if __name__ == '__main__':
    # bbox_dumper()
    # img_dumper()
    # label_dumper()

    # lst = img_loader()
    # bbox = bbox_loader()
    # labels = label_loader()
    pass