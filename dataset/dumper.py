import os
import sys
import pickle
import pandas as pd
from zipfile import ZipFile

"""
minify and dumped_bbox the origin dataset
"""

if len(sys.argv) == 1:
    DATA_DIR = '/media/cna/backpack/dataset/Open_Image/'
else:
    if str(sys.argv[1]).startswith('--'):
        DATA_DIR = str(sys.argv[1])[2:]
    else:
        DATA_DIR = '/media/cna/backpack/dataset/Open_Image/'
dump_dir = os.path.split(__file__)[0]


def bbox_dumper(dst: str = None):
    if not dst:
        dst = os.path.join(DATA_DIR, 'Train/train-annotations-bbox.csv')
    df = pd.read_csv(dst, dtype='str')
    for i in ('Source', 'Confidence', 'IsDepiction', 'IsOccluded', 'IsTruncated', 'IsInside'):
        df = df.drop(i, axis=1)

    with open('{}/dumped/dumped_bbox'.format(dump_dir), 'wb') as f:
        pickle.dump(df, f)


def img_dumper(dst: str = None):
    if not dst:
        dst = os.path.join(DATA_DIR, '/Train/train_0{}.zip')
    lst = []
    for i in range(9):
        file = dst.format(i)
        with ZipFile(file, 'r') as zip_:
            lst.extend(zip_.namelist()[1:])
    with open('{}/dumped/dumped_img_dirs'.format(dump_dir), 'wb') as f:
        pickle.dump(lst, f)


def label_dumper(dst: str = None):
    if not dst:
        dst = os.path.join(DATA_DIR, 'class-descriptions-boxable.csv')
    df = pd.read_csv(dst, dtype='str', names=['code', 'name'], index_col=[1])

    with open('{}/dumped/dumped_labels'.format(dump_dir), 'wb') as f:
        pickle.dump(df, f)


def img_loader(dst: str = None) -> list:
    if not dst:
        dst = '{}/dumped/dumped_img_dirs'.format(dump_dir)
    with open(dst, 'rb') as f:
        return pickle.load(f)


def bbox_loader(dst: str = None) -> pd.DataFrame:
    if not dst:
        dst = '{}/dumped/dumped_bbox'.format(dump_dir)
    with open(dst, 'rb') as f:
        return pickle.load(f)


def label_loader(dst: str = None) -> pd.DataFrame:
    if not dst:
        dst = '{}/dumped/dumped_labels'.format(dump_dir)
    with open(dst, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    # bbox_dumper()
    # img_dumper()
    # label_dumper()

    # lst = img_loader()
    # bbox = bbox_loader()
    # labels = label_loader()
    pass