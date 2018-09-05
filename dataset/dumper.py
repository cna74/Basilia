import os
import pickle
import pandas as pd
from zipfile import ZipFile

"""
minify and dumped_bbox the origin dataset
"""

DATADIR = '/media/cna/backpack/dataset/Open_Image/'


def bbox_dumper(dst: str = None):
    if not dst:
        dst = '/home/cna/PycharmProjects/tptpt/train-annotations-bbox.csv'
    df = pd.read_csv(dst, dtype='str', index_col='ImageID')
    for i in ('Source', 'Confidence', 'IsDepiction', 'IsOccluded', 'IsTruncated', 'IsInside'):
        df = df.drop(i, axis=1)

    with open('dumped_bbox', 'wb') as f:
        pickle.dump(df, f)


def img_dumper(dst: str = None):
    if not dst:
        dst = '/media/cna/backpack/dataset/Open_Image/Train/train_0{}.zip'
    lst = []
    for i in range(9):
        file = dst.format(i)
        with ZipFile(file, 'r') as zip_:
            print(len(zip_.namelist()) - 1)
            lst.extend(zip_.namelist()[1:])
    with open('dumped_img_dirs', 'wb') as f:
        pickle.dump(lst, f)


def img_loader(dst: str = None) -> list:
    if not dst:
        dst = './dumped_img_dirs'
    with open(dst, 'rb') as f:
        return pickle.load(f)


def bbox_loader(dst: str = None) -> pd.DataFrame:
    if not dst:
        dst = './dumped_bbox'
    with open(dst, 'rb') as f:
        return pickle.load(f)


def label_loader(dst: str = None) -> pd.DataFrame:
    if not dst:
        dst = os.path.join(DATADIR, 'class-descriptions-boxable.csv')
    df = pd.read_csv(dst, dtype='str')
    return df


# bbox_dumper()
# img_dumper()
# lst = img_loader()
# bbox = bbox_loader()
# labels = label_loader()
