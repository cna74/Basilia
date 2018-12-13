import sys

# AVAILABLE_AS jpg or csv or zip
AVAILABLE_AS = "jpg"

DATA_DIR = "FILL THIS IF YOU ARE ON A STUPID MAC !!!"

if sys.platform == 'linux':
    DATA_DIR = '/media/cna/backpack/dataset/Open_Image/'
elif sys.platform == 'win32':
    DATA_DIR = 'E:/dataset/Open_Image/'


IMG = 0
WIDTH, HEIGHT = 1, 2
LABEL = 3
BBOX_SLICE = slice(4, 8)
ROW_LENGTH = 8
headers = {0: 'ImageID', 1: 'Source', 2: 'LabelName', 3: 'Confidence',
           4: 'XMin', 5: 'XMax', 6: 'YMin', 7: 'YMax', 8: 'IsOccluded',
           9: 'IsTruncated', 10: 'IsGroupOf', 11: 'IsDepiction', 12: 'IsInside'}

"""
DF_COLS:
0 ImageID       ***
1 Source
2 LabelName     ***
3 Confidence
4 XMin          ***
5 XMax          ***
6 YMin          ***
7 YMax          ***
8 IsOccluded
9 IsTruncated
10 IsGroupOf    ***
11 IsDepiction
12 IsInside
i.e ->     ImageID, LabelName, XMin, XMax, YMin, YMax, IsGroupOf
DF_COLS = (0,       2,         4,    5,    6,    7,    10)
"""
DF_COLS = (0, 2, 4, 5, 6, 7, 10)
