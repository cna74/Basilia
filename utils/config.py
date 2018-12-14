# RESOURCE jpg or csv
RESOURCE = "jpg"

DATA_DIR = "/media/cna/backpack/dataset/Open-Image/"

IMG = 0
WIDTH, HEIGHT = 1, 2
LABEL = 3
BBOX_SLICE = slice(4, 8)
ROW_LENGTH = 8
HEADERS = {0: 'ImageID', 1: 'Source', 2: 'LabelName', 3: 'Confidence',
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
