# RESOURCE jpg or csv
RESOURCE = "csv"

DATA_DIR = "<Open-Image-directory>"

IMG = 0
WIDTH, HEIGHT = 1, 2
LABEL = 3
BBOX_SLICE = slice(4, 8)
ROW_LENGTH = 8
HEADERS = {0: 'ImageID', 1: 'Source', 2: 'LabelName', 3: 'Confidence',
           4: 'XMin', 5: 'XMax', 6: 'YMin', 7: 'YMax', 8: 'IsOccluded',
           9: 'IsTruncated', 10: 'IsGroupOf', 11: 'IsDepiction', 12: 'IsInside'}

"""
i.e ->     ImageID, LabelName, XMin, XMax, YMin, YMax, IsOccluded, IsTruncated, IsGroupOf, IsDepiction, IsInside
DF_COLS = (0,       2,         4,    5,    6,    7,    8,          9,           10,        11,          12)
"""
DF_COLS = (0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12)
