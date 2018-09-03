import pickle
import pandas as pd

"""
summarise and dump the origin dataset
"""

bbox_dir = '/home/cna/PycharmProjects/tptpt/train-annotations-bbox.csv'

df = pd.read_csv(bbox_dir, dtype='str', index_col='ImageID')
for i in ('Source', 'Confidence', 'IsDepiction', 'IsOccluded', 'IsTruncated', 'IsInside'):
    df = df.drop(i, axis=1)

with open('dump', 'wb') as f:
    pickle.dump(df, f)
