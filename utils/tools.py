from utils import dumper, finder, tf_generator
from os.path import join, exists
import matplotlib.pyplot as plt
from utils import config
from glob2 import glob
import numpy as np
import pickle
import time
import sys
import cv2

LABELS = dumper.label_loader()


def write(s):
    sys.stdout.write(s)
    sys.stdout.write("\n")
    sys.stdout.flush()
    time.sleep(.1)


def dict_of_all_classes() -> dict:
    dumped = join(dumper.DUMP_DIR, "dict_of_classes.pickle")
    if exists(dumped):
        classes = pickle.load(open(dumped, "rb"))
    else:
        raw = list(set([i[0] for i in LABELS.itertuples()]))
        classes = {}
        for i, j in enumerate(raw):
            fnd = finder.Finder(subject=j, etc=True)
            result = sorted(list(fnd.search_result))
            if not len(result) == 1:
                classes.update({j: result})
        pickle.dump(classes, open(dumped, "wb"))
    return classes


def mid_to_string(mid_or_name) -> str:
    if mid_or_name.startswith('/m'):
        sel = LABELS.loc[LABELS['code'] == mid_or_name]
        sel = sel.to_dict()
        return list(sel['code'].keys())[0]
    else:
        sel = LABELS.loc[mid_or_name]
        sel = sel.to_dict()
        return sel['code']


def draw(img, bboxes, thickness):
    bboxes = bboxes.astype(np.uint)
    for b in bboxes:
        img = cv2.rectangle(img=img, pt1=(b[0], b[1]), pt2=(b[2], b[3]), color=(0, 0, 255), thickness=thickness)
    return img


def generate(csv_input, output_path, images_dir, classes):
    tf_generator.main(csv_input=csv_input,
                      output_path=output_path,
                      images_dir=images_dir,
                      classes=classes)


def bbox_test(address, target="test", n=2, thickness=3):
    file = "{}_bbox.csv".format(target.title())
    csv = np.genfromtxt("{}/{}/{}".format(address, "records", file), delimiter=",", skip_header=1, dtype=str)
    dir_ = join(address, "images", target)
    images = glob("{}/*.jpg".format(dir_))
    selected = np.random.choice(images, n*n, replace=False)
    del images, dir_

    plt.figure(figsize=(10, 10))
    for idx, img_dir in enumerate(selected, start=1):
        img = plt.imread(img_dir)
        res = csv[np.where(csv[:, config.IMG] == img_dir.rsplit("/")[-1])]
        bboxes = res[:, config.BBOX_SLICE]
        titles = np.unique(res[:, config.LABEL])
        plt.subplot(n, n, idx)
        img = draw(img, bboxes, thickness)
        plt.imshow(img)
        plt.title(', '.join(titles))
        plt.axis('off')
    plt.show()


if __name__ == "__main__":
    bbox_test(address="/home/cna/PycharmProjects/Basilia/utils/data")
