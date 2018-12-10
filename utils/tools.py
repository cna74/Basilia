from utils import dumper, finder, tf_generator
from os.path import join, exists
import numpy as np
import pickle
import sys
import cv2

LABELS = dumper.label_loader()


def write(s):
    sys.stdout.write(s + "\n")
    sys.stdout.flush()


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


def draw(img, bboxes, size):
    bboxes = np.multiply(size[0], bboxes.astype(np.float))
    bboxes = bboxes.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for b in bboxes:
        img = cv2.rectangle(img=img, pt1=(b[0], b[1]), pt2=(b[2], b[3]), color=(0, 0, 255), thickness=2)
    return img


def generate(csv_input, output_path, images_dir, classes):
    tf_generator.main(csv_input=csv_input,
                      output_path=output_path,
                      images_dir=images_dir,
                      classes=classes)


if __name__ == "__main__":
    d = dict_of_all_classes()
    print(d)
