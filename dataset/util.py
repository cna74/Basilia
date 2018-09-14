import matplotlib.pyplot as plt
from zipfile import ZipFile
from dataset import dumper
from math import sqrt
from PIL import Image
import pandas as pd
import numpy as np
import json
import cv2
import sys
import os

LABELS = dumper.label_loader()
BBOX = dumper.bbox_loader()
IMG_DIRS = dumper.img_loader()
JSON_ = json.load(open('{}/dataset/dumped/bbox_labels_600_hierarchy.json'
                       .format(os.path.split(__file__)[0])))['Subcategory']


class Finder:

    def __init__(self, search_for=None, father=False, size=None):
        self.search_result = []
        self.dirs = []
        self._images_with_bbox = None
        self._image_df = pd.DataFrame({i: [] for i in BBOX.columns.values})

        if not size:
            sys.stdout.write("size = (224, 224) as default")
            self.size = (224, 244)

        if search_for:
            self.search(subject=search_for, father=father)

    def __add__(self, other):
        return self.search_result + other.search_result

    @staticmethod
    def list_of_classes() -> set:
        return set([i[0] for i in LABELS.itertuples()])

    @staticmethod
    def mid_to_string(mid_or_name) -> str:
        if mid_or_name.startswith('/m'):
            sel = LABELS.loc[LABELS['code'] == mid_or_name]
            sel = sel.to_dict()
            return list(sel['code'].keys())[0]
        else:
            sel = LABELS.loc[mid_or_name]
            sel = sel.to_dict()
            return sel['code']

    @staticmethod
    def _draw(img: np.ndarray, bboxes: np.ndarray, size: tuple) -> np.ndarray:
        bboxes = np.multiply(size[0], bboxes.astype(np.float))
        bboxes = bboxes.astype(np.uint8)

        for i in bboxes:
            img = cv2.rectangle(img=img, pt1=(i[0], i[2]), pt2=(i[1], i[3]), color=(0, 0, 255), thickness=2)
        return img

    @property
    def imgs_with_bbox(self):
        if len(self._images_with_bbox) == 0:
            self._fill_images_with_bbox()
        return self._images_with_bbox

    def search(self, subject, list_=None, father=False):
        if not list_:
            self.search_result = []
            list_ = JSON_

        for i in list_:
            if i['LabelName'] == self.mid_to_string(subject):
                self.search_result.append(i['LabelName'])
                if len(i.keys()) > 1:
                    if 'Subcategory' in i.keys():
                        self._dig(i['Subcategory'])
                    elif 'Part' in i.keys():
                        self._dig(i['Part'])

            elif len(i.keys()) > 1:
                if 'Subcategory' in i.keys():
                    self.search(subject=subject, list_=i['Subcategory'], father=father)
                elif 'Part' in i.keys():
                    self.search(subject=subject, list_=i['Part'], father=father)

        if not father and self.mid_to_string(subject) in self.search_result and len(self.search_result) > 1:
            self.search_result.remove(self.mid_to_string(subject))

    def bbox_test(self, size=None, many=16):
        if not size:
            size = self.size

        self._fill_images()
        idx = list(set((i[1] for i in self._image_df.itertuples())))

        choices = np.random.choice(len(idx), many, replace=False)
        selected_imgs = [idx[i] for i in choices]
        imgs_and_bbox = self._get_img_arrays_with_bboxes(selected_imgs)

        for idx, data in enumerate(imgs_and_bbox):
            plt.subplot(sqrt(many), sqrt(many), idx + 1)
            img = self._draw(data[0], data[1], size)
            plt.imshow(img)
            plt.axis('off')
        plt.show()

    def convert(self):
        tmp = []

        for i in self.search_result:
            tmp.append(self.mid_to_string(i))
        return tmp.sort()

    def _fill_images(self):
        for i in self.search_result:
            self._image_df = self._image_df.append(BBOX.loc[BBOX['LabelName'] == i])

    def _fill_images_with_bbox(self, size=(224, 224)):
        self._images_with_bbox = self._get_img_arrays_with_bboxes(self._image_df, size=size)
        return self._images_with_bbox

    def _dig(self, list_):
        for i in list_:
            self.search_result.append(i['LabelName'])
            if len(i.keys()) > 1:
                if 'Subcategory' in i.keys():
                    self._dig(i['Subcategory'])
                elif 'Part' in i.keys():
                    self._dig(i['Part'])

    def _get_img_arrays_with_bboxes(self, imgs, size=(224, 224)) -> np.ndarray:
        # get image path
        images_and_bboxes = []
        for dir_ in IMG_DIRS:
            for img in imgs:
                if img == dir_[9:-4]:
                    tmp = [dir_, ]
                    bboxes = self._image_df.loc[self._image_df['ImageID'] == img, ['XMin', 'XMax', 'YMin', 'YMax']]
                    bboxes = bboxes.values
                    tmp.append(bboxes)
                    self.dirs.append(tmp)

        dst = os.path.join(dumper.DATA_DIR, 'Train/train_0{}.zip')
        zip_numbers = set([i[0][7:8] for i in self.dirs])

        # read image as np.ndarray
        for file in zip_numbers:
            for dir_ in self.dirs:
                if dir_[0][7:8] == file:
                    with ZipFile(dst.format(file)) as zip_:
                        with zip_.open(dir_[0]) as image:
                            img = np.asarray(Image.open(image))
                            if size:
                                img = cv2.resize(img, size)
                            tmp = [img, dir_[1]]
                            images_and_bboxes.append(tmp)
        return np.array(images_and_bboxes)


if __name__ == '__main__':
    pass
    # finder = Finder('Food')
    # finder.bbox_test()
