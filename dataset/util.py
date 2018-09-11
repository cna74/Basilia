import matplotlib.pyplot as plt
from zipfile import ZipFile
from PIL import Image
import numpy as np
import dumper
import json
import cv2
import os

LABELS = dumper.label_loader()
BBOX = dumper.bbox_loader()
IMG_DIRS = dumper.img_loader()
JSON_ = json.load(open('dumped/bbox_labels_600_hierarchy.json'))['Subcategory']


class Finder:

    def __init__(self, search_for=None, father=False):
        self.classes = []
        self.dirs = []
        self.images_as_array = []
        self.images = None

        if search_for:
            self.search(subject=search_for, father=father)

    def __add__(self, other):
        return self.classes + other.classes

    def fill_images(self):
        for i in self.classes:
            self.images = BBOX.loc[BBOX['LabelName'] == i]

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

    def search(self, subject, list_=None, father=False) -> set:
        if not list_:
            self.classes = []
            list_ = JSON_

        for i in list_:
            if i['LabelName'] == self.mid_to_string(subject):
                self.classes.append(i['LabelName'])
                if len(i.keys()) > 1:
                    if 'Subcategory' in i.keys():
                        self._dig(i['Subcategory'])
                    elif 'Part' in i.keys():
                        self._dig(i['Part'])

            elif len(i.keys()) > 1:
                if 'Subcategory' in i.keys():
                    self.search(subject, i['Subcategory'])
                elif 'Part' in i.keys():
                    self.search(subject, i['Part'])

        if not father and self.mid_to_string(subject) in self.classes and len(self.classes) > 1:
            self.classes.remove(self.mid_to_string(subject))

        return set(self.classes)

    def full_result(self, size=(224, 224)):
        return self._get_img_arrays(self.images, size=size)

    def bbox_test(self):
        self.fill_images()
        idx = list(set((i[0] for i in self.images.itertuples())))

        choices = np.random.choice(len(idx), 4, replace=False)
        result = [idx[i] for i in choices]
        imgs = self._get_img_arrays(result)

        for idx, img in enumerate(imgs):
            plt.subplot(2, 2, idx + 1)
            plt.imshow(img)
            plt.axis('off')
        plt.show()

    def convert(self, list_=None):
        tmp = []
        if not list_:
            list_ = self.classes
        for i in list_:
            tmp.append(self.mid_to_string(i))
        return tmp

    def _dig(self, list_):
        for i in list_:
            self.classes.append(i['LabelName'])
            if len(i.keys()) > 1:
                if 'Subcategory' in i.keys():
                    self._dig(i['Subcategory'])
                elif 'Part' in i.keys():
                    self._dig(i['Part'])

    def _get_img_arrays(self, imgs, size=(224, 224)) -> list:
        images_as_array = []
        for dir_ in IMG_DIRS:
            for img in imgs:
                if img == dir_[9:-4]:
                    self.dirs.append(dir_)

        dst = os.path.join(dumper.DATA_DIR, 'Train/train_0{}.zip')
        zip_numbers = set([i[7:8] for i in self.dirs])

        for file in zip_numbers:
            for dir_ in self.dirs:
                if dir_[7:8] == file:
                    with ZipFile(dst.format(file)) as zip_:
                        with zip_.open(dir_) as image:
                            im = np.asarray(Image.open(image))
                            if size:
                                im = cv2.resize(im, size)
                            images_as_array.append(im)
        return images_as_array


finder = Finder('Apple')
# print(finder.classes)
# print(finder.convert())
finder.bbox_test()
# print(len(finder.images))