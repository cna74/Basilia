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

    def __init__(self, search_for=None, father=False, size=None):
        self.classes = []
        self.dirs = []
        self.images_as_array = []
        self.images = None

        if not size:
            print("size = (224, 224) as default")
            self.size = (224, 244)

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
        return self._get_img_arrays_with_bboxes(self.images, size=size)

    def bbox_test(self, size=None):
        if not size:
            size = self.size

        self.fill_images()
        idx = list(set((i[0] for i in self.images.itertuples())))

        choices = np.random.choice(len(idx), 4, replace=False)
        selected_imgs = [idx[i] for i in choices]
        imgs_and_bbox = self._get_img_arrays_with_bboxes(selected_imgs)

        for idx, data in enumerate(imgs_and_bbox):
            plt.subplot(2, 2, idx + 1)
            img = self._draw(data[0], data[1], size)
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

    def _draw(self, img: np.ndarray, bboxes: np.ndarray, size: tuple):
        bboxes = np.multiply(size[0], bboxes.astype(np.float))
        bboxes = bboxes.astype(np.uint8)

        for i in bboxes:
            img = cv2.rectangle(img, (i[0], i[2]), (i[1], i[3]), (0, 0, 255))
        return img

    def _get_img_arrays_with_bboxes(self, imgs, size=(224, 224)) -> list:
        # get image path
        images_and_bboxes = []
        for dir_ in IMG_DIRS:
            for img in imgs:
                if img == dir_[9:-4]:
                    tmp = [dir_, ]
                    bboxes = self.images.loc[[img], ['XMin', 'XMax', 'YMin', 'YMax']]
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
        return images_and_bboxes


if __name__ == '__main__':
    finder = Finder('Food')
    finder.bbox_test()