import numpy.core.defchararray as char
import matplotlib.pyplot as plt
from zipfile import ZipFile
from dataset import dumper
from PIL import Image
import pandas as pd
import numpy as np
import json
import cv2
import sys
import time
import os

LABELS = dumper.label_loader()
# BBOX = dumper.bbox_loader() call this when you really need it, it's Huge
IMG_DIRS = dumper.img_loader()
JSON_ = json.load(open('{}/dumped/bbox_labels_600_hierarchy.json'.format(os.path.split(__file__)[0])))['Subcategory']


class Finder:

    def __init__(self, search_for=None, size=None, father=False):
        self.search_result = []
        self.dirs = []
        self._images_with_bbox = None
        self._image_df = pd.DataFrame({i: [] for i in ['ImageID', 'LabelName',
                                                       'XMin', 'XMax', 'YMin', 'YMax', 'IsGroupOf']})

        if not size:
            sys.stdout.write("size = (224, 224) as default")
            self.size = (224, 244)
        else:
            self.size = size

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

    def convert(self):
        tmp = []

        for i in self.search_result:
            tmp.append(self.mid_to_string(i))
        return tmp.sort()

    def bbox_test(self, size=None, n: int=4):
        if not size:
            size = self.size

        self._fill_images()
        idx = list(set((i[1] for i in self._image_df.itertuples())))

        choices = np.random.choice(len(idx), n*n, replace=False)
        selected_imgs = np.array([idx[i] for i in choices])
        imgs_and_bbox = self._get_imgs_arrays_with_bboxes(selected_imgs, size=size, is_test=True)

        plt.figure(figsize=(10, 10))
        for idx, data in enumerate(imgs_and_bbox):
            plt.subplot(n, n, idx + 1)
            img = self._draw(data[0], np.array(data[1][:-1]), size)
            plt.imshow(img)
            plt.title(', '.join([self.mid_to_string(str(i)) for i in data[1][-1]]))
            plt.axis('off')
        plt.show()

    def _fill_images(self):
        bbox = dumper.bbox_loader()
        for i in self.search_result:
            self._image_df = self._image_df.append(bbox.loc[bbox['LabelName'] == i])

    def _fill_images_with_bbox(self, size=(224, 224)):
        self._fill_images()
        idx = np.array(list(set((i[1] for i in self._image_df.itertuples()))))

        self._images_with_bbox = self._get_imgs_arrays_with_bboxes(idx, size=size, is_test=False)
        return self._images_with_bbox

    def _dig(self, list_):
        for i in list_:
            self.search_result.append(i['LabelName'])
            if len(i.keys()) > 1:
                if 'Subcategory' in i.keys():
                    self._dig(i['Subcategory'])
                elif 'Part' in i.keys():
                    self._dig(i['Part'])

    @staticmethod
    def _draw(img: np.ndarray, bboxes: np.ndarray, size: tuple) -> np.ndarray:
        bboxes = np.multiply(size[0], bboxes.astype(np.float))
        bboxes = bboxes.astype(np.uint8)

        for i in bboxes:
            img = cv2.rectangle(img=img, pt1=(i[0], i[2]), pt2=(i[1], i[3]), color=(0, 0, 255), thickness=2)
        return img

    def _get_imgs_arrays_with_bboxes(self, imgs_id: np.ndarray, size: tuple=None, is_test=False) -> np.ndarray:
        print(time.ctime())
        if not size:
            size = self.size

        # get image path
        images_and_bboxes = []
        for img_dir in IMG_DIRS:
            # img_dir = char.partition(char.replace(img_dir, '.jpg', ''), '/')
            # np.where(img_dir[:, 2] == imgs_id)
            for img in imgs_id:
                if img == img_dir[9:-4]:
                    if is_test:
                        tmp = [img_dir, ]
                        bboxes = self._image_df.loc[self._image_df['ImageID'] == img,
                                                    ['XMin', 'XMax', 'YMin', 'YMax']]
                        bboxes = list(bboxes.values)

                        lbls = self._image_df.loc[self._image_df['ImageID'] == img, 'LabelName']
                        lbls = set(lbls.values)

                        bboxes.append(list(lbls & set(self.search_result)))
                        tmp.append(bboxes)
                    else:
                        bboxes = self._image_df.loc[self._image_df['ImageID'] == img,
                                                    ['XMin', 'XMax', 'YMin', 'YMax', 'LabelName']]
                        bboxes = np.array(bboxes.values)
                        tmp = np.c_[np.repeat(img_dir, len(bboxes)), bboxes]

                    self.dirs.append(tmp)

        dst = os.path.join(dumper.DATA_DIR, 'Train/train_0{}.zip')
        zip_numbers = set([i[0][7:8] for i in self.dirs])

        # read image as np.ndarray
        for file in zip_numbers:
            for img_dir in self.dirs:
                if img_dir[0][7:8] == file:
                    with ZipFile(dst.format(file)) as zip_:
                        with zip_.open(img_dir[0]) as image:
                            img = np.asarray(Image.open(image))
                            if size:
                                img = cv2.resize(img, size)
                            tmp = [img, img_dir[1]]
                            images_and_bboxes.append(tmp)
        print(time.ctime())
        return np.array(images_and_bboxes)


if __name__ == '__main__':
    finder = Finder('Fruit', size=(224, 224))
    finder.bbox_test(n=4)
    # print(finder._fill_images_with_bbox()[:10])
    pass
