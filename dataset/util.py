import numpy.core.defchararray as char
from collections import OrderedDict
import matplotlib.pyplot as plt
from zipfile import ZipFile
from dataset import dumper
from dataset import config
from PIL import Image
import pandas as pd
import numpy as np
import progressbar
import json
import cv2
import sys
import os

LABELS = dumper.label_loader()
# BBOX = dumper.bbox_loader() # call this when you really need it, it's Huge
IMG_DIRS = dumper.img_loader()
JSON_ = json.load(open('{}/dumped/bbox_labels_600_hierarchy.json'.format(os.path.split(__file__)[0])))['Subcategory']
headers = OrderedDict({0: 'ImageID', 1: 'Source', 2: 'LabelName', 3: 'Confidence',
                       4: 'XMin', 5: 'XMax', 6: 'YMin', 7: 'YMax', 8: 'IsOccluded',
                       9: 'IsTruncated', 10: 'IsGroupOf', 11: 'IsDepiction', 12: 'IsInside'})


class Finder:

    def __init__(self, search_for=None, size=None, father=False):
        self.search_result = []
        self._images_with_bbox = []
        self._image_df = pd.DataFrame({headers.get(i): [] for i in config.DF_COLS})

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
        return set(tmp)

    def bbox_test(self, size=None, n: int=4):
        if not size:
            size = self.size

        self._fill_images()
        idx = list(set((i[1] for i in self._image_df.itertuples())))

        choices = np.random.choice(len(idx), n*n, replace=False)
        selected_imgs = [idx[i] for i in choices]
        pathes = self._get_imgs_path_with_bboxes(selected_imgs)
        imgs_and_bbox = self._replace_path_with_img(pathes, size, True)

        plt.figure(figsize=(10, 10))
        for idx, data in enumerate(imgs_and_bbox):
            img = data[0]
            bboxes = data[1]
            titles = set(data[-1])
            plt.subplot(n, n, idx + 1)
            img = self._draw(img, bboxes, size)
            plt.imshow(img)
            plt.title(', '.join([self.mid_to_string(i) for i in titles]))
            plt.axis('off')
        plt.show()

    def _fill_images(self):
        bbox = dumper.bbox_loader()
        for i in self.search_result:
            self._image_df = self._image_df.append(bbox.loc[bbox['LabelName'] == i])

    def _fill_images_with_bbox(self, size=(224, 224)):
        self._fill_images()
        idx = list(set((i[1] for i in self._image_df.itertuples())))
        sys.stdout.write('{} unique image\n'.format(len(idx)))

        pathes = self._get_imgs_path_with_bboxes(idx)
        sys.stdout.write('{} unique object\n'.format(pathes.shape[0]))

        self._images_with_bbox = self._replace_path_with_img(pathes, size)

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

    # noinspection PyTypeChecker
    def _get_imgs_path_with_bboxes(self, imgs_id) -> np.ndarray:
        path_and_bboxes = []

        # split image dirs string and fetch names
        img_names = char.partition(IMG_DIRS, '/')[:, 2]

        # selected image dirs
        selected = IMG_DIRS[np.where(np.isin(img_names, imgs_id) == True)]

        for path in selected:
            name = path[9:]
            result = self._image_df.loc[self._image_df['ImageID'] == name,
                                        ['XMin', 'XMax', 'YMin', 'YMax', 'LabelName']]
            result = result.values

            tmp = np.c_[np.repeat(path, len(result)), result]

            path_and_bboxes.extend(tmp)

        return np.array(path_and_bboxes)

    def _replace_path_with_img(self, images_and_bboxes, size: tuple=None, is_test=False):
        if not size:
            size = self.size
        test = []

        dst = os.path.join(dumper.DATA_DIR, 'Train/{}.zip')
        zip_name = np.unique([i[:8] for i in images_and_bboxes[:, 0]])

        un = np.unique(images_and_bboxes[:, 0])
        un = un.astype(np.str)

        for folder in zip_name:
            sep = char.partition(un, '/')
            file_names = sep[np.where(sep[:, 0] == folder)][:, 2]
            sys.stdout.write(folder+'\n')
            bar = progressbar.ProgressBar()
            for img_dir in bar(file_names):
                with ZipFile(dst.format(folder)) as zip_:
                    img_path = os.path.join(folder, img_dir)
                    with zip_.open(img_path+'.jpg') as image:
                        img = np.asarray(Image.open(image))
                        if size:
                            img = cv2.resize(img, size)
                        if not is_test:
                            len_ = len(np.where(images_and_bboxes[:, 0] == img_path))
                            images_and_bboxes[len_, 0] = np.tile(img, (len_, 1, 1, 1))
                        elif is_test:
                            test.append(img)
                            bboxes = images_and_bboxes[np.where(images_and_bboxes[:, 0] == img_path), 1:-1][0]
                            test.append(bboxes)
                            label = images_and_bboxes[np.where(images_and_bboxes[:, 0] == img_path), -1][0]
                            test.append(label)
        if is_test:
            test = np.array(test).reshape(-1, 3)
            return test
        else:
            return images_and_bboxes


if __name__ == '__main__':
    finder = Finder('Fruit', size=(224, 224))
    finder.bbox_test(n=4)
    # print(finder._fill_images_with_bbox()[:10])
    pass
