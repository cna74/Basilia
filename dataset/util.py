from os.path import split, join, exists
import numpy.core.defchararray as char
from dataset import config, dumper
import matplotlib.pyplot as plt
from zipfile import ZipFile
from PIL import Image
import pandas as pd
import numpy as np
import warnings
import tqdm
import json
import cv2
import sys
import os

# region checkers
warnings.simplefilter(action='ignore', category=FutureWarning)
dumper.check_requirements()
LABELS = dumper.label_loader()
IMG_DIRS = dumper.img_loader()
JSON_ = json.load(open('{}/dumped/bbox_labels_600_hierarchy.json'.format(split(__file__)[0])))['Subcategory']
id_slice = 0
img_slice = 1
bbox_slice = slice(2, 6)
label_slice = -1
headers = {0: 'ImageID', 1: 'Source', 2: 'LabelName', 3: 'Confidence',
           4: 'XMin', 5: 'XMax', 6: 'YMin', 7: 'YMax', 8: 'IsOccluded',
           9: 'IsTruncated', 10: 'IsGroupOf', 11: 'IsDepiction', 12: 'IsInside'}
# endregion


class Finder:

    def __init__(self, search_for: str=None, size=(224, 224), father=False, address=None):
        self.__search_result = []
        self.images_with_bbox = []
        self.image_df = pd.DataFrame({headers.get(i): [] for i in config.DF_COLS})

        if not address:
            self.address = join(split(__name__)[0], 'data')
            if not exists(self.address):
                os.makedirs(self.address)
            sys.stdout.write('images will export to {}\n'.format(self.address))

        else:
            self.address = address

        self.size = size

        if search_for:
            self.search(subject=search_for.title(), father=father)

        sys.stdout.flush()

    @property
    def list_of_all_classes(self) -> list:
        return list(set([i[0] for i in LABELS.itertuples()]))

    @property
    def search_result(self) -> set:
        tmp = []
        for i in self.__search_result:
            tmp.append(self.mid_to_string(i))
        return set(tmp)

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

    def fill_images_with_bbox(self):
        if len(self.images_with_bbox) == 0:
            self._extract_data_frame()
            idx = list(set((i[1] for i in self.image_df.itertuples())))
            sys.stdout.write('{} unique image\n'.format(len(idx)))

            paths = self._get_imgs_path_with_bboxes(idx)
            sys.stdout.write('{} unique object\n'.format(paths.shape[0]))
            sys.stdout.flush()

            self.images_with_bbox = self._replace_path_with_img(paths, self.size)

    def store_data(self):
        with open(join(self.address, 'bbox'), 'w') as file:
            for dir_ in self.search_result:
                label_folder = join(self.address, dir_)

                try:
                    os.makedirs(label_folder)
                except:
                    pass

                y = np.unique(self.images_with_bbox[:, 0], return_index=True, return_counts=True)

                for n, idx, count in zip(*y):
                    row = self.images_with_bbox[idx, :]
                    bboxes = self.images_with_bbox[np.where(self.images_with_bbox[:, 0] == n), bbox_slice][0]
                    name = '{}'.format(str(n).zfill(5))
                    dst = join(label_folder, name+'.jpg')
                    if not exists(dst):
                        img = cv2.cvtColor(row[1], cv2.COLOR_BGR2RGB)
                        cv2.imwrite(dst, img)
                    for bbox in bboxes:
                        file.write('{}{}{}{}{}{}\n'.format(name.ljust(7), *[str(i).ljust(10) for i in bbox], dir_))

    def search(self, subject, list_=None, father=False):
        if not list_:
            self.__search_result = []
            list_ = JSON_

        for i in list_:
            if i['LabelName'] == self.mid_to_string(subject):
                self.__search_result.append(i['LabelName'])
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

        if not father and self.mid_to_string(subject) in self.__search_result and len(self.__search_result) > 1:
            self.__search_result.remove(self.mid_to_string(subject))

    def bbox_test(self, size=None, n: int=4):
        if not size:
            size = self.size

        self._extract_data_frame()
        idx = list(set((i[1] for i in self.image_df.itertuples())))

        choices = np.random.choice(len(idx), n*n, replace=False)
        selected_imgs = [idx[i] for i in choices]
        pathes = self._get_imgs_path_with_bboxes(selected_imgs)
        imgs_and_bbox = self._replace_path_with_img(pathes, size, True)

        plt.figure(figsize=(10, 10))
        for idx, data in enumerate(imgs_and_bbox, start=1):
            img = data[0]
            bboxes = data[1]
            titles = set(data[-1])
            plt.subplot(n, n, idx)
            img = self._draw(img, bboxes, size)
            plt.imshow(img)
            plt.title(', '.join(titles))
            plt.axis('off')
        plt.show()

    def _extract_data_frame(self):
        bbox_df = dumper.bbox_loader()
        for i in self.__search_result:
            self.image_df = self.image_df.append(bbox_df.loc[bbox_df['LabelName'] == i])

    def _dig(self, list_):
        for i in list_:
            self.__search_result.append(i['LabelName'])
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

        # add info
        for i, path in enumerate(selected, start=1):
            name = path[9:]
            result = self.image_df.loc[self.image_df['ImageID'] == name,
                                       ['XMin', 'XMax', 'YMin', 'YMax', 'LabelName']]

            result = result.values

            tmp = np.c_[np.repeat(i, len(result)).astype(np.uint),
                        np.repeat(path, len(result)),
                        result]

            path_and_bboxes.extend(tmp)

        path_and_bboxes = np.array(path_and_bboxes)

        # convert labels
        for i in np.unique(path_and_bboxes[:, -1]):
            path_and_bboxes[np.where(path_and_bboxes[:, -1] == i), -1] = self.mid_to_string(i)

        # reformat bboxes
        path_and_bboxes[:, bbox_slice] = path_and_bboxes[:, bbox_slice].astype(np.float)

        return path_and_bboxes

    def _replace_path_with_img(self, paths_and_bboxes, size: tuple=None, is_test=False):
        if not size:
            size = self.size
        test = []

        dst = join(dumper.DATA_DIR, 'Train/{}.zip')
        zip_name = np.unique([i[:8] for i in paths_and_bboxes[:, 1]])

        un = np.unique(paths_and_bboxes[:, 1])
        un = un.astype(np.str)

        for folder in zip_name:
            sep = char.partition(un, '/')
            file_names = sep[np.where(sep[:, 0] == folder)][:, 2]
            for img_dir in tqdm.tqdm(file_names, desc=folder, ncols=100):
                with ZipFile(dst.format(folder)) as zip_:
                    img_path = join(folder, img_dir)
                    if sys.platform == 'win32':
                        img_path = img_path.replace('\\', '/')
                    with zip_.open(img_path+'.jpg') as image:
                        img = np.asarray(Image.open(image))
                        img = cv2.resize(img, size)

                        if not is_test:
                            fnd = np.where(paths_and_bboxes[:, 1] == img_path)[0]
                            for i in fnd:
                                paths_and_bboxes[i, 1] = img

                        elif is_test:
                            test.append(img)
                            bboxes = paths_and_bboxes[np.where(paths_and_bboxes[:, 1] == img_path), bbox_slice][0]
                            test.append(bboxes)
                            label = paths_and_bboxes[np.where(paths_and_bboxes[:, 1] == img_path), -1][0]
                            test.append(label)
        if is_test:
            test = np.array(test).reshape(-1, 3)
            return test
        else:
            return paths_and_bboxes


if __name__ == '__main__':
    finder = Finder('lemon', size=(224, 224))
    finder.bbox_test(n=3)
    pass
