from os.path import split, join, exists
import numpy.core.defchararray as char
from progressbar import progressbar
import matplotlib.pyplot as plt
from zipfile import ZipFile
import multiprocessing
from PIL import Image
import pandas as pd
import numpy as np
import warnings
import config
import dumper
import cv2
import sys
import os

# region checkers
warnings.simplefilter(action='ignore', category=FutureWarning)
dumper.check_requirements()
LABELS = dumper.label_loader()
IMG_DIRS = dumper.img_loader()
JSON_ = dumper.json_loader()
id_slice = 0
img_slice = 1
bbox_slice = slice(2, 6)
label_slice = -1
headers = {0: 'ImageID', 1: 'Source', 2: 'LabelName', 3: 'Confidence',
           4: 'XMin', 5: 'XMax', 6: 'YMin', 7: 'YMax', 8: 'IsOccluded',
           9: 'IsTruncated', 10: 'IsGroupOf', 11: 'IsDepiction', 12: 'IsInside'}
# endregion


class Finder:

    def __init__(self, subject: str=None, size=(224, 224), etc=False, just=False, address=None):
        self.subject = subject
        self.etc = etc
        self.just = just
        self.size = size

        # MID
        self.__search_result = []
        # Str
        self.search_result = []

        self.images_with_bbox = np.delete(np.empty((1, len(config.DF_COLS))), 0, 0)
        self.image_df = pd.DataFrame({headers.get(i): [] for i in config.DF_COLS})

        if not address:
            self.address = join(split(__name__)[0], 'data')
            if not exists(self.address):
                os.makedirs(self.address)
        else:
            self.address = address

        if subject:
            self.search(subject=subject.capitalize(), etc=self.etc, just=self.just)
            self.fill_search_result()

        sys.stdout.write('images will export to {}\n'.format(self.address))
        sys.stdout.flush()

    @property
    def list_of_all_classes(self) -> list:
        return list(set([i[0] for i in LABELS.itertuples()]))

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

    def fill_search_result(self):
        tmp = []
        for i in self.__search_result:
            tmp.append(self.mid_to_string(i))
        self.search_result = set(tmp)

    def restore(self, dst):
        self.images_with_bbox: np.ndarray = np.load(dst)
        self.search_result = np.unique(self.images_with_bbox[:, label_slice])

    def fill_images_with_bbox(self):
        if len(self.images_with_bbox) == 0:
            self._extract_data_frame()

            idx = list(set((i[1] for i in self.image_df.itertuples())))
            sys.stdout.write('{} unique image\n'.format(len(idx)))
            sys.stdout.flush()

            self._get_imgs_path_with_bboxes(idx)
            sys.stdout.write('{} unique object\n'.format(self.images_with_bbox.shape[0]))
            sys.stdout.flush()

            zip_name = np.unique([i[:8] for i in self.images_with_bbox[:, 1]])
            p = multiprocessing.Pool(multiprocessing.cpu_count())
            out = p.map(func=self._replace_path_with_img, iterable=zip_name)
            p.close()
            p.join()
            
            idx, imgs = np.array([]), np.array([])
            for idxs, imgs_ in out:
                idx = np.append(idx, idxs)
                imgs = np.append(imgs, imgs_)
                
            idx = idx.astype(int)
            imgs = imgs.reshape((-1, self.size[0], self.size[1], 3)).astype(np.uint8)
            
            for i, j in zip(idx, imgs):
                self.images_with_bbox[i, 1] = j
                
        else:
            sys.stdout.write("it's already filled\n")
            sys.stdout.flush()

    def store_data(self):
        np.save('{}/{}-[etc={}]-[just={}]'.format(self.address, self.subject, self.etc, self.just),
                self.images_with_bbox)
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

    def bbox_test(self, size=None, n: int=4):
        if not size:
            size = self.size

        choose = np.unique(self.images_with_bbox[:, 0])
        choices = np.random.choice(len(choose), n*n, replace=False)

        plt.figure(figsize=(10, 10))
        for idx, choice in enumerate(choices, start=1):
            data = self.images_with_bbox[np.where(self.images_with_bbox[:, 0] == choice)]
            img = data[0, 1]
            bboxes = data[:, bbox_slice]
            titles = set(data[:, -1])
            plt.subplot(n, n, idx)
            img = self._draw(img, bboxes, size)
            plt.imshow(img)
            plt.title(', '.join(titles))
            plt.axis('off')
        plt.show()

    def search(self, subject, etc, just):
        self._search_step1(subject)
        self._search_step2(subject, etc, just)

    def _search_step1(self, subject, list_=None):
        subject = subject.capitalize()
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
                    self._search_step1(subject=subject, list_=i['Subcategory'])
                elif 'Part' in i.keys():
                    self._search_step1(subject=subject, list_=i['Part'])

    def _search_step2(self, subject, etc, just):
        if not just:
            if not etc and self.mid_to_string(subject) in self.__search_result and len(self.__search_result) > 1:
                self.__search_result.remove(self.mid_to_string(subject))
        else:
            self.__search_result = [self.mid_to_string(subject), ]

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
    def _get_imgs_path_with_bboxes(self, imgs_id):
        # path_and_bboxes = []

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

            self.images_with_bbox = np.append(self.images_with_bbox, tmp)

        self.images_with_bbox = self.images_with_bbox.reshape((-1, len(config.DF_COLS)))
        # path_and_bboxes = np.array(path_and_bboxes)

        # convert labels
        for i in np.unique(self.images_with_bbox[:, -1]):
            self.images_with_bbox[np.where(self.images_with_bbox[:, -1] == i), -1] = self.mid_to_string(i)

        # reformat bboxes
        self.images_with_bbox[:, bbox_slice] = self.images_with_bbox[:, bbox_slice].astype(np.float)

        # return path_and_bboxes

    def _replace_path_with_img(self, zip_name, size: tuple=None):
        if not size:
            size = self.size
        idx = np.array([])
        imgs = np.array([])
        
        dst = join(dumper.DATA_DIR, 'Train/{}.zip')

        un = np.unique(self.images_with_bbox[:, 1])
        un = un.astype(np.str)

        sep = char.partition(un, '/')
        file_names = sep[np.where(sep[:, 0] == zip_name)][:, 2]
        for img_dir in progressbar(file_names, prefix=zip_name+'  '):
            with ZipFile(dst.format(zip_name)) as zip_:
                img_path = join(zip_name, img_dir)
                if sys.platform == 'win32':
                    img_path = img_path.replace('\\', '/')
                with zip_.open(img_path+'.jpg') as image:
                    img = np.asarray(Image.open(image))
                    img = cv2.resize(img, size)

                    fnd = np.where(self.images_with_bbox[:, 1] == img_path)[0][0]
                    idx = np.append(idx, fnd)
                    imgs = np.append(imgs, img)

        return idx, imgs


if __name__ == '__main__':
    import time
    print('start')
    finder = Finder('lemon', size=(224, 224))
    t1 = time.time()
    # finder.bbox_test(n=3)
    finder.fill_images_with_bbox()
    t2 = time.time()
    print('multiprocessing took: {}'.format(t2-t1))
    np.save('lemon', finder.images_with_bbox)
    pass
