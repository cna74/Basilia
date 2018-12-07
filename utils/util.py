from os.path import split, join, exists
import numpy.core.defchararray as char
from progressbar import progressbar
import matplotlib.pyplot as plt
import multiprocessing
import pandas as pd
import numpy as np
import dumper
import config
import cv2
import sys
import os

# region checkers
dumper.check_requirements()
LABELS = dumper.label_loader()
IMG_DIRS = dumper.img_loader()
JSON_ = dumper.json_loader()
id_slice = 0
img_slice = 1
label_slice = 2
bbox_slice = slice(3, 7)
headers = {0: 'ImageID', 1: 'Source', 2: 'LabelName', 3: 'Confidence',
           4: 'XMin', 5: 'XMax', 6: 'YMin', 7: 'YMax', 8: 'IsOccluded',
           9: 'IsTruncated', 10: 'IsGroupOf', 11: 'IsDepiction', 12: 'IsInside'}
# endregion

# region functions


def write(s):
    sys.stdout.write(s + "\n")
    sys.stdout.flush()


def dict_of_all_classes() -> dict:
    raw = list(set([i[0] for i in LABELS.itertuples()]))
    classes = {}
    for j in raw:
        fnd = Finder(subject=j, etc=True)
        result = list(fnd.search_result)
        if not len(result) == 1:
            classes.update({j: result})
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


def draw(img: np.ndarray, bboxes: np.ndarray, size: tuple):
    bboxes = np.multiply(size[0], bboxes.astype(np.float))
    bboxes = bboxes.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for b in bboxes:
        img = cv2.rectangle(img=img, pt1=(b[0], b[1]), pt2=(b[2], b[3]), color=(0, 0, 255), thickness=2)
    return img


# endregion

# noinspection PyTypeChecker
class Finder:
    def __init__(self, subject, size=(224, 224), etc=False, just=False, address=None, is_group=None):
        self.subject = subject
        self.etc = etc
        self.just = just
        self.size = size

        if isinstance(is_group, bool):
            self.is_group = "1" if is_group else "0"
        else:
            self.is_group = is_group

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

        if isinstance(subject, str):
            self.search(subject=subject.capitalize(), etc=self.etc, just=self.just)
            self.fill_search_result()

        elif isinstance(subject, (list, tuple, set)):
            # todo fix it
            self.search_result = [str(i).capitalize() for i in subject]
            self.__search_result = [mid_to_string(i) for i in self.search_result]

        sys.stdout.flush()

    def fill_search_result(self):
        tmp = []
        for i in self.__search_result:
            tmp.append(mid_to_string(i))
        self.search_result = set(tmp)

    def restore(self, dst):
        self.images_with_bbox: np.ndarray = np.load(dst)
        self.search_result = np.unique(self.images_with_bbox[:, label_slice])

    def fill_images_with_bbox(self, threads=None):
        if not threads:
            threads = multiprocessing.cpu_count()

        if len(self.images_with_bbox) == 0:
            self._extract_data_frame()

            idx = list(set((i[1] for i in self.image_df.itertuples())))
            write('{} image'.format(len(idx)))

            self._get_imgs_path_with_bboxes(idx)
            write('{} unique object'.format(self.images_with_bbox.shape[0]))
            del idx
            zip_name = np.unique([i[:8] for i in self.images_with_bbox[:, 1]])

            pool = multiprocessing.Pool(threads)
            out = pool.map(func=self._read_img_path, iterable=zip_name)
            pool.close()
            pool.join()

            write("preparing")

            for idxs, imgs in out:
                imgs = imgs.reshape((-1, self.size[0], self.size[1], 3))
                imgs = imgs.astype(np.uint8)
                for i, j in zip(idxs, imgs):
                    self.images_with_bbox[i, 1] = j

            write("done")

        else:
            write("it's already filled")

    def store_data(self):
        width, height = self.size

        np.save('{}/{}-[etc={}]-[just={}]-[is_group={}]'.format(
            self.address, str(self.subject), self.etc, self.just, self.is_group), self.images_with_bbox)

        with open(join(self.address, 'bbox'), 'w') as file:
            file.write("filename,width,height,class,xmin,ymin,xmax,ymax\n")
            for cls in self.search_result:
                label_folder = join(self.address, cls)

                try:
                    os.makedirs(label_folder)
                except Exception:
                    pass

                y = np.unique(self.images_with_bbox[:, 0], return_index=True, return_counts=True)

                for n, idx, count in zip(*y):
                    img = self.images_with_bbox[idx, :][1]
                    bboxes = self.images_with_bbox[np.where(self.images_with_bbox[:, 0] == n), bbox_slice][0]

                    name = '{}'.format(str(n).zfill(5)) + ".jpg"
                    dst = join(label_folder, name)
                    if not exists(dst) and isinstance(img, np.ndarray):
                        cv2.imwrite(dst, img)
                    for bbox in bboxes:
                        # multiply bbox in size
                        bbox = [int(i * j) for i, j in zip(bbox, [width, height, width, height])]
                        file.write('{},{},{},{},{},{},{},{}\n'.format(
                            name, width, height, cls, *bbox))

    def bbox_test(self, size=None, n: int = 4):
        if not size:
            size = self.size

        choose = np.unique(self.images_with_bbox[:, 0])
        choices = np.random.choice(len(choose), n*n, replace=False)

        plt.figure(figsize=(10, 10))
        for idx, choice in enumerate(choices, start=1):
            data = self.images_with_bbox[np.where(self.images_with_bbox[:, 0] == choice)]
            img = data[0, 1]
            bboxes = data[:, bbox_slice]
            titles = set(data[:, label_slice])
            plt.subplot(n, n, idx)
            img = draw(img, bboxes, size)
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
            if i['LabelName'] == mid_to_string(subject):
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
            if not etc and mid_to_string(subject) in self.__search_result and len(self.__search_result) > 1:
                self.__search_result.remove(mid_to_string(subject))
        else:
            self.__search_result = [mid_to_string(subject), ]

    def _extract_data_frame(self):
        bbox_df = dumper.bbox_loader()
        for i in self.__search_result:
                if isinstance(self.is_group, str):
                    self.image_df = self.image_df.append(
                        bbox_df.loc[(bbox_df['LabelName'] == i) & (bbox_df["IsGroupOf"] == self.is_group)])
                else:
                    self.image_df = self.image_df.append(bbox_df.loc[bbox_df['LabelName'] == i])

    def _dig(self, list_):
        for i in list_:
            self.__search_result.append(i['LabelName'])
            if len(i.keys()) > 1:
                if 'Subcategory' in i.keys():
                    self._dig(i['Subcategory'])
                elif 'Part' in i.keys():
                    self._dig(i['Part'])

    def _get_imgs_path_with_bboxes(self, imgs_id):
        # split image dirs string and fetch names
        img_names = char.partition(IMG_DIRS, '/')[:, 2]

        # selected image dirs
        selected = IMG_DIRS[np.where(np.isin(img_names, imgs_id) == True)]

        for i, path in enumerate(selected, start=1):
            name = path[9:]
            # find them
            result = self.image_df.loc[self.image_df['ImageID'] == name, ['LabelName', 'XMin', 'YMin', 'XMax', 'YMax']]
            result = result.values

            # add info(id and path of image)
            tmp = np.c_[np.repeat(i, len(result)).astype(np.uint),
                        np.repeat(path, len(result)),
                        result]

            self.images_with_bbox = np.append(self.images_with_bbox, tmp)

        self.images_with_bbox = self.images_with_bbox.reshape((-1, len(config.DF_COLS)))

        # convert labels
        for i in np.unique(self.images_with_bbox[:, label_slice]):
            self.images_with_bbox[np.where(self.images_with_bbox[:, label_slice] == i), label_slice] = mid_to_string(i)

        # reformat bboxes
        self.images_with_bbox[:, bbox_slice] = self.images_with_bbox[:, bbox_slice].astype(np.float)

    def _read_img_path(self, zip_name, size: tuple = None, z=False):
        if not size:
            size = self.size

        idx = np.array([], dtype=int)
        imgs = np.array([], dtype=int)

        if z:
            dst = join(dumper.DATA_DIR, 'Train/{}.zip')
        else:
            dst = join(dumper.DATA_DIR, 'Train/{}.jpg')

        dirs = np.unique(self.images_with_bbox[:, 1])
        dirs = dirs.astype(np.str)

        sep = char.partition(dirs, "/")
        file_names = sep[np.where(sep[:, 0] == zip_name)][:, 2]

        for img_dir in progressbar(file_names, prefix=zip_name+'  '):
            img_path = join(zip_name, img_dir)
            dir_ = dst.format(img_path)

            if sys.platform == 'win32':
                img_path = img_path.replace('\\', '/')

            img = cv2.imread(dir_)
            img = cv2.resize(img, size)

            fnd = np.where(self.images_with_bbox[:, 1] == img_path)[0][0]
            idx = np.append(idx, fnd)
            imgs = np.append(imgs, img)

        return idx, imgs


if __name__ == '__main__':
    import time
    finder = Finder('apple', is_group=False)
    t1 = time.time()
    finder.fill_images_with_bbox()
    t2 = time.time()
    print('multiprocessing took: {}'.format(t2-t1))
    finder.store_data()
    finder.bbox_test(n=4)
    pass
