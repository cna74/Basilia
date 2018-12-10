from os.path import split, join, exists
from utils import dumper, config, tools
import numpy.core.defchararray as char
from progressbar import progressbar
import matplotlib.pyplot as plt
from glob2 import glob
import multiprocessing
import pandas as pd
import numpy as np
import cv2
import sys
import os

# region checkers
dumper.check_requirements()
IMG_DIRS = dumper.img_dirs()
JSON_ = dumper.json_loader()
ID_SLICE = 0
IMG_SLICE = 1
LABEL_SLICE = 2
BBOX_SLICE = slice(3, 7)
headers = {0: 'ImageID', 1: 'Source', 2: 'LabelName', 3: 'Confidence',
           4: 'XMin', 5: 'XMax', 6: 'YMin', 7: 'YMax', 8: 'IsOccluded',
           9: 'IsTruncated', 10: 'IsGroupOf', 11: 'IsDepiction', 12: 'IsInside'}
# endregion


# noinspection PyTypeChecker
class Finder:
    def __init__(self, subject, size=(100, 100),
                 etc=False, just=False, address=None,
                 is_group=None):
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

        # empty numpy arrays to append later
        self.data = np.delete(np.empty((1, len(config.DF_COLS))), 0, 0)

        self.image_df = pd.DataFrame({headers.get(i): [] for i in config.DF_COLS})

        if not address:
            self.address = join(split(__file__)[0], 'data')
            if not exists(self.address):
                os.makedirs(self.address)
            for i in ["images/train", "images/validation", "images/test", "records"]:
                if not exists(join(self.address, i)):
                    os.makedirs(join(self.address, i))
        else:
            self.address = address

        if isinstance(subject, str):
            self.search(subject=subject.capitalize(), etc=self.etc, just=self.just)
            self.fill_search_result()

        self.classes = dict([(cls, i) for i, cls in enumerate(self.search_result, start=1)])
        sys.stdout.flush()

    def fill_search_result(self):
        tmp = []
        for i in self.__search_result:
            tmp.append(tools.mid_to_string(i))
        self.search_result = set(tmp)

    def extract_images(self, threads=None):
        if not threads:
            threads = multiprocessing.cpu_count()

        for out in ["Validation", "Test", "Train"]:
            tools.write(out)

            output_path = join(self.address, "records/{}.record".format(out.lower()))
            csv_out = join(self.address, "records/{}_bbox.csv".format(out.lower()))
            images_dir = join(self.address, "images/{}".format(out.lower()))

            self._extract_data_frame(folder_name=out)

            imgs_id = list(set((i[IMG_SLICE] for i in self.image_df.itertuples())))
            tools.write('{} image'.format(len(imgs_id)))

            self._get_imgs_path_with_bboxes(imgs_id=imgs_id)
            tools.write('{} unique object'.format(self.data.shape[0]))
            folders = glob(config.DATA_DIR + "{}/*/".format(out))

            pool = multiprocessing.Pool(threads)
            pool.map(func=self._imread_imwrite,
                     iterable=zip(folders, np.repeat(images_dir, len(folders))))
            pool.close()
            pool.join()

            self.save_csv(csv_out=csv_out)
            # save some memory for next step
            self.image_df = self.image_df[:0]
            del imgs_id
            self.data = np.delete(np.empty((1, len(config.DF_COLS))), 0, 0)

            # generate tf.record
            tools.generate(csv_input=csv_out, images_dir=images_dir,
                           output_path=output_path, classes=self.classes)
            tools.write("done")

    def save_csv(self, csv_out):
        width, height = self.size

        with open(csv_out, 'w') as file:
            file.write("filename,width,height,class,xmin,ymin,xmax,ymax\n")
            for row in self.data:
                bbox = row[BBOX_SLICE]
                bbox = [int(i * j) for i, j in zip(bbox, [width, height, width, height])]
                name = row[IMG_SLICE].split("/")[-1] + ".jpg"
                cls = row[LABEL_SLICE]
                out = '{},{},{},{},{},{},{},{}\n'.format(name, width, height, cls, *bbox)
                file.write(out)

    # todo not working at all
    def bbox_test(self, n=4):
        choose = np.unique(self.data[:, 0])
        choices = np.random.choice(len(choose), n*n, replace=False)

        plt.figure(figsize=(10, 10))
        for idx, choice in enumerate(choices, start=1):
            data = self.data[np.where(self.data[:, 0] == choice)]
            img = data[0, 1]
            bboxes = data[:, BBOX_SLICE]
            titles = set(data[:, LABEL_SLICE])
            plt.subplot(n, n, idx)
            img = tools.draw(img, bboxes, self.size)
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
            if i['LabelName'] == tools.mid_to_string(subject):
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
            if not etc and tools.mid_to_string(subject) in self.__search_result and len(self.__search_result) > 1:
                self.__search_result.remove(tools.mid_to_string(subject))
        else:
            self.__search_result = [tools.mid_to_string(subject), ]

    def _dig(self, list_):
        for i in list_:
            self.__search_result.append(i['LabelName'])
            if len(i.keys()) > 1:
                if 'Subcategory' in i.keys():
                    self._dig(i['Subcategory'])
                elif 'Part' in i.keys():
                    self._dig(i['Part'])

    def _extract_data_frame(self, folder_name):
        bbox_df = dumper.annotation_loader(folder_name=folder_name)
        for i in self.__search_result:
            if isinstance(self.is_group, str):
                self.image_df = self.image_df.append(
                    bbox_df.loc[(bbox_df['LabelName'] == i) & (bbox_df["IsGroupOf"] == self.is_group)])
            else:
                self.image_df = self.image_df.append(bbox_df.loc[bbox_df['LabelName'] == i])

    def _get_imgs_path_with_bboxes(self, imgs_id):
        # split image dirs string and fetch names
        img_names = char.rpartition(IMG_DIRS, '/')[:, -1]

        # selected image dirs
        selected = IMG_DIRS[np.where(np.isin(img_names, imgs_id) == True)]

        for i, path in enumerate(selected, start=1):
            name = path.rsplit("/")[-1]
            # find them
            result = self.image_df.loc[self.image_df['ImageID'] == name, ['LabelName', 'XMin', 'YMin', 'XMax', 'YMax']]
            result = result.values

            # add info(id, path)
            tmp = np.c_[np.repeat(i, len(result)).astype(np.uint),
                        np.repeat(path, len(result)),
                        result]

            # id, path, LabelName, XMin, YMin, XMax, YMax
            self.data = np.append(self.data, tmp)

        self.data = self.data.reshape((-1, len(config.DF_COLS)))
        for i in np.unique(self.data[:, LABEL_SLICE]):
            self.data[np.where(self.data[:, LABEL_SLICE] == i), LABEL_SLICE] = tools.mid_to_string(i)
            self.data[:, BBOX_SLICE] = self.data[:, BBOX_SLICE].astype(np.float)

    def _imread_imwrite(self, fl_n_out):
        folders = fl_n_out[0]
        out = fl_n_out[1]

        folders = folders[:-1]
        dst = join(folders, '{}.jpg')

        dirs = np.unique(self.data[:, 1]).astype(np.str)
        sep = char.rpartition(dirs, "/")
        file_names = sep[np.where(sep[:, 0] == folders)][:, 2]

        for img_dir in progressbar(file_names, prefix=folders[-2:] + " "):
            img_path = dst.format(img_dir)
            save_to = join(out, img_dir) + ".jpg"

            if sys.platform == 'win32':
                save_to = save_to.replace('\\', '/')

            img = cv2.imread(img_path)
            img = cv2.resize(img, self.size)
            cv2.imwrite(save_to, img)


if __name__ == '__main__':
    import time
    finder = Finder('fruit', size=(500, 500), etc=True, is_group=True)
    t1 = time.time()
    finder.extract_images()

    t2 = time.time()
    print('multiprocessing took: {}'.format(t2-t1))
    # finder.bbox_test(n=4)
    pass
