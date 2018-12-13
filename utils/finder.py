from os.path import split, join, exists
from utils import dumper, config, tools
import numpy.core.defchararray as char
from progressbar import progressbar
from os import makedirs
from glob2 import glob
import multiprocessing
import pandas as pd
import numpy as np
import imageio
import cv2
import sys

# region checkers
dumper.check_requirements()
IMG_DIRS = dumper.img_dirs()
JSON_ = dumper.json_loader()
# endregion


# noinspection PyTypeChecker
class Finder:
    def __init__(self, subject, size=None, etc=False,
                 just=False, address=None, is_group=None):
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
        self.data = np.delete(np.empty((1, config.ROW_LENGTH)), 0, 0)

        self.image_df = pd.DataFrame({config.headers.get(i): [] for i in config.DF_COLS})

        if not address:
            self.address = join(split(__file__)[0], 'data')
            if not exists(self.address):
                makedirs(self.address)
            for i in ["images/train", "images/validation", "images/test", "records"]:
                if not exists(join(self.address, i)):
                    makedirs(join(self.address, i))
        else:
            self.address = address

        if isinstance(subject, str):
            self.search(subject=subject.capitalize(), etc=self.etc, just=self.just)
            self.fill_search_result()
        else:
            raise ValueError("subject excepted str but got {}".format(type(subject).__name__))

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

            output_path = join(self.address, "records/{}.record".format(out))
            csv_out = join(self.address, "records/{}_bbox.csv".format(out))
            images_dir = join(self.address, "images/{}".format(out.lower()))

            self._extract_data_frame(folder_name=out)

            imgs_id = list(set((i[1] for i in self.image_df.itertuples())))
            tools.write('{} image'.format(len(imgs_id)))

            self._get_imgs_path_with_bboxes(imgs_id=imgs_id)
            tools.write('{} unique object'.format(self.data.shape[0]))

            if config.AVAILABLE_AS == "jpg":
                folders = glob(config.DATA_DIR + "{}/*/".format(out))
                folders = char.rpartition(np.array(folders), "/")[:, 0]
            elif config.AVAILABLE_AS == "zip":
                folders = None
            else:
                folders = None

            result = None
            if config.AVAILABLE_AS in ("jpg", "zip"):
                z = zip(folders, np.repeat(images_dir, len(folders)))
                pool = multiprocessing.Pool(threads)
                result = pool.map(func=self._imread_imwrite,
                                  iterable=z)
                pool.close()
                pool.join()

            elif config.AVAILABLE_AS == "csv":
                result = self._imread_imwrite(fl_n_out=(self.data[:, 1], images_dir))

            for folder in result:
                for row in folder:
                    self.data[row[0], config.WIDTH:config.HEIGHT+1] = row[1], row[2]

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
        with open(csv_out, 'w') as file:
            file.write("filename,width,height,class,xmin,ymin,xmax,ymax\n")
            for row in self.data:
                width, height = row[config.WIDTH:config.HEIGHT + 1]
                bbox = row[config.BBOX_SLICE]
                bbox = [int(i * j) for i, j in zip(bbox, [width, height, width, height])]
                name = row[config.IMG].split("/")[-1]
                cls = row[config.LABEL]
                out = '{},{},{},{},{},{},{},{}\n'.format(name, width, height, cls, *bbox)
                file.write(out)

    def bbox_test(self, target, n=4, thickness=3):
        tools.bbox_test(address=self.address, target=target,
                        n=n, thickness=thickness)

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
                        self._search_dig(i['Subcategory'])
                    elif 'Part' in i.keys():
                        self._search_dig(i['Part'])

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

    def _search_dig(self, list_):
        for i in list_:
            self.__search_result.append(i['LabelName'])
            if len(i.keys()) > 1:
                if 'Subcategory' in i.keys():
                    self._search_dig(i['Subcategory'])
                elif 'Part' in i.keys():
                    self._search_dig(i['Part'])

    def _extract_data_frame(self, folder_name):
        bbox_df = dumper.annotation_loader(folder_name=folder_name)
        for i in self.__search_result:
            if isinstance(self.is_group, str):
                self.image_df = self.image_df.append(
                    bbox_df.loc[(bbox_df['LabelName'] == i) & (bbox_df["IsGroupOf"] == self.is_group)])
            else:
                self.image_df = self.image_df.append(bbox_df.loc[bbox_df['LabelName'] == i])

    def _get_imgs_path_with_bboxes(self, imgs_id):
        for path in progressbar(IMG_DIRS, prefix="fetch object"):
            name = None

            if config.AVAILABLE_AS == "jpg":
                name = path.rsplit("/")[-1][:-4]
            elif config.AVAILABLE_AS == "csv":
                name = path[0]
                path = path[1]

            # find them
            if name in imgs_id:
                result = self.image_df.loc[self.image_df['ImageID'] == name,
                                           ['LabelName', 'XMin', 'YMin', 'XMax', 'YMax']]
                result = result.values

                # add info(path, width, height)
                tmp = np.c_[np.repeat(path, len(result)),
                            np.repeat(0, len(result)),
                            np.repeat(0, len(result)),
                            result]

                # path, LabelName, XMin, YMin, XMax, YMax
                self.data = np.append(self.data, tmp)

        self.data = self.data.reshape((-1, config.ROW_LENGTH))
        for label in np.unique(self.data[:, config.LABEL]):
            self.data[np.where(self.data[:, config.LABEL] == label), config.LABEL] = tools.mid_to_string(label)
        self.data[:, config.BBOX_SLICE] = self.data[:, config.BBOX_SLICE].astype(np.float)

    def _imread_imwrite(self, fl_n_out):
        folders = fl_n_out[0]
        out = fl_n_out[1]
        file_names = prefix = None
        ret = np.array([])
        state = config.AVAILABLE_AS

        if state == "jpg":
            dirs = np.unique(self.data[:, config.IMG]).astype(np.str)
            sep = char.rpartition(dirs, "/")
            file_names = sep[np.where(sep[:, 0] == folders)][:, 2]
            prefix = "{} ".format(folders.rsplit("/")[-1])
        elif state == "csv":
            file_names = np.unique(self.data[:, config.IMG])
            prefix = "Downloading "

        for img_dir in progressbar(file_names, prefix=prefix):
            save_to = join(out, img_dir) if state == "jpg" else join(out, img_dir.rsplit("/")[-1])
            save_to = save_to.replace('\\', '/') if sys.platform == "win32" else save_to
            imread = join(folders, img_dir) if state == "jpg" else img_dir

            if not exists(save_to):
                img = cv2.imread(imread) if state == "jpg" else imageio.imread(imread)
            else:
                img = cv2.imread(save_to)

            if isinstance(self.size, tuple):
                img = cv2.resize(img, self.size)

            height, width, _ = img.shape
            fnd = np.where(self.data[:, config.IMG] == imread)
            ret = np.append(ret, [fnd, width, height])
            cv2.imwrite(save_to, img)

        ret = ret.reshape((-1, 3))
        return ret if state == "jpg" else [ret, ]


if __name__ == '__main__':
    import time
    finder = Finder('apple', size=None, is_group=True)
    t1 = time.time()
    finder.extract_images()
    t2 = time.time()
    finder.bbox_test(target="train", n=3, thickness=5)
    print('multiprocessing took: {}'.format(t2 - t1))
