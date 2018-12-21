from multiprocessing import Pool, cpu_count
from os.path import split, join, exists
from utils import dumper, config, tools
import numpy.core.defchararray as char
from progressbar import progressbar
from pandas import DataFrame
from shutil import rmtree
from sys import platform
from os import makedirs
from glob2 import glob
import numpy as np
import imageio
import cv2


class Finder:
    def __init__(self, subject=None, resource=config.RESOURCE, input_dir=config.DATA_DIR,
                 out_dir=None, just=False, other=True, is_group=None, is_occluded=None,
                 is_truncated=None, size=None, is_depiction=None, automate=False,
                 is_inside=None, just_count=False):
        """
        extract desired images with given conditions
        :param subject:
            subject's name that should be in `tools.dict_of_classes()`
        :param resource:
            "jpg" if images are downloaded already
            "csv" if need to download the selected subjects
        :param input_dir:
            `Open-Image` DataSet directory path
        :param out_dir:
            where to save images, out_dir=join(out_dir, "data")
        :param size:
            None uses original size of images and saves them in out_dir [Default]
            tuple resize images to this and saves them in out_dir
        :param just:
            True search_result=[subject,]
            False doesn't affect on search procedure
        :param other:
            True if subject has sub-class search_result all of the sub-classes, [Default]
            False if subject does't have any sub-class search_result would be empty,
            look at examples
        :param is_group:
            True finder will search for bboxes that represent a group of object in a bbox
            False finder will search for bboxes that represent a single object in bbox
            None finder will search for both type of bboxes (single and grouped) [Default]
        :param is_occluded:
            True Indicates that the object is occluded by another object in the image
            False object is't occluded by another object in the image
            None both of situation [Default]
        :param is_truncated:
            True Indicates that the object extends beyond the boundary of the image
            False Indicates that all part of the object is in boundary of the image
            None both of situation [Default]
        :param is_depiction:
            True Indicates that the object is a depiction (e.g., a cartoon or drawing of the object,
                                                           not a real physical instance).
            False object is not depiction
            None both of situation [Default]
        :param is_inside:
            True Indicates a picture taken from the inside of the object (e.g., a car interior or inside of a building).
            False object is not inside
            None both of situation [Default]
        :param just_count:
            True just_count_images set this True to just check how many images finder may extract
            False Default value and will extract images [Default]
        :param automate: True automate all steps of finder LIKE A BOSS!!!
                         False [Default]

        Examples ------------------------------------------------------------------------------------------

        >>> Finder(subject='apple', input_dir="/media/home/cna/backpack/dataset/Open-Image/")
            -> result = all kind(single and grouped) of apples in original size

        >>> Finder(subject='apple', size=(600, 600), input_dir="/media/home/cna/backpack/dataset/Open-Image/")
            -> result = all kind(single and grouped) of apples in (600, 600) size

        >>> Finder(subject='fruit', just=True, input_dir="/media/home/cna/backpack/dataset/Open-Image/")
            -> result = just kind(single and grouped) fruits(basically fruit is those type of fruit that are't exactly
                        labeled but we know they are fruit, like persimmon so fruits like this are in category named
                        fruit witch is actually THE OTHERS) in (600, 600) size

        >>> Finder(subject='fruit', etc=False input_dir="/media/home/cna/backpack/dataset/Open-Image/")
            -> result = all kind(single and grouped) of fruits(apple, orange, lemon, banana, ...) except other fruits

        >>> Finder(subject='fruit', etc=True, input_dir="/media/home/cna/backpack/dataset/Open-Image/")
            -> result = all kind(single and grouped) of fruits(Banana, Apple, Grapefruit, Lemon, Strawberry, Grape,
                        Tomato, Watermelon, Common fig, Pineapple, Mango, Pomegranate, Orange, Peach, Cantaloupe,
                        Pear) and the fruits(other type of fruits) in original size
        """

        # region params
        self.other = other
        self.just = just
        self.size = size
        self.just_count = just_count
        self.is_group = self.is_depiction = self.is_inside = self.is_truncated = self.is_occluded = ["0", "1"]
        # MID type of result
        self._search_result = []
        # Str type of result
        self.search_result = []
        # empty numpy array and data frame to append later
        self.image_df = None
        self.table = DataFrame(index=["Train", "Validation", "Test"], columns=['Images', 'Objects'], dtype=int)
        self.data = np.delete(np.empty((1, config.ROW_LENGTH)), 0, 0)
        # resources should be jpg or csv
        self.resource = resource
        # Open-Image DataSet path
        self.out_dir = out_dir
        self.input_dir = input_dir
        self.dirs = ("Train", "Validation", "Test")
        # endregion

        # region conditions
        if isinstance(is_group, bool):
            self.is_group = ["1", ] if is_group else ["0", ]
        if isinstance(is_depiction, bool):
            self.is_depiction = ["1", ] if is_depiction else ["0", ]
        if isinstance(is_inside, bool):
            self.is_inside = ["1", ] if is_inside else ["0", ]
        if isinstance(is_truncated, bool):
            self.is_truncated = ["1", ] if is_truncated else ["0", ]
        if isinstance(is_occluded, bool):
            self.is_occluded = ["1", ] if is_occluded else ["0", ]
        # endregion

        # region out_dir
        if not just_count:
            self.out_dir = join(split(__name__)[0], 'data')
            if out_dir:
                self.out_dir = join(out_dir, 'data')
            if exists(self.out_dir):
                y_n = input("\n{} is already exist,\n"
                            "(y) remove files and folders in it\n"
                            "(Enter) i was working on this dir now i want to resume\n"
                            "(y/n)?".format(self.out_dir))
                if y_n == "y":
                    rmtree(self.out_dir)
                else:
                    print("resuming ")
            else:
                makedirs(self.out_dir)
            for d in ["images/Train", "images/Validation", "images/Test", "records"]:
                mkd = join(self.out_dir, d)
                if not exists(mkd):
                    makedirs(mkd)
        # endregion

        # region subject
        if isinstance(subject, str):
            self.search(subject=subject.capitalize(), other=self.other, just=self.just)
            self._fill_search_result()
        elif isinstance(subject, (tuple, list)):
            for subj in subject:
                self.search(subject=subj.capitalize(), other=self.other, just=self.just)
                self._fill_search_result()
            self._search_result = [tools.mid_to_string(s) for s in self.search_result]
        elif not subject:
            self._search_result = dumper.label_loader(dir_=self.input_dir).code.tolist()
            self._fill_search_result()
        else:
            raise ValueError("subject excepted str or iterable(tuple, list) but got {}".format(type(subject).__name__))
        # endregion

        self.classes = dict([(cls, i) for i, cls in enumerate(self.search_result, start=1)])
        tools.colored_print("searching for {}".format(self.search_result), text_color="blue")

        # region automate
        if automate:
            self.extract_images()
            if not self.just_count:
                self.bbox_test(thickness=10)
            print(self.table)
        # endregion

    def extract_images(self, dirs=None, threads=cpu_count()):
        dirs = dirs if dirs is not None and isinstance(dirs, tuple) else self.dirs

        for out in dirs:
            tools.colored_print(out, text_color="cyan", condition=not self.just_count)
            result = output_path = csv_out = images_dir = None
            a = b = "unknown"
            if not self.just_count:
                output_path = join(self.out_dir, "records/{}.record".format(out))
                csv_out = join(self.out_dir, "records/{}_bbox.csv".format(out))
                images_dir = join(self.out_dir, "images/{}".format(out))

            self._extract_data_frame(folder_name=out)
            imgs_id = list(set((i[1] for i in self.image_df.itertuples())))

            a = len(imgs_id)
            tools.colored_print("{} images".format(a), text_color="green", condition=not self.just_count)
            self.table.loc[out] = a, b
            if self.just_count:
                continue
            if a > 0:
                self._get_imgs_path_with_bboxes(imgs_id=imgs_id)
            b = len(self.data)
            tools.colored_print("{} objects".format(b), text_color="green", condition=not self.just_count)

            self.table.loc[out] = a, b
            if b > 0:
                if self.resource == "jpg":
                    folders = glob(self.input_dir + "{}/*/".format(out))
                    folders = char.rpartition(np.array(folders), "/")[:, 0]

                    z = zip(folders, np.repeat(images_dir, len(folders)))
                    pool = Pool(threads)
                    result = pool.map(func=self._imread_imwrite, iterable=z)
                    pool.close()
                    pool.join()

                elif self.resource == "csv":
                    result = self._imread_imwrite(fl_n_out=(self.data[:, config.IMG], images_dir))

                for folder in result:
                    for row in folder:
                        self.data[row[0], config.WIDTH:config.HEIGHT + 1] = row[1], row[2]

                self.save_csv(csv_out=csv_out)
                # save some memory for next step
                self.image_df = self.image_df[:0]
                del imgs_id
                self.data = np.delete(np.empty((1, len(config.DF_COLS))), 0, 0)

                # generate tf.record
                tools.generate(csv_input=csv_out, images_dir=images_dir, output_path=output_path, classes=self.classes)

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

    def bbox_test(self, max_=4, thickness=4):
        select = self.table.loc[((self.table.Images > 0) & (self.table.Objects > 0))]
        select = select.head(1)
        if not select.empty:
            target = select.index[0]
            n = np.modf(np.sqrt(np.arange(select.Images[0])))[1]
            n = int(n[np.where(n <= max_)].max())
            tools.bbox_test(address=self.out_dir, target=target, n=n, thickness=thickness)

    def search(self, subject, other, just):
        self._search_begin(subject=subject)
        self._search_harvest(subject=subject, other=other, just=just)

    def _search_begin(self, subject, json=None):
        subject = subject.capitalize()
        if not json:
            self._search_result = []
            json = dumper.json_loader(dir_=self.input_dir)

        for node in json:
            if node['LabelName'] == tools.mid_to_string(subject, dir_=self.input_dir):
                self._search_result.append(node['LabelName'])
                if len(node.keys()) > 1:
                    if 'Subcategory' in node.keys():
                        self._search_dig(node['Subcategory'])
                    elif 'Part' in node.keys():
                        self._search_dig(node['Part'])

            elif len(node.keys()) > 1:
                if 'Subcategory' in node.keys():
                    self._search_begin(subject=subject, json=node['Subcategory'])
                elif 'Part' in node.keys():
                    self._search_begin(subject=subject, json=node['Part'])

    def _search_harvest(self, subject, other, just):
        if just:
            self._search_result = [tools.mid_to_string(subject, dir_=self.input_dir), ]
        else:
            if not other and tools.mid_to_string(subject, dir_=self.input_dir) in self._search_result and len(
                    self._search_result) > 1:
                self._search_result.remove(tools.mid_to_string(subject, dir_=self.input_dir))

    def _search_dig(self, list_):
        for i in list_:
            self._search_result.append(i['LabelName'])
            if len(i.keys()) > 1:
                if 'Subcategory' in i.keys():
                    self._search_dig(i['Subcategory'])
                elif 'Part' in i.keys():
                    self._search_dig(i['Part'])

    def _fill_search_result(self):
        tmp = set()
        for i in self._search_result:
            tmp.add(tools.mid_to_string(i, dir_=self.input_dir))
        self.search_result.extend(tmp)

    def _extract_data_frame(self, folder_name):
        bbox_df = dumper.annotation_loader(folder_name=folder_name, dir_=self.input_dir)
        self.image_df = bbox_df.loc[
            (bbox_df['LabelName'].isin(self._search_result)) &
            (bbox_df["IsGroupOf"].isin(self.is_group)) &
            (bbox_df["IsDepiction"].isin(self.is_depiction)) &
            (bbox_df["IsOccluded"].isin(self.is_occluded)) &
            (bbox_df["IsTruncated"].isin(self.is_truncated)) &
            (bbox_df["IsInside"].isin(self.is_inside)),
            ['ImageID', 'LabelName', 'XMin', 'YMin', 'XMax', 'YMax']]

    def _get_imgs_path_with_bboxes(self, imgs_id):
        img_dirs = dumper.img_dirs(resource=self.resource, dir_=self.input_dir)
        for path in progressbar(img_dirs, prefix="find objects"):
            name = None

            if self.resource == "jpg":
                name = path.rsplit("/")[-1][:-4]
            elif self.resource == "csv":
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
            self.data[np.where(self.data[:, config.LABEL] == label),
                      config.LABEL] = tools.mid_to_string(label, dir_=self.input_dir)
        self.data[:, config.BBOX_SLICE] = self.data[:, config.BBOX_SLICE].astype(np.float)
        self.data[:, config.IMG] = self.data[:, config.IMG].astype(np.str)

    def _imread_imwrite(self, fl_n_out):
        folders = fl_n_out[0]
        out = fl_n_out[1]
        file_names = prefix = None
        ret = np.array([])

        if self.resource == "jpg":
            dirs = np.unique(self.data[:, config.IMG]).astype(np.str)
            sep = char.rpartition(dirs, "/")
            file_names = sep[np.where(sep[:, 0] == folders)][:, 2]
            prefix = "{} ".format(folders.rsplit("/")[-1])
        elif self.resource == "csv":
            file_names = np.unique(self.data[:, config.IMG])
            prefix = "Downloading "

        for img_dir in progressbar(file_names, prefix=prefix):
            save_to = join(out, img_dir) if self.resource == "jpg" else join(out, img_dir.rsplit("/")[-1])
            save_to = save_to.replace('\\', '/') if platform == "win32" else save_to
            img_dir = join(folders, img_dir) if self.resource == "jpg" else img_dir

            if not exists(save_to):
                downloaded_before = False
                img = cv2.imread(img_dir) if self.resource == "jpg" else imageio.imread(img_dir)
            else:
                downloaded_before = True
                img = cv2.imread(save_to)

            if isinstance(self.size, tuple):
                img = cv2.resize(img, self.size)

            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            if self.resource == "csv" and not downloaded_before:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            height, width = img.shape[:2]
            fnd = np.where(self.data[:, config.IMG] == img_dir)
            ret = np.append(ret, [fnd, width, height])
            cv2.imwrite(save_to, img)

        ret = ret.reshape((-1, 3))
        return ret if self.resource == "jpg" else [ret, ]


if __name__ == '__main__':
    Finder(subject="apple", is_group=True, is_depiction=True, is_inside=True, resource="jpg",
           out_dir="/home/cna/Desktop/", input_dir="/media/cna/backpack/dataset/Open-Image/", automate=True)
