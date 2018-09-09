from dataset import dumper
import json


class Finder:

    def __init__(self, search_for=None):
        self.labels = dumper.label_loader()
        self.bbox = dumper.bbox_loader()
        self.img_dirs = dumper.img_loader()
        self.json = json.load(open('bbox_labels_600_hierarchy.json'))['Subcategory']

        self.lst = []
        if search_for:
            self.search(search_for)

    def mid_to_string(self, mid_or_name) -> str:
        rev = False if mid_or_name.startswith('/m') else True
        if rev:
            sel = self.labels.loc[mid_or_name]
            sel = sel.to_dict()
            return sel['code']
        else:
            sel = self.labels.loc[self.labels['code'] == mid_or_name]
            sel = sel.to_dict()
            return list(sel['code'].keys())[0]

    def _dig(self, l):
        for i in l:
            self.lst.append(i['LabelName'])
            if len(i.keys()) > 1:
                if 'Subcategory' in i.keys():
                    self._dig(i['Subcategory'])
                elif 'Part' in i.keys():
                    self._dig(i['Part'])

    def search(self, subject, list_=None) -> set:
        if not list_:
            self.lst = []
            list_ = self.json
        for i in list_:
            if i['LabelName'] == self.mid_to_string(subject):
                self.lst.append(i['LabelName'])
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
        return set(self.lst)

    def convert(self, list_=None):
        tmp = []
        if not list_:
            list_ = self.lst
        for i in list_:
            tmp.append(self.mid_to_string(i))
        return tmp


# finder = Finder('Food')
# print(finder.lst)
# print(finder.convert())
