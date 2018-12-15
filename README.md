# Basilia
## Downloading DataSet

[Open-Image-V4](https://www.figure-eight.com/dataset/open-images-annotated-with-bounding-boxes/)
if you want to use online just download [csv](https://github.com/cna74/Basilia/blob/master/#csv) and [json](https://github.com/cna74/Basilia/blob/master/#)
- csv:

    |               |       |
    | ------------- | ----- |
    |[train-annotations-bbox.csv](https://datasets.figure-eight.com/figure_eight_datasets/open-images/train-annotations-bbox.csv)|[train-images-boxable.csv](https://datasets.figure-eight.com/figure_eight_datasets/open-images/train-images-boxable.csv)|    
    |[validation-annotations-bbox.csv](https://datasets.figure-eight.com/figure_eight_datasets/open-images/validation-annotations-bbox.csv)|[validation-images.csv](https://datasets.figure-eight.com/figure_eight_datasets/open-images/validation-images.csv)|    
    |[test-annotations-bbox.csv](https://datasets.figure-eight.com/figure_eight_datasets/open-images/test-annotations-bbox.csv)|[test-images.csv](https://datasets.figure-eight.com/figure_eight_datasets/open-images/test-images.csv)|
    |[labels](https://storage.googleapis.com/openimages/2018_04/class-descriptions-boxable.csv)|[json_hierarchy](https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy.json)|

- zip *: if you want to use it online zip files are't necessary

    |               |               |       |
    | ------------- | ------------- | ----- |
    |[train_00](https://datasets.figure-eight.com/figure_eight_datasets/open-images/train_00.zip)| [train_01](https://datasets.figure-eight.com/figure_eight_datasets/open-images/train_01.zip) |[train_02](https://datasets.figure-eight.com/figure_eight_datasets/open-images/train_02.zip)|
    |[train_03](https://datasets.figure-eight.com/figure_eight_datasets/open-images/train_03.zip)| [train_04](https://datasets.figure-eight.com/figure_eight_datasets/open-images/train_04.zip)|[train_05](https://datasets.figure-eight.com/figure_eight_datasets/open-images/train_05.zip)|
    |[train_06](https://datasets.figure-eight.com/figure_eight_datasets/open-images/train_06.zip)| [train_07](https://datasets.figure-eight.com/figure_eight_datasets/open-images/train_06.zip)| [train_08](https://datasets.figure-eight.com/figure_eight_datasets/open-images/train_05.zip)|

- store like files like this, stared items are't necessary for online use 

```
└── Open_Image
    ├── class-descriptions-boxable.csv
    ├── bbox_labels_600_hierarchy.json
    ├── Test
    │   ├── test-annotations-bbox.csv
    │   ├── test-images.csv
    │   └── test       *
    ├── Train
    │   ├── train_00   *
    │   ├── train_01   *
    │   ├── train_02   *
    │   ├── train_03   *
    │   ├── train_04   *
    │   ├── train_05   *
    │   ├── train_06   *
    │   ├── train_07   *
    │   ├── train_08   *
    │   ├── train-annotations-bbox.csv
    │   └── train-images-boxable.csv
    └── Validation
        ├── validation *
        ├── validation-annotations-bbox.csv
        └── validation-images.csv
```

## install object-detection-api
- from home directory `cd ~`
- clone repo `git clone https://github.com/tensorflow/models.git`
- go to research directory `cd models/research/`
- install object-detection-api `python3 setup.py install`

## install Basilia
- from home directory `cd ~`
- clone repo `git clone https://github.com/cna74/Basilia.git`
- go to  directory `cd Basilia`
- install requirements `pip install -r requierments.txt`

## edit `config.py`
- change directory to utils folder `cd ~/Basilia/utils/`
- open `config.py`
  ```python
  DATA_DIR="<path-to-open-image-directory>"
  # if you downloaded zip files and extracted them
  RESOURCE="jpg"
  # if you just downloaded csv files
  RESOURCE="csv"
  ```
- open jupyter `jupyter lab` or `jupyter notebook`

  ```python
  from Basil import Finder
  finder = Finder(subject=("fox", "tiger", "Jaguar"), out_dir="/home/cna/Desktop/", automate=True)
  ```
more examples in https://github.com/cna74/Basilia/blob/master/Demo.ipynb