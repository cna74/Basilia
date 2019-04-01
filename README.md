# Basilia
An API for [Open-Image-V4](https://www.figure-eight.com/dataset/open-images-annotated-with-bounding-boxes/) to gather your desired images with bboxes in just **TWO LINES OF CODE!**
[![open-image-v4-1024x655.png](https://i.postimg.cc/mrh0X00q/open-image-v4-1024x655.png)](https://postimg.cc/8FQKsnkB)

## Downloading DataSet

if you want to use online just download [csv](https://github.com/cna74/Basilia/blob/master/#csv) and [json](https://github.com/cna74/Basilia/blob/master/#)
- csv:

    |               |       |
    | ------------- | ----- |
    |[train-annotations-bbox.csv](https://datasets.figure-eight.com/figure_eight_datasets/open-images/train-annotations-bbox.csv)|[train-images-boxable.csv](https://datasets.figure-eight.com/figure_eight_datasets/open-images/train-images-boxable.csv)|    
    |[validation-annotations-bbox.csv](https://datasets.figure-eight.com/figure_eight_datasets/open-images/validation-annotations-bbox.csv)|[validation-images.csv](https://datasets.figure-eight.com/figure_eight_datasets/open-images/validation-images.csv)|    
    |[test-annotations-bbox.csv](https://datasets.figure-eight.com/figure_eight_datasets/open-images/test-annotations-bbox.csv)|[test-images.csv](https://datasets.figure-eight.com/figure_eight_datasets/open-images/test-images.csv)|
    |[class-descriptions-boxable.csv](https://storage.googleapis.com/openimages/2018_04/class-descriptions-boxable.csv)|[json_hierarchy](https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy.json)|

- zip *: if you want to use it online zip files are't necessary

    |               |               |       |
    | ------------- | ------------- | ----- |
    |[train_00](https://datasets.figure-eight.com/figure_eight_datasets/open-images/train_00.zip)| [train_01](https://datasets.figure-eight.com/figure_eight_datasets/open-images/train_01.zip) |[train_02](https://datasets.figure-eight.com/figure_eight_datasets/open-images/train_02.zip)|
    |[train_03](https://datasets.figure-eight.com/figure_eight_datasets/open-images/train_03.zip)| [train_04](https://datasets.figure-eight.com/figure_eight_datasets/open-images/train_04.zip)|[train_05](https://datasets.figure-eight.com/figure_eight_datasets/open-images/train_05.zip)|
    |[train_06](https://datasets.figure-eight.com/figure_eight_datasets/open-images/train_06.zip)| [train_07](https://datasets.figure-eight.com/figure_eight_datasets/open-images/train_06.zip)| [train_08](https://datasets.figure-eight.com/figure_eight_datasets/open-images/train_05.zip)|
    |[validation](https://datasets.figure-eight.com/figure_eight_datasets/open-images/validation.zip)|[test](https://datasets.figure-eight.com/figure_eight_datasets/open-images/test.zip)| |
    download and extract them

- store files like this, stared items are't necessary for online use 

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

## create environment [optional]
- `conda create -n ElBasil python=3.6`
- `source activate ElBasil`

## install Basilia
- clone repo `git clone https://github.com/cna74/Basilia.git`
- install requirements `pip install -r Basilia/requirements.txt`

## install object-detection-api
- clone repo `git clone https://github.com/tensorflow/models.git`
- move to research directory `cd models/research/`
- install object-detection-api `python3 setup.py install`

## edit `config.py`
- open `Basilia/utils/config.py`
  ```python
  DATA_DIR="<path-to-open-image-directory>"
  # if you downloaded zip files and extracted them
  RESOURCE="jpg"
  # if you just downloaded csv files
  RESOURCE="csv"
  ```
- from directory `Basilia/`:

  ```python
  from Basil import Finder
  # find Punching bags
  Finder(subject="Punching bag", out_dir="/home/cna/Desktop/", automate=True)
  ```
  [![Punching-bag.png](https://i.postimg.cc/hPgq0ZM8/output-1-9.png)](https://postimg.cc/jC89xXpC)
  ```python
  from Basil import Finder
  # find fox, tiger and jaguar
  Finder(subject=("fox", "tiger", "Jaguar"), out_dir="/home/cna/Desktop/", automate=True)
  ```
  [![Animal.png](https://i.postimg.cc/br0fh1Y2/output-7-13.png)](https://postimg.cc/m1r53zHb)
  ```python
  from Basil import Finder
  # just count all type of fruits, won't extract them
  Finder(subject="fruit", out_dir="/home/cna/Desktop/", just_count=True, automate=True)
  '''  just count images:
            Images  Objects
  Train        13271        0
  Validation     786        0
  Test          2545        0
  '''
  ```
more examples in https://github.com/cna74/Basilia/blob/master/Examples.ipynb

### Tutorial
[part 1: Using Basilia to gather data](https://www.youtu.be/frCVWcl8eas)

[part 2: understanding Basilia [conditions]](https://www.youtu.be/hpAIZRFbseM)