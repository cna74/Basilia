# Downloading DataSet

- Download Dataset from here
[Open-Image](https://www.figure-eight.com/dataset/open-images-annotated-with-bounding-boxes/)

-  store like this
```
└── Open_Image
    ├── class-descriptions-boxable.csv
    ├── bbox_labels_600_hierarchy.json
    ├── Test
    │   ├── test-annotations-bbox.csv
    │   ├── test-images.csv
    │   └── test
    ├── Train
    │   ├── train_00
    │   ├── train_01
    │   ├── train_02
    │   ├── train_03
    │   ├── train_04
    │   ├── train_05
    │   ├── train_06
    │   ├── train_07
    │   ├── train_08
    │   ├── train-annotations-bbox.csv
    │   └── train-images-boxable.csv
    └── Validation
        ├── validation-annotations-bbox.csv
        └── validation-images.csv
```

## install Basilia
- get package `git clone https://github.com/cna74/Basilia.git`
- go to  directory `cd Basilia`
- create a virtual environment `virtualenv -p ~/%your-python-path% /%venv dir%/` after replacing python path and venv dir it should be like this `virtualenv -p ~/anaconda3/bin/python3.6 ~/ElBasil`
- install requirements `pip install -r requierments.txt`

## in action
- open jupyter `jupyter lab` or `jupyter notebook`

![](https://github.com/cna74/Basilia/blob/master/Demo.png)

