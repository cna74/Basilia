# Downloading DataSet

- Download Dataset from here
[Open-Image](https://www.figure-eight.com/dataset/open-images-annotated-with-bounding-boxes/ "Open-Image")

-  store it like this
```
└── Open_Image
    ├── class-descriptions-boxable.csv
    ├── Test
    │   ├── test-annotations-bbox.csv
    │   ├── test-images.csv
    │   └── test.zip
    ├── Train
    │   ├── train_00.zip
    │   ├── train_01.zip
    │   ├── train_02.zip
    │   ├── train_03.zip
    │   ├── train_04.zip
    │   ├── train_05.zip
    │   ├── train_06.zip
    │   ├── train_07.zip
    │   ├── train_08.zip
    │   ├── train-annotations-bbox.csv
    │   └── train-images-boxable.csv
    └── Validation
        ├── validation-annotations-bbox.csv
        └── validation-images.csv
```

## install Basilia
- get package `git clone https://github.com/cna74/Basilia.git`
- go to  directory `cd Basilia`
- create a virtual enviornment `virtualenv -p ~/%your-python-path% /%venv dir%/` after replacing python path and venv dir it should be like this `virtualenv -p ~/anaconda3/bin/python3.6 ~/ElBasil`
- install requirements `pip install -r requierments.txt`

## in action
- open jupyter `jupyter lab` or `jupyter notebook`

![](https://github.com/cna74/Basilia/blob/master/Demo.png)

