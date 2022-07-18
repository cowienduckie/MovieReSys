# The Movie Recommender System - Movie ReSys

Project for Introduction of Machine Learning subject

## Install packages

```shell
pip3 install wheel
pip3 install pandas
pip3 install sklearn
pip3 install nltk
pip3 install tkinter
pip3 install customtkinter
pip3 install pandastable
pip3 install surprise  
```

If `pip3 install surprise` command occurred errors, try to download [Anaconda](https://www.anaconda.com/products/distribution) then run below command with Anaconda command prompt

```shell
conda install -c conda-forge scikit-surprise
```

Or build Surprise from source code with the below commands (this required `numpy` and `cython`)

```shell
pip install numpy cython
git clone https://github.com/NicolasHug/surprise.git
cd surprise
python setup.py install
```

## Download datasets

Download 2 input datasets below then put them into the right folder in the input directory.

1. [the-movies-dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset): Full MovieLens Dataset
2. [movies-data](https://www.kaggle.com/datasets/bentan233/movies-data): forked from the above dataset for optimization

## Build and Run

```shell
cd src
python app.py
```
