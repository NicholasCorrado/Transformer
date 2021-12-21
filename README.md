# Transformer

We use the code from [here](https://github.com/harvardnlp/annotated-transformer) as the foundation for this codebase.

This repo has two sections. Code for the language translation task is found in the `ipynb` directory. The rest of the repo is for the stock translation and sentence completion tasks. There are separate installation instructions for both applications.

We have only tested on CSL Linux machines, and do not guarantee support for other platforms.

## Concat Transformer: Financial Market Application
### Installation
```
conda env create --file environment.yml
conda install pytorch cudatoolkit=10.2 -c pytorch
```

### Data
Stock price data is in the `data` directory as `.npy` files.

### Replicating Results

To run all experiments for the stock translation task:
```angular2html
./run.sh
```
After running, you can run `python plot.py` to plot loss curves.

To run all experiments for the stock sentence completion task:
```angular2html
./run_2.sh
```
After running, you can run `python plot_2.py` to plot loss curves.

## Concat Transformer: Language Translation

All code for language translation tasks are found in the `ipynb` directory. All code is in iPython notebooks.

### Installation
Dependencies are listed below:
```
1. Requirement
Package                            Version
---------------------------------- -------------------
jupyter                            1.0.0
jupyter-client                     6.1.12
jupyter-console                    6.4.0
jupyter-core                       4.7.1
jupyter-packaging                  0.7.12
jupyter-server                     1.4.1
jupyterlab                         3.0.14
jupyterlab-pygments                0.1.2
jupyterlab-server                  2.4.0
jupyterlab-widgets                 1.0.0
matplotlib                         3.3.4
numpy                              1.20.1
scikit-learn                       0.24.1
seaborn                            0.11.1
spacy                              2.3.0
spacy-legacy                       3.0.8
torch                              1.10.0
torchtext                          0.11.0
```
### Getting the Data
1. Visit https://wit3.fbk.eu/ ;
2. Click "link" and download the corresponding file (e.g., dataset for WMT-14);
3. Find the "de-en.tgz" file (for English-German translation). Put this file in "your_work_dir/.data/iwslt/".

### Replicating Results
Run the cells in the .ipynb files in order.
