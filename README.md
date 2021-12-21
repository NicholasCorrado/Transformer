# Transformer

We use the code from [here](https://github.com/harvardnlp/annotated-transformer) as the foundation for this codebase.

## Financial Market Application
### Installation
```
conda env create --file environment.yml
```

### Replicating Results

We have only tested on CSL Linux machines, and do not guarantee support for other platforms.

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

## Getting the datasets
From a comment in the 
Harvard NLP notebook:

I solved this problem, just go to this website: https://wit3.fbk.eu/2016-01, click "link" and download the "2016-01.tgz" file. In this file you can find the "2016-01/texts/de/en/de-en.tgz" file. Simply put this file in "your_work_dir/.data/iwslt/", then it works fine
