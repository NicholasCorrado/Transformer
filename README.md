# Transformer

We use the code from [here](https://github.com/harvardnlp/annotated-transformer) as the foundation for this codebase.

Running:
```angular2html
python driver.py
```

## Dependencies
TODO: add env .yml file
```angular2html
python 3.8 (Haven't tried it with different versions)
numpy
pytorch
matplotlib
```

## Getting the datasets
From a comment in the 
Harvard NLP notebook:

I solved this problem, just go to this website: https://wit3.fbk.eu/2016-01, click "link" and download the "2016-01.tgz" file. In this file you can find the "2016-01/texts/de/en/de-en.tgz" file. Simply put this file in "your_work_dir/.data/iwslt/", then it works fine
