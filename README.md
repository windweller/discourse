Place download Wikitext-103 and put it in "data" directory,
simple unzip would do!

https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip

To preprocess the data and generate files of BECAUSE and BUT, use:
`python data_gen.py `

To generate the ids files with glove indices, download desired pretrained [GloVe](https://nlp.stanford.edu/projects/glove/) vectors and use `python data.py`.

## Running Instructions

```
python train.py --run_dir wikitext_run_cause --task cause --epochs 3 --print_every 100 --dataset wikitext-103
```

## Examining Result

```
python train.py --best_epoch 3 --run_dir ptb_run_but --dev True --correct_example True
```

## Winograd Scheme

```
python train.py --best_epoch 1 --run_dir wiki_run_but --dev True --winograd True --dataset winograd
```

## Input format for models

* `{split}_S1.txt` - first part of sentence (e.g. before discourse marker), 1 sentence per line, space separated words
* `{split}_S2.txt` - second part of sentence (e.g. after discourse marker), 1 sentence per line, space separated words
* `{split}_Y.txt` - labels for 1 sentence per line (possibly multiple columns per line)

