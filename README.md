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

## File Structure

* `data_gen.py`
	- does the split (train, test, valid) if necessary
	- tokenization if necessary
	- have patterns for each discourse marker we care about
	- make tuple (S1, S2, label) for each instance of discourse marker (if we can)
	- for each corpus (wikitext and gigaword, maybe book corpus):
		- make files `S1.txt`, `S2.txt`, `labels.txt`
	- also make master files aggregated accross "all" corpora
* `data.py`
	- takes in pre-processed files (S1, S2, label)
	- build vocabulary and save `vocab.dat`
	- process trimmed down glove
	- map each word onto id and generate `ids.txt` files
* `util.py`
	- exports `pair_iter`
	- should have an option to limit to a subset of labels

## To Do

high priority:

* ☑ decide what other discourse markers to include
   1). ☑ get candidate list of frequent markers from papers [Erin]
   2). ☑ vote on slack [all]
* extend corpus to include those discourse markers
   - ☐ record patterns for each marker [Erin and Allen]
   - ☐ finish sentence extraction code [Erin]
   - ☐ move extract_sentences.py code to data_gen.py [Erin]
   - ☐ revise `pair_iter` and data to use S1 file, S2 file, and labels file [Erin]
* get more training data
	- ☐ get Gigaword to Allen
	- ☐ look for Book Corpus
* set up for SentEval evaluation 
    2). load classifier to generate sentence representations (Allen)
* other evaluation tasks
    - infer sent
    - skip thought experiments

low priority:

* reimplement in pytorch [Erin]



