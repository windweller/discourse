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

## Extracting Sentence Pairs and Discourse Markers

1. Grab all explicit discourse markers that showed up in PDTB as more than 1% of the tokens:
	* 
	*
2. Of these, filter out the ones that are very poorly behaved (we can't consistently automatically extract usable pairs)
	* and
	* also
	* as
	* then (tentatively: we can revisit this later)
3. Extraction rules for remaining discourse markers:
	* because, although
	* 

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

## Setting up AWS to run model

[helpful script here](https://bitbucket.org/jhong1/azure-gpu-setup/src/4f736634c9a714fba988e664805c19cf4ca05508/gpu-setup-part2.sh?at=master&fileviewer=file-view-default)

* Use this image: `Deep Learning AMI Ubuntu Linux - 2.2_Aug2017 - ami-599a7721`
* check that CUDA 8.0 is installed: `nvcc --version`. if it's not, bail. find another image or whatever.
* add new users
	```
	sudo adduser erindb
	sudo adduser anie
	```
	* add ssh keys to `~/.ssh/authorized_keys` for each user (see, e.g. [github.com/windweller.keys](https://github.com/windweller.keys))
* get tensorflow from [?](?)
* setup tensorflow with CUDA
	```
	scp ~/Downloads/cudnn-8.0-linux-x64-v5.1.tgz erindb@52.38.236.178:~/

	tar -xzvf cudnn-8.0-linux-x64-v5.1.tgz
	sudo cp -P cuda/include/cudnn.h /usr/local/cuda/include/
	sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64/
	sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

	sudo apt-get update
	sudo apt-get -y install python-dev libffi-dev libssl-dev libcupti-dev

	sudo pip install --upgrade pip
	pip install pyOpenSSL ndg-httpsclient pyasn1

	# update bashrc
	# Note: this will create duplicates if you run it more than once. Not elegant...
	echo "Updating bashrc"
	echo >> $HOME/.bashrc '
	export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
	export CUDA_HOME=/usr/local/cuda

	export PATH=$PATH:~/bin

	white="\[\033[1;37m\]"
	pink="\[\033[1;35m\]"
	yellow="\[\033[1;33m\]"
	green="\[\033[1;32m\]"
	blue="\[\033[1;36m\]"

	time="$pink\t"
	user="$yellow\u"
	host="$green\h"
	wd="$blue\w"

	export PS1="$time $user@$host:$wd$ $white"
	'

	source $HOME/.bashrc

	# create bash_profie
	# Note: this will destroy your existing .bash_profile if have one...
	echo "Creating bash_profile"
	echo > $HOME/.bash_profile '
	if [ -f ~/.bashrc ]; then
	    source ~/.bashrc
	fi
	'

	# install tensorflow
	TF_VERSION="https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp27-none-linux_x86_64.whl"

	export TF_BINARY_URL=$TF_VERSION
	sudo pip install --upgrade $TF_BINARY_URL
	```
* clone our repo
	```
	git clone https://github.com/windweller/discourse.git
	mkdir discourse/data/books
	```
* then on the data pre-processing instance:
	```
	scp *but_because_if_when_so.* erindb@52.38.236.178:~/discourse/data/books/
	```



