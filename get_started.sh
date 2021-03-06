#!/usr/bin/env bash

CODE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PYTHONPATH:$CODE_DIR

DATA_DIR=data
GLOVE_DIR="glove.6B"
GLOVE_FILE="$GLOVE_DIR.zip"

if [ -f "$DATA_DIR/$GLOVE_FILE" ]; then
	echo "File $GLOVE_FILE exists."
else
	GLOVE_URL="http://nlp.stanford.edu/data/$GLOVE_DIR.zip"
	echo "File $GLOVE_FILE does not exist. Downloading from $GLOVE_URL"
	cd $DATA_DIR
	wget $GLOVE_URL
	unzip $GLOVE_FILE -d glove.6B
	cd ..
fi

python extract_sentences.py

python data.py --source_dir data/ptb --glove_dir data/glove.6B \
	--vocab_dir data/ptb --glove_dim 100

python data.py --source_dir data/wikitext-103 --glove_dir data/glove.6B \
	--vocab_dir data/wikitext-103 --glove_dim 100

python data.py --source_dir data/winograd --glove_dir data/glove.6B \
	--vocab_dir data/winograd --glove_dim 100
