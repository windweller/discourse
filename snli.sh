#!/usr/bin/env bash

DATA_DIR=data
SNLI_DIR="snli"
SNLI_FILE="snli_1.0.zip"

if [ -f "$DATA_DIR/$SNLI_FILE" ]; then
	echo "File $SNLI_FILE exists."
else
    SNLI_URL=https://nlp.stanford.edu/projects/snli/snli_1.0.zip
    echo "File $SNLI_FILE does not exist. Downloading from $SNLI_URL"
    cd $DATA_DIR
	wget $SNLI_URL
	unzip $SNLI_FILE -d snli
	cd ..
fi

