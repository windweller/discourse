#!/usr/bin/env bash

python data.py --source_dir data/ptb --glove_dir data/glove.6B \
	--vocab_dir data/ptb --glove_dim 300

python data.py --source_dir data/wikitext-103 --glove_dir data/glove.6B \
	--vocab_dir data/wikitext-103 --glove_dim 300

# #     parser.add_argument("--source_dir", default=source_dir)
# #     parser.add_argument("--glove_dir", default=glove_dir)
# #     parser.add_argument("--vocab_dir", default=vocab_dir)
# #     parser.add_argument("--glove_dim", default=100, type=int)
# #     parser.add_argument("--random_init", action='store_true')


# CODE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# export PYTHONPATH=$PYTHONPATH:$CODE_DIR

# DATA_DIR=data
# GLOVE_DIR="glove.6B"
# GLOVE_FILE="$DATA_DIR/$GLOVE_DIR.zip"

# if [ -f "$GLOVE_FILE" ]; then
# 	echo "File $GLOVE_FILE exists."
# else
# 	GLOVE_URL="http://nlp.stanford.edu/data/$GLOVE_DIR.zip"
# 	echo "File $GLOVE_FILE does not exist. Downloading from $GLOVE_URL"
# 	cd $DATA_DIR
# 	wget $GLOVE_URL
# 	unzip $GLOVE_FILE
# 	cd ..
# fi

# # wget 
# # mv $GLOVE_DIR.zip $DATA_DIR
# # unzip $DATA_DIR/$GLOVE_DIR.zip

# # python data.py --source_dir $DATA_DIR --glove_dir

# # python2 $CODE_DIR/preprocessing/dwr.py

# # # Data processing for TensorFlow
# # python2 $CODE_DIR/qa_data.py --glove_dim 100
