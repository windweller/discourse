Place download Wikitext-103 and put it in "data" directory,
simple unzip would do!

https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip

To preprocess the data and generate files of BECAUSE and BUT, use:
`python data_gen.py `

To generate the ids files with glove indices, download desired pretrained [GloVe](https://nlp.stanford.edu/projects/glove/) vectors and use `python data.py`.
