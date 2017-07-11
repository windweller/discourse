"""
We load in the sentence representation
that is pre-trained and we evaluate on different
tasks including:
1. SNLI
2. SST
"""

from torchtext import data
from torchtext import datasets

from os.path import join as pjoin

if __name__ == '__main__':
    inputs = data.Field(lower=True)
    answers = data.Field(sequential=False)

    train, dev, test = datasets.SNLI.splits(inputs, answers, root="data/snli")
