import re
import io
import sys
import nltk
from os.path import join as pjoin

def get_wiki_pairs(file_path, connective):
    but_sents = []
    with io.open(file_path, 'rU', encoding="utf-8") as f:
        tokens = f.read().replace("\n", ". ")
        sent_list = nltk.sent_tokenize(tokens)
        print("sent num in total: " + str(len(sent_list)))
        for sent in sent_list:
            if connective in sent.split():
                # words = sent.split()
                # if "but" in words[1:]:  # no sentence from beginning has but
                but_sents.append(sent)

    print("discourse number: " + str(len(but_sents)))
    return but_sents


if __name__ == '__main__':
    wikitext_103_train_path = pjoin("data", "wikitext-103", "wiki.train.tokens")
    wikitext_103_valid_path = pjoin("data", "wikitext-103", "wiki.valid.tokens")

    # wikitext_103_train = get_wiki_pairs(wikitext_103_train_path, "then")
    wikitext_103_valid = get_wiki_pairs(wikitext_103_valid_path, "also")

    for i in range(10):
        print(wikitext_103_valid[i])
        print("")