#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import io
import nltk
import pickle
import requests
import re

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import os
from os.path import join as pjoin

import json
from itertools import izip

np.random.seed(123)

_PAD = b"<pad>" # no need to pad
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _UNK]

PAD_ID = 0
UNK_ID = 1

DISCOURSE_MARKERS = [
    "after",
    "also",
    "although",
    "and",
    "as",
    "because",
    "before",
    "but",
    "for example",
    "however",
    "if",
    "meanwhile",
    "so",
    "still",
    "then",
    "though",
    "when",
    "while"
]
DISCOURSE_MARKER_SET_TAG = "ALL18"

# patterns = {
#     "because": ("IN", "mark", "advcl"),
# }

def setup_args():
    parser = argparse.ArgumentParser()
    code_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    glove_dir = os.path.join("data", "glove.6B")
    parser.add_argument("--dataset", default="wikitext-103", type=str)
    parser.add_argument("--train_size", default=0.9, type=float)
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--method", default="string_ssplit_int_init", type=str)
    parser.add_argument("--caching", action='store_true')
    parser.add_argument("--action", default='collect_raw', type=str)
    parser.add_argument("--glove_dim", default=300, type=int)
    parser.add_argument("--random_init", action='store_true')
    parser.add_argument("--max_seq_len", default=50, type=int)
    parser.add_argument("--min_seq_len", default=5, type=int)
    parser.add_argument("--max_ratio", default=5.0, type=float)
    parser.add_argument("--undersamp_cutoff", default=0, type=int)
    return parser.parse_args()

def initialize_vocabulary(vocabulary_path):
    # map vocab to word embeddings
    if os.path.isfile(vocabulary_path):
        rev_vocab = []
        with open(vocabulary_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)

def process_glove(args, vocab_dict, save_path, random_init=True):
    """
    :param vocab_list: [vocab]
    :return:
    """
    if os.path.isfile(save_path + ".npz"):
        print("Glove file already exists at %s" % (save_path + ".npz"))
    else:
        glove_path = os.path.join(args.glove_dir, "glove.840B.{}d.txt".format(args.glove_dim))
        if random_init:
            glove = np.random.randn(len(vocab_dict), args.glove_dim)
        else:
            glove = np.zeros((len(vocab_dict), args.glove_dim))

        found = 0

        with open(glove_path, 'r') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in vocab_dict:  # all cased
                    idx = vocab_dict[word]
                    glove[idx, :] = np.fromstring(vec, sep=' ')
                    found += 1

        # print("{}/{} of word vocab have corresponding vectors in {}".format(found, len(vocab), glove_path))
        np.savez_compressed(save_path, glove=glove)
        print("saved trimmed glove matrix at: {}".format(save_path))

def create_vocabulary(vocabulary_path, sentence_pairs_data, discourse_markers=None):
    if os.path.isfile(vocabulary_path):
        print("Vocabulary file already exists at %s" % vocabulary_path)
    else:
        print("Creating vocabulary {}".format(vocabulary_path))
        vocab = {}
        counter = 0

        for s1, s2, label in sentence_pairs_data:
            counter += 1
            if counter % 100000 == 0:
                print("processing line %d" % counter)
            for w in s1:
                if not w in _START_VOCAB:
                    if w in vocab:
                        vocab[w] += 1
                    else:
                        vocab[w] = 1
            for w in s2:
                if not w in _START_VOCAB:
                    if w in vocab:
                        vocab[w] += 1
                    else:
                        vocab[w] = 1

        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        print("Vocabulary size: %d" % len(vocab_list))
        with open(vocabulary_path, mode="wb") as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + b"\n")

def sentence_to_token_ids(sentence, vocabulary):
    return [vocabulary.get(w, UNK_ID) for w in sentence]

def merge_dict(dict_list1, dict_list2):
    for key, list_sent in dict_list1.iteritems():
        dict_list1[key].extend(dict_list2[key])
    return dict_list1

def data_to_token_ids(data, rev_class_labels, class_label_dict, target_path, vocabulary_path, data_dir):
    if os.path.isfile(target_path):
        print("file {} already exists".format(target_path))
    else:
        vocab, _ = initialize_vocabulary(vocabulary_path)

        ids_data = []
        text_data = []

        counter = 0
        for s1, s2, text_label in data:
            label = class_label_dict[text_label]
            counter += 1
            if counter % 1000000 == 0:
                print("converting %d" % (counter))
            token_ids_s1 = sentence_to_token_ids(s1, vocab)
            token_ids_s2 = sentence_to_token_ids(s2, vocab)
            ids_data.append((token_ids_s1, token_ids_s2, label))

        # shuffled_idx = range(len(ids_data))
        # np.random.shuffle(shuffled_idx)
        # shuffled_ids_data = [ids_data[idx] for idx in shuffled_idx]
        # shuffled_text_data = [text_data[idx] for idx in shuffled_idx]

        print("writing {}".format(target_path))
        pickle.dump(ids_data, open(target_path, mode="wb"))

def undo_rephrase(lst):
    return " ".join(lst).replace("for_example", "for example").split()

def rephrase(str):
    return str.replace("for example", "for_example")

def string_ssplit_int_init(sentence, previous_sentence, marker):

    if marker=="for example":
        words = rephrase(sentence).split()
        if "for_example"==words[0].lower():
            s1 = previous_sentence
            s2 = " ".join(undo_rephrase(words[1:]))
        else:
            idx = [w.lower() for w in words].index("for_example")
            s1 = " ".join(undo_rephrase(words[:idx]))
            s2 = " ".join(undo_rephrase(words[idx+1:]))
    else:
        words = sentence.split()
        if marker==words[0].lower(): # sentence-initial
            s1 = previous_sentence
            s2 = " ".join(words[1:])
        else: # sentence-internal
            idx = [w.lower() for w in words].index(marker)
            s1 = " ".join(words[:idx])
            s2 = " ".join(words[idx+1:])
    return (s1.strip(), s2.strip(), marker)

def string_ssplit_clean_markers():
    raise Exception("haven't included clean ssplit in this script yet")

def depparse_ssplit_v1():
    raise Exception("haven't included old combination depparse ssplit in this script yet")

def depparse_ssplit_v2(sentence, previous_sentence, marker):
    dangerous_dependencies = ["mark", "advcl", "acl"]

    dependency_patterns = {
      "after": {
        "POS": "IN",
        "S2": "mark", # S2 head (full S head) ---> connective
        "S1": ["advcl", "acl"]
      },
      "also": {
        "POS": "RB",
        "S2": "advmod",
        "S1": ["advcl"]
      },
      "although": {
        "POS": "IN",
        "S2": "mark",
        "S1": ["advcl"]
      },
      "before": {
        "POS": "IN",
        "S2": "mark",
        "S1": ["advcl"]
      },
      "so": {
        "POS": "IN",
        "S2": "mark",
        "S1": ["advcl"]
      },
      "still": {
        "POS": "RB",
        "S2": "advmod",
        "S1": ["parataxis", "dep"]
      },
      "though": {
        "POS": "IN",
        "S2": "mark",
        "S1": ["advcl"]
      },
      "because": {
        "POS": "IN",
        "S2": "mark",
        "S1": ["advcl"]
      },
      "however": {
        "POS": "RB",
        "S2": "advmod",
        "S1": ["dep", "parataxis"]
      },
      "if": {
        "POS": "IN",
        "S2": "mark",
        "S1": ["advcl"]
      },
      "while": {
        "POS": "IN",
        "S2": "mark",
        "S1": ["advcl"]
      }
    }

    # search for pattern:
    # "[discourse marker] S2, S1" (needs dependency parse)
    def search_for_reverse_pattern_pair(sent, marker, words, previous_sentence):
        parse_string = get_parse(sent, depparse=True)

        # book corpus maybe has carriage returns and new lines and other things?
        try: 
            parse = json.loads(parse_string.replace('\r\n', ''))
        except ValueError:
            parse = json.loads(re.sub("[^A-z0-9.,!:?\"'*&/\{\}\[\]()=+-]", "", parse_string))        
        sentence = Sentence(parse["sentences"][0], sent)
        return sentence.find_pair(marker, "s2 discourse_marker s1", previous_sentence)


    def is_verb_tag(tag):
        return tag[0] == "V" and not tag[-2:] in ["BG", "BN"]

    """
    POS-tag string as if it's a sentence
    and see if it has a verb that could plausibly be the predicate.
    """
    def has_verb(string):
        parse = get_parse(string, depparse=False)
        tokens = json.loads(parse)["sentences"][0]["tokens"]
        return any([is_verb_tag(t["pos"]) for t in tokens])


    """
    using the depparse, look for the desired pattern, in any order
    """
    def search_for_dep_pattern(marker, current_sentence, previous_sentence):  
        parse_string = get_parse(current_sentence, depparse=True)

        # book corpus maybe has carriage returns and new lines and other things?
        try: 
            parse = json.loads(parse_string.replace('\r\n', ''))
        except ValueError:
            parse = json.loads(re.sub("[^A-z0-9.,!?:\"'*&/\{\}\[\]()=+-]", "", parse_string))

        sentence = Sentence(parse["sentences"][0], current_sentence)
        return sentence.find_pair(marker, "any", previous_sentence)


    # https://stackoverflow.com/a/18669080
    def get_indices(lst, element, case="sensitive"):
      result = []
      starting_index = -1
      while True:
        try:
            found_index = lst.index(element, starting_index+1)
            starting_index = found_index
        except ValueError:
            return result
        result.append(found_index)

    def get_nearest(lst, element):
        distances = [abs(e-element) for e in lst]
        return lst[np.argmin(distances)]


    def separate_at_signs(lst):
        s = " ".join(lst)
        separated_s = re.sub(" @([^ ]+)@ ", " @ \1 @ ", s)
        return separated_s.split()


    """
    parsed tokenization is different from original tokenization.
    try to re-align and extract the correct words given the
    extraction_indices (which are 1-indexed into parsed_words)

    fix me to catch more cases?
    """
    def extract_subphrase(orig_words, parsed_words, extraction_indices):
        extraction_indices = [i-1 for i in extraction_indices]

        orig_words = separate_at_signs(orig_words)

        if len(orig_words) == len(parsed_words):
            return " ".join([orig_words[i] for i in extraction_indices])
        else:
            first_parse_index = extraction_indices[0]
            first_word_indices = get_indices(orig_words, parsed_words[first_parse_index])

            last_parse_index = extraction_indices[-1]

            last_word_indices = get_indices(orig_words, parsed_words[last_parse_index])

            if len(first_word_indices)>0 and len(last_word_indices)>0:
                first_orig_index = get_nearest(first_word_indices, first_parse_index)
                last_orig_index = get_nearest(last_word_indices, last_parse_index)
                if last_orig_index-first_orig_index == last_parse_index-first_parse_index:
                    # maybe it's just shifted
                    shift = first_orig_index - first_parse_index
                    extraction_indices = [i+shift for i in extraction_indices]
                    return " ".join([orig_words[i] for i in extraction_indices])
                else:
                    # or maybe there's funny stuff happening inside the subphrase
                    # in which case T-T
                    return None
            else:
                if len(first_word_indices)>0 and abs(last_parse_index-len(parsed_words))<3:
                    # the end of the sentence is always weird. assume it's aligned

                    # grab the start of the subphrase
                    first_orig_index = get_nearest(first_word_indices, first_parse_index)
                    # shift if necessary
                    shift = first_orig_index - first_parse_index
                    extraction_indices = [i+shift for i in extraction_indices]

                    if len(orig_words) > extraction_indices[-1]:
                        # extend to the end of the sentence if we're not already there
                        extraction_indices += range(extraction_indices[-1]+1, len(orig_words))
                    else:
                        extraction_indices = [i for i in extraction_indices if i<len(orig_words)]

                    return " ".join([orig_words[i] for i in extraction_indices])

                else:
                    # or maybe the first and/or last words have been transformed,
                    # in which case T-T
                    return None
            


    """
    use corenlp server (see https://github.com/erindb/corenlp-ec2-startup)
    to parse sentences: tokens, dependency parse
    """
    def get_parse(sentence, depparse=True):
        sentence = sentence.replace("'t ", " 't ")
        if depparse:
            url = "http://localhost:12345?properties={annotators:'tokenize,ssplit,pos,depparse'}"
        else:
            url = "http://localhost:12345?properties={annotators:'tokenize,ssplit,pos'}"
        data = sentence
        parse_string = requests.post(url, data=data).text
        return json.loads(parse_string)["sentences"][0]

    class Sentence():
        def __init__(self, json_sentence, original_sentence):
            self.json = json_sentence
            self.dependencies = json_sentence["basicDependencies"]
            self.tokens = json_sentence["tokens"]
            self.original_sentence = original_sentence
        def indices(self, word):
            if len(word.split(" ")) > 1:
                words = word.split(" ")
                indices = [i for lst in [self.indices(w) for w in words] for i in lst]
                return indices
            else:
                return [i+1 for i in get_indices([t["word"].lower() for t in self.tokens], word)]
        def token(self, index):
            return self.tokens[int(index)-1]
        def word(self, index):
            return self.token(index)["word"]

        def find_parents(self, index, filter_types=False, needs_verb=False):
            deps = self.find_deps(index, dir="parents", filter_types=filter_types)

            if needs_verb:
                deps = [d for d in deps if self.gov_is_verb(d) or self.dep_is_verb(d)]

            return [d["governor"] for d in deps]

        def find_children(self, index, filter_types=False, exclude_types=False, needs_verb=False):
            deps = self.find_deps(
                index,
                dir="children",
                filter_types=filter_types,
                exclude_types=exclude_types
            )

            if needs_verb:
                deps = [d for d in deps if self.dep_is_verb(d)]

            # print(deps)
            return [d["dependent"] for d in deps]

        def is_verb(self, index):
            pos = self.token(index)["pos"]
            if pos[0] == "V":
                return True
            else:
                cop_relations = self.find_deps(index, dir="children", filter_types="cop")
                has_cop_relation = len(cop_relations)>0
                if has_cop_relation:
                    return True
                else:
                    return False

        def gov_is_verb(self, d):
            index = d["governor"]
            return self.is_verb(index)

        def dep_is_verb(self, d):
            index = d["dependent"]
            return self.is_verb(index)

        def find_deps(self, index, dir=None, filter_types=False, exclude_types=False):
            deps = []
            if dir=="parents" or dir==None:
                deps += [d for d in self.dependencies if d['dependent']==index]
            if dir=="children" or dir==None:
                deps += [d for d in self.dependencies if d['governor']==index]

            if filter_types:
                deps = [d for d in deps if d["dep"] in filter_types]
            if exclude_types:
                deps = [d for d in deps if not d["dep"] in exclude_types]

            return deps

        def find_dep_types(self, index, dir=None, filter_types=False):
            deps = self.find_deps(index, dir=dir, filter_types=filter_types)
            return [d["dep"] for d in deps]

        def __str__(self):
            return " ".join([t["word"] for t in self.tokens])

        def get_subordinate_indices(self, acc, explore, depth=0, exclude_indices=[], exclude_types=[]):
            # print("acc: {}\nexplore: {}\ndepth: {}\nexclude_indices: {}".format(acc, explore, depth, exclude_indices))
            exclude_indices.sort()
            acc.sort()
            explore.sort()
            # print("exclude: " + " ".join([self.tokens[t_ind-1]["word"] for t_ind in exclude_indices]))
            # print("acc: " + " ".join([self.tokens[t_ind-1]["word"] for t_ind in acc]))
            # print("explore: " + " ".join([self.tokens[t_ind-1]["word"] for t_ind in explore]))
            # print("*****")
            children = [c for i in explore for c in self.find_children(i, exclude_types=exclude_types) if not c in exclude_indices]
            if len(children)==0:
                return acc
            else:
                return self.get_subordinate_indices(
                    acc=acc + children,
                    explore=children,
                    depth=depth+1,
                    exclude_indices=exclude_indices
                )

        def get_phrase_from_head(self, head_index, exclude_indices=[], exclude_types=[]):
            # given an index,
            # grab every index that's a child of it in the dependency graph
            subordinate_indices = self.get_subordinate_indices(
                acc=[head_index],
                explore=[head_index],
                exclude_indices=exclude_indices,
                exclude_types=exclude_types
            )
            if not subordinate_indices:
                return None
            subordinate_indices.sort()
            
            # make string of subordinate phrase from parse
            parse_subordinate_string = " ".join([self.word(i) for i in subordinate_indices])

            # correct subordinate phrase from parsed version to wikitext version
            # (tokenization systems are different)
            orig_words = self.original_sentence.split()
            parsed_words = [t["word"] for t in self.tokens]

            subordinate_phrase = extract_subphrase(orig_words, parsed_words, subordinate_indices)

            # make a string from this to return
            return subordinate_phrase

        def get_valid_marker_indices(self, marker):
            pos = dependency_patterns[marker]["POS"]
            return [i for i in self.indices(marker) if pos == self.token(i)["pos"] ]

        def get_candidate_S2_indices(self, marker, marker_index, needs_verb=False):
            connection_type = dependency_patterns[marker]["S2"]
            # Look for S2
            return self.find_parents(marker_index, filter_types=[connection_type], needs_verb=needs_verb)

        def get_candidate_S1_indices(self, marker, s2_head_index, needs_verb=False):
            valid_connection_types = dependency_patterns[marker]["S1"]
            return self.find_parents(
                s2_head_index,
                filter_types=valid_connection_types,
                needs_verb=needs_verb
            ) + self.find_children(
                s2_head_index,
                filter_types=valid_connection_types,
                needs_verb=needs_verb
            )

        def find_pair(self, marker, order, previous_sentence):
            assert(order in ["s2 discourse_marker s1", "any"])
            # fix me
            # (this won't quite work if there are multiple matching connections)
            # (which maybe never happens)
            S1 = None
            S2 = None
            s1_ind = 1000
            s2_ind = 0

            for marker_index in self.get_valid_marker_indices(marker):

                for s2_head_index in self.get_candidate_S2_indices(marker, marker_index, needs_verb=True):
                    s2_ind = s2_head_index

                    possible_S1s = []

                    for s1_head_index in self.get_candidate_S1_indices(marker, s2_head_index, needs_verb=True):
                        # store S1 if we have one
                        S1 = self.get_phrase_from_head(
                            s1_head_index,
                            exclude_indices=[s2_head_index]
                        )
                        # we'll lose some stuff here because of alignment between
                        # wikitext tokenization and corenlp tokenization.
                        # if we can't get a phrase, reject this pair
                        if not S1:
                            break

                        # if we are only checking for the "reverse" order, reject anything else
                        if order=="s2 discourse_marker s1":
                            if s1_ind < s2_ind:
                                break

                        possible_S1s.append((s1_head_index, S1))

                    # to do: fix this. it is wrong. we're just grabbing the first if there are multiple matches for the S1 pattern rather than choosing in a principled way
                    if len(possible_S1s) > 0:
                        s1_ind, S1 = possible_S1s[0]

                    # store S2 if we have one
                    S2 = self.get_phrase_from_head(
                        s2_head_index,
                        exclude_indices=[marker_index, s1_ind],
                        # exclude_types=dependency_patterns[marker]["S1"]
                    )
                    # we'll lose some stuff here because of alignment between
                    # wikitext tokenization and corenlp tokenization.
                    # if we can't get a phrase, reject this pair
                    # update: we fixed some of these with the @ correction
                    if not S2:
                        return None

            # if S2 is the whole sentence *and* we're missing S1, let S1 be the previous sentence
            if S2 and not S1:
                words_in_sentence = len(sentence.tokens)
                words_in_s2 = len(S2.split())
                if words_in_sentence - 1 == words_in_s2:
                    S1 = previous_sentence

            if S1 and S2:
                return S1, S2
            else:
                return None

    parse = get_parse(sentence)
    # print(json.dumps(parse, indent=4))
    sentence = Sentence(parse, sentence)
    return(sentence.find_pair(marker, "any", previous_sentence))

def collect_raw_sentences(source_dir, dataset, caching):
    markers_dir = pjoin(source_dir, "markers_" + DISCOURSE_MARKER_SET_TAG)
    output_dir = pjoin(markers_dir, "files")

    if not os.path.exists(markers_dir):
        os.makedirs(markers_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if dataset == "wikitext-103":
        filenames = [
            "wiki.train.tokens",
            "wiki.valid.tokens", 
            "wiki.test.tokens"
        ]
    else:
        raise Exception("not implemented")

    sentences = {marker: {"sentence": [], "previous": []} for marker in DISCOURSE_MARKERS}
    
    for filename in filenames:
        print("reading {}".format(filename))
        file_path = pjoin(source_dir, "orig", filename)
        with io.open(file_path, 'rU', encoding="utf-8") as f:
            # tokenize sentences
            sentences_cache_file = file_path + ".CACHE_SENTS"
            if caching and os.path.isfile(sentences_cache_file):
                sent_list = pickle.load(open(sentences_cache_file, "rb"))
            else:
                tokens = f.read().replace("\n", ". ")
                print("tokenizing")
                sent_list = nltk.sent_tokenize(tokens)
                if caching:
                    pickle.dump(sent_list, open(sentences_cache_file, "wb"))

        # check each sentence for discourse markers
        previous_sentence = ""
        for sentence in sent_list:
            words = rephrase(sentence).split()  # replace "for example"
            for marker in DISCOURSE_MARKERS:
                if marker == "for example":
                    proxy_marker = "for_example" 
                else:
                    proxy_marker = marker

                if proxy_marker in [w.lower() for w in words]:
                    sentences[marker]["sentence"].append(sentence)
                    sentences[marker]["previous"].append(previous_sentence)
            previous_sentence = sentence

    print('writing files')
    statistics_lines = []
    for marker in sentences:
        sentence_path = pjoin(output_dir, "{}_s.txt".format(marker))
        previous_path = pjoin(output_dir, "{}_prev.txt".format(marker))
        n_sentences = len(sentences[marker]["sentence"])
        statistics_lines.append("{}\t{}".format(marker, n_sentences))
        with open(sentence_path, "w") as sentence_file:
            for s in sentences[marker]["sentence"]:
                sentence_file.write(s + "\n")
        with open(previous_path, "w") as previous_file:
            for s in sentences[marker]["previous"]:
                previous_file.write(s + "\n")

    statistics_report = "\n".join(statistics_lines)
    open(pjoin(markers_dir, "VERSION.txt"), "w").write(
        "commit: \n\ncommand: \n\nmarkers:\n" + statistics_report
    )

def split_raw(source_dir, train_size):
    assert(train_size < 1 and train_size > 0)

    markers_dir = pjoin(source_dir, "markers_" + DISCOURSE_MARKER_SET_TAG)
    input_dir = pjoin(markers_dir, "files")

    split_dir = pjoin(markers_dir, "split_train{}".format(train_size))
    output_dir = pjoin(split_dir, "files")
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    statistics_lines = []
    for marker in DISCOURSE_MARKERS:
        sentences = open(pjoin(input_dir, "{}_s.txt".format(marker)), "rU").readlines()
        previous_sentences = open(pjoin(input_dir, "{}_prev.txt".format(marker)), "rU").readlines()
        assert(len(sentences)==len(previous_sentences))

        indices = range(len(sentences))
        np.random.shuffle(indices)

        test_proportion = (1-train_size)/2
        n_test = round(len(indices) * test_proportion)
        n_valid = n_test
        n_train = len(indices) - (n_test + n_valid)

        splits = {split: {"s": [], "prev": []} for split in ["train", "valid", "test"]}

        for i in range(len(indices)):
            sentence_index = indices[i]
            sentence = sentences[sentence_index]
            previous = previous_sentences[sentence_index]
            if i<n_test:
                split="test"
            elif i<(n_test + n_valid):
                split="valid"
            else:
                split="train"
            splits[split]["s"].append(sentence)
            splits[split]["prev"].append(previous)

        for split in splits:
            n_sentences = len(splits[split]["s"])
            statistics_lines.append("{}\t{}\t{}".format(split, marker, n_sentences))
            for sentence_type in ["s", "prev"]:
                write_path = pjoin(output_dir, "{}_{}_{}.txt".format(split, marker, sentence_type))
                with open(write_path, "w") as write_file:
                    for sentence in splits[split][sentence_type]:
                        write_file.write(sentence)

    statistics_report = "\n".join(statistics_lines)
    open(pjoin(split_dir, "VERSION.txt"), "w").write(
        "commit: \n\ncommand: \n\nstatistics:\n" + statistics_report
    )

def ssplit(method, source_dir, train_size):

    methods = {
        "string_ssplit_int_init": string_ssplit_int_init,
        "string_ssplit_clean_markers": string_ssplit_clean_markers,
        "depparse_ssplit_v1": depparse_ssplit_v1
    }
    assert(args.method in methods)

    markers_dir = pjoin(source_dir, "markers_" + DISCOURSE_MARKER_SET_TAG)
    split_dir = pjoin(markers_dir, "split_train{}".format(train_size))
    input_dir = pjoin(split_dir, "files")

    ssplit_dir = pjoin(split_dir, "ssplit_" + method)
    output_dir = pjoin(ssplit_dir, "files")

    if not os.path.exists(ssplit_dir):
        os.makedirs(ssplit_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    open(pjoin(ssplit_dir, "VERSION.txt"), "w").write("commit: \n\ncommand: \n\n")

    def get_data(split, marker, sentence_type):
        filename = "{}_{}_{}.txt".format(split, marker, sentence_type)
        file_path = pjoin(input_dir, filename)
        return open(file_path, "rU").readlines()

    # (a dictionary {train: {...}, valid: {...}, test: {...}})
    splits = {}
    for split in ["train", "valid", "test"]:
        print("extracting {}".format(split))
        data = {"s1": [], "s2": [], "label": []}
        for marker in DISCOURSE_MARKERS:
            sentences = get_data(split, marker, "s")
            previous = get_data(split, marker, "prev")
            assert(len(sentences) == len(previous))
            for i in range(len(sentences)):
                sentence = sentences[i]
                previous_sentence = previous[i]
                s1, s2, label = methods[method](sentence, previous_sentence, marker)
                data["label"].append(marker)
                data["s1"].append(s1)
                data["s2"].append(s2)
        splits[split] = data

    for split in splits:
        print("randomizing {}".format(split))
        # randomize the order at this point
        labels = splits[split]["label"]
        s1 = splits[split]["s1"]
        s2 = splits[split]["s2"]

        assert(len(labels) == len(s1) and len(s1) == len(s2))
        indices = range(len(labels))
        np.random.shuffle(indices)

        print("writing {}".format(split))
        for element_type in ["label", "s1", "s2"]:
            filename = "{}_{}_{}.txt".format(method, split, element_type)
            file_path = pjoin(output_dir, filename)
            with open(file_path, "w") as write_file:
                for index in indices:
                    element = splits[split][element_type][index]
                    write_file.write(element + "\n")

def filtering(source_dir, args):

    args.min_ratio = 1/args.max_ratio

    marker_dir = pjoin(source_dir, "markers_" + DISCOURSE_MARKER_SET_TAG)
    split_dir = pjoin(marker_dir, "split_train{}".format(args.train_size))
    ssplit_dir = pjoin(split_dir, "ssplit_" + args.method)
    input_dir = pjoin(ssplit_dir, "files")

    filter_dir = pjoin(ssplit_dir, "filter_max{}_min{}_ratio{}_undersamp{}".format(
        args.max_seq_len,
        args.min_seq_len,
        args.max_ratio,
        args.undersamp_cutoff
    ))
    output_dir = pjoin(filter_dir, "files")

    if not os.path.exists(filter_dir):
        os.makedirs(filter_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    def get_data(element_type, split):
        filename = "{}_{}_{}.txt".format(args.method, split, element_type)
        file_path = pjoin(input_dir, filename)
        return open(file_path, "rU").readlines()

    frequencies = {}
    for split in ["train", "valid", "test"]:
        frequencies[split] = {}
        for marker in DISCOURSE_MARKERS:
            frequencies[split][marker] = 0

    statistics_lines = []
    for split in ["train", "valid", "test"]:
        keep = {"s1": [], "s2": [], "label": []}

        # length-based filtering
        s1s = get_data("s1", split)
        s2s = get_data("s2", split)
        labels = get_data("label", split)
        assert(len(s1s) == len(s2s) and len(s2s) == len(labels))
        for i in range(len(s1s)):
            s1 = s1s[i][:-1]
            s2 = s2s[i][:-1]
            label = labels[i][:-1]
            len1 = len(s1.split())
            len2 = len(s2.split())
            ratio = float(len2)/len1
            if args.min_seq_len<len1 and len1<args.max_seq_len \
                    and args.min_seq_len<len2 and len2<args.max_seq_len \
                    and args.min_ratio<ratio and ratio<args.max_ratio:
                keep["s1"].append(s1)
                keep["s2"].append(s2)
                keep["label"].append(label)
                frequencies[split][label] += 1

        # write new filtered files
        for element_type in ["s1", "s2", "label"]:
            filename = "{}_{}_{}_{}_{}_{}.txt".format(
                split, 
                element_type,
                args.method, 
                args.max_seq_len, 
                args.min_seq_len, 
                args.max_ratio
            )
            file_path = pjoin(output_dir, filename)
            with open(file_path, "w") as write_file:
                for element in keep[element_type]:
                    write_file.write(element + "\n")

    statistics_lines = []
    for split in frequencies:
        for marker in frequencies[split]:
            freq = frequencies[split][marker]
            statistics_lines.append("{}\t{}\t{}".format(split, marker, freq))
    statistics_report = "\n".join(statistics_lines)
    open(pjoin(filter_dir, "VERSION.txt"), "w").write(
        "commit: \n\ncommand: \n\statistics:\n" + statistics_report
    )

def indexify(source_dir, args):

    marker_dir = pjoin(source_dir, "markers_" + DISCOURSE_MARKER_SET_TAG)
    split_dir = pjoin(marker_dir, "split_train{}".format(args.train_size))
    ssplit_dir = pjoin(split_dir, "ssplit_" + args.method)
    filter_dir = pjoin(ssplit_dir, "filter_max{}_min{}_ratio{}_undersamp{}".format(
        args.max_seq_len,
        args.min_seq_len,
        args.max_ratio,
        args.undersamp_cutoff
    ))
    input_dir = pjoin(filter_dir, "files")

    indexified_dir = pjoin(filter_dir, "indexified")
    if not os.path.exists(indexified_dir):
        os.makedirs(indexified_dir)
    output_dir = indexified_dir

    splits = {
        "train": [],
        "valid": [],
        "test": []
    }


    class_labels = {DISCOURSE_MARKERS[i]: i for i in range(len(DISCOURSE_MARKERS))}
    reverse_class_labels = [marker for marker in class_labels]
    
    for marker in class_labels:
        index = class_labels[marker]
        reverse_class_labels[index] = marker

    def get_filename(split, element_type):
        return "{}_{}_{}_{}_{}_{}.txt".format(
            split, 
            element_type,
            args.method, 
            args.max_seq_len, 
            args.min_seq_len, 
            args.max_ratio
        )

    for split in splits:
        s1_path = pjoin(input_dir, get_filename(split, "s1"))
        s2_path = pjoin(input_dir, get_filename(split, "s2"))
        labels_path = pjoin(input_dir, get_filename(split, "label"))
        with open(s1_path) as f1, open(s2_path) as f2, open(labels_path) as flab: 
            for s1, s2, label in izip(f1, f2, flab):
                s1 = s1.strip().split()
                s2 = s2.strip().split()
                label = label.strip()
                # if label in all_labels:
                splits[split].append((s1, s2, label))

    all_examples = splits["train"] + splits["valid"] + splits["test"]

    vocab_path = pjoin(output_dir, "vocab.dat")
    create_vocabulary(vocab_path, all_examples)
    vocab, rev_vocab = initialize_vocabulary(vocab_path)

    # ======== Trim Distributed Word Representation =======
    # If you use other word representations, you should change the code below

    process_glove(args, vocab, pjoin(output_dir, "glove.trimmed.{}.npz".format(args.glove_dim)),
                  random_init=args.random_init)

    pickle.dump(class_labels, open(pjoin(output_dir, "class_labels.pkl"), "w"))
    json.dump(reverse_class_labels, open(pjoin(output_dir, "reverse_class_labels.json"), "w"))

    for split in splits:
        data = splits[split]
        print("Converting data in {}".format(split))
        ids_path = pjoin(
            output_dir,
            "{}.ids.pkl".format(split)
        )
        data_to_token_ids(data, reverse_class_labels, class_labels, ids_path, vocab_path, output_dir)

def test():
    test_items = [
        {
            "sentence": "After release , it received downloadable content , along with an expanded edition in November of that year .",
            "previous_sentence": "It met with positive sales in Japan , and was praised by both Japanese and western critics .",
            "marker": "after",
            "output": None
        },
        {
            "sentence": "After the game 's completion , additional episodes are unlocked , some of them having a higher difficulty than those found in the rest of the game .",
            "previous_sentence": ".",
            "marker": "after",
            "output": None
        },
        {
            "sentence": "After the release of Valkyria Chronicles II , the staff took a look at both the popular response for the game and what they wanted to do next for the series .",
            "previous_sentence": "NA",
            "marker": "after",
            "output": None
        },
        {
            "sentence": "Kotaku 's Richard Eisenbeis was highly positive about the game , citing is story as a return to form after Valkyria Chronicles II and its gameplay being the best in the series .",
            "previous_sentence": "NA",
            "marker": "after",
            "output": None
        },
        {
            "sentence": "Valkyria of the Battlefield 3 : The Flower of the Nameless Oath ) , illustrated by Naoyuki Fujisawa and eventually released in two volumes after being serialized in Dengeki Maoh between 2011 and 2012 ; and Senjō no Valkyria 3 : <unk> Unmei no <unk> <unk> ( 戦場のヴァルキュリア3 <unk> , lit .",
            "previous_sentence": "They were Senjō no Valkyria 3 : Namo <unk> <unk> no Hana ( 戦場のヴァルキュリア3 <unk> , lit .",
            "marker": "after",
            "output": ('eventually released in two volumes', 'being serialized in Dengeki Maoh between 2011 and 2012')
        },
        {
            "sentence": "After taking up residence , her health began to deteriorate .",
            "previous_sentence": "She restored a maisonette in Storrington , Sussex , England , bequeathed by her friend Edith Major , and named it St. Andrew 's .",
            "marker": "after",
            "output": (', her health began to deteriorate .', 'taking up residence')
        },
        {
            "sentence": "While it retained the standard features of the series , it also underwent multiple adjustments , such as making the game more forgiving for series newcomers .",
            "previous_sentence": "The game began development in 2010 , carrying over a large portion of the work done on Valkyria Chronicles II .",
            "marker": "also",
            "output": ('While it retained the standard features of the series', ', it underwent multiple adjustments , such as making the game more forgiving for series newcomers .')
        },
        {
            "sentence": "It was also adapted into manga and an original video animation series .",
            "previous_sentence": "After release , it received downloadable content , along with an expanded edition in November of that year .",
            "marker": "also",
            "output": ('After release , it received downloadable content , along with an expanded edition in November of that year .', 'It was adapted into manga and an original video animation series .')
        },
        {
            "sentence": "There are also love simulation elements related to the game 's two main heroines , although they take a very minor role .",
            "previous_sentence": "After the game 's completion , additional episodes are unlocked , some of them having a higher difficulty than those found in the rest of the game .",
            "marker": "also",
            "output": ("After the game 's completion , additional episodes are unlocked , some of them having a higher difficulty than those found in the rest of the game .", "There are love simulation elements related to the game 's two main heroines , although they take a very minor role .")
        },
        {
            "sentence": "The music was composed by Hitoshi Sakimoto , who had also worked on the previous Valkyria Chronicles games .",
            "previous_sentence": ".",
            "marker": "also",
            "output": None
        },
        {
            "sentence": "Gallian Army Squad 422 , also known as \" The Nameless \" , are a penal military unit composed of criminals , foreign deserters , and military offenders whose real names are erased from the records and thereon officially referred to by numbers .",
            "previous_sentence": "The game takes place during the Second Europan War .",
            "marker": "also",
            "output": None
        },
        {
            "sentence": "In a preview of the TGS demo , Ryan Geddes of IGN was left excited as to where the game would go after completing the demo , along with enjoying the improved visuals over Valkyria Chronicles II .",
            "previous_sentence": ".",
            "marker": "after",
            "output": ('as to where the game would go', 'completing the demo')
        },
        {
            "sentence": "The units comprising the infantry force of Van Dorn 's Army of the West were the 1st and 2nd Arkansas Mounted Rifles were also armed with M1822 flintlocks from the Little Rock Arsenal .",
            "previous_sentence": "The 9th and 10th Arkansas , four companies of Kelly 's 9th Arkansas Battalion , and the 3rd Arkansas Cavalry Regiment were issued flintlock Hall 's Rifles .",
            "marker": "also",
            "output": None
        },
        {
            "sentence": "The Tower Building of the Little Rock Arsenal , also known as U.S. Arsenal Building , is a building located in MacArthur Park in downtown Little Rock , Arkansas .",
            "previous_sentence": ".",
            "marker": "also",
            "output": None
        },
        {
            "sentence": "It has also been the headquarters of the Little Rock Æsthetic Club since 1894 .",
            "previous_sentence": "It was home to the Arkansas Museum of Natural History and Antiquities from 1942 to 1997 and the MacArthur Museum of Arkansas Military History since 2001 .",
            "marker": "also",
            "output": ('It was home to the Arkansas Museum of Natural History and Antiquities from 1942 to 1997 and the MacArthur Museum of Arkansas Military History since 2001 .', 'It has been the headquarters of the Little Rock \xc3\x86sthetic Club since 1894 .')
        },
        {
            "sentence": "It was also the starting place of the Camden Expedition .",
            "previous_sentence": "Besides being the last remaining structure of the original Little Rock Arsenal and one of the oldest buildings in central Arkansas , it was also the birthplace of General Douglas MacArthur , who became the supreme commander of US forces in the South Pacific during World War II .",
            "marker": "also",
            "output": ('Besides being the last remaining structure of the original Little Rock Arsenal and one of the oldest buildings in central Arkansas , it was also the birthplace of General Douglas MacArthur , who became the supreme commander of US forces in the South Pacific during World War II .', 'It was the starting place of the Camden Expedition .')
        },
        {
            "sentence": "Fey 's projects after 2008 include a voice role in the English @-@ language version of the Japanese animated film Ponyo .",
            "previous_sentence": ".",
            "marker": "after",
            "output": None
        },
        {
            "sentence": "The main cave ( Cave 1 , or the Great Cave ) was a Hindu place of worship until Portuguese rule began in 1534 , after which the caves suffered severe damage .",
            "previous_sentence": ".",
            "marker": "after",
            "output": None
        },
        {
            "sentence": "This movement , although not authorized by me , has assumed such an aspect that it becomes my duty , as the executive of this <unk> , to interpose my official authority to prevent a collision between the people of the State and the Federal troops under your command .",
            "previous_sentence": "This movement is prompted by the feeling that pervades the citizens of this State that in the present emergency the arms and munitions of war in the Arsenal should be under the control of the State authorities , in order to their security .",
            "marker": "although",
            "output": None
        },
        {
            "sentence": "Dunnington was selected to head the ordnance works at Little Rock , and although he continued to draw his pay from the Confederate Navy Department , he was placed in charge of all Confederate ordnance activities ( which included artillery functions ) there with the rank of lieutenant colonel .",
            "previous_sentence": "Ponchartrain , which had been brought to Little Rock in hopes of converting it to an ironclad .",
            "marker": "although",
            "output": (', he was placed in charge of all Confederate ordnance activities ( which included artillery functions ) there with the rank of lieutenant colonel', 'he continued to draw his pay from the Confederate Navy Department')
        },
        {
            "sentence": "The development of a national team faces challenges similar to those across Africa , although the national football association has four staff members focusing on women 's football .",
            "previous_sentence": "The Gambia has two youth teams , an under @-@ 17 side that has competed in FIFA U @-@ 17 Women 's World Cup qualifiers , and an under @-@ 19 side that withdrew from regional qualifiers for an under @-@ 19 World Cup .",
            "marker": "although",
            "output": ('The development of a national team faces challenges similar to those across Africa , .', "the national football association has four staff members focusing on women 's football")
        },
        {
            "sentence": "Although this species is discarded when caught , it is more delicate @-@ bodied than other maskrays and is thus unlikely to survive encounters with trawling gear .",
            "previous_sentence": "In the present day , this is mostly caused by Australia 's Northern Prawn Fishery , which operates throughout its range .",
            "marker": "although",
            "output": (', it is more delicate @ \x01 @ bodied than other maskrays and is thus unlikely to survive encounters with trawling gear .', 'this species is discarded when caught')
        },
        {
            "sentence": "In the nineteenth @-@ century , the mound was higher on the western end of the tomb , although this was removed by excavation to reveal the sarsens beneath during the 1920s .",
            "previous_sentence": "The earthen mound that once covered the tomb is now visible only as an undulation approximately 1 foot , 6 inches in height .",
            "marker": "although",
            "output": ('In the nineteenth @ \x01 @ century , the mound was higher on the western end of the tomb , .', 'this was removed by excavation to reveal the sarsens beneath during the 1920s')
        },
        {
            "sentence": "In 1880 , the archaeologist Flinders Petrie included the existence of the stones at \" <unk> \" in his list of Kentish earthworks ; although noting that a previous commentator had described the stones as being in the shape of an oval , he instead described them as forming \" a rectilinear enclosure \" around the chamber .",
            "previous_sentence": "He believed that the monument consisted of both a \" chamber \" and an \" oval \" of stones , suggesting that they were \" two distinct erections \" .",
            "marker": "although",
            "output": (', he instead described them as forming " a rectilinear enclosure " around the chamber', 'noting that a previous commentator had described the stones as being in the shape of an oval')
        },
        # {
        #     "sentence": "",
        #     "previous_sentence": "",
        #     "marker": "although",
        #     "output": None
        # },
        # {
        #     "sentence": "",
        #     "previous_sentence": "",
        #     "marker": "although",
        #     "output": None
        # }
    ]
    curious_cases = [
        {
            "sentence": "But , after inspecting the work and observing the spirit of the men I decided that a garrison 500 strong could hold out against Fitch and that I would lead the remainder - about 1500 - to Gen 'l Rust as soon as shotguns and rifles could be obtained from Little Rock instead of pikes and lances , with which most of them were armed .",
            "previous_sentence": "",
            "marker": "after",
            "output": None,
            "explanation": "incorrect parse. it thinks 'the men I decided ...' forms a relative clause. different from what's up at http://nlp.stanford.edu:8080/corenlp/process"
        },
        {
            "sentence": "In 1864 , after Little Rock fell to the Union Army and the arsenal had been recaptured , General Fredrick Steele marched 8 @,@ 500 troops from the arsenal beginning the Camden Expedition .",
            "previous_sentence": "NA",
            "marker": "after",
            "output": None
        },
        {
            "sentence": "In addition to Sega staff from the previous games , development work was also handled by <unk> The original scenario was written Kazuki Yamanobe , while the script was written by Hiroyuki Fujii , Koichi Majima , <unk> Miyagi , Seiki <unk> and Takayuki <unk> .",
            "previous_sentence": "Speaking in an interview , it was stated that the development team considered Valkyria Chronicles III to be the series ' first true sequel : while Valkyria Chronicles II had required a large amount of trial and error during development due to the platform move , the third game gave them a chance to improve upon the best parts of Valkyria Chronicles II due to being on the same platform .",
            "marker": "also",
            "output": ('while the script was written by Hiroyuki Fujii , Koichi Majima , <unk> Miyagi , Seiki <unk> and Takayuki <unk>', 'In addition to Sega staff from the previous games , development work was handled by <unk> The original scenario was written Kazuki Yamanobe , .')
        },
        ## parse is just wrong :(
        {
            "sentence": "There are also love simulation elements related to the game 's two main heroines , although they take a very minor role .",
            "previous_sentence": "After the game 's completion , additional episodes are unlocked , some of them having a higher difficulty than those found in the rest of the game .",
            "marker": "although",
            "output": ("love simulation elements related to the game 's two main heroines ,", 'they take a very minor role')
        },
        {
            "sentence": "The remainder held professional pilot licences , either a Commercial Pilot Licence or an Airline Transport Pilot Licence , although not all of these would be engaged in GA activities .",
            "previous_sentence": "The number of pilots licensed by the CAA to fly powered aircraft in 2005 was 47 @,@ 000 , of whom 28 @,@ 000 held a Private Pilot Licence .",
            "marker": "although",
            "output": ('either a Commercial Pilot Licence or an Airline Transport Pilot Licence ,', 'not all of these would be engaged in GA activities')
        },
    ]
        
    print("{} cases are weird and I can't figure out how to handle them. :(".format(len(curious_cases)))
    curious=False
    if curious:
        print("running curious cases...")
        for item in curious_cases:
            print("====================")
            print(item["sentence"])
            output = depparse_ssplit_v2(item["sentence"], item["previous_sentence"], item["marker"])
            print(output)
        print("====================")
        print("====================")
        print("====================")


    n_tests = 23
    i = 0
    failures = 0
    print("running tests...")

    for item in test_items:
        if i < n_tests:
            output = depparse_ssplit_v2(item["sentence"], item["previous_sentence"], item["marker"])
            try:
                assert(output == item["output"])
            except AssertionError:
                print("====== TEST FAILED ======" + "\nsentence: " + item["sentence"] + "\nmarker: " + item["marker"] + "\nactual output: " + str(output) + "\ndesired output: " + str(item["output"]))
                failures += 1
        else:
            print("====================")
            print(item["sentence"])
            output = depparse_ssplit_v2(item["sentence"], item["previous_sentence"], item["marker"])
            print(output)
        i += 1

    if failures==0:
        print("All tests passed.")

if __name__ == '__main__':
    args = setup_args()

    source_dir = os.path.join("data", args.dataset)

    if args.action == "collect_raw":
        collect_raw_sentences(source_dir, args.dataset, args.caching)
    elif args.action == "split":
        split_raw(source_dir, args.train_size)
    elif args.action == "ssplit":
        ssplit(args.method, source_dir, args.train_size)
    elif args.action == "filtering":
        if args.undersamp_cutoff != 0:
            raise Exception("not implemented")
        filtering(source_dir, args)
    elif args.action == "indexify":
        indexify(source_dir, args)
    elif args.action == "test":
        test()


