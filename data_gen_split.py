#! /usr/bin/env python

import numpy as np
import argparse
import io
import nltk
import pickle

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import os
from os.path import join as pjoin

np.random.seed(123)

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
    parser.add_argument("--dataset", default="wikitext-103", type=str)
    parser.add_argument("--train_size", default=0.9, type=float)
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
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="rb") as f:
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
    if gfile.Exists(save_path + ".npz"):
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

        print("{}/{} of word vocab have corresponding vectors in {}".format(found, len(vocab), glove_path))
        np.savez_compressed(save_path, glove=glove)
        print("saved trimmed glove matrix at: {}".format(save_path))


def create_vocabulary(vocabulary_path, sentence_pairs_data, discourse_markers=None):
    if gfile.Exists(vocabulary_path):
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
        with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + b"\n")


def sentence_to_token_ids(sentence, vocabulary):
    return [vocabulary.get(w, UNK_ID) for w in sentence]


def merge_dict(dict_list1, dict_list2):
    for key, list_sent in dict_list1.iteritems():
        dict_list1[key].extend(dict_list2[key])
    return dict_list1

def data_to_token_ids(data, all_labels, class_label_dict, target_path, vocabulary_path, data_dir):
    rev_class_labels = all_labels
    if gfile.Exists(target_path):
        print("file {} already exists".format(target_path))
    else:
        vocab, _ = initialize_vocabulary(vocabulary_path)

        ids_data = []
        text_data = []

        counter = 0
        for s1, s2, text_label in data:
            label = class_label_dict[text_label]
            counter += 1
            if counter % 100000 == 0:
                print("converting %d" % (counter))
            token_ids_s1 = sentence_to_token_ids(s1, vocab)
            token_ids_s2 = sentence_to_token_ids(s2, vocab)
            ids_data.append((token_ids_s1, token_ids_s2, label))
            text_data.append((s1, s2, label))

        shuffled_idx = range(len(ids_data))
        random.shuffle(shuffled_idx)
        shuffled_ids_data = [ids_data[idx] for idx in shuffled_idx]
        shuffled_text_data = [text_data[idx] for idx in shuffled_idx]

        print("writing {} and {}".format(target_path, text_path))
        pickle.dump(shuffled_ids_data, gfile.GFile(target_path, mode="wb"))

        with gfile.GFile(text_path, mode="wb") as f:
            for t in shuffled_text_data:
                f.write(str([" ".join(t[0]), " ".join(t[1]), rev_class_labels[t[2]]]) + "\n")


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

def depparse_ssplit_v2():
    raise Exception("haven't included new depparse ssplit in this script yet")

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

    def get_data(split, marker, sentence_type):
        filename = "{}_{}_{}.txt".format(split, marker, sentence_type)
        file_path = pjoin(input_dir, filename)
        return open(file_path, "rU").readlines()

    # (a dictionary {train: {...}, valid: {...}, test: {...}})
    splits = {}
    for split in ["train", "valid", "test"]:
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

    for split in splits:
        # randomize the order at this point
        labels = splits[split]["label"]
        s1 = splits[split]["s1"]
        s2 = splits[split]["s2"]

        assert(len(labels) == len(s1) and len(s1) == len(s2))
        indices = range(len(labels))
        np.random.shuffle(indices)

        for element_type in ["label", "s1", "s2"]:
            filename = "{}_{}_{}.txt".format(method, split, element_type)
            file_path = pjoin(output_dir, filename)
            with open(file_path, "w") as write_file:
                for index in indices:
                    element = splits[split][element_type][index]
                    write_file.write(element + "\n")

    open(pjoin(ssplit_dir, "VERSION.txt"), "w").write("commit: \n\ncommand: \n\n")

def filtering(source_dir, train_size, method, max_seq_len, min_seq_len, max_ratio, undersamp_cutoff):

    min_ratio = 1/max_ratio

    marker_dir = pjoin(source_dir, "markers_" + DISCOURSE_MARKER_SET_TAG)
    split_dir = pjoin(marker_dir, "split_train" + train_size)
    ssplit_dir = pjoin(split_dir, "ssplit_" + method)
    filter_dir = pjoin(ssplit_dir, "filter_max{}_min{}_ratio{}_undersamp{}".format(
        max_seq_len,
        min_seq_len,
        max_ratio,
        undersamp_cutoff
    ))
    
    for split in ["train", "valid", "test"]:
        for element_type in ["s", "prev"]:
            filename = "{}_{}_{}.txt".format(method, split, element_type)
            file_path = pjoin(ssplit_dir, filename)

    # length-based filtering

    # write new filtered files

def indexify(method, source_dir, train_size, glove_dim, random_init):

    marker_dir = pjoin(source_dir, "markers_" + DISCOURSE_MARKER_SET_TAG)
    split_dir = pjoin(marker_dir, "split_train" + train_size)
    ssplit_dir = pjoin(split_dir, "ssplit_" + method)
    filter_dir = pjoin(ssplit_dir, "filter_max{}_min{}_ratio{}_undersamp{}".format(
        max_seq_len,
        min_seq_len,
        max_ratio,
        undersamp_cutoff
    ))
    indexified_dir = pjoin(filter_dir, "indexified")

    # marker_dir = pjoin(source_dir, "markers_" + DISCOURSE_MARKER_SET_TAG)
    # split_dir = pjoin(marker_dir, "split_train" + train_size)
    # ssplit_dir = pjoin(split_dir, "ssplit_" + method)

    # "{}_{}_{}.txt".format(method, split, element_type)

    sub_directory = "{}_train{}{}".format(
        args.method,
        train_size,
        extra_tag
    )

    vocab_path = pjoin(sub_directory, "vocab.dat")

    splits = {
        "train": [],
        "valid": [],
        "test": []
    }

    def get_file_path(split, element_type):
        filename = sub_directory + "_{}_{}.txt".format(split, element_type)
        return pjoin(source_dir, sub_directory, filename)

    for split in splits:
        s1_path = get_file_path(split, "s1")
        s2_path = get_file_path(split, "s2")
        labels_path = get_file_path(split, "label")
        with open(s1_path) as f1, open(s2_path) as f2, open(labels_path) as flab: 
            for s1, s2, label in izip(f1, f2, flab):
                s1 = s1.strip().split()
                s2 = s2.strip().split()
                label = label.strip()
                # if label in all_labels:
                splits[split].append((s1, s2, label))

    all_examples = splits["train"] + splits["valid"] + splits["test"]

    create_vocabulary(vocab_path, all_examples)

    output_dir = pjoin(source_dir, sub_directory)

    vocab, rev_vocab = initialize_vocabulary(pjoin(output_dir, "vocab.dat"))

    # ======== Trim Distributed Word Representation =======
    # If you use other word representations, you should change the code below

    process_glove(args, vocab, pjoin(output_dir, "glove.trimmed.{}.npz".format(args.glove_dim)),
                  random_init=args.random_init)


    class_labels = {all_labels[i]: i for i in range(len(all_labels))}

    pickle.dump(class_labels, open(pjoin(output_dir, "class_labels.pkl"), "wb"))
    pickle.dump(reverse_class_labels, open(pjoin(output_dir, "reverse_class_labels.pkl"), "wb"))

    for split in splits:
        data = splits[split]
        print("Converting data in {}".format(split))
        ids_path = pjoin(
            args.run_dir,
            "{}.ids.pkl".format(split)
        )
        data_to_token_ids(data, all_labels, class_labels, ids_path, vocab_path, args.run_dir)

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
        filtering()
    elif args.action == "indexify":
        indexify(args.method, source_dir, args.train_size, args.glove_dim, args.random_init)


