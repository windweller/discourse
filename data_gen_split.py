#! /usr/bin/env python

import numpy as np

import sys
reload(sys)
sys.setdefaultencoding('utf8')

np.random.seed(123)

def setup_args():
    parser = argparse.ArgumentParser()
    code_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    parser.add_argument("--dataset", default="wikitext-103", type=str)
    parser.add_argument("--data_tag", default="", type=str)
    parser.add_argument("--train_size", default=0.9, type=float)
    parser.add_argument("--method", default="string_ssplit_int_init", type=str)

def string_ssplit_int_init():
    pass

def string_ssplit_clean_markers():
    pass

def depparse_ssplit_v1():
    pass

def split_dictionary(data_dict, train_size):
    split_proportions = {
        "train": args.train_size,
        "valid": (1-args.train_size)/2,
        "test": (1-args.train_size)/2
    }
    assert(sum([split_proportions[split] for split in split_proportions])==1)
    splits = {split: {} for split in split_proportions}

    n_total = 0
    for marker in data_dict:
        examples_for_this_marker = data_dict[marker]
        n_marker = len(examples_for_this_marker)
        n_total += n_marker

        print("number of examples for {}: {}".format(marker, n_marker))

        # make valid and test sets (they will be equal size)
        valid_size = int(np.floor(split_proportions["valid"]*n_marker))
        test_size = valid_size
        splits["valid"][marker] = examples_for_this_marker[0:valid_size]
        splits["test"][marker] = examples_for_this_marker[valid_size:valid_size+test_size]
        # make train set with remaining examples
        splits["train"][marker] = examples_for_this_marker[valid_size+test_size:]

    print("total number of examples: {}".format(n_total))

    return splits


if __name__ == '__main__':
    args = setup_args()

    source_dir = os.path.join(args.prefix, "data", args.dataset)

    # collect sentence pairs
    methods = {
        "string_ssplit_int_init": string_ssplit_int_init,
        "string_ssplit_clean_markers": string_ssplit_clean_markers,
        "depparse_ssplit_v1": depparse_ssplit_v1,
        "split_dictionary": split_dictionary
    }
    data_dict = methods[args.method]

    assert(args.train_size < 1 and args.train_size > 0)
    assert(args.method in methods)

    # split train, valid, test
    # (returns a dictionary {train: {...}, valid: {...}, test: {...}})
    splits = split_dictionary(data_dict, args.train_size)

    if args.data_tag == "":
        extra_tag = ""
    else:
        extra_tag = "_" + args.data_tag

    for split in splits:
        data = splits[split]
        filename = "{}_all_pairs_{}_train{}{}".format(
            split,
            args.method,
            args.train_size,
            extra_tag
        )
        pickle.dump(data, open(filename, "wb"))


