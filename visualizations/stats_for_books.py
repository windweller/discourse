import pickle



versions = [
	("trunc5", "../data/books/truncated/", "_but_because_when_if_so", "truncated", "new5"),
	("trunc8", "../data/books/truncated/", "_but_because_when_if_for_example_so_before_still", "truncated", "simple8"),
	("new5", "../data/books/non-truncated", "_but_because_if_when_so", "non-truncated", "new5"),
	("simple8", "../data/books/non-truncated", "_but_because_when_if_for_example_so_before_still", "non-truncated", "simple8"),
]

def get_stats(version):
	print version

	splits = {split: pickle.load(open(version[1] + split + version[2] + ".ids.pkl")) for split in ["train", "valid", "test"]}

	labels = pickle.load(open("class_labels" + version[2], + ".pkl" "rb"))
	rev_labels = [None for marker in labels]
	for marker in labels:
		rev_labels[labels[marker]] = marker

	truncation = version[3]
	subset = version[4]

	stats = []
	for split in splits:
		print split
		lengths = {}
		for s1, s2, label in splits[split]:
			length_label = "{} {} {}".format(len(s1), len(s2), rev_labels[label], truncation, subset)
			if length_label in lengths:
				lengths[length_label] += 1
			else:
				lengths[length_label] = 1
		for key in lengths:
			stats.append("{} {}".format(key, lengths[key]))

	open(version[0] + ".csv", "w").write("\n".join(stats))



for version in versions:
	get_stats(version)


