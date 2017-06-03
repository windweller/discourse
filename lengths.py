# run `python lengths.py > data/lengths.csv`

import re

csv_lines = []
for version in ["test", "train", "valid"]:
	for source in ["ptb", "wikitext-103"]:
		for connective in ["because", "but"]:
			filename = "data/" + source + "/" + version + "_" + connective.upper() + ".txt"

			for line in open(filename):
				m = re.match("(.*) " + connective + " (.*)\n", line)
				if m:
					s1, s2 = m.groups()
					csv_line = ",".join([
						str(len(s1.split(" "))),
						str(len(s2.split(" "))),
						connective,
						version,
						source
					])
					csv_lines.append(csv_line)

for line in csv_lines:
	print line