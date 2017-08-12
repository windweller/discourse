import pickle

def get_all_data():

	wiki = pickle.load(pairs, open("data/wikitext-103/all_sentence_pairs.pkl", "rb"))
	books = pickle.load(pairs, open("data/books/all_sentence_pairs.pkl", "rb"))

	all_data = {}

	for marker in books:
		all_data[marker] = wiki[marker] + books[marker]

	return all_data

all_data = get_all_data()

pickle.dump(pairs, open("data/books_and_wiki_sentence_pairs.pkl", "wb"))
