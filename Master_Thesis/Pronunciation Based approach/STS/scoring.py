import torch
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings, DocumentPoolEmbeddings, WordEmbeddings
from nltk.corpus import stopwords
from nltk import download
import gensim.downloader as api
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class ScoreEval:
    def __init__(self, gensim=True, vocab_glove=True):
        if gensim:
            if not vocab_glove:
                self.model = api.load('word2vec-google-news-300')
            else:
                self.model = api.load('glove-wiki-gigaword-100')
            self.model.init_sims(replace=True)
            self.model.wv.vocab = {k.lower(): v for k, v in self.model.wv.vocab.items()}
            download('stopwords')  # Download stopwords list.
            self.stop_words = stopwords.words('english')

        self.embeddings = DocumentPoolEmbeddings(
            [WordEmbeddings('en-crawl'),
             WordEmbeddings('en'),
             FlairEmbeddings('en-forward'),
             FlairEmbeddings('en-backward')])

    def preprocess_wmd(self, sentence):
        return [w for w in sentence.lower().split() if w not in self.stop_words]

    def wmd(self, sentence1, sentence2, normalisation=True):
        #         sentence1 = " ".join(self.preprocess_wmd(sentence1))
        #         sentence2 = " ".join(self.preprocess_wmd(sentence2))
        sentence1 = [word for word in sentence1.lower().split() if self.model.wv.vocab.get(word) is not None]

        sentence2 = [word for word in sentence2.lower().split() if self.model.wv.vocab.get(word) is not None]
        wmd_distance = self.model.wmdistance(sentence1, sentence2)
        wmd_normalised = min(max((1.0 - 0.4 * wmd_distance), 0.0), 1.0)
        if not normalisation:
            return wmd_distance
        else:
            return wmd_normalised

    def flair_similarity(self, sentence1, sentence2):
        s1 = Sentence(sentence1)
        s2 = Sentence(sentence2)
        self.embeddings.embed([s1, s2])
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        cosine_similarity = cos(s1.embedding, s2.embedding).item()
        return cosine_similarity

    def levenshtein_ratio_and_distance(self, s, t, ratio_calc=True):
        """ levenshtein_ratio_and_distance:
        Calculates levenshtein distance between two strings.
        If ratio_calc = True, the function computes the
        levenshtein distance ratio of similarity between two strings
        For all i and j, distance[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t
        """
        # Initialize matrix of zeros
        rows = len(s) + 1
        cols = len(t) + 1
        distance = np.zeros((rows, cols), dtype=int)

        # Populate matrix of zeros with the indices of each character of both strings
        for i in range(1, rows):
            for k in range(1, cols):
                distance[i][0] = i
                distance[0][k] = k

        # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions
        for col in range(1, cols):
            for row in range(1, rows):
                if s[row - 1] == t[col - 1]:
                    cost = 0  # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
                else:
                    # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                    # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                    if ratio_calc == True:
                        cost = 2
                    else:
                        cost = 1
                distance[row][col] = min(distance[row - 1][col] + 1,  # Cost of deletions
                                         distance[row][col - 1] + 1,  # Cost of insertions
                                         distance[row - 1][col - 1] + cost)  # Cost of substitutions
        if ratio_calc == True:
            # Calculation of the Levenshtein Distance Similarity
            levenshtein_similarity = ((len(s) + len(t)) - distance[row][col]) / (len(s) + len(t))
            return levenshtein_similarity
        else:
            # print(distance) # Uncomment if you want to see the matrix showing how the algorithm computes the cost of deletions,
            # insertions and/or substitutions
            # This is the minimum number of edits needed to convert string a to string b
            return "The Levenshtein Distance is {}.".format(distance[row][col])


if __name__ == '__main__':
    obj=ScoreEval()