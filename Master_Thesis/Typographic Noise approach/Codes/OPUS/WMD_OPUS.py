import pandas as pd
import os
import random
import warnings
from pyemd import emd
import math
from math import sqrt
import numpy as np
import string
import gensim
from gensim.models import KeyedVectors
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from random import sample
import seaborn as sns
import matplotlib.pyplot as plt
import gensim.downloader as api




from nltk.corpus import stopwords
from nltk import download
data = pd.read_csv("/home/arghya/Documents/Thesis/Datasets/OPUS/opusparcus_en/opusparcus_v1/en/dev/en-dev.txt", sep='\t', header=None)
data.columns = ["serial", "sentence1", "sentence2", "similarity"]

download('stopwords')
stop_words = stopwords.words('english')


if not os.path.exists('/home/arghya/Documents/Thesis/GoogleNews-vectors-negative300.bin.gz'):
    raise ValueError("SKIP: You need to download the google news model")
warnings.filterwarnings('ignore')

def replace_letters(string, ratio_changes=0.2):
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
               'v', 'w', 'x', 'y', 'z']
    letter_map = {}
    letter_map["a"] = ["s", "z", "q"]
    letter_map["b"] = ["v", "g", "n", "h"]
    letter_map["c"] = ["x", "d", "f", "v"]
    letter_map["d"] = ["s", "f", "e", "x", "c"]
    letter_map["e"] = ["r", "w", "s", "d"]
    letter_map["f"] = ["d", "g", "r", "c", "v"]
    letter_map["g"] = ["f", "t", "h", "v", "b"]
    letter_map["h"] = ["g", "y", "j", "b", "n"]
    letter_map["i"] = ["u", "o", "j", "k"]
    letter_map["j"] = ["u", "h", "k", "n", "m"]
    letter_map["k"] = ["i", "j", "l", "m"]
    letter_map["l"] = ["k", "o"]
    letter_map["m"] = ["n", "j", "k"]
    letter_map["n"] = ["b", "m", "h", "j"]
    letter_map["o"] = ["i", "p", "l"]
    letter_map["p"] = ["o", "l"]
    letter_map["q"] = ["a", "w"]
    letter_map["r"] = ["e", "t", "d", "f"]
    letter_map["s"] = ["a", "w", "d", "z", "x"]
    letter_map["t"] = ["r", "y", "f", "g"]
    letter_map["u"] = ["y", "i", "h", "j"]
    letter_map["v"] = ["c", "f", "g", "b"]
    letter_map["w"] = ["q", "e", "a", "s"]
    letter_map["x"] = ["s", "z", "d", "c"]
    letter_map["y"] = ["t", "u", "g", "h"]
    letter_map["z"] = ["a", "s", "x"]

    letter_index = []
    for i in range(len(string)):
        if (string[i].isalpha()):
            letter_index.append(i)

    indices = sample(letter_index, int(len(letter_index) * ratio_changes))
    temp_letter = []
    for i in range(len(string)):
        if i in indices:
            if string[i] in letters:
                val = random.randint(0, len(letter_map[string[i]]) - 1)
                temp_letter.append(letter_map[string[i]][val])
            else:
                temp_letter.append(string[i])

        else:
            temp_letter.append(string[i])

    return "".join(temp_letter)


sim_res = []
sim_given = []

word_vectors = KeyedVectors.load_word2vec_format('/home/arghya/Documents/Thesis/GoogleNews-vectors-negative300.bin',
                                                 binary=True)
word_vectors.init_sims(replace=True)
word_vectors.wv.vocab = {k.lower(): v for k, v in word_vectors.wv.vocab.items()}

results = []

for error_rate in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
    for _, row in data.iterrows():
        str1 = row['sentence1'].lower().replace(" '", "'")
        str2 = row['sentence2'].lower().replace(" '", "'")
        sent1 = replace_letters(str1, ratio_changes=error_rate)
        sent2 = replace_letters(str2, ratio_changes=error_rate)

        ground_truth = row['similarity'] / 4
        wmd_score = word_vectors.wmdistance(sent1, sent2)
        final_score = min(max((0.6649 - 0.3479 * wmd_score), 0.0), 1.0)

        error = abs(final_score - ground_truth)
        results.append({"sentence1": sent1, "sentence2": sent2, "method": "WMD",
                        "ground_truth": ground_truth, "score": final_score, "noise_rate": error_rate,
                        "absolute_error": error})

        sim_given.append(ground_truth)
        sim_res.append(final_score)

        print("error rate:", error_rate)
        print("sentence_1-->{:>30} || sentence_2-->{:>30}".format(sent1, sent2))
        print('Measured WMD Similarity for Noisy Texts:{:>20}'.format(final_score))
        print('Original Similarity:{:>10}'.format(ground_truth))

rmse = sqrt(mean_squared_error(sim_given, sim_res))
mae = mean_absolute_error(sim_given, sim_res)

print('Measured WMD Similarity for Noisy Texts:',sim_res)
print('Original Similarity:', sim_given)
print('Root Mean Squared Error:{:>20}'.format(rmse))
print('Mean Absolute Error is:{:>20}'.format(mae))

pd_results = pd.DataFrame(results)
print(pd_results)

pd_results.to_csv("/home/arghya/Documents/Thesis/NewResults/WMD_OPUS.csv")

ax = sns.boxplot(x="noise_rate", y="absolute_error", hue="method", data=pd_results)
ax.set(ylim=(-0.05, 1.05))
ax.legend(bbox_to_anchor=(0.99, 1), loc='upper right', borderaxespad=0.2)
plt.savefig("/home/arghya/Documents/Thesis/Plots/WMD_OPUS.pdf")