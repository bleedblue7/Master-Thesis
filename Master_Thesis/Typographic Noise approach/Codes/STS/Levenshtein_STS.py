import numpy as np
from pyemd import emd
import os
import random
import gensim
from random import sample
import math
from gensim.models import KeyedVectors
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_excel("/home/arghya/Documents/Thesis/Datasets/STS_test.xlsx", engine='odf')

def levenshtein_ratio_and_distance(s, t, ratio_calc = False):
    """ levenshtein_ratio_and_distance:
        Calculates levenshtein distance between two strings.
        If ratio_calc = True, the function computes the
        levenshtein distance ratio of similarity between two strings
        For all i and j, distance[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t
    """
    # Initialize matrix of zeros
    rows = len(s)+1
    cols = len(t)+1
    distance = np.zeros((rows,cols),dtype = int)

    # Populate matrix of zeros with the indices of each character of both strings
    for i in range(1, rows):
        for k in range(1,cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0 # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                if ratio_calc == True:
                    cost = 2
                else:
                    cost = 1
            distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
                                 distance[row][col-1] + 1,          # Cost of insertions
                                 distance[row-1][col-1] + cost)     # Cost of substitutions
    if ratio_calc == True:
        # Calculation of the Levenshtein Distance Similarity
        levenshtein_similarity = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
        return levenshtein_similarity
    else:
        # print(distance) # Uncomment if you want to see the matrix showing how the algorithm computes the cost of deletions,
        # insertions and/or substitutions
        # This is the minimum number of edits needed to convert string a to string b
        return "The Levenshtein Distance is {}.".format(distance[row][col])

def replace_letters(string, ratio_changes=0.2):
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
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

results = []

for error_rate in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
    for _, row in data.iterrows():
        str1 = row['sentence1'].lower().replace(" '", "'")
        str2 = row['sentence2'].lower().replace(" '", "'")
        sent1 = replace_letters(str1, ratio_changes=error_rate)
        sent2 = replace_letters(str2, ratio_changes=error_rate)

        ground_truth = row['similarity'] / 5
        levenshtein_Similarity = levenshtein_ratio_and_distance(sent1, sent2, ratio_calc=True)
        sim_res.append(levenshtein_Similarity)
        sim_given.append(ground_truth)

        error = abs(levenshtein_Similarity - ground_truth)

        results.append({"sentence1": sent1, "sentence2": sent2, "method": "Levenshtein",
                        "ground_truth": ground_truth, "score": levenshtein_Similarity,
                        "noise_rate": error_rate, "absolute_error": error})

        print("error rate:", error_rate)
        print("sentence_1-->{:>30} || sentence_2-->{:>30}".format(sent1, sent2))
        print('Measured Levenshtein Similarity for Noisy Texts:{:>20}'.format(levenshtein_Similarity))
        print('Original Similarity:{:>10}'.format(ground_truth))
rmse = sqrt(mean_squared_error(sim_given, sim_res))
mae = mean_absolute_error(sim_given, sim_res)

print('Measured Levenshtein Similarity for Noisy Texts:', sim_res)
print('Original Similarity:', sim_given)
print('Root Mean Squared Error:{:>20}'.format(rmse))
print('Mean Absolute Error is:{:>20}'.format(mae))

pd_results = pd.DataFrame(results)
print(pd_results)

# pd_results.to_csv("/home/arghya/Documents/Thesis/NewResults/STS_Levenshtein.csv")

ax = sns.boxplot(x="noise_rate", y="absolute_error", hue="method", data=pd_results)
ax.set(ylim=(-0.05, 1.05))
ax.legend(bbox_to_anchor=(0.99, 1), loc='upper right', borderaxespad=0.2)
plt.savefig("/home/arghya/Documents/Thesis/Plots/STS_Levenshtein.pdf")