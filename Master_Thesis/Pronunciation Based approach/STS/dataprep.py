import pandas as pd
import pickle
import numpy as np
import random
import string
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from random import sample
import torch
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings, DocumentPoolEmbeddings, WordEmbeddings


# dataset structure

class Data:
    def __init__(self, dset_name):
        self.dataset_name = dset_name
        self.letter_consonant_noise_ratio = None
        self.letter_vowel_noise_ratio = None
        self.letter_random_noise_ratio = None
        self.pair_noise_ratio = None
        self.original_dset = {}
        self.sentences_noise_consonant = {}
        self.sentences_noise_vowel = {}
        self.sentences_noise_random = {}
        self.sentences_noise_swap = {}

    def _assign_dname(self, name):
        self.dataset_name = name
        return True

    def _assign_original(self, dset):
        self.original_dset = dset
        return True

    def _assign_noise_consonant(self, dset, ratio):
        self.letter_consonant_noise_ratio = ratio
        self.sentences_noise_consonant = dset
        return True

    def _assign_noise_vowel(self, dset, ratio):
        self.letter_vowel_noise_ratio = ratio
        self.sentences_noise_vowel = dset
        return True

    def _assign_noise_random(self, dset, ratio):
        self.letter_random_noise_ratio = ratio
        self.sentences_noise_random = dset
        return True

    def _assign_noise_swap(self, dset, ratio):
        self.letter_swap_noise_ratio = ratio
        self.sentences_noise_swap = dset
        return True

    def _get_dname(self):
        return self.dataset_name

    def _get_original(self):
        return self.original_dset

    def _get_noise_consonant(self):
        return (self.letter_consonant_noise_ratio, self.sentences_noise_consonant)

    def _get_noise_vowel(self):
        return (self.letter_vowel_noise_ratio, self.sentences_noise_vowel)

    def _get_noise_random(self):
        return (self.letter_random_noise_ratio, self.sentences_noise_random)

    def _get_noise_swap(self):
        return (self.letter_swap_noise_ratio, self.sentences_noise_swap)


# data preparation class
class DataPrepare:
    def __init__(self, excel_path):
        self.data = pd.read_excel(excel_path, engine='odf')
        self.original_dset = {}
        self.original_dset['sentence1'] = []
        self.original_dset['sentence2'] = []
        self.original_dset['ground_truth_similarity'] = []

        for _, row in self.data.iterrows():
            str1 = row['sentence1'].lower().replace(" '", "'").split(" ")
            str2 = row['sentence2'].lower().replace(" '", "'").split(" ")
            self.original_dset['ground_truth_similarity'].append(row['similarity'])
            self.original_dset['sentence1'].append(" ".join(str1))
            self.original_dset['sentence2'].append(" ".join(str2))

    def replace_letter(self, noise_type, string, ratio_changes=0.2):
        consonants = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'o', 'p', 'q', 's', 't', 'v', 'w', 'z']
        vowels = ['a', 'e', 'i', 'o', 'u']
        vowel_map = {}
        consonant_map = {}
        consonant_map["b"] = ["v"]
        consonant_map["c"] = ["s", "sh"]
        consonant_map["d"] = ["t"]
        consonant_map["f"] = ["ph"]
        consonant_map["g"] = ["j", "z"]
        consonant_map["h"] = ["o"]
        consonant_map["j"] = ["g", "z"]
        consonant_map["k"] = ["ch"]
        consonant_map["q"] = ["ko"]
        consonant_map["s"] = ["c", "sh"]
        consonant_map["t"] = ["d"]
        consonant_map["v"] = ["b", "w"]
        consonant_map["w"] = ["v"]
        consonant_map["z"] = ["g", "j"]
        consonant_map["o"] = ["o"]
        consonant_map["p"] = ["p"]
        vowel_map["a"] = ["e", "u"]
        vowel_map["e"] = ["a", "i"]
        vowel_map["i"] = ["e", "y"]
        vowel_map["o"] = ["ou"]
        vowel_map["u"] = ["a", "e"]

        letter_index = []
        for i in range(len(string)):
            if (string[i].isalpha()):
                letter_index.append(i)
        indices = sample(letter_index,int(len(letter_index)*ratio_changes)+1)

        temp_letter = []

        for i in range(len(string)):
            if "c" in noise_type:
                if i in indices:
                    if string[i] in consonants:
                        val = random.randint(0, len(consonant_map[string[i]]) - 1)
                        temp_letter.append(consonant_map[string[i]][val])
                    else:
                        temp_letter.append(string[i])
                else:
                    temp_letter.append(string[i])

            elif "v" in noise_type:
                if i in indices:
                    if string[i] in vowels:
                        val = random.randint(0, len(vowel_map[string[i]]) - 1)
                        temp_letter.append(vowel_map[string[i]][val])
                    else:
                        temp_letter.append(string[i])
                else:
                    temp_letter.append(string[i])
            else:
                if string[i] in consonants:
                    if i in indices:
                        val = random.randint(0, len(consonant_map[string[i]]) - 1)
                        temp_letter.append(consonant_map[string[i]][val])
                    else:
                        temp_letter.append(string[i])
                elif string[i] in vowels:
                    if i in indices:
                        val = random.randint(0, len(vowel_map[string[i]]) - 1)
                        temp_letter.append(vowel_map[string[i]][val])
                    else:
                        temp_letter.append(string[i])
                else:
                    temp_letter.append(string[i])
        string = "".join(temp_letter)
        return string

    def noise(self, noise_type, max_noise_ratio=0.2, change_all_strings=False):
        mod_dset = {}
        mod_dset['sentence1'] = []
        mod_dset['sentence2'] = []
        for sent1, sent2, ground_truth in zip(self.original_dset['sentence1'], self.original_dset['sentence2'],
                                              self.original_dset['ground_truth_similarity']):
            temp1 = self.replace_letter(noise_type, sent1, max_noise_ratio)
            mod_dset['sentence1'].append(temp1)
            temp2 = self.replace_letter(noise_type, sent2, max_noise_ratio)
            mod_dset['sentence2'].append(temp2)
        return mod_dset

    def view(self):
        for sent1, sent2, ground_truth in zip(self.original_dset['sentence1'], self.original_dset['sentence2'],
                                              self.original_dset['ground_truth_similarity']):
            print(sent1, sent2)

    def _get_original_dset(self):
        return self.original_dset


def save_dataset(dset, path):
    pickle.dump(dset, open(path, "wb"))
    return True


def load_dataset(path):
    obj = pickle.load(open(path, "rb"))
    return obj



if __name__ == '__main__':
    path = "/home/arghya/Documents/Thesis/Datasets/STS_test.xlsx"
    dataprep = DataPrepare(path)
    consonant = dataprep.noise("cons")
    vowel = dataprep.noise("vowel")
    random = dataprep.noise("rand")

    original = dataprep._get_original_dset()
##creating dataset
    dataset = Data("STS")
    dataset._assign_original(original)
    dataset._assign_noise_consonant(consonant, ratio=0.2)
    dataset._assign_noise_vowel(vowel, ratio=0.2)
    dataset._assign_noise_random(random, ratio=0.2)

    path = "/home/arghya/PycharmProjects/sts_letter.pkl"
    save_dataset(dataset, path)

    load_dset = load_dataset(path)
