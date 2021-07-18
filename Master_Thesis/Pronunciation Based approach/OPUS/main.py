import pandas as pd
import pickle
import numpy as np
import random
import string
import math
import sys

from dataprep import DataPrepare, Data, save_dataset, load_dataset
from scoring import ScoreEval

def prepare():
    #### preparing dataset main
    path = "/home/arghya/Documents/Thesis/Datasets/OPUS/opusparcus_en/opusparcus_v1/en/dev/en-dev.txt"
    dataprep = DataPrepare(path)
    # obj.view()
    consonant = dataprep.noise("cons")
    vowel = dataprep.noise("vowel")
    random = dataprep.noise("rand")

    original = dataprep._get_original_dset()
    ##creating dataset
    dataset = Data("OPUS")
    dataset._assign_original(original)
    dataset._assign_noise_consonant(consonant, ratio=0.2)
    dataset._assign_noise_vowel(vowel, ratio=0.2)
    dataset._assign_noise_random(random, ratio=0.2)

    # saving dataset

    path = "/home/arghya/PycharmProjects/opus_letter.pkl"
    save_dataset(dataset, path)

    # load_dset = load_dataset(path)


if __name__ == '__main__':
    import time

    start_time = time.time()
    # prepare()
    path = "/home/arghya/PycharmProjects/opus_letter.pkl"
    load_dset = load_dataset(path)

    dictionary = {}
    dictionary["ground_truth_original"] = []
    dictionary["original_sent1"] = []
    dictionary["original_sent2"] = []
    dictionary["consonant_sent1"] = []
    dictionary["consonant_sent2"] = []
    dictionary["vowel_sent1"] = []
    dictionary["vowel_sent2"] = []
    dictionary["random_sent1"] = []
    dictionary["random_sent2"] = []

    load_original = load_dset._get_original()
    for sent1, sent2, ground_truth in zip(load_original['sentence1'], load_original['sentence2'],
                                          load_original['ground_truth_similarity']):
        dictionary["ground_truth_original"].append(ground_truth)
        dictionary["original_sent1"].append(sent1)
        dictionary["original_sent2"].append(sent2)

    _, load_consonant = load_dset._get_noise_consonant()

    for sent1, sent2 in zip(load_consonant['sentence1'], load_consonant['sentence2']):
        dictionary["consonant_sent1"].append(sent1)
        dictionary["consonant_sent2"].append(sent2)

    _, load_vowel = load_dset._get_noise_vowel()

    for sent1, sent2 in zip(load_vowel['sentence1'], load_vowel['sentence2']):
        dictionary["vowel_sent1"].append(sent1)
        dictionary["vowel_sent2"].append(sent2)

    _, load_random = load_dset._get_noise_random()

    for sent1, sent2 in zip(load_random['sentence1'], load_random['sentence2']):
        dictionary["random_sent1"].append(sent1)
        dictionary["random_sent2"].append(sent2)

    obj = ScoreEval()
    counter = 1

    for gt, a, b, c, d, e, f, g, h in zip(dictionary["ground_truth_original"], dictionary["original_sent1"],
                                          dictionary["original_sent2"], dictionary["consonant_sent1"],
                                          dictionary["consonant_sent2"], dictionary["vowel_sent1"],
                                          dictionary["vowel_sent2"], dictionary["random_sent1"],
                                          dictionary["random_sent2"]):
        cosine, leven, wmd = (obj.flair_similarity(a, b), obj.levenshtein_ratio_and_distance(a, b), obj.wmd(a, b))
        print("Label: {}  --- ground truth: {}--- Pair: {} --Sentence1: {} --Sentence2: {}".format("original",
                                                                                                   float(gt / 4),
                                                                                                   str(counter), a, b))
        print("Cosine Similarity: {} --- Levenshtein Similarity: {} --- WMD Similarity: {}".format(cosine, leven, wmd))
        cosine, leven, wmd = (obj.flair_similarity(c, d), obj.levenshtein_ratio_and_distance(c, d), obj.wmd(c, d))
        print("Label: {} --- Pair: {} --Sentence1: {} --Sentence2: {}".format("consonant_noise", str(counter), c, d))
        print("Cosine Similarity: {} --- Levenshtein Similarity: {} --- WMD Similarity: {}".format(cosine, leven, wmd))
        cosine, leven, wmd = (obj.flair_similarity(e, f), obj.levenshtein_ratio_and_distance(e, f), obj.wmd(e, f))
        print("Label: {} --- Pair: {} --Sentence1: {} --Sentence2: {}".format("vowel_noise", str(counter), e, f))
        print("Cosine Similarity: {} --- Levenshtein Similarity: {} --- WMD Similarity: {}".format(cosine, leven, wmd))
        cosine, leven, wmd = (obj.flair_similarity(g, h), obj.levenshtein_ratio_and_distance(g, h), obj.wmd(g, h))
        print("Label: {} --- Pair: {} --Sentence1: {} --Sentence2: {}".format("mixed_noise", str(counter), g, h))
        print("Cosine Similarity: {} --- Levenshtein Similarity: {} --- WMD Similarity: {}".format(cosine, leven, wmd))

        counter = counter + 1

    print("--- %s seconds ---" % (time.time() - start_time))


