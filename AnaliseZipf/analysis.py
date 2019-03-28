#!/usr/bin/env python3
import argparse
import string
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from nltk.tokenize import word_tokenize

PARSER = argparse.ArgumentParser()
PARSER.add_argument('--dataset', '-d', type=str, required=True,
                    help='Path to data set directory with multiple plain text files')
PARSER.add_argument('--randomize', '-r', type=int,
                    required=False, default=0,
                    help='Number of samples to get random tokens from each text in data set')


def tokenize(plain):
    tokens = word_tokenize(plain, language='portuguese')
    tokens = [t.lower() for t in tokens if t not in string.punctuation]
    return tokens


def get_count(tokens):
    words = set(tokens)
    count = [(w, tokens.count(w)) for w in words]
    count.sort(key=lambda x: x[1], reverse=True)
    return count


def plot_zipf_curve(count, filename):
    X = list(range(len(count)))
    labels, Y = zip(*count)

    plt.figure()

    # Zipf
    plt.plot(X, Y)

    # Luhn
    data = list()
    for i in X:
        data.extend(Y[i] * [i])
    cut_min = np.round(np.percentile(data, 50))
    cut_max = np.round(np.percentile(data, 75))
    plt.axvline(x=cut_min, color='red')
    plt.axvline(x=cut_max, color='red')

    ticks = [int(np.round(np.percentile(X, i))) for i in range(0, 100, 3)]
    ticks_labels = [labels[int(t)] for t in ticks]
    plt.xticks(ticks, ticks_labels, rotation=90)
    plt.xlabel('Palavra')
    plt.ylabel('Frequência absoluta')
    plt.title('Curva de frequência das palavras no texto ' + filename)
    plt.tight_layout()
    save_filepath = 'figs/' + os.path.splitext(filename)[0] + '.png'
    plt.savefig(save_filepath)


def analyse_random_texts(dataset_path, num_samples):
    words = list()

    for _file in os.listdir(dataset_path):
        filename = dataset_path + '/' + os.fsdecode(_file)
        with open(filename) as f:
            tokens = tokenize(f.read())
            random_tokens = random.sample(tokens, num_samples)
            words.extend(random_tokens)
    count = get_count(words)
    plot_zipf_curve(count, 'randomized.txt')


def main(args):
    dataset = os.fsencode(args.dataset)

    if not os.path.exists('figs'):
        os.makedirs('figs')

    if not args.randomize:
        for _file in os.listdir(dataset):
            filename = args.dataset + '/' + os.fsdecode(_file)
            with open(filename) as f:
                plain = f.read()
                tokens = tokenize(plain)
                count = get_count(tokens)

                plot_zipf_curve(count, os.path.split(filename)[1])
    else:
        analyse_random_texts(args.dataset, args.randomize)


if __name__ == "__main__":
    ARGS = PARSER.parse_args()
    main(ARGS)
