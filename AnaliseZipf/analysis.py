#!/usr/bin/env python3
import argparse
import string
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from nltk.tokenize import word_tokenize

import scipy.stats as stats
from scipy.stats import norm
import math

PARSER = argparse.ArgumentParser()
PARSER.add_argument('--dataset', '-d', type=str, required=True,
                    help='Path to data set directory with multiple plain text files or a single file (.txt)')
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

    # Gaussian
    mu, sigma = norm.fit(Y)

    count, bins, ignored = plt.hist(Y, len(Y), density=True)

    plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
             np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)), linewidth=3, color='y')

    plt.xlabel('Frequência')
    plt.ylabel('Probabilidade')
    plt.title('Análise da Distribuição do texto ' + filename)
    plt.tight_layout()
    save_filepath = 'figs/norm_' + os.path.splitext(filename)[0] + '.png'
    plt.savefig(save_filepath)

    plt.figure()

    # BoxPlot

    fig, ax = plt.subplots()
    ax.boxplot(Y)

    save_filepath = 'figs/boxplot_' + os.path.splitext(filename)[0] + '.png'
    plt.savefig(save_filepath)

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


def process_curve(filename):
    with open(filename) as f:
        plain = f.read()
        tokens = tokenize(plain)
        count = get_count(tokens)

        plot_zipf_curve(count, os.path.split(filename)[1])


def main(args):
    dataset = os.fsencode(args.dataset)

    if not os.path.exists('figs'):
        os.makedirs('figs')

    if not args.randomize:
        if os.path.isdir(dataset):
            for _file in os.listdir(dataset):
                filename = args.dataset + '/' + os.fsdecode(_file)
                process_curve(filename)

        elif os.path.isfile(dataset) and str(dataset).replace('\'', '').split('.')[-1] == 'txt':
            filename = os.fsdecode(dataset)
            process_curve(filename)
        else:
            raise ValueError(
                'Invalid File Format. Please, use TXT File.\n'
                ' You could use lerolero_extractor.py to create a TXT test base')

    else:
        analyse_random_texts(args.dataset, args.randomize)


if __name__ == "__main__":
    ARGS = PARSER.parse_args()
    main(ARGS)
