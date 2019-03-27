#!/usr/bin/env python3
import argparse
import string
import os
import random
import matplotlib.pyplot as plt
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
    count = count[:35]
    X = list(range(len(count)))
    labels, Y = zip(*count)

    plt.figure()
    plt.plot(X, Y)
    plt.xticks(X, labels, rotation=90)
    plt.xlabel('Palavra')
    plt.ylabel('Frequência absoluta')
    plt.title('Curva de frequência das palavras no texto ' + filename)
    plt.tight_layout()
    save_filepath = 'figs/' + os.path.splitext(filename)[0] + '.png'
    plt.savefig(save_filepath)


def analyse_text(filepath):
    with open(filepath) as f:
        plain = f.read()
        tokens = tokenize(plain)
        count = get_count(tokens)

        plot_zipf_curve(count, os.path.split(filepath)[1])


def analyse_random_texts(dataset_path, num_samples):
    words = list()

    for _file in os.listdir(dataset):
        filename = dataset_path + '/' + os.fsdecode(_file)
        with open(filename) as f:
            tokens = tokenize(f.read())
            random_tokens = random.sample(tokens, num_samples)
            words.extend(random_tokens)
    count = get_count(words)
    plot_zipf_curve(count, 'randomized.txt')


if __name__ == "__main__":
    ARGS = PARSER.parse_args()
    dataset = os.fsencode(ARGS.dataset)

    if not os.path.exists('figs'):
        os.makedirs('figs')

    if not ARGS.randomize:
        for _file in os.listdir(dataset):
            filename = ARGS.dataset + '/' + os.fsdecode(_file)
            analyse_text(filename)
    else:
        analyse_random_texts(ARGS.dataset, ARGS.randomize)
