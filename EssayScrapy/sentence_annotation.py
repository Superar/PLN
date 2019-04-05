from nltk import sent_tokenize
import os


def main():
    for root, dirs, files in os.walk("essay"):
        for file_path in files:

            file_r = open('essay/{}'.format(file_path), 'r')
            file_w = open('essay/annotation/{}'.format(file_path.split('/')[-1]), 'w')

            for line in file_r:

                for sentence in sent_tokenize(line):
                    file_w.write(sentence)
                    file_w.write('\n\n')

            file_r.close()
            file_w.close()


if __name__ == "__main__":
    main()
