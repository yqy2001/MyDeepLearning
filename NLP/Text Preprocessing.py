"""
@Author:        禹棋赢
@StartTime:     2021/8/20 9:53
@Filename:      Text Preprocessing.py

preprocess a simple "time machine" dataset:
1 preprocess text.
2 Construct Vocabulary.
"""
import collections
import re
from d2l import torch as d2l

d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')


def read_time_machine():
    """Load the time machine dataset into a list of text lines."""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


def tokenize(lines, token_type='word'):
    """
    tokenize text lines into word or char tokens
    :param lines: list, list of str, texts
    :param token_type: str, token's type, choices: ['word', 'char']
    :return: tokenized list of lines
    """
    if token_type == 'word':
        return [line.split() for line in lines]
    elif token_type == 'char':
        return [list(line) for line in lines]
    else:
        print('ERROR: unknown token type: ' + token_type)


class Vocab:
    def __init__(self, tokens, min_freq=0, reserved_tokens=None):
        if not reserved_tokens:
            reserved_tokens = []
        self.unk = 0

        counter = count_corpus(tokens)
        # sort the counter items ((token, freq) tuple), use the idx 1 element of each tuple
        # reverse = True means sort by descending order
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx2token = ['<unk>'] + reserved_tokens
        self.token2idx = {token: idx for idx, token in enumerate(self.idx2token)}
        for token, freq in self.token_freqs:
            if freq < min_freq:
                break
            self.idx2token.append(token)
            self.token2idx[token] = len(self.idx2token) - 1

    def __len__(self):
        return len(self.idx2token)

    def __getitem__(self, tokens):
        """
        serialize the input tokens, the input and the output are of the same type
        """
        if not isinstance(tokens, (list, tuple)):
            return self.token2idx.get(tokens, 0)  # if token not in dict, return 0, i.e. unk token
        else:
            return [self.__getitem__(x) for x in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, list):
            return self.idx2token[indices]
        else:
            return [self.idx2token[token] for token in self.idx2token]


def count_corpus(tokens):
    """
    count token frequencies
    :param tokens: list, list of token list
    :return:
    """
    tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


def main():
    lines = read_time_machine()
    print("# text lines: {%d}" % len(lines))
    lines = tokenize(lines)
    vocab = Vocab(lines)
    corpus = [vocab[token] for line in lines for token in line]
    print(len(vocab), len(corpus))


if __name__ == '__main__':
    main()
