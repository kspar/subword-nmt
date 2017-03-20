#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Rico Sennrich

"""Use byte pair encoding (BPE) to learn a variable-length encoding of the vocabulary in a text.
Unlike the original BPE, it does not compress the plain text, but can be used to reduce the vocabulary
of a text to a configurable number of symbols, with only a small increase in the number of tokens.

Reference:
Rico Sennrich, Barry Haddow and Alexandra Birch (2016). Neural Machine Translation of Rare Words with Subword Units.
Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany.
"""

from __future__ import unicode_literals

import argparse
import codecs
import copy
import re
import sys
from collections import defaultdict, Counter
from functools import reduce
from io import open

argparse.open = open

# python 2/3 compatibility
if sys.version_info < (3, 0):
    sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
    sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
    sys.stdin = codecs.getreader('UTF-8')(sys.stdin)


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="learn BPE-based word segmentation")

    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Input text (default: standard input)")
    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w'), default=sys.stdout,
        metavar='PATH',
        help="Output file for BPE codes (default: standard output)")
    parser.add_argument(
        '--symbols', '-s', type=int, default=10000,
        help="Create this many new symbols (each representing a character n-gram) (default: %(default)s)")
    parser.add_argument(
        '--min-frequency', type=int, default=2, metavar='FREQ',
        help='Stop if no symbol pair has frequency >= FREQ (default: %(default)s)')
    parser.add_argument(
        '--verbose', '-v', action="store_true",
        help="verbose mode.")
    parser.add_argument(
        '--morph-as-char', '-m', action='store_true',
        help='Start with morphemes as BPE starting tokens instead of characters. ' +
             'Input words must be segmented into morphemes.')
    parser.add_argument(
        '--delimiter', '-d', type=str, default='==',
        help='Delimiter used to separate morphemes (default: %(default)s)')
    parser.add_argument(
        '--morph-aware', action='store_true',
        help="If you need help, you're stupid and should become a codemonkey-sheeple-person instead.")

    return parser


def get_vocabulary(fobj):
    """Read text and return dictionary that encodes vocabulary
    """
    vocab = Counter()
    for line in fobj:
        for word in line.split():
            vocab[word] += 1
    return vocab


def update_pair_statistics(pair, changed, stats, indices):
    """Minimally update the indices and frequency of symbol pairs

    if we merge a pair of symbols, only pairs that overlap with occurrences
    of this pair are affected, and need to be updated.
    """
    stats[pair] = 0
    indices[pair] = defaultdict(int)
    first, second = pair
    new_pair = first + second
    for j, word, old_word, freq in changed:

        if '==' in old_word and '==' not in word:
            # Big changes (in your life tomorrow)
            # for morpheme in split_tuple(old_word, '=='):
            #     prev_char = morpheme[0]
            #     for char in morpheme[1:]:
            #         indices[prev_char, char][j] -= 1
            #         prev_char = char
            # indices[first, second][j] -=1

            # Kõik indeksid invalideeritakse jube enne

            # Ei ole vaja paaride tõenäosuseid vanas sõnas uutest statsidest lahutada,
            # sest seal sai olla ainult üks paar (see, mida viimasena liideti) ja seda on uutes statsides niikuinii 0

            # add newly discovered pairs to stats & index
            prev_token = word[0]
            for token in word[1:]:
                    stats[prev_token, token] += freq
                    indices[prev_token, token][j] += 1
                    prev_token = token
        else:
            # find all instances of pair, and update frequency/indices around it
            i = 0
            while True:
                try:
                    i = old_word.index(first, i)
                except ValueError:
                    break
                if i < len(old_word) - 1 and old_word[i + 1] == second:
                    if i:
                        prev = old_word[i - 1:i + 1]
                        if '==' not in prev:
                            stats[prev] -= freq
                            indices[prev][j] -= 1
                    if i < len(old_word) - 2:
                        # don't double-count consecutive pairs
                        if old_word[i + 2] != first or i >= len(old_word) - 3 or old_word[i + 3] != second:
                            nex = old_word[i + 1:i + 3]
                            if '==' not in nex:
                                stats[nex] -= freq
                                indices[nex][j] -= 1
                    i += 2
                else:
                    i += 1

            i = 0
            while True:
                try:
                    i = word.index(new_pair, i)
                except ValueError:
                    break
                if i:
                    prev = word[i - 1:i + 1]
                    if '==' not in prev:
                        stats[prev] += freq
                        indices[prev][j] += 1
                # don't double-count consecutive pairs
                if i < len(word) - 1 and word[i + 1] != new_pair:
                    nex = word[i:i + 2]
                    if '==' not in nex:
                        stats[nex] += freq
                        indices[nex][j] += 1
                i += 1


def split_tuple(tpl, delim):
    lst = []
    start_idx = 0
    end_idx = -1
    while True:
        try:
            end_idx = tpl.index(delim, start_idx)
        except ValueError:
            lst.append(tpl[end_idx + 1:])
            break
        lst.append(tpl[start_idx: end_idx])
        start_idx = end_idx + 1

    return tuple(lst)


def get_pair_statistics(vocab):
    """Count frequency of all symbol pairs, and create index"""

    # data structure of pair frequencies
    stats = defaultdict(int)

    # index from pairs to words
    indices = defaultdict(lambda: defaultdict(int))

    for i, (word, freq) in enumerate(vocab):
        for morpheme in split_tuple(word, '=='):
            prev_char = morpheme[0]
            for char in morpheme[1:]:
                stats[prev_char, char] += freq
                indices[prev_char, char][i] += 1
                prev_char = char

    return stats, indices


def replace_pair(pair, vocab, indices):
    """Replace all occurrences of a symbol pair ('A', 'B') with a new symbol 'AB'"""
    first, second = pair
    pair_str = ''.join(pair)
    pair_str = pair_str.replace('\\', '\\\\')
    changes = []
    pattern = re.compile(r'(?<!\S)' + re.escape(first + ' ' + second) + r'(?!\S)')
    if sys.version_info < (3, 0):
        iterator = indices[pair].iteritems()
    else:
        iterator = indices[pair].items()
    for j, freq in iterator:
        if freq < 1:
            continue
        word, freq = vocab[j]
        new_word = ' '.join(word)
        new_word = pattern.sub(pair_str, new_word)
        new_word_tpl = tuple(new_word.split())

        if len(new_word_tpl) - 2 * new_word_tpl.count('==') - 1 == 0:
            # If magic then allow to merge morphemes
            new_word_tpl = tuple(filter(lambda t: t != '==', new_word_tpl))

        vocab[j] = (new_word_tpl, freq)
        changes.append((j, new_word_tpl, word, freq))

    return changes


def prune_stats(stats, big_stats, threshold):
    """Prune statistics dict for efficiency of max()

    The frequency of a symbol pair never increases, so pruning is generally safe
    (until we the most frequent pair is less frequent than a pair we previously pruned)
    big_stats keeps full statistics for when we need to access pruned items
    """
    for item, freq in list(stats.items()):
        if freq < threshold:
            del stats[item]
            if freq < 0:
                big_stats[item] += freq
            else:
                big_stats[item] = freq


if __name__ == '__main__':

    parser = create_parser()
    args = parser.parse_args()

    vocab = get_vocabulary(args.input)
    if args.morph_as_char:
        vocab = dict([(tuple(x.split(args.delimiter)) + ('</w>',), y) for (x, y) in vocab.items()])
    elif args.morph_aware:
        # You ask why. I ask why not.
        vocab = dict([(tuple(reduce(lambda t1, t2: t1 + (args.delimiter,) + t2, map(lambda p: tuple(p), x.split(args.delimiter)))) + ('</w>',), y) for (x, y) in
                      vocab.items()])
    else:
        vocab = dict([(tuple(x) + ('</w>',), y) for (x, y) in vocab.items()])
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)

    stats, indices = get_pair_statistics(sorted_vocab)
    big_stats = copy.deepcopy(stats)
    # threshold is inspired by Zipfian assumption, but should only affect speed
    threshold = max(stats.values()) / 10
    for i in range(args.symbols):
        if stats:
            most_frequent = max(stats, key=stats.get)

        # we probably missed the best pair because of pruning; go back to full statistics
        if not stats or (i and stats[most_frequent] < threshold):
            prune_stats(stats, big_stats, threshold)
            stats = copy.deepcopy(big_stats)
            most_frequent = max(stats, key=stats.get)
            # threshold is inspired by Zipfian assumption, but should only affect speed
            threshold = stats[most_frequent] * i / (i + 10000.0)
            prune_stats(stats, big_stats, threshold)

        if stats[most_frequent] < args.min_frequency:
            sys.stderr.write('no pair has frequency >= {0}. Stopping\n'.format(args.min_frequency))
            break

        if args.verbose:
            sys.stderr.write('pair {0}: {1} {2} -> {1}{2} (frequency {3})\n'.format(i, most_frequent[0], most_frequent[1], stats[most_frequent]))
        args.output.write('{0} {1}\n'.format(*most_frequent))
        changes = replace_pair(most_frequent, sorted_vocab, indices)
        update_pair_statistics(most_frequent, changes, stats, indices)
        stats[most_frequent] = 0
        if not i % 100:
            prune_stats(stats, big_stats, threshold)
