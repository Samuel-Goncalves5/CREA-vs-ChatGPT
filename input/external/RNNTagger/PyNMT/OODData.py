#!/usr/bin/python3

import sys
from collections import Counter
import itertools
import random

boundary_symbol = '##'


def select(elems, cum_probs, k=None):
    if k is None:
        return random.choices(elems, cum_weights=cum_probs)[0]
    return random.choices(elems, cum_weights=cum_probs, k=k)


class OODData:

    @staticmethod
    def elements_and_cumprobs(elems):
        freq = Counter(elems)   # count element freqs
        total = sum(freq.values()) # compute total number
        prob = [(e, f/total) for e, f in freq.most_common()] # normalize to probs
        elems, probs = zip(*prob)
        return elems, list(itertools.accumulate(probs))

    
    def __init__(self, batches, boundaryID, oodID):
        self.oodID = oodID
        self.boundaryID = boundaryID
        
        srcID_seqs = [srcIDs for (seqs, _), _ in batches for srcIDs in seqs]
        letterID_seqs = [seq[:seq.index(boundaryID)] for seq in srcID_seqs]

        self.letters, self.cum_letter_probs = \
            self.elements_and_cumprobs(ID for IDs in letterID_seqs for ID in IDs)

     
    def get_batch(self, dummy_batch):
        # An OOD sample is created by
        # * replacing the target sequences with [<ood>]
        # * and replacing the source letter sequences with random letter sequences
        src_seqs, src_lengths = dummy_batch[0]
        lengths = [seq.index(self.boundaryID) for seq in src_seqs]
        new_words = [select(self.letters, self.cum_letter_probs, length)
                     for length in lengths]
        src_seqs = [word + seq[l:] for word, seq, l in zip(new_words, src_seqs, lengths)]
        return (src_seqs, src_lengths), ([[self.oodID]] * len(src_seqs), [1] * len(src_seqs))

