
import sys
import pickle
import random
from collections import Counter

from OODData import OODData, select, boundary_symbol

pad_symbol = '<pad>'
unk_symbol = '<unk>'
ood_symbol = '<ood>'

padID = 0
unkID = 1

num_batches_in_big_batch = 5


def rstrip_zeros(wordIDs):
   """ removes trailing padding symbols """
   wordIDs = list(wordIDs)
   if padID in wordIDs:
      wordIDs = wordIDs[:wordIDs.index(padID)]
   return wordIDs


### helper functions ###############################################

def read_sentences(path):
   with open(path) as file:
      return [line.split() for line in file]

   
def build_dict(data, max_vocab_size):
   """ 
   builds
   - a dictionary which maps the most frequent words to indices and
   - a table which maps indices to the most frequent words
   """
   word_freq = Counter(word for sent in data for word in sent)
   max_vocab_size = min(max_vocab_size, len(word_freq))
            
   words, _ = zip(*word_freq.most_common(max_vocab_size))
   # ID of pad_symbol must be 0
   words = [pad_symbol, unk_symbol] + list(words)
   word2ID = {w:i for i,w in enumerate(words)}
   
   return word2ID, words


def pad_batch(batch):
   """ pad sequences in batch with 0s to obtain sequences of identical length """
   
   seq_len = [len(sent) for sent in batch]
   max_len = max(seq_len)
   padded_batch = [seq + [0] * (max_len - len(seq)) for seq in batch]
   return padded_batch, seq_len
      
      
def words2IDs(words, word2ID):
   """ maps a list of words to a list of IDs """
   unkID = word2ID[unk_symbol]
   return [word2ID.get(w, unkID) for w in words]

   
### class Data ####################################################

class Data(object):
   """ class for data preprocessing """

   def __init__(self, *args):
      if len(args) == 2:
         # Initialisation for translation
         self.init_test(*args)
      else:
         # Initialisation for training
         self.init_train(*args)
   
         
   ### functions needed during training ##########################

   def init_train(self, path_train_src, path_train_tgt, path_dev_src, path_dev_tgt,
                  max_src_vocab_size, max_tgt_vocab_size, max_sent_len, max_batch_size,
                  ood_rate):
      """ reads the training and development data and generates the mapping tables """
      
      self.max_sent_len = max_sent_len
      self.max_batch_size = max_batch_size
      self.ood_rate = ood_rate

      train_src_sentences = read_sentences(path_train_src)
      train_tgt_sentences = read_sentences(path_train_tgt)
      dev_src_sentences = read_sentences(path_dev_src)
      dev_tgt_sentences = read_sentences(path_dev_tgt)

      self.max_tgt_len_factor = max((len(t) - 5) / len(s)
                                    for s, t in zip(train_src_sentences, train_tgt_sentences))
      
      self.src2ID, self.ID2src = build_dict(train_src_sentences, max_src_vocab_size)

      self.tgt2ID, self.ID2tgt = build_dict(train_tgt_sentences, max_tgt_vocab_size)

      self.train_batches = self.build_batches(train_src_sentences, train_tgt_sentences)
      self.dev_batches = self.build_batches(dev_src_sentences, dev_tgt_sentences, dev=True)
      
      if ood_rate > 0.0:
         
         # Create a module which generated synthetic data for
         # recognizing out of distribution data.  This is only
         # applicable in lemmatization applications whose input
         # consists of a sequence of single-letter tokens (the word),
         # the separator token '##' and another sequence of
         # single-letter tokens (the POS tag).

         # add the OOD symbol to the target vocabulary
         self.tgt2ID[ood_symbol] = oodID = len(self.ID2tgt) # new ID
         self.ID2tgt.append(ood_symbol)

         bID = self.src2ID[boundary_symbol]
         
         self.ood = OODData(self.train_batches, self.src2ID[boundary_symbol], oodID)

      self.src_vocab_size = len(self.ID2src)
      self.tgt_vocab_size = len(self.ID2tgt)


   def build_batches(self, src_data, tgt_data, dev=False):
      # sort sentence pairs by length
      data = sorted(zip(src_data, tgt_data), key=lambda x: len(x[0])+len(x[1]), reverse=True)
      
      def add_batch(batches, batch):
         # sort batch by source sentence length
         batch.sort(key=lambda x: len(x[0]), reverse=True)
         src_vecs, tgt_vecs = zip(*batch)
         # padding
         batches.append((pad_batch(src_vecs), pad_batch(tgt_vecs)))
         
      # extract mini-batches of similar length (in number of words)
      batches, batch = [], []
      for src_sent, tgt_sent in data:
         # filter out sentences which are too long
         ls, lt = len(src_sent), len(tgt_sent)
         if (ls > self.max_sent_len or
             (dev and ls * (1.0 + self.max_tgt_len_factor + 7) > self.max_batch_size) or
             (not dev and ls + lt + 2 > self.max_batch_size)):
            continue
         # add the sentence to the current batch
         batch.append((words2IDs(src_sent, self.src2ID),
                       words2IDs(tgt_sent, self.tgt2ID)))
         # compute the size of the current batch
         max_tl = max(len(src_sent) for src_sent, _ in batch) * self.max_tgt_len_factor + 7 \
            if dev else max(len(tgt_sent) for _, tgt_sent in batch) + 2
            
         batch_size = sum(len(src_sent) for src_sent, _ in batch) + len(batch) * max_tl
         if batch_size > self.max_batch_size:
            # cut off the last sentence
            next_batch, batch = batch[:-1], batch[-1:]
            add_batch(batches, next_batch)
      # add the last batch
      add_batch(batches, batch)
      return batches
      
      # batches, batch, L = [], [], 0
      # for src_sent, tgt_sent in data:
      #    l = len(src_sent) + len(tgt_sent)
      #    if len(src_sent) > self.max_sent_len or l > self.max_batch_size:
      #       continue   # ignore sentence
      #    if L + l > self.max_batch_size:  # the batch is full
      #       # sort batch by source sentence length
      #       batch.sort(key=lambda x: len(x[0]), reverse=True)
      #       src_vecs, tgt_vecs = zip(*batch)
      #       batches.append((pad_batch(src_vecs), pad_batch(tgt_vecs)))
      #       batch, L = [], 0
      #    batch.append((words2IDs(src_sent, self.src2ID),
      #                  words2IDs(tgt_sent, self.tgt2ID)))
      #    L += l
      # batches.append((pad_batch(src_vecs), pad_batch(tgt_vecs)))
      # return batches
      
      # # extract mini-batches of similar length (in number of words)
      # batches = []
      # src_vecs, tgt_vecs, L = [], [], 0
      # for src_sent, tgt_sent in data:
      #    l = len(src_sent) + len(tgt_sent)
      #    if len(src_sent) > self.max_sent_len or l > self.max_batch_size:
      #       continue   # ignore sentence
      #    if L + l > self.max_batch_size:  # the batch is full
      #       batches.append((pad_batch(src_vecs), pad_batch(tgt_vecs)))
      #       src_vecs, tgt_vecs, L = [], [], 0
      #    src_vecs.append(words2IDs(src_sent, self.src2ID))
      #    tgt_vecs.append(words2IDs(tgt_sent, self.tgt2ID))
      #    L += l
      # batches.append((pad_batch(src_vecs), pad_batch(tgt_vecs)))
      # return batches
               
   def training_batches(self):
      # shuffle sentences before each new epoch
      random.shuffle(self.train_batches)
      if self.ood_rate == 0:
         for batch in self.train_batches:
            yield batch
      else:
         i = 0
         while i < len(self.train_batches):
            # decide whether to generate an OOD sample
            if select([True, False], [self.ood_rate, 1.0]):
               # Turn a random batch into an OOD batch
               yield self.ood.get_batch(random.choice(self.train_batches))
            else:
               yield self.train_batches[i]
               i += 1
      
   def development_batches(self):
      for batch in self.dev_batches:
         yield batch

         
   def save_parameters(self, filename):
      """ save the module's parameters to a file """
      all_params = (self.ID2src, self.ID2tgt, self.max_tgt_len_factor)
      with open(filename, "wb") as file:
         pickle.dump(all_params, file)

      
   ### functions needed for translation ############################

   def init_test(self, filename, batch_size):
      """ load parameters from a file """

      self.batch_size = batch_size
      with open(filename, "rb") as file:
         self.ID2src, self.ID2tgt, self.max_tgt_len_factor = pickle.load(file)
         self.src2ID = {w:i for i,w in enumerate(self.ID2src)}
         self.tgt2ID = {w:i for i,w in enumerate(self.ID2tgt)}
         
   def build_test_batch(self, batch):
      batch_IDs = [words2IDs(srcWords, self.src2ID) for srcWords in batch]
      result = sorted(enumerate(batch_IDs), key=lambda x: -len(x[1]))
      orig_sent_pos, sorted_batch_IDs = zip(*result)

      new_sent_pos, _ = zip(*sorted(enumerate(orig_sent_pos), key=lambda x: x[1]))

      return batch, new_sent_pos, pad_batch(sorted_batch_IDs)

   def test_batches(self, file):
      """ yields the next batch of test sentences """

      batch = []
      for line in file:
         srcWords = line.split()
         batch.append(srcWords)
         if len(batch) == self.batch_size:
            yield self.build_test_batch(batch)
            batch = []

      if len(batch) > 0:
         yield self.build_test_batch(batch)
         
   def source_words(self, wordIDs):
      """ maps IDs to source word strings """
      return [self.ID2src[id] for id in wordIDs]

   def target_words(self, wordIDs):
      """ maps IDs to target word strings """
      return [self.ID2tgt[id] for id in wordIDs if id > 0]

