
import sys
from collections import Counter, OrderedDict 
import pickle
import numpy

unk_string   = '<UNK>'
pad_string   = '<PAD>'

def read_tagged_sentences(path, max_sent_len):
   """
   Read a dataset.
   Each line consists of a token and a tag separated by a tab character
   """
   sentences, words, tags = [], [], []
   with open(path) as file:
      for line in file:
         line = line.rstrip()
         if line:
            word, tag, *_ = line.split("\t")
            words.append(word)
            tags.append(tag)
         else:
            # empty line marking the end of a sentence
            if 0 < len(words) < max_sent_len:
               sentences.append((words, tags))
            words, tags = [], []
   return sentences


def read_word_embeddings(filename):
   # Read word embeddings from file.
   word_embeddings = []
   if filename is not None:
      print("reading word embeddings ...", file=sys.stderr)
      with open(filename) as file:
         for line in file:
            word, *vec = line.rstrip().split(' ')
            if word != unk_string:
               word_embeddings.append((word, numpy.array(vec, dtype=numpy.float32)))
      print("done", file=sys.stderr)
   word_emb_size = len(word_embeddings[0][1]) if word_embeddings else 0
   return word_embeddings, word_emb_size
            

def make_dict(counter, min_freq=0, add_pad_symbol=False):
   """
   Create a dictionary which maps strings with some minimal frequency to numbers.
   We don't use pack_padded sequence, so it is OK to assign ID 1 to the
   padding symbol.
   """
   symlist = [unk_string] + ([pad_string] if add_pad_symbol else []) + \
             [elem for elem,freq in counter.most_common() if freq>=min_freq]
   string2ID = {elem:i for i,elem in enumerate(symlist)}
   return string2ID, symlist


class Data(object):
   """
   class for reading a tagged training and development corpus or a test corpus
   """
   
   IGNORE_INDEX = -100

   def __init__(self, *args):
      if len(args) == 1:
         self.init_test(*args)
      else:
         self.init_train(*args)

   ### functions needed during training ###############################################

   def init_train(self, path_train, path_dev, word_trunc_len,
                  min_char_freq, max_sent_len, word_embeddings, ignore_tag):

      self.word_trunc_len = word_trunc_len  # length to which words are truncated or filled up

      # reading the datasets
      self.train_sentences = read_tagged_sentences(path_train, max_sent_len)
      self.dev_sentences   = read_tagged_sentences(path_dev, max_sent_len)
   
      ### create dictionaries which map characters or tags to IDs
      char_counter = Counter()
      tag_counter  = Counter()
      for words, tags in self.train_sentences:
         tag_counter.update(tags)
         for word in words:
            char_counter.update(word)
      self.char2ID, _ = make_dict(char_counter, min_char_freq, add_pad_symbol=True)

      if ignore_tag is not None:
         tag_counter.pop(ignore_tag, None) # remove this special tag if present
         self.tag2ID, self.ID2tag  = make_dict(tag_counter)
         self.tag2ID[ignore_tag] = self.IGNORE_INDEX  # empty tags will not be trained
      else:
         self.tag2ID, self.ID2tag  = make_dict(tag_counter)

      ### sizes of the symbol inventories
      self.num_char_types = len(self.char2ID)
      self.num_tag_types  = len(self.ID2tag)

      self.word_embeddings, self.word_emb_size = read_word_embeddings(word_embeddings)
      

   def get_charIDs(self, word):
      '''
      maps a word to a sequence of character IDs
      '''

      unkID = self.char2ID[unk_string]
      padID = self.char2ID[pad_string]

      charIDs = [self.char2ID.get(c, unkID) for c in word]

      # add enough padding symbols
      fwd_charIDs = [padID] * self.word_trunc_len + charIDs
      bwd_charIDs = [padID] * self.word_trunc_len + charIDs[::-1]

      # truncate
      fwd_charIDs = fwd_charIDs[-self.word_trunc_len:]
      bwd_charIDs = bwd_charIDs[-self.word_trunc_len:]

      return fwd_charIDs, bwd_charIDs


   def words2charIDvec(self, words):
      """
      converts words to char-ID vectors
      """

      ### convert words to character ID sequences
      fwd_charID_seqs = []
      bwd_charID_seqs = []
      for word in words:
         fwd_charIDs, bwd_charIDs = self.get_charIDs(word)
         fwd_charID_seqs.append(fwd_charIDs)
         bwd_charID_seqs.append(bwd_charIDs)

      fwd_charID_seqs = numpy.asarray(fwd_charID_seqs, dtype='int32')
      bwd_charID_seqs = numpy.asarray(bwd_charID_seqs, dtype='int32')

      return fwd_charID_seqs, bwd_charID_seqs


   def tags2IDs(self, tags):
      """
      takes a list of tags and converts them to IDs using the tag2ID dictionary
      """
      unkID = self.tag2ID[unk_string]
      IDs = [self.tag2ID.get(tag, unkID) for tag in tags]
      return numpy.asarray(IDs, dtype='int32')


   def save_parameters(self, filename):
      """ save parameters to a file """
      all_params = (self.word_trunc_len, self.char2ID, self.ID2tag)
      with open(filename, "wb") as file:
         pickle.dump(all_params, file)


   ### functions needed during tagging ###############################################

   def init_test(self, filename):
      """ load parameters from a file """
      with open(filename, "rb") as file:
         self.word_trunc_len, self.char2ID, self.ID2tag = pickle.load(file)

   def sentences(self, filename):
      """ read data to be tagged. One token per line. Empty line follows a sentence """
      with open(filename) as f:
         words = []
         for line in f:
            line = line.rstrip()
            if line != '':
               words.append(line)
            elif len(words) > 0:
               # empty line indicates the end of a sentence
               yield words
               words = []

   def IDs2tags(self, IDs):
      """ takes a list of IDs and converts them to tags using the ID2tag dictionary """
      return [self.ID2tag[int(ID)] for ID in IDs]
