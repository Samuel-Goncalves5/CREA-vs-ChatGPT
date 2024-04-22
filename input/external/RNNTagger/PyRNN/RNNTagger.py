
import sys
import torch
from torch import nn


class WordRepresentation(nn.Module):
   '''
   RNN for computing character-based word representations
   '''
   def __init__(self, num_chars, emb_size, rec_size, dropout_rate):
      super().__init__()

      # character embedding lookup table
      self.embeddings = nn.Embedding(num_chars, emb_size)

      # character-based LSTMs
      self.fwd_rnn = nn.LSTM(emb_size, rec_size)
      self.bwd_rnn = nn.LSTM(emb_size, rec_size)

      self.dropout = nn.Dropout(dropout_rate)
      
         
   def forward(self, fwd_charIDs, bwd_charIDs):
      # swap the 2 dimensions and lookup the embeddings
      fwd_embs = self.embeddings(fwd_charIDs.t())
      bwd_embs = self.embeddings(bwd_charIDs.t())

      # run the biLSTM over characters
      fwd_outputs, _ = self.fwd_rnn(fwd_embs)
      bwd_outputs, _ = self.bwd_rnn(bwd_embs)

      # concatenate the forward and backward final states to form
      # word representations
      word_reprs = torch.cat((fwd_outputs[-1], bwd_outputs[-1]), -1)
      
      return word_reprs


class ResidualLSTM(nn.Module):
   ''' Deep BiRNN with residual connections '''
   
   def __init__(self, input_size, rec_size, num_rnns, dropout_rate):
      super().__init__()
      self.rnn = nn.LSTM(input_size, rec_size, 
                         bidirectional=True, batch_first=True)

      self.deep_rnns = nn.ModuleList([
         nn.LSTM(2*rec_size, rec_size, bidirectional=True, batch_first=True)
         for _ in range(num_rnns-1)])
      
      self.dropout = nn.Dropout(dropout_rate)

   def forward(self, state):
      state, _ = self.rnn(state)
      for rnn in self.deep_rnns:
            hidden, _ = rnn(self.dropout(state))
            state = state + hidden # residual connection
      return state


class RNNTagger(nn.Module):
   ''' main tagger module '''

   def __init__(self, num_chars, num_tags, char_emb_size, char_rec_size, 
                word_rec_size, word_rnn_depth, dropout_rate, word_emb_size):

      super().__init__()

      # character-based BiLSTMs
      self.word_representations = WordRepresentation(num_chars, char_emb_size, 
                                                     char_rec_size, dropout_rate)
      # word-based BiLSTM
      self.word_rnn = ResidualLSTM(char_rec_size*2, word_rec_size, word_rnn_depth,
                                   dropout_rate)
      # output feed-forward network
      self.output_layer = nn.Linear(2*word_rec_size, num_tags)

      # dropout layers
      self.dropout = nn.Dropout(dropout_rate)

      # word embedding projection layer for finetuning on word embeddings
      if word_emb_size > 0:
         self.projection_layer = nn.Linear(2*char_rec_size, word_emb_size)


   def forward(self, fwd_charIDs, bwd_charIDs, word_embedding_training=False):
         
      # compute the character-based word representations
      word_reprs = self.word_representations(fwd_charIDs, bwd_charIDs)

      if word_embedding_training:
         if not hasattr(self, 'projection_layer'):
            sys.exit("Error: The embedding projection layer is undefined!")
         # Project the word representations to word embedding vectors
         # for finetuning on word embeddings as an auxiliary task
         word_embs = self.projection_layer(word_reprs)
         return word_embs

      # apply dropout
      word_reprs = self.dropout(word_reprs)
         
      # run the BiLSTM over words
      reprs = self.word_rnn(word_reprs.unsqueeze(0)).squeeze(0)
      reprs = self.dropout(reprs)  # and apply dropout
      
      # apply the output layers
      scores = self.output_layer(reprs)
      
      return scores
      
