
import sys
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

device = None


### Encoder Layer #############################################################

class EncoderLayer(nn.Module):
   """ implements an encoder layer """

   def __init__(self, input_size, rnn_size, dropout_rate):
      super().__init__()

      # add a residual connection if input and output sizes match
      self.residualConnection = (input_size == rnn_size*2)
      self.rnn = nn.LSTM(input_size, rnn_size,
                         batch_first=True, bidirectional=True)
      self.dropout = nn.Dropout(dropout_rate)

   def forward(self, reprs, seq_len):
      packed_input = pack_padded_sequence(self.dropout(reprs), seq_len,
                                          batch_first=True, enforce_sorted=False)
      output, _ = self.rnn(packed_input)
      result, _ = pad_packed_sequence(output, batch_first=True)
      if self.residualConnection:
         result = reprs + result
      return result


### Encoder ###################################################################

class Encoder(nn.Module):
   """ implements the encoder of the encoder-decoder architecture """
   
   def __init__(self, vocab_size, word_emb_size, rnn_size, rnn_depth,
                dropout_rate, emb_dropout_rate):
      
      super().__init__()

      self.embeddings = nn.Embedding(vocab_size, word_emb_size, padding_idx=0)
      input_size, dropout = word_emb_size, emb_dropout_rate
      self.rnns = nn.ModuleList()
      for _ in range(rnn_depth):
         self.rnns.append(EncoderLayer(input_size, rnn_size, dropout))
         input_size, dropout = 2*rnn_size, dropout_rate

      self.dropout = nn.Dropout(dropout_rate)

      
   def forward(self, wordIDs, seq_len):

      repr = self.embeddings(wordIDs)
      for rnn in self.rnns:
         repr = rnn(repr, seq_len)

      return self.dropout(repr)


### Attention #################################################################

class Attention(nn.Module):

   def __init__(self, enc_rnn_size, dec_rnn_size):
      
      super().__init__()

      self.projection = nn.Linear(enc_rnn_size*2+dec_rnn_size, dec_rnn_size)
      self.final_weights = nn.Parameter(torch.randn(dec_rnn_size))


   def forward(self, enc_states, dec_state, src_len=None):

      # Replicate dec_state along a new (sentence length) dimension
      exp_dec_states = dec_state.unsqueeze(1).expand(-1,enc_states.size(1),-1)

      # Replicate enc_state along the first (batch) dimension if it is 1
      # needed during beam search
      exp_enc_states = enc_states.expand(dec_state.size(0),-1,-1)

      # Append the decoder state to each encoder state
      input = torch.cat((exp_enc_states, exp_dec_states), dim=-1)
      
      # Apply a fully connected layer
      proj_input = torch.tanh(self.projection(input))
      
      # Multiply with the final weight vector to get a single attention score
      # Division by a normalization constant facilitates training
      scores = torch.matmul(proj_input, self.final_weights) / \
               math.sqrt(self.final_weights.size(0))

      if src_len:  # not beam search
         # mask all padding positions
         mask = [[0]*l + [-float('inf')]*(enc_states.size(1)-l) for l in src_len]
         mask = torch.Tensor(mask).to(device)
         scores = scores + mask
      
      # softmax across all encoder positions
      attn_probs = F.softmax(scores, dim=-1)
      
      # weighted average of encoder representations
      enc_context = torch.sum(enc_states*attn_probs.unsqueeze(2), dim=1)
      
      return enc_context


### Decoder Layer #############################################################

class DecoderLayer(nn.Module):
   """ implements a decoder layer """

   def __init__(self, repr_size, rnn_size, ctx_size, dropout_rate):
      super().__init__()

      input_size = repr_size + ctx_size
      self.rnn_size = rnn_size
      self.rnn = nn.LSTMCell(input_size, rnn_size)
      self.dropout = nn.Dropout(dropout_rate)
      self.residualConnection = (repr_size == rnn_size)

   def init_states(self, batch_size):
      self.hidden_states = torch.zeros(batch_size, self.rnn_size).to(device)
      self.cell_states = self.hidden_states

   def forward(self, reprs, ctx_vec):
      # LSTM sublayer
      tmp = self.dropout(torch.cat((reprs, ctx_vec), dim=-1))
      self.hidden_states, self.cell_states = \
         self.rnn(tmp, (self.hidden_states, self.cell_states))
      if self.residualConnection:
         self.hidden_states += reprs
      return self.hidden_states

         
### Decoder ###################################################################

class NMTDecoder(nn.Module):

   def __init__(self, src_vocab_size, tgt_vocab_size, word_emb_size,
                enc_rnn_size, dec_rnn_size, enc_depth, dec_depth, 
                dropout_rate, emb_dropout_rate, tie_embeddings=True):
      ''' intialize the model before training starts '''
      
      super().__init__()

      self.dec_depth = dec_depth
      self.tie_embeddings = tie_embeddings
      self.ctx_size = enc_rnn_size*2

      # create the encoder and attention sub-modules
      self.encoder = Encoder(src_vocab_size, word_emb_size, enc_rnn_size, enc_depth,
                             dropout_rate, emb_dropout_rate)
      self.attention = Attention(enc_rnn_size, dec_rnn_size)

      self.tgt_embeddings = nn.Embedding(tgt_vocab_size, word_emb_size)

      # create the (deep) decoder RNN
      input_size, dropout = word_emb_size, emb_dropout_rate
      self.layers = nn.ModuleList()
      for _ in range(dec_depth):
         self.layers.append(DecoderLayer(input_size, dec_rnn_size, self.ctx_size, dropout))
         input_size, dropout = dec_rnn_size, dropout_rate
      
      self.dropout = nn.Dropout(dropout_rate)

      # If we use tied input and output embeddings, we need to
      # project the final hidden state to the target embedding size.
      if self.tie_embeddings:
         # Create the projection layer
         self.output_proj = nn.Linear(dec_rnn_size+self.ctx_size, word_emb_size)
         # Create the output layer weight matrix
         self.output_layer = nn.Linear(word_emb_size, tgt_vocab_size)
         # Tie input and output embeddings
         self.output_layer.weight = self.tgt_embeddings.weight
      else:
         self.output_layer = nn.Linear(dec_rnn_size+self.ctx_size, tgt_vocab_size)

   
   def init_decoder(self, src_wordIDs, src_len):

      # Initialize the LSTM hidden and cell states
      batch_size = len(src_wordIDs)
      for layer in self.layers:
         layer.init_states(batch_size)

      # Run the encoder with the input sequence
      src_wordIDs = torch.LongTensor(src_wordIDs).to(device)
      enc_states  = self.encoder(src_wordIDs, src_len)
      enc_context = torch.zeros(batch_size, self.ctx_size).to(device)

      return enc_states, enc_context
   
   
   def decoder_step(self, prev_word_embs, enc_states,
                    enc_context, src_len=None):
      ''' runs a single decoder step '''

      reprs = prev_word_embs
      for i, layer in enumerate(self.layers):
         reprs = layer(reprs, enc_context)
         if i==0:
            enc_context = self.attention(enc_states, reprs, src_len)
         
      return reprs, enc_context

   
   def compute_scores(self, hidden_states, enc_contexts):
      """ computes the values of the output layer """
      hidden_states = self.dropout(torch.cat((hidden_states, enc_contexts), dim=-1))
      if self.tie_embeddings:
         hidden_states = self.output_proj(hidden_states)
      return self.output_layer(hidden_states)

   
   def forward(self, src_wordIDs, src_len, tgt_wordIDs):
      ''' forward pass of the network during training and evaluation on dev data '''

      self.train(True)
      enc_states, enc_context = self.init_decoder(src_wordIDs, src_len)
      
      # Look up the target word embeddings
      tgt_word_embs = self.tgt_embeddings(tgt_wordIDs)

      # Run the decoder for each target word and collect the hidden states
      hidden_states = []
      enc_contexts = []
      for i in range(tgt_word_embs.size(1)):
         hidden_state, enc_context = self.decoder_step(
            tgt_word_embs[:,i,:], enc_states, enc_context, src_len)
         hidden_states.append(hidden_state)
         enc_contexts.append(enc_context)

      # Compute the scores of the output layer
      hidden_states = torch.stack(hidden_states, dim=1)
      enc_contexts = torch.stack(enc_contexts, dim=1)
      scores = self.compute_scores(hidden_states, enc_contexts)

      return scores


   ### Translation ########################

   def translate(self, src_wordIDs, src_len, max_tgt_len_factor, beam_size=0):
      ''' forward pass of the network during translation '''

      self.train(False)

      if beam_size > 0:
         return self.beam_translate(src_wordIDs, src_len, max_tgt_len_factor, beam_size)

      # run the encoder and initialize the decoder states
      enc_states, enc_context = self.init_decoder(src_wordIDs, src_len)

      tgt_wordIDs = []
      prev_wordIDs = torch.zeros(len(src_wordIDs)).to(device).long()
      tgt_logprobs = torch.zeros(len(src_wordIDs)).to(device)
      nonfinal     = torch.ones(len(src_wordIDs)).to(device)
      
      # Limit the maximal target sentence length
      for i in range(int(src_len[0] * max_tgt_len_factor + 6)):

         # run the decoder RNN for a single step
         hidden_state, enc_context = self.decoder_step(
            self.tgt_embeddings(prev_wordIDs), enc_states, enc_context, src_len)
         scores = self.compute_scores(hidden_state, enc_context)

         # extract the most likely target word for each sentence
         best_logprobs, best_wordIDs = F.log_softmax(scores, dim=-1).max(dim=-1)
         
         tgt_wordIDs.append(best_wordIDs)
         prev_wordIDs = best_wordIDs

         # sum up log probabilities until the end symbol with index 0 is encountered
         tgt_logprobs += best_logprobs * nonfinal
         nonfinal *= (best_wordIDs != 0).float()
         
         # stop if all output symbols are boundary/padding symbols
         if (best_wordIDs == 0).all():
            break

      tgt_wordIDs = torch.stack(tgt_wordIDs).t().tolist()
      return tgt_wordIDs, tgt_logprobs


   ### Translation with beam decoding ########################

   def build_beam(self, logprobs, beam_size, dec_rnn_states):

      # get the threshold which needs to be exceeded by all hypotheses in the new beam
      top_logprobs, _ = logprobs.view(-1).topk(beam_size+1)
      threshold = top_logprobs[-1]

      # extract the most likely extensions for each hypothesis
      top_logprobs, top_wordIDs = logprobs.topk(beam_size, dim=-1)

      # extract the most likely extended hypotheses overall
      new_wordIDs = []
      new_logprobs = []
      prev_pos = []
      for i in range(top_logprobs.size(0)):
         for k in range(top_logprobs.size(1)):
            if (top_logprobs[i,k] <= threshold).all(): # without all() it doesn't work
               break # ignore the rest
            prev_pos.append(i)
            new_wordIDs.append(top_wordIDs[i,k])
            new_logprobs.append(top_logprobs[i,k])
      new_wordIDs = torch.stack(new_wordIDs)
      new_logprobs = torch.stack(new_logprobs)
      new_dec_states = []
      for d in range(self.dec_depth):
         hidden_states = torch.stack([dec_rnn_states[d][0][i] for i in prev_pos])
         cell_states   = torch.stack([dec_rnn_states[d][1][i] for i in prev_pos])
         new_dec_states.append((hidden_states, cell_states))

      return new_wordIDs, new_logprobs, new_dec_states, prev_pos

   
   def beam_translate(self, src_wordIDs, src_len, max_tgt_len_factor, beam_size):
      ''' processes a single sentence with beam search '''
      
      enc_states, enc_context = self.init_decoder(src_wordIDs, src_len)
      
      tgt_wordIDs = []
      prev_pos = []
      prev_wordIDs = torch.zeros(1).to(device).long()
      prev_logprobs = torch.zeros(1).to(device)
      
      # Limit the maximal target sentences length
      for i in range(src_len[0] * max_tgt_len_factor + 6):

         # compute scores for the next target word candidates
         tgt_word_embs = self.tgt_embeddings(prev_wordIDs)
         hidden_state, enc_context = self.decoder_step(
            tgt_word_embs, enc_states, enc_context, src_len)
         scores = self.compute_scores(hidden_state, enc_context)

         # add the current logprob to the logprob of the previous hypothesis
         logprobs = prev_logprobs.unsqueeze(1) + F.log_softmax(scores, dim=-1)

         # extract the best hypotheses
         best_wordIDs, prev_logprobs, dec_rnn_states, prev \
            = self.build_beam(logprobs, beam_size, dec_rnn_states)

         # store information for computing the best translation at the end
         tgt_wordIDs.append(best_wordIDs.cpu().data.numpy().tolist())
         prev_pos.append(prev)
         prev_wordIDs = best_wordIDs
         
         # stop if all output symbols are boundary/padding symbols
         if (best_wordIDs == 0).all():
            break

      # extract the best translation
      # get the position of the most probable hypothesis
      logprob, pos = prev_logprobs.max(-1)
      pos = int(pos)

      # extract the best translation backward using prev_pos
      wordIDs = []
      for i in range(len(prev_pos)-1,0,-1):
         pos = prev_pos[i][pos]
         wordIDs.append(tgt_wordIDs[i-1][pos])
      wordIDs.reverse()

      return [wordIDs], [logprob]
