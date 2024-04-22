#!/usr/bin/python3

import sys
import argparse
import random
import operator
import numpy
import pickle

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from Data import Data, rstrip_zeros
import NMT


def build_optimizer(optim, model, learning_rate):
   optimizer = {
      'SGD':      torch.optim.SGD,
      'RMSprop':  torch.optim.RMSprop,
      'Adagrad':  torch.optim.Adagrad,
      'Adadelta': torch.optim.Adadelta,
      'Adam':     torch.optim.Adam,
      'AdamW':    torch.optim.AdamW
   }
   return optimizer[optim](model.parameters(), lr=learning_rate)


def process_batch(data, batch, model, optimizer=None):

   (src_wordIDs, src_len), (tgt_wordIDs, tgt_len) = batch

   training_mode = optimizer is not None
   model.train(training_mode)
      
   # add boundary symbols to target sequence
   tgt_wordIDs = torch.LongTensor(tgt_wordIDs)
   boundaries = torch.zeros(tgt_wordIDs.size(0), 1).long()
   tgt_wordIDs = torch.cat((boundaries, tgt_wordIDs, boundaries), dim=-1).to(NMT.device)

   scores = model(src_wordIDs, src_len, tgt_wordIDs[:,:-1])

   # flatten the first dimension of the tensors and compute the loss
   scores = scores.view(-1, scores.size(2))
   wordIDs = tgt_wordIDs[:,1:].contiguous().view(-1)
   loss = F.cross_entropy(scores, wordIDs)

   # compute gradient and perform weight updates
   if training_mode:
      optimizer.zero_grad()
      loss.backward()
      if args.grad_threshold > 0.0:
         torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_threshold)
      optimizer.step()

   return float(loss)


def training(args):

   random.seed(args.random_seed)

   data = Data(args.path_train_src, args.path_train_tgt,
               args.path_dev_src, args.path_dev_tgt,
               args.max_src_vocab_size, args.max_tgt_vocab_size,
               args.max_len, args.batch_size, args.ood_rate)
   data.save_parameters( args.path_param+".io" )

   hyper_params = data.src_vocab_size, data.tgt_vocab_size, args.word_emb_size, \
      args.enc_rnn_size, args.dec_rnn_size, args.enc_depth, args.dec_depth, \
      args.dropout_rate, args.emb_dropout_rate, args.tie_embeddings
   
   with open(args.path_param+".hyper", "wb") as file:
      pickle.dump(hyper_params, file)

   ### creation of the network
   model = NMT.NMTDecoder(*hyper_params)
   
   if args.gpu >= 0:
      model = model.to(NMT.device)

   optimizer = build_optimizer(args.optimizer, model, args.learning_rate)
   scheduler = StepLR(optimizer, step_size=1, gamma=args.learning_rate_decay)

   ### training loop ##################################################

   best_result = 0.0
   for epoch in range(1, args.num_epochs+1):
      for batch_no, batch in enumerate(data.training_batches(), 1):
         loss = process_batch(data, batch, model, optimizer)

         if batch_no % 100 == 1:
            ### translate a few training sentences #######################
            (src_wordIDs, src_len), (tgt_wordIDs, tgt_len) = batch
            
            predIDs, logprobs = model.translate(src_wordIDs[:1], src_len[:1],
                                                data.max_tgt_len_factor)
            # print the source sentence, translation and reference translation
            words = data.source_words(rstrip_zeros(src_wordIDs[0]))
            print("src:", *words, file=sys.stderr)
            words = data.target_words(rstrip_zeros(tgt_wordIDs[0]))
            print("ref:", *words, file=sys.stderr)
            words = data.target_words(rstrip_zeros(predIDs[0]))
            print("tgt:", *words, file=sys.stderr)
            print('', file=sys.stderr, flush=True)

      scheduler.step()
      
      print("Evaluation on dev data", file=sys.stderr)
      with torch.no_grad():
         correct = 0; all = 0
         for batch in data.development_batches():
            (src_wordIDs, src_len), (tgt_wordIDs, tgt_len) = batch
            predIDs, logprobs = model.translate(src_wordIDs, src_len,
                                                data.max_tgt_len_factor)
            for i in range(len(tgt_wordIDs)):
               all += 1
               pred_words = ' '.join(data.target_words(predIDs[i]))
               correct_words =  ' '.join(data.target_words(tgt_wordIDs[i]))
               if pred_words == correct_words:
                  correct += 1
      acc = correct*100/all
      print("Accuracy: %.2f"%(acc), flush=True)

      if best_result < acc:
         best_result = acc
         print("storing parameters", file=sys.stderr)
         torch.save(model.state_dict(), args.path_param+".nmt" )
      


###########################################################################
# main function
###########################################################################

if __name__ == "__main__":

   parser = argparse.ArgumentParser(description='Training program of the RNN-Tagger.')

   parser.add_argument('path_train_src', type=str,
                       help='file containing the source training data')
   parser.add_argument('path_train_tgt', type=str,
                       help='file containing the target training data')
   parser.add_argument('path_dev_src', type=str,
                       help='file containing the source development data')
   parser.add_argument('path_dev_tgt', type=str,
                       help='file containing the target development data')
   parser.add_argument('path_param', type=str,
                       help='base name of the files in which the parameters are stored')
   parser.add_argument('--word_emb_size', type=int, default=100,
                       help='size of the word embedding vectors')
   parser.add_argument('--enc_rnn_size', type=int, default=400,
                       help='size of the hidden state of the RNN encoder')
   parser.add_argument('--dec_rnn_size', type=int, default=400,
                       help='size of the hidden state of the RNN decoder')
   parser.add_argument('--enc_depth', type=int, default=2,
                       help='number of encoder BiLSTM layers')
   parser.add_argument('--dec_depth', type=int, default=2,
                       help='number of decoder LSTM layers')
   parser.add_argument('--dropout_rate', type=float, default=0.3,
                       help='dropout rate')
   parser.add_argument('--emb_dropout_rate', type=float, default=0.0,
                       help='dropout rate for embeddings')
   parser.add_argument('--optimizer', type=str, default='Adam', 
                       choices=['SGD', 'Adagrad', 'Adadelta', 'RMSprop', 'Adam', 'AdamW'],
                       help='seletion of the optimizer')
   parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='the learning rate')
   parser.add_argument('--learning_rate_decay', type=float, default=0.95,
                       help='the learning rate multiplier applied after each evaluation')
   parser.add_argument('--grad_threshold', type=float, default=1.0,
                       help='gradient clipping threshold')
   parser.add_argument('--max_src_vocab_size', type=int, default=0,
                       help='maximal number of words in the source vocabulary')
   parser.add_argument('--max_tgt_vocab_size', type=int, default=0,
                       help='maximal number of words in the target vocabulary')
   parser.add_argument('--batch_size', type=int, default=1000,
                       help='size of each batch')
   parser.add_argument('--num_epochs', type=int, default=50,
                       help='number of training epochs')
   parser.add_argument('--random_seed', type=int, default=32,
                       help='seed for the random number generators')
   parser.add_argument('--max_len', type=int, default=50,
                       help='maximal sentence length')
   parser.add_argument('--ood_rate', type=float, default=0.0,
                       help='probability of sampling synthesized OOD data')
   parser.add_argument('--gpu', type=int, default=0,
                       help='selection of the GPU. The default is: 0 (CPU=-1)')
   parser.add_argument("--tie_embeddings", action="store_true", default=False,
                       help="Decoder input and output embeddings are tied")
   args = parser.parse_args()

   if args.gpu >= 0:
      if not torch.cuda.is_available():
         print('No gpu available. Using cpu instead.', file=sys.stderr)
         args.gpu = -1
      else:
         if args.gpu >= torch.cuda.device_count():
            print('gpu '+str(args.gpu)+' not available. Using gpu 0 instead.', file=sys.stderr)
            args.gpu = 0
         torch.cuda.set_device(args.gpu)
   NMT.device = torch.device('cuda' if args.gpu >= 0 else 'cpu')

   if args.dec_depth < 2:
      print("Error: decoder must have at least 2 layers!", file=sys.stderr)
      sys.exit(1)

   training(args)
