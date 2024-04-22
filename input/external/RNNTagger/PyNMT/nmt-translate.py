#!/usr/bin/python3

import sys
import argparse
import random
import operator
import pickle
import numpy

import torch
import torch.nn.functional as F

from Data import Data, rstrip_zeros
import NMT


def translate(model, data, inputfile, max_tgt_len_factor, args):
   # translate the contents of an input file
   batch_no = 0
   model.eval()
   with torch.no_grad():
      for src_words, sent_idx, (src_wordIDs, src_len) in data.test_batches(inputfile):
         if not args.quiet:
            print(batch_no*args.batch_size+len(src_words), end="\r", file=sys.stderr)
         batch_no += 1
   
         tgt_wordIDs, logprobs = model.translate(src_wordIDs, src_len, 
                                                 max_tgt_len_factor, args.beam_size)
         # undo the sorting of sentences by length
         tgt_wordIDs = [tgt_wordIDs[i] for i in sent_idx]
         
         for swords, twordIDs, lp in zip(src_words, tgt_wordIDs, logprobs):
            if args.print_source:
               print(*swords)
            twords = data.target_words(rstrip_zeros(twordIDs))
            print(*twords)
            if args.print_probs:
               print(float(lp.exp()))
            if args.print_source:
               print('')
         


###########################################################################
# main function
###########################################################################

if __name__ == "__main__":

   parser = argparse.ArgumentParser(description='Decoder program of the RNN-Tagger.')

   parser.add_argument('path_param', type=str,
                       help='base name of the files in which the parameters are stored')
   parser.add_argument('path_data', type=str,
                       help='file containing the input data')
   parser.add_argument('--batch_size', type=int, default=32,
                       help='size of each batch')
   parser.add_argument('--beam_size', type=int, default=0,
                       help='size of the search beam')
   parser.add_argument('--gpu', type=int, default=0,
                       help='selection of the GPU. The default is: 0 (CPU=-1)')
   parser.add_argument("--quiet", action="store_true", default=False,
                       help="print status messages")
   parser.add_argument("--print_probs", action="store_true", default=False,
                       help="print the translation probabilities")
   parser.add_argument("--print_source", action="store_true", default=False,
                       help="print source sentences")
   args = parser.parse_args()

   if args.beam_size < 0 or args.beam_size > 1000:
      sys.exit("beam size is out of range: "+str(args.beam_size))
   if args.beam_size > 0:
      args.batch_size = 1
   elif args.batch_size < 1 or args.batch_size > 1000:
      sys.exit("batch size is out of range: "+str(args.batch_size))

   # load parameters
   data  = Data(args.path_param+".io", args.batch_size) # read the symbol mapping tables

   ### creation of the network
   with open(args.path_param+".hyper", "rb") as file:
      hyper_params = pickle.load(file)
   model = NMT.NMTDecoder(*hyper_params)
   model.load_state_dict(torch.load(args.path_param+".nmt", 
                         map_location=torch.device('cpu')))   # read the model
   
   # Select the processing device
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
   model = model.to(NMT.device)

   with open(args.path_data) as inputfile:
      translate(model, data, inputfile, data.max_tgt_len_factor, args)
