#!/usr/bin/python3

import sys
import argparse
import pickle
import torch

from Data import Data
from RNNTagger import RNNTagger
from CRFTagger import CRFTagger


###########################################################################
# main function
###########################################################################

if __name__ == "__main__":

   parser = argparse.ArgumentParser(description='Annotation program of the RNN-Tagger.')

   parser.add_argument('path_param', type=str,
                       help='name of parameter file')
   parser.add_argument('path_data', type=str,
                       help='name of the file with input data')
   parser.add_argument('--crf_beam_size', type=int, default=10,
                       help='size of the CRF beam (if the system contains a CRF layer)')
   parser.add_argument('--gpu', type=int, default=0,
                       help='selection of the GPU. The default is: 0 (CPU=-1)')
   parser.add_argument("--min_prob", type=float, default=-1.0,
                       help="print all tags whose probability exceeds the probability of the best tag times this threshold")
   parser.add_argument("--print_probs", action="store_true", default=False,
                       help="print the tag probabilities")

   args = parser.parse_args()

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
   device = torch.device('cuda' if args.gpu >= 0 else 'cpu')

   # load parameters
   data  = Data(args.path_param+'.io')   # read the symbol mapping tables

   with open(args.path_param+'.hyper', 'rb') as file:
      hyper_params = pickle.load(file)
   model = CRFTagger(*hyper_params) if len(hyper_params)==10 \
           else RNNTagger(*hyper_params)
   model.load_state_dict(torch.load(args.path_param+'.rnn', 
                         map_location=torch.device('cpu')))

   model = model.to(device)

   if type(model) is CRFTagger:
      for optvar, option in zip((args.min_prob, args.print_probs),
                                ("min_prob","print_probs")):
         if optvar:
            print(f"Warning: Option --{option} is ignored because the model has a CRF output layer", file=sys.stderr)
   
   model.eval()
   with torch.no_grad():
      for i, words in enumerate(data.sentences(args.path_data)):
         print(i, end='\r', file=sys.stderr, flush=True)
   
         # map words to numbers and create Torch variables
         fwd_charIDs, bwd_charIDs = data.words2charIDvec(words)
         fwd_charIDs = torch.LongTensor(fwd_charIDs).to(device)
         bwd_charIDs = torch.LongTensor(bwd_charIDs).to(device)
         
         # run the model
         if type(model) is RNNTagger:
            tagscores = model(fwd_charIDs, bwd_charIDs)
            if args.min_prob == -1.0:
               # only print the word and tag with the highest score
               tagIDs = tagscores.argmax(-1)
               tags = data.IDs2tags(tagIDs.to("cpu"))
               if not args.print_probs:
                  for word, tag in zip(words, tags):
                     print(word, tag, sep="\t")
               else:
                  # print probabilities as well
                  tagprobs = torch.nn.functional.softmax(tagscores, dim=-1)
                  # get the probabilities of the highest-scoring tags
                  probs = tagprobs[range(len(tagIDs)), tagIDs].to("cpu")
                  # print the result
                  for word, tag, prob in zip(words, tags, probs):
                     print(word, tag, round(float(prob), 4), sep="\t")
            else:
               # print the best tags for each word
               tagprobs = torch.nn.functional.softmax(tagscores, dim=-1)
               # get the most probable tag and its probability
               best_probs, _ = tagprobs.max(-1)
               # get all tags with a probability above best_prob * min_prob
               thresholds = best_probs * args.min_prob
               greaterflags = (tagprobs > thresholds.unsqueeze(1))
               for word, flags, probs in zip(words, greaterflags, tagprobs):
                  # get the IDs of the best tags
                  IDs = flags.nonzero()
                  # get the best tags and their probabilities
                  best_probs = probs[IDs].to("cpu")
                  best_tags = data.IDs2tags(IDs.to("cpu"))
                  # sort the tags by decreasing probability
                  sorted_list = sorted(zip(best_tags, best_probs), key=lambda x:-x[1])
                  best_tags, best_probs = zip(*sorted_list)
                  # generate the output
                  if args.print_probs:
                     # append the probabilities to the tags
                     best_tags = [f"{t} {float(p):.4f}" for t, p in zip(best_tags, best_probs)]
                  print(word, ' '.join(best_tags), sep="\t")
         elif type(model) is CRFTagger:
            tagIDs = model(fwd_charIDs, bwd_charIDs)
            tags = data.IDs2tags(tagIDs)
            for word, tag in zip(words, tags):
               print(word, tag, sep='\t')
         else:
            sys.exit('Error')
   
         print('')
