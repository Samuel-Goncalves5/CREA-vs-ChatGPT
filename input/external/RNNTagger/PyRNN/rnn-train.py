#!/usr/bin/python3

import sys
import argparse
import random
import pickle
import numpy

import torch
from torch.optim.lr_scheduler import StepLR

from Data import Data
from RNNTagger import RNNTagger
from CRFTagger import CRFTagger

EmbeddingBatchSize = 20

def build_optimizer(optim, model, learning_rate):
   optimizer = {
      'sgd':      torch.optim.SGD,
      'rmsprop':  torch.optim.RMSprop,
      'adagrad':  torch.optim.Adagrad,
      'adadelta': torch.optim.Adadelta,
      'adam':     torch.optim.Adam
   }
   return optimizer[optim](model.parameters(), lr=learning_rate)


def run_tagger(sentences, data, model, optimizer=None, embedding_training=False):

   training_mode = True if optimizer else False
   model.train(training_mode)

   loss_function = torch.nn.CrossEntropyLoss(reduction='sum',
                                             ignore_index=data.IGNORE_INDEX)
   cosine_similarity = torch.nn.CosineSimilarity()

   ### iterate over the data 
   num_tags, num_correct, loss_sum, emb_loss_sum = 0, 0, 0.0, 0.0
   for iteration, (words, tags) in enumerate(sentences):

      # map words and tags to numbers and create Torch variables
      fwd_charIDs, bwd_charIDs = data.words2charIDvec(words)
      fwd_charIDs = torch.LongTensor(fwd_charIDs).to(device)
      bwd_charIDs = torch.LongTensor(bwd_charIDs).to(device)
   
      tagIDs = torch.LongTensor(data.tags2IDs(tags)).to(device)

      # run the model
      if type(model) is RNNTagger:
         tagscores = model(fwd_charIDs, bwd_charIDs)
         loss = loss_function(tagscores, tagIDs)

         # compute the tag predictions
         _, predicted_tagIDs = tagscores.max(dim=-1)
         
      elif type(model) is CRFTagger:
         predicted_tagIDs, loss = \
             model(fwd_charIDs, bwd_charIDs, tagIDs)
      else:
         sys.exit("Error in function run_tagger")

      num_tags += len(tagIDs)
      num_correct += sum([1 for t, t2 in zip(tagIDs, predicted_tagIDs) if t == t2])
      loss_sum += float(loss)

      if training_mode:
         # compute gradient and perform weight updates
         optimizer.zero_grad()
         loss.backward()
         if args.grad_threshold > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_threshold)
         optimizer.step()

         if iteration%1000 == 0:
            if args.verbose:
               predicted_tags = data.IDs2tags(predicted_tagIDs)
               print("training items:", iteration, file=sys.stderr)
               for word, tag, ptag in zip(words, tags, predicted_tags):
                  if tag == ptag:
                     print(word, tag, sep="\t", file=sys.stderr)
                  else:
                     print(word, tag, ptag, sep="\t", file=sys.stderr)
               print(file=sys.stderr)
            else:
               print(iteration, end='\r', file=sys.stderr)
            sys.stderr.flush()

         # Auxiliary training of the character-based LSTMs using  word embeddings
         if data.word_embeddings:
            # sampling of word embeddings 
            batch = random.sample(data.word_embeddings, EmbeddingBatchSize)
            words, embeddings = zip(*batch)
            fwd_charIDs, bwd_charIDs = data.words2charIDvec(words)
            target_embeddings = numpy.array(embeddings, dtype=numpy.float32)
            target_embeddings = torch.tensor(target_embeddings).to(device)
            word_embeddings = model(torch.LongTensor(fwd_charIDs).to(device),
                                    torch.LongTensor(bwd_charIDs).to(device),
                                    word_embedding_training=True)
            emb_loss = (1.0 - cosine_similarity(word_embeddings, target_embeddings)).sum()
            emb_loss_sum +=  float(emb_loss)
            emb_loss *= args.emb_train_weight
            # compute gradient and perform weight updates
            optimizer.zero_grad()
            emb_loss.backward()
            if args.grad_threshold > 0.0:
               torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_threshold)
            optimizer.step()

   accuracy = num_correct * 100.0 / num_tags
   print("losses:", loss_sum/(iteration+1), emb_loss_sum/(iteration+1), sep="\t", file=sys.stderr)

   return loss_sum, accuracy
      


###########################################################################
# tagger training
###########################################################################

def training(args):

   random.seed(args.random_seed)

   data = Data(args.path_train, args.path_dev, args.word_trunc_len,
               args.min_char_freq, args.max_sent_len, args.word_embeddings,
               args.ignore_tag)
   data.save_parameters(args.path_param+".io")

   ### creation of the network
   hyper_params = data.num_char_types, data.num_tag_types, \
                  args.char_embedding_size, args.char_recurrent_size, \
                  args.word_recurrent_size, args.word_rnn_depth, \
                  args.dropout_rate, data.word_emb_size, args.crf_beam_size
   if args.crf_epochs > 0:  # use a CRF?
      crf_model = CRFTagger(*hyper_params).to(device)
      model = crf_model.base_tagger  # initial training of the base tagger only
   else:
      model = RNNTagger(*hyper_params[:-1]).to(device)

   optimizer = build_optimizer(args.optimizer, model, args.learning_rate)
   scheduler = StepLR(optimizer, step_size=1, gamma=args.learning_rate_decay)
   args.emb_train_weight *= args.learning_rate_decay

   ### training
   max_accuracy = -1.
   for epoch in range(args.epochs + args.crf_epochs):

      if epoch == args.epochs:
         # start CRF training
         model = crf_model
         current_lr = optimizer.param_groups[0]['lr']
         optimizer = build_optimizer(args.optimizer, model, current_lr)
         scheduler = StepLR(optimizer, step_size=1, gamma=args.learning_rate_decay)
         max_accuracy = -1.0
         args.path_param += "-crf"

      random.shuffle(data.train_sentences)  # data is shuffled after each epoch
      loss, accuracy = run_tagger(data.train_sentences, data, model,
                                  optimizer, data.word_emb_size>0)
      print("Epoch:", epoch+1, file=sys.stderr)
      print("TrainLoss: %.0f" % loss, "TrainAccuracy: %.2f" % accuracy, file=sys.stderr)
      sys.stderr.flush();

      if epoch >= args.burn_in_epochs:
         scheduler.step()
         
      loss, accuracy = run_tagger(data.dev_sentences, data, model)
      print(epoch+1, "DevLoss: %.0f" % loss, "DevAccuracy: %.2f" % accuracy)
      sys.stdout.flush()

      ### keep the model which performs best on dev data
      if max_accuracy < accuracy:
         max_accuracy = accuracy

         with open(args.path_param+".hyper", "wb") as file:
            if epoch >= args.epochs:
               # model is a CRFTagger
               pickle.dump(hyper_params, file)
            else:
               # model is an RNNTagger
               pickle.dump(hyper_params[:-1], file)
         
         model = model.cpu()
         torch.save(model.state_dict(), args.path_param+".rnn")
         model = model.to(device)


###########################################################################
# main function
###########################################################################

if __name__ == "__main__":

   parser = argparse.ArgumentParser(description='Training program of the RNN-Tagger.')

   parser.add_argument('path_train', type=str,
                       help='file containing the training data')
   parser.add_argument('path_dev', type=str,
                       help='file containing the development data')
   parser.add_argument('path_param', type=str,
                       help='file in which the network parameters are stored')
   parser.add_argument('--char_embedding_size', type=int, default=100,
                       help='size of the character embedding vectors')
   parser.add_argument('--char_recurrent_size', type=int, default=400,
                       help='size of the hidden states of the RNN over characters')
   parser.add_argument('--word_recurrent_size', type=int, default=400,
                       help='size of the hidden states of the RNN over words')
   parser.add_argument('--word_rnn_depth', type=int, default=1,
                       help='number of word-based BiLSTM layers')
   parser.add_argument('--crf_beam_size', type=int, default=0,
                       help='CRF beam search size')
   parser.add_argument('--dropout_rate', type=float, default=0.5,
                       help='dropout rate')
   parser.add_argument('--optimizer', type=str, default='sgd', 
                       choices=['sgd', 'adagrad', 'adadelta', 'rmsprop', 'adam'],
                       help='seletion of the optimizer')
   parser.add_argument('--learning_rate', type=float, default=1.0,
                       help='initial learning rate')
   parser.add_argument('--learning_rate_decay', type=float, default=0.95,
                       help='learning rate decay after each epoch')
   parser.add_argument('--burn_in_epochs', type=int, default=5,
                       help='number of initial epochs without a learning rate change')
   parser.add_argument('--crf_epochs', type=int, default=0,
                       help='number of final CRF training epochs')
   parser.add_argument('--grad_threshold', type=float, default=1.0,
                       help='gradient clipping threshold')
   parser.add_argument('--word_trunc_len', type=int, default=10,
                       help='words longer than this are truncated')
   parser.add_argument('--min_char_freq', type=int, default=2,
                       help='characters less frequent than this are replaced by <unk>')
   parser.add_argument('--max_sent_len', type=int, default=256,
                       help='Sentences longer than this are ignored.')
   parser.add_argument('--epochs', type=int, default=50,
                       help='number of training epochs')
   parser.add_argument('--word_embeddings', type=str, default=None,
                       help='pretrained word embeddings')
   parser.add_argument('--emb_train_weight', type=float, default=0.2,
                       help='weight of the embedding training loss')
   parser.add_argument('--ignore_tag', type=str, default=None,
                       help='No loss is computed for words with this tag')
   parser.add_argument('--random_seed', type=int, default=32,
                       help='seed for the random number generators')
   parser.add_argument('--gpu', type=int, default=0,
                       help='selection of the GPU. The default is: 0 (CPU=-1)')
   parser.add_argument("--verbose", action="store_true", default=False,
                       help="increase output verbosity")
   
   args = parser.parse_args()

   if args.gpu >= 0:
      if not torch.cuda.is_available():
         sys.exit("Sorry, no gpu available. Please drop the -gpu option.")
      if args.gpu >= torch.cuda.device_count():
         sys.exit("Sorry, the given gpu index was too large. Please choose a valid gpu index.")
      torch.cuda.set_device(args.gpu)
      torch.cuda.manual_seed(args.random_seed)
      device = torch.device('cuda')
   else:
      torch.manual_seed(args.random_seed)
      device = torch.device('cpu')

   training(args)
