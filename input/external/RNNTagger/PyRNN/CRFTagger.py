
import sys
import torch
from torch import nn
from torch.nn import functional as F

from RNNTagger import RNNTagger


### auxiliary functions ############################################

def logsumexp(x, dim):
    """ sums up log-scale values """
    offset, _ = torch.max(x, dim=dim)
    offset_broadcasted = offset.unsqueeze(dim) 
    safe_log_sum_exp = torch.log(torch.exp(x-offset_broadcasted).sum(dim=dim))
    return safe_log_sum_exp + offset

def lookup(T, indices):
    """ look up probabilities of tags in a vector, matrix, or 3D tensor """
    if T.dim() == 3:
        return T.gather(2, indices.unsqueeze(2)).squeeze(2)
    elif T.dim() == 2:
        return T.gather(1, indices.unsqueeze(1)).squeeze(1)
    elif  T.dim() == 1:
        return T[indices]
    else:
        raise Exception('unexpected tensor size in function "lookup"')

    
### tagger class ###############################################

class CRFTagger(nn.Module):
    """ implements a CRF tagger """
    
    def __init__(self, num_chars, num_tags, char_emb_size,
                 char_rec_size, word_rec_size, word_rnn_depth, 
                 dropout_rate, word_emb_size, beam_size):

        super(CRFTagger, self).__init__()

        # simple LSTMTagger which computes tag scores
        self.base_tagger = RNNTagger(num_chars, num_tags, char_emb_size,
                                     char_rec_size, word_rec_size,
                                     word_rnn_depth, dropout_rate, word_emb_size)
        self.beam_size = beam_size if 0 < beam_size < num_tags else num_tags
        self.weights = nn.Parameter(torch.zeros(num_tags, num_tags))
        self.dropout = nn.Dropout(dropout_rate)

        
    def forward(self, fwd_charIDs, bwd_charIDs, tags=None):

        annotation_mode = (tags is None)

        scores = self.base_tagger(fwd_charIDs, bwd_charIDs)
        
        # extract the highest-scoring tags for each word and their scores
        best_scores, best_tags = scores.topk(self.beam_size, dim=-1)

        if self.training:  # not done during dev evaluation
            # check whether the goldstandard tags are among the best tags
            gs_contained = (best_tags == tags.unsqueeze(1)).sum(dim=-1)

            # replace the tag with the lowest score at each position
            # by the gs tag if the gs tag is not in the list
            last_column = gs_contained * best_tags[:,-1] + (1-gs_contained) * tags
            s = lookup(scores, last_column)
            best_tags   = torch.cat((best_tags[:,:-1], last_column.unsqueeze(1)), dim=1)
            best_scores = torch.cat((best_scores[:,:-1], s.unsqueeze(1)), dim=1)

        best_previous = []  # stores the backpointers of the Viterbi algorithm
        viterbi_scores = best_scores[0]
        if not annotation_mode:
            forward_scores = best_scores[0]
        for i in range(1,scores.size(0)):   # for all word positions except the first
            # lookup of the tag-pair weights
            w = self.weights[best_tags[i-1]][:,best_tags[i]]
            
            # Viterbi algorithm
            values = viterbi_scores.unsqueeze(1) + best_scores[i].unsqueeze(0) + w
            viterbi_scores, best_prev = torch.max(values, dim=0)
            best_previous.append(best_prev)
            
            # Forward algorithm
            if not annotation_mode:
                values = forward_scores.unsqueeze(1) + best_scores[i].unsqueeze(0) + w
                forward_scores = logsumexp(values, dim=0)

        # Viterbi algorithm
        _, index = torch.max(viterbi_scores, dim=0)
        best_indices = [index]
        for i in range(len(best_previous)-1, -1, -1):
            index = best_previous[i][index]
            best_indices.append(index)

        # reverse the indices and map them to tag IDs
        best_indices = torch.stack(best_indices[::-1])
        predicted_tags = lookup(best_tags, best_indices)

        if annotation_mode:
            return predicted_tags
        else:
            # loss computation
            basetagger_scores = lookup(scores, tags).sum()
            CRFweights = self.weights[tags[:-1], tags[1:]].sum() if tags.size(0)>1 else 0
            logZ = logsumexp(forward_scores, dim=0)  # log partition function
            logprob = basetagger_scores + CRFweights - logZ
            
            return predicted_tags, -logprob
        
