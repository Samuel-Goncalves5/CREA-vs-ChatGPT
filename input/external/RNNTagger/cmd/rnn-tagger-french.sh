#!/bin/sh

# Set these paths appropriately

BIN=preprocessing/RNNTagger/bin
SCRIPTS=preprocessing/RNNTagger/scripts
LIB=preprocessing/RNNTagger/lib
PyRNN=preprocessing/RNNTagger/PyRNN
PyNMT=preprocessing/RNNTagger/PyNMT
TMP=/tmp/rnn-tagger$$
LANGUAGE=french

TOKENIZER="perl ${SCRIPTS}/tokenize.pl"
ABBR_LIST=${LIB}/Tokenizer/${LANGUAGE}-abbreviations
TAGGER="python3 $PyRNN/rnn-annotate.py"
RNNPAR=${LIB}/PyRNN/${LANGUAGE}
REFORMAT="perl ${SCRIPTS}/reformat.pl"
LEMMATIZER="python3 $PyNMT/nmt-translate.py"
NMTPAR=${LIB}/PyNMT/${LANGUAGE}

$TOKENIZER -f -a $ABBR_LIST $1 > $TMP.tok

$TAGGER $RNNPAR $TMP.tok > $TMP.tagged

$REFORMAT $TMP.tagged > $TMP.reformatted

$LEMMATIZER --print_source $NMTPAR $TMP.reformatted > $TMP.lemmas

$SCRIPTS/lemma-lookup.pl $TMP.lemmas $TMP.tagged 

rm $TMP.tok  $TMP.tagged  $TMP.reformatted $TMP.lemmas
