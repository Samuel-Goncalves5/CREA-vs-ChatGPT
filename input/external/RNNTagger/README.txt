
General remarks:

RNNTagger is a part-of-speech tagger and lemmatizer currently
supporting 51 languages.  RNNTagger is written in Python3 and uses
the PyTorch machine learning library.


Installation:

Required software: Linux, Python3, Perl, PyTorch, (CUDA)

In order to use RNNTagger, you need to install Python3, Perl and
PyTorch.  It is highly recommended to use a computer with a recent
Nvidia graphics card and to install the CUDA library. Without graphics
card support, the software will be very slow.

Using the tagger with other operating systems (e.g. Windows) should be
possible if the shell scripts (files with extension ".sh") are
replaced by e.g. batch files on Windows.


Usage:

After unpacking the software archive, you can use the software as follows:

> cd RNNTagger
> echo "cetero censeo carthaginem esse delendam." > file
> cmd/rnn-tagger-latin.sh file
cetero	11B---G---1	ceterus
censeo	3-KA1--4---	censeo
carthaginem	11C---D2---	carthago
esse	3-NH1------	sum
delendam	2-KO-1D2---	deleo
.	SENT	.

The taggers for other languages are called in the same way:
> cmd/rnn-tagger-<language>.sh file

The processing usually consists of three steps:

The included simple tokenizer first splits the input text into
"tokens" (i.e. words, punctuation, parentheses etc.) Each token is
written on a separate line and each sentence is followed by an empty
line.

The part-of-speech (POS) tagger reads the token sequence and assigns a
POS tag to each token.

The lemmatizer extracts all word-POS tag combinations from the tagged
token sequence, computes the lemma for each pair, and then looks up
the precomputed lemma for each token-tag pair of the POS tagged
token sequence.

The Korean tagger uses a special XML-based output format which shows
the eojeols and their components.

rnn-tagger-old-english.sh and rnn-tagger-swiss-german.sh do not generate lemmas.

rnn-tagger-syriac.sh expects tokenized input with one sentence per line.

Information on the part-of-speech tagsets is available via the URLs
listed in the file "Tagset-Information".

The tagger for old-french and middle-french additionally uses a
separate lexicon for lemmatization. If the lemma for a given word/tag
combination is not found in the lexicon, then these tagger print the
lemma predicted by the NMT lemmatizer in parentheses. If the predicted
lemma is not known from the lexicon, it is printed in double parentheses.
