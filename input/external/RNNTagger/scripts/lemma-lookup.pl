#!/usr/bin/env perl

use warnings;
use strict;
use utf8;
use open ':utf8';       # all open() use UTF-8
use open ':std';        # standard filehandles too

my $lemma_file = shift or die;

my $with_lexicon = 0;
my %lex_lemma;
my %lemmaset;
if ($lemma_file eq '-l') {
    # read an optional lexicon file
    $with_lexicon = 1;
    my $lexicon_file = shift or die;
    $lemma_file = shift or die;

    # read lemma information from a dictionary
    open(FILE, $lexicon_file) or die "Error: unable to open \"$lexicon_file\"";
    while (<FILE>) {
	chomp;
	my @F = split(/\t/);
	my $word = $F[0];
	for(my $i=1; $i<=$#F; $i++) {
	    die unless $F[$i] =~ /^(.*?)\s(.*)$/;
	    my $tag = $1;
	    my $lemma = $2;
	    $lex_lemma{"$word\t$tag"} = $lemma;
	    $lemmaset{$lemma} = $lemma;
	}
    }
}

# read the lemmas predicted by the lemmatizer
open(FILE, $lemma_file) or die "Error: unable to open \"$lemma_file\"";
my %lemma;
while (<FILE>) {
    die unless /^(.*?) ## (.*)$/;
    my $word = $1;
    my $tag = $2;
    
    $word =~ s/ //g;
    $word =~ s/<>/ /g;
    $tag =~ s/ //g;

    my $lemma = <FILE>;
    chomp($lemma);
    $lemma =~ s/ //g;
    $lemma =~ s/<>/ /g;

    if ($lemma =~ /<(unk|ood)>/) {
	if (length($word) == 1) {
	    $lemma = $word;
	}
	else {
	    $lemma = '<unknown>';
	}
    }
    
    $lemma{"$word\t$tag"} = $lemma;

    die unless <FILE> eq "\n"; # empty line must follow
}

# lemmatize the input file
while (<>) {
    chomp;
    if ($_ eq "") {
	print("\n");
    }
    else {
	chomp;
	if ($with_lexicon) {
	    if (exists $lex_lemma{$_}) {
		# lemma obtained from the lexicon
		print("$_\t$lex_lemma{$_}\n");
	    }
	    elsif (exists $lemmaset{$lemma{$_}}) {
		# The predicted lemma is known from the lexicon
		print("$_\t($lemma{$_})\n");
	    }
	    else {
		# The predicted lemma is unknown
		print("$_\t(($lemma{$_}))\n");
	    }
	}
	else {
	    print("$_\t$lemma{$_}\n");
	}
    }
    
}
