#!/usr/bin/env perl

use warnings;
use strict;
#use utf8::all;
use utf8;
use open ':utf8';       # all open() use UTF-8
use open ':std';        # standard filehandles too

my %lines;
while (<>) {
    chomp;
    next if $_ eq '';
    my($w,$t) = split(/\t/);

    # split the letter sequence for lemmatization with PyNMT
    $w =~ s/(.)/$1 /g;
    $w =~ s/   / <> /g;
    $t =~ s/(.)/$1 /g;
    
    $lines{"$w ## $t\n"} = 1;
}

# print unique lines
for my $line (sort keys %lines) {
    print $line;
}
