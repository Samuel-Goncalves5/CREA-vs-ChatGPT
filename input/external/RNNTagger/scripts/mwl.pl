#!/usr/bin/perl

use warnings;
use strict;
use utf8;
use open ':utf8';       # all open() use UTF-8
use open ':std';        # standard filehandles too

my $word = '';
while (<>) {
    chomp;
    if ($_ eq '') {
	print("\n");
    }
    else {
	my($w, $t) = split(/\t/);
	$word = $word.$w;
	$t = '$_' if $w =~ /^[().,;:()\/"?!'-]$/;
	if ($t ne 'MWLpart' && $t ne 'PART') {
	    print("$word\t$t\n");
	    $word = '';
	}
    }
}
