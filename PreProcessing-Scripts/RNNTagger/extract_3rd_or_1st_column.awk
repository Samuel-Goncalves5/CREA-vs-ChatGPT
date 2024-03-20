#! /bin/awk

# Copy the 3rd column (input) at 1st column (output), except if there are not
#  enough column (in that case, the 1st column is copied on output) or if 3rd
#  column is "<unknown>"
#  Another exception : if the 1st column contains special characters, it is kept
#  (for example : "close()" is kept as is)

# Variables :
# OFS = "\t"

# Input file :
# Full Word \t POS \t Lemmatized Word
# fonctionnement	N.C.m.s	fonctionnement
# name.surname@mail	ET	<unknown>
# <>

# Output file :
# Lemmatized Word
# fonctionnement
# name.surname@mail
# <>

{
    # If there are not 3 columns, write 1st column
    if (NF != 3)
    {
	printf("%s", $1);
    }
    else
    {
	# If word is unknown... let's copy it as is
	if ($3 == "<unknown>")
	{
	    printf("%s", $1);
	}
	else
	{
	    # If there are parenthesis in the 1st col, let's keep it
	    if ($0 ~ /[:cntrl:]/)
	    {
		printf("%s", $1);
	    }
	    else
	    {
		# If everything is well, let's copy the lemmatized word
		printf("%s", $3);
	    }
	}
    }

    # End of line
    printf("\n");
}

## Match lines with special characters

#$0 ~ /\(/ || /\)/ {
#    print $1;
#}


## Match within statements (1)
#if (($0 ~ /\(/) || ($0 ~ /\)/) ||
#  ($0 ~ /</) || ($0 ~ />/) ||
#  ($0 ~ /=/) ||
#  ($0 ~ /^/) ||
#  ($0 ~ /$/))

## Match within statements (2)
#if ((match($0, /\(/)) or (match($0, /\)/)))
