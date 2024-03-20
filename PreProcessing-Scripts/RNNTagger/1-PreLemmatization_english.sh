#!/bin/sh

## Script for lemmatizing each file in parameter

OUTPUT_PREFIX=prelemmatized_

######

if [ "$#" = "0" ]; then
    echo "Missing arguments"
    echo "$0 input_file1 [input_file2] ..."
    exit 255
fi

for ARG in "$@"
do
    if [ -f "${ARG}" ]; then
        IN_NAME="${ARG}"
        IN_BASENAME=`basename "${IN_NAME}"`

	OUTPUT_NAME="${OUTPUT_PREFIX}${IN_BASENAME}"

	cmd/rnn-tagger-english.sh "${IN_NAME}" > "${OUTPUT_NAME}"
    fi
done
