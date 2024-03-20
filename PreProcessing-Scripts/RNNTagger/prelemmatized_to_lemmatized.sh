#! /bin/sh

OUTPUT_PREFIX=lemmatized_

CONVERT_SCRIPT=extract_3rd_or_1st_column.awk

AWK=awk

if [ $# -lt 1 ]; then
    echo "Missing arguments"
    echo "$0 input_file1.csv [...]"
    exit -1
fi

#############################################

for ARG in "$@"
do
    if [ -f "${ARG}" ]; then
        IN_NAME="${ARG}"
        IN_BASENAME=`basename "${IN_NAME}"`

        echo "-- ${IN_BASENAME} --"

        OUTPUT_NAME="${OUTPUT_PREFIX}${IN_BASENAME}"

        TMP1_IN=`mktemp tmp1.XXXXX`
        TMP2_OUT=`mktemp tmp2.XXXXX`

        cp -f "${IN_NAME}" ${TMP1_IN}

        # Extract the lemmatized word (3rd column usually, sometimes the 1st)
        ${AWK} -F "\t" -v OFS="\t" \
               -f ${CONVERT_SCRIPT} \
               ${TMP1_IN} > ${TMP2_OUT}

        cp -f ${TMP2_OUT} "${OUTPUT_NAME}"

        rm -f ${TMP1_IN}
        rm -f ${TMP2_OUT}
    fi
done
