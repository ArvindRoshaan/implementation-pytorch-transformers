#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=2000 #10000

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

if [ $# -ne 6 ]; then
    echo "Mismatch in # of arguments provided"
    echo "Please provide -n model_unique_identifier_name ; -s source_language ; -t target_language"
    exit 1
fi

while getopts n:s:t: flag
do
    case "${flag}" in
        n) foldername=${OPTARG};;
        s) source_lang=${OPTARG};;
        t) target_lang=${OPTARG};;
    esac
done

src=$source_lang
tgt=$target_lang
lang=$source_lang-$target_lang
prep=$foldername
tmp=$prep/tmp

<< 'MULTILINE-COMMENT'
echo "creating valid..."
for l in $src $tgt; do
    cp $tmp/train.$l $tmp/train_tmp.$l
    awk '{if (NR%23 == 0)  print $0; }' $tmp/train_tmp.$l > $tmp/valid.$l
    awk '{if (NR%23 != 0)  print $0; }' $tmp/train_tmp.$l > $tmp/train.$l
    rm -f $tmp/train_tmp.$l
done

echo "creating train, test..."
for l in $src $tgt; do
    cp $tmp/train.$l $tmp/train_tmp.$l
    awk '{if (NR%23 == 0)  print $0; }' $tmp/train_tmp.$l > $tmp/test.$l
    awk '{if (NR%23 != 0)  print $0; }' $tmp/train_tmp.$l > $tmp/train.$l
    rm -f $tmp/train_tmp.$l
done
MULTILINE-COMMENT

#python preprocess_custom.py $foldername $src $tgt

TRAIN=$tmp/train.$lang
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $prep/$f
    done
done
