#!/bin/bash

# Path to main pcfg data folder, and name of dev file used for all experiments
PATH="..."
DEV=dev

# Experiment name, could be ( pcfg | systematicity | productivity | substitutivity )
# Subtype is only applicable for substitutivity (primitive | equally_distributed)
EXPERIMENT_NAME=$1
SUBTYPE=$2

# Parameters for saving, DATA_FOLDER contains the preprocessed Fairseq data
DATA_FOLDER="..."
# MODEL_FOLDER contains all trained models, the training traces and the model's predictions
MODEL_FOLDER="..."

if [ "${EXPERIMENT_NAME}" = "pcfg" ]; then
    # Parameters for PCFG data files to use for training and testing
    TRAIN=pcfgset/train
    TEST=pcfgset/test
elif [ "${EXPERIMENT_NAME}" = "systematicity" ]; then
    # Parameters for PCFG data files to use for training and testing
    TRAIN=systematicity/train2
    TEST=systematicity/test2
elif [ "${EXPERIMENT_NAME}" = "productivity" ]; then
    # Parameters for PCFG data files to use for training and testing
    TRAIN=productivity/train
    TEST=productivity/test
elif [ "${EXPERIMENT_NAME}" = "substitutivity" ]; then
    # Parameters for PCFG data files to use for training and testing
    TRAIN="experiments/substitutivity/${SUBTYPE}/train"
    TEST="experiments/substitutivity/${SUBTYPE}/test"

    # Parameters for saving, DATA_FOLDER contains the preprocessed Fairseq data
    DATA_FOLDER="data/${EXPERIMENT_NAME}_${SUBTYPE}"
    # MODEL_FOLDER contains all trained models, the training traces and the model's predictions
    MODEL_FOLDER="models/${EXPERIMENT_NAME}_${SUBTYPE}"
else
    echo "Experiment name ${EXPERIMENT_NAME} is unknown!"
    exit 0
fi

mkdir $MODEL_FOLDER

# Repeat three times to get averaged performance metrics
for i in 1 2 3
do
    FILENAME="${MODEL_FOLDER}/lstms2s_run=${i}_experiment=${EXPERIMENT_NAME}"

    # Use Fairseq file to preprocess the data into format usable during training
    python3.6 preprocess.py --source-lang src --target-lang tgt --trainpref "${PATH}/${TRAIN}" --validpref "${PATH}/${DEV}" --destdir "${DATA_FOLDER}_run=${i}"
    wait

    # Train bidirectional model with embeddings of size 512, using batch size 64
    python3.6 train.py "${DATA_FOLDER}_run=${i}" --no-epoch-checkpoints --arch fconv_wmt_en_de --lr 0.25 --max-tokens 3000 --save-dir "${MODEL_FOLDER}/" \
                                                 --clip-norm 0.1 --dropout 0.1 --max-epoch 25 --save-interval 1 --no-epoch-checkpoints \
                                                 --encoder-embed-dim 512 --decoder-embed-dim 512 \
                                                 --batch-size 64 --distributed-world-size 1 --device-id 1 > "${FILENAME}_trace.txt"
    wait
    mv "${MODEL_FOLDER}/checkpoint_best.pt" "${MODEL_FOLDER}/checkpoint_run=${i}.pt"
    model="${MODEL_FOLDER}/checkpoint_run=${i}.pt"


    if [ "${EXPERIMENT_NAME}" = "substitutivity" ]; then
        python3.6 translate.py "${DATA_FOLDER}_run=${i}" --src "${PATH}/${TEST}_twin.src" --tgt "${PATH}/${TEST}_twin.tgt" \
                                                         --path $model --batch-size 64 --max-tokens 3000 \
                                                         --buffer-size 64 --beam 5 --quiet --file "${FILENAME}_pred_twin.txt"
        wait
        python3.6 translate.py "${DATA_FOLDER}_run=${i}" --src "${PATH}/${TEST}.src" --tgt "${PATH}/${TEST}.tgt" \
                                                         --path $model --batch-size 64 --max-tokens 3000 \
                                                         --buffer-size 64 --beam 5 --quiet --file "${FILENAME}_pred.txt"
        wait
        python3.6 consistency.py --file1 -output "${FILENAME}_pred.txt" --file2 -output "${FILENAME}_pred_twin.txt" > "${FILENAME}_consistency.txt"
        wait
    else
        # Evaluate model on test data
        python3.6 translate.py "${DATA_FOLDER}_run=${i}" --src "${PATH}/${TEST}.src" --tgt "${PATH}/${TEST}.tgt" \
                                                         --path $model --batch-size 64 --max-tokens 3000 \
                                                         --buffer-size 64 --beam 5 --quiet --file "${FILENAME}_pred.txt"
        wait
    fi
done
