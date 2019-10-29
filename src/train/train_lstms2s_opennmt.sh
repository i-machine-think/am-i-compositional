#!/bin/bash

# Path to main pcfg data folder, and name of dev file used for all experiments
PATH="..."
DEV=dev

# Experiment name, could be ( pcfg | systematicity | productivity | substitutivity )
# Subtype is only applicable for substitutivity (primitive | equally_distributed)
EXPERIMENT_NAME=$1
SUBTYPE=$2

# Parameters for saving, DATA_FOLDER contains the preprocessed OpenNMT data
DATA_FOLDER="..."
# MODEL_FOLDER contains all trained models, the training traces and the model's predictions
MODEL_FOLDER="..."

if [ "${EXPERIMENT_NAME}" = "pcfg" ]; then
    # Parameters for PCFG data files to use for training and testing
    TRAIN=pcfgset/train
    TEST=pcfgset/test

    # Length of training should equal 25 * #steps for one epoch
    EPOCH_STEPS=1280
    TOTAL_STEPS=32125 
elif [ "${EXPERIMENT_NAME}" = "systematicity" ]; then
    # Parameters for PCFG data files to use for training and testing
    TRAIN=systematicity/train2
    TEST=systematicity/test2

    # Length of training should equal 25 * #steps for one epoch
    EPOCH_STEPS=1277
    TOTAL_STEPS=31925
elif [ "${EXPERIMENT_NAME}" = "productivity" ]; then
    # Parameters for PCFG data files to use for training and testing
    TRAIN=productivity/train
    TEST=productivity/test

    # Length of training should equal 25 * #steps for one epoch
    EPOCH_STEPS=1265
    TOTAL_STEPS=31625
elif [ "${EXPERIMENT_NAME}" = "substitutivity" ]; then
    # Parameters for PCFG data files to use for training and testing
    TRAIN="experiments/substitutivity/${SUBTYPE}/train"
    TEST="experiments/substitutivity/${SUBTYPE}/test"

    # Parameters for saving, DATA_FOLDER contains the preprocessed OpenNMT data
    DATA_FOLDER="data/${EXPERIMENT_NAME}_${SUBTYPE}"
    # MODEL_FOLDER contains all trained models, the training traces and the model's predictions
    MODEL_FOLDER="models/${EXPERIMENT_NAME}_${SUBTYPE}"

    # Length of training should equal 25 * #steps for one epoch
    EPOCH_STEPS=1290
    TOTAL_STEPS=32250
else
    echo "Experiment name ${EXPERIMENT_NAME} is unknown!"
    exit 0
fi

mkdir $MODEL_FOLDER

# Repeat three times to get averaged performance metrics
for i in 1
do
    FILENAME="${MODEL_FOLDER}/lstms2s_run=${i}_experiment=${EXPERIMENT_NAME}"

    # Use OpenNMT file to preprocess the data into format usable during training
    python3.6 preprocess.py -train_src "${PATH}/${TRAIN}.src" -train_tgt "${PATH}/${TRAIN}.tgt" \
                            -valid_src "${PATH}/${DEV}.src" -valid_tgt "${PATH}/${DEV}.tgt" \
                            -save_data $DATA_FOLDER
    wait

    # Train bidirectional model with embeddings of size 512, using batch size 64
    python3.6 train.py -word_vec_size 512 -encoder_type brnn -data $DATA_FOLDER -save_model $FILENAME \
                       -gpu_ranks 0 -valid_steps $EPOCH_STEPS -train_steps $TOTAL_STEPS \
                       -batch_size 64 -save_checkpoint_steps $EPOCH_STEPS 2> "${FILENAME}_trace.txt"
    wait

    # Out of the saved models, use the best-performing one based on validation data
    python3.6 select_model.py --model_name $FILENAME --steps $EPOCH_STEPS \
                              --trace "${FILENAME}_trace.txt" --folder $MODEL_FOLDER
    wait

    if [ "${EXPERIMENT_NAME}" = "substitutivity" ]; then
        python3.6 translate.py -src "${PATH}/${TEST}.src" -tgt "${PATH}/${TEST}.tgt" \
                               -model "${FILENAME}.pt" -gpu 0 -batch_size 64 -output "${FILENAME}_pred.txt"
        wait
        python3.6 translate.py -src "${PATH}/${TEST}_twin.src" -tgt "${PATH}/${TEST}_twin.tgt" \
                               -model "${FILENAME}.pt" -gpu 0 -batch_size 64 -output "${FILENAME}_pred_twin.txt"
        wait
        python3.6 consistency_compare.py --file1 -output "${FILENAME}_pred.txt" --file2 -output "${FILENAME}_pred_twin.txt" > "${FILENAME}_consistency.txt"
        wait
    else
        # Evaluate model on test data
        python3.6 translate.py -src "${PATH}/${TEST}.src" -tgt "${PATH}/${TEST}.tgt" \
                               -model "${FILENAME}.pt" -gpu 0 -batch_size 64 \
                               -output "${FILENAME}_pred.txt" 2> "${FILENAME}_taskaccuracy.txt"
        wait
    fi
done
