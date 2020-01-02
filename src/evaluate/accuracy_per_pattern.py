"""
Script to compute am-i-compositional accuracy scores for lines containing a
predefined pattern of functions.

This scripts takes in prediction files from OpenNMT or Fairseq
and the custom `translate.py` files provided in the am-i-compositional
repository.

These prediction files contain columns for "source", "target" and "prediction".

Example usage:
    python3 sequence_accuracy_per_function.py \
        --traces run=1.txt run=2.txt run=3.txt --pattern "append repeat"
"""

import csv
import argparse
import numpy as np


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=str, nargs="+",
                        help="Filenames containing srcs, tgts and predictions.")
    parser.add_argument("--pattern", type=str,
                        help="Consecutive function names to filter on.")
    args = vars(parser.parse_args())


    accs_over_traces = []

    # Collect accuracies per trace given as input, with one trace corresponding
    # to one [randomized] run
    for trace in args["traces"]:
        correct, incorrect = 0, 0
        for item in csv.DictReader(open(trace), delimiter="\t"):
            source = item["source"].strip()
            target = item["target"].strip()
            prediction = item["prediction"].strip()

            # Only consider the accuracy for src-tgt pairs containing the
            # user-defined pattern
            if args["pattern"] in source:
                if target == prediction:
                    correct += 1
                else:
                    incorrect += 1
        accs_over_traces.append(correct / (correct + incorrect))

    mean = np.mean(accs_over_traces)
    std = np.std(accs_over_traces)
    print(f"Accuracies: {' '.join([str(round(x,3)) for x in accs_over_traces])}")
    print(f"Mean: {mean:.3f}, Std: {std:.3f}")

