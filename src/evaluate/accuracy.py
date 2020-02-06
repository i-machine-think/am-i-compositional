"""
Script to compute am-i-compositional (sequence) accuracy score.
"""

import csv
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pred', type=str,
        help="Output file of translate.py."
    )
    args = vars(parser.parse_args())

    correct, all_samples = 0, 0

    # Open the prediction files, that contain src, tgt & pred
    predictions = csv.DictReader(open(args["pred"]), delimiter="\t")

    for item in predictions:
        prediction = item["prediction"].strip()
        target = item["target"].strip()

        if prediction == target:
            correct += 1
        all_samples += 1

    # Report statistics to user
    print(f"Sequence Accuracy: {correct / all_samples}")
