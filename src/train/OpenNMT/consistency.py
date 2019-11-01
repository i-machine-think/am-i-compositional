"""
Script to compute am-i-compositional consistency scores.

Python script acting on two predictions saved by `translate.py`,
(1) should be the output of the original substitutivity test set,
(2) should be the output of the synonyms replaced substitutivity test set.

OpenNMT version november 2018, using the adapted am-i-compositional
`translate.py` script.
"""

import csv
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file1', type=str,
        help="Output file of translate.py."
    )
    parser.add_argument(
        '--file2', type=str,
        help="Output file of translate.py., synonyms replaced"
    )
    args = vars(parser.parse_args())

    consistent, consistent_correct, consistent_wrong, wrong, all_samples = \
        0, 0, 0, 0, 0

    # Open the prediction files for the regular and synonym replaced test set
    predictions = csv.DictReader(open(args["file1"]), delimiter="\t")
    predictions_twin = csv.DictReader(open(args["file2"]), delimiter="\t")

    for item, item_twin in zip(predictions, predictions_twin):   
        l1 = item["prediction"].strip()
        l2 = item_twin["prediction"].strip()
        l3 = item["target"].strip()

        # Consider two outputs consistent if and only if they are the same
        # and non-empty
        if l1 == l2 and len(l1) > 0 and len(l2) > 0:
            consistent += 1
            if l1 == l3:
                consistent_correct += 1
            else:
                consistent_wrong += 1

        # Consider the output "wrong" if any of the two outputs is wrong
        if l1 != l3 or l2 != l3:
            wrong += 1
        all_samples += 1

    # Report statistics to user
    print("Consistent / All samples", consistent/all_samples)
    print("Consistent & correct / All samples", consistent_correct/all_samples)
    print("Consistent & wrong / All samples", consistent_wrong/all_samples)
    print("Consistent & wrong / Wrong samples", consistent_wrong/wrong)
