import csv
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', type=str)
    parser.add_argument('--file2', type=str)
    args = vars(parser.parse_args())

    consistent = 0
    consistent_correct = 0
    consistent_wrong = 0
    wrong = 0
    all_samples = 0
    predictions = csv.DictReader(open(args["file1"]), delimiter="\t")
    predictions_twin = csv.DictReader(open(args["file2"]), delimiter="\t")

    for item, item_twin in zip(predictions, predictions_twin):   
        l1 = item["prediction"].strip()
        l2 = item_twin["prediction"].strip()
        l3 = item["target"].strip()
        if l1 == l2 and len(l1) > 0 and len(l2) > 0:
            consistent += 1
            if l1 == l3:
                consistent_correct += 1
            else:
                consistent_wrong += 1
        if l1 != l3 or l2 != l3:
            wrong += 1
        all_samples += 1
    print("Consistent / All samples", consistent/all_samples)
    print("Consistent & correct / All samples", consistent_correct/all_samples)
    print("Consistent & wrong / All samples", consistent_wrong/all_samples)
    print("Consistent & wrong / Wrong samples", consistent_wrong/wrong)
