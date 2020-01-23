# Systematicity

### About the test

In the systematicity test, we focus on models' ability to interpret pairs of functions that were never seen together while training.
We evaluate four pairs of functions: `swap repeat`, `append remove_second`, `repeat remove_second` and `append swap`.
We redistribute the training and test data such that the training data does not contain any input sequences including these specific four pairs, and all sequences in the test data contain at least one.
After this redistribution, the training set contains 82 thousand input-output pairs, while the test set contains 10 thousand examples.
Note that while the training data does not contain any of the function pairs listed above, it still may contain sequences that contain both functions.
E.g. `reverse repeat remove_second A B , C D` cannot appear in the training set, but `repeat reverse remove second A B , C D` might.

### About the data

This Systematicity data folder contains the following items:
1. `train.src` and `train.tgt`
2. `test.src` and `test.tg`

### Data format

The format equals the regular PCFG SET format.

