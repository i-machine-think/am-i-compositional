# Productivity

### About the test

With our productivity test, we focus purely on this extrapolation aspect, by studying models' ability to successfully generalise to longer sequences, which we will call the model's productive power.
For PCFG SET, we simply pooled the training and testing data (see the `../pcfgset/` folder) and split the samples based on the number of functions in the input sequence.
Sequences containing up to eight functions are collected in the training set, consisting of 81 thousand sequences, while input sequences containing at least nine functions are used for evaluation and collected in a test set containing 11 thousand sequences.

### About the data

This Productivity data folder contains the following items:
1. `train.src` and `train.tgt`
2. `test.src` and `test.tg`

### Data format

The format equals the regular PCFG SET format. The samples in the source files are ordered based on the number of functions, such that the training data includes all sequences from 1 to 8 functions,

```
copy R11 B10
...
remove_first swap_first_last Y7 H8 A6 , copy remove_second reverse reverse swap_first_last swap_first_last X9 Y12 L10 J13 J4 , T6 Y18
```

and the testing data includes all sequences ranging from 9 to 35 functions.