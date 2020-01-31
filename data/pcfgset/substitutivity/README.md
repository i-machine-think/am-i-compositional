# Substitutivity

### About the test

We study under what conditions models treat words as synonyms.
We consider what happens when synonyms are equally distributed in the input sequences and the case in which one of the synonyms only occurs in primitive contexts.
For functions `swap`, `repeat`, `append` and `remove_second`, we artificially introduce synonyms during training that have the same interpretation functions as the terms they substitute.
We consider two conditions that differ in the syntactic distribution of the synonyms in the training data.

### About the data

The Substitutivity data folder contains two subfolders, including the following files:
1. The `primitive` folder corresponding to the primitive test condition.
	a. The `train.src` and `train.tgt` files, that correspond to the regular PCFG SET data, but with extra primitives added for synonyms.
	b. The `test.src` and `test.tgt` files with lines that contain at least one function that was coupled to a synonym.
	c. The `test_twin.src` and `test_twin.tgt` files with affected functions replaced with their synonym.
2. The `equally_distributed` folder corresponding to the equally_distributed test condition.
	a. The `train.src` and `train.tgt` files, that correspond to the regular PCFG SET data adapted to include synonyms.
	b. The `test.src` and `test.tgt` files with lines that contain at least one function that was coupled to a synonym.
	c. The `test_twin.src` and `test_twin.tgt` files with affected functions replaced with their synonym.

### Data format

- The data format equals the regular PCFG SET format.
- The difference between `test.src` and `test_twin.src` is that affected functions from the test are replaced with their synonyms in `test_twin.src`, according to the following format: `FUNCTION` becomes `FUNCTION_twin`.
  Every line contains one function that is replaced by its synonym, to be able to isolate model performance per function.

```
swap_first_last append Q7 O3 P1 Y19 , G9 T14 U6 T9
swap_first_last append_twin Q7 O3 P1 Y19 , G9 T14 U6 T9
```

To process the sample above (line 10 from `test.src` and line 10 from `test_twin.src`), the following steps are carried out:
1. `swap_first_last append Q7 O3 P1 Y19 , G9 T14 U6 T9` is processed by the model and the output is collected.
2. `swap_first_last append_twin Q7 O3 P1 Y19 , G9 T14 U6 T9` is processed by the model and the output is collected.
3. If the outputs of (1) and (2) are equal, the output is considered consistent irrespective of whether it is accurate. If both outputs equal the target listed in `test.tgt` (and `test_twin.tgt`), they are considered accurate.
