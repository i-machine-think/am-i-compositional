# Basis corpus

### About PCFG SET

The input alphabet of PCFG SET contains three types of tokens:
1. Tokens for unary and binary funtions that represent string edit operations (e.g. append, copy, reverse);
2. Tokens to form the string sequences that these functions can be applied to (e.g. A, B, A1, B1);
3. A separator to separate the arguments of a binary function (,).
The input sequences that are formed with these task are sequences describing how a series of such operations are to be applied to a string argument.
Further information on the generation of sequences is provided in the paper.

### About the data

This pcfgset folder contains the following data:
1. Training set: `train.src` and `train.tgt`
2. Development set: `dev.src` and `dev.tgt` - notice that this is the development set for all other compositionality tests as well. While the training and testing set vary per test, the development set does not.
3. Testing set: `test.src` and `test.tgt`
4. The `per_function_test` folder: contains source and target files per function. All unary functions are applied to the same input and all binary functions are applied to the same inputs.

### Data format

- All files in this folder contain function names or characters separated using spaces.
- For every input sequence from a source file, its corresponding target is listed in the target file on the same line number.

```
reverse copy O14 O4 C12 J14 W3
```

To process the sample listed above (line 16 from the train files), the following tasks are carried out:
1. Feed `reverse copy O14 O4 C12 J14 W3` to your model.
2. Find the corresponding target from the target file (`W3 J14 C12 O4 O14`).
3. If the model output matches the target exactly, the output is considered accurate. Otherwise, the output is inaccurate.

