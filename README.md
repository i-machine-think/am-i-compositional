# am-i-compositional

This repository contains data and scrips to use the tests from the compositionality evaluation paradigm described in the paper.

Dieuwke Hupkes, Verna Dankers, Mathijs Mul and Elia Bruni. 2020. [ompositionality decomposed: how do neural networks generalise](https://jair.org/index.php/jair/article/view/11674). Journal of Artificial Intelligence Research (JAIR).

Below we provide a brief description of the structure of the repository.
The explanation for usage of the individual scripts can be found in the corresponding folders.
Please cite the paper if you want to use the resources in this repository.
If you have any questions, feel free to ask them: <dieuwkehupkes@gmail.com>.

## 1 - Data: PCFG & Compositionality tests

The folder [data/pcfgset](https://github.com/i-machine-think/am-i-compositional/tree/master/data/pcfgset) contains six folders with the data to compute the overall task accuracy for PCFG set for a model, and the data and scripts to conduct the five main tests in the evaluation paradigm -- [systematicity](https://github.com/i-machine-think/am-i-compositional/tree/master/data/pcfgset/systematicity), [productivity](https://github.com/i-machine-think/am-i-compositional/tree/master/data/pcfgset/productivity), [substitutivity](https://github.com/i-machine-think/am-i-compositional/tree/master/data/pcfgset/substitutivity), [localism](https://github.com/i-machine-think/am-i-compositional/tree/master/data/pcfgset/localism) and [overgeneralisation](https://github.com/i-machine-think/am-i-compositional/tree/master/data/pcfgset/overgeneralisation).
In the folders for these tests you can find also the data to conduct the auxiliary tests that we conducted in the paper to further clarify the results of the different tests in the evaluation paradigm.

## 2 - Source code: Training and auxiliary scripts

In addition to the data, we provide scripts necessary for training models on this data in `src/train` and scripts used to interpret the results in `src/evaluate`, such as scripts to plot the accuracy of a model per function.
The scripts for training are specific to the [OpenNMT](https://github.com/OpenNMT/OpenNMT-py) and [Fairseq](https://github.com/pytorch/fairseq) toolkits.
