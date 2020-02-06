# am-i-compositional

This repository contains data and scrips to use the tests from the compositionality evaluation paradigm described in the paper 

Dieuwke Hupkes, Verna Dankers, Mathijs Mul and Elia Bruni. 2018. [The compositionality of neural networks: integrating symbolism and connectionism](https://arxiv.org/abs/1908.08351).

Below we describe the structure of the repository, explanation for usage of the individual scripts can be found in the corresponding folders.
Please cite the paper if you want to use the resources in this repository.
If you have any questions, feel free to ask them: <dieuwkehupkes@gmail.com>.

## Tests

The folder `data/pcfgset` contains six folders with the data to compute the overall task accuracy for PCFG set for a model, and the data and scripts to conduct the five main tests in the evaluation paradigm -- `systematicity`, `productivity`, `substitutivity`, `localism` and `overgeneralisation`.
In the folders for these tests you can find also the data to conduct the auxiliary tests that we conducted in the paper to further clarify the results of the different tests in the evaluation paradigm.

## Auxiliary scripts

In addition to the data, we also made several scripts available that we used to interpret the results, such as scripts to plot the accuracy of a model per function.
