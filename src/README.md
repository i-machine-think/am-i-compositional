# am-i-compositional - Source Code

We will continue to add more useful scripts, if you miss a script that you think could be useful to you, please reach out to <dieuwkehupkes@gmail.com>.

## 1 - train: Scripts to train models

Please fill out the following parameters in the training scripts:
- `PATH`: path to the pcfgset data.
- `DATA_FOLDER`: folder to store the OpenNMT preprocessed data in.
- `MODEL_FOLDER`: folder to store the models and related files in.

And indicate the following two parameters in the command line:
- `EXPERIMENT_NAME`: `pcfg | systematicity | productivity | substitutivity`.
- `SUBTYPE`: `primitive | equally_distributed`, only applies to substitutivity.

Afterwards, train models using the following commands:
- `bash train_lstms2s_opennmt.sh pcfg`
- `bash train_lstms2s_opennmt.sh systematicity`
- `bash train_lstms2s_opennmt.sh productivity`
- `bash train_lstms2s_opennmt.sh substitutivity primitive`
- `bash train_lstms2s_opennmt.sh substitutivity equally_distributed`

## 2 - evaluate: Auxiliary scripts

In this folder, you can find several scripts that are helpful to visualise results obtained by using the compositionality evaluation paradigm.
Currently, it contains just one script, `plot_per_function.py` that generates a plot of the accuracy of a model per function.

