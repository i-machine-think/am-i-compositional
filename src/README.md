# am-i-compositional - Source Code

We will continue to add more useful scripts, if you miss a script that you think could be useful to you, please reach out to <dieuwkehupkes@gmail.com>.

Three types of scripts are provided:
- Compositionality tests that require model training using OpenNMT or Fairseq named `train_lstms2s.sh`, `train_transformer.sh`, `train_convs2s.sh`.
- Compositionality tests that use a trained model and require translating using OpenNMT or Fairseq `translate_lstms2s.sh`, `translate_transformer.sh`, `translate_convs2s.sh`.
- Scripts that take emitted translations and compute new statistics or visualise results: `accuracy.py`, `consistency.py`, any script named `plot_..._.py`, `accuracy_per_pattern.py`, `significance_testing.py` and `eos_problem.py`.

Please fill out the following parameters in the train and translate scripts, that are set to match the paths from this repository by default:
- `DATA_FOLDER`: path to the pcfgset data.
- `PROCESSED_DATA_FOLDER`: folder to store the OpenNMT preprocessed data in.
- `MODEL_FOLDER`: folder to store the models and related files in.

## Experiment: Task accuracy (Section 7.1)

#### 1. <span style="color:blue">train</span> -- Main experiment (Section 7.1)

Dependent on the model used, run `./train_lstms2s.sh pcfg` or `./train_transformer.sh pcfg` from within OpenNMT, or `./train_convs2s.sh pcfg` from within Fairseq.

#### 2. Correlation length & depth (Section 7.1.1)

- Run the following command per subplot, with all emitted predictions on test data for all model types listed:
`python plot_length_depth.py --lstms2s "lstm_run=1.txt" "lstm_run=2.txt" "lstm_run=3.txt" --convs2s "convs2s_run=1.txt" "convs2s_run=2.txt" "convs2s_run=3.txt" 
--transformer "transformer_run=1.txt" "transformer_run=2.txt" "transformer_run=3.txt" --output_filename depth.pdf --mode depth`
- The available modes are `depth | length | number of functions`

#### 3. Function difficulty (Section 7.1.2)

- Dependent on the model used, run `./translate_lstms2s.sh per_function` or `./translate_transformer.sh per_function` from within OpenNMT, or `./translate_convs2s.sh per_function` from within Fairseq.
- The translation script will gather performances per PCFG SET function in one file per run. Afterwards, the following script can be used to visualise the results (after replacing the filenames).
`python plot_per_function.py --traces "run=1_per_function.txt" "run=2_per_function.txt" "run=3_per_function.txt" --output_filename per_function.pdf`

## Experiment: Systematicity (Section 7.2)

#### 1. <span style="color:blue">train</span> -- Main experiment (Section 7.2)

Dependent on the model used, run `./train_lstms2s.sh systematicity` or `./train_transformer.sh systematicity` from within OpenNMT, or `./train_convs2s.sh systematicity` from within Fairseq.

#### 2. Breakdown the performance per heldout composition (Section 7.2)

Run the following script with the emitted predictions on test data for the model type of interest:
`python sequence_accuracy_per_function.py --pred run1.txt run2.txt run3.txt --pattern "reverse echo"`.
The `pattern` argument refers to the consecutive functions of interest.

## Experiment: Productivity (Section 7.3)

#### 1. <span style="color:blue">train</span> -- Main experiment (Section 7.3)
Dependent on the model used, run `./train_lstms2s.sh productivity` or `./train_transformer.sh productivity` from within OpenNMT, or `./train_convs2s.sh productivity` from within Fairseq.

#### 2. <span style="color:purple">evaluate</span> -- Correlation length & depth (Section 7.3.1)

#### 3. EOS problem (Section 7.3)

## Experiment: Substitutivity (Section 7.4)

#### 1. Main experiment, condition equally distributed (Section 7.4.1)
Dependent on the model used, run `./train_lstms2s.sh substitutivity equally_distributed` or `./train_transformer.sh substitutivity equally_distributed` from within OpenNMT, or `./train_convs2s.sh substitutivity equally_distributed` from within Fairseq.

#### 2. Main experiment, condition primitive (Section 7.4.2)
Dependent on the model used, run `./train_lstms2s.sh substitutivity primitive` or `./train_transformer.sh substitutivity primitive` from within OpenNMT, or `./train_convs2s.sh substitutivity primitive` from within Fairseq.

#### 3. Consistency score breakdown (Section 7.4.1-2) 

#### 4. Embedding distances (Section 7.4.1-2)

## Experiment: Localism (Section 7.5)

#### 1. Main experiment with unrolled sequences (7.5.1)
Dependent on the model used, run `./translate_lstms2s.sh localism` or `./translate_transformer.sh localism` from within OpenNMT, or `./translate_convs2s.sh localism` from within Fairseq.

#### 2. Primitive setup with varying input lengths (Section 7.5.2)
Dependent on the model used, run `./translate_lstms2s.sh localism_primitive` or `./translate_transformer.sh localism_primitive` from within OpenNMT, or `./translate_convs2s.sh localism_primitive` from within Fairseq.

## Experiment: Overgeneralisation peak (Section 7.6)

#### 1. Overgeneralisation profile (Section 7.6.2)

To generate the overgeneralisation profiles (Section 7.6.2), one needs to run the following command: `./train_lstms2s_overgeneralisation.sh 0.01`, with the desired ratio indicated as argument.
Available ratios are `0.005 | 0.001 | 0.0005 | 0.0001`, that correspond with the named datasets in the `data/overgeneralisation` folder.

#### 2. Main experiment / overgeneralisation peak (Section 7.6.1)

From the overgeneralisation profiles the peaks are to be calculated by averaging the peak heights per randomly initialised runs.