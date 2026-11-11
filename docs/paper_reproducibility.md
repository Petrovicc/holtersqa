# Reproducing the HolterSQA ECG signal-quality result

The HolterSQA source code and the scripts used for these checks are available in the [project repository](https://github.com/Petrovicc/holtersqa).

The reported result is the internal subject-independent evaluation of a specific saved multilayer perceptron (MLP). It includes 2,739 overlapping windows from six people. The saved weights give 1,927 correct predictions and 812 disagreements with the consensus reference. The application also supports logistic regression, random forests and gradient boosting. The earlier exploratory search over 270 configurations used a different preprocessing and evaluation protocol. Its scores cannot establish which model performs best under the protocol reported here.

## Reproduce the predictions from BUT QDB

Install the dependencies documented in the project README. Obtain BUT QDB version 1.0.0 from PhysioNet and retain its `SHA256SUMS.txt`. This check needs records 104001, 114001, 115001, 123001, 125001 and 126001, each with its ECG, accelerometer and annotation files. Run from the reviewed project checkout.

```shell
python scripts/reproduce_public.py --dataset-root /path/to/butqdb --bundle-dir artifacts --output-dir public_check
```

The script validates the model and 30 source files, extracts the consensus-labelled windows directly from WFDB, and runs the saved pipeline. It compares each result with `docs/reproducibility/predictions.csv` by subject, record, sample bounds and reference class. The 11 November 2026 check matched all 2,739 window identifiers, labels, predicted classes and probability vectors. The largest absolute probability difference was zero (allowed absolute and relative tolerances both 1e−12). The output includes new predictions and a JSON record of the check. This route requires only the released model, training-results JSON, reference CSV, project code and the specified BUT QDB files.

The archived bundle was created at 00:04:41 UTC on 24 August 2026 in format `ekg-monitor-model/2`. Its SHA-256 is `0b0adf27512aa4812d43f4fcc9214d7e8faed6a321c635a0b30a75925cda45df`. The export identifies candidate 19 of 20 and seed 42. There is no exported run UUID, historical Git commit, complete environment lock or separate epoch-by-epoch weight archive. The estimator serialization records scikit-learn 1.8.0. The checked inference environment was Windows AMD64, Python 3.14.4, scikit-learn 1.8.0, NumPy 2.2.6, SciPy 1.16.3, pandas 2.3.3, joblib 1.5.3, PyArrow 23.0.0, WFDB 4.3.1 and Matplotlib 3.10.8. The manifest records the reviewed source-file hashes and distinguishes these versions from historical metadata.

A second check in the clean Windows release environment (CPython 3.13.11) reproduced every window identifier and predicted class. The maximum absolute probability difference was 1.22 × 10⁻¹⁵. The full installed package versions and the result are recorded in [the release environment](release_environment.json) and [the clean-install reproduction check](reproducibility/clean_install_public_check.json).

## Allocation, resampling and model selection

The source contains 18 recordings from 15 people. Each partition uses the three-digit person ID. For example, 100001 and 100002 both belong to 100 and both entered the training partition of candidate 19. Neither record was used for validation or evaluation in that candidate. The original prepared feature frame and all 20 logged allocations agree with the grouping. The `subject_record_class_counts.csv` file gives the individual record and class counts before and after resampling.

| Candidate 19 partition | Subject IDs | Original Class 1 / 2 / 3 | After resampling Class 1 / 2 / 3 |
| --- | --- | ---: | ---: |
| Training | 100, 103, 111, 113, 118, 121, 122, 124 | 20,302 / 14,411 / 878 | 20,302 / 20,302 / 20,302 |
| Validation | 105 | 10,729 / 5,926 / 9,467 | 10,729 / 10,729 / 10,729 |
| Reported evaluation | 104, 114, 115, 123, 125, 126 | 2,248 / 326 / 165 | 2,248 / 326 / 165 |

After assigning subjects, the implementation sampled each class to the size of the largest class in that partition. It used replacement for classes below the target and shuffled the combined rows, with seeds 42 for training and 43 for validation. Class weights for imbalance were disabled by the oversampling branch. A separate sample weight equal to the stored dangerous-error coefficient was applied to actual Class 3 during MLP fitting. The validation objective therefore reflects a balanced validation distribution. The reported evaluation retained the original class frequencies.

The saved development log contains all 20 ordered choices of validation and evaluation roles among five shuffled stratified subject groups (seed 42). The remaining three groups trained each candidate under the same model settings. Each candidate has disjoint training, validation and evaluation subjects. The 20 candidates share people and data across roles. `development_candidates.csv` gives the scores, subjects and original window counts for each candidate. The reviewed implementation computes and stores candidate evaluation metrics, then chooses the first maximum validation objective. Candidate 19 has the highest logged validation objective. Candidate 12 has the highest reported-partition accuracy, balanced accuracy and macro F1. These evaluation metrics have no input to the implemented argmax. Logs cannot determine whether informal decisions between runs used already visible evaluation results, or why this run was chosen for reporting. Reuse of a subject across development candidates does not by itself demonstrate leakage within a candidate. The paper reports one selected development result. It does not report a pooled out-of-fold estimate, 20 independent replications or external validation.

## Features, standardization and checkpoint

The archived feature configuration specifies 10-s windows and 50% overlap. Offline windows advance by 5 s within eligible consensus intervals. ECG is read in µV at 1,000 Hz, acceleration in mg per axis at 100 Hz. Both streams use the same record-relative time range. ECG sample boundaries determine time in seconds. The dataset reader rounds start times and durations to the corresponding acceleration sampling grid. When building blocks of nearby windows, offsets within the block are likewise rounded to the acceleration grid. There is no interpolation to the ECG rate. The project reader converts WFDB physical values to float32. The feature extractor uses float64 arithmetic.

The exported preprocessing removes the ECG window mean and applies a zero-phase, fourth-order Butterworth design with cutoffs 0.5 and 40 Hz, using second-order sections passed forward and backward over the complete window. The 64 inputs, in exported order, are eight ECG time statistics, 32 ECG magnitude bins and eight acceleration-difference statistics for each of three axes. The eight statistics are minimum, maximum, mean, population standard deviation, root mean square, peak-to-peak range, median and unscaled median absolute deviation from the median. Each acceleration axis uses its first difference times the sampling rate (mg/s), with an initial difference of zero.

The FFT input is the filtered ECG with its within-window mean subtracted. The implementation takes absolute values of the real FFT, omits DC and averages contiguous magnitudes into 32 groups using integer-truncated evenly spaced index edges. For 10,000 samples at 1,000 Hz, the included positive frequencies run from 0.1 to 500 Hz, including Nyquist. The Butterworth passband acts before the FFT. Each grouped value is the mean of its unnormalized Fourier magnitudes, with no taper, power transform, division by window length or normalization by total spectral energy. `fft_bins.csv` gives the exact first and last frequencies in each group. `features.csv` gives the 64 named inputs in order with the learned scaler statistics.

The saved scikit-learn pipeline contains a fitted StandardScaler followed by a 64–128–64 ReLU MLP with three softmax outputs. All 64,452 archived feature rows are finite, so the reviewed code's pre-split median imputation branch had no effect. Independently reconstructing the seed-42 oversampled training rows reproduced all 64 learned means, population variances and scale factors within rtol 1e−11 and atol 1e−9. The scaler reports 60,906 fitted rows. The server loads that same fitted pipeline. Training used Adam with a stored constant learning rate of 0.0010000000474974513, L2 alpha 0.0001, batch size `auto` (200 effective), seed 42, and a 200-epoch limit. The training wrapper calls the estimator with one warm-started iteration per epoch and controls validation and early stopping itself.

For the stored overall-accuracy setting, the reviewed selection function is

```
S = n_correct / N − λ × n31 / max(1, n3)
λ = 5.999999046325684
```

Here N counts all evaluated validation windows, n3 counts reference Class 3 windows, and n31 counts those classified as Class 1. An absent Class 3 gives zero penalty. The score is neither clipped nor further normalized and may be negative. The curve labelled `accuracy_curve` in the historical JSON records this objective. On the oversampled validation set a strict improvement exceeding delta = 0.0010000000474974513 both resets the patience counter and copies the layer coefficients and intercepts. The wrapper restores those coefficients and intercepts on stopping. Epoch 3 scored 0.9122627147606176. Epoch 5 scored 0.9132258365178488, an increase of 0.0009631217572312, below the retention threshold. The run stopped at epoch 103 after 100 epochs without a qualifying improvement. The saved estimator reproduces the retained epoch-3 validation objective exactly on all 32,187 reconstructed validation rows. The stored training loss curve also matches the estimator's 103 recorded losses. This verifies the checkpoint's reported behavior. The available files do not allow an independent comparison with a separately saved epoch-3 weight hash or reconstruction of its historical optimizer trajectory.

The reviewed training source's filesystem timestamp predates the model export and its rule agrees with the complete logged candidate scores and retained epoch. An older project source archive lacks this 20-candidate procedure and cannot identify the exact historical source snapshot. Historical training was not repeated. The 270-run exploratory material used a different preprocessing and evaluation protocol.

## Class-specific interpretation

| Reference class | Predicted 1 / 2 / 3 | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |
| 1 | 1,480 / 762 / 6 | 98.54% | 65.84% | 78.93% |
| 2 | 22 / 285 / 19 | 27.14% | 87.42% | 41.42% |
| 3 | 0 / 3 / 162 | 86.63% | 98.18% | 92.05% |

Accuracy was 70.354%, balanced accuracy 83.814% and macro F1 70.801%. Of 1,050 predicted Class 2 windows, 285 had that reference class. This may create additional review needs, but window counts do not measure clinical alarms or staff time. A 3-to-2 error may matter if a Class 2 decision permits QRS analysis. There were zero 3-to-1 errors among 165 Class 3 windows, all from subject 114. Between-person Class 3 sensitivity and zero population risk cannot be inferred.

The companion table `per_subject_class_performance.csv` supplies numerator, denominator, precision and recall. Among six subjects with reference Class 1, recall ranged from 2.21% (10/453) to 99.73% (376/377). For four subjects with reference Class 2, recall ranged from 66.67% (10/15) to 100% (101/101 and 80/80). Empty denominators have blank estimates. These are observed descriptive ranges. The small number of contributing people, single selected development partition, overlapping windows and one-person Class 3 support limit uncertainty estimates. No nominal 95% population intervals were generated. The historical export's individual-window intervals do not account for subject clustering.

## Packet processing and the Figure 4 demonstration

Figure 4 was captured during a localhost TCP replay on 127.0.0.1:50070. Three replay clients streamed records 100001, 100002 and 103001 from two people. The requested packet interval was 2 s. The server accumulates a complete 10-s ECG buffer, then predicts on the most recent complete window after each incoming eligible packet. With contiguous 2-s data packets this yields a 2-s inference step. Zero-phase filtering acts on the complete buffered window. The Figure 4b snapshot shows 56–66 s of the third recording. The retained stdout records startup, streaming, quality messages and connection closure at shutdown with automatic reconnection disabled. It lacks timestamped packet/ACK pairs, an end-to-end reference clock and a sample-continuity report. Total replay duration, latency distributions, sustained throughput, uninterrupted delivery and reconnect performance cannot be established. Older one-record packet archives are unrelated to this figure capture.

On 11 November 2026 a separate functional check passed five real two-second packets from record 104001, ECG samples [28,900,000, 28,910,000), through the production encoder and server packet handler. The first four packets returned a warm-up state. The fifth returned Class 2. The complete server feature vector, standardized vector and all three class probabilities matched the offline prepared window. The measured maximum absolute difference was zero for each comparison. Tested tolerances were rtol 1e−12, atol 1e−9 for features and standardized values, and atol 1e−12 for probabilities. The probabilities in class order were 0.4749891962, 0.5221083801 and 0.0029024237. `server_parity.json` records the values. This check covers one uninterrupted model-input window. Network and clinical performance require separate measurements.

## Additional archival checks

With access to the original project archive, run the following commands from this checkout. These checks use the archived private feature file `DBs/newDB.parquet`, `Models/New Top`, and the original `butqdb` directory. They verify stored scaling, grouping, selection logs and sample boundaries. They do not fit an estimator or generate a new evaluation split.

```shell
python scripts/reproduce_paper.py --project-root /path/to/archive --code-root . --output-dir paper_check
python scripts/verify_paper_waveforms.py --project-root /path/to/archive --code-root . --predictions paper_check/predictions.csv --output-dir waveform_check
```

Supplementary Material S1 provides all prediction identifiers and reference/predicted classes, all 812 discordant windows, and five 10-s examples with source record, sample bounds and raw ECG. The first example in each observed error direction was chosen by sorting by record ID and start sample. Raw waveform values, consensus labels and all 64 features were verified for all five examples against the official WFDB/annotation files. No expert re-adjudication or individual feature-attribution study was performed.

BUT QDB source: Andrea Nemcova, Radovan Smisek, Kamila Opravilová, Martin Vitek, Lukas Smital and Lucie Maršánová, *Brno University of Technology ECG Quality Database*, version 1.0.0, PhysioNet, 2020. Dataset identifiers and derived excerpts are used under CC BY 4.0. Derived outputs add window boundaries, model decisions, scores and display formatting. See the [dataset and citations](https://physionet.org/content/butqdb/1.0.0/) and [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).


The dataset's official site requests citation of Smital et al., *Real-world 24h+ ECG dataset with quality annotations and motion context* ([Scientific Data](https://doi.org/10.1038/s41597-026-07905-w)), and Pollard et al., *PhysioNet as a global platform for biomedical research* ([Nature Health](https://doi.org/10.1038/s44360-026-00096-z)). Dataset DOI [10.13026/kah4-0w24](https://doi.org/10.13026/kah4-0w24).
