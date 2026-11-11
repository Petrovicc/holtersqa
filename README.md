# HolterSQA

**A Framework for ECG Signal Quality Assessment and Telemonitoring**

[Paper snapshot 0.1.1](https://github.com/Petrovicc/holtersqa/releases/tag/paper-v0.1.1)

**Research prototype—not for clinical decision-making.**

HolterSQA combines an offline training application with a replay client and a physician-facing server. It classifies the technical usability of a single-lead ECG into three quality classes using ECG and synchronized triaxial acceleration. The implementation supports logistic regression, random forest, gradient boosting and multilayer perceptrons (MLPs). The paper reports internal subject-independent evaluation of one saved MLP and demonstrates its transfer to the monitoring server. An earlier 270-configuration search used a different protocol. The paper result establishes neither a matched comparison between all four families nor the incremental value of individual feature groups.

## Install

The clean release check ran on 64-bit Windows 11 with CPython 3.13.11. From the repository root, run these PowerShell commands.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-win-py313.lock
.\.venv\Scripts\python.exe -m pip check
```

`requirements.txt` records the ten direct dependencies. The lock file captures the complete installed distributions for the tested Windows environment. The [release environment](docs/release_environment.json) separates this installation from the historical training environment.

## Data

Download [BUT QDB version 1.0.0 from PhysioNet](https://physionet.org/content/butqdb/1.0.0/). Point the application at the extracted directory containing identifiers such as `100001/` and `subject-info.csv`. Keep each record's original ECG and ACC WFDB header/data pairs and annotation CSV. The first three digits of each record identifier identify a person, so records 100001 and 100002 represent two recordings from the same person. See [data citations and licensing](DATA_SOURCES.md).

## Run one local replay

```powershell
.\.venv\Scripts\python.exe scripts/demo_replay.py --dataset "D:\data\butqdb" --record 100001
```

This command starts the actual client and server services on IPv4 loopback, loads the saved model, transmits ECG and acceleration and waits for the first complete model window. The release check used 2 s packets, accumulated 10 s of signals, evaluated all 64 features and returned quality class 2. The measurement describes one functional replay. See [test details](docs/functional_check.json) and the [transport and inference protocol](docs/protocol.md).

For the graphical applications, start the server, import `artifacts/model_bundle.joblib`, set its bind address to `127.0.0.1` and start listening. Open the client in a second terminal, select the dataset folder and record, then connect to the same port. The default port is 50070.

```powershell
.\.venv\Scripts\python.exe run_server.py
.\.venv\Scripts\python.exe run_client.py
```

`run_clients.py` opens multiple replay client windows. `run_subject.py` streams selected records without a GUI. `run_training.py` opens the offline training interface. The installation check used a headless replay and the CPU model. Running the released example requires no training.

## Architecture and saved model

- `ekg_monitor/presentation/` contains the Dear PyGui client, server and training windows.
- `ekg_monitor/application/` implements dataset preparation, model configuration, signal processing and streaming services.
- `ekg_monitor/infrastructure/` handles WFDB records, TCP framing, disk-backed signal history and model import/export.
- `artifacts/model_bundle.joblib` is the saved scikit-learn pipeline with its fitted scaler, MLP parameters, feature order and preprocessing settings. Its SHA-256 is `0b0adf27512aa4812d43f4fcc9214d7e8faed6a321c635a0b30a75925cda45df`. Load the trusted artifact supplied with this release. Python model serialization can execute code on load.
- `artifacts/model_bundle.json` and `artifacts/training_results.json` are the accompanying archived, machine-readable exports.

## Reported evaluation

To reconstruct the reported evaluation from the public recordings, use the dataset's `SHA256SUMS.txt` and original files for records 104001, 114001, 115001, 123001, 125001 and 126001. The script verifies input hashes, extracts features from the annotated windows, applies the saved model and compares the resulting window IDs, class predictions and probabilities with the released reference CSV.

```powershell
.\.venv\Scripts\python.exe scripts/reproduce_public.py --dataset-root "D:\data\butqdb" --bundle-dir artifacts --output-dir public_check
```

The evaluation contains 2,739 windows, 1,927 correct predictions and 812 disagreements. Accuracy was 70.354%, balanced accuracy 83.814% and macro F1 70.801%. Class 2 precision was 27.1% and F1 was 41.4%. All 165 Class 3 evaluation windows came from a single person. There were zero Class 3-to-1 predictions in that partition and three Class 3-to-2 predictions. Window counts do not measure clinical alerts or time spent reviewing recordings. Scientific scope, reconstruction steps and unresolved historical details are recorded in [paper reproducibility](docs/paper_reproducibility.md).

## Availability

See [software license status](LICENSE_STATUS.md) and [data sources](DATA_SOURCES.md) for separate code and data terms. The three applications are HolterSQA Client Simulator, HolterSQA Monitoring Server and HolterSQA Model Training.
