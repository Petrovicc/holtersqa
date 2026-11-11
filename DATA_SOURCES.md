# Data and attribution

The ECG and acceleration recordings, annotations and recording identifiers come from Andrea Nemcova, Radovan Smisek, Kamila Opravilová, Martin Vitek, Lukas Smital and Lucie Maršánová, *Brno University of Technology ECG Quality Database (BUT QDB)*, version 1.0.0, PhysioNet (2020). [Dataset DOI](https://doi.org/10.13026/kah4-0w24). [Versioned PhysioNet record and downloads](https://physionet.org/content/butqdb/1.0.0/). The original recordings and annotations are distributed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). Follow the license and keep its notices with copies or adaptations.

The saved model was fitted using features derived from annotated BUT QDB windows. The reference prediction table retains source record identifiers and consensus labels and adds window boundaries, predicted classes and model probabilities. The retained database material carries the [original dataset license](docs/BUT_QDB_LICENSE.txt). Full recordings are obtained from PhysioNet. This repository's software license status is separate from the dataset license. Dependencies are installed from their respective distributions with their own notices.

The dataset page also requests the following citations.

- L. Smital et al. *Real-world 24h+ ECG dataset with quality annotations and motion context*. Scientific Data (2026). [DOI](https://doi.org/10.1038/s41597-026-07905-w).
- T. Pollard et al. *PhysioNet as a global platform for biomedical research*. Nature Health 1 (2026), 792–795. [DOI](https://doi.org/10.1038/s44360-026-00096-z).
