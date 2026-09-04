\# Brain Perturbation \& Working Memory EEG Analytics



An end-to-end, research-oriented EEG analytics pipeline for investigating working-memory-related neural activity through quality control, preprocessing, behavioral alignment, feature engineering, statistical analysis, machine learning, explainability, robustness analysis, empirical null validation, and evidence reconciliation.



The project demonstrates how raw EEG recordings can be transformed into a validated, interpretable, and structured analytical evidence layer.



> \*\*The project treats EEG analysis as a multi-layer evidence problem rather than simply a prediction problem.\*\*



\---



\## Project Overview



This repository presents a modular EEG analytics workflow built around the \*\*Sternberg Working Memory\*\* paradigm.



The pipeline integrates signal processing, behavioral reconstruction, statistical analysis, machine learning, explainability, robustness analysis, and evidence synthesis.



\### End-to-End Workflow



```text

Raw EEG

&#x20;  ↓

Quality Control

&#x20;  ↓

Preprocessing

&#x20;  ↓

ICA \& Artifact Processing

&#x20;  ↓

Epoch Validation

&#x20;  ↓

Behavioral Reconstruction

&#x20;  ↓

EEG–Behavior Alignment

&#x20;  ↓

Feature Engineering

&#x20;  ↓

Statistical Analysis

&#x20;  ↓

Machine Learning

&#x20;  ↓

Explainability

&#x20;  ↓

Robustness / LOSO Sensitivity

&#x20;  ↓

Empirical Null Validation

&#x20;  ↓

Evidence Reconciliation

&#x20;  ↓

Final Analytical Dataset

&#x20;  ↓

Power BI Analytics

```



\---



\## Why This Project Matters



EEG machine-learning projects can produce strong-looking predictive results while overlooking issues such as:



\* signal quality

\* channel integrity

\* event decoding

\* behavioral alignment

\* subject variability

\* target leakage

\* statistical reliability

\* robustness across subjects



This project explicitly addresses these layers.



The analytical goal is therefore not only:



> \*\*Can a model predict the target?\*\*



but also:



> \*\*Are the observed EEG-derived effects consistent, interpretable, robust, and supported across multiple analytical layers?\*\*



\---



\## Project Highlights



\* End-to-end EEG analytics pipeline

\* Working-memory EEG analysis

\* Multi-stage EEG quality control

\* Channel, sampling-rate, and signal-scale validation

\* Event inspection and decoding

\* ICA-based artifact processing

\* Epoch-level validation

\* Behavioral reconstruction and EEG–behavior alignment

\* Subject-level feature engineering

\* Statistical analysis

\* Machine learning

\* SHAP-based explainability

\* Subject-aware robustness analysis

\* Leave-One-Subject-Out (LOSO) sensitivity analysis

\* Directional robustness analysis

\* Empirical null validation

\* Cross-target consistency analysis

\* Multi-layer evidence reconciliation

\* Validated final analytical dataset

\* Power BI-ready analytical layer

\* Modular and reproducibility-oriented project structure



\---



\## Dataset



The project uses the publicly available:



\*\*OpenNeuro DS004117 — Sternberg Working Memory\*\*



Dataset version:



```text

1.0.1

```



Dataset characteristics include:



\* 23 healthy young adults

\* 85 original EEG recordings

\* 71 EEG channels

\* BIDS organization

\* Working-memory task

\* HED annotations

\* CC0 licensing



The original dataset should be distinguished from the analytical subset used by this project:



| Dataset Layer                  | Count |

| ------------------------------ | ----: |

| Original EEG recordings        |    85 |

| Validated EEG epoch files/runs |    82 |

| Subjects                       |    23 |

| Subject × feature observations | 1,265 |

| Unique EEG-derived features    |    55 |



Raw EEG recordings are intentionally excluded from this repository because of their size and data-management considerations.



Users wishing to reproduce the analysis should obtain the original dataset directly from OpenNeuro.



\---



\## Original Study



The dataset is associated with:



\*\*Onton, J., Delorme, A., \& Makeig, S. (2005).\*\*



\*Frontal midline EEG dynamics during working memory.\*



\*NeuroImage, 27(2), 341–356.\*



DOI:



```text

10.1016/j.neuroimage.2005.04.014

```



The experimental paradigm is based on a modified \*\*Sternberg Working Memory task\*\*, in which participants memorize relevant letters, ignore irrelevant information, and subsequently respond to a probe indicating whether a memorized item was present.



\---



\# Analytical Pipeline



\## 1. Raw EEG Quality Control



The initial QC layer evaluates:



\* EEG channel structure

\* Channel metadata

\* Sampling rate

\* Signal scale

\* Recording integrity

\* Event structure

\* Suspicious channels

\* Subject-level recording characteristics



Representative QC workflows include:



```text

01\_quality\_control.py

02\_bad\_channel\_detection.py

03\_channel\_information.py

04\_event\_inspection.py

04\_inspect\_suspicious\_channels.py

05\_sampling\_rate\_check.py

06\_data\_scale\_check.py

06\_final\_qc.py

07\_subject\_scale\_inspection.py

08\_scale\_summary.py

```



The purpose of this stage is to identify acquisition and signal-quality problems before downstream processing.



\---



\## 2. Event Inspection \& Decoding



Experimental events are inspected and decoded to reconstruct the structure of the working-memory task.



This stage establishes the connection between:



```text

EEG recordings

&#x20;     ↓

Experimental events

&#x20;     ↓

Behavioral information

```



Representative workflows include:



```text

05\_decode\_events.py

analyze\_event\_distribution\_v5.py

```



\---



\## 3. Preprocessing



EEG recordings undergo preprocessing and harmonization before feature extraction and downstream analysis.



The preprocessing layer prepares the data for:



\* artifact handling

\* epoching

\* feature extraction

\* statistical analysis

\* machine learning



\---



\## 4. ICA \& Artifact Processing



Independent Component Analysis (ICA) is used to identify and handle artifact-related components.



Representative workflows include:



```text

apply\_ica.py

apply\_ica\_run2.py

artifact\_localization\_82runs.py

```



The objective is to reduce major artifact contamination while preserving relevant EEG information.



\---



\## 5. Epoch Validation



Processed EEG recordings are converted into validated epochs suitable for downstream analysis.



The final analytical workflow uses:



\*\*82 validated EEG epoch files/runs\*\*



Multiple QC layers are used to evaluate epoch integrity before feature extraction.



\---



\## 6. Behavioral Reconstruction \& Alignment



Behavioral information is reconstructed and aligned with EEG-derived data.



The analytical chain is:



```text

Experimental Events

&#x20;       ↓

Behavioral Responses

&#x20;       ↓

EEG Epochs

&#x20;       ↓

Analytical Targets

```



Two principal behavioral targets are used:



```text

target\_remember

target\_correct

```



Representative workflow:



```text

audit\_behavioral\_mapping\_82runs.py

```



This stage is critical for ensuring that EEG features are paired with the correct behavioral observations.



\---



\## 7. Feature Engineering



EEG-derived features are constructed for subject-level analysis.



The current analytical feature space contains:



\*\*55 EEG-derived features\*\*



The subject-level analytical layer contains:



\*\*1,265 subject × feature observations\*\*



The feature engineering stage is designed to produce interpretable signal characteristics that can subsequently be evaluated through statistical and machine-learning approaches.



\---



\## 8. Statistical Analysis



Statistical analysis evaluates relationships between EEG-derived features and behavioral targets.



The repository includes both Python- and R-based statistical workflows.



Representative R analysis:



```text

r\_statistical\_analysis\_v1.R

```



Statistical evidence is treated as one component of the broader analytical framework rather than as the sole decision criterion.



\---



\## 9. Machine Learning



Machine-learning workflows evaluate whether EEG-derived features contain predictive information related to the behavioral targets.



The current ML pipeline emphasizes:



\* subject-aware analysis

\* feature-level modeling

\* target validation

\* leakage auditing

\* model interpretation

\* robustness analysis



Representative current workflow:



```text

baseline\_ml\_v3.py

```



Machine-learning results are interpreted as one evidence layer alongside statistical, robustness, and null-validation analyses.



\---



\## 10. Explainability



SHAP-based explainability is incorporated to examine feature contributions to model outputs.



The analytical transition is:



```text

Prediction

&#x20;   ↓

Feature Contribution

&#x20;   ↓

Interpretation

```



This allows important EEG-derived features to be investigated beyond predictive performance alone.



\---



\## 11. Robustness \& LOSO Sensitivity



The project includes subject-level robustness and Leave-One-Subject-Out (LOSO) sensitivity analysis.



Representative workflow:



```text

publication\_grade\_loso\_sensitivity\_v1.py

```



The LOSO component is intentionally described as:



\*\*Leave-One-Subject-Out sensitivity / robustness analysis\*\*



rather than being presented as a conventional LOSO machine-learning benchmark.



Directional robustness analysis considers quantities such as:



\* Positive fraction

\* Negative fraction

\* Dominant direction

\* Nonzero fraction

\* Mean

\* Median

\* Standard deviation



These analyses help assess whether observed feature-level patterns remain stable across subjects.



\---



\## 12. Empirical Null Validation



An empirical null-validation layer is used to evaluate whether observed feature-target associations could plausibly arise under a null distribution.



Current validation includes:



| Metric                        | Value |

| ----------------------------- | ----: |

| EEG-derived features          |    55 |

| Behavioral targets            |     2 |

| Feature-target observations   |   110 |

| Null-significant observations |     1 |

| NaN values                    |     0 |

| Infinite values               |     0 |

| Duplicate observations        |     0 |



The empirical null is treated as an additional statistical-control layer rather than definitive proof of scientific validity.



\---



\# Evidence Reconciliation



The project integrates evidence from multiple analytical layers.



The current evidence-reconciliation framework uses:



| Evidence Layer                      | Weight |

| ----------------------------------- | -----: |

| Statistical / perturbation evidence |    25% |

| Machine-learning evidence           |    20% |

| LOSO sensitivity                    |    20% |

| Cross-target consistency            |    15% |

| R statistical analysis              |    10% |

| Empirical null validation           |    10% |



The resulting framework produces a feature-level evidence ranking.



The current highest-ranked feature is:



```text

alpha\_beta\_central\_ratio

```



with an integrated evidence score of approximately:



```text

0.653

```



The score represents a \*\*multi-layer evidence ranking score\*\*.



It should \*\*not\*\* be interpreted as:



\* a probability

\* an effect size

\* model accuracy

\* a causal estimate



\---



\# Perturbation / Feature-Effect Analysis



The project includes a feature-effect / perturbation analysis layer for evaluating feature-level changes across conditions and subjects.



Representative workflows include:



```text

perturbation\_effect\_analysis.py

perturbation\_feature\_analysis.py

perturbation\_statistical\_analysis\_v2.py

```



The analysis considers:



\* global condition effects

\* subject-level effects

\* directional consistency

\* feature stability

\* robustness characteristics



The current implementation should be interpreted as \*\*feature-effect / perturbation analysis\*\*, not as causal intervention or causal inference.



\---



\# Final Analytical Product



The final analytical product layer provides a structured feature-level output for downstream analysis and reporting.



Current validation includes:



| Metric                      | Value |

| --------------------------- | ----: |

| Feature rows                |    55 |

| Input columns               |    29 |

| Final columns               |    15 |

| Selected analytical columns |    10 |

| NaN values                  |     0 |

| Infinite values             |     0 |

| Duplicate features          |     0 |



Representative workflow:



```text

final\_product\_dataset\_v1.py

```



The final product layer is designed to provide a compact analytical representation of the upstream EEG pipeline.



\---



\# Power BI Analytics



A Power BI-ready analytical layer is also generated to connect scientific analysis with structured analytical reporting.



Current validation includes:



\* 55 unique features

\* 31 columns

\* 0 NaN

\* 0 Inf



Representative workflow:



```text

powerbi\_dashboard\_dataset\_v1.py

```



The Power BI/dashboard layer is intended to provide an interpretable interface for exploring feature-level analytical results.



\*\*Dashboard development is currently in progress.\*\*



\---



\# Repository Structure



```text

BrainPerturbationProject/

│

├── dashboard/

│   └── Dashboard-related files and outputs

│

├── data/

│   └── Raw/input EEG data

│

├── epochs/

├── epochs\_clean/

├── epochs\_v2/

├── epochs\_v3/

│   └── Local EEG epoch outputs

│

├── features/

│   ├── scientific\_v1/

│   ├── scientific\_v2/

│   ├── ml\_ready\_v2/

│   └── perturbation\_analysis/

│

├── final\_dataset/

│   └── Final analytical outputs

│

├── models/

│   └── Machine-learning models and outputs

│

├── notebooks/

│   └── Exploratory and analytical notebooks

│

├── perturbation/

│   └── Perturbation-related outputs

│

├── preprocessed/

├── preprocessed\_v2/

├── processed/

│   └── Local/intermediate EEG processing outputs

│

├── qc/

│   └── Quality-control and validation reports

│

├── results/

│   └── Statistical and analytical results

│

├── simulation/

│   └── Simulation-related outputs

│

├── pipeline.png

│   └── End-to-end pipeline visualization

│

├── README.md

└── Python / R analytical scripts

```



Large raw and intermediate EEG data are maintained locally and are not intended to be versioned in GitHub.



\---



\# How to Run



This repository is a \*\*modular research pipeline\*\*, not a single one-click application.



A typical analytical sequence is:



\### Step 1 — Quality Control



Run the relevant QC workflows:



```text

01\_quality\_control.py

02\_bad\_channel\_detection.py

03\_channel\_information.py

04\_event\_inspection.py

05\_sampling\_rate\_check.py

06\_data\_scale\_check.py

06\_final\_qc.py

```



\### Step 2 — Event Processing



```text

05\_decode\_events.py

analyze\_event\_distribution\_v5.py

```



\### Step 3 — Preprocessing \& ICA



```text

apply\_ica.py

apply\_ica\_run2.py

artifact\_localization\_82runs.py

```



\### Step 4 — Behavioral Alignment



```text

audit\_behavioral\_mapping\_82runs.py

```



\### Step 5 — Feature Engineering



Generate the scientific EEG feature datasets using the configured feature-engineering workflows.



The current pipeline includes:



```text

feature\_engineering\_scientific\_v1.py

feature\_engineering\_scientific\_v2.py

```



\### Step 6 — ML Dataset Preparation



The current ML workflow uses the v2 analytical dataset and feature-selection layer.



Representative workflows include:



```text

build\_ml\_ready\_dataset\_v2.py

scientific\_feature\_selection\_v2.py

build\_subject\_level\_split\_v2.py

```



\### Step 7 — Machine Learning \& Explainability



Run the current ML workflow:



```text

baseline\_ml\_v3.py

```



followed by the relevant explainability analyses.



\### Step 8 — Statistical Analysis



Run the relevant Python and R statistical workflows, including:



```text

r\_statistical\_analysis\_v1.R

```



\### Step 9 — Robustness \& Sensitivity



Representative workflows include:



```text

publication\_grade\_loso\_sensitivity\_v1.py

publication\_grade\_robustness\_validation\_v1.py

```



\### Step 10 — Empirical Null Validation



Run the configured empirical-null validation workflow.



\### Step 11 — Evidence Reconciliation



Generate the final evidence ranking and analytical product outputs.



Representative workflows include:



```text

final\_scientific\_evidence\_ranking\_v2.py

final\_evidence\_reconciliation\_v1.py

final\_product\_dataset\_v1.py

powerbi\_dashboard\_dataset\_v1.py

```



> The scripts are organized as modular analytical stages. Input/output paths and upstream artifacts should be configured according to the local project environment before execution.



\---



\# Reproducibility \& Data Management



The project follows a modular and auditable workflow.



Key principles include:



\* Explicit preprocessing stages

\* Dedicated QC outputs

\* Subject-level analytical records

\* Explicit behavioral target definitions

\* Separate EEG–behavior alignment

\* Leakage auditing

\* Multiple validation layers

\* Structured intermediate outputs

\* Final dataset validation

\* Git/GitHub version control

\* Exclusion of large raw EEG files from version control



Raw EEG recordings and other large intermediate artifacts are intentionally excluded from the public repository.



The `.gitignore` configuration excludes local/raw data and large EEG formats such as:



```text

\*.fif

\*.edf

\*.bdf

\*.set

\*.fdt

\*.h5

\*.hdf5

```



as well as local processing directories and temporary development artifacts.



The original OpenNeuro dataset must be obtained separately for full reproduction.



\---



\# Technologies



\## Programming



\* Python

\* R



\## Python Ecosystem



\* NumPy

\* Pandas

\* SciPy

\* scikit-learn

\* MNE-Python

\* SHAP

\* Matplotlib



\## Analytics



\* Statistical analysis

\* Machine learning

\* Explainable AI

\* Feature engineering

\* Signal processing

\* Data validation

\* Robustness analysis



\## Business Intelligence



\* Power BI

\* Analytical dataset design

\* Feature-level reporting



\## Version Control



\* Git

\* GitHub



\---



\# Scientific Interpretation



The analytical philosophy of the project is:



```text

Data Quality

&#x20;    ↓

Signal Integrity

&#x20;    ↓

Behavioral Alignment

&#x20;    ↓

Feature Engineering

&#x20;    ↓

Statistics

&#x20;    ↓

Machine Learning

&#x20;    ↓

Explainability

&#x20;    ↓

Robustness

&#x20;    ↓

Null Control

&#x20;    ↓

Evidence Synthesis

```



This framework reduces the risk of interpreting a single model result as definitive scientific evidence.



Instead, EEG-derived features are evaluated from multiple complementary perspectives.



\---



\# Key Results



The current analytical pipeline produces:



\* \*\*23 subjects\*\*

\* \*\*85 original EEG recordings\*\*

\* \*\*82 validated EEG epoch files/runs\*\*

\* \*\*55 EEG-derived features\*\*

\* \*\*1,265 subject × feature observations\*\*

\* \*\*2 principal behavioral targets\*\*

\* \*\*55 final feature-level product rows\*\*

\* \*\*15 final product columns\*\*

\* \*\*10 selected analytical columns\*\*

\* \*\*0 NaN values in the final dataset\*\*

\* \*\*0 infinite values in the final dataset\*\*

\* \*\*0 duplicate features\*\*

\* \*\*110 empirical-null feature-target observations\*\*

\* \*\*1 null-significant observation\*\*

\* \*\*Top reconciled feature:\*\* `alpha\_beta\_central\_ratio`

\* \*\*Integrated evidence score:\*\* approximately `0.653`



The integrated score is a multi-layer evidence ranking and is not equivalent to probability, model accuracy, or causal effect size.



\---



\# Limitations



\## Dataset Size



The dataset contains a relatively limited number of participants for modern machine-learning standards.



Results should therefore be interpreted as analytical evidence rather than definitive population-level conclusions.



\## Subject Generalization



EEG signals exhibit substantial inter-subject variability.



Subject-level robustness and sensitivity analyses help evaluate stability but do not eliminate the challenge of generalization.



\## Causal Interpretation



The perturbation/effect analysis does not establish causal relationships.



Observed associations should therefore not be interpreted as causal neural mechanisms.



\## Machine-Learning Interpretation



Machine-learning evidence should be interpreted together with statistical, robustness, explainability, and null-validation results.



\## Empirical Null Interpretation



The empirical null layer provides an additional control mechanism, but a small number of significant observations should be interpreted cautiously and do not independently establish broad statistical validity.



\---



\# Future Work



Potential extensions include:



\## 1. Advanced EEG Feature Engineering



\* Time-frequency decomposition

\* Connectivity features

\* Phase synchronization

\* Cross-frequency coupling

\* Advanced spatial features



\## 2. Advanced Machine Learning



\* Nested cross-validation

\* Subject-independent evaluation

\* Model comparison

\* Hyperparameter optimization

\* Calibration analysis



\## 3. Deep Learning



Potential architectures include:



\* CNN

\* Temporal CNN

\* LSTM

\* Transformer-based EEG models



\## 4. Advanced Explainability



\* SHAP interaction analysis

\* Feature clustering

\* Subject-level explanations

\* Temporal explanations

\* Channel-level interpretation



\## 5. Counterfactual / What-If Simulation



A future extension could implement controlled feature perturbations and counterfactual simulations to investigate how changes in individual analytical features affect model predictions.



\## 6. Neurotechnology Product Layer



A longer-term direction is to evolve the research prototype toward an analytical neurotechnology workflow:



```text

EEG Recording

&#x20;     ↓

Automated QC

&#x20;     ↓

Preprocessing

&#x20;     ↓

Feature Extraction

&#x20;     ↓

Prediction

&#x20;     ↓

Explainability

&#x20;     ↓

Subject-Level Report

```



\---



\# Project Status



\*\*Status: Research-oriented analytical prototype / portfolio-ready pipeline\*\*



Current components include:



\* Raw EEG engineering

\* Multi-layer QC

\* Event reconstruction

\* Behavioral alignment

\* ICA processing

\* Epoch validation

\* Scientific feature engineering

\* Statistical analysis

\* Machine learning

\* Explainability

\* Robustness analysis

\* LOSO sensitivity analysis

\* Empirical null validation

\* Evidence reconciliation

\* Final analytical dataset

\* Power BI-ready analytical layer



\*\*Dashboard development is ongoing.\*\*



\---



\# Ethical \& Data Considerations



This project uses a publicly available research dataset.



The repository does not distribute the original raw EEG recordings.



Users interested in reproducing the analysis should obtain the dataset directly from OpenNeuro and follow its licensing and usage requirements.



No personally identifying information is intentionally included in the analytical repository.



\---



\# Citation



If you use this project or build upon its methodology, please cite the original dataset and associated study.



\## Dataset



Onton, J., Delorme, A., Makeig, S., et al.



\*\*OpenNeuro Dataset DS004117 — Sternberg Working Memory\*\*



Dataset version 1.0.1.



```text

10.18112/openneuro.ds004117.v1.0.1

```



\## Original Study



Onton, J., Delorme, A., \& Makeig, S. (2005).



\*\*Frontal midline EEG dynamics during working memory.\*\*



\*NeuroImage, 27(2), 341–356.\*



```text

10.1016/j.neuroimage.2005.04.014

```



\---



\# Author



\*\*Ali Sadeghian\*\*



Business \& Data Analytics | Machine Learning | EEG Analytics



\---



\# Contact \& Collaboration



\*\*Email:\*\* \[alisadeqiann@gmail.com](mailto:alisadeqiann@gmail.com)



\*\*GitHub:\*\* https://github.com/alisadeghiann



\*\*LinkedIn:\*\* https://www.linkedin.com/in/ali-sadeghian-3a4107418



\---



\# License



The analytical code and documentation in this repository are provided for educational, research, and portfolio purposes.



The underlying EEG dataset is distributed separately through OpenNeuro under its own licensing terms.



Please refer to the original dataset documentation before redistributing or using the underlying data.



\---



\# Acknowledgment



This project builds upon the publicly available \*\*Sternberg Working Memory EEG dataset (OpenNeuro DS004117)\*\* and the research contributions of its original authors and contributors.



Special acknowledgment to:



\* Julie Onton

\* Arnaud Delorme

\* Scott Makeig

\* Dung Truong

\* Kay Robbins



for the original dataset and research work underlying this project.



\---



\## Final Pipeline Summary



```text

Raw EEG

&#x20;  ↓

Quality Control

&#x20;  ↓

Channel / Scale / Sampling Validation

&#x20;  ↓

Event Inspection \& Decoding

&#x20;  ↓

Preprocessing

&#x20;  ↓

ICA \& Artifact Processing

&#x20;  ↓

Epoch Validation

&#x20;  ↓

Behavioral Reconstruction

&#x20;  ↓

EEG–Behavior Alignment

&#x20;  ↓

Feature Engineering

&#x20;  ↓

Statistical Analysis

&#x20;  ↓

Machine Learning

&#x20;  ↓

Explainability

&#x20;  ↓

Robustness / LOSO Sensitivity

&#x20;  ↓

Empirical Null Validation

&#x20;  ↓

Evidence Reconciliation

&#x20;  ↓

Final Feature Dataset

&#x20;  ↓

Power BI Analytics

```



\*\*Brain Perturbation \& Working Memory EEG Analytics\*\*



\*From raw EEG recordings to validated, interpretable, and evidence-oriented analytical outputs.\*



