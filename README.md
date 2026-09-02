\# Brain Perturbation \& Working Memory EEG Analytics



An end-to-end EEG analytics and machine learning project for investigating \*\*working memory-related neural activity\*\* through rigorous data quality control, preprocessing, behavioral alignment, feature engineering, statistical analysis, machine learning, explainability, robustness analysis, empirical null validation, and evidence reconciliation.



The project is designed as a \*\*research-grade analytical pipeline and neurotechnology portfolio project\*\*, demonstrating how raw EEG data can be transformed into a validated, interpretable, and analysis-ready evidence layer.



\---



\# Repository Purpose



The purpose of this repository is to demonstrate a complete workflow for transforming raw EEG recordings into scientifically interpretable analytical outputs.



Rather than focusing only on predictive modeling, the project emphasizes the full analytical lifecycle:



\*\*Raw EEG → Quality Control → Preprocessing → ICA → Epoch Validation → Behavioral Alignment → Feature Engineering → Statistical Analysis → Machine Learning → Explainability → Robustness → Empirical Null Validation → Evidence Reconciliation → Final Analytical Dataset\*\*



The repository is intended to demonstrate practical skills across:



\* EEG signal processing

\* Scientific data quality control

\* Behavioral reconstruction and event alignment

\* Feature engineering

\* Statistical analysis

\* Machine learning

\* Explainable AI

\* Robustness and sensitivity analysis

\* Subject-level analysis

\* Data validation

\* Reproducible research workflows

\* Analytical dataset design

\* Neurotechnology-oriented data science



\---



\# Why This Project Matters



EEG projects can easily produce attractive machine learning results while hiding problems in data quality, event alignment, subject variability, leakage, or statistical reliability.



This project therefore treats EEG analysis as a \*\*multi-layer evidence problem\*\*, rather than simply a prediction problem.



The workflow explicitly addresses:



\* Raw EEG quality and channel integrity

\* Sampling-rate and signal-scale validation

\* Event decoding and experimental structure

\* Artifact detection and ICA-based cleaning

\* Epoch-level validation

\* Behavioral-response reconstruction

\* Subject-aware feature generation

\* Statistical effects

\* Machine-learning evidence

\* Feature explainability

\* Leave-One-Subject-Out sensitivity analysis

\* Directional robustness

\* Empirical null validation

\* Cross-target consistency

\* Final evidence reconciliation



The result is a pipeline designed to answer not only:



> \*\*"Can a model predict the target?"\*\*



but also:



> \*\*"Are the observed EEG effects consistent, interpretable, robust, and supported across multiple analytical layers?"\*\*



\---



\# Project Highlights



\* End-to-end EEG analytics pipeline

\* Working-memory EEG analysis

\* Multi-stage quality control

\* Behavioral and EEG event alignment

\* ICA-based artifact processing

\* Subject-level feature engineering

\* Statistical and machine-learning analysis

\* SHAP-based explainability

\* Leave-One-Subject-Out sensitivity analysis

\* Directional robustness analysis

\* Empirical null validation

\* Multi-layer evidence reconciliation

\* Final publication-oriented analytical dataset

\* Power BI-ready analytical layer

\* Reproducibility-oriented project structure



\---



\# Skills Demonstrated



\### EEG \& Neuroscience



\* EEG preprocessing

\* Channel quality control

\* Event and trigger inspection

\* Epoch validation

\* Artifact analysis

\* ICA

\* Behavioral alignment

\* Working-memory experimental analysis



\### Data Science



\* Python

\* Pandas

\* NumPy

\* SciPy

\* Scikit-learn

\* Data validation

\* Feature engineering

\* Statistical analysis

\* Machine learning



\### Machine Learning



\* Classification workflows

\* Subject-aware analysis

\* Feature importance

\* SHAP explainability

\* Robustness analysis

\* Sensitivity analysis

\* Leakage auditing



\### Scientific Analytics



\* Statistical effect analysis

\* Multiple analytical evidence layers

\* Empirical null validation

\* Cross-target consistency

\* Subject-level robustness

\* Evidence reconciliation

\* Scientific QC



\### Data Visualization \& BI



\* Power BI

\* Analytical dataset design

\* Feature-level reporting

\* Research-oriented visualization



\### Reproducibility \& Engineering



\* Modular Python scripts

\* Structured project organization

\* Intermediate validation layers

\* Automated QC outputs

\* Git/GitHub version control

\* Large-data exclusion from repository



\---



\# Overview



The project uses EEG recordings from a publicly available working-memory dataset and develops an analytical pipeline from raw recordings to a final feature-level evidence dataset.



The pipeline combines \*\*signal processing, behavioral information, statistics, machine learning, explainability, robustness analysis, and evidence synthesis\*\*.



Two principal behavioral targets are used throughout the analytical workflow:



\* `target\_remember`

\* `target\_correct`



\---



\# Key Results



The current analytical pipeline produces:



| Metric                                          |                     Result |

| ----------------------------------------------- | -------------------------: |

| Original EEG recordings in dataset              |                         85 |

| Validated EEG epoch files/runs used in pipeline |                         82 |

| Subject-feature records                         |                      1,265 |

| Final feature space                             |                55 features |

| Behavioral targets                              |                          2 |

| Final product dataset                           |                    55 rows |

| Final product dataset columns                   |                         15 |

| Selected analytical columns                     |                         10 |

| Final dataset NaN                               |                          0 |

| Final dataset Inf                               |                          0 |

| Duplicate features                              |                          0 |

| Null validation feature-target observations     |                        110 |

| Null validation significant observations        |                          1 |

| Top reconciled feature                          | `alpha\_beta\_central\_ratio` |

| Top integrated evidence score                   |                    ≈ 0.653 |



The integrated evidence score should be interpreted as a \*\*multi-layer evidence ranking score\*\*, not as a probability, effect size, or model accuracy.



\---



\# Dataset



This project uses:



\*\*OpenNeuro — DS004117: Sternberg Working Memory\*\*



Dataset version:



\*\*1.0.1\*\*



Source:



https://openneuro.org/datasets/ds004117/versions/1.0.1



Dataset characteristics include:



\* 23 healthy young adults

\* 85 EEG recordings in the original dataset

\* 71 EEG channels

\* BIDS organization

\* Working-memory task

\* HED annotations

\* CC0 license



The original dataset and this project's processed analytical subset should not be conflated:



\*\*Original dataset:\*\* 85 EEG recordings



\*\*Project analytical pipeline:\*\* 82 validated EEG epoch files/runs



Raw EEG files are intentionally excluded from this GitHub repository because of file size and reproducibility/data-management considerations.



\---



\# Original Study



The dataset is associated with the study:



\*\*Onton, J., Delorme, A., \& Makeig, S. (2005).\*\*



\*Frontal midline EEG dynamics during working memory.\*



NeuroImage, 27(2), 341–356.



DOI:



10.1016/j.neuroimage.2005.04.014



The original experimental paradigm is based on a modified \*\*Sternberg Working Memory task\*\*, in which participants memorize relevant letters while ignoring irrelevant information and subsequently respond to a probe indicating whether a memorized item was present.



\---



\# Experimental Concept



The analytical goal is to investigate whether EEG-derived features contain systematic information related to working-memory behavior.



Conceptually:



\*\*EEG activity → Neural features → Behavioral relationship → Statistical evidence → ML evidence → Robustness → Final evidence ranking\*\*



The project therefore integrates both \*\*neuroscientific interpretation\*\* and \*\*data-science methodology\*\*.



\---



\# End-to-End Pipeline



\---



\# Analytical Pipeline



\## 1. Raw EEG Quality Control



Initial quality-control procedures inspect:



\* EEG channel structure

\* Channel metadata

\* Sampling rate

\* Signal scale

\* Recording integrity

\* Event structure

\* Suspicious channels

\* Subject-level recording properties



Representative scripts include:



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



\---



\## 2. Event Inspection \& Decoding



Experimental events are inspected and decoded to reconstruct the task structure.



This stage ensures that EEG recordings can be reliably connected to experimental events and behavioral information.



Representative scripts include:



```text

05\_decode\_events.py

analyze\_event\_distribution\_v5.py

```



\---



\## 3. Preprocessing



EEG data undergoes preprocessing and harmonization before downstream analysis.



The preprocessing layer prepares the recordings for:



\* artifact handling

\* epoching

\* feature extraction

\* statistical analysis

\* machine learning



\---



\## 4. ICA \& Artifact Processing



Independent Component Analysis is applied to identify and handle artifact-related components.



Representative scripts include:



```text

apply\_ica.py

apply\_ica\_run2.py

artifact\_localization\_82runs.py

```



The goal is to reduce major artifact contamination while preserving relevant EEG information.



\---



\## 5. Epoch Validation



The processed EEG recordings are converted into validated epochs suitable for downstream analysis.



The project ultimately works with:



\*\*82 validated EEG epoch files/runs\*\*



Multiple QC layers are used to verify the integrity of the resulting epochs.



\---



\## 6. Behavioral Reconstruction \& Alignment



Behavioral information is reconstructed and aligned with EEG-derived data.



This layer connects:



\*\*Experimental events → Behavioral responses → EEG epochs → Analytical targets\*\*



Two principal targets are used:



```text

target\_remember

target\_correct

```



Representative analytical scripts include:



```text

audit\_behavioral\_mapping\_82runs.py

```



\---



\## 7. Feature Engineering



EEG-derived features are constructed at the subject level.



The final feature space contains:



\*\*55 features\*\*



The feature engineering layer is designed to capture interpretable signal characteristics relevant to working-memory analysis.



The resulting subject-level analytical dataset contains:



\*\*1,265 subject-feature records\*\*



\---



\## 8. Statistical Analysis



Statistical analysis evaluates relationships between EEG-derived features and behavioral targets.



The project includes both Python- and R-based statistical analysis.



Representative script:



```text

r\_statistical\_analysis\_v1.R

```



The statistical layer provides evidence complementary to machine-learning models.



\---



\## 9. Machine Learning



Machine-learning analysis evaluates whether EEG-derived features contain predictive information related to the behavioral targets.



The project emphasizes subject-aware analysis and explicitly audits potential target leakage.



Representative script:



```text

audit\_ml\_target\_leakage.py

```



Machine-learning results are treated as one evidence layer rather than the sole basis for scientific conclusions.



\---



\## 10. Explainability



Explainability methods are used to identify which features contribute most strongly to model outputs.



SHAP-based analysis is incorporated to move from:



\*\*Prediction → Feature contribution → Interpretation\*\*



This allows important EEG-derived features to be investigated beyond raw predictive performance.



\---



\## 11. Robustness \& Leave-One-Subject-Out Sensitivity



The project includes subject-level robustness and sensitivity analysis.



Representative script:



```text

publication\_grade\_loso\_sensitivity\_v1.py

```



This analysis evaluates whether observed feature effects remain stable when individual subjects are considered separately.



Importantly, this layer is treated as:



\*\*Leave-One-Subject-Out sensitivity / robustness analysis\*\*



rather than claiming it as a conventional LOSO machine-learning validation benchmark.



Directional robustness analysis evaluates quantities including:



\* Positive fraction

\* Negative fraction

\* Dominant direction

\* Nonzero fraction

\* Mean

\* Median

\* Standard deviation



\---



\## 12. Empirical Null Validation



An empirical null validation layer is used to evaluate whether observed feature-target associations could plausibly arise under a null distribution.



The current null validation contains:



\* 55 features

\* 2 behavioral targets

\* 110 feature-target observations

\* 1 null-significant observation

\* 0 NaN

\* 0 Inf

\* 0 duplicates



This provides an additional statistical-control layer beyond the main feature ranking.



\---



\# Evidence Reconciliation



The project combines evidence from multiple analytical layers rather than relying on a single metric.



The final evidence reconciliation uses the following weighting framework:



| Evidence Layer                      | Weight |

| ----------------------------------- | -----: |

| Statistical / perturbation evidence |    25% |

| Machine-learning evidence           |    20% |

| LOSO sensitivity                    |    20% |

| Cross-target consistency            |    15% |

| R statistical analysis              |    10% |

| Empirical null validation           |    10% |



The final reconciliation produces a feature-level evidence ranking.



The current highest-ranked feature is:



```text

alpha\_beta\_central\_ratio

```



with an integrated evidence score of approximately:



```text

0.653

```



This score represents a \*\*combined evidence ranking\*\*, not a probability or causal effect estimate.



\---



\# Perturbation / Effect Analysis



The project includes an analytical effect layer that evaluates feature-level changes across conditions and subjects.



Representative script:



```text

perturbation\_effect\_analysis.py

```



This layer currently focuses on:



\* Global condition effects

\* Subject-level effects

\* Directional consistency

\* Stability

\* Quality-control characteristics



The current implementation should therefore be interpreted as \*\*feature-effect / perturbation analysis\*\*, rather than claiming causal intervention or causal inference.



\---



\# Final Product Dataset



The final analytical product layer has been validated with:



\* 55 feature rows

\* 29 input columns

\* 15 final columns

\* 10 selected analytical columns

\* 0 NaN

\* 0 Inf

\* 0 duplicate features



Representative script:



```text

final\_product\_dataset\_v1.py

```



This layer is designed as the final structured analytical output of the research pipeline.



\---



\# Power BI Analytics



A dedicated Power BI-ready analytical dataset is also generated.



Validation results include:



\* 55 unique features

\* 31 columns

\* 0 NaN

\* 0 Inf



Representative script:



```text

powerbi\_dashboard\_dataset\_v1.py

```



The Power BI layer provides a bridge between scientific analysis and practical analytical reporting.



\---



\# Folder-by-Folder Explanation



```text

BrainPerturbationProject/

│

├── dashboard/

│   └── Dashboard-oriented outputs

│

├── data/

│   └── Raw/input EEG data

│

├── epochs/

│   └── Initial epoch outputs

│

├── epochs\_clean/

│   └── Cleaned epoch data

│

├── epochs\_v2/

│   └── Second-stage epoch outputs

│

├── epochs\_v3/

│   └── Final-stage epoch outputs

│

├── features/

│   └── EEG-derived feature datasets

│

├── final\_dataset/

│   └── Final analytical datasets

│

├── models/

│   └── Machine-learning models and outputs

│

├── notebooks/

│   └── Exploratory and analytical notebooks

│

├── perturbation/

│   └── Feature-effect and perturbation-related outputs

│

├── preprocessed/

│   └── Preprocessed EEG data

│

├── preprocessed\_v2/

│   └── Updated preprocessing outputs

│

├── processed/

│   └── Intermediate processed data

│

├── qc/

│   └── Quality-control reports and validation outputs

│

├── results/

│   └── Statistical and analytical results

│

├── simulation/

│   └── Simulation-related analytical outputs

│

├── pipeline.png

│   └── End-to-end analytical pipeline diagram

│

├── \*.py

│   └── Modular analysis and processing scripts

│

├── \*.R

│   └── R-based statistical analysis

│

├── .gitignore

│   └── Large/raw/generated data exclusion rules

│

└── README.md

&#x20;   └── Project documentation

```



\---



\# How to Run



This repository contains a modular research pipeline rather than a single one-click script.



A typical execution sequence is:



\### Step 1 — Quality Control



Run the initial QC scripts:



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



Generate subject-level EEG feature datasets using the feature-engineering scripts and configured input/output paths.



\### Step 6 — Statistical Analysis



Run the relevant Python statistical scripts and:



```text

r\_statistical\_analysis\_v1.R

```



\### Step 7 — Machine Learning \& Explainability



Run the configured ML workflows and SHAP-based analysis.



\### Step 8 — Robustness \& Sensitivity



```text

publication\_grade\_loso\_sensitivity\_v1.py

publication\_grade\_robustness\_validation\_v1.py

```



\### Step 9 — Empirical Null Validation



Run the null-validation workflow.



\### Step 10 — Evidence Reconciliation



Generate the final evidence ranking and analytical product dataset.



Representative scripts include:



```text

final\_scientific\_evidence\_ranking\_v2.py

final\_evidence\_reconciliation\_v1.py

final\_product\_dataset\_v1.py

powerbi\_dashboard\_dataset\_v1.py

```



> The scripts are organized as modular analytical stages. Input/output paths and upstream artifacts should be configured according to the local project environment before execution.



\---



\# Repository Structure



```text

BrainPerturbationProject/

│

├── dashboard/

├── data/

├── epochs/

├── epochs\_clean/

├── epochs\_v2/

├── epochs\_v3/

├── features/

├── final\_dataset/

├── models/

├── notebooks/

├── perturbation/

├── preprocessed/

├── preprocessed\_v2/

├── processed/

├── qc/

├── results/

├── simulation/

│

├── pipeline.png

├── README.md

├── .gitignore

│

└── Python / R analytical scripts

```



\---



\# Data Management



Raw and intermediate EEG data are intentionally excluded from the public repository.



The `.gitignore` configuration excludes:



```text

data/

epochs/

epochs\_clean/

epochs\_v2/

epochs\_v3/

preprocessed/

preprocessed\_v2/

processed/

final\_dataset/

```



It also excludes large EEG file formats including:



```text

\*.fif

\*.edf

\*.bdf

\*.set

\*.fdt

\*.h5

\*.hdf5

```



Temporary files, Python environments, logs, and other generated artifacts are also excluded.



This keeps the repository focused on:



\* Code

\* Documentation

\* Analytical methodology

\* Reproducible project structure

\* Lightweight analytical outputs



rather than large binary EEG recordings.



\---



\# Reproducibility



The project is structured around modular and auditable analytical stages.



Key reproducibility principles include:



\* Explicit preprocessing stages

\* Dedicated QC outputs

\* Subject-level analytical records

\* Separate behavioral alignment

\* Explicit target definitions

\* Leakage auditing

\* Multiple validation layers

\* Structured intermediate outputs

\* Final dataset validation

\* Git-based version control



The repository is designed so that each major analytical stage can be inspected independently.



\---



\# Technologies



\### Programming



\* Python

\* R



\### Python Ecosystem



\* NumPy

\* Pandas

\* SciPy

\* Scikit-learn

\* MNE-Python

\* SHAP

\* Matplotlib



\### Analytics \& BI



\* Power BI

\* Statistical analysis

\* Machine learning

\* Explainable AI



\### Version Control



\* Git

\* GitHub



\---



\# Scientific Interpretation



The project investigates EEG-derived evidence related to working memory at the intersection of neuroscience and data science.



The analytical philosophy is:



\*\*Data Quality → Signal Integrity → Behavioral Alignment → Features → Statistics → ML → Explainability → Robustness → Null Control → Evidence Synthesis\*\*



This approach reduces the risk of interpreting a single model result as definitive scientific evidence.



Instead, features are evaluated through multiple complementary perspectives.



\---



\# Key Takeaways



\* Built an \*\*end-to-end EEG analytics pipeline\*\* from raw recordings to final analytical evidence.

\* Implemented a \*\*multi-layer quality-control framework\*\* rather than relying on a single preprocessing step.

\* Connected EEG recordings with behavioral information through \*\*event decoding and alignment\*\*.

\* Developed a \*\*55-feature analytical space\*\* across 1,265 subject-feature records.

\* Combined \*\*statistics and machine learning\*\* instead of treating ML performance as the only evidence.

\* Added \*\*SHAP-based explainability\*\* for feature-level interpretation.

\* Used \*\*subject-level robustness and Leave-One-Subject-Out sensitivity analysis\*\*.

\* Added \*\*empirical null validation\*\* as an additional statistical-control layer.

\* Built a \*\*multi-layer evidence reconciliation framework\*\* for ranking EEG-derived features.

\* Produced a validated \*\*final analytical dataset\*\* and a \*\*Power BI-ready data layer\*\*.

\* Demonstrated the ability to transform complex neuroscience data into a structured analytical product.



\---



\# Limitations



Several limitations should be considered when interpreting the results.



\### Dataset Size



The dataset contains a relatively limited number of participants for modern machine-learning standards.



Therefore, findings should be interpreted as analytical evidence rather than definitive population-level conclusions.



\### Subject Generalization



Subject-level variability is substantial in EEG research.



The inclusion of sensitivity and robustness analyses helps assess stability, but does not eliminate the challenge of generalizing to new populations.



\### Causal Interpretation



The current perturbation/effect layer does not establish causal relationships.



Observed associations should therefore not be interpreted as causal neural mechanisms.



\### Machine-Learning Interpretation



Machine-learning evidence should be interpreted together with statistical, robustness, and null-validation results.



\### Null Validation



The empirical null validation provides an additional control layer, but one null-significant observation among the tested feature-target combinations should be interpreted cautiously and does not by itself establish broad statistical validity.



\---



\# Future Work



Potential extensions include:



\### 1. Advanced EEG Feature Engineering



\* Time-frequency decomposition

\* Connectivity features

\* Phase synchronization

\* Cross-frequency coupling

\* More advanced spatial features



\### 2. Advanced Machine Learning



\* Nested cross-validation

\* Subject-independent evaluation

\* Model comparison

\* Hyperparameter optimization

\* Calibration analysis



\### 3. Deep Learning



Potential architectures include:



\* CNN

\* Temporal CNN

\* LSTM

\* Transformer-based EEG models



\### 4. Advanced Explainability



\* SHAP interaction analysis

\* Feature clustering

\* Subject-level explanations

\* Temporal explanations

\* Channel-level interpretation



\### 5. Counterfactual / What-If Simulation



A future extension could implement explicit controlled feature perturbations and counterfactual simulations to investigate how changing individual analytical features affects model predictions.



\### 6. Neurotechnology Product Layer



Potential future development:



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



This could evolve the research prototype toward a practical neurotechnology analytics platform.



\---



\# Project Status



\*\*Status: Completed analytical prototype / portfolio-ready research pipeline\*\*



The current project includes:



\* Raw EEG engineering

\* Multi-layer QC

\* Event reconstruction

\* Behavioral alignment

\* ICA processing

\* Epoch validation

\* Feature engineering

\* Statistical analysis

\* Machine learning

\* Explainability

\* Robustness analysis

\* LOSO sensitivity

\* Empirical null validation

\* Evidence reconciliation

\* Final analytical dataset

\* Power BI-ready analytical layer

\* GitHub-ready project structure



\---



\# Project Scale



The project combines:



\* EEG signal processing

\* Behavioral data reconstruction

\* Scientific quality control

\* Statistical analysis

\* Machine learning

\* Explainable AI

\* Robustness testing

\* Empirical null validation

\* Data engineering

\* Business intelligence



This makes the project relevant to both:



\*\*Data Science / Business Analytics\*\*



and



\*\*Neuroscience / Neurotechnology\*\*



applications.



\---



\# Ethical \& Data Considerations



The project uses a publicly available research dataset.



The repository does not distribute the original raw EEG recordings.



Users interested in reproducing the analysis should obtain the dataset directly from OpenNeuro and follow the dataset's licensing and usage conditions.



No personal identifying information is intentionally included in the analytical repository.



\---



\# Citation



If you use this project or build upon its methodology, please cite the original dataset and study.



\### Dataset



Onton, J., Delorme, A., Makeig, S., et al.



\*\*OpenNeuro Dataset DS004117 — Sternberg Working Memory.\*\*



Dataset version 1.0.1.



DOI:



```text

10.18112/openneuro.ds004117.v1.0.1

```



\### Original Study



Onton, J., Delorme, A., \& Makeig, S. (2005).



\*\*Frontal midline EEG dynamics during working memory.\*\*



NeuroImage, 27(2), 341–356.



DOI:



```text

10.1016/j.neuroimage.2005.04.014

```



\---



\# Author



\*\*Ali Sadeghian\*\*



Business \& Data Analytics | Machine Learning



\---



\# Contact / Collaboration



Interested in collaboration across:



\* Data Analytics

\* Machine Learning

\* EEG Analysis

\* Neurotechnology

\* Data Science

\* Research-oriented AI



For research, MSc, technical, or interdisciplinary collaboration:



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



This project builds upon the publicly available \*\*Sternberg Working Memory EEG dataset (OpenNeuro DS004117)\*\* and the research contributions of the original dataset authors and contributors.



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



\*From raw EEG data to validated, interpretable, and evidence-oriented analytical outputs.\*



