# BERT-CNN-GRU for SQL Injection Detection

This repository provides the implementation and experimental workflow for a hybrid deep-learning framework for SQL Injection (SQLi) detection based on BERT, Convolutional Neural Networks (CNN), and Gated Recurrent Units (GRU).

The model combines contextual language representations from BERT, local pattern extraction through a one-dimensional convolutional layer, and bidirectional sequential modeling through a GRU. The repository is intended to support reproducibility of the experiments, evaluation protocol, and supplementary analyses reported in the associated study.

---

## Overview

SQL Injection remains a major security threat to web applications because malicious payloads can manipulate backend SQL statements and bypass traditional rule-based filters.

The proposed architecture processes SQL query strings using BERT WordPiece tokenization and learns complementary representations at three levels:

1. **BERT** captures contextual relationships among query tokens.
2. **CNN** extracts local discriminative patterns, including short-range SQL syntax and injection signatures.
3. **GRU** models sequential and long-range dependencies across the extracted feature sequence.

The resulting representation is passed through a dropout layer and a fully connected classification layer to distinguish between benign and SQL injection queries.

---

## Datasets

### 1. Primary SQL Injection Dataset

The main experiments use the publicly available **Biggest SQL Injection Dataset** from Kaggle:

https://www.kaggle.com/datasets/gambleryu/biggest-sql-injection-dataset

The implementation expects:

```text
clean_sql_dataset.csv
```

with the main columns:

```text
Query
Label
```

The repository does not redistribute the Kaggle dataset. Users should obtain the data directly from the official dataset page and comply with its license and terms of use.

Example download:

```bash
kaggle datasets download \
    -d gambleryu/biggest-sql-injection-dataset

unzip biggest-sql-injection-dataset.zip -d data/
```

### 2. Supplementary Superviz26-SQL Dataset

Additional experiments were conducted using the independently released **Superviz26-SQL Cross-Domain SQL Attack Detection Dataset**.

Zenodo record:

https://zenodo.org/records/19627322

DOI:

```text
10.5281/zenodo.19627322
```

Superviz26-SQL contains SQL-query workloads from multiple application/database domains, including:

- OurAirports
- Sakila
- AdventureWorks
- Oracle Human Resources

The dataset provides fields such as:

```text
full_query
label
split
```

where:

```text
0 = Benign
1 = Attack
```

The supplementary supervised experiment included in this repository uses Superviz26-SQL labels for model training. Therefore, those results should be interpreted as a **supervised cross-dataset experiment**, rather than strict zero-shot external validation.

The dataset is not redistributed in this repository and should be downloaded from the official Zenodo record.

---

## Data Preprocessing

Separate preprocessing procedures are used for deep-learning and traditional machine-learning models.

### BERT-based Deep-Learning Pipeline

For the proposed BERT-CNN-BiGRU architecture, query strings are used as provided in the loaded dataset after removing samples with missing query or label values.

No explicit punctuation stripping, special-character removal, or SQL-symbol removal is applied before BERT tokenization. This preserves potentially discriminative SQLi syntax such as quotation marks, comment markers, semicolons, equality operators, and parentheses.

Queries are processed using the BERT WordPiece tokenizer and padded or truncated to a maximum sequence length of 128 tokens. The tokenizer produces `input_ids` and `attention_mask`, which are passed to the BERT encoder.

### Traditional Machine-Learning Pipeline

Traditional baselines use TF-IDF representations with:

```text
Maximum features: 10,000
N-gram range:     (1, 2)
Stop words:       English
```

TF-IDF vectors are then supplied to classifiers such as Decision Tree and K-Nearest Neighbors.

---

## Proposed Architecture

```text
SQL Query
   |
   v
BERT WordPiece Tokenizer
   |
   v
Pretrained BERT Encoder
   |
   v
Conv1D
128 filters
Kernel size = 5
   |
   v
ReLU
   |
   v
Bidirectional GRU
128 hidden units
   |
   v
Dropout = 0.3
   |
   v
Fully Connected Layer
   |
   v
Binary Classification
Benign / SQLi
```

---

## Model Configuration

| Parameter | Value |
|---|---:|
| BERT model | `bert-base-uncased` |
| BERT hidden dimension | 768 |
| Maximum sequence length | 128 |
| CNN filters | 128 |
| CNN kernel size | 5 |
| BiGRU hidden units | 128 |
| Dropout | 0.3 |
| Batch size | 32 |
| Learning rate | `2e-5` |
| Optimizer | Adam |
| Loss function | Cross-Entropy |
| Maximum training epochs | 20 |
| Early stopping | Enabled |
| Early-stopping patience | 3 epochs |
| Checkpoint criterion | Minimum validation loss |

---

## Evaluation Methodology

### Primary Dataset Split

The complete primary dataset is first partitioned using a stratified:

```text
80% development set
20% independent test set
```

The independent test partition is kept separate from cross-validation.

### Five-Fold Stratified Cross-Validation

Five-fold stratified cross-validation is conducted only on the 80% development partition.

For each fold:

```text
4 folds -> training
1 fold  -> validation
```

The independent 20% test set is excluded from all cross-validation folds.

This strategy is intended to preserve class distributions, measure variation across training subsets, reduce dependence on a single split, and prevent test-set leakage.

---

## Final Model Training

Training is allowed for a maximum of 20 epochs with early-stopping patience of 3 epochs. Whenever validation loss improves, the model checkpoint is updated. Training terminates when no validation-loss improvement occurs for three consecutive epochs.

The checkpoint corresponding to the minimum validation loss is retained for evaluation.

---

## Evaluation Metrics

The repository reports:

- Accuracy
- Precision
- Recall
- F1-score
- Area Under the ROC Curve (AUC)
- False Positive Rate (FPR)
- False Negative Rate (FNR)
- Matthews Correlation Coefficient (MCC)

Confusion matrices report:

```text
TN = True Negatives
FP = False Positives
FN = False Negatives
TP = True Positives
```

---




## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/bert-cnn-bigru-sqli-detection.git
cd bert-cnn-bigru-sqli-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Typical dependencies include:

```text
torch
transformers
pandas
numpy
scikit-learn
matplotlib
tqdm
requests
```

---

## Reproducibility

For reproducibility, experiments use a fixed random seed where applicable:

```text
random_state = 42
```

The repository records model configuration, dataset splitting parameters, and training settings for each experiment. Users are encouraged to retain `run_configuration.json` along with reported results so that the exact experimental configuration can be reconstructed.

---

## Supplementary Superviz26-SQL Experiment

The repository includes a separate script for supervised evaluation on Superviz26-SQL.

Example domain files:

```text
a-a.csv -> OurAirports
b-b.csv -> Sakila
c-c.csv -> AdventureWorks
d-d.csv -> Oracle HR
```

The experiment generates training/validation curves, a classification report, confusion matrix, ROC curve, AUC, FPR, FNR, and MCC.

Because Superviz26-SQL labels are used during training in this experiment, the results are reported separately from the primary independent-test evaluation.

---

## Data Availability

Primary dataset:

**Biggest SQL Injection Dataset, Kaggle**

https://www.kaggle.com/datasets/gambleryu/biggest-sql-injection-dataset

Supplementary dataset:

**Superviz26-SQL, Zenodo**

https://zenodo.org/records/19627322

DOI:

```text
10.5281/zenodo.19627322
```

---

## Citation

If you use this repository, please cite the associated manuscript.

```bibtex
@article{YOUR_CITATION_KEY,
  title   = {Improving SQL Injection Detection Using NLP-Integrated CNN-GRU Hybrid Deep Learning Architecture},
  author  = {Author names},
  journal = {Journal name},
  year    = {Year}
}
```

The citation information can be updated after publication.

---

## License

The source code in this repository is released under the MIT License unless otherwise stated.

Dataset licenses and usage conditions remain governed by their respective original providers.

---

## Disclaimer

This repository is intended for academic research and defensive cybersecurity applications. SQL injection samples are used for detection, classification, benchmarking, and security research purposes.
