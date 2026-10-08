# Data


The scripts expect a CSV with two columns:

- `Query`: SQL/query payload text
- `Label`: binary label (`0` benign, `1` SQLi), or a supported textual equivalent

The original notebook downloaded the Kaggle dataset `gambleryu/biggest-sql-injection-dataset` and used `clean_sql_dataset.csv`.

## Download

Configure Kaggle credentials outside this repository and run:

```bash
python scripts/download_kaggle_data.py --output-dir data/raw
```

Then pass the CSV path explicitly to training/evaluation scripts, for example:

```bash
python scripts/run_cv.py --data data/raw/clean_sql_dataset.csv
```
