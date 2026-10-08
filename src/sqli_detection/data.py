from __future__ import annotations

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


LABEL_MAP = {
    "benign": 0,
    "normal": 0,
    "safe": 0,
    "0": 0,
    "sqli": 1,
    "sql injection": 1,
    "malicious": 1,
    "attack": 1,
    "1": 1,
}


def load_dataset(path: str) -> pd.DataFrame:
    """Load Query/Label data and preserve an original row identifier."""
    df = pd.read_csv(path)
    df = df[["Query", "Label"]].dropna().copy()
    df["Query"] = df["Query"].astype(str)

    if not pd.api.types.is_numeric_dtype(df["Label"]):
        df["Label"] = (
            df["Label"].astype(str).str.strip().str.lower().map(LABEL_MAP)
        )

    df = df.dropna(subset=["Label"]).copy()
    df["Label"] = df["Label"].astype(int)
    if not set(df["Label"].unique()).issubset({0, 1}):
        raise ValueError(f"Labels must be binary 0/1; found {sorted(df.Label.unique())}")

    return df.reset_index(drop=False).rename(columns={"index": "original_id"})


def make_primary_split(df: pd.DataFrame, test_size: float = 0.20, seed: int = 42):
    """Create one stratified development/test split."""
    dev_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["Label"],
        random_state=seed,
    )
    dev_df = dev_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    dev_ids = set(dev_df["original_id"])
    test_ids = set(test_df["original_id"])
    assert dev_ids.isdisjoint(test_ids), "Development/test overlap detected."
    assert len(dev_ids) + len(test_ids) == len(df)
    return dev_df, test_df


class SQLiDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, max_len: int = 128):
        self.texts = frame["Query"].astype(str).tolist()
        self.labels = frame["Label"].astype(int).tolist()
        self.row_ids = frame["original_id"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "original_id": torch.tensor(self.row_ids[idx], dtype=torch.long),
        }
