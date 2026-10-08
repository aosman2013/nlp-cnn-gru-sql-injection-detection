#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sqli_detection.data import SQLiDataset, load_dataset, make_primary_split
from sqli_detection.evaluation import evaluate
from sqli_detection.metrics import compute_binary_metrics
from sqli_detection.model import BERTCNNBiGRU
from sqli_detection.plots import save_confusion_matrix, save_roc_curve
from sqli_detection.reproducibility import set_global_seed


def parse_args():
    p = argparse.ArgumentParser(description="Leakage-controlled 5-fold CV for BERT-CNN-BiGRU.")
    p.add_argument("--data", required=True, help="CSV containing Query and Label columns.")
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "cv_bert_cnn_bigru.json"))
    p.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "cv_results"))
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg["seed"])
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    df = load_dataset(args.data)
    dev_df, heldout_test_df = make_primary_split(df, cfg["test_size"], seed)
    dev_ids = set(dev_df["original_id"])
    test_ids = set(heldout_test_df["original_id"])

    pd.DataFrame([
        {"partition": "development_80_percent", "n_samples": len(dev_df)},
        {"partition": "heldout_test_20_percent_not_used", "n_samples": len(heldout_test_df)},
    ]).to_csv(output_dir / "split_summary.csv", index=False)

    tokenizer = BertTokenizerFast.from_pretrained(cfg["model_name"])
    skf = StratifiedKFold(
        n_splits=cfg["n_splits"], shuffle=True, random_state=seed
    )

    fold_metrics, fold_histories, oof_rows, fold_sizes = [], [], [], []
    X_dev, y_dev = dev_df["Query"], dev_df["Label"]

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_dev, y_dev), start=1):
        print(f"\n{'='*72}\nFOLD {fold}/{cfg['n_splits']}\n{'='*72}")
        fold_train_df = dev_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = dev_df.iloc[val_idx].reset_index(drop=True)

        train_ids = set(fold_train_df["original_id"])
        val_ids = set(fold_val_df["original_id"])
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)
        assert len(train_ids) + len(val_ids) == len(dev_df)

        fold_sizes.append({
            "fold": fold,
            "train_samples": len(fold_train_df),
            "validation_samples": len(fold_val_df),
            "heldout_test_samples_used": 0,
        })

        train_ds = SQLiDataset(fold_train_df, tokenizer, cfg["max_sequence_length"])
        val_ds = SQLiDataset(fold_val_df, tokenizer, cfg["max_sequence_length"])
        train_loader = DataLoader(
            train_ds, batch_size=cfg["batch_size"], shuffle=True,
            num_workers=cfg["num_workers"], pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg["batch_size"], shuffle=False,
            num_workers=cfg["num_workers"], pin_memory=torch.cuda.is_available()
        )

        model = BERTCNNBiGRU(
            model_name=cfg["model_name"],
            cnn_filters=cfg["cnn_filters"],
            kernel_size=cfg["cnn_kernel_size"],
            gru_hidden=cfg["gru_hidden_size"],
            dropout=cfg["dropout"],
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
        criterion = nn.CrossEntropyLoss()

        history = {"fold": [], "epoch": [], "train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": [], "val_f1": []}
        start = time.time()
        for epoch in range(1, cfg["epochs_per_fold"] + 1):
            model.train()
            running_loss = 0.0
            train_true, train_pred = [], []
            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                train_true.extend(labels.detach().cpu().numpy())
                train_pred.extend(torch.argmax(logits, dim=1).detach().cpu().numpy())

            train_loss = running_loss / max(1, len(train_loader))
            train_acc = accuracy_score(train_true, train_pred)
            val_loss, val_true, val_pred, val_prob, _ = evaluate(model, val_loader, criterion, device)
            val_acc = accuracy_score(val_true, val_pred)
            val_f1 = f1_score(val_true, val_pred, zero_division=0)
            history["fold"].append(fold)
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)
            history["val_f1"].append(val_f1)
            print(f"Epoch {epoch}/{cfg['epochs_per_fold']} | Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f} | Val F1 {val_f1:.4f}")

        training_seconds = time.time() - start
        val_loss, y_true, y_pred, y_prob, row_ids = evaluate(model, val_loader, criterion, device)
        metrics = compute_binary_metrics(y_true, y_pred, y_prob)
        metrics.update({
            "fold": fold,
            "train_samples": len(fold_train_df),
            "validation_samples": len(fold_val_df),
            "validation_loss": val_loss,
            "training_seconds": training_seconds,
        })
        fold_metrics.append(metrics)
        fold_histories.append(pd.DataFrame(history))

        id_to_query = dev_df.set_index("original_id")["Query"].to_dict()
        for rid, yt, yp, pr in zip(row_ids, y_true, y_pred, y_prob):
            oof_rows.append({
                "fold": fold,
                "original_id": int(rid),
                "Query": id_to_query[int(rid)],
                "True_Label": int(yt),
                "Predicted_Label": int(yp),
                "SQLi_Probability": float(pr),
                "Correct": int(yt == yp),
            })

        save_confusion_matrix(y_true, y_pred, f"Confusion Matrix - CV Fold {fold}", output_dir / f"fold_{fold}_confusion_matrix.png")
        save_roc_curve(y_true, y_prob, f"ROC Curve - CV Fold {fold}", output_dir / f"fold_{fold}_roc_curve.png")

        del model, optimizer, train_loader, val_loader, train_ds, val_ds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_df.to_csv(output_dir / "cv_fold_metrics.csv", index=False)
    pd.DataFrame(fold_sizes).to_csv(output_dir / "cv_fold_sizes.csv", index=False)
    pd.concat(fold_histories, ignore_index=True).to_csv(output_dir / "cv_training_history.csv", index=False)

    oof_df = pd.DataFrame(oof_rows).sort_values("original_id").reset_index(drop=True)
    oof_df.to_csv(output_dir / "cv_out_of_fold_predictions.csv", index=False)
    assert len(oof_df) == len(dev_df)
    assert oof_df["original_id"].nunique() == len(dev_df)
    assert set(oof_df["original_id"]).isdisjoint(test_ids)

    metric_cols = ["accuracy", "precision", "recall", "f1", "auc", "fpr", "fnr", "mcc"]
    summary = []
    t_crit = 2.7764451051977987  # df=4, 95% CI
    for metric in metric_cols:
        values = fold_metrics_df[metric].astype(float).values
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        se = std / math.sqrt(cfg["n_splits"])
        summary.append({
            "metric": metric,
            "mean": mean,
            "std": std,
            "standard_error": se,
            "ci95_low": mean - t_crit * se,
            "ci95_high": mean + t_crit * se,
        })
    pd.DataFrame(summary).to_csv(output_dir / "cv_summary.csv", index=False)

    oof_metrics = compute_binary_metrics(oof_df["True_Label"], oof_df["Predicted_Label"], oof_df["SQLi_Probability"])
    with open(output_dir / "cv_oof_metrics.json", "w", encoding="utf-8") as f:
        json.dump(oof_metrics, f, indent=2)

    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({**cfg, "data": args.data}, f, indent=2)

    print("\nFinished. Outputs written to", output_dir)


if __name__ == "__main__":
    main()
