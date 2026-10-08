from __future__ import annotations

import numpy as np
import torch


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_true, all_pred, all_prob, all_ids = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, attention_mask)
            total_loss += criterion(logits, labels).item()
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            all_true.extend(labels.cpu().numpy())
            all_pred.extend(preds.cpu().numpy())
            all_prob.extend(probs.cpu().numpy())
            all_ids.extend(batch["original_id"].cpu().numpy())

    return (
        total_loss / max(1, len(loader)),
        np.asarray(all_true),
        np.asarray(all_pred),
        np.asarray(all_prob),
        np.asarray(all_ids),
    )
