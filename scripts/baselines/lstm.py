#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class LSTMDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts.tolist(); self.labels = labels.tolist(); self.tokenizer = tokenizer; self.max_len = max_len
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        tokens = self.tokenizer(self.texts[idx], padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
        return {'input_ids': tokens['input_ids'].squeeze(0), 'labels': torch.tensor(self.labels[idx], dtype=torch.long)}


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        _, (h_n, _) = self.lstm(x)
        h = torch.cat((h_n[-2], h_n[-1]), dim=1)
        return self.fc(self.dropout(h))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=10)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    df = pd.read_csv(args.data)[['Query', 'Label']].dropna()
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['Query'], df['Label'], test_size=0.2, stratify=df['Label'], random_state=args.seed
    )
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_ds = LSTMDataset(train_texts, train_labels, tokenizer)
    test_ds = LSTMDataset(test_texts, test_labels, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMClassifier(tokenizer.vocab_size).to(device)
    criterion = nn.CrossEntropyLoss(); optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(args.epochs):
        model.train()
        for batch in train_loader:
            ids, labels = batch['input_ids'].to(device), batch['labels'].to(device)
            optimizer.zero_grad(); loss = criterion(model(ids), labels); loss.backward(); optimizer.step()

    model.eval(); y_true=[]; y_pred=[]; y_prob=[]
    with torch.no_grad():
        for batch in test_loader:
            ids, labels = batch['input_ids'].to(device), batch['labels'].to(device)
            logits = model(ids); prob = torch.softmax(logits, dim=1)[:,1]; pred = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().numpy()); y_pred.extend(pred.cpu().numpy()); y_prob.extend(prob.cpu().numpy())

    print(classification_report(y_true, y_pred, target_names=['Benign','SQLi']))
    print('AUC:', roc_auc_score(y_true, y_prob))
    print('Confusion matrix:\n', confusion_matrix(y_true, y_pred))


if __name__ == '__main__':
    main()
