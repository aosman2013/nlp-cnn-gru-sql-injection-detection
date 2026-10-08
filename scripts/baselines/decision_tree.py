#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    df = pd.read_csv(args.data)[['Query', 'Label']].dropna()
    X_train, X_test, y_train, y_test = train_test_split(
        df['Query'], df['Label'], test_size=0.2, stratify=df['Label'], random_state=args.seed
    )
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = DecisionTreeClassifier(max_depth=20, random_state=args.seed)
    model.fit(X_train_tfidf, y_train)
    pred = model.predict(X_test_tfidf)
    prob = model.predict_proba(X_test_tfidf)[:, 1]
    print(classification_report(y_test, pred, target_names=['Benign', 'SQLi']))
    print('AUC:', roc_auc_score(y_test, prob))
    print('Confusion matrix:\n', confusion_matrix(y_test, pred))


if __name__ == '__main__':
    main()
