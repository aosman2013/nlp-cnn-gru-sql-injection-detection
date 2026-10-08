from __future__ import annotations

import torch
import torch.nn as nn
from transformers import BertModel


class BERTCNNBiGRU(nn.Module):
    """BERT -> 1D CNN -> BiGRU -> dropout -> 2-class classifier."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        cnn_filters: int = 128,
        kernel_size: int = 5,
        gru_hidden: int = 128,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        bert_dim = self.bert.config.hidden_size
        self.conv = nn.Conv1d(
            in_channels=bert_dim,
            out_channels=cnn_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.relu = nn.ReLU()
        self.gru = nn.GRU(
            input_size=cnn_filters,
            hidden_size=gru_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(gru_hidden * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv(x))
        x = x.permute(0, 2, 1)
        _, h = self.gru(x)
        h = torch.cat((h[-2], h[-1]), dim=1)
        return self.fc(self.dropout(h))
