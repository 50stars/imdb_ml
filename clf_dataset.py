from typing import List

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class MultiLabelDataset(Dataset):
    """
    Custom dataset for multi-label classification.

    Args:
        texts (List[str]): List of texts.
        labels (List[List[int]]): List of label lists.
        tokenizer (PreTrainedTokenizer): Tokenizer object for tokenizing the texts.
        max_length (int): Maximum sequence length. Defaults to 512.

    """

    def __init__(self, texts: List[str], labels: List[List[int]], tokenizer: AutoTokenizer,
                 max_length: int = 512) -> None:
        self.texts \
            = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize and encode the text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        # Convert the label to a tensor
        label_tensor = torch.tensor(label, dtype=torch.float)

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": label_tensor,
        }
