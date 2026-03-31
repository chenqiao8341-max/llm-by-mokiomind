import os

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):
    """Simple next-token-prediction dataset backed by a json/jsonl file."""

    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=data_path, split="train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        text = str(sample["text"]).strip()

        token_ids = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length - 2,
        ).input_ids

        input_ids = [
            self.tokenizer.bos_token_id,
            *token_ids,
            self.tokenizer.eos_token_id,
        ]
        pad_length = self.max_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_length
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        return input_ids, labels, attention_mask
