import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorCTCWithPadding:
    def __init__(self, processor, padding=True):
        self.processor = processor
        self.padding = padding

    def __call__(self, features):
        # 處理 input_values
        input_features = [{"input_values": feature["input_values"]} for feature in features if "input_values" in feature]
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        # 處理標籤
        label_features = [feature["labels"] for feature in features if "labels" in feature]
        labels_batch = self.processor.tokenizer.pad(
            {"input_ids": label_features},
            padding=self.padding,
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

class EvalDataCollatorCTCWithPadding(DataCollatorCTCWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)
        if "input_values" in batch:
            batch["input_values"] = batch["input_values"].float()  # 強制轉換為 float32
        return batch
