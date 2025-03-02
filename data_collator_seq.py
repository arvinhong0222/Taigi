import numpy as np
import torch

def custom_collator(features, tokenizer):
    # 若 features 為空，直接回傳空字典
    if not features:
        return {}

    # ----------------------------
    # 處理 input_features
    # ----------------------------
    try:
        input_features = []
        for f in features:
            feat = f.get("input_features")
            if feat is None:
                continue  # 若沒有 input_features，則跳過
            # 若不是 numpy array，嘗試轉換
            if not isinstance(feat, np.ndarray):
                try:
                    feat = np.array(feat)
                except Exception as e:
                    print("Error converting input_features to numpy array:", e)
                    continue
            input_features.append(feat)
        
        if len(input_features) == 0:
            raise ValueError("No valid input_features found in the batch.")
        
        # 取得所有 input_features 的最大長度
        max_input_len = max(feat.shape[0] for feat in input_features)
        padded_input_features = []
        for feat in input_features:
            pad_length = max_input_len - feat.shape[0]
            # 補零 (假設補在後面)
            if pad_length > 0:
                feat = np.pad(feat, (0, pad_length), mode="constant")
            padded_input_features.append(feat)
        
        # 使用 np.stack 將列表轉為單一 numpy array，再轉成 tensor
        padded_input_features = torch.tensor(np.stack(padded_input_features), dtype=torch.float)
    except Exception as e:
        raise RuntimeError(f"Error processing input_features in collator: {e}")

    # ----------------------------
    # 處理 labels
    # ----------------------------
    try:
        label_ids_list = []
        for f in features:
            labels = f.get("labels")
            if labels is None or "input_ids" not in labels:
                continue
            label_ids_list.append(labels["input_ids"])
        
        if len(label_ids_list) == 0:
            raise ValueError("No valid labels found in the batch.")
        
        # 使用 tokenizer.pad 進行標籤 padding
        padded_labels = tokenizer.pad({"input_ids": label_ids_list}, padding=True, return_tensors="pt")["input_ids"]
    except Exception as e:
        raise RuntimeError(f"Error processing labels in collator: {e}")

    return {"input_features": padded_input_features, "labels": padded_labels}
