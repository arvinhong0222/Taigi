import os
import re
import torch
import torchaudio
import jiwer
import numpy as np
import torch.distributed as dist
from datasets import load_dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    TrainingArguments,
    Trainer
)
from transformers.trainer_utils import get_last_checkpoint
from data_collator_seq import custom_collator  # 自訂 collator，負責處理 input_features 與 labels

# ===========================
# 1. 系統與硬體設定
# ===========================
print("設定系統與硬體參數...")
os.environ["DS_BUILD_CPU_ADAM"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
torch.backends.cudnn.benchmark = True

# ===========================
# 2. 模型與資料路徑設定
# ===========================
print("設定模型與資料路徑...")
MODEL_NAME = "../sslab_model/whisper-large-zh-cv11"
DATA_PATH = "../sslab_dataset/sutiau-wav"
OUTPUT_DIR = "/mnt/disk_2/arvin/sslab_trained_model/whisper_taiwanese"
TRAIN_AUDIO_PATH = os.path.join(DATA_PATH, "train")
VAL_AUDIO_PATH = os.path.join(DATA_PATH, "val")
print(f"模型路徑: {MODEL_NAME}")
print(f"資料路徑: {DATA_PATH}")
print(f"Train 音檔資料夾: {TRAIN_AUDIO_PATH}")
print(f"Validation 音檔資料夾: {VAL_AUDIO_PATH}")

# ===========================
# 3. 載入處理器與模型
# ===========================
print("載入 Whisper Processor 與模型...")
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="zh", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
model.config.use_cache = False  # 關閉快取機制以避免反向傳播錯誤
model.gradient_checkpointing_enable()  # 啟用梯度檢查點以節省記憶體

# ===========================
# 4. 載入資料集
# ===========================
print("載入資料集...")
data_files = {
    "train": f"{DATA_PATH}/train/train_filtered.json",
    "validation": f"{DATA_PATH}/val/valid_filtered.json"
}
dataset = load_dataset("json", data_files=data_files)
print(f"資料集載入完成，Train 數量： {len(dataset['train'])}, Validation 數量： {len(dataset['validation'])}")

# ===========================
# 5. 資料前處理（文本清理）
# ===========================
print("開始文本清理...")
def remove_special_characters(batch):
    if batch.get("漢字"):
        batch["漢字"] = re.sub(r"[^\w\s]", "", batch["漢字"]).strip()
    return batch

dataset = dataset.map(remove_special_characters, num_proc=16)
print("文本清理完成。")

# ===========================
# 6. 音訊處理與標籤生成
# ===========================
print("開始音訊處理與標籤生成...")
def speech_file_to_array_fn(batch, audio_folder):
    file_path = os.path.join(audio_folder, batch["音檔檔名"] + ".wav")
    if not os.path.exists(file_path):
        print(f"檔案不存在：{file_path}")
        batch["input_features"] = None
        batch["labels"] = None
        return batch

    try:
        speech_array, sampling_rate = torchaudio.load(file_path)
        if speech_array.ndim > 1:
            speech_array = torch.mean(speech_array, dim=0, keepdim=True)
        speech_array = speech_array.squeeze().numpy()

        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
        speech_tensor = torch.tensor(speech_array)
        if speech_tensor.ndim == 1:
            speech_tensor = speech_tensor.unsqueeze(0)
        speech_tensor = resampler(speech_tensor)
        speech_array = speech_tensor.squeeze().numpy()

        processed = processor(speech_array, sampling_rate=16000)
        batch["input_features"] = processed.input_features[0]

        encoded = processor(text=batch["漢字"])
        if isinstance(encoded.input_ids[0], list):
            batch["labels"] = {"input_ids": encoded.input_ids[0]}
        else:
            batch["labels"] = {"input_ids": [encoded.input_ids[0]]}
    except Exception as e:
        print(f"Error processing {file_path}: {repr(e)}")
        batch["input_features"] = None
        batch["labels"] = None
    return batch

print("對 Train 資料集進行音訊處理...")
dataset["train"] = dataset["train"].map(
    lambda batch: speech_file_to_array_fn(batch, TRAIN_AUDIO_PATH),
    num_proc=8,
    remove_columns=["音檔檔名"]
)
print("對 Validation 資料集進行音訊處理...")
dataset["validation"] = dataset["validation"].map(
    lambda batch: speech_file_to_array_fn(batch, VAL_AUDIO_PATH),
    num_proc=8,
    remove_columns=["音檔檔名"]
)
print("音訊處理完成。")

print("過濾處理失敗的資料...")
dataset = dataset.filter(
    lambda x: x.get("漢字") is not None and x.get("input_features") is not None and
            x.get("labels") is not None and len(x["labels"]["input_ids"]) > 0,
    num_proc=16
)
print(f"過濾完成，Train 數量： {len(dataset['train'])}, Validation 數量： {len(dataset['validation'])}")

print("移除不必要的欄位...")
dataset = dataset.remove_columns(["id", "漢字", "羅馬字"])
print(f"移除後，Train 數量： {len(dataset['train'])}, Validation 數量： {len(dataset['validation'])}")

# ===========================
# 7. 建立 Collator 與定義評估指標
# ===========================
print("建立自訂 Collator 與定義評估指標...")
data_collator = lambda features: custom_collator(features, tokenizer=processor.tokenizer)

def compute_metrics(pred):
    # 檢查是否為 tuple，若是則取第一個元素
    pred_logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    pred_ids = pred_logits.argmax(-1)
    label_ids = pred.label_ids
    # 將 label 中的 -100 轉換為 tokenizer 的 pad token id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids)
    label_str = processor.tokenizer.batch_decode(label_ids)
    wer = jiwer.wer(label_str, pred_str)
    return {"wer": wer}


# ===========================
# 8. 建立抽樣驗證集
# ===========================
print("建立抽樣驗證集...")
num_val = min(5000, len(dataset["validation"]))
sampled_eval_dataset = dataset["validation"].shuffle(seed=42).select(range(num_val))
print(f"抽樣驗證集建立完成，驗證集數量： {len(sampled_eval_dataset)}")

# ===========================
# 9. 訓練參數設定（儘可能降低 GPU 記憶體使用）
# ===========================
print("設定訓練參數...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,    # 最小 batch size
    per_device_eval_batch_size=1,     # 最小 eval batch size
    eval_accumulation_steps=1,        # 減少同時搬移到 GPU 的資料量
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    num_train_epochs=1,               # 減少 epoch 數（測試時可先設為1）
    save_steps=1000,
    logging_steps=1000,               # 減少 log 輸出頻率
    learning_rate=3e-4,
    weight_decay=0.0,                 # 關閉 weight decay
    save_total_limit=1,
    report_to="none",
    fp16=True,
    gradient_accumulation_steps=1,
    warmup_ratio=0.0,                 # 關閉 warmup
    logging_dir=f"{OUTPUT_DIR}/logs",
    dataloader_num_workers=0,         # 使用單個 worker 避免額外開銷
    optim="adamw_torch",
    ddp_find_unused_parameters=False,
    gradient_checkpointing=True,      # 已啟用梯度檢查點以節省記憶體
    remove_unused_columns=False,
    dataloader_drop_last=True,
    deepspeed="./config/ds_config.json"  # ← 新增這一行，設定 DeepSpeed 配置檔的路徑
)
print("訓練參數設定完成。")

# 在這裡印出 DeepSpeed 設定，確認是否正確傳入
print("DeepSpeed 設定:", training_args.deepspeed)

# ===========================
# 10. 建立 Trainer 並開始訓練
# ===========================
print("建立 Trainer 並開始訓練...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=sampled_eval_dataset,
    processing_class=processor,  # 改用 processing_class 取代 tokenizer
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
if last_checkpoint is not None:
    print(f"發現 checkpoint: {last_checkpoint}，將從 checkpoint 恢復訓練。")
    torch.cuda.empty_cache()
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    print("未發現有效 checkpoint，將正常開始訓練。")
    torch.cuda.empty_cache()
    trainer.train()

print("訓練完成。")

if dist.is_initialized():
    dist.destroy_process_group()  # 銷毀分散式進程群，避免 NCCL 警告
