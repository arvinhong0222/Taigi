import os
import re
import torch
import torchaudio
import jiwer
from datasets import load_dataset
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer
)
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration, TrainingArguments, Trainer

processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="zh", task="translate")

from transformers.trainer_utils import get_last_checkpoint
from data_collator_ctc import DataCollatorCTCWithPadding, EvalDataCollatorCTCWithPadding  # 請確認檔案名稱正確

# ===========================
# 1. 系統與硬體設定
# ===========================
# 使用 GPU 2 與 GPU 3
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["OMP_NUM_THREADS"] = "8"

# 限制 CUDA 記憶體分配區塊最大尺寸，降低碎片化風險
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
torch.backends.cudnn.benchmark = True

# ===========================
# 2. 模型與資料路徑設定
# ===========================
MODEL_NAME = "./model/wav2vec2-large-xlsr-53"
DATA_PATH = "../sslab_dataset/sutiau-wav"
OUTPUT_DIR = "./wav2vec2_taiwanese"
AUDIO_PATH = os.path.join(DATA_PATH, "train")

# ===========================
# 3. 載入處理器與模型
# ===========================
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to(device)

# ===========================
# 4. 載入資料集
# ===========================
data_files = {
    "train": f"{DATA_PATH}/train/train.json",
    "validation": f"{DATA_PATH}/val/valid.json"
}
dataset = load_dataset("json", data_files=data_files)

# ===========================
# 5. 資料前處理（文本清理）
# ===========================
def remove_special_characters(batch):
    # 這裡使用「漢字」欄位作為標籤文本，依需要也可以改成「羅馬字」
    if batch.get("漢字"):
        # 移除掉非字詞字符（依需求修改正則表達式）
        batch["漢字"] = re.sub(r"[^\w\s]", "", batch["漢字"]).strip()
    return batch

dataset = dataset.map(remove_special_characters)

# ===========================
# 6. 音訊處理與標籤生成
# ===========================
def speech_file_to_array_fn(batch):
    file_path = os.path.join(AUDIO_PATH, batch["音檔檔名"] + ".wav")
    if not os.path.exists(file_path):
        # 檔案不存在時，設定為 None（後續會過濾掉這筆資料）
        batch["input_values"] = None
        batch["labels"] = None
        return batch
    try:
        speech_array, sampling_rate = torchaudio.load(file_path)
        if speech_array.ndim > 1:
            speech_array = torch.mean(speech_array, dim=0, keepdim=True)
        speech_array = speech_array.squeeze().numpy()
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
        speech_tensor = torch.tensor(speech_array)
        speech_array = resampler(speech_tensor).squeeze().numpy()
        # 處理音訊輸入
        batch["input_values"] = processor(speech_array, sampling_rate=16000).input_values[0]
        # 使用「漢字」欄位作為標籤；若希望使用羅馬字，請改成 batch["羅馬字"]
        batch["labels"] = processor(text=batch["漢字"]).input_ids
    except Exception as e:
        batch["input_values"] = None
        batch["labels"] = None
    return batch

# 在 map 過程中移除掉不需要的欄位（這邊移除「音檔檔名」）
dataset = dataset.map(speech_file_to_array_fn, remove_columns=["音檔檔名"])

# 過濾掉標籤或輸入處理失敗的資料
dataset = dataset.filter(lambda x: x.get("漢字") is not None and x.get("input_values") is not None)

# ===========================
# 7. 數據處理器與評估指標
# ===========================
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

def compute_metrics(pred):
    pred_ids = pred.predictions.argmax(-1)
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(label_ids)
    wer = jiwer.wer(label_str, pred_str)
    return {"wer": wer}

# ===========================
# 8. 建立抽樣驗證集
# ===========================
# 對驗證集隨機抽樣 2500 筆（固定 seed 以確保重現性）
sampled_eval_dataset = dataset["validation"].shuffle(seed=42).select(range(5000))

# ===========================
# 9. 訓練參數設定（訓練在 GPU 上）
# ===========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=1,
    eval_accumulation_steps=32,    # 每次搬移較小的資料量
    eval_strategy="steps",
    eval_steps=1000,                # 每 500 個訓練步驟執行一次 eval
    save_strategy="steps",
    num_train_epochs=5,
    save_steps=1000,                # 每 500 步存一次 checkpoint
    logging_steps=100,
    learning_rate=3e-4,
    weight_decay=0.005,
    save_total_limit=1,
    report_to="none",
    fp16=True,
    gradient_accumulation_steps=1,
    warmup_ratio=0.1,
    logging_dir=f"{OUTPUT_DIR}/logs",
    dataloader_num_workers=16,
    optim="adamw_torch",
    ddp_find_unused_parameters=True,
    gradient_checkpointing=False,
    remove_unused_columns=False,
    dataloader_drop_last=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=sampled_eval_dataset,  # 使用抽樣驗證集
    tokenizer=processor.feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
if last_checkpoint is not None:
    print(f"發現 checkpoint: {last_checkpoint}，將從 checkpoint 恢復訓練。")
    # 清理 GPU 記憶體
    torch.cuda.empty_cache()
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    print("未發現有效 checkpoint，將正常開始訓練。")
    # 清理 GPU 記憶體
    torch.cuda.empty_cache()
    trainer.train()
