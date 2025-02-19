import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# 設定模型與處理器存檔的目錄
OUTPUT_DIR = "./wav2vec2_taiwanese/checkpoint-5250/"

# 載入儲存好的處理器與模型
processor = Wav2Vec2Processor.from_pretrained(OUTPUT_DIR)
model = Wav2Vec2ForCTC.from_pretrained(OUTPUT_DIR)

# 設定模型為 evaluation 模式
model.eval()

# 讀取待推理的音訊檔（請將路徑改為您的音訊檔案位置）
audio_file = "./example/213(1).mp3"
speech_array, sampling_rate = torchaudio.load(audio_file)

# 若音訊為多聲道，則轉換成單聲道（取平均）
if speech_array.ndim > 1:
    speech_array = torch.mean(speech_array, dim=0, keepdim=True)

# 移除多餘的維度，轉為 numpy 陣列
speech_array = speech_array.squeeze().numpy()

# 如果原始音訊採樣率不是 16000 Hz，則進行重取樣
if sampling_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
    speech_tensor = torch.tensor(speech_array)
    speech_array = resampler(speech_tensor).squeeze().numpy()
    sampling_rate = 16000

# 使用處理器將音訊轉換為模型所需的輸入格式
inputs = processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt")

# 將模型輸入送入模型進行推理，並使用 no_grad 避免計算梯度
with torch.no_grad():
    logits = model(**inputs).logits

# 取得每個時間步預測的 token ID（最大機率）
predicted_ids = torch.argmax(logits, dim=-1)

# 將 token ID 轉換成文字
transcription = processor.batch_decode(predicted_ids)
print("轉錄結果：", transcription)
