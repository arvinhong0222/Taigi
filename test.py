from transformers import Wav2Vec2CTCTokenizer

# 指定 vocab.json 文件路徑
tokenizer = Wav2Vec2CTCTokenizer(
    vocab_file="./model/wav2vec2-large-xlsr-53/vocab.json",
    unk_token="<unk>",
    pad_token="<pad>"
)

print(tokenizer)
print(tokenizer.get_vocab())
