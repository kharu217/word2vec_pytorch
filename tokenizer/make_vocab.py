from tokenizers import ByteLevelBPETokenizer
import os

# 1. 텍스트 파일이 있는 폴더 지정
data_dir = r"C:\Users\User\Downloads\archive (1)\wikitext-103-raw"
all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".txt")]

# 2. BPE tokenizer 초기화
tokenizer = ByteLevelBPETokenizer()

# 3. BPE tokenizer 학습
tokenizer.train(
    files=all_files,
    vocab_size=50000,        # 원하는 vocab 크기, 제한 없이 하고 싶으면 None
    min_frequency=2,         # 최소 등장 횟수
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
)

# 4. 결과 저장
save_path = r"tokenizer\bpe_vocab"

# create directory if not exists
os.makedirs(save_path, exist_ok=True)

tokenizer.save_model(save_path)