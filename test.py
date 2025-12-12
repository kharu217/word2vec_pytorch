from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer(
        r"tokenizer\bpe_vocab\vocab.json",
        r"tokenizer\bpe_vocab\merges.txt"
    )

print(tokenizer.encode("orange").ids)
print(tokenizer.decode([14895]))
