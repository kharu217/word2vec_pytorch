import torch
from torch.utils.data import Dataset
from tokenizers import ByteLevelBPETokenizer
import numpy as np
import os
from tqdm import tqdm
import random

class text_dataset(Dataset) :
    def __init__(self, file_path, window_size, vocap_path, merges_path):
        super().__init__()
        assert window_size % 2 == 1
        self.window_size = window_size
        self.file_path = file_path
        self.tokenizer = ByteLevelBPETokenizer(
            vocap_path,
            merges_path
        )
        
        self.raw_tokens = []
        self.lines = open(self.file_path, 'r', encoding="utf-8").readlines()
    
    def __len__(self) :
        return len(self.lines)

    def __getitem__(self, index):
        temp_encoding = []

        while len(temp_encoding) < self.window_size :
            temp_encoding = self.tokenizer.encode(self.lines[random.randint(0, len(self.lines)-1)]).ids
        temp_encoding = np.array(temp_encoding)

        idx = random.randint(0, len(temp_encoding)-self.window_size)
        temp = temp_encoding[idx:idx+self.window_size]
        word = temp[self.window_size//2]
        seq = np.delete(temp, self.window_size//2)
        return torch.tensor(word), torch.tensor(seq)
    
if __name__ == "__main__" :
    test_dataset = text_dataset(file_path=r"C:\Users\User\Downloads\archive (1)\wikitext-103-raw\train.txt",
                                window_size=5,
                                vocap_path=r"tokenizer\bpe_vocab\vocab.json",
                                merges_path=r"tokenizer\bpe_vocab\merges.txt")
