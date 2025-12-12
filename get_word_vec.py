import torch
import model
from tokenizers import ByteLevelBPETokenizer

def get_closest_word(word:str, model:torch.nn.Module) :
    tokenizer = ByteLevelBPETokenizer(
            r"tokenizer\bpe_vocab\vocab.json",
            r"tokenizer\bpe_vocab\merges.txt"
        )

    embedding_set = model.state_dict()["embeddings.weight"]

    token = tokenizer.encode(word).ids
    if len(token) > 1 :
        print("token is more than 1")
        exit()
    embed_v = embedding_set[token[0]]

    min_dist = 1e10
    min_idx = -1
    for i, vec in enumerate(embedding_set):
        if i == token[0] :
            print(i)
            continue
        else :
            dist = float(torch.cdist(embed_v.unsqueeze(0), vec.unsqueeze(0)))
            if min_dist > dist :
                min_dist = dist
                min_idx = i
    print(f"{tokenizer.decode([min_idx])} is closest word from {word}")

if __name__ == "__main__" :
    test_model = model.CBOW(vocab_size=50000, embedding_dim=300)
    test_model.load_state_dict(torch.load(r"weight\cbow_11.pt"))
    get_closest_word("man", test_model)