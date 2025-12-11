import torch
from model import CBOW

model = CBOW(vocab_size=50000, embedding_dim=300)
model.load_state_dict(torch.load("weight/cbow_1.pt"))

embedding_layer = None
for n, p in model.named_parameters() :
    print(n)
    if n == "embeddings.weight" :
        embedding_layer = p
print(embedding_layer[0])
