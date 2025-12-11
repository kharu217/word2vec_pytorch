import torch
import torch.nn as nn
import torch.nn.functional as F

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()

        #out: 1 x emdedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.activation_function1 = nn.ReLU()

        self.dropout = nn.Dropout(p=0.2)
        
        #out: 1 x vocab_size
        self.linear2 = nn.Linear(128, vocab_size)
        

    def forward(self, inputs):
        embeds = torch.mean(self.embeddings(inputs), dim=1)
        out = self.linear1(embeds)
        out = self.dropout(out)
        out = self.activation_function1(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return out

    def get_word_emdedding(self, word_idx):
        word = torch.tensor(word_idx)
        return self.embeddings(word).view(1,-1)

class skip_gram(nn.Module) :
    def __init__(self, vocab_size, embedding_dim, word_range):
        super().__init__()
        self.vocab_size = vocab_size
        self.word_range = word_range

        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        #out -> B, 1, D
        self.linear1 = nn.Linear(embedding_dim, vocab_size*(word_range*2))

    def forward(self, inputs) :
        B, D = inputs.shape

        embedding = self.embeddings(inputs)
        outputs = self.linear1(embedding).view(B*self.word_range*2, self.vocab_size)
        return outputs

if __name__ == "__main__" :
    model = skip_gram(embedding_dim=300, vocab_size=10, word_range=2)
    A = torch.randint(0, 9, (10, 1))

    output = model(A)

    print(model(A).shape)
    loss = nn.CrossEntropyLoss()
    
    target = F.softmax(output, dim=1).argmax(dim=1)
    print(target.shape)
    print(loss(output, target).item())

    print(output)