import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from dataset import text_dataset
from model import CBOW, skip_gram

def train(data_path:dict, epoch=10, batch_size=8,model="cbow") :
    train_dataset = text_dataset(file_path=data_path["train"], vocap_path=r"tokenizer\bpe_vocab\vocab.json", merges_path=r"tokenizer\bpe_vocab\merges.txt", window_size=5)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    valid_dataset = text_dataset(file_path=data_path["valid"], vocap_path=r"tokenizer\bpe_vocab\vocab.json", merges_path=r"tokenizer\bpe_vocab\merges.txt", window_size=5)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

    test_dataset = text_dataset(file_path=data_path["test"],  vocap_path=r"tokenizer\bpe_vocab\vocab.json", merges_path=r"tokenizer\bpe_vocab\merges.txt", window_size=5)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    if model == "cbow" :
        train_model = CBOW(vocab_size=50000, embedding_dim=300)
    elif model == "skip_gram" :
        train_model = skip_gram(vocab_size=50000, embedding_dim=300, word_range=5)
    
    #(B, C, D)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=train_model.parameters())

    total_train_loss = 0
    total_valid_loss = 0

    print("train with", len(train_dataset), "lines of text")
    print("valid with", len(valid_dataset), "lines of text")
    #train start
    for epch in range(1, epoch+1) :
        for word, seq in tqdm(train_dataloader) :
            optimizer.zero_grad()

            if model == "cbow" : 
                pred = train_model(seq)
                loss = loss_fn(pred, word)

            #FIXME : batch size issue
            # elif model == "skip_gram" :
            #     pred = train_model(word.unsqueeze(-1))
            #     print(pred.shape)
            #     loss = loss_fn(pred, seq)
            
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        #valid
        with torch.no_grad() :
            for word, seq in tqdm(valid_dataloader) :
                if model == "cbow" : 
                    pred = train_model(seq)
                    loss = loss_fn(pred, word)

                #FIXME : batch size issue
                # elif model == "skip_gram" :
                #     pred = train_model(word.unsqueeze(-1))
                #     print(pred.shape)
                #     loss = loss_fn(pred, seq)

                total_valid_loss += loss.item()
        print(f"{epch} epoch : avg train loss => {total_train_loss/len(train_dataloader)}")
        print(f"{epch} epoch : avg valid loss => {total_valid_loss/len(valid_dataloader)}")

        total_valid_loss = 0
        total_train_loss = 0
    torch.save(train_model.state_dict(), f"weight/cbow_{epoch}.pt")
if __name__ == "__main__" :
    data_path = {
        "train" : r"C:\Users\User\Downloads\archive (1)\wikitext-103-raw\train.txt",
        "valid" : r"C:\Users\User\Downloads\archive (1)\wikitext-103-raw\valid.txt",
        "test" : r"C:\Users\User\Downloads\archive (1)\wikitext-103-raw\test.txt"
    }

    train(data_path=data_path, model="cbow", epoch=1, batch_size=64)
