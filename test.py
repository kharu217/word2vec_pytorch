import torch
import torch.nn as nn


MSAencoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=1, dim_feedforward=128, batch_first=True)
MSA_encoder = nn.TransformerEncoder(MSAencoder_layer, num_layers=4)

A = torch.rand((32, 10, 128))
print(MSA_encoder(A).shape)
