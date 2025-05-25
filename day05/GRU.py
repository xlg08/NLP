

import torch
import torch.nn as nn

gru = nn.GRU(input_size=6, hidden_size=8, num_layers=1, bidirectional=False)

x = torch.randn(2, 6, 6)

h = torch.randn(1, 6, 8)

out, hn = gru(x, h)

print(f'out: {out}, out.shape:{out.shape}')
print(f'hn: {hn}, hn.shape:{hn.shape}')
