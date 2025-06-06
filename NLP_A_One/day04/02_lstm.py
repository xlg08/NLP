'''

'''

import torch
from torch import nn

lstm = nn.LSTM(input_size=5, hidden_size=4, num_layers=1, bidirectional=True, batch_first=False)

x = torch.randn(5, 3, 5)

h0 = torch.randn(2, 3, 4)
c0 = torch.randn(2, 3, 4)

output, (hn, cn) = lstm(x, (h0, c0))

print(output)
print(output.shape)

print(hn)
print(hn.shape)

print(cn)
print(cn.shape)
