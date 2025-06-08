from decoder import *


# 创建全连接层类
class Generator(nn.Module):
	def __init__(self, d_model, vocab_size):
		super().__init__()
		self.out = nn.Linear(d_model, vocab_size)
	
	def forward(self, x):
		return torch.log_softmax(self.out(x), dim=-1)
