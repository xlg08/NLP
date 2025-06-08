from decoder import *


# 创建全连接层类
class Generator(nn.Module):
	def __init__(self, d_model, vocab_size):
		super().__init__()
		# 输出维度=词表大小
		self.out = nn.Linear(d_model, vocab_size)
	
	def forward(self, x):
		return torch.log_softmax(self.out(x), dim=-1)


if __name__ == '__main__':
	# 获取解码器结果
	dl_result = te01_decoder()
	d_model = 512
	vocab_size = 10000
	# 创建输出对象
	my_generator = Generator(d_model, vocab_size)
	output = my_generator(dl_result)
	print(output.shape, output)
